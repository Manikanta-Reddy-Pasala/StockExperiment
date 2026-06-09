# Per-Model "Invest More" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-model "Invest more" action on `/v2/portfolio` that suggests deploying a model's idle cash into its own current picks, and on Approve places the live BUY (market hours only).

**Architecture:** A pure sizing/policy core (`model_invest_service.py`, no I/O — unit-tested) plus two routes in `momrot_routes.py` that wire it to existing infra: ledger cash (`model_ledger_service`), current picks (`ranking_dir/{today}.json`), live LTP + order placement (`_fyers_*` helpers), ledger write (`record_buy`), and the pg-advisory `trading_lock`. UI is a button + modal in `portfolio.html`.

**Tech Stack:** Python/Flask, SQLAlchemy ledger, Fyers API, Dragonfly cache (idempotency token), vanilla JS + Bootstrap modal.

---

## Key facts (verified in code)

- **Idle cash to deploy** = model ledger cash (`model_ledger_service` portfolio row `cash`), e.g. n100 = ₹57,689 after the 45k deposit. NOT allocated−invested.
- **Broker free cash** = `momrot_routes._fyers_available_cash(user_id)`.
- **Model's current picks** = `MODEL_PATHS[model]["ranking_dir"]/{YYYY-MM-DD}.json` → `top_n` list (each has `symbol`, `price`). Single-position models use `top_n[0]`; Retest uses `top_n[:max_holdings]`. `max_holdings` via `model_ledger_service.model_max_holdings(model)`.
- **Current holdings** (to skip filled slots / identify top-up) = `momrot_routes._fyers_holdings(user_id)` (broker) and/or `model_ledger_service.get_trades`.
- **Place order** = `momrot_routes._fyers_place_market(symbol, qty, "BUY", user_id)`; **live LTP** = `_fyers_live_ltp(symbol, user_id)`; **lock** = `src.services.trading.trade_lock.trading_lock`; **notify** = `_notify_tg`.
- **Record to ledger** = `model_ledger_service.record_buy(model_name, symbol, qty, price, fyers_order_id=order_id)`.
- **Trading-day** = `tools.shared.nse_calendar.is_trading_day(date)`.

---

## File structure

- Create: `src/services/trading/model_invest_service.py` — pure policy/sizing core (no Fyers/DB).
- Modify: `src/web/momrot_routes.py` — add `invest-preview` (GET) + `invest-execute` (POST) routes that orchestrate infra around the core.
- Modify: `src/web/templates/v2/portfolio.html` — per-model "Invest more" button + modal + fetch wiring.
- Modify: `src/web/static/sw.js` — bump `CACHE_VERSION`.
- Create: `tests/test_model_invest_service.py` — unit tests for the pure core.
- Create: `tests/test_invest_routes_dryrun.py` — route tests in dry-run (no real order).

---

## Task 1: Pure sizing core — `compute_buys`

**Files:**
- Create: `src/services/trading/model_invest_service.py`
- Test: `tests/test_model_invest_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_invest_service.py
from src.services.trading.model_invest_service import compute_buys

def test_single_position_topup_uses_min_of_idle_and_broker():
    # idle 57689, broker 40000 -> deployable 40000; rank-1 LTP 800 -> 49 sh (0.5% buffer)
    buys = compute_buys(idle_cash=57689, broker_cash=40000, max_holdings=1,
                        targets=[{"symbol": "ABC", "ltp": 800.0}], open_symbols=set())
    assert len(buys) == 1
    assert buys[0]["symbol"] == "ABC"
    assert buys[0]["qty"] == int((40000 * 0.995) // 800)   # 49
    assert buys[0]["amount"] == buys[0]["qty"] * 800.0

def test_retest_fills_only_empty_slots():
    # max 4, already hold A,B -> only C,D get budget; split deployable across 2
    buys = compute_buys(idle_cash=100000, broker_cash=100000, max_holdings=4,
                        targets=[{"symbol": s, "ltp": 100.0} for s in ("A","B","C","D")],
                        open_symbols={"A", "B"})
    syms = {b["symbol"] for b in buys}
    assert syms == {"C", "D"}
    assert sum(b["amount"] for b in buys) <= 100000 * 0.995 + 0.01

def test_zero_when_no_deployable():
    assert compute_buys(0, 50000, 1, [{"symbol":"X","ltp":10.0}], set()) == []
    assert compute_buys(50000, 0, 1, [{"symbol":"X","ltp":10.0}], set()) == []

def test_drops_zero_qty_targets():
    # deployable too small for a 5000-rupee share
    assert compute_buys(1000, 1000, 1, [{"symbol":"PRICEY","ltp":5000.0}], set()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd StockExperiment && python -m pytest tests/test_model_invest_service.py -q`
Expected: FAIL (ModuleNotFoundError: model_invest_service).

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/trading/model_invest_service.py
"""Per-model 'Invest more': pure policy/sizing core (no I/O).

Suggests how to deploy a model's idle sleeve cash into its OWN current picks:
single-position models top up rank-1; multi-slot (Retest) fills empty slots.
Routes in momrot_routes.py supply the live numbers (ledger cash, broker cash,
ranking targets + LTP, open symbols) and place/record the orders.
"""
from __future__ import annotations
from typing import Dict, List, Set

CASH_BUFFER = 0.995  # 0.5% headroom for brokerage/STT/GST


def compute_buys(idle_cash: float, broker_cash: float, max_holdings: int,
                 targets: List[Dict], open_symbols: Set[str]) -> List[Dict]:
    """Return a list of {symbol, ltp, qty, amount} buys, total <= deployable.

    idle_cash    : model's uninvested ledger cash
    broker_cash  : Fyers available_cash
    max_holdings : model's slot count (1 = single position)
    targets      : model's current picks, ordered best-first, each {symbol, ltp}
    open_symbols : bare symbols the model already holds (skip filled slots,
                   except a single-position model may top up its held rank-1)
    """
    deployable = min(max(0.0, idle_cash), max(0.0, broker_cash))
    if deployable <= 0 or not targets:
        return []

    if max_holdings <= 1:
        # single position: deploy all into rank-1 (top up even if already held)
        slots = targets[:1]
    else:
        # multi-slot: only fill slots not already held
        slots = [t for t in targets[:max_holdings]
                 if t["symbol"] not in open_symbols]
    if not slots:
        return []

    per_slot = (deployable * CASH_BUFFER) / len(slots)
    buys = []
    for t in slots:
        ltp = float(t["ltp"] or 0)
        qty = int(per_slot // ltp) if ltp > 0 else 0
        if qty < 1:
            continue
        buys.append({"symbol": t["symbol"], "ltp": ltp,
                     "qty": qty, "amount": round(qty * ltp, 2)})
    return buys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd StockExperiment && python -m pytest tests/test_model_invest_service.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/services/trading/model_invest_service.py tests/test_model_invest_service.py
git commit -m "feat(invest): pure sizing core compute_buys for per-model Invest More"
```

---

## Task 2: Market-open gate + idempotency token

**Files:**
- Modify: `src/services/trading/model_invest_service.py`
- Test: `tests/test_model_invest_service.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_model_invest_service.py
from datetime import datetime
from src.services.trading.model_invest_service import is_market_open, make_token

def test_market_open_window(monkeypatch):
    import src.services.trading.model_invest_service as M
    monkeypatch.setattr(M, "is_trading_day", lambda d=None: True)
    assert is_market_open(datetime(2026, 6, 11, 10, 0)) is True    # 10:00 weekday
    assert is_market_open(datetime(2026, 6, 11, 9, 0)) is False    # pre-open
    assert is_market_open(datetime(2026, 6, 11, 15, 45)) is False  # post-close
    monkeypatch.setattr(M, "is_trading_day", lambda d=None: False)
    assert is_market_open(datetime(2026, 6, 11, 10, 0)) is False   # holiday

def test_token_deterministic_and_buy_sensitive():
    b = [{"symbol": "ABC", "qty": 10, "ltp": 800.0, "amount": 8000.0}]
    t1 = make_token("n100", b, "2026-06-11")
    t2 = make_token("n100", b, "2026-06-11")
    t3 = make_token("n100", [{**b[0], "qty": 11}], "2026-06-11")
    assert t1 == t2 and t1 != t3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd StockExperiment && python -m pytest tests/test_model_invest_service.py -q`
Expected: FAIL (cannot import is_market_open / make_token).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/services/trading/model_invest_service.py
import hashlib
import json as _json
from datetime import datetime, time as _time

try:
    from tools.shared.nse_calendar import is_trading_day
except Exception:  # pragma: no cover - import shim for test envs
    def is_trading_day(d=None):
        return True

_OPEN = _time(9, 15)
_CLOSE = _time(15, 30)


def is_market_open(now: datetime) -> bool:
    """True only on an NSE trading day, 09:15..15:30 IST. `now` must be IST."""
    if not is_trading_day(now.date()):
        return False
    return _OPEN <= now.time() <= _CLOSE


def make_token(model_name: str, buys: list, day: str) -> str:
    """Deterministic idempotency token = hash(model, day, symbol+qty list)."""
    payload = _json.dumps(
        {"m": model_name, "d": day,
         "b": sorted((b["symbol"], int(b["qty"])) for b in buys)},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd StockExperiment && python -m pytest tests/test_model_invest_service.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/services/trading/model_invest_service.py tests/test_model_invest_service.py
git commit -m "feat(invest): market-open gate + idempotency token"
```

---

## Task 3: `invest-preview` route (read-only suggestion)

**Files:**
- Modify: `src/web/momrot_routes.py` (add route near `/buy-now`)
- Test: `tests/test_invest_routes_dryrun.py`

Helper to add in `momrot_routes.py` (reads the model's ranking file + overlays live LTP):

```python
def _model_targets(model_name: str, user_id: int = 1):
    """Return the model's current picks [{symbol, ltp}] from its ranking file,
    LTP overlaid live. Empty list if no ranking yet."""
    import json, os
    from datetime import datetime
    from src.web.admin_routes import MODEL_PATHS
    paths = MODEL_PATHS.get(model_name) or {}
    rdir = paths.get("ranking_dir")
    if not rdir:
        return []
    f = os.path.join(rdir, datetime.now().strftime("%Y-%m-%d") + ".json")
    if not os.path.exists(f):
        return []
    try:
        top = json.load(open(f)).get("top_n") or []
    except Exception:
        return []
    out = []
    for r in top:
        sym = (r.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
        if not sym:
            continue
        ltp = _fyers_live_ltp(sym, user_id) or float(r.get("price") or 0)
        out.append({"symbol": sym, "ltp": float(ltp)})
    return out
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_invest_routes_dryrun.py
import json
import pytest

@pytest.fixture
def client(monkeypatch):
    from src.web import momrot_routes as MR
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_model_targets",
                        lambda m, uid=1: [{"symbol": "ABC", "ltp": 800.0}])
    monkeypatch.setattr(MR, "_fyers_holdings", lambda uid=1: [])
    monkeypatch.setattr(MR, "_model_idle_cash", lambda m: 57689.0)
    from src.web.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()

def test_invest_preview_returns_sized_buy(client):
    r = client.get("/momrot/models/momentum_n100_top5_max1/invest-preview")
    d = r.get_json()
    assert d["success"] is True
    assert d["deployable"] == 40000.0
    assert d["buys"][0]["symbol"] == "ABC"
    assert d["buys"][0]["qty"] == int((40000 * 0.995) // 800)
    assert "token" in d and "market_open" in d
```

(Note: confirm the blueprint url_prefix for `momrot_bp` — adjust the path in the test + UI to match, e.g. `/momrot/...`. Grep `momrot_bp = Blueprint` for `url_prefix`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd StockExperiment && python -m pytest tests/test_invest_routes_dryrun.py -q`
Expected: FAIL (404 — route not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/web/momrot_routes.py
def _model_idle_cash(model_name: str) -> float:
    from src.services.trading import model_ledger_service as L
    for row in (L.get_all_settings() or []):
        if row.get("model_name") == model_name:
            return float(row.get("cash") or 0)
    return 0.0


@momrot_bp.route("/models/<model_name>/invest-preview", methods=["GET"])
def api_invest_preview(model_name):
    """Read-only: suggest deploying the model's idle cash into its own picks."""
    try:
        from datetime import datetime
        from src.services.trading.model_invest_service import compute_buys, is_market_open, make_token
        from src.services.trading.model_ledger_service import model_max_holdings
        user_id = int(request.args.get("user_id", 1))
        idle = _model_idle_cash(model_name)
        broker = _fyers_available_cash(user_id)
        targets = _model_targets(model_name, user_id)
        open_syms = {(p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
                     for p in _fyers_holdings(user_id)}
        maxh = model_max_holdings(model_name) or 1
        buys = compute_buys(idle, broker, maxh, targets, open_syms)
        day = datetime.now().strftime("%Y-%m-%d")
        return jsonify({
            "success": True, "model": model_name,
            "idle_cash": idle, "broker_free": broker,
            "deployable": min(max(0.0, idle), max(0.0, broker)),
            "market_open": is_market_open(datetime.now()),
            "buys": buys, "token": make_token(model_name, buys, day),
        })
    except Exception as e:
        logger.exception("invest-preview fail")
        return jsonify({"success": False, "error": str(e)}), 500
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd StockExperiment && python -m pytest tests/test_invest_routes_dryrun.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/web/momrot_routes.py tests/test_invest_routes_dryrun.py
git commit -m "feat(invest): invest-preview route (read-only suggestion)"
```

---

## Task 4: `invest-execute` route (live buy, gated + idempotent)

**Files:**
- Modify: `src/web/momrot_routes.py`
- Test: `tests/test_invest_routes_dryrun.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_invest_routes_dryrun.py
def test_invest_execute_blocks_when_market_closed(client, monkeypatch):
    from src.web import momrot_routes as MR
    monkeypatch.setattr(MR, "is_market_open_now", lambda: False)
    r = client.post("/momrot/models/momentum_n100_top5_max1/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "deadbeef"})
    d = r.get_json()
    assert d["success"] is False
    assert "market" in d["error"].lower()

def test_invest_execute_places_and_records(client, monkeypatch):
    from src.web import momrot_routes as MR
    placed = []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)  # fresh token
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 800.0)
    monkeypatch.setattr(MR, "_fyers_holdings", lambda uid=1: [])
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda sym, qty, side, uid=1: (placed.append((sym, qty, side)) or
                                                       {"ok": True, "result": {"data": {"orderid": "OID1"}}}))
    recorded = []
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "record_buy",
                        lambda m, s, q, p, fyers_order_id=None: recorded.append((m, s, q)))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    r = client.post("/momrot/models/momentum_n100_top5_max1/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "tok1"})
    d = r.get_json()
    assert d["success"] is True
    assert placed == [("ABC", 10, "BUY")]
    assert recorded == [("momentum_n100_top5_max1", "ABC", 10)]

def test_invest_execute_rejects_reused_token(client, monkeypatch):
    from src.web import momrot_routes as MR
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_token_consume", lambda t: False)  # already used
    r = client.post("/momrot/models/momentum_n100_top5_max1/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "used"})
    assert r.get_json()["success"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd StockExperiment && python -m pytest tests/test_invest_routes_dryrun.py -q`
Expected: FAIL (route + helpers undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/web/momrot_routes.py
from datetime import datetime as _dt

def is_market_open_now() -> bool:
    from src.services.trading.model_invest_service import is_market_open
    return is_market_open(_dt.now())

def _token_consume(token: str) -> bool:
    """Return True if token is fresh (and mark used); False if already used.
    Backed by Dragonfly so it is shared across gunicorn workers."""
    try:
        from src.services.cache import get_cache  # existing Dragonfly client
        c = get_cache()
        key = f"invest:token:{token}"
        if c.get(key):
            return False
        c.set(key, "1", ex=3600)
        return True
    except Exception:
        return True  # cache down -> do not hard-block the operator

@momrot_bp.route("/models/<model_name>/invest-execute", methods=["POST"])
def api_invest_execute(model_name):
    """Place the previewed BUY list for the model. Market-hours + idempotent."""
    _lk = None
    try:
        from src.services.trading.trade_lock import trading_lock
        from src.services.trading.model_ledger_service import record_buy
        body = request.get_json(silent=True) or {}
        buys = body.get("buys") or []
        token = body.get("token") or ""
        if not buys:
            return jsonify({"success": False, "error": "no buys"}), 400
        if not is_market_open_now():
            return jsonify({"success": False, "error": "market closed — deploys next session"}), 400
        if not _token_consume(token):
            return jsonify({"success": False, "error": "already submitted (duplicate token)"}), 409
        user_id = int(request.args.get("user_id", 1))
        _lk = trading_lock()
        if not _lk.__enter__():
            return jsonify({"success": False, "error": "another trade/rebalance in progress — retry shortly"}), 409
        broker = _fyers_available_cash(user_id)
        spent, fills = 0.0, []
        for b in buys:
            sym, qty = b["symbol"], int(b["qty"])
            ltp = _fyers_live_ltp(sym, user_id) or float(b.get("ltp") or 0)
            if qty < 1 or ltp <= 0:
                continue
            if spent + qty * ltp > broker * 0.995:   # re-validate against live broker cash
                continue
            res = _fyers_place_market(sym, qty, "BUY", user_id)
            if res.get("ok"):
                oid = ((res.get("result") or {}).get("data") or {}).get("orderid") or ""
                try:
                    record_buy(model_name, sym, qty, ltp, fyers_order_id=oid)
                except Exception as e:
                    logger.warning(f"record_buy failed {model_name} {sym}: {e}")
                spent += qty * ltp
                fills.append({"symbol": sym, "qty": qty, "price": ltp, "order_id": oid})
                _notify_tg(f"✅ *INVEST* `{model_name}` BUY {sym} x{qty} @ ₹{ltp:.2f} (₹{qty*ltp:,.0f})")
            else:
                err = (res.get("result") or {}).get("message") or res.get("error") or "unknown"
                fills.append({"symbol": sym, "qty": qty, "error": err})
                _notify_tg(f"❌ *INVEST FAIL* `{model_name}` {sym} x{qty} — {err}")
        return jsonify({"success": True, "model": model_name, "fills": fills, "deployed": round(spent, 2)})
    except Exception as e:
        logger.exception("invest-execute fail")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if _lk is not None:
            try:
                _lk.__exit__(None, None, None)
            except Exception:
                pass
```

(Build-time check: confirm the Dragonfly client import path `src.services.cache.get_cache` — grep for the existing cache accessor and adjust if named differently.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd StockExperiment && python -m pytest tests/test_invest_routes_dryrun.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/web/momrot_routes.py tests/test_invest_routes_dryrun.py
git commit -m "feat(invest): invest-execute route (live buy, market-gated, idempotent, ledger-recorded)"
```

---

## Task 5: Portfolio UI — button + modal

**Files:**
- Modify: `src/web/templates/v2/portfolio.html`
- Modify: `src/web/static/sw.js`

- [ ] **Step 1: Locate the per-model row render** in `portfolio.html` (grep for where each model card/row is built; reuse its model_name + cash fields). Add an "Invest more" button shown only when the model's `cash > 0`.

```html
<!-- inside each per-model row, where actions live -->
<button class="btn btn-sm btn-outline-success invest-more-btn"
        data-model="${m.model_name}"
        ${(m.cash > 0) ? '' : 'style="display:none"'}>Invest more</button>
```

- [ ] **Step 2: Add the modal + JS** (preview → render → approve → execute):

```html
<div class="modal fade" id="investModal" tabindex="-1"><div class="modal-dialog"><div class="modal-content">
  <div class="modal-header"><h5 class="modal-title">Invest more</h5>
    <button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
  <div class="modal-body" id="investBody">Loading…</div>
  <div class="modal-footer">
    <button class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
    <button class="btn btn-success" id="investApprove" disabled>Approve &amp; Invest</button>
  </div>
</div></div></div>
<script>
let _investCtx = null;
document.addEventListener('click', async (e) => {
  const btn = e.target.closest('.invest-more-btn');
  if (!btn) return;
  const model = btn.dataset.model;
  const body = document.getElementById('investBody');
  const approve = document.getElementById('investApprove');
  body.innerHTML = 'Loading suggestion…'; approve.disabled = true;
  new bootstrap.Modal(document.getElementById('investModal')).show();
  try {
    const d = await fetch(`/momrot/models/${model}/invest-preview`, {cache:'no-store'}).then(r=>r.json());
    if (!d.success) { body.innerHTML = `Error: ${d.error}`; return; }
    _investCtx = {model, buys: d.buys, token: d.token};
    const rows = (d.buys||[]).map(b => `<tr><td>${b.symbol}</td><td>₹${b.ltp.toFixed(2)}</td><td>${b.qty}</td><td>₹${Number(b.amount).toLocaleString('en-IN')}</td></tr>`).join('');
    body.innerHTML = `
      <div class="small text-muted mb-2">Idle ₹${Number(d.idle_cash).toLocaleString('en-IN')} · broker free ₹${Number(d.broker_free).toLocaleString('en-IN')} · deployable ₹${Number(d.deployable).toLocaleString('en-IN')}</div>
      ${rows ? `<table class="table table-sm"><thead><tr><th>Symbol</th><th>LTP</th><th>Qty</th><th>₹</th></tr></thead><tbody>${rows}</tbody></table>` : '<div class="text-muted">Nothing to deploy.</div>'}
      ${d.market_open ? '' : '<div class="text-warning small">Market closed — deploys next session.</div>'}`;
    approve.disabled = !(d.market_open && (d.buys||[]).length);
  } catch (err) { body.innerHTML = 'Load failed.'; }
});
document.getElementById('investApprove').addEventListener('click', async () => {
  if (!_investCtx) return;
  const approve = document.getElementById('investApprove');
  approve.disabled = true; approve.textContent = 'Investing…';
  try {
    const d = await fetch(`/momrot/models/${_investCtx.model}/invest-execute`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({buys: _investCtx.buys, token: _investCtx.token})
    }).then(r=>r.json());
    document.getElementById('investBody').innerHTML = d.success
      ? `<div class="text-success">Deployed ₹${Number(d.deployed).toLocaleString('en-IN')}. ${d.fills.map(f=>f.order_id?`${f.symbol} x${f.qty} ✅`:`${f.symbol} ❌ ${f.error||''}`).join(', ')}</div>`
      : `<div class="text-danger">${d.error}</div>`;
  } catch (e) { document.getElementById('investBody').innerHTML = '<div class="text-danger">Execute failed.</div>'; }
  approve.textContent = 'Approve & Invest';
  setTimeout(()=>location.reload(), 1500);
});
</script>
```

- [ ] **Step 3: Bump SW** `CACHE_VERSION` in `src/web/static/sw.js` to `v59-2026-06-10-invest-more`.

- [ ] **Step 4: Manual smoke (local or VM dev)** — open `/v2/portfolio`, click "Invest more" on a model with idle cash, confirm modal shows the suggestion. (Do NOT approve against live unless intended.)

- [ ] **Step 5: Commit**

```bash
git add src/web/templates/v2/portfolio.html src/web/static/sw.js
git commit -m "feat(invest): portfolio Invest More button + suggest/approve modal; SW v59"
```

---

## Task 6: Deploy + VM dry verification

- [ ] **Step 1:** Run full unit suite: `cd StockExperiment && python -m pytest tests/test_model_invest_service.py tests/test_invest_routes_dryrun.py -q` → all pass.
- [ ] **Step 2:** Deploy app-only: `./DEPLOY_PRODUCTION.sh --app-only`.
- [ ] **Step 3:** VM preview check (read-only, no order) for each of the 3 funded models:
  `ssh root@77.42.45.12 'curl -s "http://localhost:5001/momrot/models/momentum_n100_top5_max1/invest-preview" | python3 -m json.tool'`
  Confirm: `deployable == min(idle ledger cash, broker free)`, suggested `buys[0].symbol` == the model's actual current rank-1 (cross-check `/admin/<model>/ranking`), `market_open` correct for the time of day.
- [ ] **Step 4:** Confirm Approve is disabled out of hours; (optionally) one real small approve during market hours with a single low-priced share to validate the fill + ledger update, then verify in portfolio.

---

## Self-review notes

- Spec coverage: preview (T3), approve/execute (T4), suggest-per-model-logic (T1 compute_buys), market-hours gate (T2+T4), idempotency (T2+T4), UI (T5), tests (T1–T4), deploy/verify (T6). All covered.
- Build-time confirmations flagged inline (not placeholders in shipped code): `momrot_bp` url_prefix, Dragonfly cache accessor import path, exact per-model-row location in portfolio.html.
- Type consistency: `compute_buys` returns `{symbol, ltp, qty, amount}`; routes + UI consume those keys; `make_token`/`is_market_open` signatures match their tests.
