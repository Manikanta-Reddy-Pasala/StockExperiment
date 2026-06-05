"""Per-model capital ledger service.

Business logic owning:
  - settings (allocated capital, enabled flag) for each trading model
  - cash + open-position state for each model
  - trade audit log
  - portfolio aggregate stats

Used by live executors (route every buy/sell through ledger), admin/settings
UI (display + edit), and dashboard (per-model + portfolio totals).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import text

from src.models.database import get_database_manager
from src.models.model_ledger_models import (
    ModelLedger, ModelSettings, ModelTrade, ModelHolding,
)

log = logging.getLogger(__name__)


def _normalize_symbol(sym: str) -> str:
    """Normalize bare symbol (HFCL) to Fyers form (NSE:HFCL-EQ) for
    consistent MTM lookup against historical_data table (Fyers symbols)."""
    if not sym or ":" in sym:
        return sym
    return f"NSE:{sym}-EQ"


# Order ids that never identify a unique broker fill, so are never deduped.
_NON_DEDUPABLE_ORDER_IDS = ("", "DRY")


def _order_already_recorded(order_id, existing_ids) -> bool:
    """True if this Fyers order id was already recorded (idempotency guard).

    Prevents a retried / re-invoked executor from double-recording the same
    fill (cash debited twice, qty doubled). Blank / "DRY" / None ids never
    identify a real fill, so are treated as non-dedupable (always False).
    """
    if not order_id or order_id in _NON_DEDUPABLE_ORDER_IDS:
        return False
    return order_id in set(existing_ids)


def partial_sell_outcome(open_qty: int, requested_qty):
    """Resolve a (possibly partial) sell against the open position.

    Returns (sell_qty, remaining_qty, is_full). A partial broker fill must NOT
    book the whole position flat — that strands the unfilled shares at the
    broker while the ledger thinks the model is flat. ``requested_qty`` None or
    <=0 (or >= open) means a full close (defensive: never strand on a bad qty).
    """
    oq = int(open_qty)
    if requested_qty is None:
        return oq, 0, True
    rq = int(requested_qty)
    if rq <= 0 or rq >= oq:
        return oq, 0, True
    return rq, oq - rq, False


def _order_seen(session, model_name: str, order_id) -> bool:
    """DB-backed idempotency check: has model_name already logged this order id?"""
    if not order_id or order_id in _NON_DEDUPABLE_ORDER_IDS:
        return False
    rows = session.query(ModelTrade.fyers_order_id).filter_by(
        model_name=model_name, fyers_order_id=order_id).all()
    return _order_already_recorded(order_id, [r[0] for r in rows])


# Models the system knows about. Adding a new model = append here.
# `enabled` controls auto-seeding default (per-row UI toggle still wins later).
# `default_capital` ₹30K = small live test slug; user can deposit more via UI.
KNOWN_MODELS = [
    {
        "name": "momentum_n100_top5_max1",
        "default_capital": 30000,
        "enabled": True,
        "description": "Equity monthly rotation top-1 from real NSE Nifty 100 by 30d return",
    },
    {
        "name": "momentum_pseudo_n100_adv",
        "default_capital": 30000,
        # LIVE. Universe is rebuilt at each year-start using only data
        # observable at that date — PIT-safe for live deployment.
        "enabled": True,
        "description": "Equity monthly rotation top-1 from pseudo-N100 (top-100 ADV from N500, yearly PIT rebuild)",
    },
    {
        "name": "midcap_narrow_60d_breakout",
        "default_capital": 30000,
        "enabled": True,
        "description": "Equity 60d-high swing on midcap_narrow (event-driven)",
    },
    {
        "name": "n20_daily_large_only",
        "default_capital": 30000,
        "enabled": True,
        "description": "Equity daily rotation top-20-ADV ∩ Nifty 100 by 30d return",
    },
]

# Models intentionally removed from the system. ensure_models_seeded() purges
# their settings+ledger rows on boot IF they carry no trade history — so a
# stale row (left behind by an incomplete cleanup, or seeded by an old image)
# can never silently resurrect a retired model in the UI with a phantom
# allocation. Rows WITH trades are left intact (forensic safety) and logged.
RETIRED_MODELS = [
    "finnifty_ic_otm4_w300_lots5",  # FinNifty options IC — removed 2026-05-25
    "orb_momentum_intraday",        # ORB intraday — ARCHIVED 2026-06-05, no edge
                                    # (lookahead-inflated backtest; faithful −63%)
]

# ---- Order product-type policy (single source of truth) -------------------
# ONLY models listed here trade INTRADAY (MIS, broker auto-square-off + intraday
# margin/cheaper STT). EVERY other model trades CNC (delivery, multi-day hold).
# The executor resolves product via product_for_model() so a swing model can
# never accidentally fire an MIS order, and an intraday model can never hold
# overnight as CNC. Add a new intraday model's name here to switch it to MIS.
INTRADAY_MODELS = {
    # (ORB archived 2026-06-05; no intraday models currently active.)
}

# ---- Multi-holding models: max concurrent positions (K) -------------------
# Models that hold a BASKET of names (positions in model_holdings, not the
# single ledger.open_symbol slot). The value is K — the max concurrent
# positions the model may hold. RiskManager.from_model uses this so a K>1
# model is NOT capped at the env MAX_CONCURRENT (=1 on prod), which would
# wrongly treat momentum_retest_n500 as a single-position model.
MULTI_HOLDING_MODELS = {
    "momentum_retest_n500": 4,   # K4 (see tools/models/momentum_retest_n500/strategy.py)
}


def model_max_holdings(model_name: str):
    """K (max concurrent positions) for a multi-holding model, else None.

    None = single-position model; the caller keeps its env/default concurrency.
    """
    return MULTI_HOLDING_MODELS.get(model_name)


def product_for_model(model_name: str) -> str:
    """Canonical order product for a model: 'INTRADAY' for intraday models,
    'CNC' (delivery) for everything else (the safe default)."""
    return "INTRADAY" if model_name in INTRADAY_MODELS else "CNC"


def ensure_models_seeded() -> None:
    """Create settings+ledger rows for any KNOWN_MODELS missing from DB.

    Safe to call repeatedly. Allocated capital defaults to the value above
    but user can edit via settings UI without losing trade history.
    """
    db = get_database_manager()
    with db.get_session() as s:
        for m in KNOWN_MODELS:
            existing = s.query(ModelSettings).filter_by(model_name=m["name"]).first()
            if existing:
                continue
            s.add(ModelSettings(
                model_name=m["name"],
                enabled=m.get("enabled", True),
                invested_amount=Decimal(m["default_capital"]),
                current_amount=Decimal(m["default_capital"]),
                description=m["description"],
            ))
            # Flush so settings row exists before ledger FK
            s.flush()
            s.add(ModelLedger(
                model_name=m["name"],
                cash=Decimal(m["default_capital"]),
                realized_pnl=Decimal(0),
                total_trades=0,
                wins=0,
                losses=0,
            ))
            s.flush()
            log.info(f"Seeded ledger for {m['name']} cap={m['default_capital']} "
                     f"enabled={m.get('enabled', True)}")

        # Purge retired models so they can't linger in the UI with a phantom
        # allocation. Only delete rows with NO trade history.
        for name in RETIRED_MODELS:
            settings = s.query(ModelSettings).filter_by(model_name=name).first()
            ledger = s.query(ModelLedger).filter_by(model_name=name).first()
            if not settings and not ledger:
                continue
            trades = s.query(ModelTrade).filter_by(model_name=name).count()
            if trades > 0:
                log.warning(f"Retired model {name} has {trades} trades — "
                            "leaving rows intact (manual review needed)")
                continue
            if ledger:
                s.delete(ledger)
            if settings:
                s.delete(settings)
            log.info(f"Purged retired model {name} (0 trades)")


# ---- Settings ----

def get_all_settings() -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = s.query(ModelSettings).order_by(ModelSettings.model_name).all()
        return [_settings_dict(r) for r in rows]


def deposit(model_name: str, amount: float) -> Dict:
    """Add fresh capital to a model.

    Effects:
      invested_amount += amount   (principal in)
      current_amount  += amount   (NAV, before any market move)
      cash            += amount   (immediately deployable)

    Use case: monthly top-up from user's bank. Other models untouched.
    """
    if amount <= 0:
        raise ValueError("deposit amount must be > 0")
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        delta = Decimal(str(amount))
        settings.invested_amount = (settings.invested_amount or Decimal(0)) + delta
        settings.current_amount = (settings.current_amount or Decimal(0)) + delta
        ledger.cash = (ledger.cash or Decimal(0)) + delta
        s.add(ModelTrade(
            model_name=model_name,
            side="DEPOSIT",
            symbol="-",
            qty=0,
            price=Decimal(0),
            value=delta,
            reason="DEPOSIT",
        ))
        log.info(f"{model_name}: deposit ₹{amount:,.0f} (invested now "
                 f"₹{float(settings.invested_amount):,.0f}, current "
                 f"₹{float(settings.current_amount):,.0f}, cash "
                 f"₹{float(ledger.cash):,.0f})")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def withdraw(model_name: str, amount: float) -> Dict:
    """Pull cash out of a model.

    Effects:
      invested_amount -= amount   (principal out; floored at 0)
      current_amount  -= amount   (NAV cache)
      cash            -= amount

    Safety: refuses if cash < amount.
    """
    if amount <= 0:
        raise ValueError("withdraw amount must be > 0")
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        delta = Decimal(str(amount))
        if ledger.cash < delta:
            # Cash is `idle` only — money already sunk into an open position
            # is locked at the broker until that position is sold. Tell the
            # caller exactly that rather than the bare "insufficient cash".
            locked = ""
            if ledger.open_qty and ledger.open_qty > 0:
                pos_cost = float(Decimal(str(ledger.open_qty)) *
                                 (ledger.open_entry_px or Decimal(0)))
                locked = (f" ({ledger.open_symbol} x{int(ledger.open_qty)} "
                          f"locks ₹{pos_cost:,.0f}; sell first to free cash)")
            raise ValueError(
                f"Insufficient cash in {model_name}: have "
                f"₹{float(ledger.cash):,.0f}, want to withdraw "
                f"₹{amount:,.0f}{locked}"
            )
        ledger.cash = ledger.cash - delta
        settings.invested_amount = max(
            Decimal(0), (settings.invested_amount or Decimal(0)) - delta
        )
        settings.current_amount = max(
            Decimal(0), (settings.current_amount or Decimal(0)) - delta
        )
        s.add(ModelTrade(
            model_name=model_name,
            side="WITHDRAW",
            symbol="-",
            qty=0,
            price=Decimal(0),
            value=delta,
            reason="WITHDRAW",
        ))
        log.info(f"{model_name}: withdraw ₹{amount:,.0f} (invested now "
                 f"₹{float(settings.invested_amount):,.0f}, current "
                 f"₹{float(settings.current_amount):,.0f}, cash "
                 f"₹{float(ledger.cash):,.0f})")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def auto_bootstrap_from_json_ledger(json_path: str, model_name: str,
                                    cash_buffer: float = 0.0) -> Dict:
    """Migrate legacy JSON ledger (e.g. momrot_ledger.json) into model_ledger.

    For each open position in the JSON:
      - Sets model's allocated_capital = sum(qty * entry_price) + cash_buffer
      - Adds cash_buffer to ledger.cash (= leftover from last buy)
      - Seeds model_ledger.open_symbol/qty/entry_px/date

    If the model already has allocated_capital > 0 or open_symbol, refuses
    unless reset_model was called first.

    Returns the seeded ledger snapshot.
    """
    import json
    from datetime import date as _date

    try:
        with open(json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Legacy ledger file not found: {json_path}")

    open_positions = data.get("open", [])
    if not open_positions:
        raise ValueError(f"No open positions in {json_path}")

    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        if ledger.open_symbol:
            raise ValueError(
                f"{model_name} already has open position {ledger.open_symbol}, "
                f"reset first"
            )

        # Take the first open position (mc=1 model)
        p = open_positions[0]
        symbol = p["symbol"]
        qty = int(p["qty"])
        entry_px = float(p["entry_price"])
        cost = Decimal(str(qty)) * Decimal(str(entry_px))
        total_allocated = cost + Decimal(str(cash_buffer))

        # Increase invested_amount + current_amount + ledger cash by buffer
        # (cash leftover), then deduct position cost (now in market value).
        settings.invested_amount = (settings.invested_amount or Decimal(0)) + total_allocated
        settings.current_amount = (settings.current_amount or Decimal(0)) + total_allocated
        ledger.cash = (ledger.cash or Decimal(0)) + total_allocated

        # Seed position (eats cost from cash, leaves cash_buffer)
        # Normalize symbol to Fyers format (NSE:XXX-EQ) so MTM lookup
        # against historical_data table works — that table stores Fyers form.
        ledger.open_symbol = _normalize_symbol(symbol)
        ledger.open_qty = qty
        ledger.open_entry_px = Decimal(str(entry_px))
        # entry_date from JSON if present, else today
        entry_ts = p.get("entry_ts") or data.get("updated_at")
        if entry_ts:
            try:
                ledger.open_entry_date = datetime.fromisoformat(entry_ts.split("T")[0]).date()
            except Exception:
                ledger.open_entry_date = _date.today()
        else:
            ledger.open_entry_date = _date.today()
        ledger.cash = ledger.cash - cost

        # Audit trail
        s.add(ModelTrade(
            model_name=model_name,
            side="DEPOSIT",
            symbol="-",
            qty=0,
            price=Decimal(0),
            value=total_allocated,
            reason="BOOTSTRAP_DEPOSIT",
        ))
        s.add(ModelTrade(
            model_name=model_name,
            side="BUY",
            symbol=symbol,
            qty=qty,
            price=Decimal(str(entry_px)),
            value=cost,
            reason="BOOTSTRAP_POSITION",
        ))
        log.info(
            f"Bootstrapped {model_name}: position {symbol} x{qty} @ ₹{entry_px} "
            f"(cost ₹{float(cost):,.0f}) + cash buffer ₹{cash_buffer:,.0f} "
            f"= total deposited ₹{float(total_allocated):,.0f}"
        )
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
            "bootstrapped_position": {
                "symbol": symbol, "qty": qty, "entry_px": entry_px,
                "cost": float(cost),
            },
            "cash_buffer": cash_buffer,
        }


def reset_model(model_name: str) -> Dict:
    """Hard reset: zero invested_amount + current_amount, zero cash, zero
    realized_pnl, clear open position, reset counters. Trade audit log is NOT
    deleted (history kept). Use to start fresh before depositing real cost basis.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        settings.invested_amount = Decimal(0)
        settings.current_amount = Decimal(0)
        ledger.cash = Decimal(0)
        ledger.realized_pnl = Decimal(0)
        ledger.total_trades = 0
        ledger.wins = 0
        ledger.losses = 0
        ledger.open_symbol = None
        ledger.open_qty = None
        ledger.open_entry_px = None
        ledger.open_entry_date = None
        log.info(f"{model_name}: model reset to zero")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def set_enabled(model_name: str, enabled: bool) -> Dict:
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        if not settings:
            raise ValueError(f"Unknown model: {model_name}")
        settings.enabled = enabled
        return _settings_dict(settings)


def set_signals_only(model_name: str, signals_only: bool) -> Dict:
    """Flip a model to observe-only (signals_only=True) or live (False).

    When True the model still emits signals + ranking, but executors place no
    real orders and don't mutate the ledger. Independent of `enabled`.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        if not settings:
            raise ValueError(f"Unknown model: {model_name}")
        settings.signals_only = signals_only
        return _settings_dict(settings)


# ---- Ledger snapshot ----

def get_ledger(model_name: str) -> Optional[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l:
            return None
        return _ledger_dict(l)


def get_all_ledgers() -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = s.query(ModelLedger).order_by(ModelLedger.model_name).all()
        return [_ledger_dict(r) for r in rows]


# ---- Bootstrap an existing position (e.g. momentum_n100 already live) ----

def seed_position(model_name: str, symbol: str, qty: int,
                  entry_px: float, entry_date_str: str) -> Dict:
    """Manually seed a model's open position (no Fyers order placed)."""
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l:
            raise ValueError(f"Unknown model: {model_name}")
        if l.open_symbol:
            raise ValueError(
                f"{model_name} already has open position {l.open_symbol}, "
                f"reset first"
            )
        cost = Decimal(str(qty)) * Decimal(str(entry_px))
        if l.cash < cost:
            raise ValueError(
                f"Not enough cash in {model_name} ledger "
                f"(₹{float(l.cash):,.0f}) to seed position cost ₹{float(cost):,.0f}"
            )
        norm = _normalize_symbol(symbol)
        l.open_symbol = norm
        l.open_qty = qty
        l.open_entry_px = Decimal(str(entry_px))
        l.open_entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        l.cash = l.cash - cost
        s.add(ModelTrade(
            model_name=model_name,
            side="BUY",
            symbol=norm,
            qty=qty,
            price=Decimal(str(entry_px)),
            value=cost,
            reason="SEED",
        ))
        return _ledger_dict(l)


def reset_position(model_name: str) -> Dict:
    """Mark position as flat, returning estimated NAV back to cash.

    DOES NOT place any Fyers order — manual reconciliation tool only.
    """
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not l.open_symbol:
            return _ledger_dict(l) if l else None
        cost = Decimal(str(l.open_qty)) * l.open_entry_px
        l.cash = l.cash + cost
        l.open_symbol = None
        l.open_qty = None
        l.open_entry_px = None
        l.open_entry_date = None
        return _ledger_dict(l)


# ---- Live buy/sell hooks (called by executor) ----

def _compute_real_charges(side: str, qty: int, price: float,
                          product: str = "CNC") -> Decimal:
    """Return full SEBI-rate charges total (brokerage+STT+exchange+SEBI+stamp+GST+DP).

    Falls back to a flat-rate approximation if the calculator isn't importable
    (defensive — module is in tools/, not src/).
    """
    try:
        from tools.live.broker_charges import compute_charges
        br = compute_charges(side, qty, price, product)
        return Decimal(str(br.get("total", 0)))
    except Exception:
        approx = Decimal("20")
        if side.upper() == "SELL":
            approx += Decimal(str(qty)) * Decimal(str(price)) * Decimal("0.001")
        return approx


def record_buy(model_name: str, symbol: str, qty: int, price: float,
               brokerage: float = None, fyers_order_id: str = None,
               product: str = "CNC") -> Dict:
    """Record a BUY fill.

    Cash flow: cash -= (qty*price + charges)
    NAV flow:  current_amount -= charges
               (qty*price stays as position value, so net NAV moves only by fees)

    Same-symbol behavior: ACCUMULATES into the open position (qty += new_qty,
    entry_px = weighted average). Different-symbol while holding still raises.
    Accumulation matters when an upstream race / UI bug / retry path produces
    multiple Fyers fills on the same symbol — the previous "raise on already
    holding" guard dropped 2nd+ fills, leaving Fyers and ledger out of sync
    (the May 18 ADANIPOWER incident lost track of ~134 shares this way).

    charges = full SEBI-rate broker_charges.compute_charges (brokerage + exchange
    + SEBI + stamp + GST), not the legacy ₹20 flat. brokerage kwarg is ignored —
    kept for back-compat with older call sites.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not settings:
            raise ValueError(f"Unknown model: {model_name}")
        # Idempotency: a retried/re-invoked executor must not re-apply a fill.
        if _order_seen(s, model_name, fyers_order_id):
            log.warning(f"{model_name}: BUY order {fyers_order_id} already "
                        f"recorded — skipping duplicate (idempotent).")
            return _ledger_dict(l)
        norm = _normalize_symbol(symbol)
        # Different-symbol guard remains — model logic owns single-symbol per
        # position; rotating to a different symbol must SELL first.
        if l.open_symbol and l.open_symbol != norm:
            raise ValueError(
                f"{model_name}: already holding {l.open_symbol}, cannot buy {norm}"
            )
        qty_d = Decimal(str(qty))
        price_d = Decimal(str(price))
        charges = _compute_real_charges("BUY", qty, price, product)
        cost = qty_d * price_d + charges
        if l.cash < cost:
            # Fyers fill already executed — cannot raise, ledger MUST follow
            # broker truth or drift forever. Log shortfall + allow cash dip
            # (rare edge: slippage > sizer's charges buffer).
            shortfall = float(cost - l.cash)
            log.warning(
                f"{model_name}: cash shortfall ₹{shortfall:,.2f} on BUY "
                f"{qty}x{symbol}@{price} (cost=₹{float(cost):,.2f}, "
                f"cash=₹{float(l.cash):,.2f}) — ledger absorbs"
            )
            try:
                from tools.live.telegram_notify import send as _tg
                _tg(
                    f"⚠️ *Post-fill cash shortfall absorbed*\n"
                    f"Model: `{model_name}`\n"
                    f"Symbol: `{symbol}` x{qty} @ ₹{float(price):,.2f}\n"
                    f"Cost: ₹{float(cost):,.2f}  Cash: ₹{float(l.cash):,.2f}\n"
                    f"Short: ₹{shortfall:,.2f} — ledger cash will go negative"
                )
            except Exception:
                pass
        l.cash = l.cash - cost
        # FIX 7 — absorbing the cost can drive cash negative (ledger MUST follow
        # broker truth). A negative balance means the model can no longer enter
        # and needs manual reconciliation, so escalate it as a CRITICAL alert
        # (not the quiet shortfall line above). notify_order funnels to DB +
        # Telegram; both calls are guarded so a notify/log issue never breaks
        # the ledger write.
        if l.cash < 0:
            new_cash = float(l.cash)
            try:
                log.error(
                    f"{model_name}: ledger cash NEGATIVE (₹{new_cash:.2f}) after "
                    f"{symbol} buy — model will stop entering; manual "
                    f"reconciliation needed"
                )
            except Exception:
                pass
            try:
                from src.services.notification_service import notify_order
                notify_order(
                    f"🚨 CRITICAL: {model_name} ledger cash NEGATIVE "
                    f"(₹{new_cash:.2f}) after {symbol} buy — model will stop "
                    f"entering; manual reconciliation needed"
                )
            except Exception:
                pass
        if l.open_symbol == norm and l.open_qty:
            # Accumulate same-symbol fill: weighted-average entry price.
            prev_qty = Decimal(str(l.open_qty))
            prev_px = l.open_entry_px or Decimal(0)
            total_qty = prev_qty + qty_d
            new_avg = ((prev_qty * prev_px) + (qty_d * price_d)) / total_qty
            l.open_qty = int(total_qty)
            l.open_entry_px = new_avg
            # open_entry_date stays as original (earliest fill) for hold-period
            # accounting; weighted avg doesn't change first-entry date.
            log.info(
                f"{model_name}: ACCUMULATED {norm} +{qty}@{price} "
                f"(was {int(prev_qty)}@{float(prev_px):.4f}, "
                f"now {int(total_qty)}@{float(new_avg):.4f})"
            )
        else:
            l.open_symbol = norm
            l.open_qty = qty
            l.open_entry_px = price_d
            l.open_entry_date = date.today()
        settings.current_amount = (settings.current_amount or Decimal(0)) - charges
        s.add(ModelTrade(
            model_name=model_name,
            side="BUY",
            symbol=norm,
            qty=qty,
            price=price_d,
            value=cost,
            reason="ENTRY",
            fyers_order_id=fyers_order_id,
        ))
        return _ledger_dict(l)


def record_sell(model_name: str, exit_price: float, reason: str,
                brokerage: float = None, stt_pct: float = None,
                fyers_order_id: str = None, product: str = "CNC",
                qty: int = None) -> Dict:
    """Record a SELL fill.

    Cash flow: cash += (sell_qty*exit_price - charges)
    NAV flow:  full close  -> current_amount = cash (position flat)
               partial sell -> current_amount = cash + remaining_qty*entry_px

    ``qty`` = the ACTUAL filled quantity. None / >= open / <=0 means a full
    close (back-compat + defensive). A genuine partial fill (0<qty<open) sells
    only that many shares and RETAINS the residual open position, so the ledger
    never marks flat while real shares remain at the broker.

    charges = full SEBI-rate broker_charges.compute_charges (incl. STT, DP, GST).
    brokerage + stt_pct kwargs are ignored — kept for back-compat.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not settings:
            raise ValueError(f"{model_name}: unknown model")
        # Idempotency FIRST: a replayed SELL (same order_id) must be skipped
        # silently — even after a full close left the position flat — rather
        # than raising "no open position" and firing a false ledger-FAIL alert.
        if _order_seen(s, model_name, fyers_order_id):
            log.warning(f"{model_name}: SELL order {fyers_order_id} already "
                        f"recorded — skipping duplicate (idempotent).")
            return _ledger_dict(l)
        if not l.open_symbol or not l.open_qty:
            raise ValueError(f"{model_name}: no open position")
        sell_qty, remaining_qty, is_full = partial_sell_outcome(l.open_qty, qty)
        entry_px = l.open_entry_px
        proc = Decimal(str(sell_qty)) * Decimal(str(exit_price))
        fees = _compute_real_charges("SELL", sell_qty, float(exit_price), product)
        net = proc - fees
        # P&L = exit proceeds (net of sell-side charges) minus entry cost. Entry
        # cost in the ledger already reflects buy-side charges (cash was
        # decremented by qty*entry_px + buy_charges at record_buy time), so the
        # cumulative realized_pnl across a round-trip captures BOTH legs.
        pnl = net - (Decimal(str(sell_qty)) * entry_px)
        l.cash = l.cash + net
        l.realized_pnl = (l.realized_pnl or Decimal(0)) + pnl
        l.total_trades = (l.total_trades or 0) + 1
        if pnl > 0:
            l.wins = (l.wins or 0) + 1
        elif pnl < 0:
            l.losses = (l.losses or 0) + 1   # exact break-even counts as neither
        symbol = l.open_symbol
        if is_full:
            l.open_symbol = None
            l.open_qty = None
            l.open_entry_px = None
            l.open_entry_date = None
            # Position flat — current_amount snaps to cash (no open MTM)
            settings.current_amount = l.cash
        else:
            # PARTIAL exit — keep the residual position; only the sold shares
            # leave. NAV = realized cash + remaining shares at entry cost.
            l.open_qty = remaining_qty
            settings.current_amount = l.cash + Decimal(str(remaining_qty)) * entry_px
            log.warning(f"{model_name}: PARTIAL SELL {sell_qty}/{sell_qty + remaining_qty} "
                        f"{symbol} — retaining {remaining_qty} open shares.")
        s.add(ModelTrade(
            model_name=model_name,
            side="SELL",
            symbol=symbol,
            qty=sell_qty,
            price=Decimal(str(exit_price)),
            value=net,
            pnl=pnl,
            reason=reason,
            fyers_order_id=fyers_order_id,
        ))
        return _ledger_dict(l)


# ---- Trade history ----

def get_trades(model_name: str, limit: int = 50) -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = (s.query(ModelTrade)
                  .filter_by(model_name=model_name)
                  .order_by(ModelTrade.trade_at.desc())
                  .limit(limit).all())
        return [_trade_dict(r) for r in rows]


# ---- Aggregate ----

def get_portfolio_stats(price_lookup=None, quote_lookup=None) -> Dict:
    """Return per-model stats + portfolio total.

    price_lookup: optional callable(symbol) -> last_price for MTM of open
    positions. If None, uses last entry price as proxy.
    quote_lookup: optional callable(symbol) -> {"ltp", "prev_close"} for the
    live MTM + day-change. Enables P&L the way the user wants it:
      - unrealized / open P&L  = qty × (live LTP − entry)        [Fyers LTP]
      - realized P&L           = Σ closed-trade pnl from model_trades
      - total P&L              = realized + unrealized
      - TODAY's P&L            = Σ qty × (LTP − prev_close) on open holds
                                 + realized from trades closed today
    Falls back to price_lookup (ltp only; no day-change) when not given.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings_rows = {x.model_name: x for x in s.query(ModelSettings).all()}
        ledger_rows = s.query(ModelLedger).all()
        # Multi-holding models (e.g. momentum_retest_n500) keep their positions
        # in model_holdings, not ledger.open_symbol. Group them once so each
        # model's NAV / position_value includes ALL its holdings (otherwise a
        # K>1 model shows only cash and its real shares look "untracked").
        holdings_by_model: Dict[str, list] = {}
        for h in s.query(ModelHolding).all():
            holdings_by_model.setdefault(h.model_name, []).append(h)

        # Approx broker charges per model = sum of model_trades.charges_inr for
        # trades that ACTUALLY hit the broker (a real fyers_order_id). This is
        # the right discriminator, NOT the model's signals_only flag: an
        # observe-mode model can still have placed real orders (those DO incur
        # charges and must count), while a paper trade (no order_id) must not.
        charges_by_model: Dict[str, float] = {}
        try:
            from sqlalchemy import func as _func, or_ as _or
            for mn, csum in (s.query(ModelTrade.model_name,
                                     _func.coalesce(_func.sum(ModelTrade.charges_inr), 0))
                               .filter(ModelTrade.side.in_(("BUY", "SELL")))
                               # real broker charge = a placed order (has order_id)
                               # OR a linked real broker holding (LINK_FYERS_POSITION,
                               # which has no order_id but is a real position that
                               # cost money). Only genuine paper/sim trades excluded.
                               .filter(_or(_func.coalesce(ModelTrade.fyers_order_id, "") != "",
                                           ModelTrade.reason == "LINK_FYERS_POSITION"))
                               .group_by(ModelTrade.model_name).all()):
                charges_by_model[mn] = float(csum or 0)
        except Exception:
            charges_by_model = {}

        # Realized P&L per model FROM TRADES (closed-trade pnl on real broker
        # sells), + the slice realized TODAY — drives total + today's P&L
        # without trusting the stored ledger.realized_pnl aggregate.
        realized_by_model: Dict[str, float] = {}
        today_realized_by_model: Dict[str, float] = {}
        try:
            from sqlalchemy import text as _t
            for r in s.execute(_t(
                "SELECT model_name, COALESCE(SUM(pnl),0) FROM model_trades "
                "WHERE side='SELL' AND pnl IS NOT NULL AND COALESCE(fyers_order_id,'')<>'' "
                "GROUP BY model_name")).fetchall():
                realized_by_model[r[0]] = float(r[1] or 0)
            for r in s.execute(_t(
                "SELECT model_name, COALESCE(SUM(pnl),0) FROM model_trades "
                "WHERE side='SELL' AND pnl IS NOT NULL AND COALESCE(fyers_order_id,'')<>'' "
                "AND trade_at::date = CURRENT_DATE GROUP BY model_name")).fetchall():
                today_realized_by_model[r[0]] = float(r[1] or 0)
        except Exception:
            pass

        def _q(sym):
            """{'ltp','prev_close'} for sym — quote_lookup if given, else
            price_lookup (ltp only, prev_close 0 → today MTM falls back to 0)."""
            if quote_lookup:
                try:
                    qd = quote_lookup(sym) or {}
                    return {"ltp": float(qd.get("ltp") or 0),
                            "prev_close": float(qd.get("prev_close") or 0)}
                except Exception:
                    pass
            lp = 0.0
            if price_lookup:
                try:
                    lp = float(price_lookup(sym) or 0)
                except Exception:
                    lp = 0.0
            return {"ltp": lp, "prev_close": 0.0}

        models = []
        total_allocated = Decimal(0)
        total_nav = Decimal(0)
        total_realized = Decimal(0)
        total_trades = 0
        total_charges_all = 0.0
        total_unrealized = 0.0
        total_today_pnl = 0.0
        total_realized_trades = 0.0

        for l in ledger_rows:
            cfg = settings_rows.get(l.model_name)
            cash = l.cash or Decimal(0)
            pos_value = Decimal(0)
            mtm_price = None
            unrealized = 0.0       # Σ qty × (LTP − entry)        — open P&L (Fyers LTP)
            today_unrealized = 0.0  # Σ qty × (LTP − prev_close)  — day MTM change
            if l.open_symbol and l.open_qty:
                q = _q(l.open_symbol)
                ltp = q["ltp"]
                entry = float(l.open_entry_px or 0)
                if ltp <= 0:
                    ltp = entry  # no live price → flat MTM
                mtm_price = ltp
                qty = int(l.open_qty or 0)
                pos_value = Decimal(str(ltp)) * Decimal(str(qty))
                if entry > 0:
                    unrealized += qty * (ltp - entry)
                if q["prev_close"] > 0:
                    today_unrealized += qty * (ltp - q["prev_close"])

            # Multi-holding positions: MTM each, add to pos_value + unrealized.
            holdings_list = []
            for h in holdings_by_model.get(l.model_name, []):
                h_qty = int(h.qty or 0)
                if h_qty <= 0:
                    continue
                hq = _q(h.symbol)
                h_ltp = hq["ltp"]
                h_entry = float(h.entry_px or 0)
                if h_ltp <= 0:
                    h_ltp = h_entry
                h_val = Decimal(str(h_ltp)) * Decimal(str(h_qty))
                pos_value += h_val
                if h_entry > 0:
                    unrealized += h_qty * (h_ltp - h_entry)
                if hq["prev_close"] > 0:
                    today_unrealized += h_qty * (h_ltp - hq["prev_close"])
                holdings_list.append({
                    "symbol": h.symbol,
                    "qty": h_qty,
                    "entry_px": float(h.entry_px) if h.entry_px else None,
                    "entry_date": h.entry_date.isoformat() if h.entry_date else None,
                    "mtm_price": h_ltp,
                    "value": float(h_val),
                })

            nav = cash + pos_value
            invested = cfg.invested_amount if cfg else Decimal(0)
            current_cache = cfg.current_amount if cfg else Decimal(0)
            pnl_total = nav - invested
            return_pct = (
                float(pnl_total / invested * 100) if invested > 0 else 0
            )
            # P&L the user wants: realized from TRADES, open from live (Fyers LTP),
            # total = realized + unrealized, today = day MTM + realized today.
            realized_trades = realized_by_model.get(l.model_name, 0.0)
            today_realized = today_realized_by_model.get(l.model_name, 0.0)
            total_pnl_trades = round(realized_trades + unrealized, 2)
            today_pnl = round(today_unrealized + today_realized, 2)

            models.append({
                "model_name": l.model_name,
                "enabled": bool(cfg and cfg.enabled),
                "signals_only": bool(cfg and getattr(cfg, "signals_only", False)),
                "invested_amount": float(invested),
                "current_amount": float(current_cache),
                # Legacy alias for any UI still reading old field name
                "allocated_capital": float(invested),
                "cash": float(cash),
                "position_value": float(pos_value),
                "nav": float(nav),
                "pnl_total": float(pnl_total),
                "return_pct": round(return_pct, 2),
                "realized_pnl": float(l.realized_pnl or 0),
                # P&L (user spec): open=live LTP MTM, realized=from trades,
                # total=realized+unrealized, today=day MTM + realized today.
                "unrealized_pnl": round(unrealized, 2),
                "realized_pnl_trades": round(realized_trades, 2),
                "total_pnl_trades": total_pnl_trades,
                "today_pnl": today_pnl,
                "today_realized": round(today_realized, 2),
                "today_unrealized": round(today_unrealized, 2),
                "open_symbol": l.open_symbol,
                "open_qty": l.open_qty,
                "open_entry_px": float(l.open_entry_px) if l.open_entry_px else None,
                "open_entry_date": l.open_entry_date.isoformat() if l.open_entry_date else None,
                "open_mtm_price": mtm_price,
                "holdings": holdings_list,
                "is_multi": bool(holdings_list),
                "total_trades": l.total_trades or 0,
                "wins": l.wins or 0,
                "losses": l.losses or 0,
                "win_rate_pct": round(
                    100.0 * (l.wins or 0) / max(1, l.total_trades or 0), 1
                ),
                # Lifetime approx broker charges for this model + net-of-charges P&L.
                "total_charges": round(charges_by_model.get(l.model_name, 0.0), 2),
                "net_realized_pnl": round(float(l.realized_pnl or 0)
                                          - charges_by_model.get(l.model_name, 0.0), 2),
            })

            total_allocated += invested
            total_nav += nav
            total_realized += l.realized_pnl or Decimal(0)
            total_trades += l.total_trades or 0
            total_unrealized += unrealized
            total_today_pnl += today_pnl
            total_realized_trades += realized_trades
            # charges_by_model already counts only broker-executed trades (real
            # fyers_order_id), so paper trades are excluded at the source and an
            # observe model's REAL trades are correctly included here.
            total_charges_all += charges_by_model.get(l.model_name, 0.0)

        total_pnl = total_nav - total_allocated
        total_return_pct = (
            float(total_pnl / total_allocated * 100) if total_allocated > 0 else 0
        )

        return {
            "models": models,
            "total": {
                "invested_amount": float(total_allocated),
                "current_amount": float(total_nav),
                # Legacy aliases
                "allocated_capital": float(total_allocated),
                "nav": float(total_nav),
                "pnl_total": float(total_pnl),
                "return_pct": round(total_return_pct, 2),
                "realized_pnl": float(total_realized),
                "total_trades": total_trades,
                # Aggregate P&L (user spec): open=live, realized=trades,
                # total=realized+unrealized, today=day MTM + realized today.
                "unrealized_pnl": round(total_unrealized, 2),
                "realized_pnl_trades": round(total_realized_trades, 2),
                "total_pnl_trades": round(total_realized_trades + total_unrealized, 2),
                "today_pnl": round(total_today_pnl, 2),
                # Formula estimate (model_trades) — reference. The headline
                # "txn_charges" is set by the /models/portfolio endpoint from the
                # Fyers API (_fyers_account_txn_charges); default to formula here.
                "total_charges": round(total_charges_all, 2),
                "net_realized_pnl": round(float(total_realized) - total_charges_all, 2),
                "txn_charges": round(total_charges_all, 2),
            },
            "as_of": datetime.now().isoformat(),  # IST (container TZ=Asia/Kolkata)
        }


# ---- internal helpers ----

def _settings_dict(s: ModelSettings) -> Dict:
    return {
        "model_name": s.model_name,
        "enabled": s.enabled,
        "signals_only": bool(getattr(s, "signals_only", False)),
        "invested_amount": float(s.invested_amount or 0),
        "current_amount": float(s.current_amount or 0),
        # Legacy alias for any caller still using old field name
        "allocated_capital": float(s.invested_amount or 0),
        "description": s.description,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _ledger_dict(l: ModelLedger) -> Dict:
    return {
        "model_name": l.model_name,
        "cash": float(l.cash or 0),
        "open_symbol": l.open_symbol,
        "open_qty": l.open_qty,
        "open_entry_px": float(l.open_entry_px) if l.open_entry_px else None,
        "open_entry_date": l.open_entry_date.isoformat() if l.open_entry_date else None,
        "realized_pnl": float(l.realized_pnl or 0),
        "total_trades": l.total_trades or 0,
        "wins": l.wins or 0,
        "losses": l.losses or 0,
        "win_rate_pct": round(
            100.0 * (l.wins or 0) / max(1, l.total_trades or 0), 1
        ),
        "updated_at": l.updated_at.isoformat() if l.updated_at else None,
    }


def _trade_dict(t: ModelTrade) -> Dict:
    # Stored approx charges (before_insert listener / backfill). Fall back to an
    # on-the-fly estimate for any legacy row whose column is still null so the
    # UI never shows a blank.
    charges = float(t.charges_inr) if t.charges_inr is not None else None
    if charges is None and (t.side or "").upper() in ("BUY", "SELL"):
        try:
            from tools.live.broker_charges import compute_charges
            charges = compute_charges(t.side, int(t.qty or 0), float(t.price or 0),
                                      product_for_model(t.model_name)).get("total", 0.0)
        except Exception:
            charges = 0.0
    return {
        "id": t.id,
        "model_name": t.model_name,
        "side": t.side,
        "symbol": t.symbol,
        "qty": t.qty,
        "price": float(t.price),
        "value": float(t.value),
        "pnl": float(t.pnl) if t.pnl is not None else None,
        "charges": round(charges, 2) if charges is not None else 0.0,
        "reason": t.reason,
        "fyers_order_id": t.fyers_order_id,
        "trade_at": t.trade_at.isoformat() if t.trade_at else None,
    }
