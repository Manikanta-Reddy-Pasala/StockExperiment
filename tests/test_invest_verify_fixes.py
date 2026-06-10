"""Verification tests for the FIXED 'Invest More' behaviors.

These exercise the server-side guards in src/web/momrot_routes.py's
invest-execute route (clamp-down, drop-unknown-symbol, multi-vs-single ledger
routing, token replay protection) plus the cache_service.set_if_absent
idempotency primitive.

Everything is monkeypatched: create_app() builds without a live DB/Dragonfly,
and every Fyers/ledger/token touchpoint is stubbed. No network, no real orders,
no DB writes. Follows the monkeypatch style of tests/test_invest_routes_dryrun.py.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

PREFIX = "/admin/momrot/models/momentum_n100_top5_max1"          # single-position model
MULTI_PREFIX = "/admin/momrot/models/momentum_retest_n500"       # multi-holding model


@pytest.fixture
def app_client():
    from src.web import momrot_routes as MR
    from src.web.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return MR, app.test_client()


def _single(buys):
    return {"idle": 57689.0, "broker": 40000.0, "deployable": 40000.0,
            "buys": buys, "is_multi": False}


def _multi(buys):
    return {"idle": 100000.0, "broker": 100000.0, "deployable": 100000.0,
            "buys": buys, "is_multi": True}


def _stub_market(MR, monkeypatch, allowed, *, cash=100000.0, ltp=None):
    """Wire the common happy-path stubs; return the `placed` accumulator."""
    placed = []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest", lambda m, uid=1: allowed)
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: cash)
    monkeypatch.setattr(MR, "_fyers_live_ltp",
                        lambda s, uid=1: (ltp if ltp is not None else None))
    monkeypatch.setattr(
        MR, "_fyers_place_market",
        lambda s, q, side, uid=1: (placed.append((s, q, side)) or
                                   {"ok": True, "result": {"data": {"orderid": "O" + s}}}))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    return placed


# --------------------------------------------------------------------------- #
# 1. execute clamps qty DOWN to the server-derived max                         #
# --------------------------------------------------------------------------- #
def test_execute_clamps_qty_down_to_server_max(app_client, monkeypatch):
    MR, client = app_client
    placed = _stub_market(
        MR, monkeypatch,
        _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]),
        cash=40000.0, ltp=800.0)
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "record_buy", lambda *a, **k: None)

    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 9999}], "token": "t"}).get_json()

    assert d["success"] is True
    assert placed == [("ABC", 49, "BUY")]            # 9999 clamped DOWN to max 49


# --------------------------------------------------------------------------- #
# 2. execute DROPS a client symbol not in the server-derived suggestion        #
# --------------------------------------------------------------------------- #
def test_execute_drops_symbol_not_in_suggestion(app_client, monkeypatch):
    MR, client = app_client
    placed = _stub_market(
        MR, monkeypatch,
        _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]),
        cash=40000.0, ltp=800.0)

    resp = client.post(f"{PREFIX}/invest-execute",
                       json={"buys": [{"symbol": "EVIL", "qty": 10}], "token": "t"})
    d = resp.get_json()

    assert d["success"] is False
    assert placed == []                               # nothing placed for the injected symbol


# --------------------------------------------------------------------------- #
# 3. MULTI model routes to record_buy_multi, NOT record_buy                     #
# --------------------------------------------------------------------------- #
def test_execute_multi_routes_to_record_buy_multi_only(app_client, monkeypatch):
    MR, client = app_client
    multi_rec, single_rec = [], []
    _stub_market(
        MR, monkeypatch,
        _multi([{"symbol": "C", "ltp": 100.0, "qty": 400, "amount": 40000.0},
                {"symbol": "D", "ltp": 100.0, "qty": 400, "amount": 40000.0}]),
        cash=100000.0, ltp=100.0)
    import src.services.trading.multi_holding_service as MH
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(MH, "record_buy_multi",
                        lambda m, s, q, p, fyers_order_id=None: multi_rec.append((m, s, q)))
    monkeypatch.setattr(L, "record_buy",
                        lambda *a, **k: single_rec.append(a))

    d = client.post(f"{MULTI_PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "C", "qty": 200}, {"symbol": "D", "qty": 100}],
                          "token": "t"}).get_json()

    assert d["success"] is True
    assert sorted(multi_rec) == [("momentum_retest_n500", "C", 200),
                                 ("momentum_retest_n500", "D", 100)]
    assert single_rec == []                           # single path NEVER used for multi model


# --------------------------------------------------------------------------- #
# 4. SINGLE model routes to record_buy, NOT record_buy_multi                    #
# --------------------------------------------------------------------------- #
def test_execute_single_routes_to_record_buy_only(app_client, monkeypatch):
    MR, client = app_client
    multi_rec, single_rec = [], []
    _stub_market(
        MR, monkeypatch,
        _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]),
        cash=40000.0, ltp=800.0)
    import src.services.trading.multi_holding_service as MH
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(MH, "record_buy_multi",
                        lambda *a, **k: multi_rec.append(a))
    monkeypatch.setattr(L, "record_buy",
                        lambda m, s, q, p, fyers_order_id=None: single_rec.append((m, s, q)))

    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10}], "token": "t"}).get_json()

    assert d["success"] is True
    assert single_rec == [("momentum_n100_top5_max1", "ABC", 10)]
    assert multi_rec == []                            # multi path NEVER used for single model


# --------------------------------------------------------------------------- #
# 5. reused/expired token -> 409, nothing placed                               #
# --------------------------------------------------------------------------- #
def test_execute_reused_token_returns_409_no_order(app_client, monkeypatch):
    MR, client = app_client
    placed = _stub_market(
        MR, monkeypatch,
        _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]),
        cash=40000.0, ltp=800.0)
    monkeypatch.setattr(MR, "_token_consume", lambda t: False)   # already used

    resp = client.post(f"{PREFIX}/invest-execute",
                       json={"buys": [{"symbol": "ABC", "qty": 10}], "token": "used"})
    d = resp.get_json()

    assert resp.status_code == 409
    assert d["success"] is False
    assert placed == []                               # token guard fires BEFORE any order


# --------------------------------------------------------------------------- #
# 6. cache_service.set_if_absent — true first, false second (NX semantics)      #
# --------------------------------------------------------------------------- #
def test_set_if_absent_true_then_false():
    from src.services.utils.cache_service import CacheService

    class _FakeRedis:
        def __init__(self):
            self._calls = 0

        def set(self, key, value, nx=False, ex=None):
            self._calls += 1
            # SET NX returns truthy when the key was absent, None when present
            return True if self._calls == 1 else None

    svc = CacheService.__new__(CacheService)          # bypass __init__/_connect
    svc.redis_client = _FakeRedis()

    assert svc.set_if_absent("k", "v", 60) is True    # first caller wins
    assert svc.set_if_absent("k", "v", 60) is False   # second caller loses


def test_derive_invest_full_book_tops_up_existing(monkeypatch):
    """Retest full 4-name book + idle cash -> top up the 4 held (no 5th name)."""
    from src.web import momrot_routes as MR
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "model_max_holdings", lambda m: 4)
    monkeypatch.setattr(MR, "_model_idle_cash", lambda m: 40000.0)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 100000.0)
    monkeypatch.setattr(MR, "_model_own_held", lambda m: {"A", "B", "C", "D"})
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 100.0)
    d = MR._derive_invest("momentum_retest_n500")
    assert d["is_multi"] is True
    assert {b["symbol"] for b in d["buys"]} == {"A", "B", "C", "D"}  # tops up held


def test_derive_invest_free_slots_fill_new(monkeypatch):
    """Retest holding 2 of 4 -> fill the 2 free slots with top unheld names."""
    from src.web import momrot_routes as MR
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "model_max_holdings", lambda m: 4)
    monkeypatch.setattr(MR, "_model_idle_cash", lambda m: 40000.0)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 100000.0)
    monkeypatch.setattr(MR, "_model_own_held", lambda m: {"A", "B"})
    monkeypatch.setattr(MR, "_model_ranking_targets",
                        lambda m, uid=1: [{"symbol": s, "ltp": 100.0} for s in ("C", "D", "E")])
    d = MR._derive_invest("momentum_retest_n500")
    syms = {b["symbol"] for b in d["buys"]}
    assert syms == {"C", "D"}            # exactly 2 free slots, no overfill
