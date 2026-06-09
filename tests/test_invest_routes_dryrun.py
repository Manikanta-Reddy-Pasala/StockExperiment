"""Route tests for per-model Invest More — fully mocked, no real orders/DB.

create_app() builds without a live DB/Dragonfly (it warns and degrades). We
exercise the routes via the test client and monkeypatch every Fyers/ledger
touchpoint. The execute route RE-DERIVES the allowed buys server-side via
_derive_invest and CLAMPS the client request to it (decrease-only), so tests
stub _derive_invest to define the allowed set. No network, no orders.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

PREFIX = "/admin/momrot/models/momentum_n100_top5_max1"
MULTI_PREFIX = "/admin/momrot/models/momentum_retest_n500"


@pytest.fixture
def app_client(monkeypatch):
    from src.web import momrot_routes as MR
    from src.web.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return MR, app.test_client()


def _single(buys):
    return {"idle": 57689.0, "broker": 40000.0, "deployable": 40000.0,
            "buys": buys, "is_multi": False}


def test_invest_preview_returns_suggestion(app_client, monkeypatch):
    MR, client = app_client
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]))
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    d = client.get(f"{PREFIX}/invest-preview").get_json()
    assert d["success"] is True
    assert d["deployable"] == 40000.0
    assert d["buys"][0] == {"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}
    assert "token" in d and d["market_open"] is True


def test_execute_blocks_when_market_closed(app_client, monkeypatch):
    MR, client = app_client
    monkeypatch.setattr(MR, "is_market_open_now", lambda: False)
    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10}], "token": "t"}).get_json()
    assert d["success"] is False and "market" in d["error"].lower()


def test_execute_places_and_records_single(app_client, monkeypatch):
    MR, client = app_client
    placed, recorded = [], []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]))
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 800.0)
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda s, q, side, uid=1: (placed.append((s, q, side)) or
                                                   {"ok": True, "result": {"data": {"orderid": "OID1"}}}))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "record_buy",
                        lambda m, s, q, p, fyers_order_id=None: recorded.append((m, s, q)))
    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10}], "token": "t"}).get_json()
    assert d["success"] is True
    assert placed == [("ABC", 10, "BUY")]            # qty 10 honored (<= max 49)
    assert recorded == [("momentum_n100_top5_max1", "ABC", 10)]
    assert d["deployed"] == 8000.0


def test_execute_clamps_qty_to_server_max(app_client, monkeypatch):
    """Client cannot INCREASE beyond the re-derived max (decrease-only)."""
    MR, client = app_client
    placed = []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]))
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 800.0)
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda s, q, side, uid=1: (placed.append((s, q)) or
                                                   {"ok": True, "result": {"data": {"orderid": "X"}}}))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "record_buy", lambda *a, **k: None)
    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 9999}], "token": "t"}).get_json()
    assert d["success"] is True
    assert placed == [("ABC", 49)]                   # clamped down to the max


def test_execute_rejects_unknown_symbol(app_client, monkeypatch):
    """A symbol not in the server-derived suggestion is dropped → nothing to do."""
    MR, client = app_client
    placed = []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]))
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda *a, **k: placed.append(a) or {"ok": True})
    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "EVIL", "qty": 9999}], "token": "t"}).get_json()
    assert d["success"] is False
    assert placed == []                               # no order placed


def test_execute_rejects_reused_token(app_client, monkeypatch):
    MR, client = app_client
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: _single([{"symbol": "ABC", "ltp": 800.0, "qty": 49, "amount": 39200.0}]))
    monkeypatch.setattr(MR, "_token_consume", lambda t: False)
    d = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10}], "token": "used"}).get_json()
    assert d["success"] is False


def test_execute_multi_uses_record_buy_multi(app_client, monkeypatch):
    """Retest (multi-holding) records via record_buy_multi, NOT record_buy."""
    MR, client = app_client
    placed, multi_rec, single_rec = [], [], []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_derive_invest",
                        lambda m, uid=1: {"idle": 100000.0, "broker": 100000.0, "deployable": 100000.0,
                                          "is_multi": True,
                                          "buys": [{"symbol": "C", "ltp": 100.0, "qty": 400, "amount": 40000.0},
                                                   {"symbol": "D", "ltp": 100.0, "qty": 400, "amount": 40000.0}]})
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 100000.0)
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 100.0)
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda s, q, side, uid=1: (placed.append((s, q)) or
                                                   {"ok": True, "result": {"data": {"orderid": "O" + s}}}))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    import src.services.trading.multi_holding_service as MH
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(MH, "record_buy_multi",
                        lambda m, s, q, p, fyers_order_id=None: multi_rec.append((s, q)))
    monkeypatch.setattr(L, "record_buy",
                        lambda *a, **k: single_rec.append(a))
    d = client.post(f"{MULTI_PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "C", "qty": 200}, {"symbol": "D", "qty": 400}], "token": "t"}).get_json()
    assert d["success"] is True
    assert sorted(placed) == [("C", 200), ("D", 400)]
    assert sorted(multi_rec) == [("C", 200), ("D", 400)]
    assert single_rec == []                           # single-position path NOT used


def test_execute_empty_buys_rejected(app_client, monkeypatch):
    MR, client = app_client
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    d = client.post(f"{PREFIX}/invest-execute", json={"buys": [], "token": "x"}).get_json()
    assert d["success"] is False
