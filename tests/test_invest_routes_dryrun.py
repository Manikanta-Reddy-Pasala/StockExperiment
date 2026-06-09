"""Route tests for per-model Invest More — fully mocked, no real orders/DB.

create_app() builds without a live DB/Dragonfly (it warns and degrades), so we
exercise the routes via the test client and monkeypatch every Fyers/ledger
touchpoint. No network, no orders.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

PREFIX = "/admin/momrot/models/momentum_n100_top5_max1"


@pytest.fixture
def app_client(monkeypatch):
    from src.web import momrot_routes as MR
    # read-side stubs
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_model_targets",
                        lambda m, uid=1: [{"symbol": "ABC", "ltp": 800.0}])
    monkeypatch.setattr(MR, "_fyers_holdings", lambda uid=1: [])
    monkeypatch.setattr(MR, "_model_idle_cash", lambda m: 57689.0)
    from src.web.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return MR, app.test_client()


def test_invest_preview_returns_sized_buy(app_client):
    MR, client = app_client
    r = client.get(f"{PREFIX}/invest-preview")
    d = r.get_json()
    assert d["success"] is True
    assert d["deployable"] == 40000.0
    assert d["buys"][0]["symbol"] == "ABC"
    assert d["buys"][0]["qty"] == int((40000 * 0.995) // 800)
    assert "token" in d and "market_open" in d


def test_invest_execute_blocks_when_market_closed(app_client):
    MR, client = app_client
    import pytest as _p
    MR_module = MR
    # force market closed
    import src.web.momrot_routes as M
    M.is_market_open_now = lambda: False
    r = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "deadbeef"})
    d = r.get_json()
    assert d["success"] is False
    assert "market" in d["error"].lower()


def test_invest_execute_places_and_records(app_client, monkeypatch):
    MR, client = app_client
    placed = []
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_token_consume", lambda t: True)   # fresh token
    monkeypatch.setattr(MR, "_fyers_available_cash", lambda uid=1: 40000.0)
    monkeypatch.setattr(MR, "_fyers_live_ltp", lambda s, uid=1: 800.0)
    monkeypatch.setattr(MR, "_fyers_holdings", lambda uid=1: [])
    monkeypatch.setattr(MR, "_fyers_place_market",
                        lambda sym, qty, side, uid=1: (placed.append((sym, qty, side)) or
                                                       {"ok": True, "result": {"data": {"orderid": "OID1"}}}))
    monkeypatch.setattr(MR, "_notify_tg", lambda *a, **k: None)
    recorded = []
    import src.services.trading.model_ledger_service as L
    monkeypatch.setattr(L, "record_buy",
                        lambda m, s, q, p, fyers_order_id=None: recorded.append((m, s, q)))
    r = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "tok1"})
    d = r.get_json()
    assert d["success"] is True
    assert placed == [("ABC", 10, "BUY")]
    assert recorded == [("momentum_n100_top5_max1", "ABC", 10)]
    assert d["deployed"] == 8000.0


def test_invest_execute_rejects_reused_token(app_client, monkeypatch):
    MR, client = app_client
    monkeypatch.setattr(MR, "is_market_open_now", lambda: True)
    monkeypatch.setattr(MR, "_token_consume", lambda t: False)  # already used
    r = client.post(f"{PREFIX}/invest-execute",
                    json={"buys": [{"symbol": "ABC", "qty": 10, "ltp": 800.0}],
                          "token": "used"})
    assert r.get_json()["success"] is False


def test_invest_execute_empty_buys_rejected(app_client):
    MR, client = app_client
    r = client.post(f"{PREFIX}/invest-execute", json={"buys": [], "token": "x"})
    assert r.get_json()["success"] is False
