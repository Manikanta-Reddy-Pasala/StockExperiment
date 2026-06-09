from datetime import datetime

from src.services.trading.model_invest_service import (
    compute_buys, is_market_open, make_token,
)


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
                        targets=[{"symbol": s, "ltp": 100.0} for s in ("A", "B", "C", "D")],
                        open_symbols={"A", "B"})
    syms = {b["symbol"] for b in buys}
    assert syms == {"C", "D"}
    assert sum(b["amount"] for b in buys) <= 100000 * 0.995 + 0.01


def test_zero_when_no_deployable():
    assert compute_buys(0, 50000, 1, [{"symbol": "X", "ltp": 10.0}], set()) == []
    assert compute_buys(50000, 0, 1, [{"symbol": "X", "ltp": 10.0}], set()) == []


def test_drops_zero_qty_targets():
    # deployable too small for a 5000-rupee share
    assert compute_buys(1000, 1000, 1, [{"symbol": "PRICEY", "ltp": 5000.0}], set()) == []


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
