"""Gap coverage for the pure sizing core in
src/services/trading/model_invest_service.py.

No I/O — these exercise compute_buys edge cases the existing
tests/test_model_invest_service.py does not cover:
  * max_holdings=None and =0 (treated as single position)
  * Retest where every top-N slot is already held -> empty buys
  * ltp=0 / ltp=None target dropped (qty<1 skipped)
  * broker cash smaller than one share -> empty buys
"""
import pytest

from src.services.trading.model_invest_service import compute_buys, CASH_BUFFER


def test_max_holdings_none_raises_typeerror():
    # Core contract: compute_buys does NOT coerce None itself (None <= 1 is a
    # TypeError in Py3). The route guards this by passing
    # `model_max_holdings(...) or 1`, so None never reaches the core in prod.
    # This pins the boundary so a future "or 1" regression is caught here.
    targets = [{"symbol": "AAA", "ltp": 100.0}]
    with pytest.raises(TypeError):
        compute_buys(50000.0, 50000.0, None, targets, set())


def test_max_holdings_one_is_single_position():
    # The route-coerced value (None -> 1) deploys the whole sleeve into rank-1.
    targets = [{"symbol": "AAA", "ltp": 100.0}, {"symbol": "BBB", "ltp": 100.0}]
    buys = compute_buys(50000.0, 50000.0, 1, targets, set())
    assert len(buys) == 1
    assert buys[0]["symbol"] == "AAA"
    assert buys[0]["qty"] == int((50000.0 * CASH_BUFFER) // 100.0)


def test_max_holdings_zero_treated_as_single_position():
    # max_holdings <= 1 branch also covers 0 (and negatives).
    targets = [{"symbol": "AAA", "ltp": 100.0}, {"symbol": "BBB", "ltp": 100.0}]
    buys = compute_buys(50000.0, 50000.0, 0, targets, set())
    assert len(buys) == 1
    assert buys[0]["symbol"] == "AAA"


def test_retest_all_top4_already_held_gives_empty():
    targets = [{"symbol": s, "ltp": 100.0} for s in ("A", "B", "C", "D")]
    held = {"A", "B", "C", "D"}
    buys = compute_buys(40000.0, 40000.0, 4, targets, held)
    assert buys == []


def test_retest_partial_held_funds_only_empty_slots():
    targets = [{"symbol": s, "ltp": 100.0} for s in ("A", "B", "C", "D")]
    held = {"A", "C"}
    buys = compute_buys(40000.0, 40000.0, 4, targets, held)
    syms = {b["symbol"] for b in buys}
    assert syms == {"B", "D"}
    # per-slot split is across the 2 open slots, not 4
    expected_qty = int((40000.0 * CASH_BUFFER / 2) // 100.0)
    assert all(b["qty"] == expected_qty for b in buys)


def test_ltp_zero_target_dropped():
    targets = [{"symbol": "ZERO", "ltp": 0.0}, {"symbol": "OK", "ltp": 100.0}]
    # single position picks rank-1 (ZERO) -> qty 0 -> dropped -> empty
    buys = compute_buys(50000.0, 50000.0, 1, targets, set())
    assert buys == []


def test_ltp_none_target_dropped_multislot():
    targets = [{"symbol": "NONE", "ltp": None}, {"symbol": "OK", "ltp": 100.0}]
    buys = compute_buys(50000.0, 50000.0, 2, targets, set())
    syms = {b["symbol"] for b in buys}
    assert "NONE" not in syms
    assert "OK" in syms


def test_broker_cash_smaller_than_one_share_gives_empty():
    targets = [{"symbol": "PRICEY", "ltp": 1000.0}]
    # plenty of idle, but broker only has 500 -> can't afford 1 share
    buys = compute_buys(50000.0, 500.0, 1, targets, set())
    assert buys == []


def test_broker_cash_is_the_binding_constraint():
    targets = [{"symbol": "X", "ltp": 100.0}]
    # idle huge, broker small -> deployable = broker (the min)
    buys = compute_buys(1_000_000.0, 5000.0, 1, targets, set())
    assert buys[0]["qty"] == int((5000.0 * CASH_BUFFER) // 100.0)


def test_retest_full_book_buys_nothing():
    # holds 4 of 4 (even if ranking rotated to new names) -> no new buys
    held = {"A", "B", "C", "D"}
    targets = [{"symbol": s, "ltp": 100.0} for s in ("E", "F", "A", "B")]
    assert compute_buys(100000, 100000, 4, targets, held) == []


def test_retest_caps_to_free_slots_on_rotation():
    # holds 3 (X,Y,Z); ranking top names are new -> only 1 FREE slot may fill,
    # never pushing the book to 5.
    held = {"X", "Y", "Z"}
    targets = [{"symbol": s, "ltp": 100.0} for s in ("M", "N", "O", "P")]
    buys = compute_buys(100000, 100000, 4, targets, held)
    assert len(buys) == 1            # exactly one free slot
    assert buys[0]["symbol"] == "M"  # highest-ranked unheld
