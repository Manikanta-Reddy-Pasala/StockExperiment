"""Backtest realism convention + PIT-before-ADV fix (2026-06-13) — momentum_retest_n500.

Guards two changes:
  A. Realism convention in tools/models/momentum_retest_n500/backtest.py:
     1. Real Fyers CNC charges per fill (same calculator as the live ledger).
     2. Fills at the NEXT bar's open (decision on bar d close -> fill d+1 open);
        window-end fills at d's close; missing open falls back to that bar close.
     3. Buy sizing leaves room for charges (no negative cash).
  B. PIT-before-ADV universe fix in strategy.rank_targets: the eligible_at PIT
     filter is applied BEFORE the top-TOPN ADV cut (via the new optional
     `eligible` arg), so historically-ineligible names can no longer displace
     eligible ones at the ADV margin. Live passes no `eligible` -> unchanged.

No DB/Fyers needed: the helpers under test are pure functions on in-memory
pandas panels (run() itself needs postgres and is NOT exercised here).
"""
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.broker_charges import compute_charges
from tools.models.momentum_retest_n500 import backtest as B
from tools.models.momentum_retest_n500 import strategy as S


# ---------------- convention flags ----------------

def test_convention_constants_documented():
    assert B.FILL_AT_NEXT_OPEN is True
    assert B.CHARGES == "fyers_cnc"


def test_flat_cost_constant_removed():
    # The old flat 0.0015/side COST is gone — real per-fill charges only.
    assert not hasattr(B, "COST")


def test_run_wires_pit_eligible_into_rank_targets():
    src = inspect.getsource(B.run)
    assert "eligible=eligible_at(" in src        # PIT filter BEFORE the ADV cut
    # the old AFTER-the-cut post-filter on rk must be gone
    assert "rk = [s for s in rk" not in src


# ---------------- charges: same calculator as live ledger ----------------

def test_fill_charges_matches_live_cnc_calculator():
    for side in ("BUY", "SELL"):
        expected = compute_charges(side, 100, 200.0, "CNC")["total"]
        assert B.fill_charges(side, 100, 200.0) == pytest.approx(expected)
        assert B.fill_charges(side, 100, 200.0) > 0


def test_fill_charges_is_cnc_not_intraday():
    # CNC pays 0.1% STT both sides + DP on sell — far above intraday.
    cnc_sell = B.fill_charges("SELL", 100, 200.0)
    mis_sell = compute_charges("SELL", 100, 200.0, "INTRADAY")["total"]
    assert cnc_sell > mis_sell


# ---------------- next_open_fill timing ----------------

def _panels():
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"])
    sym = "NSE:ABC-EQ"
    cl = pd.DataFrame({sym: [100.0, 102.0, 104.0]}, index=dates)
    op = pd.DataFrame({sym: [99.0, 101.0, 103.0]}, index=dates)
    return op, cl, dates, sym


def test_decision_on_d_close_fills_at_d_plus_1_open():
    op, cl, dates, sym = _panels()
    px, fdate = B.next_open_fill(op, cl, dates, sym, 0)
    assert px == 101.0                      # NOT 100.0 (bar 0 close)
    assert fdate == "2026-01-06"


def test_last_bar_of_window_fills_at_own_close():
    op, cl, dates, sym = _panels()
    px, fdate = B.next_open_fill(op, cl, dates, sym, 2)
    assert px == 104.0                      # window-end bookkeeping: d's close
    assert fdate == "2026-01-07"


def test_missing_next_open_falls_back_to_next_close():
    op, cl, dates, sym = _panels()
    op.loc[dates[1], sym] = float("nan")
    px, fdate = B.next_open_fill(op, cl, dates, sym, 0)
    assert px == 102.0                      # bar d+1 close, NOT bar d close
    assert fdate == "2026-01-06"


def test_symbol_missing_from_open_panel_falls_back_to_next_close():
    op, cl, dates, sym = _panels()
    px, fdate = B.next_open_fill(op.drop(columns=[sym]), cl, dates, sym, 0)
    assert px == 102.0
    assert fdate == "2026-01-06"


# ---------------- buy sizing: charges come out of cash ----------------

def test_size_buy_qty_reserves_cash_for_charges():
    cap, px = 10_000.0, 100.0
    q = B.size_buy_qty(cap, px)
    assert q == 99                          # naive int(cap/px)=100 would overdraw
    assert q * px + B.fill_charges("BUY", q, px) <= cap


def test_size_buy_qty_never_goes_negative_cash():
    for cap, px in [(1e6, 333.33), (54_000.0, 187.4), (101.0, 100.0), (5.0, 100.0)]:
        q = B.size_buy_qty(cap, px)
        assert q >= 0
        if q:
            assert q * px + B.fill_charges("BUY", q, px) <= cap


def test_size_buy_qty_zero_for_bad_price():
    assert B.size_buy_qty(10_000.0, 0.0) == 0
    assert B.size_buy_qty(10_000.0, -5.0) == 0


# ---------------- PIT-before-ADV in rank_targets ----------------

def _rank_panels():
    """121 symbols (TOPN=120 + 1) that ALL pass every momentum filter, with a
    strictly descending ADV ladder: BAD has the HIGHEST ADV but is NOT in the
    PIT index; EDGE has the LOWEST ADV and IS in the index. Pre-fix, BAD's
    presence in the head(TOPN) cut displaced EDGE from the universe."""
    assert S.TOPN == 120
    n_days = 210                            # > SMA_LONG so sma200 is non-NaN
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    fillers = [f"NSE:S{i:03d}-EQ" for i in range(S.TOPN - 1)]   # 119 names
    syms = ["NSE:BAD-EQ"] + fillers + ["NSE:EDGE-EQ"]           # 121 total
    base = pd.Series([100.0 * (1.01 ** i) for i in range(n_days)], index=dates)
    cl = pd.DataFrame({s: base for s in syms})                  # rising, <3000
    adv = pd.DataFrame({s: [2000.0 - j] * n_days for j, s in enumerate(syms)},
                       index=dates)                             # BAD top, EDGE last
    sma200 = cl.rolling(S.SMA_LONG).mean()
    return cl, adv, sma200, syms


def test_eligible_filter_applied_before_topn_adv_cut():
    cl, adv, sma200, syms = _rank_panels()
    di = len(cl) - 1
    eligible = {s.replace("NSE:", "").replace("-EQ", "") for s in syms} - {"BAD"}
    rk = S.rank_targets(cl, adv, sma200, set(), di, eligible=eligible)
    assert "NSE:BAD-EQ" not in rk           # ineligible name excluded
    assert "NSE:EDGE-EQ" in rk              # marginal eligible name now makes
    #                                         the top-120 ADV cut (the old
    #                                         post-filter dropped it forever)


def test_no_eligible_arg_keeps_old_behavior_live_path():
    cl, adv, sma200, syms = _rank_panels()
    di = len(cl) - 1
    # Live calls rank_targets with 5 positional args — must work unchanged:
    rk = S.rank_targets(cl, adv, sma200, set(), di)
    assert "NSE:BAD-EQ" in rk               # no PIT info -> ranked as before
    assert "NSE:EDGE-EQ" not in rk          # displaced at the ADV margin
    assert len(rk) == S.TOPN


def test_eligible_none_identical_to_omitted():
    cl, adv, sma200, syms = _rank_panels()
    di = len(cl) - 1
    assert S.rank_targets(cl, adv, sma200, set(), di) == \
        S.rank_targets(cl, adv, sma200, set(), di, eligible=None)
