"""Backtest realism convention + PIT smallcap fix (2026-06-13) — momentum_pseudo_n100_adv.

Guards the convention applied to tools/models/momentum_pseudo_n100_adv/backtest.py:
  1. Real Fyers CNC charges per fill (same calculator as the live ledger).
  2. Fills at the NEXT bar's open (decision on bar d close -> fill d+1 open);
     window-end fills at d's close; missing open falls back to that bar close.
  3. Buy sizing leaves room for charges (no negative cash).
  4. PIT smallcap exclusion: each yearly universe anchor subtracts the
     period-correct smallcap250_YYYYMMDD.csv snapshot (latest <= anchor,
     earliest as pre-history fallback) — NOT today's nifty_smallcap250.csv.

No DB/Fyers needed: the helpers under test are pure functions on in-memory
pandas panels / repo CSV files (run() itself needs postgres and is NOT
exercised here).
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.broker_charges import compute_charges
from tools.models.momentum_pseudo_n100_adv import backtest as B


# ---------------- convention flags ----------------

def test_convention_constants_documented():
    assert B.FILL_AT_NEXT_OPEN is True
    assert B.CHARGES == "fyers_cnc"


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


# ---------------- PIT smallcap snapshots (non-PIT bug fix) ----------------

def test_smallcap_snapshots_loaded_from_index_history():
    # Repo carries 20210331..20260331 snapshots; loader must find them.
    assert B._SML_SNAPS, "no smallcap250_*.csv snapshots loaded"
    assert date(2021, 3, 31) in B._SML_SNAPS
    assert date(2026, 3, 31) in B._SML_SNAPS
    for syms in B._SML_SNAPS.values():
        assert len(syms) >= 200             # ~250 names per snapshot


def test_smallcap_at_picks_latest_snapshot_not_after_anchor():
    # May-15 yearly universe anchor -> the Mar-31 snapshot of the SAME year.
    assert B.smallcap_at(date(2021, 5, 15)) == B._SML_SNAPS[date(2021, 3, 31)]
    assert B.smallcap_at(date(2026, 5, 15)) == B._SML_SNAPS[date(2026, 3, 31)]
    # Between snapshots: 2023-10-01 -> the 2023-09-30 snapshot.
    assert B.smallcap_at(date(2023, 10, 1)) == B._SML_SNAPS[date(2023, 9, 30)]
    # NOT the same-year later snapshot (no lookahead).
    assert B.smallcap_at(date(2023, 5, 15)) == B._SML_SNAPS[date(2023, 3, 31)]


def test_smallcap_at_pre_history_falls_back_to_earliest_snapshot():
    # The warmup anchor (start.year-1 = 2020-05-15) predates every snapshot:
    # last-known-state rule falls back to the FIRST snapshot, mirroring
    # tools/shared/index_membership semantics.
    earliest = B._SML_SNAPS[min(B._SML_SNAPS)]
    assert B.smallcap_at(date(2020, 5, 15)) == earliest


def test_smallcap_membership_is_point_in_time_not_static():
    # The old bug: today's list applied to 2021 (only ~52% overlap). The PIT
    # snapshots must genuinely differ across years.
    s2021 = B.smallcap_at(date(2021, 5, 15))
    s2026 = B.smallcap_at(date(2026, 5, 15))
    assert s2021 != s2026
    overlap = len(s2021 & s2026) / max(1, len(s2021))
    assert overlap < 0.8                    # substantial index churn 2021->2026
