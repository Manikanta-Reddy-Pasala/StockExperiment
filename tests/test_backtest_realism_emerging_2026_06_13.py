"""Backtest realism convention (2026-06-13) — emerging_momentum.

Guards the convention applied to tools/models/emerging_momentum/backtest.py:
  1. Real Fyers CNC charges per fill (same calculator as the live ledger).
  2. Fills at the NEXT bar's open (decision on bar d close -> fill d+1 open);
     window-end fills at d's close; missing open falls back to that bar close.
  3. Buy sizing leaves room for charges (no negative cash).

No DB/Fyers needed: the helpers under test are pure functions on in-memory
pandas panels (run() itself needs postgres and is NOT exercised here).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.broker_charges import compute_charges
from tools.models.emerging_momentum import backtest as B


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
