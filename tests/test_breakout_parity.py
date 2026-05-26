"""Parity + unit tests for the shared breakout strategy core.

midcap_narrow_60d_breakout's backtest.py and live_signal.py now both import
tools.shared.breakout_strategy for entry qualification (is_breakout) and the
exit rule (breakout_exit_reason). These tests pin the rule and assert the live
check_exit returns exactly what the core decides — so the two paths can't drift.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.shared.breakout_strategy import is_breakout, breakout_exit_reason

EXIT = dict(target_pct=1.00, stop_pct=0.20, trail_pct=0.20,
            profit_trigger=0.10, max_hold_days=120)


# ---------------- is_breakout ----------------

def test_breakout_qualifies_when_all_three_fire():
    ok, vr = is_breakout(close=110, prior_high=108, sma_long=90,
                         volume=300, vol_avg20=100, vol_mult=2.0)
    assert ok and vr == 3.0


def test_breakout_fails_below_prior_high():
    ok, _ = is_breakout(105, 108, 90, 300, 100, vol_mult=2.0)
    assert not ok


def test_breakout_fails_below_sma():
    ok, _ = is_breakout(110, 108, 120, 300, 100, vol_mult=2.0)
    assert not ok


def test_breakout_fails_low_volume():
    ok, _ = is_breakout(110, 108, 90, 150, 100, vol_mult=2.0)  # vr=1.5 < 2.0
    assert not ok


def test_breakout_zero_volume_baseline():
    ok, vr = is_breakout(110, 108, 90, 300, 0, vol_mult=2.0)
    assert not ok and vr == 0.0


# ---------------- breakout_exit_reason ----------------

def test_exit_target():
    assert breakout_exit_reason(100, 200, 200, 5, **EXIT) == "TARGET"


def test_exit_stop():
    assert breakout_exit_reason(100, 79, 100, 5, **EXIT) == "STOP"


def test_exit_trail_peak40_exits_at_plus12_not_plus30():
    # peak 140 (entry 100). 20% below peak = 112 = +12% -> TRAIL fires.
    assert breakout_exit_reason(100, 112, 140, 5, **EXIT) == "TRAIL"
    # at +30% (close 130) drop-from-peak is only 7.1% < 20% -> no trail.
    assert breakout_exit_reason(100, 130, 140, 5, **EXIT) is None


def test_exit_trail_needs_profit_arm():
    # down 20% from a peak that's only +5% from entry -> below +10% arm, no trail.
    # peak 105, close 84 -> ret_e=-16% (>-20% so no stop), ret_pk=20%, but ret_e<10% -> None
    assert breakout_exit_reason(100, 84, 105, 5, **EXIT) is None


def test_exit_max_hold():
    assert breakout_exit_reason(100, 105, 106, 120, **EXIT) == "MAX_HOLD"


def test_exit_hold_none():
    assert breakout_exit_reason(100, 105, 106, 10, **EXIT) is None


# ---------------- live check_exit parity with the core ----------------

def test_midcap_live_check_exit_matches_core():
    import importlib
    mod = importlib.import_module(
        "tools.models.midcap_narrow_60d_breakout.live_signal")

    entry = (datetime.now().date() - timedelta(days=10))
    # closes since entry: peak 140, last 112 -> TRAIL per the core.
    rows = [
        {"date": pd.Timestamp(entry), "close": 100.0},
        {"date": pd.Timestamp(entry + timedelta(days=3)), "close": 140.0},
        {"date": pd.Timestamp(entry + timedelta(days=6)), "close": 112.0},
    ]
    df_sym = pd.DataFrame(rows)
    pos = {"open_entry_px": 100.0, "open_entry_date": entry.strftime("%Y-%m-%d")}

    out = mod.check_exit(pos, df_sym)
    core = breakout_exit_reason(100.0, 112.0, 140.0, 6, **EXIT)
    assert core == "TRAIL"
    assert out is not None and out["reason"] == core
