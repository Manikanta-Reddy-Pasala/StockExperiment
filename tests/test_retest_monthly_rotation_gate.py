"""momentum_retest_n500 monthly rotation gate (2026-06 fix).

Bug: the old is_first_trading_day_of_month(dates, today) returned True only if
TODAY's bar already existed in the OHLCV panel — never true at the 09:26 cron
(data is pulled nightly ~20:42, so the panel's last bar is the PREVIOUS
session). Monthly RANK_DROP exits and the WATCH_FILE refresh therefore NEVER
fired from the scheduler, only on manual --force runs.

Fix semantics (backtest parity — backtest.py ranks the month-first-session
close at its rebalance): monthly_rotation_due(dates, di, watch_state) is True
exactly when the panel's LAST bar IS the month's first NSE trading session
(holiday/weekend-aware via tools.shared.nse_calendar), i.e. on the first cron
morning AFTER that session closed — ranking that session's close. A
"ranked_session" stamp in the watch state stops a re-fire when a failed
nightly pull leaves the panel parked on the first session.

No DB / Fyers needed: the gate is pure (pandas dates + nse_calendar fallback
holiday list).
"""
import inspect
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.models.momentum_retest_n500 import live_signal as ls  # noqa: E402


def _panel(*iso_dates):
    """A panel date index ending on the given sessions; returns (dates, di)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in iso_dates])
    return idx, len(idx) - 1


# ---------------------------------------------------------------------------
# Plain month start: June 2026 — Mon 2026-06-01 is the first NSE session.
# ---------------------------------------------------------------------------

def test_fires_morning_after_first_session():
    # Cron morning of Tue Jun 2: nightly pull has landed Jun 1's bar -> fire,
    # ranking the Jun 1 close (the bar backtest.py ranks at its rebalance).
    dates, di = _panel("2026-05-28", "2026-05-29", "2026-06-01")
    assert ls.monthly_rotation_due(dates, di, {}) is True


def test_no_fire_on_first_session_morning():
    # Cron morning of Mon Jun 1 itself: panel still ends on May's last session
    # (Fri May 29) — the first session hasn't closed yet. Must NOT fire.
    dates, di = _panel("2026-05-27", "2026-05-28", "2026-05-29")
    assert ls.monthly_rotation_due(dates, di, {}) is False


def test_no_fire_later_in_month():
    # Cron morning of Jun 3: panel ends Jun 2 (second session) — no fire.
    dates, di = _panel("2026-05-29", "2026-06-01", "2026-06-02")
    assert ls.monthly_rotation_due(dates, di, {}) is False


# ---------------------------------------------------------------------------
# Month starting on a weekend: Feb 2026 — Feb 1 is a Sunday, first session is
# Mon Feb 2.
# ---------------------------------------------------------------------------

def test_weekend_month_start_shifts_gate():
    # Morning of Mon Feb 2 (first session): panel ends Fri Jan 30 -> no fire.
    dates, di = _panel("2026-01-29", "2026-01-30")
    assert ls.monthly_rotation_due(dates, di, {}) is False
    # Morning of Tue Feb 3: panel ends Feb 2 = month's first session -> fire.
    dates, di = _panel("2026-01-30", "2026-02-02")
    assert ls.monthly_rotation_due(dates, di, {}) is True


# ---------------------------------------------------------------------------
# Holiday-shifted month start: May 2026 — Fri May 1 is Maharashtra Day, so the
# first session is Mon May 4.
# ---------------------------------------------------------------------------

def test_holiday_month_start_shifts_gate():
    # Panel ending Apr 30 (May 1 holiday not yet traded): no fire.
    dates, di = _panel("2026-04-29", "2026-04-30")
    assert ls.monthly_rotation_due(dates, di, {}) is False
    # Panel ending Mon May 4 (the actual first session) -> fire (Tue morning).
    dates, di = _panel("2026-04-30", "2026-05-04")
    assert ls.monthly_rotation_due(dates, di, {}) is True
    # Panel ending May 5 (second session): no fire.
    dates, di = _panel("2026-05-04", "2026-05-05")
    assert ls.monthly_rotation_due(dates, di, {}) is False


# ---------------------------------------------------------------------------
# First session on a Friday: Aug 2025 — Fri Aug 1 is the first session; the
# next cron is Mon Aug 4 (weekend crons skip). Gate keys off the PANEL bar,
# so the Monday-morning run still fires.
# ---------------------------------------------------------------------------

def test_friday_first_session_fires_on_monday_cron():
    dates, di = _panel("2025-07-31", "2025-08-01")
    assert ls.monthly_rotation_due(dates, di, {}) is True


# ---------------------------------------------------------------------------
# Re-entry guard (no double-fire) + stale-data behavior.
# ---------------------------------------------------------------------------

def test_ranked_session_stamp_blocks_refire():
    # Nightly pull failed after the Jun-2 rotation: Jun-3 morning the panel
    # STILL ends on Jun 1. The ranked_session stamp must block a second fire.
    dates, di = _panel("2026-05-29", "2026-06-01")
    state = {"month": "2026-06", "watch": [], "ranked_session": "2026-06-01"}
    assert ls.monthly_rotation_due(dates, di, state) is False


def test_stale_panel_fires_on_first_morning_bar_present():
    # If Jun 1's bar only lands a night late (panel ends Jun 1 on the Jun 3
    # cron, no rotation recorded yet), fire on that first available morning.
    dates, di = _panel("2026-05-29", "2026-06-01")
    assert ls.monthly_rotation_due(dates, di, None) is True
    assert ls.monthly_rotation_due(dates, di, {"watch": []}) is True


def test_force_stamp_on_non_first_session_cannot_suppress_cron():
    # A manual --force run mid-month stamps ranked_session with a NON-first
    # session date; the real cron rotation next month must still fire.
    dates, di = _panel("2026-05-29", "2026-06-01")
    state = {"month": "2026-05", "watch": [], "ranked_session": "2026-05-15"}
    assert ls.monthly_rotation_due(dates, di, state) is True


def test_empty_panel_is_safe():
    dates = pd.DatetimeIndex([])
    assert ls.monthly_rotation_due(dates, -1, {}) is False
    assert ls.monthly_rotation_due(None, -1, {}) is False


# ---------------------------------------------------------------------------
# Source-level guards: main() must use the new gate; the old today-bar gate
# must be gone (style of test_retest_n40_eligibility_pit.py).
# ---------------------------------------------------------------------------

def test_main_wired_to_new_gate():
    src = inspect.getsource(ls.main)
    assert "monthly_rotation_due(dates, di" in src, \
        "main() must gate the monthly block on monthly_rotation_due(dates, di, ...)"
    assert "args.force or monthly_rotation_due" in src, \
        "--force must still bypass the monthly gate"
    assert "ranked_session" in src, \
        "monthly block must stamp ranked_session into WATCH_FILE (re-fire dedup)"


def test_old_today_bar_gate_removed():
    src = inspect.getsource(ls)
    assert "fut[0].normalize() == pd.Timestamp(today_ts).normalize()" not in src, \
        "old gate required TODAY's bar in the panel — never true at the 09:26 cron"
    # Gate must be holiday-aware via the shared NSE calendar.
    assert "nse_calendar" in src
