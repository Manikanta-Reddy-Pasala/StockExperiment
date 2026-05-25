"""Rebalance-cycle gate tests for the monthly rotation models.

These lock the contract that was violated by the "VEDL on the 25th" bug:
a model must only emit an order-eligible ENTRY on its scheduled rebalance
day (1st-7th weekday) or its mid-month check day (first weekday on/after
the 15th). Any other calendar day must gate to "no trade".

Pure datetime logic — no DB, no network. Run:
    python3 -m pytest tests/test_rebalance_cycle.py -q
"""
from datetime import datetime

import pytest

from tools.models.momentum_n100_top5_max1.live_signal import (
    is_rebalance_day,
    is_mid_month_check_day,
)
from tools.models.momentum_pseudo_n100_adv.live_signal import (
    is_rebalance_day as pseudo_is_rebalance_day,
)


# ---------------------------------------------------------------------------
# is_rebalance_day — True only on the 1st-7th of the month AND a weekday.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, expected", [
    (datetime(2026, 5, 1),  True),   # Fri, day 1
    (datetime(2026, 5, 4),  True),   # Mon, day 4
    (datetime(2026, 6, 1),  True),   # Mon, day 1
    (datetime(2026, 5, 2),  False),  # Sat, day 2 — weekend
    (datetime(2026, 5, 3),  False),  # Sun, day 3 — weekend
    (datetime(2026, 5, 8),  False),  # Fri, day 8 — past the window
    (datetime(2026, 5, 15), False),  # Fri, mid-month, not rebalance
    (datetime(2026, 5, 25), False),  # Mon, day 25 — the bug day
    (datetime(2026, 5, 29), False),  # Fri, day 29
])
def test_is_rebalance_day(d, expected):
    assert is_rebalance_day(d) is expected
    # Pseudo-N100 shares the identical monthly gate.
    assert pseudo_is_rebalance_day(d) is expected


def test_rebalance_skipped_if_already_rotated_this_month():
    # Already rotated earlier this month -> no second rebalance, even on a
    # valid 1st-7th weekday.
    day3 = datetime(2026, 5, 4)            # Mon, day 4 — would normally be True
    last = datetime(2026, 5, 1)            # rotated on the 1st, same month
    assert is_rebalance_day(day3, last_rotation=last) is False


def test_rebalance_allowed_if_last_rotation_was_previous_month():
    day3 = datetime(2026, 6, 1)            # Mon, day 1 of June
    last = datetime(2026, 5, 4)            # last rotation in May
    assert is_rebalance_day(day3, last_rotation=last) is True


# ---------------------------------------------------------------------------
# is_mid_month_check_day — True only on the first weekday on/after the 15th.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, expected", [
    (datetime(2026, 5, 15), True),   # Fri 15th — the anchor itself is a weekday
    (datetime(2026, 5, 18), False),  # Mon — past the anchor, not the trigger
    (datetime(2026, 5, 14), False),  # before day 15
    (datetime(2026, 5, 25), False),  # day 25 — the bug day
    (datetime(2026, 8, 15), False),  # Sat — anchor rolls forward
    (datetime(2026, 8, 16), False),  # Sun
    (datetime(2026, 8, 17), True),   # Mon — first weekday on/after the 15th
    (datetime(2026, 8, 18), False),  # Tue — past the trigger
])
def test_is_mid_month_check_day(d, expected):
    assert is_mid_month_check_day(d) is expected


# ---------------------------------------------------------------------------
# Regression: the exact day the off-cycle VEDL order fired. On 2026-05-25
# (Monday, day 25) BOTH gates must be closed -> no order-eligible signal.
# ---------------------------------------------------------------------------

def test_25th_is_a_no_trade_day_for_monthly_models():
    bug_day = datetime(2026, 5, 25)
    assert is_rebalance_day(bug_day) is False
    assert pseudo_is_rebalance_day(bug_day) is False
    assert is_mid_month_check_day(bug_day) is False


def test_every_non_window_day_in_a_month_is_no_trade():
    """Across a full month, a trade is permissible ONLY on a 1st-7th weekday
    or the mid-month anchor weekday. Everything else must gate closed."""
    for day in range(1, 29):
        d = datetime(2026, 5, day)
        permissible = is_rebalance_day(d) or is_mid_month_check_day(d)
        in_rebalance_window = (day <= 7 and d.weekday() < 5)
        in_midmonth_window = is_mid_month_check_day(d)  # already validated above
        assert permissible == (in_rebalance_window or in_midmonth_window)
