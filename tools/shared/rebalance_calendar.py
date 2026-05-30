"""Shared rebalance-calendar rule — single source of truth for WHEN the
rotation models rebalance, so every model's backtest and live signal agree.

Rule:
  full = first NSE trading day of each month
  mid  = first NSE trading day ON/AFTER the 15th (optional; monthly models that
         run a mid-month lead-check use it, e.g. emerging + n100)

Backtest derives the calendar from its trading-day index (`build_calendar`);
live mirrors the SAME rule on a single date via nse_calendar
(`is_mid_month_check_day`), so the two can never disagree.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

MID_MONTH_FROM_DAY = 15


def build_calendar(dates, start, end, mid_check: bool = True, mid_from: int = MID_MONTH_FROM_DAY):
    """Sorted [(timestamp, 'full'|'mid')] over [start, end] from a trading-day index.

    full = month's first trading day; mid = first trading day on/after `mid_from`
    (only if mid_check). A mid coinciding with a full is dropped.
    """
    s = pd.Series(dates, index=dates)
    firsts = {pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).first().values}
    full = [d for d in dates if start <= d.date() <= end and d in firsts]
    cal = [(d, "full") for d in full]
    if mid_check:
        mids, seen = [], set()
        for d in dates:
            if (start <= d.date() <= end and d.day >= mid_from
                    and (d.year, d.month) not in seen):
                mids.append(d)
                seen.add((d.year, d.month))
        full_set = set(full)
        cal += [(d, "mid") for d in mids if d not in full_set]
    return sorted(cal, key=lambda x: x[0])


def is_mid_month_check_day(today, mid_from: int = MID_MONTH_FROM_DAY) -> bool:
    """Live mirror of the backtest 'mid' rule: True iff `today` is the FIRST NSE
    trading day on/after `mid_from` of its month (holiday-aware)."""
    from tools.shared.nse_calendar import is_trading_day
    d0 = today.date() if hasattr(today, "date") else today
    if d0.day < mid_from or not is_trading_day(d0):
        return False
    probe = d0.replace(day=mid_from)
    while probe < d0:                       # an earlier trading day on/after mid_from?
        if is_trading_day(probe):
            return False
        probe += timedelta(days=1)
    return True
