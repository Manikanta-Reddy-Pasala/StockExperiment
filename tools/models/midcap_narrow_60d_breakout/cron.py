"""Cron registration for midcap_narrow_60d_breakout.

Two register functions:
  register_data_jobs(schedule)    -- called by data_scheduler.py
  register_trading_jobs(schedule) -- called by scheduler.py (not yet wired)
"""
from __future__ import annotations

import logging
from datetime import datetime

from tools.models.midcap_narrow_60d_breakout.data_pull import (
    pull_daily_ohlcv, refresh_universe,
)

log = logging.getLogger(__name__)


def _monthly_universe():
    """Run universe refresh only on 1st of month."""
    if datetime.now().day == 1:
        refresh_universe()


def register_data_jobs(schedule):
    """Daily OHLCV + monthly universe refresh."""
    # Equity OHLCV — daily after market close (post momentum_n100 pull at 20:30)
    schedule.every().day.at("20:45").do(pull_daily_ohlcv)
    # Universe refresh — first of every month
    schedule.every().day.at("06:35").do(_monthly_universe)


def register_trading_jobs(schedule):
    """Not wired — backtest-only winner. Live executor TODO."""
    pass
