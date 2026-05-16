"""Cron registration for FinNifty IC — DATA ONLY (execution unwired).

Only register_data_jobs is implemented. No trading-side jobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.data_pull import (  # noqa: E402
    fetch_index_spots, fetch_option_bhav,
)


def register_data_jobs(schedule):
    """Daily data pulls. Called by data_scheduler."""
    # Index spots: Fyers EOD, run after market close + buffer
    schedule.every().day.at("18:00").do(fetch_index_spots)
    # Option bhavcopy: NSE archives publish ~17:30 IST, fetch 18:30
    schedule.every().day.at("18:30").do(fetch_option_bhav)


def register_trading_jobs(schedule):
    """No-op — model is unwired for execution."""
    pass
