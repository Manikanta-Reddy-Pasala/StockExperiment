"""SHARED core for momentum_n100_top5_max1 — single source of truth for the
params + rebalance calendar, imported by BOTH backtest.py and live_signal.py so
they can never drift.

Strategy: real NSE Nifty 100, rank by LOOKBACK-trading-day return, hold rank-1
(retain band RETAIN), monthly (1st trading day) + mid-month lead check. Universe
is the live config: retain_top_n=1 + mid-month ON.

Universe SOURCE differs by context (necessarily): backtest uses PIT
eligible_at("n100", d) over its price panel; live uses today's official
nifty100.csv. The RANKING (LOOKBACK return), retain band, mid-month timing and
lead gate are shared from here.
"""
from __future__ import annotations

# Rebalance calendar from the single shared source (same rule emerging uses).
from tools.shared.rebalance_calendar import (
    build_calendar, is_mid_month_check_day, MID_MONTH_FROM_DAY)

LOOKBACK = 15        # momentum window (TRADING days). 6-yr sweep winner (15>30).
RETAIN = 3           # exit band — hold while in top-3 (the LIVE default: --retain-top-n=3).
MIDMONTH_LEAD = 5.0  # mid-month rotates only if new rank-1 leads held by >= 5pp.
INDEX_NAME = "n100"
