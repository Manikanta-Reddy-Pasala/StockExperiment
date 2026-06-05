"""SHARED core for momentum_n100_top5_max1 — single source of truth for the
params + rebalance calendar, imported by BOTH backtest.py and live_signal.py so
they can never drift.

Strategy: real NSE Nifty 100, rank by LOOKBACK-trading-day return, hold while in
top-RETAIN, monthly (1st trading day) + mid-month lead check. Live config:
LOOKBACK=15, retain_top_n=3, MIDMONTH_LEAD=5pp, mid-month ON.

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
# From-entry FIXED-% hard stop (backtest-validated 2026-06-02 via
# tools/analysis/n100_improve_sweep.py). -12% won across windows on these
# large-caps — full 2021-26 CAGR 56.2->59.9 / DD 56.8->46.4; crash 2022-23
# 68.7->95.7 / DD 42.9->27.8; recent (bull) neutral. ATR was DD-only and a
# price-floor was threshold-fragile, so a fixed % fits n100's uniform vol.
# SHARED with backtest + live --stop-check via tools.shared.stops. Level =
# entry*(1-STOP_PCT), checked daily on the LOW. Set STOP_PCT=0 to disable.
STOP_PCT = 0.12

# Partial profit-take: book HALF the position once price closes >= entry*(1+PCT),
# the rest rides under the from-entry stop. Default OFF (0.0) for n100 — large-cap
# Nifty-100 names rarely spike enough to trigger it; validated as a both-axes win
# only on the high-vol emerging universe (2026-06-06). Set >0 to enable.
PROFIT_TAKE_PCT = 0.0
