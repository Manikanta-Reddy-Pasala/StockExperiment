"""SHARED core for n20_daily_large_only (functionally n40 — UNIV_SIZE=40) —
params imported by BOTH backtest.py and live_signal.py so they can't drift.

Strategy: WEEKLY rotation (2026-05-30: was daily — weekly cut the whipsaw).
Universe = top-UNIV_SIZE by 20d ADV from N500, intersected with PIT Nifty 100,
uptrend (close > SMA_LONG); rank by LOOKBACK return; hold rank-1 (RETAIN band).
Rebalances on the first trading day of each ISO week (shared build_weekly_calendar
/ is_week_rebalance_day). (Model dir keeps the legacy 'n20' name; config is n40.)
"""
from __future__ import annotations

UNIV_SIZE = 40       # top-N by 20d ADV (n40; was 20 — n40 beats n20 on CAGR + DD)
LOOKBACK = 30        # momentum window (TRADING days)
ADV_WIN = 20         # ADV averaging window
SMA_LONG = 200       # uptrend filter: close > 200d SMA
RETAIN = 1           # exit band — top-1 weekly rotation
# CURRENT headline (2026-06-13 realism regen w/ STOP=0.10 — net of real Fyers
# CNC charges, next-open fills): full 2021-03→2026-05 +32.4% CAGR / 38.9% DD /
# Calmar 0.83. From-entry FIXED-% hard stop, checked daily on the LOW
# (entry*(1-STOP_PCT)). SHARED with backtest + live --stop-check via
# tools.shared.stops. Set STOP_PCT=0 to disable.
# 2026-06-13 RE-TUNE 0.12 -> 0.10: the old 0.12 was tuned PRE-realism (close-MTM,
# zero charges, "−12% won both axes"). Under the realism convention 0.12 is a
# local dip; a fresh realism stop-sweep (tools/research/n40_cagr.py) shows the
# in-sample optimum is ~8-10% — 0.10 lifts full-window CAGR 28.4->32.4 AND cuts
# DD 43.9->38.9 (Calmar 0.65->0.83), a plateau (8/10/15 all beat 12). Anchored
# WF: the lift is concentrated in the 2021-22 bear so forward (2023-26 OOS) CAGR
# is ~neutral — 0.10 is a stale-param fix + DD reducer, not a forward-CAGR boost.
STOP_PCT = 0.10

# Partial profit-take: book HALF once price closes >= entry*(1+PCT), rest rides
# under the from-entry stop. Default OFF (0.0) — n40 is large-cap (top-40 ADV ∩
# N100), rarely spikes enough to trigger; validated as a win only on the high-vol
# emerging universe (2026-06-06). Set >0 to test/enable.
PROFIT_TAKE_PCT = 0.0
