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
# CURRENT headline (2026-06-13 realism regen — net of real Fyers CNC charges,
# next-open fills): full 2021-03→2026-05 +28.4% CAGR / 43.9% DD / Calmar 0.65;
# 3-yr 2023-05→2026-05 +48.6% / 30.9% / 1.58. Sweep numbers below are
# pre-realism (close fills, zero charges).
# From-entry FIXED-% hard stop (backtest-validated 2026-06-04 via n40 daily-MTM
# stop sweep). n40 was the only momentum model without a stop. -12% won on BOTH
# axes: full 2021-26 CAGR 41.1->48.1 / DD 41.4->37.1 / Calmar 0.99->1.30
# (-10% ~equal: 46.4/35.6/1.30; smooth plateau, not a spike). SHARED with
# backtest + live --stop-check via tools.shared.stops. Level = entry*(1-STOP_PCT),
# checked daily on the LOW. Set STOP_PCT=0 to disable.
STOP_PCT = 0.12

# Partial profit-take: book HALF once price closes >= entry*(1+PCT), rest rides
# under the from-entry stop. Default OFF (0.0) — n40 is large-cap (top-40 ADV ∩
# N100), rarely spikes enough to trigger; validated as a win only on the high-vol
# emerging universe (2026-06-06). Set >0 to test/enable.
PROFIT_TAKE_PCT = 0.0
