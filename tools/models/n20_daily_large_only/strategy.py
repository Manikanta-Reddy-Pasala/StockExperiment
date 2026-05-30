"""SHARED core for n20_daily_large_only (functionally n40 — UNIV_SIZE=40) —
params imported by BOTH backtest.py and live_signal.py so they can't drift.

Strategy: DAILY rotation. Universe = top-UNIV_SIZE by 20d ADV from N500,
intersected with PIT Nifty 100, uptrend (close > SMA_LONG); rank by LOOKBACK
return; hold rank-1 (RETAIN band). No monthly/mid calendar — rebalances every
trading day. (Model dir keeps the legacy 'n20' name; the live config is n40.)
"""
from __future__ import annotations

UNIV_SIZE = 40       # top-N by 20d ADV (n40; was 20 — n40 beats n20 on CAGR + DD)
LOOKBACK = 30        # momentum window (TRADING days)
ADV_WIN = 20         # ADV averaging window
SMA_LONG = 200       # uptrend filter: close > 200d SMA
RETAIN = 1           # exit band — top-1 daily rotation
