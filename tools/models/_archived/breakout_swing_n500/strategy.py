"""breakout_swing_n500 — constants + thin wrappers (single source of truth).

Short-hold (1-5 trading-day) momentum-burst breakout swing over the liquid
N500. Decision logic itself lives in the SHARED pure core
``tools.shared.breakout_strategy`` (is_breakout + breakout_exit_reason) so the
backtest and live emit cannot drift — exactly the pattern the other breakout
model (midcap_narrow_60d_breakout) and the rotation models already follow.

The CONFIG values below are the ship-candidate defaults. They are NOT final
until the backtest sweep (backtest.py --sweep) picks the winning grid point and
this file is updated to match. Until then treat them as placeholders that the
sweep overrides via the BacktestConfig.
"""
from __future__ import annotations

# --- Entry / universe selection ---
HH_WIN   = 40      # fresh-breakout lookback: close must clear the prior 40d high
SMA_LONG = 200     # Stage-2 uptrend gate (close > 200d SMA)
VOL_MULT = 2.0     # volume surge: breakout-day vol >= VOL_MULT * 20d avg vol
ADV_WIN  = 20      # liquidity-rank window (20d average ₹ value traded)
UNIV_SIZE = 150    # top-150 by 20d ADV from PIT N500 (breakout breadth)
MIN_PRICE = 100.0  # ORB lesson: sub-₹100 pennies whipsaw into fake breakouts
MAX_PRICE = 3000.0 # share-granularity cap, consistent with other models

# --- Position sizing ---
SLOTS = 3          # concurrent positions (swept {1,3,5}; equal-weight)

# --- Exit (short-hold) ---
TARGET_PCT  = 0.10  # take profit at +10% from entry
STOP_PCT    = 0.06  # catastrophe stop at -6% from entry
TRAIL_PCT   = 0.05  # exit on -5% off the peak close, once in profit
PROFIT_TRIG = 0.05  # trail arms only after >= +5% gain
MAX_HOLD    = 5     # the SWING constraint: force-exit after 5 TRADING days
