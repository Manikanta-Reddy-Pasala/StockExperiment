"""Swing Pullback Breakout (daily) model wrapper (KISS registry entry)."""
from src.services.technical.ema_pullback_breakout import (
    EMAPullbackBreakoutStrategy,
    PullbackConfig,
)

NAME = "swing_pullback"
DESCRIPTION = "Swing pullback-to-EMA breakout on daily bars (multi-day swings)."
STRATEGY_CLASS = EMAPullbackBreakoutStrategy
CONFIG_CLASS = PullbackConfig
DEFAULT_CONFIG = PullbackConfig()
BARS_INTERVAL = "daily"
DEFAULT_WINDOW_DAYS = 365
HARNESS = "tools.backtests.run_swing_pullback_backtest"
