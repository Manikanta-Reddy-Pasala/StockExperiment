"""EMA 9/21 1H crossover model wrapper (KISS registry entry)."""
from src.services.technical.ema_crossover_strategy import (
    EMACrossoverStrategy,
    StrategyConfig,
)

NAME = "ema_9_21"
DESCRIPTION = "EMA 9/21 crossover on 1H bars (short-term trend follow)."
STRATEGY_CLASS = EMACrossoverStrategy
CONFIG_CLASS = StrategyConfig
DEFAULT_CONFIG = StrategyConfig(ema_fast_period=9, ema_slow_period=21)
BARS_INTERVAL = "1h"
DEFAULT_WINDOW_DAYS = 365
HARNESS = "tools.backtests.run_ema_200_400_backtest"
