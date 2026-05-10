"""EMA 200/400 1H crossover model wrapper (KISS registry entry)."""
from src.services.technical.ema_crossover_strategy import (
    EMACrossoverStrategy,
    StrategyConfig,
)

NAME = "ema_200_400"
DESCRIPTION = "EMA 200/400 crossover on 1H bars (long-term trend follow)."
STRATEGY_CLASS = EMACrossoverStrategy
CONFIG_CLASS = StrategyConfig
DEFAULT_CONFIG = StrategyConfig(ema_fast_period=200, ema_slow_period=400)
BARS_INTERVAL = "1h"
DEFAULT_WINDOW_DAYS = 365
HARNESS = "tools.backtests.run_ema_200_400_backtest"
