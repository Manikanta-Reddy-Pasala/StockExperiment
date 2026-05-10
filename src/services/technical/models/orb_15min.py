"""15-minute Opening Range Breakout (5m bars) model wrapper (KISS registry entry)."""
from src.services.technical.orb_15min import (
    ORB15MinStrategy,
    ORBConfig,
)

NAME = "orb_15min"
DESCRIPTION = "15-min opening-range breakout intraday on 5m bars (same-day exit)."
STRATEGY_CLASS = ORB15MinStrategy
CONFIG_CLASS = ORBConfig
DEFAULT_CONFIG = ORBConfig()
BARS_INTERVAL = "5m"
DEFAULT_WINDOW_DAYS = 90
HARNESS = "tools.backtests.run_orb_intraday_backtest"
