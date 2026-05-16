"""Cron registration for midcap_narrow_60d_breakout.

Currently backtest-only (live exec unwired).
"""


def register_data_jobs(schedule):
    """No-op — equity OHLCV pulled by momentum_n100_top5_max1 data cron."""
    pass


def register_trading_jobs(schedule):
    """Not wired — backtest-only winner. Live executor TODO."""
    pass
