"""Cron registration for breakout_60d_high_volume.

Data side: NO-OP (shares N100 equity cache + universe file with Model 3).
Trading side: NOT YET WIRED — model is standalone, can be enabled when
live signal generator is built. For now backtest-only.
"""
from __future__ import annotations


def register_data_jobs(schedule):
    """No-op — daily equity OHLCV + N100 universe already pulled by
    momentum_n100_top5_max1 model. This model reuses that cache."""
    pass


def register_trading_jobs(schedule):
    """Not yet wired. Live signal generator TODO."""
    pass
