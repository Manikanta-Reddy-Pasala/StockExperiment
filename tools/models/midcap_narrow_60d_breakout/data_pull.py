"""Data pulls for midcap_narrow_60d_breakout.

Shares equity OHLCV with momentum_n100_top5_max1 (same `historical_data` table).
Universe file `midcap_narrow.json` lives under `logs/momrot/universes/`.
"""
import logging

log = logging.getLogger(__name__)


def noop():
    log.info("midcap_narrow_60d_breakout: equity OHLCV shared with momentum_n100_top5_max1")
