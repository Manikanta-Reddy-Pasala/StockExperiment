"""Data pulls required by breakout_60d_high_volume model.

Daily:
  - N100 daily OHLCV (close + volume) — same as Model 3 momentum, shares cache
  - Universe file (pseudo-N100) — reused from momentum model

Monthly:
  - Universe refresh (handled by momentum model — same N100 file)

Net effect: this model PIGGYBACKS on Model 3's data pulls. No separate
schedule jobs needed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def noop():
    """Breakout model uses same equity OHLCV cache as Model 3.
    Daily pulls + universe refresh are covered by momentum_n100_top5_max1.
    """
    log.info("breakout_60d_high_volume: data shared with momentum_n100_top5_max1")
