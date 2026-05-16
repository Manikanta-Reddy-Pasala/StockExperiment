"""Data pulls required by Model 3 — momentum_n100_top5_max1.

Daily (post-market close):
  - NIFTY 500 daily close OHLCV (cache via prefetch_ohlcv)
  - Compute pseudo-N100 universe from 20-day ADV ranking

Monthly (1st trading day):
  - Refresh n100_current.json (universe drift across rotations)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

UNIVERSE_OUT = "/app/logs/momrot/universes/n100_current.json"


def _run(cmd: list, label: str, timeout: int = 1800) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            log.info(f"  ✅ {label} ok")
            return True
        log.error(f"  ❌ {label} failed (rc={r.returncode})")
        if r.stderr:
            log.error(r.stderr[-500:])
    except subprocess.TimeoutExpired:
        log.error(f"  ❌ {label} timeout ({timeout}s)")
    except Exception as e:
        log.error(f"  ❌ {label} error: {e}")
    return False


def pull_daily_ohlcv():
    """Incremental N500 daily OHLCV (2 days lookback, just-in-case backfill)."""
    log.info("=" * 80)
    log.info("Model 3 daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n50,n500", "--days", "2",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Rebuild pseudo-N100 universe by 20d ADV. Runs monthly + bootstrap."""
    log.info("=" * 80)
    log.info("Model 3 universe refresh (pseudo-N100 by ADV)")
    log.info("=" * 80)
    Path(UNIVERSE_OUT).parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["python3", "tools/models/momentum_n100_top5_max1/build_universe.py",
         "--top", "100", "--out", UNIVERSE_OUT],
        "build_universe", timeout=600,
    )
