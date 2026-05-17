"""Data pulls required by momentum_pseudo_n100_adv.

Daily (post-market close):
  - N500 daily close OHLCV (shared with momentum_n100 and midcap_narrow via
    tools/shared/prefetch_ohlcv.py — same historical_data table). The
    pseudo-N100 PIT universe is a subset of N500 so this covers it.

Yearly (May rebalance):
  - Rebuild yearly_universes.json by ranking N500 by 20d ADV at year-start.
    NOTE: current yearly_universes.json was built with forward-looking ADV
    (lookahead bias). Rebuild is gated by --allow-lookahead flag.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

UNIVERSES_FILE = (
    "/app/tools/models/momentum_pseudo_n100_adv/yearly_universes.json"
)


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
    """Incremental N500 daily OHLCV (2 days lookback)."""
    log.info("=" * 80)
    log.info("momentum_pseudo_n100_adv daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n50,n500", "--days", "2",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Rebuild yearly PIT universe via build_universe.py (top-100 by ADV).

    Called on month-1 of each year (May). Output: yearly_universes.json
    appended/updated with new year-start key.
    """
    log.info("=" * 80)
    log.info("momentum_pseudo_n100_adv yearly PIT universe refresh")
    log.info("=" * 80)
    end_date = datetime.now().strftime("%Y-%m-%d")
    out_file = f"/app/exports/backtests/pseudo_n100_{end_date}.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["python3",
         "tools/models/momentum_pseudo_n100_adv/build_universe.py",
         "--top", "100", "--end-date", end_date, "--out", out_file],
        "build_pseudo_n100_universe", timeout=600,
    )
