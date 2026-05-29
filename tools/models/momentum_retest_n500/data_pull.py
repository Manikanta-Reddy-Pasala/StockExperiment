"""Data pull for momentum_retest_n500.

The model ranks the top-120-ADV liquid pool out of Nifty-500, so it needs the
N500 daily OHLCV panel — pulled via the SHARED prefetch tool (same infra the
other models use; upserts so a shared/duplicate pull is harmless). The Smallcap-
250 exclusion list comes from the self-healing universe CSVs (data_scheduler).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _run(cmd, label, timeout=1800):
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            log.info(f"  ✅ {label}")
            return True
        log.error(f"  ❌ {label}: {r.stderr[-500:]}")
    except Exception as e:
        log.error(f"  ❌ {label}: {e}")
    return False


def pull_daily_ohlcv():
    """Refresh the N500 daily OHLCV panel via the shared prefetch tool."""
    return _run(["python3", "tools/shared/prefetch_ohlcv.py",
                 "--universe", "n500", "--days", "5", "--interval", "D"],
                "momentum_retest_n500 prefetch n500 daily")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pull_daily_ohlcv()
