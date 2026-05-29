"""Data pull for midcap_breakout_k3 — N500 daily OHLCV via the shared prefetch."""
from __future__ import annotations
import logging, subprocess
from pathlib import Path
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _run(cmd, label, timeout=1800):
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            log.info(f"  ✅ {label}"); return True
        log.error(f"  ❌ {label}: {r.stderr[-500:]}")
    except Exception as e:
        log.error(f"  ❌ {label}: {e}")
    return False


def pull_daily_ohlcv():
    """Refresh N500 daily OHLCV (the model ranks the PIT midcap pool from N500)."""
    return _run(["python3", "tools/shared/prefetch_ohlcv.py",
                 "--universe", "n500", "--days", "5", "--interval", "D"],
                "midcap_breakout_k3 prefetch n500 daily")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); pull_daily_ohlcv()
