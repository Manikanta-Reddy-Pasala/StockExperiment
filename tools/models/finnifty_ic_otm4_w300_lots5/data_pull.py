"""Data pulls required by FinNifty IC option model.

Daily:
  - Index spots: NIFTY50, BANKNIFTY, FINNIFTY (Fyers history)
  - Option bhavcopy: NIFTY/BANKNIFTY/FINNIFTY OPTIDX (NSE archives EOD)

No trading-side jobs — model is UNWIRED for execution. Data still flows
so backtests + future research can pick up where they left off.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def _run(cmd: list, label: str, timeout: int = 900) -> bool:
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


def fetch_index_spots():
    """Pull NIFTY50 / BANKNIFTY / FINNIFTY spot daily OHLC."""
    log.info("=" * 80)
    log.info("FinNifty data: index spots (NIFTY50 / BANKNIFTY / FINNIFTY)")
    log.info("=" * 80)
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    for sym in ("NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"):
        _run(
            ["python3", "tools/shared/fetch_index_spot.py",
             "--symbol", sym, "--from", start, "--to", today],
            f"index_spot:{sym}", timeout=300,
        )


def fetch_option_bhav():
    """Pull NSE FO bhavcopy for NIFTY/BANKNIFTY/FINNIFTY (3-day lookback)."""
    log.info("=" * 80)
    log.info("FinNifty data: option bhavcopy (NIFTY / BANKNIFTY / FINNIFTY)")
    log.info("=" * 80)
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    _run(
        ["python3", "tools/shared/prefetch_bhav.py",
         "--from", start, "--to", today,
         "--underlying", "NIFTY,BANKNIFTY,FINNIFTY",
         "--instrument", "OPTIDX"],
        "option_bhav", timeout=900,
    )
