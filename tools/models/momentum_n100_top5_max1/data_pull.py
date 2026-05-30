"""Data pulls required by Model 3 — momentum_n100_top5_max1.

This is the FIRST stage of the model's flow:

    data_pull (this file)  ->  build_universe  ->  live_signal (emits signals)
                           ->  cron (schedules)  ->  backtest (validates)

It fills two prerequisites that every later stage depends on:
  1. The OHLCV cache (historical_data table) that ranking reads from.
  2. The real NSE Nifty 100 constituent list + the universe JSON that
     live_signal.py / backtest.py rank against.

Daily (post-market close):
  - NIFTY 100 daily close OHLCV (cache via prefetch_ohlcv)

Quarterly (NSE rebalance: Mar/Sep):
  - Refresh src/data/symbols/nifty100.csv from NSE archives
  - Rebuild n100_current.json from updated CSV

These functions are registered as scheduled jobs in cron.py
(register_data_jobs). They shell out to the shared prefetch/refresh tools
rather than doing the work inline, so the model owns only the schedule, not
the data-fetch implementation. Data source is Fyers (project rule: never
yfinance in production).
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
    """Run a subprocess command, log the outcome, and report success.

    Args:
        cmd: argv list passed straight to subprocess.run (no shell).
        label: human-readable name used in the log lines.
        timeout: seconds before the child is killed (default 30 min — these
            are bulk OHLCV/universe pulls and can be slow).

    Returns:
        True only on exit code 0. Any non-zero exit, timeout, or exception is
        caught and logged (last 500 chars of stderr) and returns False — the
        scheduler must not crash because a single nightly pull failed.
    """
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
    """Incremental N500 daily OHLCV (2 days lookback, just-in-case backfill).

    Refreshes the daily-close cache that ranking reads. Pulls the n50,n500
    universes (a superset of the Nifty 100 the model actually ranks) so any
    constituent change is already covered. `--days 5` re-fetches the last few
    sessions to backfill late corrections; `--sleep 0.2` throttles the Fyers
    API to stay under rate limits.

    No return value — failures are logged by _run and tolerated; the next
    nightly run retries.
    """
    log.info("=" * 80)
    log.info("Model 3 daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n50,n500", "--days", "5",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Refresh real Nifty 100 from NSE CSV + rebuild universe file.

    Two sequential steps (each tolerant of failure via _run):
      1. tools/refresh_nifty100.py — pulls the latest official constituent
         list from NSE archives into src/data/symbols/nifty100.csv.
      2. build_universe.py — turns that CSV into the n100_current.json the
         live signal/backtest consume (see UNIVERSE_OUT).

    Triggered monthly by cron._monthly_universe (only fires on the 1st), which
    covers the quarterly NSE index rebalances (Mar/Sep) plus any ad-hoc change.
    build_universe is idempotent, so re-running on a no-change month is safe.
    """
    log.info("=" * 80)
    log.info("Model 3 universe refresh (real NIFTY 100 from NSE)")
    log.info("=" * 80)
    Path(UNIVERSE_OUT).parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["python3", "tools/analysis/download_niftyindices.py"],
        "refresh_nifty_universes", timeout=180,
    )
    _run(
        ["python3", "tools/models/momentum_n100_top5_max1/build_universe.py",
         "--out", UNIVERSE_OUT],
        "build_universe", timeout=120,
    )
