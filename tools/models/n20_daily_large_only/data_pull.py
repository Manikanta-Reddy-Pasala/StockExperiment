"""Data pulls for n20_daily_large_only.

Daily (post-market close):
  - N500 daily close OHLCV (shared infra via prefetch_ohlcv.py). Daily PIT
    ranking pool is N500, narrowed to top-20 ADV ∩ N100 at signal time.

Quarterly (NSE Nifty 100 rebalance: Mar/Sep):
  - Refresh nifty100.csv (handled by momentum_n100 already; we register a
    no-op fallback here for self-containment).

Role in the model flow (data_pull -> live_signal -> cron -> backtest)
---------------------------------------------------------------------
This is the FIRST leg: it keeps the inputs the rest of the model reads up to
date. Both functions here are thin wrappers that shell out to shared tools and
are invoked by cron.py's data jobs — they perform no ranking or trading.

  - pull_daily_ohlcv : feeds the N500 price panel that live_signal.py loads
    and that backtest.py replays.
  - refresh_universe : refreshes the Nifty-100 large-cap filter CSV used by
    both live_signal.py and backtest.py.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def _run(cmd: list, label: str, timeout: int = 1800) -> bool:
    """Run a subprocess, log the outcome, and never raise.

    Args:
        cmd: argv list to execute.
        label: human-readable name used in the log lines.
        timeout: seconds before the subprocess is killed (default 1800).

    Returns:
        bool: True on exit code 0; False on non-zero exit, timeout, or any
        exception (errors are logged with the last 500 chars of stderr).
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
    """Incremental N500 daily OHLCV — also covers n20 PIT pool.

    Shells out to the shared prefetch_ohlcv tool for the n50,n500 universes,
    daily ("D") interval, last 5 days only (incremental top-up, not a full
    refetch). The N500 panel it maintains is exactly the point-in-time ranking
    pool that live_signal.py and backtest.py rank from. No return value.
    """
    log.info("=" * 80)
    log.info("n20_daily_large_only daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        # --days 5: small incremental window since this runs every day.
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n50,n500", "--days", "5",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Refresh nifty100.csv from NSE archive (proxies N100 rebalance).

    Shells out to the shared refresh_nifty100 tool to rewrite the local
    Nifty-100 CSV that supplies the large-cap intersection filter. Invoked by
    cron.py only on the quarterly NSE rebalance day. No return value.
    """
    log.info("=" * 80)
    log.info("n20_daily_large_only universe refresh (Nifty 100 from NSE)")
    log.info("=" * 80)
    _run(
        ["python3", "tools/analysis/download_niftyindices.py"],
        "refresh_nifty_universes", timeout=180,
    )
