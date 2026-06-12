"""Data pulls for midcap_narrow_60d_breakout — STEP 1 of the model flow.

Pipeline position (data_pull -> build_universe -> live_signal -> cron -> backtest):
  This is the very first stage. It refreshes the raw OHLCV the rest of the
  model reads from, and triggers the universe rebuild. Everything downstream
  (build_universe ADV ranking, live_signal breakout scan, backtest replay)
  reads the `historical_data` table this module keeps current.

Daily (post-market close):
  - NIFTY 500 daily close OHLCV (shared with momentum_n100_top5_max1 via
    tools/shared/prefetch_ohlcv.py — same `historical_data` table)
  - Symbols in midcap_narrow universe are a subset of N500, so the N500
    pull already covers them. We include an explicit incremental pull
    here as a model-local fallback.

Monthly (1st trading day):
  - Refresh midcap_narrow.json universe (ADV rank drift)

Both functions here are thin subprocess wrappers; the heavy lifting lives in
tools/shared/prefetch_ohlcv.py (OHLCV) and build_universe.py (universe). They
are invoked by cron.py's register_data_jobs().
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

UNIVERSE_OUT = "/app/logs/momrot/universes/midcap_narrow.json"
# SKIP_TOP=0 to match backtest.py (V3 rule: top-100 ADV from N500 minus Large,
# SKIP_TOP=0). Was 30, which silently skipped the 30 most-liquid names and built
# a different live universe than the validated backtest. KEEP_NEXT(=top) stays.
SKIP_TOP = 0
KEEP_NEXT = 100


def _run(cmd: list, label: str, timeout: int = 1800) -> bool:
    """Run a child process, log a tidy pass/fail line, and report success.

    Args:
        cmd: argv list passed straight to subprocess.run (no shell).
        label: human-readable name used in the success/failure log lines.
        timeout: seconds before the child is killed (default 1800 = 30 min,
            sized for a full N500 OHLCV pull).

    Returns:
        True if the child exited 0, else False. All failure modes
        (non-zero exit, timeout, unexpected exception) are caught and logged
        so a single failed pull never crashes the scheduler thread.
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
    """Incremental N500 daily OHLCV pull. Covers the midcap_narrow subset.

    Shells out to the shared prefetcher with --days 5 (small lookback so the
    last few sessions are refreshed/back-filled, not the whole history). Runs
    daily post-close via cron.py register_data_jobs(). No return value — result
    is logged by _run; the refreshed `historical_data` table is the side effect
    every other stage consumes.
    """
    log.info("=" * 80)
    log.info("midcap_narrow_60d_breakout daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n500", "--days", "5",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Rebuild the midcap_narrow universe JSON by ADV ranking. Monthly.

    Shells out to build_universe.py with this module's SKIP_TOP/KEEP_NEXT
    constants and writes UNIVERSE_OUT. Re-running monthly lets the universe
    drift as liquidity (ADV) ranks change. No return value — result logged by
    _run; the produced JSON is what live_signal.py loads each day.

    NOTE: these SKIP_TOP=30/KEEP_NEXT=100 constants pre-date the V3 universe
    rule (top-100 ADV minus Large = SKIP_TOP=0 in backtest.py). build_universe.py
    itself treats large-cap removal as the Nifty-100 exclusion, so the net band
    still lands on ~42 midcaps.
    """
    log.info("=" * 80)
    log.info("midcap_narrow universe refresh (ADV-ranked, skip large caps)")
    log.info("=" * 80)
    Path(UNIVERSE_OUT).parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["python3", "tools/models/midcap_narrow_60d_breakout/build_universe.py",
         "--skip-top", str(SKIP_TOP),
         "--top", str(KEEP_NEXT),
         "--out", UNIVERSE_OUT],
        "build_midcap_narrow_universe", timeout=600,
    )
