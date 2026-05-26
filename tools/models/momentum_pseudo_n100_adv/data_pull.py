"""Data pulls required by momentum_pseudo_n100_adv.

Pipeline position (data_pull -> build_universe -> live_signal -> cron -> backtest):
  This is the first stage. It (a) keeps the local N500 OHLCV cache current so
  build_universe can rank by ADV and live_signal can score momentum, and
  (b) orchestrates the yearly universe rebuild by shelling out to
  build_universe.py and merging its snapshot into yearly_universes.json — the
  file live_signal reads to choose the universe for a trading date. cron.py
  schedules both functions here.

Daily (post-market close):
  - N500 daily close OHLCV (shared with momentum_n100 and midcap_narrow via
    tools/shared/prefetch_ohlcv.py — same historical_data table). The
    pseudo-N100 PIT universe is a subset of N500 so this covers it.

Yearly (May rebalance):
  - Rebuild yearly_universes.json by ranking N500 by 20d ADV at year-start.
    Uses only data observable at the rebuild date — PIT-safe for live
    deployment.
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
    """Run a subprocess command, logging a labelled success/failure result.

    Args:
        cmd: argv list passed to subprocess.run (no shell).
        label: Human-readable tag prefixed onto the ok/fail/timeout log lines.
        timeout: Hard wall-clock limit in seconds before the child is killed.

    Returns:
        True only on exit code 0; False on non-zero exit, timeout, or any
        exception. On failure the tail of stderr (last 500 chars) is logged.
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
    """Refresh the local N500 daily OHLCV cache (incremental, short lookback).

    Runs after market close (scheduled by cron.register_data_jobs). Shells out
    to the shared prefetch_ohlcv.py so this model reuses the same historical
    data table as momentum_n100 / midcap_narrow rather than its own copy.

    Returns:
        None. Side effect: upserts recent daily bars into the shared cache.

    Non-obvious logic:
        - --universe n50,n500: pulls both lists; pseudo-N100 is a subset of
          N500 so the N500 fetch already covers every symbol this model needs.
        - --days 5: only a 5-day lookback window each run (incremental top-up
          of the rolling cache, not a full re-download).
        - --intervals D: daily candles only; --sleep 0.2 throttles API calls.
    """
    log.info("=" * 80)
    log.info("momentum_pseudo_n100_adv daily OHLCV pull (N500)")
    log.info("=" * 80)
    _run(
        # 5-day incremental top-up of the shared N500 daily-bar cache.
        ["python3", "tools/shared/prefetch_ohlcv.py",
         "--universe", "n50,n500", "--days", "5",
         "--intervals", "D", "--sleep", "0.2"],
        "prefetch_ohlcv_daily", timeout=1800,
    )


def refresh_universe():
    """Rebuild PIT universe via build_universe.py (top-100 by ADV) and
    MERGE the result into yearly_universes.json under today's date key.

    Called on month-1 of each year (May) by cron, or on-demand from the
    /admin 'Pull Data Now' button. Output side-effect:
      - One-off snapshot at /app/exports/backtests/pseudo_n100_{date}.json
      - Same symbols merged into yearly_universes.json under "YYYY-MM-DD"
        key so live_signal.pick_universe_for() finds it immediately.

    Returns:
        None. Returns early (no merge) if build_universe.py fails or the
        snapshot yields fewer than 50 symbols (treated as a bad/partial build).

    Non-obvious logic:
        - PIT-safe: the snapshot is keyed by today's date and built only from
          data observable now, so live deployment never sees future data.
        - Merge (not overwrite): existing date keys in yearly_universes.json
          are preserved; only today's key is added/replaced.
        - The snapshot reader tolerates both dict ({"stocks"/"symbols": ...})
          and bare-list shapes, and both {"symbol": ...} dicts and plain
          string entries.
    """
    import json
    log.info("=" * 80)
    log.info("momentum_pseudo_n100_adv yearly PIT universe refresh")
    log.info("=" * 80)
    # Today's date doubles as both the build's PIT as-of date and the merge key.
    end_date = datetime.now().strftime("%Y-%m-%d")
    out_file = f"/app/exports/backtests/pseudo_n100_{end_date}.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    ok = _run(
        # Delegate the actual top-100-by-ADV ranking to build_universe.py.
        ["python3",
         "tools/models/momentum_pseudo_n100_adv/build_universe.py",
         "--top", "100", "--end-date", end_date, "--out", out_file],
        "build_pseudo_n100_universe", timeout=600,
    )
    if not ok:
        # Don't clobber a good yearly_universes.json with a failed build.
        log.error("build_universe failed — yearly_universes.json NOT updated")
        return
    try:
        snapshot = json.loads(Path(out_file).read_text())
        # Accept dict-with-list-key or bare-list snapshot shapes.
        if isinstance(snapshot, dict):
            entries = snapshot.get("stocks") or snapshot.get("symbols") or []
        else:
            entries = snapshot
        # Normalize each entry to a plain symbol string.
        symbols = [e["symbol"] if isinstance(e, dict) else e for e in entries]
        # Sanity floor: a healthy pseudo-N100 build has ~100 names.
        if len(symbols) < 50:
            log.warning(f"build_universe returned only {len(symbols)} — skipping merge")
            return
        yearly = {}
        # Preserve prior year keys — load existing file before adding today's.
        if Path(UNIVERSES_FILE).exists():
            yearly = json.loads(Path(UNIVERSES_FILE).read_text())
        # Add/replace only today's key; live_signal looks it up by date.
        yearly[end_date] = [{"symbol": s} for s in symbols]
        Path(UNIVERSES_FILE).write_text(json.dumps(yearly, indent=2))
        log.info(f"  merged {len(symbols)} symbols into yearly_universes.json as '{end_date}'")
    except Exception as e:
        log.error(f"  yearly_universes.json merge failed: {e}", exc_info=True)
