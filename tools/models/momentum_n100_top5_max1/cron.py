"""Cron registration for Model 3 — momentum_n100_top5_max1.

Two register functions:
  register_data_jobs(schedule)   -- called by data_scheduler.py
  register_trading_jobs(schedule) -- called by scheduler.py (technical_scheduler)

Keeps schedule definitions co-located with the model so adding a new model
later is a single import + register call from the main scheduler.
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

from tools.models.momentum_n100_top5_max1.data_pull import (  # noqa: E402
    pull_daily_ohlcv, refresh_universe,
)

log = logging.getLogger(__name__)


def _alert(msg: str):
    """FIX 9 — best-effort Telegram alert for scheduler failures. Wrapped so a
    notify failure never crashes the scheduler thread (log-only was silent)."""
    try:
        from tools.live.telegram_notify import send as _tg
        _tg(msg)
    except Exception as e:
        log.debug(f"tg alert skipped: {e}")


# ---- Trading-side jobs (technical_scheduler) ----

def emit_signal(force: bool = False):
    """Emit Model 3 signal. Rebalance-gated unless force=True.

    Shells out to live_signal.py with --rebalance-only so the signal file is
    only populated on the monthly rebalance trigger day; live_signal itself
    decides whether today qualifies and writes an empty file otherwise. This
    is the SIGNAL half of the daily trading pair (execute_orders is the
    EXECUTION half, fired 5 minutes later).

    Args:
        force: pass --force instead of --rebalance-only, bypassing the
            monthly date gate (used for initial deploy / manual reruns). Note:
            forced runs are intentionally NOT audited inside live_signal.

    Side effects: writes /app/logs/momrot/signals/{today}_momrot_n100.json.
    Subprocess failures are logged, never raised — the scheduler keeps running.
    """
    label = "Model 3 momentum signal" + (" (FORCE)" if force else " (rebalance-gated)")
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {label}")
    log.info("=" * 80)
    universe = os.environ.get("UNIVERSE_FILE",
                              "/app/logs/momrot/universes/n100_current.json")
    today = datetime.now().strftime("%Y-%m-%d")
    signals_dir = Path("/app/logs/momrot/signals")
    signals_dir.mkdir(parents=True, exist_ok=True)
    signals_out = signals_dir / f"{today}_momrot_n100.json"
    cmd = [
        "python3", "tools/models/momentum_n100_top5_max1/live_signal.py",
        "--universe-file", universe, "--top-n", "5",
        "--signals-out", str(signals_out),
    ]
    # Gate the signal: forced runs ignore the date; normal runs self-skip
    # unless today is the monthly rebalance trigger.
    cmd.append("--force" if force else "--rebalance-only")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info(f"✅ Model 3 signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
            # Telegram/PWA notification is emitted inside live_signal.py via the
            # unified notification service (pings even on no-change). See
            # src/services/notification_service.py::notify_model_decision.
        else:
            log.error(f"❌ Model 3 signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ momentum_n100_top5_max1 emit_signal failed "
                   f"(rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ Model 3 signal error: {e}")
        _alert(f"❌ momentum_n100_top5_max1 emit_signal error: {e}")


def execute_orders():
    """Place Fyers orders from today's signal file.

    EXECUTION half of the monthly pair. Reads the signal file emit_signal
    wrote and hands it to tools/live/fyers_executor.py, which places the
    real BUY/SELL orders (max_concurrent=1, so a duplicate ENTRY1 for an
    already-held symbol is a no-op). No return value; failures are logged.

    Gotcha: on non-rebalance days the file is absent (or empty), so this
    correctly skips — the missing file is the gate, not an error.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = Path(f"/app/logs/momrot/signals/{today}_momrot_n100.json")
    if not signals_file.exists():
        log.info(f"Model 3 execute: no signal at {signals_file}, skipping.")
        return
    log.info("PLACING MODEL 3 FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", "momentum_n100_top5_max1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode == 0:
            log.info("✅ Model 3 Fyers execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ Model 3 Fyers execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ momentum_n100_top5_max1 execute failed "
                   f"(rc={r.returncode}) — orders may not be placed")
    except Exception as e:
        log.error(f"❌ Model 3 Fyers execute error: {e}")
        _alert(f"❌ momentum_n100_top5_max1 execute error: {e}")


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Daily/monthly data pulls. Called by data_scheduler.

    Args:
        schedule: the `schedule` library instance owned by data_scheduler.py;
            we attach our jobs to it so all model data jobs share one clock.

    Registers the data-stage jobs (stage 1 of the model flow): a nightly
    OHLCV refresh and a daily universe-refresh wrapper that self-gates to the
    1st of the month. No return value.
    """
    # Equity OHLCV — daily after market close (saga step 3 also covers this;
    # this is the model-explicit fallback)
    schedule.every().day.at("20:30").do(pull_daily_ohlcv)
    # Universe refresh — first of every month (build_universe.py is idempotent)
    schedule.every().day.at("06:30").do(_monthly_universe)


def _monthly_universe():
    """Wrapper that only runs universe refresh on 1st of month.

    The scheduler fires this every day at 06:30; the day-of-month guard turns
    it into an effective monthly job without needing a monthly cron primitive.
    No-op on days 2-31.
    """
    if datetime.now().day == 1:  # only the 1st triggers the actual refresh
        refresh_universe()


def emit_mid_month_signal():
    """Day-15 weekday mid-month check. Live_signal applies the 5pp
    lead gate; on non-day-15 it writes an empty signals file. Cron
    can fire daily and the model self-skips.

    This is the SIGNAL half of the second (mid-cycle) rebalance opportunity.
    It runs live_signal.py with --mid-month-check, which both date-gates to
    the first weekday on/after the 15th AND only rotates when the new rank-1
    leads the held stock by >= MID_MONTH_LEAD_PCT (see live_signal). Writes to
    a SEPARATE *_midmonth.json file so it never collides with the monthly
    signal. No return value; subprocess failures are logged.
    """
    log.info("\n" + "=" * 80)
    log.info("RUNNING Model 3 mid-month check")
    log.info("=" * 80)
    universe = os.environ.get("UNIVERSE_FILE",
                              "/app/logs/momrot/universes/n100_current.json")
    today = datetime.now().strftime("%Y-%m-%d")
    signals_dir = Path("/app/logs/momrot/signals")
    signals_dir.mkdir(parents=True, exist_ok=True)
    signals_out = signals_dir / f"{today}_momrot_n100_midmonth.json"
    cmd = [
        "python3", "tools/models/momentum_n100_top5_max1/live_signal.py",
        "--universe-file", universe, "--top-n", "5",
        "--signals-out", str(signals_out),
        "--mid-month-check",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info("✅ Model 3 mid-month check complete")
            if r.stdout:
                log.info(r.stdout[-500:])
            # Telegram/PWA notification is emitted inside live_signal.py via the
            # unified notification service (pings even on no-change). See
            # src/services/notification_service.py::notify_model_decision.
        else:
            log.error(f"❌ Model 3 mid-month check failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ momentum_n100_top5_max1 mid-month emit failed "
                   f"(rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ Model 3 mid-month error: {e}")
        _alert(f"❌ momentum_n100_top5_max1 mid-month emit error: {e}")


def execute_mid_month_orders():
    """Execute mid-month signal file (separate from monthly to avoid
    double-execution race).

    EXECUTION half of the mid-cycle pair. Reads the *_midmonth.json file and
    places its orders via fyers_executor.py. Most days the file is empty
    (gate not met) so this is usually a no-op. Using a distinct file from the
    monthly path is deliberate: it stops the 09:30 monthly executor and the
    09:35 mid-month executor from racing on the same signal. No return value.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = Path(f"/app/logs/momrot/signals/{today}_momrot_n100_midmonth.json")
    if not signals_file.exists():
        log.info(f"Model 3 mid-month execute: no signal at {signals_file}, skipping.")
        return
    # Skip if the file exists but holds an empty list (the common case — the
    # mid-month gate suppressed rotation). Parse errors fall through and let
    # the executor decide.
    try:
        import json as _j
        sigs = _j.loads(signals_file.read_text())
        if not sigs:
            log.info("Model 3 mid-month: no signals to execute.")
            return
    except Exception:
        pass
    log.info("PLACING MODEL 3 MID-MONTH FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", "momentum_n100_top5_max1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode == 0:
            log.info("✅ Model 3 mid-month execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ Model 3 mid-month execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ momentum_n100_top5_max1 mid-month execute failed "
                   f"(rc={r.returncode}) — orders may not be placed")
    except Exception as e:
        log.error(f"❌ Model 3 mid-month execute error: {e}")
        _alert(f"❌ momentum_n100_top5_max1 mid-month execute error: {e}")


def register_trading_jobs(schedule):
    """Signal + execute. Called by technical_scheduler."""
    schedule.every().day.at("09:25").do(emit_signal)        # rebalance-gated
    schedule.every().day.at("09:30").do(execute_orders)
    # Mid-month rank check (day-15 weekday). live_signal self-skips on
    # non-day-15 so safe to fire daily. 5pp lead threshold.
    schedule.every().day.at("09:27").do(emit_mid_month_signal)
    schedule.every().day.at("09:35").do(execute_mid_month_orders)
    # DAILY from-entry fixed-% hard-stop check (2026-06-02). Every trading day,
    # staggered last so a same-day rotation entry isn't falsely stopped.
    schedule.every().day.at("09:37").do(stop_check)


def stop_check():
    """DAILY from-entry fixed-% hard-stop check (every trading day). live_signal
    --stop-check emits a STOP_HIT SELL to a *_stop.json file if the held name's
    last completed bar breached entry*(1-STOP_PCT); we then execute it. Shared
    stop helper -> matches the backtest exactly."""
    import json as _json
    universe = os.environ.get("UNIVERSE_FILE",
                              "/app/logs/momrot/universes/n100_current.json")
    today = datetime.now().strftime("%Y-%m-%d")
    signals_dir = Path("/app/logs/momrot/signals"); signals_dir.mkdir(parents=True, exist_ok=True)
    stop_out = signals_dir / f"{today}_momrot_n100_stop.json"
    cmd = ["python3", "tools/models/momentum_n100_top5_max1/live_signal.py",
           "--universe-file", universe, "--top-n", "5",
           "--signals-out", str(stop_out), "--stop-check"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode != 0:
            log.error(f"❌ n100 stop-check failed ({r.returncode}): {r.stderr[-300:]}")
            _alert(f"❌ n100 stop-check failed (rc={r.returncode})"); return
        if r.stdout:
            log.info(r.stdout[-300:])
        if stop_out.exists() and _json.loads(stop_out.read_text() or "[]"):
            user_id = os.environ.get("USER_ID", "1")
            ex = subprocess.run(
                ["python3", "tools/live/fyers_executor.py", "--signals", str(stop_out),
                 "--user-id", user_id, "--model-name", "momentum_n100_top5_max1"],
                capture_output=True, text=True, timeout=1200)
            (log.info if ex.returncode == 0 else log.error)(
                f"{'✅' if ex.returncode==0 else '❌'} n100 stop execute"
                + ("" if ex.returncode == 0 else f" rc={ex.returncode}: {ex.stderr[-300:]}"))
            if ex.returncode != 0:
                _alert(f"❌ n100 stop execute failed (rc={ex.returncode})")
    except Exception as e:
        log.error(f"❌ n100 stop-check error: {e}"); _alert(f"❌ n100 stop-check error: {e}")
