"""Cron registration for momentum_pseudo_n100_adv.

Pipeline position (data_pull -> build_universe -> live_signal -> cron -> backtest):
  This is the scheduling glue that wires the model into the two running
  schedulers. It does no analytics itself — it registers timed jobs that call
  data_pull (OHLCV cache top-up + yearly universe rebuild) and live_signal
  (daily signal emit), then hands the resulting signal file to the Fyers
  executor. backtest is the offline counterpart that replays the same
  universe/signal logic on history.

Two register functions:
  register_data_jobs(schedule)   -- called by data_scheduler.py
  register_trading_jobs(schedule) -- called by scheduler.py (technical_scheduler)

The yearly-PIT universe is rebuilt at year-start (mid-May) using current
data at that time — PIT-safe for live deployment. The live_signal will
short-circuit if the enabled flag is False (toggle via UI).
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from tools.models.momentum_pseudo_n100_adv.data_pull import (  # noqa: E402
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


MODEL_NAME = "momentum_pseudo_n100_adv"
UNIVERSES_FILE = (
    "/app/tools/models/momentum_pseudo_n100_adv/yearly_universes.json"
)
SIGNALS_DIR = Path("/app/logs/momrot_pseudo/signals")


# ---- Trading-side jobs ----

def emit_signal(force: bool = False):
    """Run live_signal.py to produce today's pseudo-N100 momentum signal file.

    Args:
        force: When True, pass --force so live_signal emits regardless of the
            monthly rotation cadence. When False (the scheduled path), pass
            --rebalance-only so live_signal only acts on its rebalance day.

    Returns:
        None. Side effect: writes today's signal JSON to SIGNALS_DIR and logs
        the outcome. live_signal itself further short-circuits if
        model_settings.enabled is False and sends the Telegram/PWA ping.

    Non-obvious logic:
        - --top-n 5: live_signal ranks candidates and the model holds rank-1
          from the top-5 (monthly rotation); this caps the candidate slice.
        - subprocess timeout 600s guards against a hung signal computation.
        - Only the last 500 chars of stdout/stderr are logged to keep cron
          logs bounded.
    """
    label = ("pseudo-N100 momentum signal"
             + (" (FORCE)" if force else " (rebalance-gated)"))
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {label}")
    log.info("=" * 80)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = SIGNALS_DIR / f"{today}_pseudo_n100.json"
    cmd = [
        "python3", "tools/models/momentum_pseudo_n100_adv/live_signal.py",
        "--universes-file", UNIVERSES_FILE,
        "--top-n", "5",
        "--signals-out", str(signals_out),
    ]
    # Gate selector: force always emits; otherwise defer to the rebalance day.
    cmd.append("--force" if force else "--rebalance-only")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
            # Telegram/PWA notification is emitted inside live_signal.py via the
            # unified notification service (pings even on no-change).
        else:
            log.error(f"❌ {MODEL_NAME} signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ {MODEL_NAME} emit_signal failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} signal error: {e}")
        _alert(f"❌ {MODEL_NAME} emit_signal error: {e}")


def execute_orders():
    """Place Fyers orders from today's signal file, if one exists.

    Returns:
        None. No-op (logs and returns) when emit_signal produced no file for
        today — e.g. a non-rebalance day where the gate suppressed the signal.

    Non-obvious logic:
        - Reads USER_ID from env (default "1") so the executor books to the
          right account; --model-name tags audit_orders for this model.
        - subprocess timeout 300s; only the last 500 chars of output logged.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}_pseudo_n100.json"
    # No signal file today (gate suppressed it) -> nothing to execute.
    if not signals_file.exists():
        log.info(f"{MODEL_NAME} execute: no signal at {signals_file}, skipping.")
        return
    log.info(f"PLACING {MODEL_NAME} FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", MODEL_NAME]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} Fyers execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ {MODEL_NAME} Fyers execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ {MODEL_NAME} execute failed (rc={r.returncode}) — "
                   f"orders may not be placed")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} Fyers execute error: {e}")
        _alert(f"❌ {MODEL_NAME} execute error: {e}")


def emit_mid_month_signal():
    """Day-15 weekday mid-month check (2026-05-30 config). live_signal applies
    the 3pp lead gate and self-skips on non-day-15 days, so this is safe to fire
    daily. Writes a SEPARATE *_midmonth.json so it never collides with the
    monthly signal file. SIGNAL half of the mid-cycle rebalance opportunity."""
    log.info("RUNNING pseudo-N100 mid-month check")
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = SIGNALS_DIR / f"{today}_pseudo_n100_midmonth.json"
    cmd = [
        "python3", "tools/models/momentum_pseudo_n100_adv/live_signal.py",
        "--universes-file", UNIVERSES_FILE, "--top-n", "5",
        "--signals-out", str(signals_out), "--mid-month-check",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} mid-month check -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ {MODEL_NAME} mid-month check failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ {MODEL_NAME} mid-month emit failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} mid-month error: {e}")
        _alert(f"❌ {MODEL_NAME} mid-month emit error: {e}")


def execute_mid_month_orders():
    """Execute the mid-month signal file (separate from the monthly file to
    avoid a double-execution race). Usually a no-op — the lead gate suppresses
    most mid-cycle rotations, leaving an empty list."""
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}_pseudo_n100_midmonth.json"
    if not signals_file.exists():
        log.info(f"{MODEL_NAME} mid-month execute: no signal at {signals_file}, skipping.")
        return
    try:
        import json as _j
        if not _j.loads(signals_file.read_text()):
            log.info(f"{MODEL_NAME} mid-month: no signals to execute.")
            return
    except Exception:
        pass
    log.info(f"PLACING {MODEL_NAME} MID-MONTH FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", MODEL_NAME]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} mid-month execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ {MODEL_NAME} mid-month execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ {MODEL_NAME} mid-month execute failed (rc={r.returncode}) — "
                   f"orders may not be placed")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} mid-month execute error: {e}")
        _alert(f"❌ {MODEL_NAME} mid-month execute error: {e}")


# ---- Data-side helpers ----

def _yearly_universe():
    """Rebuild the yearly PIT universe, but only on the May rebalance date.

    Wraps refresh_universe behind a date gate so the same daily-scheduled job
    is a no-op every day except May 15. May 15 is used as a proxy for the NSE
    semi-annual index rebalance.

    Returns:
        None. Triggers data_pull.refresh_universe() only on May 15.

    Non-obvious logic:
        - The check is intentionally inside the daily job (rather than a
          once-a-year schedule) so the scheduler library only needs simple
          daily timers.
    """
    # Date gate: fire the rebuild on May 15 only; no-op every other day.
    if datetime.now().month == 5 and datetime.now().day == 15:
        refresh_universe()


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Register the data-side timers on the data_scheduler's schedule object.

    Args:
        schedule: The shared `schedule` library instance owned by
            data_scheduler.py.

    Returns:
        None. Registers two daily jobs (mutates `schedule`).

    Non-obvious logic:
        - 20:35 OHLCV pull runs post-market-close (IST) so the cache holds the
          day's settled bars before any ranking.
        - 06:32 universe job runs daily but is internally gated to May 15 by
          _yearly_universe (no-op on all other days).
    """
    schedule.every().day.at("20:35").do(pull_daily_ohlcv)  # post-close cache top-up
    # PIT universe rebuild — May 15 only (no-op other days)
    schedule.every().day.at("06:32").do(_yearly_universe)


def register_trading_jobs(schedule):
    """Register the trading-side timers on the technical_scheduler's schedule.

    Args:
        schedule: The shared `schedule` library instance owned by scheduler.py
            (the technical_scheduler).

    Returns:
        None. Registers two daily jobs (mutates `schedule`).

    Non-obvious logic:
        - 09:25 emit runs just before the 09:15 open settles (rebalance-gated:
          emit_signal defaults force=False -> only acts on the rotation day).
        - 09:30 execute runs 5 minutes later so the signal file is ready;
          it no-ops on days where the gate produced no signal.
    """
    schedule.every().day.at("09:25").do(emit_signal)   # rebalance-gated signal emit
    # 09:31 (staggered off n100's 09:30) — all 4 models share one scheduler
    # thread + one Fyers account; distinct minutes stop later models filling late.
    schedule.every().day.at("09:31").do(execute_orders)
    # Mid-month jobs are NOT registered (2026-05-31): the mid-month config lost to
    # RET1/monthly on the fixed-anchor re-check, so pseudo runs monthly-only. The
    # emit_mid_month_signal / execute_mid_month_orders funcs remain as an opt-in
    # path if a future fixed-anchor sweep revives mid-month.
