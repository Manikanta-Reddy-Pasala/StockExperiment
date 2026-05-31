"""Cron registration for emerging_momentum — SINGLE-POSITION rotation.

Mirrors tools/models/momentum_n100_top5_max1/cron.py: scheduler glue only, owns
no strategy logic; shells out to the model's files.

  Data:    20:45 daily  pull_daily_ohlcv (data_pull.py) refresh N500 panel.
  Trading: 09:29 daily  emit_signal           -> live_signal.py (rebalance-gated)
           09:37 daily  execute_orders        -> fyers_executor.py (SINGLE)
           09:31 daily  emit_mid_month_signal  -> live_signal.py --mid-month-check
           09:39 daily  execute_mid_month_orders -> fyers_executor.py (SINGLE)

Distinct from the other models' times (09:25-09:35 + 20:30/20:42/20:43) so it
doesn't collide. live_signal.py self-gates (weekend / enabled / monthly /
mid-month) and writes an empty file when there's nothing to do, so the trading
jobs can safely fire every weekday. backtest.py is offline and NOT scheduled.

The EXECUTION half uses the SINGLE-position executor tools/live/fyers_executor.py
with --model-name emerging_momentum (single-position model_ledger.open_symbol),
NOT fyers_executor_multi — this model is max_concurrent=1.
"""
from __future__ import annotations

import json as _json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.emerging_momentum.data_pull import pull_daily_ohlcv  # noqa: E402

log = logging.getLogger(__name__)
MODEL_NAME = "emerging_momentum"
STATE_DIR = Path("/app/logs/emerging_momentum")


def _alert(msg: str):
    """Best-effort Telegram alert for scheduler failures (never raises)."""
    try:
        from tools.live.telegram_notify import send as _tg
        _tg(msg)
    except Exception as e:
        log.debug(f"tg alert skipped: {e}")


# ---- Trading-side jobs (technical_scheduler) ----

def _signals_path(today: str, mid: bool = False) -> Path:
    name = f"{today}_emerging" + ("_midmonth" if mid else "") + ".json"
    return STATE_DIR / "signals" / name


def emit_signal(force: bool = False):
    """Emit the monthly signal. Rebalance-gated unless force=True.

    Shells out to live_signal.py with --rebalance-only so the signal file is
    only populated on the monthly rebalance trigger; live_signal decides whether
    today qualifies and writes an empty file otherwise. SIGNAL half of the
    monthly trading pair (execute_orders is the EXECUTION half).
    """
    label = "emerging_momentum signal" + (" (FORCE)" if force else " (rebalance-gated)")
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {label}")
    log.info("=" * 80)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = _signals_path(today)
    signals_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "tools/models/emerging_momentum/live_signal.py",
        "--signals-out", str(signals_out),
    ]
    cmd.append("--force" if force else "--rebalance-only")
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=600, env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info(f"✅ emerging_momentum signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ emerging_momentum signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ emerging_momentum emit_signal failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ emerging_momentum signal error: {e}")
        _alert(f"❌ emerging_momentum emit_signal error: {e}")


def execute_orders():
    """Place Fyers orders from today's monthly signal file via the SINGLE
    executor. max_concurrent=1, so a duplicate ENTRY1 for an already-held symbol
    is a no-op. On non-rebalance days the file is absent/empty so this skips."""
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = _signals_path(today)
    if not signals_file.exists():
        log.info(f"emerging_momentum execute: no signal at {signals_file}, skipping.")
        return
    log.info("PLACING emerging_momentum FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", MODEL_NAME]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
        if r.returncode == 0:
            log.info("✅ emerging_momentum Fyers execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ emerging_momentum Fyers execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ emerging_momentum execute failed (rc={r.returncode}) "
                   f"— orders may not be placed")
    except Exception as e:
        log.error(f"❌ emerging_momentum Fyers execute error: {e}")
        _alert(f"❌ emerging_momentum execute error: {e}")


def emit_mid_month_signal():
    """Day-15 weekday mid-month check. live_signal applies the 5pp lead gate;
    on non-day-15 it writes an empty file, so cron can fire daily and the model
    self-skips. Writes a SEPARATE *_midmonth.json so it never collides with the
    monthly signal."""
    log.info("\n" + "=" * 80)
    log.info("RUNNING emerging_momentum mid-month check")
    log.info("=" * 80)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = _signals_path(today, mid=True)
    signals_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "tools/models/emerging_momentum/live_signal.py",
        "--signals-out", str(signals_out),
        "--mid-month-check",
    ]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=600, env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info("✅ emerging_momentum mid-month check complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ emerging_momentum mid-month check failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ emerging_momentum mid-month emit failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ emerging_momentum mid-month error: {e}")
        _alert(f"❌ emerging_momentum mid-month emit error: {e}")


def execute_mid_month_orders():
    """Execute the mid-month signal file via the SINGLE executor (separate file
    from monthly to avoid a double-execution race). Usually a no-op (the gate
    suppressed rotation -> empty list)."""
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = _signals_path(today, mid=True)
    if not signals_file.exists():
        log.info(f"emerging_momentum mid-month execute: no signal at {signals_file}, skipping.")
        return
    try:
        sigs = _json.loads(signals_file.read_text())
        if not sigs:
            log.info("emerging_momentum mid-month: no signals to execute.")
            return
    except Exception:
        pass
    log.info("PLACING emerging_momentum MID-MONTH FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id,
           "--model-name", MODEL_NAME]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
        if r.returncode == 0:
            log.info("✅ emerging_momentum mid-month execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ emerging_momentum mid-month execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ emerging_momentum mid-month execute failed (rc={r.returncode}) "
                   f"— orders may not be placed")
    except Exception as e:
        log.error(f"❌ emerging_momentum mid-month execute error: {e}")
        _alert(f"❌ emerging_momentum mid-month execute error: {e}")


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Daily N500 OHLCV pull. Called by data_scheduler."""
    schedule.every().day.at("20:45").do(pull_daily_ohlcv)


def register_trading_jobs(schedule):
    """Signal + execute (monthly + mid-month). Called by technical_scheduler.

    Distinct times from the other models; live_signal self-gates so daily firing
    is safe.
    """
    schedule.every().day.at("09:29").do(emit_signal)          # rebalance-gated
    schedule.every().day.at("09:37").do(execute_orders)
    # Mid-month rank check (day-15 weekday). live_signal self-skips otherwise.
    schedule.every().day.at("09:31").do(emit_mid_month_signal)
    schedule.every().day.at("09:39").do(execute_mid_month_orders)
