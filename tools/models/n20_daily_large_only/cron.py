"""Cron registration for n20_daily_large_only.

Daily rotation model — signal runs every weekday morning (no monthly gate).

Role in the model flow (data_pull -> live_signal -> cron -> backtest)
---------------------------------------------------------------------
This is the SCHEDULER glue that wires the model into the technical_scheduler
container. It owns no strategy logic; it just registers jobs that shell out
to the other files:

  Data side (register_data_jobs):
    - 20:40 daily: pull_daily_ohlcv (data_pull.py) refreshes the N500 panel.
    - 06:33 daily: _quarterly_universe -> refresh_universe only on the NSE
      Nifty-100 rebalance day (1st of Mar/Sep).

  Trading side (register_trading_jobs):
    - 09:25 daily: emit_signal -> live_signal.py writes today's signals file.
    - 09:30 daily: execute_orders -> fyers_executor.py places real orders.

Weekday/enabled gating lives inside live_signal.py, so the trading jobs are
registered every day and simply no-op on weekends / when disabled.
backtest.py is offline and is NOT scheduled here.
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from tools.models.n20_daily_large_only.data_pull import (  # noqa: E402
    pull_daily_ohlcv, refresh_universe,
)

log = logging.getLogger(__name__)

MODEL_NAME = "n20_daily_large_only"
SIGNALS_DIR = Path("/app/logs/n20_daily/signals")


# ---- Trading-side jobs ----

def emit_signal():
    """Emit daily rotation signal. Weekday-gated inside live_signal.py.

    Shells out to live_signal.py with --signals-out pointed at today's file
    under SIGNALS_DIR and --top-n 1, capturing stdout/stderr for the log.
    Failures are logged (last 500 chars of stderr) but not raised, so a bad
    run can't crash the scheduler loop.
    """
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {MODEL_NAME} live signal")
    log.info("=" * 80)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = SIGNALS_DIR / f"{today}_n20.json"
    cmd = [
        "python3", "tools/models/n20_daily_large_only/live_signal.py",
        "--signals-out", str(signals_out), "--top-n", "1",
    ]
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
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} signal error: {e}")


def execute_orders():
    """Place Fyers orders from today's signal file.

    Looks for today's signals JSON under SIGNALS_DIR; if missing (no signal
    run yet), logs and returns. Otherwise shells out to fyers_executor.py with
    the signals file, USER_ID (default "1"), and this model's name. Errors are
    logged but not raised.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}_n20.json"
    # No signal file means emit_signal hasn't run (or skipped) today — nothing
    # to execute, so bail out quietly.
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
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} Fyers execute error: {e}")


# ---- Data-side helpers ----

def _quarterly_universe():
    """Refresh Nifty 100 CSV on first weekday of Mar/Sep (NSE rebalance).

    Registered to run daily but self-gates: only triggers refresh_universe on
    the 1st of March or September when that day is a weekday — proxying NSE's
    semi-annual Nifty-100 index reconstitution. A no-op on every other day.
    """
    today = datetime.now()
    # Only act on the NSE rebalance month-start (Mar 1 / Sep 1) on a weekday.
    if today.month in (3, 9) and today.day == 1 and today.weekday() < 5:
        refresh_universe()


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Register data-side jobs on the shared scheduler.

    Args:
        schedule: the `schedule` library instance the scheduler owns.

    Registers the 20:40 daily N500 OHLCV pull (run after the n100 pull so the
    panel is complete) and the 06:33 daily universe-refresh check that only
    fires on the quarterly NSE rebalance day.
    """
    schedule.every().day.at("20:40").do(pull_daily_ohlcv)
    schedule.every().day.at("06:33").do(_quarterly_universe)


def register_trading_jobs(schedule):
    """Register trading-side jobs on the shared scheduler.

    Args:
        schedule: the `schedule` library instance the scheduler owns.

    Registers the 09:25 daily signal emit and the 09:30 daily order execute.
    Both run every calendar day; weekend/enabled gating happens downstream
    inside live_signal.py.
    """
    schedule.every().day.at("09:25").do(emit_signal)
    schedule.every().day.at("09:30").do(execute_orders)
