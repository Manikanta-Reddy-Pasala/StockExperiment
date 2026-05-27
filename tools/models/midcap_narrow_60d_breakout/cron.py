"""Cron registration for midcap_narrow_60d_breakout — STEP 4 of the flow.

Pipeline position (data_pull -> build_universe -> live_signal -> cron -> backtest):
  This module is the SCHEDULER glue. It registers two families of jobs on the
  shared `schedule` instance so the rest of the pipeline runs automatically:
    - data jobs wrap data_pull.py (OHLCV refresh + monthly universe rebuild)
    - trading jobs run live_signal.py (the breakout scan) then the Fyers
      executor against the signals file it produced.
  backtest.py is offline/manual and is intentionally NOT scheduled here.

Data side:
  register_data_jobs(schedule) — daily N500 OHLCV + monthly universe refresh.
Trading side:
  register_trading_jobs(schedule) — daily live_signal + Fyers execute (always live).
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from tools.models.midcap_narrow_60d_breakout.data_pull import (
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


MODEL_NAME = "midcap_narrow_60d_breakout"
UNIVERSE_FILE = "/app/logs/momrot/universes/midcap_narrow.json"
SIGNALS_DIR = Path("/app/logs/midcap_narrow/signals")


# ---- Trading-side jobs ----

def emit_signal():
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {MODEL_NAME} live signal")
    log.info("=" * 80)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = SIGNALS_DIR / f"{today}_midcap_narrow.json"
    cmd = [
        "python3",
        "tools/models/midcap_narrow_60d_breakout/live_signal.py",
        "--universe-file", UNIVERSE_FILE,
        "--signals-out", str(signals_out),
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
            _alert(f"❌ {MODEL_NAME} emit_signal failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} signal error: {e}")
        _alert(f"❌ {MODEL_NAME} emit_signal error: {e}")


def emit_signal_eod():
    """15:25 IST end-of-day breakout/exit scan — DISPLAY/AUDIT ONLY.

    Mirrors the n100 mid-month separate-file pattern. Writes to a SEPARATE
    `{today}_midcap_narrow_eod.json` file that execute_orders() never reads,
    so this run can never clobber the morning's audited canonical signal.

    Why a separate file: execute_orders() runs ~09:32 against the 09:25
    canonical file. The original 15:25 emit reused that SAME path, recomputing
    against the now-held position and overwriting the morning signal with an
    EXIT that nothing executes — corrupting the audit trail. Routing the EOD
    scan to its own file keeps the canonical morning signal intact while still
    surfacing end-of-day SMA-break exits in the PWA/Telegram feed.
    """
    log.info("\n" + "=" * 80)
    log.info(f"RUNNING {MODEL_NAME} EOD signal (display/audit only)")
    log.info("=" * 80)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    # Separate file — NEVER read by execute_orders (canonical is the 09:25 file).
    signals_out = SIGNALS_DIR / f"{today}_midcap_narrow_eod.json"
    cmd = [
        "python3",
        "tools/models/midcap_narrow_60d_breakout/live_signal.py",
        "--universe-file", UNIVERSE_FILE,
        "--signals-out", str(signals_out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MOMROT_TG_NOTIFY": "1"})
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} EOD signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ {MODEL_NAME} EOD signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
            _alert(f"❌ {MODEL_NAME} emit_signal_eod failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} EOD signal error: {e}")
        _alert(f"❌ {MODEL_NAME} emit_signal_eod error: {e}")


def execute_orders():
    """Place Fyers orders from today's signal file."""
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}_midcap_narrow.json"
    if not signals_file.exists():
        log.info(f"{MODEL_NAME} execute: no signal at {signals_file}, skipping.")
        return
    log.info(f"PLACING {MODEL_NAME} FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = [
        "python3", "tools/live/fyers_executor.py",
        "--signals", str(signals_file),
        "--user-id", user_id,
        "--model-name", MODEL_NAME,
    ]
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


# ---- Data-side jobs ----

def _monthly_universe():
    if datetime.now().day == 1:
        refresh_universe()


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Daily OHLCV + monthly universe refresh."""
    # N500 OHLCV — covers midcap_narrow (subset of N500)
    schedule.every().day.at("20:45").do(pull_daily_ohlcv)
    # Universe refresh — 1st of month only (no-op other days)
    schedule.every().day.at("06:35").do(_monthly_universe)


def register_trading_jobs(schedule):
    """Daily signal + Fyers execute (event-driven, daily check)."""
    # Signal scan — runs daily after market open to detect breakouts + exits
    schedule.every().day.at("09:25").do(emit_signal)
    # Execute orders
    schedule.every().day.at("09:32").do(execute_orders)
    # Also check exit conditions at market close (catches end-of-day SMA breaks).
    # DISPLAY/AUDIT ONLY — writes a SEPARATE *_eod.json so it never clobbers the
    # morning canonical signal that execute_orders reads.
    schedule.every().day.at("15:25").do(emit_signal_eod)
