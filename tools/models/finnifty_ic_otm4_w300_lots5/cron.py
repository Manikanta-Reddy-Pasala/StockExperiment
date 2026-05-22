"""Cron registration for finnifty_ic_otm2_w150_lots5 — monthly Iron Condor.

Promoted 2026-05-22 to safe-tight params (OTM 2% / wing 150 / 5 lots).
Folder name kept stable for cron stability; runtime identifier below.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.data_pull import (  # noqa: E402
    fetch_index_spots, fetch_option_bhav,
)

log = logging.getLogger(__name__)

MODEL_NAME = "finnifty_ic_otm2_w150_lots5"
# Folder kept as legacy name for cron-path stability; runtime model id above.
MODEL_FOLDER = Path(__file__).parent.name
SIGNALS_DIR = Path(f"/app/logs/{MODEL_NAME}/signals")


def register_data_jobs(schedule):
    """Daily data pulls. Called by data_scheduler."""
    schedule.every().day.at("18:00").do(fetch_index_spots)
    schedule.every().day.at("18:30").do(fetch_option_bhav)


def emit_signal():
    log.info("=" * 60)
    log.info(f"RUNNING {MODEL_NAME} live signal")
    log.info("=" * 60)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    signals_out = SIGNALS_DIR / f"{today}.json"
    cmd = [
        "python3",
        f"tools/models/{MODEL_FOLDER}/live_signal.py",
        "--signals-out", str(signals_out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-400:])
            try:
                from tools.live.telegram_notify import notify_signals
                notify_signals(MODEL_NAME, str(signals_out))
            except Exception as _te:
                log.debug(f"TG notify failed: {_te}")
        else:
            log.error(f"❌ {MODEL_NAME} signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-400:])
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} signal error: {e}")


def execute_orders():
    """Place Fyers orders. Option multi-leg executor calls
    fyers_executor_options.py — per-model enable/disable via ModelSettings.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}.json"
    if not signals_file.exists():
        log.info(f"{MODEL_NAME} execute: no signal at {signals_file}, skipping.")
        return
    log.info(f"PLACING {MODEL_NAME} FYERS ORDERS (4-leg IC via options executor)")
    cmd = [
        "python3", "tools/live/fyers_executor_options.py",
        "--signals", str(signals_file),
        "--model-name", MODEL_NAME,
        "--product", "MARGIN",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} Fyers execute complete")
            if r.stdout:
                log.info(r.stdout[-600:])
        else:
            log.error(f"❌ {MODEL_NAME} Fyers execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-600:])
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} Fyers execute error: {e}")


def register_trading_jobs(schedule):
    schedule.every().day.at("09:25").do(emit_signal)
    schedule.every().day.at("14:30").do(emit_signal)
    schedule.every().day.at("09:32").do(execute_orders)
