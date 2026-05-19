"""Cron registration for finnifty_ic_otm4_w300_lots5 — monthly Iron Condor."""
from __future__ import annotations

import logging
import os
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

MODEL_NAME = "finnifty_ic_otm4_w300_lots5"
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
        f"tools/models/{MODEL_NAME}/live_signal.py",
        "--signals-out", str(signals_out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info(f"✅ {MODEL_NAME} signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-400:])
        else:
            log.error(f"❌ {MODEL_NAME} signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-400:])
    except Exception as e:
        log.error(f"❌ {MODEL_NAME} signal error: {e}")


def execute_orders():
    """LIVE_TRADING-gated. Option multi-leg executor not yet wired.

    Uses the same LIVE_TRADING flag as equity models — options enable/disable
    is controlled per-model via ModelSettings.enabled, not a second env var.
    """
    if os.environ.get("LIVE_TRADING", "false").lower() != "true":
        log.info(f"{MODEL_NAME} execute: LIVE_TRADING != 'true', skipping.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = SIGNALS_DIR / f"{today}.json"
    if not signals_file.exists():
        log.info(f"{MODEL_NAME} execute: no signal at {signals_file}, skipping.")
        return
    log.warning(f"{MODEL_NAME}: option Fyers executor not yet implemented. "
                f"Signals at {signals_file} ready for manual review.")


def register_trading_jobs(schedule):
    schedule.every().day.at("09:25").do(emit_signal)
    schedule.every().day.at("14:30").do(emit_signal)
    schedule.every().day.at("09:32").do(execute_orders)
