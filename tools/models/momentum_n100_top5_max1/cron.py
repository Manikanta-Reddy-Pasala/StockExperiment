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


# ---- Trading-side jobs (technical_scheduler) ----

def emit_signal(force: bool = False):
    """Emit Model 3 signal. Rebalance-gated unless force=True."""
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
    cmd.append("--force" if force else "--rebalance-only")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            log.info(f"✅ Model 3 signal -> {signals_out}")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ Model 3 signal failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
    except Exception as e:
        log.error(f"❌ Model 3 signal error: {e}")


def execute_orders():
    """Place Fyers orders from today's signal file. Gated by LIVE_TRADING=true."""
    if os.environ.get("LIVE_TRADING", "false").lower() != "true":
        log.info("Model 3 execute: LIVE_TRADING != 'true', skipping.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    signals_file = Path(f"/app/logs/momrot/signals/{today}_momrot_n100.json")
    if not signals_file.exists():
        log.info(f"Model 3 execute: no signal at {signals_file}, skipping.")
        return
    log.info("PLACING MODEL 3 FYERS ORDERS")
    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor.py",
           "--signals", str(signals_file), "--user-id", user_id]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log.info("✅ Model 3 Fyers execute complete")
            if r.stdout:
                log.info(r.stdout[-500:])
        else:
            log.error(f"❌ Model 3 Fyers execute failed ({r.returncode})")
            if r.stderr:
                log.error(r.stderr[-500:])
    except Exception as e:
        log.error(f"❌ Model 3 Fyers execute error: {e}")


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """Daily/monthly data pulls. Called by data_scheduler."""
    # Equity OHLCV — daily after market close (saga step 3 also covers this;
    # this is the model-explicit fallback)
    schedule.every().day.at("20:30").do(pull_daily_ohlcv)
    # Universe refresh — first of every month (build_universe.py is idempotent)
    schedule.every().day.at("06:30").do(_monthly_universe)


def _monthly_universe():
    """Wrapper that only runs universe refresh on 1st of month."""
    if datetime.now().day == 1:
        refresh_universe()


def register_trading_jobs(schedule):
    """Signal + execute. Called by technical_scheduler."""
    schedule.every().day.at("09:25").do(emit_signal)        # rebalance-gated
    schedule.every().day.at("09:30").do(execute_orders)     # LIVE_TRADING gated
