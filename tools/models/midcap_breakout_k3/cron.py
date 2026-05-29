"""Cron registration for midcap_breakout_k3 (multi-holding, K=3, daily breakout).

  Data:    20:44 daily  pull_daily_ohlcv (data_pull.py)
  Trading: 09:28 daily  emit_signal  -> live_signal.py
           09:36 daily  execute_orders -> fyers_executor_multi.py
Times offset from the other multi models. live_signal.py gates on enabled/weekend
internally and runs EVERY weekday (event-driven daily breakout scan).
"""
from __future__ import annotations
import logging, subprocess
from pathlib import Path
from tools.models.midcap_breakout_k3.data_pull import pull_daily_ohlcv  # noqa

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path("/app/logs/midcap_breakout_k3")
SIGNALS = STATE_DIR / "signals" / "latest.json"


def _run(cmd, label, timeout=900):
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        (log.info if r.returncode == 0 else log.error)(
            f"  {'✅' if r.returncode==0 else '❌'} {label}"
            + ("" if r.returncode == 0 else f": {r.stderr[-400:]}"))
        return r.returncode == 0
    except Exception as e:
        log.error(f"  ❌ {label}: {e}"); return False


def emit_signal():
    SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    return _run(["python3", "tools/models/midcap_breakout_k3/live_signal.py",
                 "--signals-out", str(SIGNALS)], "midcap_breakout_k3 emit_signal")


def execute_orders():
    if not SIGNALS.exists():
        log.warning("midcap_breakout_k3: no signals file — skip execute"); return False
    return _run(["python3", "tools/live/fyers_executor_multi.py",
                 "--signals", str(SIGNALS), "--user-id", "1"],
                "midcap_breakout_k3 execute_orders")


def register_data_jobs(schedule):
    schedule.every().day.at("20:44").do(pull_daily_ohlcv)


def register_trading_jobs(schedule):
    schedule.every().day.at("09:28").do(emit_signal)
    schedule.every().day.at("09:36").do(execute_orders)
