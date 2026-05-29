"""Cron registration for regime_momentum_n500 (multi-holding, K=5).

Scheduler glue — owns no strategy logic; shells out to the model's files:
  Data:    20:43 daily  pull_daily_ohlcv (data_pull.py) refresh N500 panel.
  Trading: 09:27 daily  emit_signal  -> live_signal.py writes signals file.
           09:35 daily  execute_orders -> fyers_executor_multi.py places orders.

Times are offset from momentum_retest_n500 (09:26/09:34/20:42) so the two
multi-holding models don't fire simultaneously. The strategy is monthly-rebalance
+ daily-regime-risk, so the trading jobs run EVERY weekday and live_signal.py
handles the monthly/regime/enabled/weekend gating internally (empty file when
there's nothing to do). backtest.py is offline and NOT scheduled.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from tools.models.regime_momentum_n500.data_pull import pull_daily_ohlcv  # noqa

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "regime_momentum_n500"
STATE_DIR = Path("/app/logs/regime_momentum_n500")
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
    return _run(["python3", "tools/models/regime_momentum_n500/live_signal.py",
                 "--signals-out", str(SIGNALS)], "regime_momentum_n500 emit_signal")


def execute_orders():
    if not SIGNALS.exists():
        log.warning("regime_momentum_n500: no signals file — skip execute"); return False
    return _run(["python3", "tools/live/fyers_executor_multi.py",
                 "--signals", str(SIGNALS), "--user-id", "1"],
                "regime_momentum_n500 execute_orders")


def register_data_jobs(schedule):
    """Register data-side jobs on the shared `schedule` library instance."""
    schedule.every().day.at("20:43").do(pull_daily_ohlcv)


def register_trading_jobs(schedule):
    """Register trading-side jobs on the shared `schedule` library instance.

    09:27 emit + 09:35 execute, every weekday; weekend/enabled/monthly/regime
    gating happens inside live_signal.py.
    """
    schedule.every().day.at("09:27").do(emit_signal)
    schedule.every().day.at("09:35").do(execute_orders)
