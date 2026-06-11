"""Cron registration for price_meanrev_n500 (limit-order dip-buy, K=3).

Scheduler glue — owns no strategy logic; shells out to the model's files:
  08:55        emit_signal      live_signal.py settles the paper ledger against
                                the latest bar, then emits today's limit levels.
  09:16        place_orders     fyers_executor_limit --place: resting LIMIT BUY
                                day orders at the levels (post-open; below-market
                                limits rest until touched, expire at the close).
  every 5 min  exit_checks      fyers_executor_limit --exits: stop/target/time
  (09:30-15:10)                 square-off vs live LTP (stop first — backtest parity).
  15:20        reconcile_fills  fyers_executor_limit --reconcile: order-book poll,
                                record traded limits into model_holdings + freeze
                                stop/target meta.

⚠ EXECUTION MODEL: this model must NEVER route through fyers_executor_multi —
that path fills NOW at live LTP (market semantics), which collapses the edge
(close-fill = 36% vs 103% CAGR, see strategy.py). The limit executor preserves
the backtest's resting-limit mechanics with real broker orders. All real
placement is still gated by model_settings (enabled + signals_only=False);
while signals_only=True the executor logs and places NOTHING.

Daily OHLCV data comes from the existing nightly pulls (momentum_retest_n500's
20:42 job refreshes the same N500 panel) — no separate data job needed.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "price_meanrev_n500"
STATE_DIR = Path("/app/logs/price_meanrev_n500")
SIGNALS = STATE_DIR / "signals" / "latest.json"
EXECUTOR = "tools/live/fyers_executor_limit.py"


def _run(cmd, label, timeout=1200):
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
    return _run(["python3", "tools/models/price_meanrev_n500/live_signal.py",
                 "--signals-out", str(SIGNALS)], f"{MODEL_NAME} emit_signal")


def place_orders():
    return _run(["python3", EXECUTOR, "--mode", "place", "--user-id", "1"],
                f"{MODEL_NAME} place_orders")


def reconcile_fills():
    return _run(["python3", EXECUTOR, "--mode", "reconcile", "--user-id", "1"],
                f"{MODEL_NAME} reconcile_fills")


def exit_checks():
    """5-min intraday stop/target/time checks — self-gated to market hours so
    the every-5-minutes schedule never fires the executor off-session."""
    now = datetime.now()
    hm = now.hour * 60 + now.minute
    if now.weekday() >= 5 or not (9 * 60 + 30 <= hm <= 15 * 60 + 10):
        return True
    return _run(["python3", EXECUTOR, "--mode", "exits", "--user-id", "1"],
                f"{MODEL_NAME} exit_checks")


def register_data_jobs(schedule):
    """No data jobs — shares the nightly N500 OHLCV pull with retest."""


def register_trading_jobs(schedule):
    schedule.every().day.at("08:55").do(emit_signal)
    schedule.every().day.at("09:16").do(place_orders)
    schedule.every(5).minutes.do(exit_checks)        # self-gated to 09:30-15:10
    schedule.every().day.at("15:20").do(reconcile_fills)
