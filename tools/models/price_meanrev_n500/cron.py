"""Cron registration for price_meanrev_n500 (limit-order dip-buy, K=3, PAPER).

Scheduler glue — owns no strategy logic; shells out to the model's files:
  Trading: 08:55 daily  emit_signal -> live_signal.py settles yesterday's paper
           orders against the latest bar, then emits today's limit levels.

NO execute_orders job is registered — DELIBERATE. The model's edge lives in
limit fills at the dip level (close-fill drops CAGR 102.8% -> 36.1%, see
strategy.py); the shared executor places market orders, so real execution is
not wired. live_signal.py keeps its own paper ledger with backtest-exact
limit-fill semantics instead. Daily OHLCV data comes from the existing nightly
pulls (momentum_retest_n500's 20:42 job refreshes the same N500 panel) — no
separate data job needed.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "price_meanrev_n500"
STATE_DIR = Path("/app/logs/price_meanrev_n500")
SIGNALS = STATE_DIR / "signals" / "latest.json"


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
                 "--signals-out", str(SIGNALS)], "price_meanrev_n500 emit_signal")


def register_data_jobs(schedule):
    """No data jobs — shares the nightly N500 OHLCV pull with retest."""


def register_trading_jobs(schedule):
    """08:55 emit (pre-open), every day; weekend/enabled gating inside live_signal."""
    schedule.every().day.at("08:55").do(emit_signal)
