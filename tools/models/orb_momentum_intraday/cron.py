"""Cron wiring for orb_momentum_intraday (the only INTRADAY model).

Schedule (IST):
  every 5 min 09:30 → 15:10  scan + execute
      -> live_signal.emit_signals does BOTH, mirroring the backtest orb_trade:
         (a) ENTRY (before the 10:00 cutoff): BUY leaders that broke above ORH,
             not already held, sized to one invested/SELECT_TOP slot.
         (b) EXIT (all session): SELL a held name the moment it hits its STOP
             (ORL) or TARGET (ORH+2×width) — strategy.live_exit_reason, same
             rule as the backtest. (Without all-session scans live could not see
             a midday stop/target and would ride to EOD, breaking parity.)
         fyers_executor_multi places them (INTRADAY/MIS). Each scan re-checks
         holdings so a name is never double-bought; entries stop after 10:00.
  15:15  SQUARE-OFF: emit SELLS for every still-held name -> executor flattens.
         (MIS broker auto-square-off is the backstop; this is the explicit exit.)
         15:15 matches the backtest EOD exit so live==backtest (no CAGR drift).

INTRADAY → flat by 15:15, ZERO overnight risk. Multi-holding (up to SELECT_TOP).
live_signal self-gates on time, so firing the same job daily is safe.
"""
from __future__ import annotations

import json as _json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "orb_momentum_intraday"
STATE_DIR = Path("/app/logs/orb_momentum_intraday")


def _alert(msg: str):
    try:
        from tools.live.telegram_notify import send
        send(msg)
    except Exception:
        pass


def _signals_path(today: str) -> Path:
    return STATE_DIR / "signals" / f"{today}_orb.json"


def scan_and_execute():
    """Emit the current intraday signal set then place it (single combined step
    so the executor always acts on THIS scan's fresh buys/sells)."""
    today = datetime.now().strftime("%Y-%m-%d")
    out = _signals_path(today)
    out.parent.mkdir(parents=True, exist_ok=True)
    emit = ["python3", "tools/models/orb_momentum_intraday/live_signal.py",
            "--signals-out", str(out)]
    try:
        r = subprocess.run(emit, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            log.error(f"❌ orb emit failed: {r.stderr[-300:]}")
            _alert(f"❌ orb emit failed (rc={r.returncode})"); return
        log.info(f"orb emit: {r.stdout.strip()[-200:]}")
    except Exception as e:
        log.error(f"❌ orb emit error: {e}"); _alert(f"❌ orb emit error: {e}"); return

    # Skip the executor entirely if there's nothing to do this scan.
    try:
        sig = _json.loads(out.read_text())
        if not sig.get("buys") and not sig.get("sells"):
            return
    except Exception:
        return

    user_id = os.environ.get("USER_ID", "1")
    cmd = ["python3", "tools/live/fyers_executor_multi.py",
           "--signals", str(out), "--user-id", user_id]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        (log.info if r.returncode == 0 else log.error)(
            f"{'✅' if r.returncode==0 else '❌'} orb execute"
            + ("" if r.returncode == 0 else f" rc={r.returncode}: {r.stderr[-300:]}"))
        if r.stdout:
            log.info(r.stdout[-500:])
        if r.returncode != 0:
            _alert(f"❌ orb execute failed (rc={r.returncode})")
    except Exception as e:
        log.error(f"❌ orb execute error: {e}"); _alert(f"❌ orb execute error: {e}")


def square_off():
    """15:15 forced flatten — same path; live_signal emits SELLS for all held.
    15:15 matches the backtest EOD exit (EOD_FLAT_MIN) so live and backtest
    flatten at the same bar (CAGR parity)."""
    log.info("orb: 15:15 SQUARE-OFF")
    scan_and_execute()


# ---- Registration entrypoints ----

def register_data_jobs(schedule):
    """ORB selection uses the daily N500 close already pulled by the other
    models' 20:xx jobs; intraday 5-min bars are fetched live per scan. No
    dedicated data job needed."""
    return


def register_trading_jobs(schedule):
    """Scan every 5 min 09:30→15:10, then square off at 15:15.

    The 5-min cadence mirrors the 5-min bar and lets live_signal check BOTH:
      - ENTRY breakouts (only fire before the 10:00 cutoff; emit self-gates), and
      - intraday STOP/TARGET exits on held names (orb_trade exits 41% of trades
        this way — backtest parity needs live to check all session, not just AM).
    live_signal self-gates on time, so firing the same job every 5 min is safe
    (no double-buy: held names are skipped; nothing to do = no executor call)."""
    h, m = 9, 30
    while (h, m) <= (15, 10):
        schedule.every().day.at(f"{h:02d}:{m:02d}").do(scan_and_execute)
        m += 5
        if m >= 60:
            h, m = h + 1, m - 60
    schedule.every().day.at("15:15").do(square_off)
    # Safety re-attempt in case the 15:15 flatten partially filled.
    schedule.every().day.at("15:18").do(square_off)
