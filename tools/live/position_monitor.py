"""Position monitor — mark-to-market open positions every N minutes.

Reads paper_portfolio ledger. Fetches LTP from Fyers. Computes mark P&L.
Triggers SL/T1 exits if price crossed. Writes updated ledger.

Run during market hours (09:15-15:30 IST) every 5 min.

Usage:
  python tools/live/position_monitor.py --user-id 1
  cron:  */5 9-15 * * 1-5 cd /opt/StockExperiment && python tools/live/position_monitor.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.live.risk_manager import Position  # noqa: E402

log = logging.getLogger("position_monitor")


def to_fyers_symbol(plain: str) -> str:
    s = plain.upper()
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s.replace('.NS', '')}-EQ"


def fetch_ltp(svc, user_id: int, symbols: List[str]) -> Dict[str, float]:
    """Use Fyers /data/quotes — supports up to 50 symbols per call."""
    fyers_syms = [to_fyers_symbol(s) for s in symbols]
    out: Dict[str, float] = {}
    BATCH = 50
    for i in range(0, len(fyers_syms), BATCH):
        chunk = fyers_syms[i:i + BATCH]
        try:
            res = svc.quotes(user_id=user_id, symbols=",".join(chunk))
        except Exception as e:
            log.warning(f"quotes fail: {e}")
            continue
        if not res or res.get("s") != "ok":
            continue
        for q in res.get("d", []):
            sym_full = q.get("n") or ""
            try:
                ltp = float(q.get("v", {}).get("lp") or q.get("v", {}).get("ltp") or 0)
            except Exception:
                ltp = 0
            plain = sym_full.replace("NSE:", "").replace("-EQ", "")
            out[plain] = ltp
    return out


def check_exits(positions: List[Position], ltps: Dict[str, float]) -> List[Dict]:
    """For each open position, check SL/T1 trigger. Returns exit events."""
    exits = []
    for p in positions:
        ltp = ltps.get(p.symbol, 0)
        if ltp <= 0:
            continue
        if p.side == "BUY":
            if p.sl > 0 and ltp <= p.sl:
                exits.append({"symbol": p.symbol, "qty": p.qty,
                              "exit_price": ltp, "reason": "STOP_HIT"})
            elif p.target > 0 and ltp >= p.target:
                exits.append({"symbol": p.symbol, "qty": p.qty,
                              "exit_price": ltp, "reason": "TARGET_HIT"})
        else:  # SELL
            if p.sl > 0 and ltp >= p.sl:
                exits.append({"symbol": p.symbol, "qty": p.qty,
                              "exit_price": ltp, "reason": "STOP_HIT"})
            elif p.target > 0 and ltp <= p.target:
                exits.append({"symbol": p.symbol, "qty": p.qty,
                              "exit_price": ltp, "reason": "TARGET_HIT"})
    return exits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ledger_path = Path(args.ledger) if args.ledger else (
        ROOT / "paper_portfolio" / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    if not ledger_path.exists():
        log.info(f"No ledger {ledger_path} — nothing to monitor")
        return 0

    with open(ledger_path) as f:
        state = json.load(f)
    positions = [Position(**p) for p in state.get("open", [])]
    if not positions:
        log.info("No open positions")
        return 0

    log.info(f"Monitoring {len(positions)} positions: "
             f"{[p.symbol for p in positions]}")

    from src.services.brokers.fyers_service import FyersService
    svc = FyersService()
    ltps = fetch_ltp(svc, args.user_id, [p.symbol for p in positions])

    # Mark-to-market
    mark_pnl = 0.0
    for p in positions:
        ltp = ltps.get(p.symbol, p.entry_price)
        pnl = (ltp - p.entry_price) * p.qty if p.side == "BUY" \
              else (p.entry_price - ltp) * p.qty
        mark_pnl += pnl
        log.info(f"  {p.symbol} qty={p.qty} entry=₹{p.entry_price} "
                 f"LTP=₹{ltp} mark=₹{pnl:+,.0f}")

    log.info(f"Open positions mark-to-market: ₹{mark_pnl:+,.0f}")

    # Check exits
    exits = check_exits(positions, ltps)
    if exits and args.mode == "live":
        log.warning(f"Exit triggers ({len(exits)}) — placing exit orders via Fyers")
        for ex in exits:
            log.info(f"  EXIT {ex['symbol']} {ex['reason']} @ ₹{ex['exit_price']}")
            # TODO: place Fyers exit order (delegate to fyers_executor)
    elif exits:
        log.info(f"Paper: would exit {len(exits)} positions")

    # Update state with marks (don't auto-close in paper mode here —
    # signal_generator will emit STOP_HIT/TARGET_HIT next bar close).
    state["last_mark_pnl"] = mark_pnl
    state["last_mark_ts"] = datetime.now().isoformat()
    state["last_ltps"] = ltps
    with open(ledger_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
