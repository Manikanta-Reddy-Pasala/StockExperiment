"""Paper trading executor — consumes signals, simulates portfolio.

Reads signals JSON (from signal_generator). Applies risk_manager.
Tracks open positions in a Postgres table `paper_portfolio` (auto-created).
Logs all decisions to stdout + appends to daily ledger.

Pure Python. No real orders.

Usage:
  python tools/live/paper_executor.py --signals signals/2026-05-12_ema_200_400_nifty50.json
  python tools/live/paper_executor.py --signals signals/2026-05-12_ema_200_400_nifty50.json --replay  # backtest mode
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

from tools.live.risk_manager import RiskManager, RiskConfig, Position  # noqa: E402

log = logging.getLogger("paper_executor")


def load_signals(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def load_open_positions(ledger_path: Path) -> List[Position]:
    """Read existing open positions from ledger JSON. Empty if none."""
    if not ledger_path.exists():
        return []
    try:
        with open(ledger_path) as f:
            data = json.load(f)
        return [Position(**p) for p in data.get("open", [])]
    except Exception as e:
        log.warning(f"load_positions fail: {e}")
        return []


def save_state(ledger_path: Path, open_positions: List[Position],
                closed_today: List[Dict], day_pnl: float) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now().isoformat(),
        "day_pnl": day_pnl,
        "open": [asdict(p) for p in open_positions],
        "closed_today": closed_today,
    }
    with open(ledger_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def execute(signals: List[Dict], rm: RiskManager, ledger_path: Path) -> Dict:
    open_positions = load_open_positions(ledger_path)
    closed: List[Dict] = []
    day_pnl = 0.0
    placed = 0
    skipped = 0

    for sig in signals:
        sym = sig["symbol"]
        side = sig["side"]
        sig_type = sig["signal"]
        price = float(sig["price"])
        ts = sig["ts"]

        if sig_type in ("ENTRY1", "ENTRY2"):
            ok, reason = rm.can_enter(sym, price, side, open_positions, day_pnl)
            if not ok:
                log.info(f"SKIP ENTRY {sym} @ {price}: {reason}")
                skipped += 1
                continue
            qty = rm.size_position(price, open_positions)
            if qty < 1:
                log.info(f"SKIP ENTRY {sym}: qty<1 (slot too small for price)")
                skipped += 1
                continue
            pos = Position(symbol=sym, qty=qty, entry_price=price, side=side,
                           sl=float(sig.get("sl", 0)),
                           target=float(sig.get("target", 0)))
            open_positions.append(pos)
            placed += 1
            log.info(f"PAPER ENTRY {side} {sym} qty={qty} @ {price} "
                     f"sl={pos.sl} target={pos.target}")

        elif sig_type in ("PARTIAL", "TARGET_HIT", "STOP_HIT"):
            # Close (partial or full) earliest open same-symbol position
            for i, p in enumerate(open_positions):
                if p.symbol != sym: continue
                book_qty = p.qty if sig_type != "PARTIAL" else p.qty // 2
                if book_qty < 1: continue
                pnl = (price - p.entry_price) * book_qty if side == "BUY" \
                      else (p.entry_price - price) * book_qty
                day_pnl += pnl
                closed.append({
                    "symbol": sym, "side": side,
                    "entry_price": p.entry_price, "exit_price": price,
                    "qty_closed": book_qty, "pnl": pnl,
                    "reason": sig_type, "ts": ts,
                })
                log.info(f"PAPER {sig_type} {sym} qty={book_qty} "
                         f"@ {price} (entry {p.entry_price}) pnl=₹{pnl:,.0f}")
                if sig_type == "PARTIAL":
                    p.qty -= book_qty
                    p.sl = p.entry_price  # trail SL to entry
                else:
                    open_positions.pop(i)
                break

    save_state(ledger_path, open_positions, closed, day_pnl)
    return {
        "placed": placed, "skipped": skipped, "closed": len(closed),
        "day_pnl": day_pnl, "open_at_end": len(open_positions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="JSON file from signal_generator")
    ap.add_argument("--ledger", default=None,
                    help="Paper-portfolio ledger JSON path. "
                         "Default: paper_portfolio/{date}.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rm = RiskManager.from_env()
    ledger_path = Path(args.ledger) if args.ledger else (
        ROOT / "paper_portfolio" / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    )

    signals = load_signals(args.signals)
    log.info(f"Loaded {len(signals)} signals from {args.signals}")
    log.info(f"Risk: capital=₹{rm.cfg.capital_inr:,} max_concurrent={rm.cfg.max_concurrent}")
    log.info(f"Ledger: {ledger_path}")

    result = execute(signals, rm, ledger_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
