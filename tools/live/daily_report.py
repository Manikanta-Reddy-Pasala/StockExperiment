"""Daily P&L report — read today's paper portfolio + emit summary.

Pure Python. Stdout-only. No LLM.

Usage:
  python tools/live/daily_report.py --date 2026-05-12
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--capital", type=int, default=1_000_000)
    args = ap.parse_args()

    ledger_path = Path(args.ledger) if args.ledger else (
        ROOT / "paper_portfolio" / f"{args.date}.json"
    )
    if not ledger_path.exists():
        print(f"No ledger at {ledger_path}")
        return 1

    with open(ledger_path) as f:
        s = json.load(f)

    closed = s.get("closed_today", [])
    open_pos = s.get("open", [])
    day_pnl = s.get("day_pnl", 0)
    mark_pnl = s.get("last_mark_pnl", 0)

    print(f"# Daily Report — {args.date}\n")
    print(f"## Realized P&L")
    print(f"- Trades closed: {len(closed)}")
    wins = [c for c in closed if c.get("pnl", 0) > 0]
    print(f"- Winners: {len(wins)} / {len(closed)} "
          f"({len(wins) * 100 // max(len(closed), 1)}%)")
    print(f"- Day realized P&L: ₹{day_pnl:+,.0f}")
    print(f"- Day ROI: {day_pnl / args.capital * 100:+.2f}%")
    print()

    if open_pos:
        print(f"## Open Positions ({len(open_pos)})")
        print("| Symbol | Side | Qty | Entry | SL | Target |")
        print("|--------|------|-----|-------|----|--------|")
        for p in open_pos:
            print(f"| {p['symbol']} | {p['side']} | {p['qty']} | "
                  f"₹{p['entry_price']} | ₹{p.get('sl', 0)} | "
                  f"₹{p.get('target', 0)} |")
        print()
        print(f"Mark-to-market unrealized: ₹{mark_pnl:+,.0f} "
              f"({mark_pnl / args.capital * 100:+.2f}%)")

    if closed:
        print(f"\n## Closed Trades")
        print("| Symbol | Side | Qty | Entry | Exit | Reason | P&L |")
        print("|--------|------|-----|-------|------|--------|-----|")
        for c in closed:
            print(f"| {c['symbol']} | {c['side']} | {c.get('qty_closed', 0)} | "
                  f"₹{c['entry_price']} | ₹{c['exit_price']} | "
                  f"{c.get('reason', '')} | ₹{c.get('pnl', 0):+,.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
