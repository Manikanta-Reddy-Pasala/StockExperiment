"""Monthly P&L profile — replays a backtest case dir grouped by calendar month.

Reuses ``realistic_capital_sim.collect_events`` so the parser stays in one place
(handles the per-symbol .md ``Strategy Cycles`` table format). Walks the same
event sequence as ``simulate()`` but buckets closed legs + equity snapshots by
``YYYY-MM`` of the event timestamp.

Output: a markdown table to stdout AND ``CASE_DIR/_monthly_profile.md``.

Usage:
    venv/bin/python tools/backtests/monthly_profile.py CASE_DIR \
        [--capital 200000] [--max-concurrent 2]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.realistic_capital_sim import (  # noqa: E402
    collect_events, Event, Position,
)


@dataclass
class MonthBucket:
    ym: str
    trades_closed: int = 0
    winners: int = 0
    losers: int = 0
    realized_pnl_inr: float = 0.0
    end_equity_inr: float = 0.0
    peak_equity_inr: float = 0.0
    trough_dd_pct: float = 0.0
    legs_pct: List[float] = field(default_factory=list)


def _ym(ts: str) -> str:
    """YYYY-MM-DD HH:MM:SS -> YYYY-MM."""
    return ts[:7]


def simulate_monthly(case_dir: str, capital: int, max_concurrent: int) -> List[MonthBucket]:
    """Replay events; bucket realized P&L + equity snapshots by month.

    Mirrors ``realistic_capital_sim.simulate`` allocation rules but tracks per-
    month aggregates. End-of-month equity snapshot = mark-to-cash equity at the
    last event in that month (same definition simulate() uses for its curve).
    """
    events = collect_events(case_dir)
    cash = float(capital)
    open_positions: Dict[str, List[Position]] = {}
    cur_open = 0
    last_price: Dict[str, float] = {}
    buckets: Dict[str, MonthBucket] = {}
    peak = float(capital)

    def _equity() -> float:
        eq = cash
        for sym, poslist in open_positions.items():
            mark = last_price.get(sym, poslist[0].entry_price if poslist else 0.0)
            for p in poslist:
                eq += p.qty_remaining * mark
        return eq

    for ev in events:
        last_price[ev.symbol] = ev.price
        ym = _ym(ev.ts)
        bucket = buckets.setdefault(ym, MonthBucket(ym=ym))

        if ev.kind == "ENTRY":
            slots_left = max_concurrent - cur_open
            if slots_left <= 0:
                pass
            else:
                slot_alloc = cash / slots_left
                shares = int(slot_alloc // ev.price)
                if shares >= 1:
                    cost = shares * ev.price
                    cash -= cost
                    pos = Position(symbol=ev.symbol, entry_ts=ev.ts,
                                   entry_price=ev.price, qty=shares,
                                   qty_remaining=shares)
                    open_positions.setdefault(ev.symbol, []).append(pos)
                    cur_open += 1

        elif ev.kind == "PARTIAL":
            for p in open_positions.get(ev.symbol, []):
                if not p.partial_done and p.qty_remaining > 0:
                    book_qty = p.qty // 2
                    if book_qty < 1:
                        continue
                    proceeds = book_qty * ev.price
                    cash += proceeds
                    p.qty_remaining -= book_qty
                    p.partial_done = True
                    leg_pnl = (ev.price - p.entry_price) * book_qty
                    leg_pct = (ev.price / p.entry_price - 1.0) * 100 if p.entry_price > 0 else 0.0
                    bucket.realized_pnl_inr += leg_pnl
                    bucket.trades_closed += 1
                    if leg_pct > 0:
                        bucket.winners += 1
                    else:
                        bucket.losers += 1
                    bucket.legs_pct.append(leg_pct)

        elif ev.kind in ("TARGET", "STOP"):
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if p.qty_remaining > 0:
                    proceeds = p.qty_remaining * ev.price
                    cash += proceeds
                    leg_pnl = (ev.price - p.entry_price) * p.qty_remaining
                    leg_pct = (ev.price / p.entry_price - 1.0) * 100 if p.entry_price > 0 else 0.0
                    bucket.realized_pnl_inr += leg_pnl
                    bucket.trades_closed += 1
                    if leg_pct > 0:
                        bucket.winners += 1
                    else:
                        bucket.losers += 1
                    bucket.legs_pct.append(leg_pct)
                    p.qty_remaining = 0
                    cur_open -= 1
            open_positions[ev.symbol] = [p for p in poslist if p.qty_remaining > 0]
            if not open_positions.get(ev.symbol):
                open_positions.pop(ev.symbol, None)

        eq = _equity()
        peak = max(peak, eq)
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
        # Last-event-in-month wins for end_equity; trough_dd_pct is the worst
        # within the month.
        bucket.end_equity_inr = eq
        bucket.peak_equity_inr = max(bucket.peak_equity_inr, eq)
        if dd_pct > bucket.trough_dd_pct:
            bucket.trough_dd_pct = dd_pct

    # Sort by year-month.
    return [buckets[k] for k in sorted(buckets)]


def render_markdown(case_dir: str, capital: int, max_concurrent: int,
                     months: List[MonthBucket]) -> str:
    lines: List[str] = [
        f"# Monthly P&L Profile — {os.path.basename(case_dir.rstrip('/'))}",
        "",
        f"- Capital: INR {capital:,}",
        f"- Max concurrent: {max_concurrent}",
        f"- Months observed: {len(months)}",
        "",
        "| YYYY-MM | Trades | Win | Loss | Win% | Avg₹ | Sum₹ | EndEquity₹ | DD% |",
        "|---------|-------:|----:|-----:|-----:|-----:|-----:|-----------:|----:|",
    ]
    for m in months:
        n = m.trades_closed
        win_pct = (m.winners / n * 100) if n else 0.0
        avg_inr = (m.realized_pnl_inr / n) if n else 0.0
        lines.append(
            f"| {m.ym} | {n} | {m.winners} | {m.losers} | "
            f"{win_pct:.1f}% | {avg_inr:,.0f} | {m.realized_pnl_inr:,.0f} | "
            f"{m.end_equity_inr:,.0f} | {m.trough_dd_pct:.2f} |"
        )
    if not months:
        lines.append("| _no events_ | 0 | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00 |")
    # Totals row
    if months:
        total_trades = sum(m.trades_closed for m in months)
        total_wins = sum(m.winners for m in months)
        total_pnl = sum(m.realized_pnl_inr for m in months)
        end_eq = months[-1].end_equity_inr
        max_dd = max((m.trough_dd_pct for m in months), default=0.0)
        win_pct = (total_wins / total_trades * 100) if total_trades else 0.0
        avg_inr = (total_pnl / total_trades) if total_trades else 0.0
        lines.append(
            f"| **TOTAL** | **{total_trades}** | **{total_wins}** | "
            f"**{total_trades - total_wins}** | **{win_pct:.1f}%** | "
            f"**{avg_inr:,.0f}** | **{total_pnl:,.0f}** | "
            f"**{end_eq:,.0f}** | **{max_dd:.2f}** |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=str)
    parser.add_argument("--capital", type=int, default=200_000)
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    if not os.path.isdir(args.case_dir):
        print(f"Not a directory: {args.case_dir}", file=sys.stderr)
        return 2

    months = simulate_monthly(args.case_dir, capital=args.capital,
                               max_concurrent=args.max_concurrent)
    md = render_markdown(args.case_dir, args.capital, args.max_concurrent, months)
    sys.stdout.write(md)
    out_path = Path(args.case_dir) / "_monthly_profile.md"
    out_path.write_text(md)
    print(f"\n-> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
