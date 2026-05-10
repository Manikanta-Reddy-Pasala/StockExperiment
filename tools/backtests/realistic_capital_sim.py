"""Realistic capital simulator — time-ordered, compounding, concurrency-capped.

Reads EMA 200/400 backtest cycle traces (per-symbol .md files) and replays
ENTRY / PARTIAL / TARGET / STOP events chronologically against a single
capital pool.

Constraints
  - max_concurrent: cap on simultaneous open positions
  - cash slice per ENTRY = available_cash / (max_concurrent - currently_open)
    (i.e. equal-share what's left after current positions)
  - shares = floor(slot_alloc / entry_price); skip ENTRY if shares < 1
  - PARTIAL sells 50% qty; remaining qty exits on TARGET/STOP
  - Cash from closes returns to pool (compounding)

Usage
  python realistic_capital_sim.py <case_dir> [--capital N] [max_concurrent ...]
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CAPITAL = 200_000


@dataclass
class Position:
    symbol: str
    entry_ts: str
    entry_price: float
    qty: int
    qty_remaining: int
    partial_done: bool = False


@dataclass
class Event:
    ts: str
    symbol: str
    kind: str            # ENTRY / PARTIAL / TARGET / STOP
    price: float


_TS_RE = re.compile(r"\| ([A-Za-z][A-Za-z0-9 _\(\)]+?) \| (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| ([\d.]+) \|")


def parse_per_symbol(path: str) -> List[Event]:
    """Parse a per-symbol .md report — extract Strategy Cycles event rows."""
    events: List[Event] = []
    if not os.path.exists(path):
        return events
    sym = os.path.basename(path).replace(".md", "").upper()
    in_cycles = False
    in_table = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Strategy Cycles"):
                in_cycles = True
                continue
            if not in_cycles:
                continue
            if s.startswith("## "):
                break
            if s.startswith("| Stage"):
                in_table = True
                continue
            if in_table and s.startswith("|") and "|---|" not in s:
                cells = [c.strip() for c in s.split("|")[1:-1]]
                if len(cells) < 6:
                    continue
                stage, ts, price = cells[0], cells[1], cells[2]
                try:
                    p = float(price)
                except ValueError:
                    continue
                kind = None
                if "First Entry" in stage or "Second Entry" in stage:
                    kind = "ENTRY"
                elif "Partial book" in stage:
                    kind = "PARTIAL"
                elif "Target hit" in stage:
                    kind = "TARGET"
                elif "Stop hit" in stage:
                    kind = "STOP"
                if kind:
                    events.append(Event(ts=ts, symbol=sym, kind=kind, price=p))
            elif in_table and not s.startswith("|"):
                in_table = False
    return events


def collect_events(case_dir: str) -> List[Event]:
    out: List[Event] = []
    for fname in sorted(os.listdir(case_dir)):
        if fname.startswith("_") or not fname.endswith(".md"):
            continue
        out.extend(parse_per_symbol(os.path.join(case_dir, fname)))
    out.sort(key=lambda e: (e.ts, e.symbol, _kind_rank(e.kind)))
    return out


def _kind_rank(k: str) -> int:
    return {"ENTRY": 0, "PARTIAL": 1, "TARGET": 2, "STOP": 2}.get(k, 9)


def simulate(case_dir: str, max_concurrent: int, capital: int = CAPITAL) -> Dict:
    events = collect_events(case_dir)
    cash = float(capital)
    open_positions: Dict[str, List[Position]] = {}    # symbol -> list (re-entries)
    cur_open = 0
    realized = 0.0
    equity_curve: List[Tuple[str, float]] = []
    skipped_entries = 0
    taken_entries = 0
    closed_legs = 0
    max_dd = 0.0
    peak = float(capital)
    last_price: Dict[str, float] = {}

    for ev in events:
        last_price[ev.symbol] = ev.price
        # Mark-to-cash equity: cash + every open position at symbol's last
        # seen price (may be stale but bounded).
        equity = cash
        for sym, poslist in open_positions.items():
            mark = last_price.get(sym, poslist[0].entry_price)
            for p in poslist:
                equity += p.qty_remaining * mark
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        equity_curve.append((ev.ts, equity))

        if ev.kind == "ENTRY":
            slots_left = max_concurrent - cur_open
            if slots_left <= 0:
                skipped_entries += 1
                continue
            slot_alloc = cash / slots_left
            shares = int(slot_alloc // ev.price)
            if shares < 1:
                skipped_entries += 1
                continue
            cost = shares * ev.price
            cash -= cost
            pos = Position(symbol=ev.symbol, entry_ts=ev.ts, entry_price=ev.price,
                           qty=shares, qty_remaining=shares)
            open_positions.setdefault(ev.symbol, []).append(pos)
            cur_open += 1
            taken_entries += 1

        elif ev.kind == "PARTIAL":
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if not p.partial_done and p.qty_remaining > 0:
                    book_qty = p.qty // 2
                    if book_qty < 1:
                        continue
                    proceeds = book_qty * ev.price
                    cash += proceeds
                    p.qty_remaining -= book_qty
                    p.partial_done = True
                    realized += (ev.price - p.entry_price) * book_qty

        elif ev.kind in ("TARGET", "STOP"):
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if p.qty_remaining > 0:
                    proceeds = p.qty_remaining * ev.price
                    cash += proceeds
                    realized += (ev.price - p.entry_price) * p.qty_remaining
                    p.qty_remaining = 0
                    closed_legs += 1
                    cur_open -= 1
            # remove fully-closed positions
            open_positions[ev.symbol] = [p for p in poslist if p.qty_remaining > 0]
            if not open_positions[ev.symbol]:
                del open_positions[ev.symbol]

    # Mark remaining open at last price (use last event price as proxy)
    final_equity = cash
    if events:
        last_price_by_sym: Dict[str, float] = {}
        for ev in events:
            last_price_by_sym[ev.symbol] = ev.price
        for sym, poslist in open_positions.items():
            for p in poslist:
                final_equity += p.qty_remaining * last_price_by_sym.get(sym, p.entry_price)

    return {
        "case_dir": case_dir,
        "max_concurrent": max_concurrent,
        "events_total": len(events),
        "taken_entries": taken_entries,
        "skipped_entries": skipped_entries,
        "closed_legs": closed_legs,
        "starting_capital": capital,
        "final_equity": round(final_equity, 2),
        "realized_pnl": round(realized, 2),
        "cash_remaining": round(cash, 2),
        "roi_pct": round((final_equity - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "open_at_end": sum(len(v) for v in open_positions.values()),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    capital = CAPITAL
    if "--capital" in args:
        i = args.index("--capital")
        capital = int(args[i + 1])
        del args[i:i + 2]
    case_dir = args[0]
    caps = [int(x) for x in args[1:]] if len(args) > 1 else [2, 3, 5, 8, 10, 15, 20, 30, 50]
    print(f"Case: {case_dir}")
    print(f"Capital: INR {capital:,}")
    print()
    print(f"{'Max':>4}  {'Taken':>6}  {'Skip':>6}  {'Final':>12}  {'ROI%':>7}  {'MaxDD%':>7}  {'OpenEnd':>7}")
    print("-" * 72)
    for cap in caps:
        r = simulate(case_dir, cap, capital=capital)
        print(f"{cap:>4}  {r['taken_entries']:>6}  {r['skipped_entries']:>6}  "
              f"{r['final_equity']:>12,.0f}  {r['roi_pct']:>+7.2f}  "
              f"{r['max_drawdown_pct']:>7.2f}  {r['open_at_end']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
