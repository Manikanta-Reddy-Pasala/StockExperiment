"""Trade ledger generator — emits detailed per-trade buy/sell records.

Reads per-stock cycle .md files, replays cap-sim, captures every
ENTRY/PARTIAL/TARGET/STOP with: date, symbol, side, qty, entry_price,
exit_price, P&L₹, P&L%, holding_days, reason.

Output: CSV + markdown table per model.

Usage:
  python tools/backtests/trade_ledger.py \
    --case exports/backtests/multiyear_n50/nifty50_ema_200_400_2025_2026 \
    --capital 1000000 --max-concurrent 2 \
    --model-name "EMA 200/400" \
    --out-md exports/backtests/ledgers/ema_200_400_2025_2026.md \
    --out-csv exports/backtests/ledgers/ema_200_400_2025_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from realistic_capital_sim import collect_events, Position  # noqa: E402


log = logging.getLogger("trade_ledger")


@dataclass
class TradeRecord:
    entry_date: str
    entry_time: str
    exit_date: str
    exit_time: str
    symbol: str
    qty: int
    entry_price: float
    exit_price: float
    pnl_inr: float
    pnl_pct: float
    holding_bars: int
    reason: str   # TARGET / STOP / PARTIAL / EOD
    cash_before: float
    cash_after: float


def simulate_with_ledger(case_dir: str, max_concurrent: int,
                          capital: int) -> List[TradeRecord]:
    """Re-runs cap-sim. For each closed trade leg, emits a TradeRecord."""
    events = collect_events(case_dir)
    cash = float(capital)
    open_positions: Dict[str, List[Position]] = {}
    cur_open = 0
    records: List[TradeRecord] = []
    last_price: Dict[str, float] = {}

    # Map position -> entry event index for bars-held calc
    entry_idx_by_pos: Dict[int, int] = {}

    for i, ev in enumerate(events):
        last_price[ev.symbol] = ev.price

        if ev.kind == "ENTRY":
            slots_left = max_concurrent - cur_open
            if slots_left <= 0: continue
            slot_alloc = cash / slots_left
            shares = int(slot_alloc // ev.price)
            if shares < 1: continue
            cost = shares * ev.price
            cash -= cost
            pos = Position(symbol=ev.symbol, entry_ts=ev.ts,
                           entry_price=ev.price, qty=shares,
                           qty_remaining=shares)
            open_positions.setdefault(ev.symbol, []).append(pos)
            entry_idx_by_pos[id(pos)] = i
            cur_open += 1

        elif ev.kind == "PARTIAL":
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if not p.partial_done and p.qty_remaining > 0:
                    book_qty = p.qty // 2
                    if book_qty < 1: continue
                    proceeds = book_qty * ev.price
                    pnl = (ev.price - p.entry_price) * book_qty
                    pnl_pct = (ev.price - p.entry_price) / p.entry_price * 100
                    cash_before = cash
                    cash += proceeds
                    p.qty_remaining -= book_qty
                    p.partial_done = True
                    records.append(TradeRecord(
                        entry_date=p.entry_ts[:10],
                        entry_time=p.entry_ts[11:16],
                        exit_date=ev.ts[:10], exit_time=ev.ts[11:16],
                        symbol=ev.symbol, qty=book_qty,
                        entry_price=p.entry_price, exit_price=ev.price,
                        pnl_inr=pnl, pnl_pct=pnl_pct,
                        holding_bars=i - entry_idx_by_pos.get(id(p), i),
                        reason="PARTIAL",
                        cash_before=cash_before, cash_after=cash,
                    ))

        elif ev.kind in ("TARGET", "STOP"):
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if p.qty_remaining > 0:
                    proceeds = p.qty_remaining * ev.price
                    pnl = (ev.price - p.entry_price) * p.qty_remaining
                    pnl_pct = (ev.price - p.entry_price) / p.entry_price * 100
                    cash_before = cash
                    cash += proceeds
                    records.append(TradeRecord(
                        entry_date=p.entry_ts[:10],
                        entry_time=p.entry_ts[11:16],
                        exit_date=ev.ts[:10], exit_time=ev.ts[11:16],
                        symbol=ev.symbol, qty=p.qty_remaining,
                        entry_price=p.entry_price, exit_price=ev.price,
                        pnl_inr=pnl, pnl_pct=pnl_pct,
                        holding_bars=i - entry_idx_by_pos.get(id(p), i),
                        reason=ev.kind,
                        cash_before=cash_before, cash_after=cash,
                    ))
                    p.qty_remaining = 0
                    cur_open -= 1
            open_positions[ev.symbol] = [p for p in poslist if p.qty_remaining > 0]
            if not open_positions[ev.symbol]:
                del open_positions[ev.symbol]

    return records


def render_md(records: List[TradeRecord], model_name: str,
              out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pnl = sum(r.pnl_inr for r in records)
    wins = sum(1 for r in records if r.pnl_inr > 0)
    losses = sum(1 for r in records if r.pnl_inr <= 0)
    win_rate = (100 * wins / (wins + losses)) if (wins + losses) > 0 else 0

    lines = [
        f"# {model_name} — Trade Ledger\n",
        f"_Generated: {datetime.now().isoformat()}_\n",
        f"## Headline",
        f"- Total closed legs: {len(records)}",
        f"- Wins / Losses: {wins} / {losses}",
        f"- Win rate: {win_rate:.1f}%",
        f"- Total P&L: ₹{total_pnl:+,.2f}",
        "",
        "## Every Trade (chronological)",
        "",
        "| # | Entry Date | Entry Time | Exit Date | Exit Time | Symbol | Qty | Buy ₹ | Sell ₹ | P&L ₹ | P&L % | Bars | Reason | Cash After ₹ |",
        "|--:|-----------|------------|-----------|-----------|--------|----:|------:|-------:|------:|------:|-----:|--------|-------------:|",
    ]
    for i, r in enumerate(records, 1):
        lines.append(
            f"| {i} | {r.entry_date} | {r.entry_time} | {r.exit_date} | "
            f"{r.exit_time} | {r.symbol} | {r.qty} | "
            f"₹{r.entry_price:.2f} | ₹{r.exit_price:.2f} | "
            f"₹{r.pnl_inr:+,.0f} | {r.pnl_pct:+.2f}% | "
            f"{r.holding_bars} | {r.reason} | "
            f"₹{r.cash_after:,.0f} |"
        )
    out_path.write_text("\n".join(lines))


def render_csv(records: List[TradeRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_num", "entry_date", "entry_time", "exit_date",
                    "exit_time", "symbol", "qty", "entry_price", "exit_price",
                    "pnl_inr", "pnl_pct", "holding_bars", "reason",
                    "cash_before", "cash_after"])
        for i, r in enumerate(records, 1):
            w.writerow([i, r.entry_date, r.entry_time, r.exit_date, r.exit_time,
                        r.symbol, r.qty, r.entry_price, r.exit_price,
                        r.pnl_inr, r.pnl_pct, r.holding_bars, r.reason,
                        r.cash_before, r.cash_after])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--capital", type=int, default=1_000_000)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--model-name", default="Custom")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    recs = simulate_with_ledger(args.case, args.max_concurrent, args.capital)
    log.info(f"Generated {len(recs)} trade records")
    render_md(recs, args.model_name, Path(args.out_md))
    log.info(f"Wrote {args.out_md}")
    if args.out_csv:
        render_csv(recs, Path(args.out_csv))
        log.info(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
