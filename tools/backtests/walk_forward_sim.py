"""Walk-forward capital simulation — no lookahead bias.

Splits a per-symbol backtest dir into two halves by event timestamp:
  - First half: rank symbols by cumulative leg sum%, pick top-N
  - Second half: run realistic capital sim on ONLY those top-N symbols

Usage:
  python walk_forward_sim.py <case_dir> --capital 500000 \
      --split-date 2025-11-08 --top-n 50 [max_concurrent ...]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Reuse parser from realistic sim
sys.path.insert(0, os.path.dirname(__file__))
from realistic_capital_sim import parse_per_symbol, simulate, Event  # type: ignore


def collect_events_per_symbol(case_dir: str) -> Dict[str, List[Event]]:
    """Return {symbol: [Event, ...]} for the case dir."""
    out: Dict[str, List[Event]] = {}
    for fname in sorted(os.listdir(case_dir)):
        if fname.startswith("_") or not fname.endswith(".md"):
            continue
        sym = fname.replace(".md", "").upper()
        events = parse_per_symbol(os.path.join(case_dir, fname))
        if events:
            out[sym] = events
    return out


def rank_first_half(events_by_sym: Dict[str, List[Event]],
                    split_date: str) -> List[Tuple[str, float]]:
    """For each symbol compute pct% of legs that closed BEFORE split_date.
    Closed leg = matched ENTRY/PARTIAL/TARGET/STOP pair (FIFO per symbol).
    Returns sorted [(symbol, sum_pct), ...] descending.
    """
    ranked: List[Tuple[str, float]] = []
    for sym, evs in events_by_sym.items():
        opens: List[Tuple[float, int]] = []  # (entry_price, qty)
        sum_pct = 0.0
        for e in evs:
            if e.ts >= split_date:
                break
            if e.kind == "ENTRY":
                opens.append((e.price, 1))
            elif e.kind == "PARTIAL" and opens:
                ep, q = opens[0]
                if q > 0:
                    sum_pct += (e.price / ep - 1.0) * 100 * 0.5  # 50% qty
                    opens[0] = (ep, q)  # keep, partial doesn't close
            elif e.kind in ("TARGET", "STOP") and opens:
                ep, q = opens.pop(0)
                if q > 0:
                    sum_pct += (e.price / ep - 1.0) * 100 * 0.5
        ranked.append((sym, sum_pct))
    ranked.sort(key=lambda x: -x[1])
    return ranked


def write_subset(events_by_sym: Dict[str, List[Event]],
                 picks: List[str], split_date: str, out_dir: str) -> None:
    """Write minimal .md files for picks, only events ≥ split_date."""
    os.makedirs(out_dir, exist_ok=True)
    for sym in picks:
        evs = [e for e in events_by_sym.get(sym, []) if e.ts >= split_date]
        if not evs:
            continue
        lines = [f"# {sym}", "", "## Strategy Cycles", "",
                 "| Stage | Time | Price | EMA200 | EMA400 | Note |",
                 "|-------|------|-------|--------|--------|------|"]
        for e in evs:
            stage = {"ENTRY": "First Entry (BUY)", "PARTIAL": "Partial book",
                     "TARGET": "Target hit", "STOP": "Stop hit"}[e.kind]
            lines.append(f"| {stage} | {e.ts} | {e.price:.2f} | 0 | 0 |  |")
        path = os.path.join(out_dir, f"{sym.lower()}.md")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--capital", type=int, default=500_000)
    ap.add_argument("--split-date", default="2025-11-08",
                    help="YYYY-MM-DD (use first half before, sim second half after)")
    ap.add_argument("--top-n", type=int, default=50,
                    help="Pick top N symbols by first-half sum%%")
    ap.add_argument("--caps", type=int, nargs="*",
                    default=[2, 3, 5, 8, 10, 15, 20, 30, 50])
    args = ap.parse_args()

    print(f"Case: {args.case_dir}")
    print(f"Capital: INR {args.capital:,}")
    print(f"Split: {args.split_date} (first half ranks, second half sims)")
    print(f"Top-N: {args.top_n}")
    print()

    by_sym = collect_events_per_symbol(args.case_dir)
    ranked = rank_first_half(by_sym, args.split_date)
    picks = [s for s, p in ranked[:args.top_n] if p > 0]
    print(f"Eligible (sum%>0 in first half): {sum(1 for _, p in ranked if p > 0)}/{len(ranked)}")
    print(f"Top {args.top_n} picks (first 10 shown):")
    for s, p in ranked[:10]:
        print(f"  {s:14s} first-half sum%={p:+.1f}")
    print()

    subset_dir = args.case_dir.rstrip("/") + f"_walkfwd_top{args.top_n}"
    write_subset(by_sym, picks, args.split_date, subset_dir)

    print(f"{'Max':>4}  {'Taken':>6}  {'Skip':>6}  {'Final':>12}  {'ROI%':>7}  {'MaxDD%':>7}  {'OpenEnd':>7}")
    print("-" * 72)
    for cap in args.caps:
        r = simulate(subset_dir, cap, capital=args.capital)
        print(f"{cap:>4}  {r['taken_entries']:>6}  {r['skipped_entries']:>6}  "
              f"{r['final_equity']:>12,.0f}  {r['roi_pct']:>+7.2f}  "
              f"{r['max_drawdown_pct']:>7.2f}  {r['open_at_end']:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
