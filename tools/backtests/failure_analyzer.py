"""Failure analyzer — identifies WHY losing trades fail.

Reads a trade ledger CSV (produced by trade_ledger.py) and buckets
losses by:
  - Symbol (which stocks bleed the most)
  - Day of week (Mon/Tue/...)
  - Hour (opening vs midday vs closing)
  - Hold duration bins (<1d, 1-3d, 3-7d, >7d)
  - Loss size bins (-1%, -2%, -5%, -10%)
  - Exit reason (STOP vs PARTIAL flip)
  - Direction (BUY vs SELL — inferred from price movement)

Outputs failure_analysis.md with actionable patterns to filter.

Usage:
  python tools/backtests/failure_analyzer.py \
    --csv exports/backtests/ledgers/MODEL_1_EMA_200_400.csv \
    --out exports/backtests/FAILURE_ANALYSIS_MODEL_1.md
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("failure_analyzer")


def load_ledger(path: Path) -> List[Dict]:
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pnl_inr"] = float(row["pnl_inr"])
            row["pnl_pct"] = float(row["pnl_pct"])
            row["qty"] = int(row["qty"])
            row["entry_price"] = float(row["entry_price"])
            row["exit_price"] = float(row["exit_price"])
            row["holding_bars"] = int(row["holding_bars"])
            out.append(row)
    return out


def analyze(records: List[Dict]) -> Dict:
    losers = [r for r in records if r["pnl_inr"] < 0]
    winners = [r for r in records if r["pnl_inr"] > 0]
    if not records:
        return {}

    # By symbol
    by_symbol = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
    for r in records:
        s = by_symbol[r["symbol"]]
        if r["pnl_inr"] > 0: s["wins"] += 1
        else: s["losses"] += 1
        s["total_pnl"] += r["pnl_inr"]
    sym_summary = sorted([{"symbol": k, **v} for k, v in by_symbol.items()],
                          key=lambda x: x["total_pnl"])

    # By day of week
    by_dow = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
    for r in records:
        try:
            dt = datetime.strptime(r["entry_date"], "%Y-%m-%d")
            dow = dt.strftime("%a")
            d = by_dow[dow]
            if r["pnl_inr"] > 0: d["wins"] += 1
            else: d["losses"] += 1
            d["total_pnl"] += r["pnl_inr"]
        except Exception:
            pass

    # By hour
    by_hour = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
    for r in records:
        h = r["entry_time"][:2]
        b = by_hour[h]
        if r["pnl_inr"] > 0: b["wins"] += 1
        else: b["losses"] += 1
        b["total_pnl"] += r["pnl_inr"]

    # By hold bars
    bins = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 9999)]
    by_hold = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
    for r in records:
        for lo, hi in bins:
            if lo <= r["holding_bars"] < hi:
                key = f"{lo}-{hi if hi < 9999 else '+'}"
                d = by_hold[key]
                if r["pnl_inr"] > 0: d["wins"] += 1
                else: d["losses"] += 1
                d["total_pnl"] += r["pnl_inr"]
                break

    # By exit reason
    by_reason = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
    for r in records:
        b = by_reason[r["reason"]]
        if r["pnl_inr"] > 0: b["wins"] += 1
        else: b["losses"] += 1
        b["total_pnl"] += r["pnl_inr"]

    return {
        "total_trades": len(records),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": 100 * len(winners) / len(records),
        "total_pnl": sum(r["pnl_inr"] for r in records),
        "avg_winner": sum(r["pnl_inr"] for r in winners) / len(winners) if winners else 0,
        "avg_loser": sum(r["pnl_inr"] for r in losers) / len(losers) if losers else 0,
        "biggest_winner": max(records, key=lambda r: r["pnl_inr"]) if records else None,
        "biggest_loser": min(records, key=lambda r: r["pnl_inr"]) if records else None,
        "by_symbol": sym_summary,
        "by_dow": dict(by_dow),
        "by_hour": dict(by_hour),
        "by_hold": dict(by_hold),
        "by_reason": dict(by_reason),
    }


def render_md(analysis: Dict, model_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {model_name} — Failure Analysis\n",
        f"_Generated: {datetime.now().isoformat()}_\n",
        f"## Headline",
        f"- Total trades: {analysis['total_trades']}",
        f"- Winners / Losers: {analysis['winners']} / {analysis['losers']}",
        f"- Win rate: {analysis['win_rate']:.1f}%",
        f"- Total P&L: ₹{analysis['total_pnl']:+,.0f}",
        f"- Avg winner: ₹{analysis['avg_winner']:+,.0f}",
        f"- Avg loser: ₹{analysis['avg_loser']:+,.0f}",
    ]
    if analysis.get("biggest_winner"):
        w = analysis["biggest_winner"]
        lines.append(f"- Biggest winner: {w['symbol']} {w['entry_date']} ₹{w['pnl_inr']:+,.0f} ({w['pnl_pct']:+.2f}%)")
    if analysis.get("biggest_loser"):
        l = analysis["biggest_loser"]
        lines.append(f"- Biggest loser: {l['symbol']} {l['entry_date']} ₹{l['pnl_inr']:+,.0f} ({l['pnl_pct']:+.2f}%)")

    lines += ["\n## Worst stocks (drop candidates)\n",
              "| Symbol | Wins | Losses | Win% | Total P&L |",
              "|--------|-----:|-------:|-----:|----------:|"]
    for s in analysis["by_symbol"][:15]:
        total = s["wins"] + s["losses"]
        wr = (100 * s["wins"] / total) if total else 0
        lines.append(f"| {s['symbol']} | {s['wins']} | {s['losses']} | "
                     f"{wr:.0f}% | ₹{s['total_pnl']:+,.0f} |")

    lines += ["\n## Best stocks (keep / weight up)\n",
              "| Symbol | Wins | Losses | Win% | Total P&L |",
              "|--------|-----:|-------:|-----:|----------:|"]
    for s in reversed(analysis["by_symbol"][-10:]):
        total = s["wins"] + s["losses"]
        wr = (100 * s["wins"] / total) if total else 0
        lines.append(f"| {s['symbol']} | {s['wins']} | {s['losses']} | "
                     f"{wr:.0f}% | ₹{s['total_pnl']:+,.0f} |")

    lines += ["\n## P&L by Day of Week\n",
              "| Day | Wins | Losses | Win% | Total P&L |",
              "|-----|-----:|-------:|-----:|----------:|"]
    for dow in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        v = analysis["by_dow"].get(dow, {"wins": 0, "losses": 0, "total_pnl": 0})
        total = v["wins"] + v["losses"]
        wr = (100 * v["wins"] / total) if total else 0
        lines.append(f"| {dow} | {v['wins']} | {v['losses']} | {wr:.0f}% | ₹{v['total_pnl']:+,.0f} |")

    lines += ["\n## P&L by Entry Hour (IST)\n",
              "| Hour | Wins | Losses | Win% | Total P&L |",
              "|------|-----:|-------:|-----:|----------:|"]
    for h in sorted(analysis["by_hour"].keys()):
        v = analysis["by_hour"][h]
        total = v["wins"] + v["losses"]
        wr = (100 * v["wins"] / total) if total else 0
        lines.append(f"| {h}:xx | {v['wins']} | {v['losses']} | {wr:.0f}% | ₹{v['total_pnl']:+,.0f} |")

    lines += ["\n## P&L by Hold Duration (bars)\n",
              "| Bars | Wins | Losses | Win% | Total P&L |",
              "|------|-----:|-------:|-----:|----------:|"]
    for k, v in sorted(analysis["by_hold"].items()):
        total = v["wins"] + v["losses"]
        wr = (100 * v["wins"] / total) if total else 0
        lines.append(f"| {k} | {v['wins']} | {v['losses']} | {wr:.0f}% | ₹{v['total_pnl']:+,.0f} |")

    lines += ["\n## P&L by Exit Reason\n",
              "| Reason | Wins | Losses | Win% | Total P&L |",
              "|--------|-----:|-------:|-----:|----------:|"]
    for k, v in sorted(analysis["by_reason"].items()):
        total = v["wins"] + v["losses"]
        wr = (100 * v["wins"] / total) if total else 0
        lines.append(f"| {k} | {v['wins']} | {v['losses']} | {wr:.0f}% | ₹{v['total_pnl']:+,.0f} |")

    lines += [
        "\n## Actionable filters (based on patterns)\n",
        "Drop the worst 5-10 symbols from the universe.",
        "Skip entries on worst day-of-week.",
        "Skip entries during worst hour bucket.",
        "Skip trades with hold duration in worst bin.",
    ]

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="Custom Model")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    records = load_ledger(Path(args.csv))
    log.info(f"Loaded {len(records)} trades")
    analysis = analyze(records)
    render_md(analysis, args.model_name, Path(args.out))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
