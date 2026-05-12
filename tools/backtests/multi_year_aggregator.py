"""Aggregate multi-year backtest results into a single year+month report.

Consumes the output dirs from run_yearly_backtest.py
(format: <out_root>/<universe>_<model>_<year_start>_<year_end>/) and
emits a consolidated markdown:

  - Per-model yearly summary across N years
  - Per-model monthly P&L across all months
  - Overall yearly + monthly aggregate
  - Equity curve compact

Usage:
  python tools/backtests/multi_year_aggregator.py \
    --root /app/exports/backtests/multiyear_n50 \
    --out exports/backtests/MULTI_YEAR_REPORT.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_monthly_profile(path: Path) -> List[Dict]:
    """Parse _monthly_profile.md output from monthly_profile.py."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(
                r"\|\s*(\d{4}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
                r"\s*([\d.]+)%\s*\|\s*(-?[\d,]+)\s*\|\s*(-?[\d,]+)\s*\|"
                r"\s*(-?[\d,]+)\s*\|\s*([\d.]+)\s*\|",
                line,
            )
            if m:
                rows.append({
                    "month": m.group(1),
                    "trades": int(m.group(2)),
                    "win": int(m.group(3)),
                    "loss": int(m.group(4)),
                    "win_pct": float(m.group(5)),
                    "avg_inr": float(m.group(6).replace(",", "")),
                    "sum_inr": float(m.group(7).replace(",", "")),
                    "end_eq": float(m.group(8).replace(",", "")),
                    "dd_pct": float(m.group(9)),
                })
    return rows


def parse_capital_sim_at(path: Path, max_concurrent: int = 2) -> Dict:
    """Parse _capital_sim.txt at given max_concurrent row."""
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)\s+([+-][\d.]+)\s+([\d.]+)\s+(\d+)",
                line,
            )
            if m and int(m.group(1)) == max_concurrent:
                return {
                    "max": int(m.group(1)),
                    "taken": int(m.group(2)),
                    "skip": int(m.group(3)),
                    "final": float(m.group(4).replace(",", "")),
                    "roi_pct": float(m.group(5)),
                    "dd_pct": float(m.group(6)),
                    "open_end": int(m.group(7)),
                }
    return {}


def discover_runs(root: Path) -> List[Dict]:
    """Find all <universe>_<model>_<from>_<to> dirs."""
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        # nifty50_ema_9_21_2025_2026 → universe=nifty50, model=ema_9_21, years=2025-2026
        parts = d.name.split("_")
        if len(parts) < 4:
            continue
        try:
            year_end = int(parts[-1])
            year_start = int(parts[-2])
        except ValueError:
            continue
        universe = parts[0]
        model = "_".join(parts[1:-2])
        runs.append({
            "dir": d,
            "universe": universe,
            "model": model,
            "year_start": year_start,
            "year_end": year_end,
            "label": f"{year_start}-{year_end}",
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--capital", type=int, default=1_000_000)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    runs = discover_runs(root)
    if not runs:
        print(f"No runs found in {root}", file=sys.stderr)
        sys.exit(1)

    lines = [
        f"# Multi-Year Backtest Report\n",
        f"_Generated: {datetime.now().isoformat()}_\n",
        f"- Root: `{root}`",
        f"- Capital: ₹{args.capital:,}",
        f"- max_concurrent: {args.max_concurrent}",
        f"- Runs discovered: {len(runs)}",
        "\n## Yearly Summary (max_concurrent=" + str(args.max_concurrent) + ")\n",
        "| Model | Universe | Year | Taken | Skip | Final₹ | ROI% | MaxDD% |",
        "|-------|----------|------|------:|-----:|-------:|-----:|-------:|",
    ]
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        cap = parse_capital_sim_at(r["dir"] / "_capital_sim.txt", args.max_concurrent)
        if not cap: continue
        by_model[r["model"]].append({**r, **cap})
        lines.append(
            f"| {r['model']} | {r['universe']} | {r['label']} | "
            f"{cap['taken']} | {cap['skip']} | ₹{cap['final']:,.0f} | "
            f"{cap['roi_pct']:+.2f} | {cap['dd_pct']:.2f} |"
        )

    # Per-model yearly aggregate
    lines += ["\n## Per-Model Multi-Year Aggregate\n"]
    for model, rows in sorted(by_model.items()):
        rois = [r["roi_pct"] for r in rows]
        dds = [r["dd_pct"] for r in rows]
        years = len(rows)
        avg_roi = sum(rois) / years if years else 0
        compound_roi = 1.0
        for r in rois:
            compound_roi *= (1 + r / 100)
        compound_roi = (compound_roi - 1) * 100
        worst_dd = max(dds) if dds else 0
        lines.append(f"### {model}")
        lines.append(f"- Years: {years}")
        lines.append(f"- Avg yearly ROI: {avg_roi:+.2f}%")
        lines.append(f"- Compound multi-year ROI: {compound_roi:+.2f}%")
        lines.append(f"- Worst yearly DD: {worst_dd:.2f}%")
        lines.append(f"- Yearly ROIs: " + ", ".join(f"{r:+.2f}%" for r in rois))
        lines.append("")

    # Monthly per model
    lines += ["\n## Monthly P&L by Model (max_concurrent=" + str(args.max_concurrent) + ")\n"]
    for model, rows in sorted(by_model.items()):
        lines.append(f"### {model}")
        lines.append("| Month | Trades | Win% | P&L₹ | Cumulative₹ |")
        lines.append("|-------|-------:|-----:|-----:|------------:|")
        all_months: List[Dict] = []
        for r in rows:
            month_path = r["dir"] / "_monthly_profile.md"
            all_months.extend(parse_monthly_profile(month_path))
        cumulative = 0.0
        for m in sorted(all_months, key=lambda x: x["month"]):
            cumulative += m["sum_inr"]
            lines.append(
                f"| {m['month']} | {m['trades']} | {m['win_pct']:.0f}% | "
                f"₹{m['sum_inr']:+,.0f} | ₹{cumulative:+,.0f} |"
            )
        # Year aggregate per model
        yearly: Dict[str, float] = defaultdict(float)
        yearly_trades: Dict[str, int] = defaultdict(int)
        yearly_wins: Dict[str, int] = defaultdict(int)
        for m in all_months:
            y = m["month"][:4]
            yearly[y] += m["sum_inr"]
            yearly_trades[y] += m["trades"]
            yearly_wins[y] += m["win"]
        lines.append("")
        lines.append("| Year | Trades | Wins | Win% | Profit₹ | ROI% |")
        lines.append("|------|-------:|-----:|-----:|--------:|-----:|")
        for y in sorted(yearly.keys()):
            wr = (100 * yearly_wins[y] / yearly_trades[y]) if yearly_trades[y] else 0
            roi = yearly[y] / args.capital * 100
            lines.append(
                f"| {y} | {yearly_trades[y]} | {yearly_wins[y]} | "
                f"{wr:.0f}% | ₹{yearly[y]:+,.0f} | {roi:+.2f}% |"
            )
        lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
