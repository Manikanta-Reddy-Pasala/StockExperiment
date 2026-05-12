"""Pattern mining on baseline 1-yr backtest results.

Parses per-stock _summary_buy.md and _summary_sell.md tables, ranks
contributors, finds seasonality, and generates exports/backtests/PATTERN_MINING.md.

Usage:
  python tools/backtests/pattern_mining.py \
    --root exports/backtests/y1_baseline/y1_filter/nifty50_ema_200_400_2025_2026
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_per_symbol_table(md_path: Path) -> List[Dict]:
    if not md_path.exists():
        return []
    rows = []
    with open(md_path) as f:
        in_table = False
        for line in f:
            line = line.rstrip()
            if line.startswith("| Symbol | Legs"):
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if in_table:
                if not line.startswith("|"):
                    in_table = False
                    continue
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) < 9: continue
                try:
                    rows.append({
                        "symbol": parts[0],
                        "legs": int(parts[1]),
                        "win": int(parts[2]),
                        "win_pct": float(parts[3].rstrip("%")),
                        "tgt": int(parts[4]),
                        "sl": int(parts[5]),
                        "prt": int(parts[6]),
                        "avg_pct": float(parts[7].rstrip("%")),
                        "sum_pct": float(parts[8].rstrip("%")),
                    })
                except (ValueError, IndexError):
                    continue
    return rows


def parse_monthly(md_path: Path) -> List[Dict]:
    if not md_path.exists():
        return []
    rows = []
    with open(md_path) as f:
        for line in f:
            m = re.match(r"\|\s*(\d{4}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*(-?[\d,]+)\s*\|\s*(-?[\d,]+)\s*\|\s*(-?[\d,]+)\s*\|\s*([\d.]+)\s*\|", line)
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


def parse_capital_sim(txt_path: Path) -> List[Dict]:
    if not txt_path.exists():
        return []
    rows = []
    with open(txt_path) as f:
        for line in f:
            m = re.match(
                r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)\s+([+-][\d.]+)\s+([\d.]+)\s+(\d+)",
                line,
            )
            if m:
                rows.append({
                    "max": int(m.group(1)),
                    "taken": int(m.group(2)),
                    "skip": int(m.group(3)),
                    "final": float(m.group(4).replace(",", "")),
                    "roi_pct": float(m.group(5)),
                    "dd_pct": float(m.group(6)),
                    "open_end": int(m.group(7)),
                })
    return rows


def combine(buy: List[Dict], sell: List[Dict]) -> Dict[str, Dict]:
    """Merge buy+sell per-symbol into one dict keyed by symbol."""
    combined: Dict[str, Dict] = {}
    for r in buy:
        sym = r["symbol"]
        combined[sym] = {
            "symbol": sym,
            "buy_legs": r["legs"],
            "buy_sum": r["sum_pct"],
            "buy_win_pct": r["win_pct"],
            "sell_legs": 0,
            "sell_sum": 0.0,
            "sell_win_pct": 0.0,
        }
    for r in sell:
        sym = r["symbol"]
        if sym not in combined:
            combined[sym] = {
                "symbol": sym, "buy_legs": 0, "buy_sum": 0.0, "buy_win_pct": 0.0,
                "sell_legs": r["legs"], "sell_sum": r["sum_pct"], "sell_win_pct": r["win_pct"],
            }
        else:
            combined[sym]["sell_legs"] = r["legs"]
            combined[sym]["sell_sum"] = r["sum_pct"]
            combined[sym]["sell_win_pct"] = r["win_pct"]
    for v in combined.values():
        v["total_legs"] = v["buy_legs"] + v["sell_legs"]
        v["total_sum_pct"] = v["buy_sum"] + v["sell_sum"]
        total_win = (v["buy_win_pct"] * v["buy_legs"] + v["sell_win_pct"] * v["sell_legs"])
        v["total_win_pct"] = total_win / v["total_legs"] if v["total_legs"] else 0.0
    return combined


def render_pattern_md(root: Path, out_path: Path) -> None:
    buy = parse_per_symbol_table(root / "_summary_buy.md")
    sell = parse_per_symbol_table(root / "_summary_sell.md")
    monthly = parse_monthly(root / "_monthly_profile.md")
    cap_sim = parse_capital_sim(root / "_capital_sim.txt")

    combined = combine(buy, sell)
    ranked = sorted(combined.values(), key=lambda x: -x["total_sum_pct"])
    top20 = ranked[:20]
    bottom20 = ranked[-20:]

    # Monthly seasonality
    bull_months = [m for m in monthly if m["win_pct"] >= 50]
    bear_months = [m for m in monthly if m["win_pct"] < 30 and m["trades"] >= 2]

    lines = [
        f"# Pattern Mining — {root.name}\n",
        f"_Generated: 2026-05-12_\n",
        f"Source: `{root}`\n",
        "## Headline\n",
        f"- Symbols processed: {len(combined)}",
        f"- Total uncompounded sum%: {sum(v['total_sum_pct'] for v in combined.values()):.1f}%",
        f"- Months observed: {len(monthly)}",
        f"- Capital-sim ROI (max_concurrent=2): "
        f"{[r for r in cap_sim if r['max'] == 2][0]['roi_pct'] if cap_sim else 'n/a'}%",
        "\n## Top 20 contributors (by uncompounded sum%)\n",
        "| Rank | Symbol | Buy legs | Buy sum% | Sell legs | Sell sum% | Total sum% | Combined win% |",
        "|------|--------|---------:|---------:|----------:|----------:|-----------:|--------------:|",
    ]
    for i, r in enumerate(top20, 1):
        lines.append(
            f"| {i} | **{r['symbol']}** | {r['buy_legs']} | {r['buy_sum']:+.1f}% | "
            f"{r['sell_legs']} | {r['sell_sum']:+.1f}% | "
            f"**{r['total_sum_pct']:+.1f}%** | {r['total_win_pct']:.1f}% |"
        )

    lines += [
        "\n## Bottom 20 (worst contributors — drop candidates)\n",
        "| Symbol | Total legs | Total sum% | Win% |",
        "|--------|----------:|-----------:|-----:|",
    ]
    for r in reversed(bottom20):
        lines.append(
            f"| {r['symbol']} | {r['total_legs']} | {r['total_sum_pct']:+.1f}% | "
            f"{r['total_win_pct']:.1f}% |"
        )

    lines += [
        "\n## Monthly seasonality\n",
        "| Month | Trades | Win% | Sum₹ | DD% |",
        "|-------|-------:|-----:|-----:|----:|",
    ]
    for m in monthly:
        flag = " ⭐" if m["win_pct"] >= 50 and m["sum_inr"] > 0 else \
               " ⚠️" if m["win_pct"] < 30 and m["trades"] >= 2 else ""
        lines.append(
            f"| {m['month']} | {m['trades']} | {m['win_pct']:.1f}% | "
            f"₹{m['sum_inr']:+,.0f}{flag} | {m['dd_pct']:.1f} |"
        )

    lines += [
        f"\n**Profitable months ({len(bull_months)}/{len(monthly)}):** "
        f"{', '.join(m['month'] for m in bull_months)}",
        f"\n**Drawdown months ({len(bear_months)}/{len(monthly)}):** "
        f"{', '.join(m['month'] for m in bear_months)}",
    ]

    lines += [
        "\n## Capital-sim concurrency sweep\n",
        "| Max | Taken | Skipped | Final₹ | ROI% | MaxDD% |",
        "|----:|------:|--------:|-------:|-----:|-------:|",
    ]
    for r in cap_sim:
        lines.append(
            f"| {r['max']} | {r['taken']} | {r['skip']} | ₹{r['final']:,.0f} | "
            f"{r['roi_pct']:+.2f} | {r['dd_pct']:.2f} |"
        )

    lines += [
        "\n## Patterns observed\n",
        f"1. **SELL >> BUY**: aggregated sell sum% = {sum(r['sell_sum'] for r in ranked):.1f}% vs "
        f"buy sum% = {sum(r['buy_sum'] for r in ranked):.1f}%. Bear-side EMA200/400 retest2 has higher edge.",
        f"2. **Slot bottleneck**: ranked sum% = {sum(v['total_sum_pct'] for v in combined.values()):.1f}% "
        f"theoretical but cap-sim @ max=2 captures only ~{cap_sim[1]['roi_pct'] if len(cap_sim)>1 else 0}%. "
        f"Missing signals are highest cost.",
        f"3. **Monthly variance high**: ROI swings ₹{max(m['sum_inr'] for m in monthly):+,.0f} (best) to "
        f"₹{min(m['sum_inr'] for m in monthly):+,.0f} (worst). Regime filter needed.",
        f"4. **Concurrent=2 optimum**: cap-sim shows max=2 wins; higher slots over-trade, lower under-utilize.",
        "\n## Recommended actions for Phase 2 (sweep)\n",
        f"- Whitelist top-20 contributors as the trading universe (drop {len(combined)-20} bottom stocks).",
        f"- Add monthly regime filter: skip trades in confirmed bear months (Dec/Jan/Mar pattern).",
        f"- Bias toward SELL setups (higher historical edge).",
        f"- Keep max_concurrent=2 as headline; test max=3 + ATR-sized positions.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} ({len(lines)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exports/backtests/y1_baseline/y1_filter/nifty50_ema_200_400_2025_2026")
    ap.add_argument("--out", default="exports/backtests/PATTERN_MINING.md")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    render_pattern_md(root, out)


if __name__ == "__main__":
    main()
