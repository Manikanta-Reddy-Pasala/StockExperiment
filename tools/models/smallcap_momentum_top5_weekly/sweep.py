"""Sweep config variants for smallcap momentum model.

Try monthly vs weekly, different lookbacks, different top/max, different
universe slices (large→mid→small).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.smallcap_momentum_top5_weekly.backtest import run  # noqa: E402


def rebalance_dates_monthly(start, end, idx) -> list:
    """First trading day of each month."""
    out = []
    seen = set()
    for d in idx:
        key = (d.year, d.month)
        if key not in seen and d.date() >= start and d.date() <= end:
            out.append(d.date())
            seen.add(key)
    return out


VARIANTS = [
    # name, top, max_conc, lookback
    {"name": "weekly_top5_max3_lb30",  "top": 5, "max": 3, "lb": 30},
    {"name": "weekly_top5_max3_lb60",  "top": 5, "max": 3, "lb": 60},
    {"name": "weekly_top10_max5_lb60", "top": 10, "max": 5, "lb": 60},
    {"name": "weekly_top3_max3_lb60",  "top": 3, "max": 3, "lb": 60},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--out", default="/app/logs/smallcap_sweep.md")
    args = ap.parse_args()

    rows = []
    for v in VARIANTS:
        print(f">>> {v['name']}", flush=True)
        try:
            r = run(args.universe_file, args.frm, args.to,
                    v["top"], v["max"], v["lb"], args.capital,
                    10.0, 20.0, 0.1)
            m = r["monthly"]["ret_pct"]
            y = r["yearly"]["ret_pct"]
            rows.append({
                "name": v["name"],
                "final": r["final"],
                "yearly": y.mean(),
                "avg_mo": m.mean(),
                "best_mo": m.max(),
                "worst_mo": m.min(),
                "max_dd": r["max_dd"],
                "twenty_plus": int((m >= 20).sum()),
                "thirty_plus": int((m >= 30).sum()),
                "below_10": int((m < -10).sum()),
                "months": int(m.count()),
                "fees": r["total_fees"],
                "trades": len(r["trades"]),
            })
            print(f"  yr={y.mean():+.1f}% mo={m.mean():+.2f}% "
                  f"dd={r['max_dd']:.1f}% 20+={int((m>=20).sum())}/{int(m.count())}",
                  flush=True)
        except Exception as e:
            print(f"  ERR: {e}", flush=True)

    rows.sort(key=lambda r: -r["yearly"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Smallcap Momentum Sweep — NET of costs\n\n")
        f.write(f"Capital ₹{args.capital:,.0f} | Window {args.frm}..{args.to}\n")
        f.write(f"Slip=10bps, brokerage=₹20/trade, STT=0.1%\n\n")
        f.write("| Variant | Final | Yr | Avg/mo | Best/mo | Worst/mo | Max DD | 20%+ | 30%+ | Trades | Fees |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | ₹{r['final']:,.0f} | "
                    f"{r['yearly']:+.1f}% | {r['avg_mo']:+.2f}% | "
                    f"{r['best_mo']:+.1f}% | {r['worst_mo']:+.1f}% | "
                    f"{r['max_dd']:+.1f}% | {r['twenty_plus']}/{r['months']} | "
                    f"{r['thirty_plus']}/{r['months']} | {r['trades']} | "
                    f"₹{r['fees']:,.0f} |\n")
    print(f"\nReport: {args.out}", flush=True)


if __name__ == "__main__":
    main()
