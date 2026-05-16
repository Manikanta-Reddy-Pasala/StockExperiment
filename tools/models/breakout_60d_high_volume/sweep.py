"""Sweep config variants for breakout strategy."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.breakout_60d_high_volume.backtest import run  # noqa: E402


VARIANTS = [
    # name, breakout, vol_mult, sma_exit, trail, max_hold, max_conc
    {"name": "baseline_60d_v1.5_sma20_t8_max3",  "lb":60, "vm":1.5, "sma":20, "tr":0.08, "hold":90, "mc":3},
    {"name": "60d_v2.0_sma20_t8_max3",           "lb":60, "vm":2.0, "sma":20, "tr":0.08, "hold":90, "mc":3},
    {"name": "60d_v1.5_sma20_t10_max3",          "lb":60, "vm":1.5, "sma":20, "tr":0.10, "hold":90, "mc":3},
    {"name": "60d_v1.5_sma20_t12_max3",          "lb":60, "vm":1.5, "sma":20, "tr":0.12, "hold":90, "mc":3},
    {"name": "60d_v1.5_sma10_t8_max3",           "lb":60, "vm":1.5, "sma":10, "tr":0.08, "hold":90, "mc":3},
    {"name": "100d_v1.5_sma20_t8_max3",          "lb":100,"vm":1.5, "sma":20, "tr":0.08, "hold":120,"mc":3},
    {"name": "50d_v1.5_sma20_t8_max3",           "lb":50, "vm":1.5, "sma":20, "tr":0.08, "hold":60, "mc":3},
    {"name": "60d_v1.5_sma20_t8_max5",           "lb":60, "vm":1.5, "sma":20, "tr":0.08, "hold":90, "mc":5},
    {"name": "60d_v1.5_sma20_t8_max2",           "lb":60, "vm":1.5, "sma":20, "tr":0.08, "hold":90, "mc":2},
    {"name": "60d_v1.5_sma20_t8_max1",           "lb":60, "vm":1.5, "sma":20, "tr":0.08, "hold":90, "mc":1},
    {"name": "120d_v2.0_sma30_t10_max3",         "lb":120,"vm":2.0, "sma":30, "tr":0.10, "hold":120,"mc":3},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--out", default="/app/logs/breakout_sweep.md")
    args = ap.parse_args()

    rows = []
    for v in VARIANTS:
        print(f">>> {v['name']}", flush=True)
        try:
            r = run(args.universe_file, args.frm, args.to,
                    v["mc"], args.capital,
                    v["lb"], v["vm"], v["sma"], v["tr"], v["hold"],
                    10.0, 20.0, 0.1)
            m = r["monthly"]["ret_pct"]
            y = r["yearly"]["ret_pct"]
            wr = sum(1 for t in r["trades"] if t["pnl"] > 0) / max(1, len(r["trades"])) * 100
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
                "trades": len(r["trades"]),
                "wr": wr,
                "fees": r["total_fees"],
            })
            print(f"  yr={y.mean():+.1f}% mo={m.mean():+.2f}% best={m.max():.1f}% "
                  f"worst={m.min():.1f}% dd={r['max_dd']:.1f}% "
                  f"20+={int((m>=20).sum())}/{int(m.count())} "
                  f"trades={len(r['trades'])} wr={wr:.0f}%", flush=True)
        except Exception as e:
            print(f"  ERR: {e}", flush=True)
            import traceback; traceback.print_exc()

    rows.sort(key=lambda r: -r["yearly"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Breakout Strategy Sweep — NET of realistic costs\n\n")
        f.write(f"Capital ₹{args.capital:,.0f} | {args.frm}..{args.to}\n")
        f.write("Slip 10bps, brokerage ₹20/trade, STT 0.1%\n\n")
        f.write("| Variant | Final | Yr | Avg/mo | Best/mo | Worst/mo | Max DD | 20%+ | Trades | WR | Fees |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | ₹{r['final']:,.0f} | "
                    f"{r['yearly']:+.1f}% | {r['avg_mo']:+.2f}% | "
                    f"{r['best_mo']:+.1f}% | {r['worst_mo']:+.1f}% | "
                    f"{r['max_dd']:+.1f}% | {r['twenty_plus']}/{r['months']} | "
                    f"{r['trades']} | {r['wr']:.0f}% | ₹{r['fees']:,.0f} |\n")
    print(f"\nReport: {args.out}", flush=True)


if __name__ == "__main__":
    main()
