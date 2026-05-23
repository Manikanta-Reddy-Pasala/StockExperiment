"""Run compare_underlyings across multiple capital levels and emit one
consolidated table (3 underlyings × N variants × M capitals).

Now uses the patched run_ic that iterates EVERY trading day in a monthly
cycle (not just Monday). If Monday's wings have no volume, the backtest
falls through to Tuesday / Wednesday / Thursday / Friday until a day
with fillable legs is found. Only when no weekday in the entire monthly
cycle clears the volume filter does the cycle get skipped.

Usage (on the VM):
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.grid_run \
        --capitals 200000,500000,1000000 --min-leg-volume 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.compare_underlyings import (
    VARIANTS, UNDERLYINGS, _run_one,
)


def _row_key(r: Dict) -> str:
    return f"{r['underlying']}/{r['variant']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capitals", default="200000,500000,1000000")
    ap.add_argument("--start", default="2023-05-15")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--min-leg-volume", type=int, default=100)
    args = ap.parse_args()
    capitals = [int(x) for x in args.capitals.split(",")]

    print(f"=== Grid: {len(UNDERLYINGS)} underlyings × {len(VARIANTS)} variants × {len(capitals)} capitals ===")
    print(f"  window  = {args.start} → {args.end}")
    print(f"  min_vol = {args.min_leg_volume}")
    print(f"  entry   = every weekday in cycle (Mon/Tue/Wed/Thu/Fri fallthrough)")
    print()

    # Run once per (underlying × variant × capital). Reuse code from
    # compare_underlyings._run_one — already handles peak-safe lots.
    by_combo: Dict[str, Dict[int, Dict]] = {}
    for cap in capitals:
        print(f"\n--- capital ₹{cap:,} ---")
        for u in UNDERLYINGS:
            for v in VARIANTS:
                row = _run_one(u, v, args.start, args.end, cap,
                               args.min_leg_volume)
                k = _row_key(row)
                by_combo.setdefault(k, {})[cap] = row

    # Print one row per underlying/variant, columns = capitals.
    print()
    cap_hdr = "  ".join(f"{c/100000:.0f}L: lots/CAGR%/DD%" for c in capitals)
    print(f"{'underlying/variant':25} {'trades':>6} {'rej%':>5}  {cap_hdr}")
    print("-" * (38 + len(cap_hdr)))

    # Sort by best-capital CAGR across the row.
    def best_cagr(rec: Dict[int, Dict]) -> float:
        return max((r.get("cagr", -999) for r in rec.values()
                    if not r.get("skipped")), default=-999)
    sorted_keys = sorted(by_combo.keys(), key=lambda k: best_cagr(by_combo[k]),
                         reverse=True)
    for k in sorted_keys:
        rec = by_combo[k]
        first = next(iter(rec.values()))
        trades = first.get("trades", first.get("n_unfiltered", "—"))
        rej = first.get("vol_reject_pct", 0)
        cells = []
        for c in capitals:
            r = rec.get(c, {})
            if r.get("skipped") or not r.get("tradeable"):
                cells.append(f"{'—':>20}")
            else:
                cells.append(
                    f" {r['lots']:>2}L/{r['cagr']:>+5.1f}/{r['max_dd_pct']:>+5.1f}")
        print(f"{k:25} {str(trades):>6} {rej:>5.1f}  {'  '.join(cells)}")

    # JSON dump for downstream tooling.
    dump = {k: {str(c): {kk: vv for kk, vv in v.items()
                         if kk != "_trades_df"}
                for c, v in rec.items()}
            for k, rec in by_combo.items()}
    out = Path("/tmp") / f"ic_grid_minvol{args.min_leg_volume}.json"
    out.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nJSON dump: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
