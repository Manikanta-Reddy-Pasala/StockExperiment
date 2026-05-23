"""Run FinNifty IC at week-2 entry across wing widths W150, W200, W300.

Same 4-leg entry-volume gate as run_entry_weeks. OTM body fixed at 4 %
(current live config). Just varies wing width.

Output: exports/models/finnifty_ic_otm4_w300_lots5/wing<N>_trades.csv +
wing<N>_daily_volumes.csv + wing_variants_summary.json.

Usage:
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.try_wing_variants
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import run_ic

EXPORTS = REPO_ROOT / "exports" / "models" / "finnifty_ic_otm4_w300_lots5"
# OTM × wing grid at week-2 entry, 4-leg-volume gate active.
COMBOS = [
    (2.0, 150), (2.0, 200), (2.0, 300),
    (3.0, 150), (3.0, 200), (3.0, 300),
    (4.0, 150), (4.0, 200), (4.0, 300),
]


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []
    capital = 200_000
    for otm, wing in COMBOS:
        daily_volumes: List[Dict] = []
        df = run_ic(
            underlying="FINNIFTY", start="2023-05-15", end="2026-05-15",
            otm_pct=otm, wing_width=wing, stop_mult=3.0, slip=0.01,
            capital=capital, lots=5, realistic_slip=True,
            entry_week=2, daily_volumes=daily_volumes,
        )
        tag = f"otm{int(otm*10)}_w{wing}"
        if df.empty:
            print(f"{tag}: no trades")
            continue
        df.to_csv(EXPORTS / f"{tag}_trades.csv", index=False)
        dv = pd.DataFrame(daily_volumes)
        dv.to_csv(EXPORTS / f"{tag}_daily_volumes.csv", index=False)
        total = float(df["pnl_total"].sum())
        n = len(df)
        wins = int((df["pnl_total"] > 0).sum())
        zero_pct = ((dv["volume"] == 0).mean() * 100) if not dv.empty else 0
        risky_pct = ((dv["our_share_of_traded"] > 0.10).mean() * 100
                     if not dv.empty
                     and "our_share_of_traded" in dv
                     and dv["our_share_of_traded"].notna().any()
                     else None)
        med_trade = (float(dv["avg_trade_inr"].median())
                     if not dv.empty
                     and "avg_trade_inr" in dv
                     and dv["avg_trade_inr"].notna().any()
                     else None)
        med_ntrd = (float(dv["num_trades"].median())
                    if not dv.empty
                    and "num_trades" in dv
                    and dv["num_trades"].notna().any()
                    else None)
        summary.append({
            "otm": otm, "wing": wing, "trades": n, "wins": wins,
            "wr_pct": round(wins / n * 100, 1) if n else 0,
            "total_pnl": round(total, 2),
            "total_return_pct": round(total / capital * 100, 2),
            "pct_held_days_zero_volume": round(zero_pct, 2),
            "pct_held_days_our_share_over_10pct_of_traded":
                round(risky_pct, 2) if risky_pct is not None else None,
            "median_avg_trade_size_inr":
                round(med_trade, 0) if med_trade else None,
            "median_num_trades_per_leg_day":
                int(med_ntrd) if med_ntrd else None,
        })
        risky_str = (f"risky={risky_pct:.1f}%" if risky_pct is not None
                     else "risky=n/a")
        print(f"{tag}: {n} trades, WR {wins/n*100:.1f}%, "
              f"₹{total:+,.0f} ({total/capital*100:+.2f}%), "
              f"zero-vol={zero_pct:.1f}%, {risky_str}")
    (EXPORTS / "wing_variants_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved → {EXPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
