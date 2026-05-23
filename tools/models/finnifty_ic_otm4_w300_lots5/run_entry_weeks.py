"""Run FinNifty IC backtest 3 ways — entry on first weekday of week 1,
week 2, or week 3 of each monthly cycle. For each variant, dump:

  exports/models/finnifty_ic_otm4_w300_lots5/week<N>_trades.csv
  exports/models/finnifty_ic_otm4_w300_lots5/week<N>_daily_volumes.csv

`daily_volumes.csv` shows the per-leg traded volume on every day the
position was open — caller inspects to see whether real liquidity
existed for the 4 strikes throughout the trade.

Usage (VM):
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.run_entry_weeks
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

# Current live params (do not change — single source of truth).
PARAMS = dict(
    underlying="FINNIFTY",
    start="2023-05-15",
    end="2026-05-15",
    otm_pct=2.0,
    wing_width=150,
    stop_mult=3.0,
    slip=0.01,
    capital=200_000,
    lots=5,
    realistic_slip=True,
)


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []
    for week in (1, 2, 3):
        daily_volumes: List[Dict] = []
        df = run_ic(**PARAMS, entry_week=week, daily_volumes=daily_volumes)
        if df.empty:
            print(f"week{week}: no trades")
            continue
        df.to_csv(EXPORTS / f"week{week}_trades.csv", index=False)
        dv = pd.DataFrame(daily_volumes) if daily_volumes else pd.DataFrame()
        dv.to_csv(EXPORTS / f"week{week}_daily_volumes.csv", index=False)
        total = float(df["pnl_total"].sum())
        wins = int((df["pnl_total"] > 0).sum())
        n = len(df)
        avg_vol = (dv.groupby("leg")["volume"].mean().round(0).to_dict()
                   if not dv.empty else {})
        # How often did a leg have zero volume on a held day?
        zero_pct = ((dv["volume"] == 0).mean() * 100) if not dv.empty else 0
        summary.append({
            "entry_week": week, "trades": n, "wins": wins,
            "wr_pct": round(wins / n * 100, 1),
            "total_pnl": round(total, 2),
            "total_return_pct": round(total / PARAMS["capital"] * 100, 2),
            "avg_volume_per_leg": avg_vol,
            "pct_held_days_zero_volume": round(zero_pct, 2),
        })
        print(f"week{week}: {n} trades, WR {wins/n*100:.1f}%, "
              f"₹{total:+,.0f} ({total/PARAMS['capital']*100:+.1f}%), "
              f"zero-vol days = {zero_pct:.1f}%")

    (EXPORTS / "entry_weeks_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved → {EXPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
