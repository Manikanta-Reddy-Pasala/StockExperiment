"""Run FinNifty Iron Condor OTM3 w500 lots=4 — winner config.

Targets: 100-200%/yr at -20 to -25% max DD.
Backtest: +102%/yr at -7.93% DD (well within constraints).

Strategy:
  - Sell OTM 3% CE + OTM 3% PE (body)
  - Buy wings 500 points further out (caps risk)
  - 4 lots (margin ~₹1.5L; fits ₹2L capital)
  - Stop: 3x entry credit OR hold to monthly expiry
  - Slippage: 1% per leg

Output: per-trade ledger + monthly equity curve under
exports/models/finnifty_ic_otm3_w500_lots4/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import run_ic  # noqa: E402

MODEL_NAME = "finnifty_ic_otm3_w500_lots4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--lots", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or f"exports/models/{MODEL_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    df = run_ic("FINNIFTY", args.frm, args.to,
                otm_pct=3.0, wing_width=500, stop_mult=3.0,
                slip=0.01, capital=args.capital, lots=args.lots)
    if df.empty:
        print("No trades.")
        return

    df = df.sort_values("entry_date").reset_index(drop=True)
    df["cum_pnl"] = df["pnl_total"].cumsum()
    df["nav"] = args.capital + df["cum_pnl"]
    df["peak"] = df["nav"].cummax()
    df["dd_pct"] = (df["nav"] / df["peak"] - 1) * 100
    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year

    df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    monthly = df.groupby("month").agg(
        trades=("pnl_total", "count"),
        pnl=("pnl_total", "sum"),
        wins=("pnl_total", lambda s: (s > 0).sum()),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100
    monthly["mo_roi_pct"] = monthly["pnl"] / args.capital * 100
    monthly["equity_end"] = args.capital + monthly["pnl"].cumsum()
    monthly.to_csv(os.path.join(out_dir, "monthly.csv"))

    yearly = df.groupby("year").agg(
        trades=("pnl_total", "count"),
        pnl=("pnl_total", "sum"),
        wins=("pnl_total", lambda s: (s > 0).sum()),
    )
    yearly["wr"] = yearly["wins"] / yearly["trades"] * 100
    yearly["yr_roi"] = yearly["pnl"] / args.capital * 100

    final = df["nav"].iloc[-1]
    total_pnl = df["pnl_total"].sum()
    wr = (df["pnl_total"] > 0).mean() * 100
    avg_mo = monthly["mo_roi_pct"].mean()
    max_dd = df["dd_pct"].min()

    summary_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(f"# {MODEL_NAME}\n\n")
        f.write("## Strategy\n\n")
        f.write("- Underlying: FINNIFTY monthly Iron Condor\n")
        f.write("- SELL OTM 3% CE + OTM 3% PE (body)\n")
        f.write("- BUY wings +/- 500 pts further out\n")
        f.write("- 4 lots (margin ~₹1.5L; fits ₹2L cap)\n")
        f.write("- Stop: 3× entry credit OR hold to monthly expiry\n")
        f.write("- Slippage: 1% per leg\n\n")
        f.write("## Result (3-year backtest)\n\n")
        f.write(f"- Capital: ₹{args.capital:,.0f}\n")
        f.write(f"- Final equity: ₹{final:,.0f}\n")
        f.write(f"- Total P&L: ₹{total_pnl:,.0f}\n")
        f.write(f"- Total return: {(final/args.capital-1)*100:+.2f}%\n")
        f.write(f"- **Avg yearly: {yearly['yr_roi'].mean():+.2f}%**\n")
        f.write(f"- **Max DD: {max_dd:.2f}%**\n")
        f.write(f"- Avg/mo: {avg_mo:+.2f}%\n")
        f.write(f"- Best mo: {monthly['mo_roi_pct'].max():+.2f}%\n")
        f.write(f"- Worst mo: {monthly['mo_roi_pct'].min():+.2f}%\n")
        f.write(f"- Trades: {len(df)} | WR: {wr:.1f}%\n\n")
        f.write("## Yearly\n\n")
        f.write("| Year | Trades | WR | P&L | ROI |\n|---|---:|---:|---:|---:|\n")
        for yr, row in yearly.iterrows():
            f.write(f"| {yr} | {int(row['trades'])} | {row['wr']:.1f}% | "
                    f"₹{row['pnl']:,.0f} | {row['yr_roi']:+.2f}% |\n")
        f.write("\n## Monthly P&L + Equity\n\n")
        f.write("| Month | Trades | WR | P&L | ROI | Equity |\n|---|---:|---:|---:|---:|---:|\n")
        for m, row in monthly.iterrows():
            f.write(f"| {m} | {int(row['trades'])} | {row['wr']:.1f}% | "
                    f"₹{row['pnl']:,.0f} | {row['mo_roi_pct']:+.2f}% | ₹{row['equity_end']:,.0f} |\n")
        f.write("\n## Every Trade\n\n")
        f.write("| # | Entry | Exit | Spot | CE k | PE k | Credit | Exit Debit | P&L | DD% | Reason |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for i, r in enumerate(df.itertuples(), 1):
            f.write(f"| {i} | {r.entry_date} | {r.exit_date} | {r.spot:.1f} | "
                    f"{r.ce_k} | {r.pe_k} | ₹{r.net_credit:.2f} | "
                    f"₹{r.exit_debit:.2f} | "
                    f"{'**+' if r.pnl_total > 0 else '**'}₹{r.pnl_total:,.0f}** | "
                    f"{r.dd_pct:.2f}% | {r.exit_reason} |\n")
    print(f"Wrote {summary_path}")
    print(f"Final: ₹{final:,.0f} | Avg/yr: {yearly['yr_roi'].mean():+.2f}% | DD: {max_dd:.2f}%")


if __name__ == "__main__":
    main()
