"""Final FORWARD-applicable winner: FinNifty Monthly Iron Condor OTM4 w300 lots=5.

Defined risk Iron Condor:
  - Sell OTM 4% CE + OTM 4% PE (short strangle body)
  - Buy CE/PE wings 300 points further (caps max loss)
  - Monthly expiry (still trades post Nov 2024 SEBI changes)
  - 5 lots scaled — max loss bounded by wing width

Output: per-trade ledger + monthly equity curve under
exports/options/FINNIFTY_monthly_IC_OTM4_w300_lots5/
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import run_ic  # noqa: E402


MODEL_NAME = "FINNIFTY_monthly_IC_OTM4_w300_lots5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or f"/app/logs/{MODEL_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    df = run_ic(
        underlying="FINNIFTY",
        start=args.frm, end=args.to,
        otm_pct=4.0,
        wing_width=300,
        stop_mult=3.0,
        slip=0.01,
        capital=args.capital,
        lots=5,
    )
    if df.empty:
        print("No trades.")
        return

    df = df.sort_values("entry_date").reset_index(drop=True)
    df["running_pnl"] = df["pnl_total"].cumsum()
    df["equity"] = args.capital + df["running_pnl"]
    df["roi_pct"] = (df["equity"] / args.capital - 1) * 100

    df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
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

    summary_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        final_eq = df["equity"].iloc[-1]
        total = df["pnl_total"].sum()
        wr_total = (df["pnl_total"] > 0).mean() * 100
        avg_mo = monthly["mo_roi_pct"].mean()
        best_mo = monthly["mo_roi_pct"].max()
        worst_mo = monthly["mo_roi_pct"].min()
        twenty_plus = (monthly["mo_roi_pct"] >= 20).sum()
        thirty_plus = (monthly["mo_roi_pct"] >= 30).sum()
        max_drawdown_per_trade = df["max_loss_total"].max()

        f.write(f"# {MODEL_NAME}\n\n")
        f.write("## Strategy\n\n")
        f.write("- **Underlying:** FINNIFTY (Nifty Financial Services Index)\n")
        f.write("- **Setup:** Iron Condor monthly expiry\n")
        f.write("  - SELL OTM 4% CE + OTM 4% PE (body)\n")
        f.write("  - BUY wings +300 points further out (cap risk)\n")
        f.write("- **Position size:** 5 lots\n")
        f.write("- **Stop:** 3× entry credit OR hold to expiry\n")
        f.write("- **Capital:** ₹{:,.0f}\n".format(args.capital))
        f.write("- **Window:** {} .. {}\n".format(args.frm, args.to))
        f.write("- **Slippage:** 1% per leg\n")
        f.write("- **Forward applicable:** YES (FinNifty monthly still trades)\n\n")

        f.write("## Final Result\n\n")
        f.write(f"- **Started with:** ₹{args.capital:,.0f}\n")
        f.write(f"- **Ended with:** ₹{final_eq:,.0f}\n")
        f.write(f"- **Total profit:** ₹{total:,.0f}\n")
        f.write(f"- **Total return:** {(final_eq/args.capital-1)*100:+.2f}%\n")
        f.write(f"- **Trades:** {len(df)}\n")
        f.write(f"- **Win rate:** {wr_total:.1f}%\n")
        f.write(f"- **Months tracked:** {monthly.shape[0]}\n")
        f.write(f"- **Avg/mo:** {avg_mo:+.2f}%\n")
        f.write(f"- **Best mo:** {best_mo:+.1f}%\n")
        f.write(f"- **Worst mo:** {worst_mo:+.1f}%\n")
        f.write(f"- **Months ≥20%:** {twenty_plus}/{monthly.shape[0]} ({twenty_plus/monthly.shape[0]*100:.0f}%)\n")
        f.write(f"- **Months ≥30%:** {thirty_plus}/{monthly.shape[0]}\n")
        f.write(f"- **Max single-trade loss:** ₹{max_drawdown_per_trade:,.0f} "
                f"({max_drawdown_per_trade/args.capital*100:.1f}% of capital)\n\n")

        f.write("## Yearly\n\n")
        f.write("| Year | Trades | Wins | WR | P&L | ROI on ₹2L |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for yr, row in yearly.iterrows():
            f.write(f"| {yr} | {int(row['trades'])} | {int(row['wins'])} | "
                    f"{row['wr']:.1f}% | ₹{row['pnl']:,.0f} | "
                    f"{row['yr_roi']:+.2f}% |\n")

        f.write("\n## Monthly P&L + Equity\n\n")
        f.write("| Month | Trades | Wins | WR | P&L | ROI | Equity end-of-month |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for m, row in monthly.iterrows():
            f.write(f"| {m} | {int(row['trades'])} | {int(row['wins'])} | "
                    f"{row['wr']:.1f}% | ₹{row['pnl']:,.0f} | "
                    f"{row['mo_roi_pct']:+.2f}% | ₹{row['equity_end']:,.0f} |\n")

        f.write("\n## Every Trade\n\n")
        f.write("| # | Entry | Exit | Spot | CE k | PE k | Wing CE | Wing PE | "
                "Credit | Exit Debit | P&L | Reason | Running |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for i, r in enumerate(df.itertuples(), 1):
            f.write(f"| {i} | {r.entry_date} | {r.exit_date} | {r.spot:.1f} | "
                    f"{r.ce_k} | {r.pe_k} | {r.wce_k} | {r.wpe_k} | "
                    f"₹{r.net_credit:.2f} | ₹{r.exit_debit:.2f} | "
                    f"{'**+' if r.pnl_total > 0 else '**'}₹{r.pnl_total:,.0f}** | "
                    f"{r.exit_reason} | ₹{r.equity:,.0f} |\n")
    print(f"Wrote {summary_path}")
    print(f"All artifacts under {out_dir}/")


if __name__ == "__main__":
    main()
