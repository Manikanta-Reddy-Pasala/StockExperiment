"""Regenerate MONTHLY_INVESTED.md from trades.csv.

Reads trades.csv (output of run_winner.py) and produces a per-month
markdown ledger with trades, WR, margin, premium, worst-case loss,
realized P&L, ROI, end-of-month equity.

Idempotent — rerun whenever lot history or backtest params change.
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict, OrderedDict


def main(trades_csv: str, out_path: str,
         start_equity: float = 200_000, wing_width: int = 300):
    rows = []
    with open(trades_csv) as f:
        for r in csv.DictReader(f):
            r["entry_date"] = r["entry_date"]
            r["lot"] = int(r["lot"])
            r["lots"] = int(r["lots"])
            r["net_credit"] = float(r["net_credit"])
            r["pnl_total"] = float(r["pnl_total"])
            r["month"] = r["month"]
            rows.append(r)
    rows.sort(key=lambda x: x["entry_date"])

    monthly = OrderedDict()
    for r in rows:
        m = r["month"]
        if m not in monthly:
            monthly[m] = {"trades": [], "wins": 0}
        monthly[m]["trades"].append(r)
        if r["pnl_total"] > 0:
            monthly[m]["wins"] += 1

    md = []
    md.append("# finnifty_ic_otm4_w300_lots5 — Monthly Performance with Capital Deployed")
    md.append("")
    md.append("## What each column means")
    md.append("")
    md.append("| Column | Definition |")
    md.append("|---|---|")
    md.append("| **Trades** | Number of Iron Condor cycles opened that month |")
    md.append("| **Margin Locked** | Defined-risk capital = wing_width × lot_size × num_lots, summed across cycles |")
    md.append("| **Premium Collected** | Net credit on entry = (CE_short + PE_short) − (CE_wing + PE_wing) × lot × lots |")
    md.append("| **Worst-case Loss** | Margin Locked − Premium Collected (hard cap) |")
    md.append("| **Realized P&L** | Actual P&L when IC closed (stop or expiry) |")
    md.append("| **ROI on Equity** | Month P&L / equity at start of month |")
    md.append("| **End-of-Month Equity** | Cumulative NAV after this month's trades |")
    md.append("")
    md.append("Lot history (FinNifty): 40 → 65 (Sep 2024) → 60 (2026 SEBI). "
              "Margin scales accordingly per trade.")
    md.append("")
    md.append("## Monthly ledger")
    md.append("")
    md.append("| Month | Trades | WR | Margin Locked ₹ | Premium Collected ₹ | "
              "Worst-case Loss ₹ | Realized P&L ₹ | ROI on Equity | End-of-Month Equity ₹ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    equity = start_equity
    total_margin = 0.0
    total_premium = 0.0
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    for m, data in monthly.items():
        ts = data["trades"]
        n = len(ts)
        wins = data["wins"]
        wr = (wins / n * 100) if n else 0
        margin = sum(wing_width * t["lot"] * t["lots"] for t in ts)
        premium = sum(t["net_credit"] * t["lot"] * t["lots"] for t in ts)
        wc_loss = margin - premium
        pnl = sum(t["pnl_total"] for t in ts)
        roi_pct = (pnl / equity * 100) if equity > 0 else 0
        start_eq = equity
        equity += pnl
        md.append(f"| {m} | {n} | {wr:.0f}% | ₹{margin:,.0f} | ₹{premium:,.0f} | "
                  f"₹{wc_loss:,.0f} | ₹{pnl:+,.0f} | {roi_pct:+.2f}% | ₹{equity:,.0f} |")
        total_margin += margin
        total_premium += premium
        total_pnl += pnl
        total_trades += n
        total_wins += wins

    md.append("")
    md.append("## Headline summary")
    md.append("")
    md.append(f"- **Total cycles:** {total_trades}")
    md.append(f"- **Win rate:** {(total_wins/total_trades*100):.1f}% "
              f"({total_wins}W / {total_trades - total_wins}L)")
    md.append(f"- **Cumulative margin deployed:** ₹{total_margin:,.0f}")
    md.append(f"- **Cumulative premium collected:** ₹{total_premium:,.0f}")
    md.append(f"- **Total realized P&L:** ₹{total_pnl:+,.0f}")
    md.append(f"- **Start NAV:** ₹{start_equity:,.0f} → **End NAV:** ₹{equity:,.0f}")
    md.append(f"- **Return on starting equity:** {(total_pnl/start_equity*100):+.2f}%")

    Path(out_path).write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import sys
    trades = sys.argv[1] if len(sys.argv) > 1 else "/app/logs/FINNIFTY_monthly_IC_OTM4_w300_lots5/trades.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/app/logs/FINNIFTY_monthly_IC_OTM4_w300_lots5/MONTHLY_INVESTED.md"
    main(trades, out)
