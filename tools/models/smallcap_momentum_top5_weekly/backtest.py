"""Backtest: weekly momentum rotation on pseudo Smallcap-250.

Strategy:
  - Universe: smallcap ~250 (built via build_universe.py, ADV-ranked rows 50-250)
  - Signal: 30-day return (shorter lookback than N100 model, faster rotation)
  - Pick top-5 by ranking, hold max 3 equal-weight
  - Rebalance: weekly (every Monday)
  - Realistic costs: 0.10% slip + ₹20 brokerage + 0.10% STT (sell side)

Output: equity curve + monthly + yearly + drawdown + per-trade ledger.

Usage:
  python tools/models/smallcap_momentum_top5_weekly/backtest.py \
    --universe-file /app/logs/momrot/universes/smallcap_current.json \
    --from 2023-05-15 --to 2026-05-15 \
    --capital 200000 --top 5 --max-conc 3 --lookback 30 \
    --out exports/models/smallcap_momentum_top5_weekly/run_$(date +%F).md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

log = logging.getLogger("smallcap_backtest")


def load_universe(universe_file: str) -> List[str]:
    """Return list of Fyers symbols (NSE:XYZ-EQ) from selector JSON."""
    with open(universe_file) as f:
        data = json.load(f)
    plain = [s["symbol"] for s in data.get("stocks", [])]
    return [f"NSE:{s}-EQ" if not s.startswith("NSE:") else s for s in plain]


def load_daily(symbols: List[str], start, end) -> pd.DataFrame:
    eng = _get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(
            "SELECT symbol, date, close FROM historical_data "
            "WHERE symbol = ANY(:syms) AND date BETWEEN :a AND :b "
            "ORDER BY date"
        ), conn, params={"syms": symbols, "a": start, "b": end})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="symbol", values="close",
                          aggfunc="last").sort_index()


def rebalance_dates_weekly(start, end, idx) -> list:
    """First trading day of each week (Monday or first available after)."""
    out = []
    seen_weeks = set()
    for d in idx:
        wk = (d.year, d.isocalendar().week)
        if wk not in seen_weeks and d.date() >= start and d.date() <= end:
            out.append(d.date())
            seen_weeks.add(wk)
    return out


def run(universe_file: str, start_str: str, end_str: str,
        top_n: int, max_conc: int, lookback: int, capital: float,
        slip_bps: float, brokerage: float, stt_pct: float) -> dict:
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    syms = load_universe(universe_file)
    log.info(f"Universe: {len(syms)} symbols")
    prices = load_daily(syms, start - timedelta(days=lookback + 30), end)
    log.info(f"Price grid: {prices.shape}")

    reb_dates = rebalance_dates_weekly(start, end, prices.index)
    log.info(f"Rebalance count: {len(reb_dates)}")
    reb_set = set(reb_dates)

    equity = capital
    holdings: Dict[str, float] = {}
    daily = []
    prev_prices: Dict[str, float] = {}
    total_fees = 0.0
    trades = []
    last_holdings = set()

    slip = slip_bps / 10000.0
    stt = stt_pct / 100.0

    all_dates = prices.loc[pd.Timestamp(start):pd.Timestamp(end)].index
    for d in all_dates:
        dts = d.date()
        # mark-to-market existing holdings
        if holdings:
            val = 0.0
            for sym, qty in holdings.items():
                px = prices.at[d, sym]
                if pd.isna(px):
                    px = prev_prices.get(sym, 0)
                val += qty * px
            equity = val

        if dts in reb_set:
            past = d - pd.Timedelta(days=lookback)
            if past >= prices.index[0]:
                snap = prices.loc[past:d]
                if len(snap) >= 2:
                    rets = (snap.iloc[-1] / snap.iloc[0]) - 1
                    rets = rets.dropna()
                    top = rets.sort_values(ascending=False).head(top_n).index.tolist()
                    top = top[:max_conc]
                    if top:
                        # Sell side fees
                        sell_val = sum(qty * prices.at[d, sym]
                                       for sym, qty in holdings.items()
                                       if not pd.isna(prices.at[d, sym]))
                        sell_fees = sell_val * slip + brokerage * len(holdings) \
                                    + sell_val * stt
                        buy_fees = equity * slip + brokerage * len(top)
                        fees = sell_fees + buy_fees
                        total_fees += fees

                        # Log exits
                        for sym in holdings:
                            if sym not in top:
                                exit_px = prices.at[d, sym]
                                if pd.isna(exit_px):
                                    exit_px = prev_prices.get(sym, 0)
                                trades.append({
                                    "date": dts.isoformat(), "action": "EXIT",
                                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                                    "price": round(float(exit_px), 2),
                                })
                        # Log entries
                        for sym in top:
                            if sym not in holdings:
                                entry_px = prices.at[d, sym]
                                trades.append({
                                    "date": dts.isoformat(), "action": "ENTRY",
                                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                                    "price": round(float(entry_px), 2),
                                })

                        equity -= fees
                        each = equity / len(top)
                        new_holdings = {}
                        for sym in top:
                            px = prices.at[d, sym]
                            if pd.isna(px) or px <= 0:
                                continue
                            new_holdings[sym] = each / px
                        holdings = new_holdings
                        equity = sum(qty * prices.at[d, sym]
                                      for sym, qty in holdings.items())

        for sym in list(holdings.keys()):
            px = prices.at[d, sym]
            if not pd.isna(px):
                prev_prices[sym] = px
        daily.append({"date": dts, "equity": equity})

    eq = pd.DataFrame(daily).set_index(pd.to_datetime([r["date"] for r in daily]))
    eq.index.name = "date"
    eq["month"] = eq.index.to_period("M").astype(str)
    eq["year"] = eq.index.year
    eq["peak"] = eq["equity"].cummax()
    eq["dd_pct"] = (eq["equity"] / eq["peak"] - 1) * 100

    monthly = eq.groupby("month")["equity"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    yearly = eq.groupby("year")["equity"].agg(["first", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100

    return {
        "equity": eq, "monthly": monthly, "yearly": yearly,
        "final": float(eq["equity"].iloc[-1]),
        "max_dd": float(eq["dd_pct"].min()),
        "total_fees": total_fees,
        "trades": trades,
    }


def summarize_and_write(r: dict, capital: float, label: str, out_md: Optional[str]):
    eq = r["equity"]
    m = r["monthly"]["ret_pct"]
    y = r["yearly"]["ret_pct"]
    final = r["final"]

    print(f"\n=== {label} ===")
    print(f"Final equity: ₹{final:,.0f} ({(final/capital-1)*100:+.1f}%)")
    print(f"Fees: ₹{r['total_fees']:,.0f}")
    print(f"Avg/mo: {m.mean():+.2f}%  Median: {m.median():+.2f}%")
    print(f"Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%")
    print(f"30%+ mo: {(m>=30).sum()}/{m.count()}  20%+: {(m>=20).sum()}/{m.count()}  <-10%: {(m<-10).sum()}")
    print(f"Avg yearly: {y.mean():+.2f}%")
    print(f"Max DD: {r['max_dd']:.2f}%")
    print(f"Trades: {len(r['trades'])}")

    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        lines = []
        lines.append(f"# {label}\n")
        lines.append(f"- Capital: ₹{capital:,.0f}")
        lines.append(f"- Final equity: ₹{final:,.0f}")
        lines.append(f"- Total return: {(final/capital-1)*100:+.2f}%")
        lines.append(f"- Avg/mo: {m.mean():+.2f}%  Median/mo: {m.median():+.2f}%")
        lines.append(f"- Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%")
        lines.append(f"- Months ≥20%: {(m>=20).sum()}/{m.count()}")
        lines.append(f"- Months ≥30%: {(m>=30).sum()}/{m.count()}")
        lines.append(f"- Avg yearly: {y.mean():+.2f}%")
        lines.append(f"- Max DD: {r['max_dd']:.2f}%")
        lines.append(f"- Total fees: ₹{r['total_fees']:,.0f}")
        lines.append(f"- Total trades: {len(r['trades'])}\n")
        lines.append("## Yearly\n")
        lines.append("| Year | First | Last | Return % |")
        lines.append("|---|---:|---:|---:|")
        for yr, row in r["yearly"].iterrows():
            lines.append(f"| {yr} | ₹{row['first']:,.0f} | ₹{row['last']:,.0f} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## Monthly\n")
        lines.append("| Month | ROI % |")
        lines.append("|---|---:|")
        for mo, row in r["monthly"].iterrows():
            lines.append(f"| {mo} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## Trades (first 50)\n")
        lines.append("| Date | Action | Symbol | Price |")
        lines.append("|---|---|---|---:|")
        for t in r["trades"][:50]:
            lines.append(f"| {t['date']} | {t['action']} | {t['symbol']} | ₹{t['price']} |")
        Path(out_md).write_text("\n".join(lines))
        print(f"\nReport: {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-conc", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--slip-bps", type=float, default=10.0)
    ap.add_argument("--brokerage", type=float, default=20.0)
    ap.add_argument("--stt-pct", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = run(args.universe_file, args.frm, args.to, args.top, args.max_conc,
            args.lookback, args.capital, args.slip_bps, args.brokerage, args.stt_pct)
    label = (f"smallcap_momentum_top{args.top}_max{args.max_conc}_lb{args.lookback}_weekly "
             f"({args.frm}..{args.to})")
    summarize_and_write(r, args.capital, label, args.out)


if __name__ == "__main__":
    main()
