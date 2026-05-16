"""Backtest: 60-day-high breakout with volume confirmation.

Different alpha source from relative-momentum rotation:
  - Enters on PRICE LEVEL break, not relative ranking
  - Exits on trend reversal (close below 20d SMA), not rotation

Strategy:
  - Universe: pseudo-N100 (top 100 by ADV from N500)
  - Signal: stock closes at 60-day high AND volume > 1.5× 20d avg volume
  - Entry: next trading day at open
  - Exit: close below 20d SMA OR -8% trailing stop OR position-age 90 days
  - Hold up to MAX_CONC positions (default 3)
  - Equal-weight allocation per position
  - Realistic costs: 10bps slip + ₹20 brokerage + 0.1% STT (sell side)

Usage:
  python tools/models/breakout_60d_high_volume/backtest.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --from 2023-05-15 --to 2026-05-15 --capital 200000 --max-conc 3 \
    --out exports/models/breakout_60d_high_volume/run_$(date +%F).md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

log = logging.getLogger("breakout_bt")


def load_universe(universe_file: str) -> List[str]:
    with open(universe_file) as f:
        data = json.load(f)
    plain = [s["symbol"] for s in data.get("stocks", [])]
    return [f"NSE:{s}-EQ" if not s.startswith("NSE:") else s for s in plain]


def load_daily_full(symbols: List[str], start, end) -> pd.DataFrame:
    eng = _get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM historical_data "
            "WHERE symbol = ANY(:syms) AND date BETWEEN :a AND :b "
            "ORDER BY symbol, date"
        ), conn, params={"syms": symbols, "a": start, "b": end})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def run(universe_file: str, start_str: str, end_str: str,
        max_conc: int, capital: float,
        breakout_lookback: int, vol_mult: float,
        sma_exit: int, trail_pct: float, max_hold_days: int,
        slip_bps: float, brokerage: float, stt_pct: float) -> dict:
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    syms = load_universe(universe_file)
    log.info(f"Universe: {len(syms)} symbols")

    # Pull enough history for lookback + warmup
    df_all = load_daily_full(syms, start - timedelta(days=breakout_lookback + 60), end)
    if df_all.empty:
        raise RuntimeError("no price data")

    # Precompute per-symbol indicators
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym, grp in df_all.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        g["high_lookback"] = g["close"].rolling(breakout_lookback).max()
        g["vol_avg"] = g["volume"].rolling(20).mean()
        g["sma_exit"] = g["close"].rolling(sma_exit).mean()
        sym_data[sym] = g

    # Build trading-day grid
    all_dates = sorted({d for df in sym_data.values() for d in df["date"]})
    trading_days = [d for d in all_dates if start <= d.date() <= end]
    log.info(f"Trading days: {len(trading_days)}")

    slip = slip_bps / 10000.0
    stt = stt_pct / 100.0

    cash = capital
    positions: Dict[str, dict] = {}  # symbol -> {qty, entry_price, entry_date, peak}
    daily = []
    trades = []
    total_fees = 0.0

    for d in trading_days:
        # 1. Mark-to-market open positions + update peaks
        nav = cash
        for sym, p in list(positions.items()):
            row = sym_data[sym][sym_data[sym]["date"] == d]
            if row.empty:
                nav += p["qty"] * p["last_close"]
                continue
            close = float(row.iloc[0]["close"])
            sma = row.iloc[0]["sma_exit"]
            p["last_close"] = close
            p["peak"] = max(p.get("peak", p["entry_price"]), close)
            p["last_sma"] = float(sma) if pd.notna(sma) else 0.0
            nav += p["qty"] * close

        # 2. Exit checks on existing positions
        for sym, p in list(positions.items()):
            close = p["last_close"]
            sma = p.get("last_sma", 0.0)
            age = (d.date() - p["entry_date"]).days
            drop_from_peak = (p["peak"] - close) / p["peak"]
            exit_reason = None
            if sma > 0 and close < sma:
                exit_reason = "SMA_EXIT"
            elif drop_from_peak >= trail_pct:
                exit_reason = "TRAIL_STOP"
            elif age >= max_hold_days:
                exit_reason = "MAX_HOLD"
            if exit_reason:
                exit_px = close * (1 - slip)
                proceeds = p["qty"] * exit_px
                fees = proceeds * stt + brokerage
                total_fees += fees
                pnl = proceeds - fees - (p["qty"] * p["entry_price"])
                cash += proceeds - fees
                trades.append({
                    "entry_date": p["entry_date"].isoformat(),
                    "exit_date": d.date().isoformat(),
                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                    "entry_px": round(p["entry_price"], 2),
                    "exit_px": round(exit_px, 2),
                    "qty": p["qty"], "pnl": round(pnl, 2),
                    "hold_days": age, "reason": exit_reason,
                })
                del positions[sym]

        # 3. Entry scan — find symbols making 60d high with volume spike
        if len(positions) < max_conc:
            slots = max_conc - len(positions)
            candidates = []
            for sym in syms:
                if sym in positions:
                    continue
                df = sym_data.get(sym)
                if df is None:
                    continue
                row = df[df["date"] == d]
                if row.empty:
                    continue
                r = row.iloc[0]
                if pd.isna(r["high_lookback"]) or pd.isna(r["vol_avg"]):
                    continue
                if r["close"] >= float(r["high_lookback"]) and \
                   r["volume"] >= vol_mult * float(r["vol_avg"]) and \
                   float(r["vol_avg"]) > 0:
                    candidates.append({
                        "symbol": sym,
                        "close": float(r["close"]),
                        "vol_ratio": float(r["volume"]) / float(r["vol_avg"]),
                    })
            # Sort by volume ratio (strongest breakout first)
            candidates.sort(key=lambda c: -c["vol_ratio"])
            for c in candidates[:slots]:
                sym = c["symbol"]
                # Buy at NEXT day's open
                df = sym_data[sym]
                nxt = df[df["date"] > d]
                if nxt.empty:
                    continue
                entry_row = nxt.iloc[0]
                entry_px = float(entry_row["open"]) * (1 + slip)
                allocate = cash / max(1, slots)
                qty = int(allocate / entry_px)
                if qty < 1:
                    continue
                cost = qty * entry_px + brokerage
                total_fees += brokerage
                cash -= cost
                positions[sym] = {
                    "qty": qty, "entry_price": entry_px,
                    "entry_date": entry_row["date"].date(),
                    "peak": entry_px, "last_close": entry_px,
                    "last_sma": 0.0,
                }
                slots -= 1
                if slots <= 0:
                    break

        daily.append({"date": d.date(), "nav": nav, "cash": cash,
                      "open_pos": len(positions)})

    # Final mark-to-market
    eq = pd.DataFrame(daily).set_index(pd.to_datetime([r["date"] for r in daily]))
    eq.index.name = "date"
    eq["month"] = eq.index.to_period("M").astype(str)
    eq["year"] = eq.index.year
    eq["peak"] = eq["nav"].cummax()
    eq["dd_pct"] = (eq["nav"] / eq["peak"] - 1) * 100

    monthly = eq.groupby("month")["nav"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    yearly = eq.groupby("year")["nav"].agg(["first", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100

    return {
        "equity": eq, "monthly": monthly, "yearly": yearly,
        "final": float(eq["nav"].iloc[-1]),
        "max_dd": float(eq["dd_pct"].min()),
        "total_fees": total_fees,
        "trades": trades,
    }


def summarize_and_write(r: dict, capital: float, label: str, out_md: str | None):
    eq = r["equity"]
    m = r["monthly"]["ret_pct"]
    y = r["yearly"]["ret_pct"]
    final = r["final"]
    wins = [t for t in r["trades"] if t["pnl"] > 0]
    wr = (len(wins) / len(r["trades"]) * 100) if r["trades"] else 0

    print(f"\n=== {label} ===")
    print(f"Start ₹{capital:,.0f} -> End ₹{final:,.0f} "
          f"({(final/capital-1)*100:+.1f}% total)")
    print(f"Avg/mo: {m.mean():+.2f}%  Median: {m.median():+.2f}%")
    print(f"Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%")
    print(f"30%+: {(m>=30).sum()}/{m.count()}  20%+: {(m>=20).sum()}/{m.count()}  "
          f"<-10%: {(m<-10).sum()}/{m.count()}")
    print(f"Avg yearly: {y.mean():+.2f}%")
    print(f"Max DD: {r['max_dd']:.2f}%")
    print(f"Trades: {len(r['trades'])}  WR: {wr:.1f}%  Fees: ₹{r['total_fees']:,.0f}")

    if out_md:
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        lines = [
            f"# {label}\n",
            f"- Capital: ₹{capital:,.0f}",
            f"- Final: ₹{final:,.0f} ({(final/capital-1)*100:+.2f}%)",
            f"- Avg/mo: {m.mean():+.2f}%  Median: {m.median():+.2f}%",
            f"- Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%",
            f"- Months ≥ 20%: {(m>=20).sum()} / {m.count()}",
            f"- Months ≥ 30%: {(m>=30).sum()} / {m.count()}",
            f"- Avg yearly: {y.mean():+.2f}%",
            f"- Max DD: {r['max_dd']:.2f}%",
            f"- Trades: {len(r['trades'])}  Win rate: {wr:.1f}%",
            f"- Fees: ₹{r['total_fees']:,.0f}\n",
            "## Yearly\n",
            "| Year | Start | End | ROI |",
            "|---|---:|---:|---:|",
        ]
        for yr, row in r["yearly"].iterrows():
            lines.append(f"| {yr} | ₹{row['first']:,.0f} | ₹{row['last']:,.0f} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## Monthly\n")
        lines.append("| Month | ROI |")
        lines.append("|---|---:|")
        for mo, row in r["monthly"].iterrows():
            lines.append(f"| {mo} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## All Trades\n")
        lines.append("| Entry | Exit | Symbol | Entry₹ | Exit₹ | Qty | PnL₹ | Days | Reason |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for t in r["trades"]:
            lines.append(f"| {t['entry_date']} | {t['exit_date']} | {t['symbol']} | "
                         f"{t['entry_px']} | {t['exit_px']} | {t['qty']} | "
                         f"{t['pnl']} | {t['hold_days']} | {t['reason']} |")
        Path(out_md).write_text("\n".join(lines))
        print(f"\nReport: {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--max-conc", type=int, default=3)
    ap.add_argument("--breakout-lookback", type=int, default=60)
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--sma-exit", type=int, default=20)
    ap.add_argument("--trail-pct", type=float, default=0.08)
    ap.add_argument("--max-hold-days", type=int, default=90)
    ap.add_argument("--slip-bps", type=float, default=10.0)
    ap.add_argument("--brokerage", type=float, default=20.0)
    ap.add_argument("--stt-pct", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    r = run(args.universe_file, args.frm, args.to,
            args.max_conc, args.capital,
            args.breakout_lookback, args.vol_mult,
            args.sma_exit, args.trail_pct, args.max_hold_days,
            args.slip_bps, args.brokerage, args.stt_pct)
    label = (f"breakout_{args.breakout_lookback}dhigh_vol{args.vol_mult}x_"
             f"sma{args.sma_exit}_trail{int(args.trail_pct*100)}pct_"
             f"max{args.max_conc} ({args.frm}..{args.to})")
    summarize_and_write(r, args.capital, label, args.out)


if __name__ == "__main__":
    main()
