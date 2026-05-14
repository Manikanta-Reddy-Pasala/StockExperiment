"""ORB-30 intraday backtest restricted to safer universes.

Three variants:
    1. N50_long_only       - NIFTY 50 (53 stocks), LONG-only breakouts
    2. N50_both_sides      - NIFTY 50, LONG + SHORT
    3. N100_top_by_ADV     - top-30 N100 by ADV (reference / prior run)

Same window 2025-11-15 -> 2026-05-13, same costs (0.13% round-trip),
same logic (ORB 09:15-09:45, target 1.5xrange, SL=opposite edge, exit 15:00).
Capital INR 10L, 1 concurrent position.

Run on container:
    docker exec trading_system_app bash -c \
      "cd /app && python -m tools.backtests.intraday_orb30_n50"
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tools.backtests.ohlcv_cache import read_cached

# ----- Config -----
CAPITAL = 1_000_000.0
COST_PCT = 0.0013
START_DATE = "2025-11-15"
END_DATE = "2026-05-13"
SESSION_START = dtime(9, 15)
ENTRY_CUTOFF = dtime(14, 30)
FORCED_EXIT = dtime(15, 0)
ORB_END = dtime(9, 45)

# NIFTY 50 base list (53 names incl. recent reconstitutions).
NIFTY50_BASE = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BPCL',
    'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB',
    'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK',
    'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
    'ITC', 'INDUSINDBK', 'INFY', 'JIOFIN', 'JSWSTEEL',
    'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC',
    'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE',
    'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TCS', 'TATACONSUM',
    'TMPV', 'TATASTEEL', 'TECHM', 'TITAN', 'TRENT',
    'ULTRACEMCO', 'UPL', 'WIPRO',
]


def _load_n100_top_adv(top_n: int = 30) -> List[str]:
    p = Path("/app/logs/momrot/universes/n100_current.json")
    data = json.loads(p.read_text())
    return [s["symbol"] for s in data["stocks"][:top_n]]


def _load_bars(symbol: str) -> pd.DataFrame:
    from_ts = int(datetime.fromisoformat(START_DATE).timestamp())
    to_ts = int(datetime.fromisoformat(END_DATE).timestamp())
    df = read_cached(symbol, "15m", from_ts, to_ts)
    if df.empty:
        return df
    df = df.copy()
    df["candle_time"] = pd.to_datetime(df["candle_time"])
    df["date"] = df["candle_time"].dt.date
    df["t"] = df["candle_time"].dt.time
    return df.sort_values("candle_time").reset_index(drop=True)


# ----- Trade book-keeping -----
class TradeBook:
    def __init__(self, name: str):
        self.name = name
        self.trades: List[Dict] = []

    def record(self, symbol, side, entry_ts, exit_ts, entry, exit_, qty, reason):
        gross = (exit_ - entry) * qty if side == "L" else (entry - exit_) * qty
        cost = (entry + exit_) * qty * COST_PCT / 2
        net = gross - cost
        self.trades.append({
            "symbol": symbol, "side": side,
            "entry_ts": entry_ts, "exit_ts": exit_ts,
            "entry": entry, "exit": exit_, "qty": qty,
            "gross": gross, "cost": cost, "net": net,
            "ret_pct": net / (entry * qty) * 100,
            "reason": reason,
            "month": pd.Timestamp(exit_ts).strftime("%Y-%m"),
        })

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()


# ----- ORB-30 with long_only switch -----
def strat_orb30(df: pd.DataFrame, symbol: str, book: TradeBook, long_only: bool):
    for date, day in df.groupby("date", sort=False):
        day = day.reset_index(drop=True)
        orb = day[day["t"] <= ORB_END]
        if len(orb) < 2:
            continue
        orb_hi = orb["high"].max()
        orb_lo = orb["low"].min()
        rng = orb_hi - orb_lo
        if rng <= 0:
            continue
        post = day[(day["t"] > ORB_END) & (day["t"] < FORCED_EXIT)]
        in_pos = None
        for _, row in post.iterrows():
            if in_pos is None:
                if row["t"] >= ENTRY_CUTOFF:
                    break
                if row["high"] > orb_hi:
                    entry = orb_hi
                    in_pos = ("L", entry, orb_lo, entry + 1.5 * rng, row["candle_time"])
                elif row["low"] < orb_lo and not long_only:
                    entry = orb_lo
                    in_pos = ("S", entry, orb_hi, entry - 1.5 * rng, row["candle_time"])
            else:
                side, entry, sl, tgt, ets = in_pos
                qty = max(1, int(CAPITAL / entry))
                if side == "L":
                    if row["low"] <= sl:
                        book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                        in_pos = None
                    elif row["high"] >= tgt:
                        book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                        in_pos = None
                else:
                    if row["high"] >= sl:
                        book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                        in_pos = None
                    elif row["low"] <= tgt:
                        book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                        in_pos = None
        if in_pos is not None:
            side, entry, sl, tgt, ets = in_pos
            qty = max(1, int(CAPITAL / entry))
            close_bars = day[day["t"] >= FORCED_EXIT]
            exit_px = close_bars["open"].iloc[0] if not close_bars.empty else day["close"].iloc[-1]
            exit_ts = close_bars["candle_time"].iloc[0] if not close_bars.empty else day["candle_time"].iloc[-1]
            book.record(symbol, side, ets, exit_ts, entry, exit_px, qty, "EOD")


def run_variant(name: str, symbols: List[str], long_only: bool) -> pd.DataFrame:
    book = TradeBook(name)
    missing = []
    for sym in symbols:
        bars = _load_bars(sym)
        if bars.empty:
            missing.append(sym)
            continue
        strat_orb30(bars, sym, book, long_only)
    if missing:
        print(f"  [warn] {len(missing)} symbols had no bars: {missing[:5]}{'...' if len(missing)>5 else ''}")
    df = book.to_df()
    if df.empty:
        return df
    # 1-position-at-a-time, time-ordered greedy
    df = df.sort_values("entry_ts").reset_index(drop=True)
    keep = []
    last_exit = pd.Timestamp.min.tz_localize(None)
    for _, r in df.iterrows():
        ets = pd.Timestamp(r["entry_ts"])
        xts = pd.Timestamp(r["exit_ts"])
        if ets >= last_exit:
            keep.append(True)
            last_exit = xts
        else:
            keep.append(False)
    df = df[keep].reset_index(drop=True)
    return df


def summarize(name: str, df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"variant": name, "trades": 0, "win_rate": 0, "total_ret": 0,
                "avg_pnl": 0, "max_dd": 0, "sharpe": 0, "by_month": {}, "top_syms": []}
    df = df.copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df["day"] = df["exit_ts"].dt.date
    df["month"] = df["exit_ts"].dt.strftime("%Y-%m")
    df["net_pct_cap"] = df["net"] / CAPITAL * 100
    by_month = df.groupby("month")["net_pct_cap"].sum().to_dict()

    daily = df.groupby("day")["net"].sum().sort_index()
    eq = daily.cumsum() + CAPITAL
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min() if len(dd) else 0
    daily_ret = daily / CAPITAL
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    win_rate = (df["net"] > 0).mean() * 100
    total_ret = df["net"].sum() / CAPITAL * 100
    avg_pnl = df["net"].mean()
    months_n = max(1, len(by_month))
    avg_mo = total_ret / months_n

    sym_pnl = df.groupby("symbol").agg(
        n=("net", "size"), total_net=("net", "sum"),
        win=("net", lambda s: (s > 0).mean() * 100)
    ).sort_values("total_net", ascending=False)
    top5 = [
        {"symbol": s, "n": int(r["n"]), "total_net": float(r["total_net"]),
         "win": float(r["win"])}
        for s, r in sym_pnl.head(5).iterrows()
    ]

    return {
        "variant": name,
        "trades": len(df),
        "trades_per_month": len(df) / months_n,
        "win_rate": win_rate,
        "total_ret": total_ret,
        "avg_monthly": avg_mo,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "by_month": by_month,
        "top_syms": top5,
    }


def main():
    variants = [
        ("N50_long_only",   NIFTY50_BASE,                 True),
        ("N50_both_sides",  NIFTY50_BASE,                 False),
        ("N100_top_by_ADV", _load_n100_top_adv(30),       False),
    ]
    print(f"Window:  {START_DATE} -> {END_DATE}")
    print(f"Capital: INR {CAPITAL:,.0f}   Cost: {COST_PCT*100:.2f}% round-trip")
    print(f"Logic:   ORB-30 09:15-09:45, target 1.5xrange, SL=opp edge, exit 15:00\n")

    all_summaries = {}
    all_trades = {}
    for name, syms, long_only in variants:
        print(f"--- {name}  universe={len(syms)}  long_only={long_only} ---", flush=True)
        df = run_variant(name, syms, long_only)
        summ = summarize(name, df)
        all_summaries[name] = summ
        all_trades[name] = df
        print(f"  trades={summ['trades']}  win={summ['win_rate']:.1f}%  "
              f"total={summ['total_ret']:.2f}%  avg_mo={summ.get('avg_monthly',0):.2f}%  "
              f"dd={summ['max_dd']:.2f}%  sharpe={summ['sharpe']:.2f}\n")

    # ----- Comparison table -----
    print("=" * 100)
    print("COMPARISON TABLE: monthly ROI (% on capital)")
    print("=" * 100)
    months = sorted({m for r in all_summaries.values() for m in r["by_month"].keys()})
    header = f"{'Variant':<20}" + "".join(f"{m:>10}" for m in months) + \
             f"{'TOTAL':>10}{'AvgMo':>8}{'Win%':>8}{'DD%':>8}{'Sh':>6}{'#Tr':>6}"
    print(header)
    print("-" * len(header))
    for name, r in all_summaries.items():
        row = f"{name:<20}"
        for m in months:
            row += f"{r['by_month'].get(m, 0):>10.2f}"
        row += f"{r['total_ret']:>10.2f}{r.get('avg_monthly',0):>8.2f}{r['win_rate']:>8.1f}"
        row += f"{r['max_dd']:>8.2f}{r['sharpe']:>6.2f}{r['trades']:>6d}"
        print(row)

    print("\nTOP 5 STOCKS BY P&L (per variant):")
    for name, r in all_summaries.items():
        print(f"  {name}:")
        for t in r["top_syms"]:
            print(f"    {t['symbol']:<14}  n={t['n']:>3}  "
                  f"total=INR{t['total_net']:>11,.0f}  win={t['win']:.1f}%")

    # Save
    out_dir = Path("/app/logs/intraday")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in all_trades.items():
        if not df.empty:
            df.to_csv(out_dir / f"orb30_{name}_trades.csv", index=False)
    (out_dir / "orb30_n50_summary.json").write_text(
        json.dumps(all_summaries, indent=2, default=str))
    print(f"\nSaved: {out_dir}/orb30_n50_summary.json")


if __name__ == "__main__":
    main()
