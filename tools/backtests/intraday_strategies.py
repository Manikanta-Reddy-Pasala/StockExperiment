"""Intraday day-trading strategies backtest on NSE top-30 N100.

Tests 4 simple strategies on 15m bars over 6 months
(2025-11-15 -> 2026-05-12). Calls realistic round-trip costs of ~0.13%
per trade.

Strategies:
    1. ORB-30          - Opening range breakout (first 30 min)
    2. VWAP_pullback   - Pullback to VWAP after morning uptrend
    3. EMA_9_21_15m    - EMA 9/21 crossover on 15m
    4. Gap_and_go      - Open gap > 1% with continuation

Each: 1 concurrent position, capital INR 10L, exit by 15:00.

Run on container:
    docker exec trading_system_app bash -c \
      "cd /app && python -m tools.backtests.intraday_strategies"
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.backtests.ohlcv_cache import read_cached

# ----- Config -----
CAPITAL = 1_000_000.0
COST_PCT = 0.0013                       # 0.13% round-trip
START_DATE = "2025-11-15"
END_DATE = "2026-05-13"
TOP_N = 30
SESSION_START = dtime(9, 15)
ENTRY_CUTOFF = dtime(14, 30)            # stop new entries
FORCED_EXIT = dtime(15, 0)              # square off
ORB_END = dtime(9, 45)                  # ORB observation end


def _load_universe(top_n: int = TOP_N) -> List[str]:
    """Pick top-N most liquid N100 stocks by ADV."""
    p = Path("/app/logs/momrot/universes/n100_current.json")
    data = json.loads(p.read_text())
    return [s["symbol"] for s in data["stocks"][:top_n]]


def _load_bars(symbol: str) -> pd.DataFrame:
    """Load 15m bars for the test window."""
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
    """Accumulates trades and produces summary stats."""

    def __init__(self, name: str):
        self.name = name
        self.trades: List[Dict] = []

    def record(self, symbol: str, side: str, entry_ts, exit_ts,
               entry: float, exit_: float, qty: int, reason: str):
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


# ----- Strategy 1: ORB-30 -----
def strat_orb30(df: pd.DataFrame, symbol: str, book: TradeBook):
    """Long break > ORB high; short break < ORB low. SL = opposite edge.
    Target = 1.5x range. Exit by 15:00."""
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
        in_pos = None  # (side, entry, sl, tgt, entry_ts)
        for _, row in post.iterrows():
            if in_pos is None:
                if row["t"] >= ENTRY_CUTOFF:
                    break
                if row["high"] > orb_hi:
                    entry = orb_hi
                    in_pos = ("L", entry, orb_lo, entry + 1.5 * rng, row["candle_time"])
                elif row["low"] < orb_lo:
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


# ----- Strategy 2: VWAP pullback -----
def strat_vwap_pullback(df: pd.DataFrame, symbol: str, book: TradeBook):
    """LONG when price pulls back to VWAP after morning uptrend.
    Morning uptrend = last (10:00 -> 10:45) close > 09:15 open.
    SL = 1% below entry, target = 1.5 * SL distance.
    """
    prev_close = None
    for date, day in df.groupby("date", sort=False):
        day = day.reset_index(drop=True)
        if len(day) < 10:
            prev_close = day["close"].iloc[-1]
            continue
        day["typ"] = (day["high"] + day["low"] + day["close"]) / 3
        day["cumvp"] = (day["typ"] * day["volume"]).cumsum()
        day["cumv"] = day["volume"].cumsum()
        day["vwap"] = day["cumvp"] / day["cumv"].replace(0, np.nan)

        open_px = day["open"].iloc[0]
        # Morning uptrend: at 10:45, price up >0.3% from open
        morning = day[(day["t"] >= dtime(10, 30)) & (day["t"] <= dtime(10, 45))]
        if morning.empty:
            prev_close = day["close"].iloc[-1]
            continue
        morn_close = morning["close"].iloc[-1]
        if (morn_close - open_px) / open_px < 0.003:
            prev_close = day["close"].iloc[-1]
            continue

        in_pos = None
        post = day[(day["t"] > dtime(10, 45)) & (day["t"] < FORCED_EXIT)]
        for _, row in post.iterrows():
            if in_pos is None:
                if row["t"] >= ENTRY_CUTOFF:
                    break
                vwap = row["vwap"]
                # Pullback: low touches VWAP from above, close stays above VWAP
                if row["low"] <= vwap * 1.001 and row["close"] > vwap and row["close"] > open_px:
                    entry = row["close"]
                    sl = entry * 0.99
                    tgt = entry + 1.5 * (entry - sl)
                    in_pos = ("L", entry, sl, tgt, row["candle_time"])
            else:
                side, entry, sl, tgt, ets = in_pos
                qty = max(1, int(CAPITAL / entry))
                if row["low"] <= sl:
                    book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                    in_pos = None
                elif row["high"] >= tgt:
                    book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                    in_pos = None
        if in_pos is not None:
            side, entry, sl, tgt, ets = in_pos
            qty = max(1, int(CAPITAL / entry))
            close_bars = day[day["t"] >= FORCED_EXIT]
            exit_px = close_bars["open"].iloc[0] if not close_bars.empty else day["close"].iloc[-1]
            exit_ts = close_bars["candle_time"].iloc[0] if not close_bars.empty else day["candle_time"].iloc[-1]
            book.record(symbol, side, ets, exit_ts, entry, exit_px, qty, "EOD")
        prev_close = day["close"].iloc[-1]


# ----- Strategy 3: EMA 9/21 crossover on 15m -----
def strat_ema_9_21(df: pd.DataFrame, symbol: str, book: TradeBook):
    """LONG on EMA9 crossing above EMA21. SL 1.5%, target 3%. Exit 15:00.
    Computed per-day to keep intraday scope; EMAs reset each day."""
    for date, day in df.groupby("date", sort=False):
        day = day.reset_index(drop=True)
        if len(day) < 22:
            continue
        day["ema9"] = day["close"].ewm(span=9, adjust=False).mean()
        day["ema21"] = day["close"].ewm(span=21, adjust=False).mean()
        day["above"] = day["ema9"] > day["ema21"]
        day["cross_up"] = day["above"] & (~day["above"].shift(1).fillna(False))

        in_pos = None
        post = day[day["t"] < FORCED_EXIT]
        for _, row in post.iterrows():
            if in_pos is None:
                if row["t"] >= ENTRY_CUTOFF:
                    break
                if row["cross_up"] and row["t"] >= dtime(9, 45):
                    entry = row["close"]
                    sl = entry * 0.985
                    tgt = entry * 1.03
                    in_pos = ("L", entry, sl, tgt, row["candle_time"])
            else:
                side, entry, sl, tgt, ets = in_pos
                qty = max(1, int(CAPITAL / entry))
                if row["low"] <= sl:
                    book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                    in_pos = None
                elif row["high"] >= tgt:
                    book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                    in_pos = None
        if in_pos is not None:
            side, entry, sl, tgt, ets = in_pos
            qty = max(1, int(CAPITAL / entry))
            close_bars = day[day["t"] >= FORCED_EXIT]
            exit_px = close_bars["open"].iloc[0] if not close_bars.empty else day["close"].iloc[-1]
            exit_ts = close_bars["candle_time"].iloc[0] if not close_bars.empty else day["candle_time"].iloc[-1]
            book.record(symbol, side, ets, exit_ts, entry, exit_px, qty, "EOD")


# ----- Strategy 4: Gap-and-go -----
def strat_gap_and_go(df: pd.DataFrame, symbol: str, book: TradeBook):
    """Gap > 1% AND first 15m candle in gap direction -> ride.
    SL 1%, target 2%. Exit 15:00."""
    prev_close = None
    for date, day in df.groupby("date", sort=False):
        day = day.reset_index(drop=True)
        if prev_close is None or len(day) < 2:
            prev_close = day["close"].iloc[-1]
            continue
        open_px = day["open"].iloc[0]
        gap = (open_px - prev_close) / prev_close
        first = day.iloc[0]
        first_dir = np.sign(first["close"] - first["open"])

        side = entry = sl = tgt = ets = None
        if gap > 0.01 and first_dir > 0:
            entry = first["close"]
            side = "L"
            sl = entry * 0.99
            tgt = entry * 1.02
            ets = first["candle_time"]
        elif gap < -0.01 and first_dir < 0:
            entry = first["close"]
            side = "S"
            sl = entry * 1.01
            tgt = entry * 0.98
            ets = first["candle_time"]

        if side is None:
            prev_close = day["close"].iloc[-1]
            continue

        in_pos = (side, entry, sl, tgt, ets)
        post = day.iloc[1:]
        post = post[post["t"] < FORCED_EXIT]
        for _, row in post.iterrows():
            side, entry, sl, tgt, ets = in_pos
            qty = max(1, int(CAPITAL / entry))
            if side == "L":
                if row["low"] <= sl:
                    book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                    in_pos = None; break
                elif row["high"] >= tgt:
                    book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                    in_pos = None; break
            else:
                if row["high"] >= sl:
                    book.record(symbol, side, ets, row["candle_time"], entry, sl, qty, "SL")
                    in_pos = None; break
                elif row["low"] <= tgt:
                    book.record(symbol, side, ets, row["candle_time"], entry, tgt, qty, "TGT")
                    in_pos = None; break
        if in_pos is not None:
            side, entry, sl, tgt, ets = in_pos
            qty = max(1, int(CAPITAL / entry))
            close_bars = day[day["t"] >= FORCED_EXIT]
            exit_px = close_bars["open"].iloc[0] if not close_bars.empty else day["close"].iloc[-1]
            exit_ts = close_bars["candle_time"].iloc[0] if not close_bars.empty else day["candle_time"].iloc[-1]
            book.record(symbol, side, ets, exit_ts, entry, exit_px, qty, "EOD")
        prev_close = day["close"].iloc[-1]


# ----- Backtest runner with single-position constraint -----
def run_strategy(name: str, func, symbols: List[str]) -> pd.DataFrame:
    """Run strategy per-symbol; then enforce single concurrent position
    across all symbols by greedy time-ordered selection."""
    book = TradeBook(name)
    for sym in symbols:
        bars = _load_bars(sym)
        if bars.empty:
            continue
        func(bars, sym, book)
    df = book.to_df()
    if df.empty:
        return df

    # Enforce 1-position-at-a-time: sort by entry_ts; keep trade only if
    # entry_ts >= last accepted exit_ts.
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
    """Per-strategy summary stats."""
    if df.empty:
        return {"strategy": name, "trades": 0, "win_rate": 0, "total_ret": 0,
                "avg_pnl": 0, "max_dd": 0, "sharpe": 0, "by_month": {}}
    df = df.copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df["day"] = df["exit_ts"].dt.date
    df["month"] = df["exit_ts"].dt.strftime("%Y-%m")

    # Compound: each trade nets ret_pct% on CAPITAL. We use net / CAPITAL
    df["net_pct_cap"] = df["net"] / CAPITAL * 100

    # Per-month ROI = sum of net_pct_cap (not compounded daily to keep simple)
    by_month = df.groupby("month")["net_pct_cap"].sum().to_dict()

    # Equity curve (daily P&L sum), max DD
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

    return {
        "strategy": name,
        "trades": len(df),
        "trades_per_month": len(df) / max(1, len(by_month)),
        "win_rate": win_rate,
        "total_ret": total_ret,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "by_month": by_month,
    }


def main():
    symbols = _load_universe(TOP_N)
    print(f"Universe: {len(symbols)} symbols (top-{TOP_N} N100 by ADV)")
    print(f"Window:   {START_DATE} -> {END_DATE}  (~6 months)")
    print(f"Capital:  INR {CAPITAL:,.0f}   Cost: {COST_PCT*100:.2f}% round-trip\n")

    strategies = [
        ("ORB-30",         strat_orb30),
        ("VWAP_pullback",  strat_vwap_pullback),
        ("EMA_9_21_15m",   strat_ema_9_21),
        ("Gap_and_go",     strat_gap_and_go),
    ]

    all_results = {}
    all_trades = {}
    for name, func in strategies:
        print(f"Running {name} ...", flush=True)
        trades = run_strategy(name, func, symbols)
        summ = summarize(name, trades)
        all_results[name] = summ
        all_trades[name] = trades
        print(f"  trades={summ['trades']}  win_rate={summ['win_rate']:.1f}%  "
              f"total_ret={summ['total_ret']:.2f}%  max_dd={summ['max_dd']:.2f}%  "
              f"sharpe={summ['sharpe']:.2f}")

    # ----- Comparison table -----
    print("\n" + "=" * 90)
    print("COMPARISON: monthly ROI per strategy (% on capital)")
    print("=" * 90)
    months = sorted({m for r in all_results.values() for m in r["by_month"].keys()})
    header = f"{'Strategy':<18}" + "".join(f"{m:>10}" for m in months) + f"{'TOTAL':>10}{'Win%':>8}{'DD%':>8}{'Sh':>6}{'#Tr':>6}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        row = f"{name:<18}"
        for m in months:
            row += f"{r['by_month'].get(m, 0):>10.2f}"
        row += f"{r['total_ret']:>10.2f}{r['win_rate']:>8.1f}{r['max_dd']:>8.2f}{r['sharpe']:>6.2f}{r['trades']:>6d}"
        print(row)

    # ----- Best strategy avg monthly -----
    print("\nAVG MONTHLY RETURN (best):")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["total_ret"]):
        months_n = max(1, len(r["by_month"]))
        avg_mo = r["total_ret"] / months_n
        print(f"  {name:<18} avg_monthly={avg_mo:>6.2f}%  total={r['total_ret']:>6.2f}% over {months_n} months")

    # ----- Top symbols by expectancy across all strategies -----
    big = pd.concat([df for df in all_trades.values() if not df.empty], ignore_index=True)
    if not big.empty:
        sym_exp = big.groupby("symbol").agg(
            n=("net", "size"), total_net=("net", "sum"), avg_net=("net", "mean"),
            win=("net", lambda s: (s > 0).mean() * 100)
        ).sort_values("total_net", ascending=False)
        print("\nTOP 5 SYMBOLS BY TOTAL NET P&L (across all strategies):")
        for sym, row in sym_exp.head(5).iterrows():
            print(f"  {sym:<22}  n={int(row['n']):>4}  total=INR{row['total_net']:>11,.0f}  "
                  f"avg=INR{row['avg_net']:>8,.0f}  win={row['win']:.1f}%")

    # Save raw results
    out_dir = Path("/app/logs/intraday")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in all_trades.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}_trades.csv", index=False)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "by_month"} | {"by_month": v["by_month"]}
         for k, v in all_results.items()}, indent=2, default=str))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
