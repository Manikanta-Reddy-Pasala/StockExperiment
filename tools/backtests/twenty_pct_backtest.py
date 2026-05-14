"""TARGET ~20%/mo backtest on Indian markets.

Strategy: Aggressive Intraday ORB-15 on top liquid F&O stocks
==============================================================

Rationale (after research phase):
- 920 short straddle: 593% over 5y on BN (~9.8%/mo) — needs options chain data (UNAVAILABLE)
- ORB-15 Nifty alone: ~91% over 9y (~0.8%/mo) — too slow
- Capitalmind Momentum: ~20-25%/yr (1.7%/mo)
- Closest backtestable proxy for 20%/mo: leveraged intraday on liquid F&O stocks with
  strong pre-market signal + tight risk + intraday MIS leverage (5x typical for retail)

This backtest:
- Universe: top 30 N100 by ADV (most liquid, all F&O)
- 12-month window: 2025-05-13 -> 2026-05-12
- Capital: INR 10,00,000
- Per-trade risk: 1% of equity (stop distance sizes the position)
- Max concurrent positions: 3
- Intraday leverage (MIS): 5x (typical Zerodha/Upstox MIS for cash stocks)
- Costs: 0.13% round-trip per trade (STT + brokerage + GST + exchange)

Entry rules (each day, per symbol):
  1. ORB range = high/low of 09:15-09:30 bar (15-min)
  2. Direction filter: previous-day close > 10-day SMA daily (LONG bias only).
     This is a quasi-trend filter; intraday ORB longs in uptrending names work best.
  3. Pre-market strength: today's OPEN must be within +/- 1.5% of yesterday's close
     (filter out runaway gaps that already exhausted move).
  4. Entry: LONG when 09:30-13:00 bar closes above ORB high AND
     bar volume > 1.5x of 09:15-09:30 ORB bar volume (confirmation).
  5. Stop: ORB low (or entry - 0.6*ORB_range, whichever is tighter).
  6. Target: 2.0 * stop_distance (2R).
  7. Time exit: 14:55 forced close (no overnight on MIS).

Position sizing:
  qty = floor( (equity * 0.01) / stop_distance )
  position_value capped at equity * leverage / max_concurrent

Run:
  scp tools/backtests/twenty_pct_backtest.py root@77.42.45.12:/tmp/
  ssh root@77.42.45.12 'docker cp /tmp/twenty_pct_backtest.py trading_system_app:/app/tools/backtests/ && \
    docker exec trading_system_app bash -c "cd /app && python -m tools.backtests.twenty_pct_backtest"'
"""
from __future__ import annotations

import json
import sys
import math
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.backtests.ohlcv_cache import read_cached

# ---------- Config ----------
CAPITAL = 1_000_000.0
COST_PCT = 0.0013          # 0.13% round-trip intraday cash
RISK_PCT = 0.01            # 1% per trade
LEVERAGE = 5.0             # intraday MIS leverage
MAX_CONCURRENT = 3
TARGET_R = float(__import__("os").environ.get("TARGET_R", "2.0"))
START_DATE = "2025-05-13"
END_DATE = "2026-05-12"
SESSION_START = dtime(9, 15)
ORB_END = dtime(9, 30)     # ORB is first 15-min bar only
ENTRY_CUTOFF = dtime(13, 0)
FORCED_EXIT = dtime(14, 55)
SMA_TREND_LEN = 10
GAP_MAX = float(__import__("os").environ.get("GAP_MAX", "0.015"))
VOL_MULT = float(__import__("os").environ.get("VOL_MULT", "1.5"))
TRAIL_AT_R = float(__import__("os").environ.get("TRAIL_AT_R", "0"))  # 0=off, else move stop to BE+x*R after 1R
ALLOW_SHORT = __import__("os").environ.get("ALLOW_SHORT", "0") == "1"


def load_universe() -> List[str]:
    p = Path("/app/logs/momrot/universes/n100_current.json")
    if not p.exists():
        # Local fallback for offline iteration
        p = Path(__file__).parent.parent.parent / "logs/momrot/universes/n100_current.json"
    data = json.loads(p.read_text())
    return [s["symbol"] for s in data["stocks"][:30]]


def load_15m(symbol: str) -> pd.DataFrame:
    a = int(datetime.fromisoformat(START_DATE).timestamp()) - 86400 * 30
    b = int(datetime.fromisoformat(END_DATE).timestamp()) + 86400
    df = read_cached(symbol, "15m", a, b)
    if df.empty:
        return df
    df = df.copy()
    df["candle_time"] = pd.to_datetime(df["candle_time"])
    df["date"] = df["candle_time"].dt.date
    df["t"] = df["candle_time"].dt.time
    return df.sort_values("candle_time").reset_index(drop=True)


def load_daily(symbol: str) -> pd.DataFrame:
    a = int(datetime.fromisoformat(START_DATE).timestamp()) - 86400 * 60
    b = int(datetime.fromisoformat(END_DATE).timestamp()) + 86400
    df = read_cached(symbol, "D", a, b)
    if df.empty:
        return df
    df = df.copy()
    df["candle_time"] = pd.to_datetime(df["candle_time"])
    df["date"] = df["candle_time"].dt.date
    df = df.sort_values("candle_time").reset_index(drop=True)
    df["sma"] = df["close"].rolling(SMA_TREND_LEN).mean()
    df["prev_close"] = df["close"].shift(1)
    df["prev_sma"]   = df["sma"].shift(1)
    return df


def build_signals(df15: pd.DataFrame, dfD: pd.DataFrame) -> List[Dict]:
    """Generate one candidate trade per (symbol, day) where rules pass.
       Position sizing happens later in the equity-walked main loop."""
    sigs = []
    daily_map = {r.date: r for r in dfD.itertuples()}
    for date, day in df15.groupby("date", sort=False):
        d_row = daily_map.get(date)
        if d_row is None or pd.isna(d_row.prev_close) or pd.isna(d_row.prev_sma):
            continue
        # Trend filter (only for longs; shorts use opposite)
        trend_long_ok = d_row.prev_close > d_row.prev_sma
        trend_short_ok = ALLOW_SHORT and (d_row.prev_close < d_row.prev_sma)
        if not (trend_long_ok or trend_short_ok):
            continue
        # ORB bar = 09:15
        orb_bars = day[(day.t >= SESSION_START) & (day.t < ORB_END)]
        # 09:15-09:29 inclusive => one 15m bar at 09:15
        orb = day[day.t == SESSION_START]
        if orb.empty:
            continue
        ob = orb.iloc[0]
        # Open / prev_close gap filter
        if abs(ob.open - d_row.prev_close) / d_row.prev_close > GAP_MAX:
            continue
        orb_high = ob.high
        orb_low  = ob.low
        orb_vol  = ob.volume
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue
        # Walk subsequent bars within entry window
        post = day[(day.t > ORB_END) & (day.t <= ENTRY_CUTOFF)].copy()
        if post.empty:
            continue
        for bar in post.itertuples():
            # LONG breakout
            if trend_long_ok and bar.close > orb_high and bar.volume > VOL_MULT * orb_vol:
                entry_ts = bar.candle_time
                entry = bar.close
                stop_a = orb_low
                stop_b = entry - 0.6 * orb_range
                stop = max(stop_a, stop_b)
                if stop >= entry:
                    break
                stop_dist = entry - stop
                target = entry + TARGET_R * stop_dist
                sigs.append({
                    "date": date, "entry_ts": entry_ts, "side": "L",
                    "entry": entry, "stop": stop, "target": target,
                    "stop_dist": stop_dist, "orb_range": orb_range,
                })
                break
            # SHORT breakout (only if enabled and trend bearish)
            if trend_short_ok and bar.close < orb_low and bar.volume > VOL_MULT * orb_vol:
                entry_ts = bar.candle_time
                entry = bar.close
                stop_a = orb_high
                stop_b = entry + 0.6 * orb_range
                stop = min(stop_a, stop_b)
                if stop <= entry:
                    break
                stop_dist = stop - entry
                target = entry - TARGET_R * stop_dist
                sigs.append({
                    "date": date, "entry_ts": entry_ts, "side": "S",
                    "entry": entry, "stop": stop, "target": target,
                    "stop_dist": stop_dist, "orb_range": orb_range,
                })
                break
    return sigs


def simulate_intraday(date_to_orders: Dict[pd.Timestamp, List[Tuple[str, Dict, pd.DataFrame]]],
                      bars_by_symbol: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], List[Dict]]:
    """Walk forward day-by-day, executing up to MAX_CONCURRENT trades per day.
       Returns (trades, equity_curve)."""
    equity = CAPITAL
    trades, curve = [], []
    sorted_days = sorted(date_to_orders.keys())
    for day in sorted_days:
        orders = date_to_orders[day]
        # Earliest entry first
        orders.sort(key=lambda x: x[1]["entry_ts"])
        opened = []  # list of (symbol, sig, exit_state)
        for symbol, sig, bars in orders:
            if len(opened) >= MAX_CONCURRENT:
                break
            # Position size
            risk_inr = equity * RISK_PCT
            qty = int(risk_inr // sig["stop_dist"])
            if qty <= 0:
                continue
            # Cap by leverage / concurrent
            max_pos_inr = equity * LEVERAGE / MAX_CONCURRENT
            max_qty = int(max_pos_inr // sig["entry"])
            qty = min(qty, max_qty)
            if qty <= 0:
                continue
            # Simulate exit by walking bars after entry_ts
            entry_ts = sig["entry_ts"]
            day_bars = bars[(bars.candle_time > entry_ts)
                            & (bars.t <= FORCED_EXIT)].copy()
            if day_bars.empty:
                # Force exit at entry (shouldn't happen)
                continue
            exit_price, exit_ts, reason = None, None, None
            cur_stop = sig["stop"]
            side = sig.get("side", "L")
            entry_p = sig["entry"]
            stop_dist0 = sig["stop_dist"]
            for b in day_bars.itertuples():
                # Trailing stop activation: once price moves 1R in favor, move stop to BE+TRAIL_AT_R*R
                if TRAIL_AT_R > 0:
                    if side == "L":
                        if b.high >= entry_p + 1.0 * stop_dist0:
                            new_stop = entry_p + (TRAIL_AT_R - 1) * stop_dist0  # TRAIL_AT_R=1 -> BE; 1.5 -> 0.5R locked
                            cur_stop = max(cur_stop, new_stop)
                    else:
                        if b.low <= entry_p - 1.0 * stop_dist0:
                            new_stop = entry_p - (TRAIL_AT_R - 1) * stop_dist0
                            cur_stop = min(cur_stop, new_stop)
                # Side-aware exit
                if side == "L":
                    if b.low <= cur_stop:
                        exit_price, exit_ts, reason = cur_stop, b.candle_time, "STOP"
                        break
                    if b.high >= sig["target"]:
                        exit_price, exit_ts, reason = sig["target"], b.candle_time, "TGT"
                        break
                else:
                    if b.high >= cur_stop:
                        exit_price, exit_ts, reason = cur_stop, b.candle_time, "STOP"
                        break
                    if b.low <= sig["target"]:
                        exit_price, exit_ts, reason = sig["target"], b.candle_time, "TGT"
                        break
            if exit_price is None:
                last = day_bars.iloc[-1]
                exit_price, exit_ts, reason = float(last.close), last.candle_time, "EOD"

            sign = 1 if side == "L" else -1
            gross = (exit_price - sig["entry"]) * sign * qty
            cost = (sig["entry"] + exit_price) * qty * COST_PCT / 2.0
            net = gross - cost
            equity += net
            trades.append({
                "symbol": symbol, "side": side, "date": day,
                "entry_ts": entry_ts, "exit_ts": exit_ts,
                "entry": sig["entry"], "exit": exit_price,
                "qty": qty, "gross": gross, "cost": cost, "net": net,
                "ret_pct_on_capital": net / CAPITAL * 100,
                "reason": reason,
                "month": pd.Timestamp(exit_ts).strftime("%Y-%m"),
            })
            opened.append((symbol, sig, None))
        curve.append({"date": day, "equity": equity})
    return trades, curve


def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} symbols")
    print(f"Window:   {START_DATE} -> {END_DATE}")
    print(f"Capital:  INR {CAPITAL:,.0f}, leverage {LEVERAGE}x, risk {RISK_PCT*100}%/trade")
    print()
    # Gather all signals
    date_to_orders: Dict[pd.Timestamp, List] = {}
    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    sym_sig_count = []
    for sym in universe:
        df15 = load_15m(sym)
        dfD  = load_daily(sym)
        if df15.empty or dfD.empty:
            sym_sig_count.append((sym, 0, "no-data"))
            continue
        # Trim to window
        win_start = pd.Timestamp(START_DATE).date()
        win_end   = pd.Timestamp(END_DATE).date()
        df15w = df15[(df15.date >= win_start) & (df15.date <= win_end)]
        if df15w.empty:
            sym_sig_count.append((sym, 0, "no-bars-in-window"))
            continue
        sigs = build_signals(df15w, dfD)
        bars_by_symbol[sym] = df15w
        for s in sigs:
            date_to_orders.setdefault(s["date"], []).append((sym, s, df15w))
        sym_sig_count.append((sym, len(sigs), "ok"))

    total_sigs = sum(c for _, c, _ in sym_sig_count)
    print(f"Total signals across universe: {total_sigs}")
    print("Per-symbol signal count (top 10):")
    for sym, c, status in sorted(sym_sig_count, key=lambda x: -x[1])[:10]:
        print(f"  {sym:14s} {c:5d}  {status}")
    print()

    trades, curve = simulate_intraday(date_to_orders, bars_by_symbol)
    if not trades:
        print("NO TRADES")
        return
    tdf = pd.DataFrame(trades)
    cdf = pd.DataFrame(curve)
    final_equity = cdf.iloc[-1].equity
    total_ret = (final_equity / CAPITAL - 1) * 100
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days
    months = n_days / 30.44

    # Per-month roll-up
    monthly = tdf.groupby("month")["net"].sum().reset_index()
    monthly["ret_on_capital_pct"] = monthly["net"] / CAPITAL * 100
    # Per-month with compounding (rough): use equity at month-end
    cdf["month"] = pd.to_datetime(cdf["date"]).dt.strftime("%Y-%m")
    eom = cdf.groupby("month").tail(1).reset_index(drop=True)
    eom["prev_equity"] = eom["equity"].shift(1).fillna(CAPITAL)
    eom["ret_pct_compounded"] = (eom["equity"] / eom["prev_equity"] - 1) * 100

    wins = tdf[tdf.net > 0]
    losses = tdf[tdf.net <= 0]
    win_rate = len(wins) / len(tdf) * 100
    avg_win = wins.net.mean() if len(wins) else 0
    avg_loss = losses.net.mean() if len(losses) else 0
    pf = wins.net.sum() / abs(losses.net.sum()) if len(losses) else float("inf")
    # MaxDD
    cdf["peak"] = cdf["equity"].cummax()
    cdf["dd"]   = (cdf["equity"] / cdf["peak"] - 1) * 100
    maxdd = cdf["dd"].min()
    # Sharpe (daily, annualized)
    cdf["ret"] = cdf["equity"].pct_change().fillna(0)
    sharpe = cdf["ret"].mean() / cdf["ret"].std() * math.sqrt(252) if cdf["ret"].std() > 0 else 0

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Trades:           {len(tdf)}")
    print(f"Win rate:         {win_rate:.1f}%")
    print(f"Avg win:          INR {avg_win:,.0f}")
    print(f"Avg loss:         INR {avg_loss:,.0f}")
    print(f"Profit factor:    {pf:.2f}")
    print(f"Final equity:     INR {final_equity:,.0f}")
    print(f"Total return:     {total_ret:+.2f}%")
    print(f"Avg monthly:      {total_ret/months:+.2f}% (simple)")
    print(f"Compounded mo:    {((final_equity/CAPITAL)**(1/months) - 1)*100:+.2f}%/mo")
    print(f"Max drawdown:     {maxdd:.2f}%")
    print(f"Sharpe (annual):  {sharpe:.2f}")
    print()
    print("MONTHLY RETURNS (compounded, on month-start equity)")
    print(eom[["month", "equity", "ret_pct_compounded"]].to_string(index=False))
    print()
    print("EXIT REASON BREAKDOWN")
    print(tdf.groupby("reason").agg(n=("net","count"),
                                     total=("net","sum"),
                                     avg=("net","mean")).to_string())
    # Save artifacts
    out = Path("/app/logs/twenty_pct")
    out.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out / "trades.csv", index=False)
    cdf.to_csv(out / "equity_curve.csv", index=False)
    eom.to_csv(out / "monthly_returns.csv", index=False)
    print(f"\nArtifacts: {out}")


if __name__ == "__main__":
    main()
