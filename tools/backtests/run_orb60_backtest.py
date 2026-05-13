"""ORB-60 (Opening Range Breakout, 1-hour window) backtest on 15m bars.

Rules (mechanical):
  1. ORB window: 09:15-10:14 IST (first 1h, 4× 15m bars)
  2. ORB high = max(high) over those 4 bars
  3. ORB low  = min(low) over those 4 bars
  4. ATR(14) on daily bars for SL/target sizing
  5. ENTRY long: any 15m bar after 10:15 closes above ORB high
                AND volume > 1.5 × prev-1h avg
  6. ENTRY short: any bar closes below ORB low + volume
  7. SL: opposite side of ORB (long SL = ORB low, short SL = ORB high)
  8. Target: entry ± (ATR × 1.5)
  9. EOD exit: close all open at 15:20 IST regardless
  10. One entry per side per day

Outputs per-stock .md with Strategy Cycles for realistic_capital_sim_v2.

Usage:
  python tools/backtests/run_orb60_backtest.py \
    --universe nifty50 --from 2025-05-13 --to 2026-05-12 \
    --out /app/exports/backtests/orb60_2025_2026
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS  # noqa: E402

log = logging.getLogger("orb60")


def run_orb60(df_15m: pd.DataFrame, df_daily: pd.DataFrame,
               vol_mult: float = 1.5, target_atr: float = 1.5) -> List[Dict]:
    """Run ORB-60 on a single symbol. Returns event list."""
    events = []
    if df_15m.empty or df_daily.empty:
        return events
    df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)
    # Bars stored in UTC. NSE market is IST (UTC+5:30).
    # IST 09:15 = UTC 03:45, IST 15:30 = UTC 10:00
    df_15m["dt"] = pd.to_datetime(df_15m["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    df_15m["date"] = df_15m["dt"].dt.date
    df_15m["hm"] = df_15m["dt"].dt.strftime("%H:%M")

    # Daily ATR for target sizing
    df_daily = df_daily.sort_values("timestamp").reset_index(drop=True)
    df_daily["dt"] = pd.to_datetime(df_daily["timestamp"], unit="s")
    df_daily["date"] = df_daily["dt"].dt.date
    df_daily["high"] = df_daily["high"].astype(float)
    df_daily["low"] = df_daily["low"].astype(float)
    df_daily["close"] = df_daily["close"].astype(float)
    tr = pd.concat([
        df_daily["high"] - df_daily["low"],
        (df_daily["high"] - df_daily["close"].shift()).abs(),
        (df_daily["low"] - df_daily["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df_daily["atr14"] = tr.rolling(14).mean()
    atr_by_date = dict(zip(df_daily["date"], df_daily["atr14"]))

    # Group 15m bars by date
    for date, group in df_15m.groupby("date"):
        if date not in atr_by_date or pd.isna(atr_by_date[date]):
            continue
        group = group.sort_values("timestamp").reset_index(drop=True)

        # ORB window: 09:15-10:14 (4 bars: 09:15, 09:30, 09:45, 10:00)
        orb_bars = group[(group["hm"] >= "09:15") & (group["hm"] < "10:15")]
        if len(orb_bars) < 3:
            continue
        orb_high = float(orb_bars["high"].astype(float).max())
        orb_low = float(orb_bars["low"].astype(float).min())
        orb_vol_avg = float(orb_bars["volume"].astype(float).mean())

        # Trade window: 10:15 onwards until 15:20
        trade_bars = group[(group["hm"] >= "10:15") & (group["hm"] <= "15:15")]
        if trade_bars.empty:
            continue

        atr = atr_by_date[date]
        long_done = False
        short_done = False
        in_long = None
        in_short = None

        for _, b in trade_bars.iterrows():
            bar_close = float(b["close"])
            bar_high = float(b["high"])
            bar_low = float(b["low"])
            bar_vol = float(b["volume"])
            ts_str = b["dt"].strftime("%Y-%m-%d %H:%M:%S")

            # Manage open longs first
            if in_long is not None:
                if bar_high >= in_long["target"]:
                    events.append({"Stage": "Target hit", "ts": ts_str,
                                    "price": in_long["target"], "kind": "TARGET"})
                    in_long = None
                elif bar_low <= in_long["sl"]:
                    events.append({"Stage": "Stop hit", "ts": ts_str,
                                    "price": in_long["sl"], "kind": "STOP"})
                    in_long = None

            if in_short is not None:
                if bar_low <= in_short["target"]:
                    events.append({"Stage": "Target hit", "ts": ts_str,
                                    "price": in_short["target"], "kind": "TARGET"})
                    in_short = None
                elif bar_high >= in_short["sl"]:
                    events.append({"Stage": "Stop hit", "ts": ts_str,
                                    "price": in_short["sl"], "kind": "STOP"})
                    in_short = None

            # New entries (one per side per day)
            if (not long_done and in_long is None and
                bar_close > orb_high and bar_vol > vol_mult * orb_vol_avg):
                entry = bar_close
                sl = orb_low
                target = entry + target_atr * atr
                in_long = {"entry": entry, "sl": sl, "target": target}
                long_done = True
                events.append({"Stage": "First Entry", "ts": ts_str,
                                "price": entry, "kind": "ENTRY"})

            if (not short_done and in_short is None and
                bar_close < orb_low and bar_vol > vol_mult * orb_vol_avg):
                entry = bar_close
                sl = orb_high
                target = entry - target_atr * atr
                in_short = {"entry": entry, "sl": sl, "target": target}
                short_done = True
                events.append({"Stage": "First Entry", "ts": ts_str,
                                "price": entry, "kind": "ENTRY"})

        # EOD force close
        if in_long is not None or in_short is not None:
            last = group.iloc[-1]
            last_close = float(last["close"])
            last_ts = last["dt"].strftime("%Y-%m-%d %H:%M:%S")
            if in_long is not None:
                pnl_pos = last_close >= in_long["entry"]
                events.append({"Stage": "Target hit" if pnl_pos else "Stop hit",
                                "ts": last_ts, "price": last_close,
                                "kind": "TARGET" if pnl_pos else "STOP"})
            if in_short is not None:
                pnl_pos = last_close <= in_short["entry"]
                events.append({"Stage": "Target hit" if pnl_pos else "Stop hit",
                                "ts": last_ts, "price": last_close,
                                "kind": "TARGET" if pnl_pos else "STOP"})

    return events


def render_per_stock_md(symbol: str, events: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {symbol}\n",
        f"## Backtest Summary\n",
        f"- Strategy: ORB-60 (1H opening range on 15m bars)",
        f"- Entries: {sum(1 for e in events if e['kind'] == 'ENTRY')}",
        f"- Closes: {sum(1 for e in events if e['kind'] in ('TARGET', 'STOP'))}",
        "",
        "## Strategy Cycles",
        "",
        "| Stage | Timestamp | Price | sl | target | note |",
        "|---|---|---|---|---|---|",
    ]
    for e in events:
        lines.append(f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
    (out_dir / f"{symbol.lower()}.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty50",
                    choices=["nifty50", "nifty500"])
    ap.add_argument("--universe-file", default=None,
                    help="JSON file with {'stocks':[{'symbol':...}]}. Overrides --universe.")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--target-atr", type=float, default=1.5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.universe_file:
        with open(args.universe_file) as f:
            data = json.load(f)
        symbols = [s["symbol"] for s in data["stocks"]]
    elif args.universe == "nifty50":
        symbols = [s for s, _ in NIFTY50_SYMBOLS]
    else:
        from tools.backtests.run_ema_200_400_backtest import nifty500_symbols
        symbols = [s for s, _ in nifty500_symbols()]

    start_ts = int(datetime.strptime(args.date_from, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(args.date_to, "%Y-%m-%d").timestamp())
    warmup_start = int((datetime.strptime(args.date_from, "%Y-%m-%d") -
                         timedelta(days=60)).timestamp())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_entries = 0
    total_closes = 0
    for i, sym in enumerate(symbols):
        df_15m = read_cached(sym, "15m", start_ts, end_ts)
        df_daily = read_cached(sym, "D", warmup_start, end_ts)
        if df_15m.empty:
            log.info(f"{sym}: no 15m data")
            continue
        events = run_orb60(df_15m, df_daily, args.vol_mult, args.target_atr)
        total_entries += sum(1 for e in events if e["kind"] == "ENTRY")
        total_closes += sum(1 for e in events if e["kind"] in ("TARGET", "STOP"))
        render_per_stock_md(sym, events, out_dir)
        log.info(f"{sym}: {len(events)} events "
                 f"({sum(1 for e in events if e['kind'] == 'ENTRY')} entries)")

    log.info(f"Total: {total_entries} entries / {total_closes} closes "
             f"across {len(symbols)} symbols")


if __name__ == "__main__":
    main()
