"""Monthly Momentum Rotation backtest.

Different mode from EMA crossover / ORB / selector approaches.

Rules:
- 1st of each month, rank N50/N100/N500 stocks by 60-day return
- Hold top-3 (equal weight) for the month
- Rebalance: sell stocks no longer in top-3, buy new entrants
- One-position-per-stock cap

Outputs cap-sim-compatible cycle .md files.

Usage:
  python tools/backtests/momentum_rotation_backtest.py \
    --universe nifty50 --from 2023-05-13 --to 2024-05-12 \
    --out /app/exports/backtests/momrot_n50_2023_2024
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
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS, nifty500_symbols  # noqa: E402


log = logging.getLogger("momrot")


def get_close_at(symbol: str, target_ts: int, daily_data: Dict[str, pd.DataFrame]) -> float:
    """Find close on or before target_ts."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    valid = df[df["timestamp"] <= target_ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def get_high_low_during(symbol: str, start_ts: int, end_ts: int,
                         daily_data: Dict[str, pd.DataFrame]):
    """Return (max_high, min_low) over [start, end]."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0, 0.0
    sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if sub.empty: return 0.0, 0.0
    return float(sub["high"].astype(float).max()), float(sub["low"].astype(float).min())


def rank_momentum(symbols: List[str], rebalance_ts: int,
                   daily_data: Dict[str, pd.DataFrame],
                   lookback_days: int = 60) -> List[tuple]:
    """Return [(symbol, 60d_return)] sorted descending."""
    lookback_ts = rebalance_ts - lookback_days * 86400
    ranks = []
    for sym in symbols:
        c_now = get_close_at(sym, rebalance_ts, daily_data)
        c_60d = get_close_at(sym, lookback_ts, daily_data)
        if c_now > 0 and c_60d > 0:
            ret = (c_now / c_60d - 1) * 100
            ranks.append((sym, ret, c_now))
    ranks.sort(key=lambda x: -x[1])
    return ranks


def run_momentum_rotation(universe: List[str], start: str, end: str,
                           top_n: int = 3, out_dir: Path = None) -> List[Dict]:
    """Run monthly momentum rotation, output per-stock cycle .md files."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    # Pre-load daily bars for all symbols
    log.info(f"Loading daily bars for {len(universe)} symbols")
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe):
        if i % 50 == 0: log.info(f"  {i}/{len(universe)}")
        df = read_cached(sym, "D",
                          int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    # Generate monthly rebalance dates
    rebalance_dates = []
    cur = start_dt
    while cur < end_dt:
        rebalance_dates.append(cur)
        # First of next month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    log.info(f"Rebalance dates: {len(rebalance_dates)}")

    # Track per-symbol events for cap-sim
    per_symbol_events: Dict[str, List[Dict]] = {sym: [] for sym in universe}
    held: Dict[str, float] = {}  # symbol -> entry price

    for i, reb_dt in enumerate(rebalance_dates):
        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_dt
        reb_ts = int(reb_dt.timestamp())
        next_reb_ts = int(next_reb.timestamp())

        ranks = rank_momentum(universe, reb_ts, daily_data)
        top = ranks[:top_n]
        top_syms = {r[0] for r in top}

        # Sell stocks no longer in top
        to_sell = [s for s in held if s not in top_syms]
        for sym in to_sell:
            entry = held.pop(sym)
            exit_price = get_close_at(sym, reb_ts, daily_data)
            if exit_price > 0:
                kind = "TARGET" if exit_price > entry else "STOP"
                stage = "Target hit" if kind == "TARGET" else "Stop hit"
                per_symbol_events[sym].append({
                    "Stage": stage,
                    "ts": reb_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": exit_price, "kind": kind,
                })

        # Buy new entrants
        for sym, ret, price in top:
            if sym not in held and price > 0:
                held[sym] = price
                per_symbol_events[sym].append({
                    "Stage": "First Entry",
                    "ts": reb_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": price, "kind": "ENTRY",
                })

    # Close any remaining positions at end
    for sym, entry in held.items():
        exit_price = get_close_at(sym, int(end_dt.timestamp()), daily_data)
        if exit_price > 0:
            kind = "TARGET" if exit_price > entry else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            per_symbol_events[sym].append({
                "Stage": stage,
                "ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "price": exit_price, "kind": kind,
            })

    # Write per-stock .md files
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym, events in per_symbol_events.items():
            short = sym.split(":")[-1].replace("-EQ", "").lower()
            lines = [
                f"# {sym}", "",
                f"## Backtest Summary",
                f"- Strategy: Monthly Momentum Rotation top-{top_n}",
                f"- Entries: {sum(1 for e in events if e['kind'] == 'ENTRY')}",
                "",
                "## Strategy Cycles", "",
                "| Stage | Timestamp | Price | sl | target | note |",
                "|---|---|---|---|---|---|",
            ]
            for e in events:
                lines.append(f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
            (out_dir / f"{short}.md").write_text("\n".join(lines))

    return [
        {"symbol": sym, "events": ev}
        for sym, ev in per_symbol_events.items() if ev
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty50",
                    choices=["nifty50", "nifty500"])
    ap.add_argument("--universe-file", default=None)
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--out", required=True)
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
        symbols = [s for s, _ in nifty500_symbols()]

    log.info(f"Universe: {len(symbols)} symbols")
    results = run_momentum_rotation(symbols, args.date_from, args.date_to,
                                       args.top_n, Path(args.out))
    total_events = sum(len(r["events"]) for r in results)
    log.info(f"Generated {total_events} events across {len(results)} active symbols")


if __name__ == "__main__":
    main()
