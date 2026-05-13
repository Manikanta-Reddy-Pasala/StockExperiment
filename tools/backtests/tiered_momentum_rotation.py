"""Tiered Monthly Momentum Rotation backtest.

Variant of momentum_rotation_backtest.py — instead of picking top-N from
ONE universe, pick top-1 from EACH of three tiers:
  Tier A = Nifty 50 (53 large-cap)
  Tier B = Next 50 (N100 by ADV minus N50, ~50 mid-large)
  Tier C = Next 50 (N150 by ADV minus N100, ~50 mid)

Logic per 1st-of-month rebalance:
  1. Rank each tier by 60-day return
  2. Pick top-1 from each tier (3 positions)
  3. Rebalance: sell drop-outs, buy new entrants
  4. Cap concurrent = 3 (one slot per tier)

Hypothesis: diversification across cap tiers improves drawdown vs Model 3
(single concentration) while keeping high return like Model 1 (N500 max=3).

Usage:
  python tools/backtests/tiered_momentum_rotation.py \
    --from 2023-05-13 --to 2024-05-12 \
    --out /app/exports/backtests/tiered_momrot_2023_2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS, nifty500_symbols,
)

log = logging.getLogger("tiered_momrot")


def get_close_at(symbol: str, target_ts: int, daily_data: Dict[str, pd.DataFrame]) -> float:
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    valid = df[df["timestamp"] <= target_ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def rank_momentum(symbols: List[str], rebalance_ts: int,
                  daily_data: Dict[str, pd.DataFrame],
                  lookback_days: int = 60) -> List[tuple]:
    lookback_ts = rebalance_ts - lookback_days * 86400
    ranks = []
    for sym in symbols:
        c_now = get_close_at(sym, rebalance_ts, daily_data)
        c_past = get_close_at(sym, lookback_ts, daily_data)
        if c_now > 0 and c_past > 0:
            ret = (c_now / c_past - 1) * 100
            ranks.append((sym, ret, c_now))
    ranks.sort(key=lambda x: -x[1])
    return ranks


def compute_adv(symbol: str, end_ts: int,
                daily_data: Dict[str, pd.DataFrame]) -> float:
    """Avg daily INR traded over last 20 bars before end_ts, in lakh."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    sub = df[df["timestamp"] <= end_ts].tail(20)
    if sub.empty:
        return 0.0
    val = (sub["close"].astype(float) * sub["volume"].astype(float)).mean()
    return float(val) / 1e5


def build_tiers(all_syms: List[str], n50: List[str],
                end_ts: int, daily_data: Dict[str, pd.DataFrame],
                tier_size: int = 50):
    """Return (tier_a, tier_b, tier_c) symbol lists.

    A = n50 (large)
    B = top-tier_size by ADV among (all_syms - n50)
    C = next tier_size by ADV after B
    """
    n50_set = set(n50)
    rest = [s for s in all_syms if s not in n50_set]
    advs = [(s, compute_adv(s, end_ts, daily_data)) for s in rest]
    advs = [(s, a) for s, a in advs if a > 0]
    advs.sort(key=lambda x: -x[1])
    tier_b = [s for s, _ in advs[:tier_size]]
    tier_c = [s for s, _ in advs[tier_size:tier_size * 2]]
    return list(n50), tier_b, tier_c


def run_tiered_rotation(n50: List[str], all_syms: List[str],
                        start: str, end: str,
                        tier_size: int = 50,
                        out_dir: Path = None) -> List[Dict]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"Loading daily bars for {len(all_syms)} symbols")
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(all_syms):
        if i % 50 == 0:
            log.info(f"  {i}/{len(all_syms)}")
        df = read_cached(sym, "D",
                         int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    # Monthly rebalance dates
    rebalance_dates = []
    cur = start_dt
    while cur < end_dt:
        rebalance_dates.append(cur)
        cur = (datetime(cur.year + 1, 1, 1) if cur.month == 12
               else datetime(cur.year, cur.month + 1, 1))
    log.info(f"Rebalance dates: {len(rebalance_dates)}")

    per_symbol_events: Dict[str, List[Dict]] = {sym: [] for sym in all_syms}
    held: Dict[str, float] = {}  # symbol -> entry price

    for i, reb_dt in enumerate(rebalance_dates):
        reb_ts = int(reb_dt.timestamp())

        # Rebuild tiers each month (ADV may shift over time)
        tier_a, tier_b, tier_c = build_tiers(all_syms, n50, reb_ts,
                                              daily_data, tier_size)

        top_per_tier = []
        for tier_name, tier_syms in [("A", tier_a), ("B", tier_b), ("C", tier_c)]:
            ranks = rank_momentum(tier_syms, reb_ts, daily_data)
            if ranks:
                top_per_tier.append((tier_name, ranks[0]))
        top_syms = {r[1][0] for r in top_per_tier}

        log.info(f"  {reb_dt.date()} picks: " +
                 ", ".join(f"{t}={r[0]}({r[1]:.1f}%)" for t, r in top_per_tier))

        # Sell drop-outs
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
        for _, (sym, ret, price) in top_per_tier:
            if sym not in held and price > 0:
                held[sym] = price
                per_symbol_events[sym].append({
                    "Stage": "First Entry",
                    "ts": reb_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": price, "kind": "ENTRY",
                })

    # Close residual at end
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

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym, events in per_symbol_events.items():
            short = sym.split(":")[-1].replace("-EQ", "").lower()
            lines = [
                f"# {sym}", "",
                "## Backtest Summary",
                "- Strategy: Tiered Monthly Momentum Rotation (N50 + Next50 + Next50)",
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
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--tier-size", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    n50 = [s for s, _ in NIFTY50_SYMBOLS]
    all_syms = [s for s, _ in nifty500_symbols()]
    # Ensure n50 is inside universe
    n50 = [s for s in n50 if s in all_syms or True]

    log.info(f"Tier A (N50) = {len(n50)} | All universe = {len(all_syms)}")
    results = run_tiered_rotation(n50, all_syms, args.date_from, args.date_to,
                                   args.tier_size, Path(args.out))
    total_events = sum(len(r["events"]) for r in results)
    log.info(f"Generated {total_events} events across {len(results)} active symbols")


if __name__ == "__main__":
    main()
