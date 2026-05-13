"""Momentum Rotation backtest with rebalance frequency + safety filters.

Variant of momentum_rotation_backtest.py that adds:
  --frequency {daily, weekly, monthly}
  --safety (enable safety filters)

Safety filters when --safety is on:
  1. Liquidity floor: pick must have 20d ADV >= ₹50 cr (5000 lakh)
                      → fall through to rank-2, rank-3, ... if rank-1 fails
  2. Volatility ceiling: pick must have 20d daily-return stdev <= 4.0%
                          (~ 60% annualized) → fall through likewise
  3. Catastrophic stop: between rebalances, if held drops ≥ 15% from entry,
                        exit immediately and stay cash until next rebalance
  4. Index circuit: if Nifty50 (^NSEI) had a single-day return <= -5% in
                    last 2 trading days at rebalance time, SKIP new entries
                    (existing positions retained)

All other rules identical:
  - 60d return ranker
  - top-N (equal weight)
  - no SL / target on ranking (except catastrophic stop)
  - ranking-driven exit (sell when no longer in top-N)

Outputs cap-sim-compatible cycle .md files.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS, nifty500_symbols  # noqa: E402


log = logging.getLogger("momrot_freq")


# ---- Safety filter thresholds ---------------------------------------------
MIN_ADV_LAKH = 5000.0          # ₹50 cr = 5000 lakh
MAX_VOL_DAILY = 4.0            # 4% daily stdev ~ 60% annualized
CATASTROPHIC_STOP_PCT = -15.0  # exit if held drops 15% from entry
INDEX_CIRCUIT_PCT = -5.0       # single-day drop triggering 2-day pause
INDEX_CIRCUIT_LOOKBACK = 2     # look back N trading days


def get_close_at(symbol: str, target_ts: int, daily_data: Dict[str, pd.DataFrame]) -> float:
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    valid = df[df["timestamp"] <= target_ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def compute_adv_lakh(symbol: str, target_ts: int,
                     daily_data: Dict[str, pd.DataFrame], n: int = 20) -> float:
    """Average daily turnover in ₹ lakh over last n bars at or before target_ts."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    sub = df[df["timestamp"] <= target_ts].tail(n)
    if sub.empty:
        return 0.0
    val = (sub["close"].astype(float) * sub["volume"].astype(float)).mean()
    return float(val) / 1e5


def compute_vol_daily(symbol: str, target_ts: int,
                      daily_data: Dict[str, pd.DataFrame], n: int = 20) -> float:
    """Daily return stdev (%) over last n bars."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    sub = df[df["timestamp"] <= target_ts].tail(n + 1)
    if len(sub) < n:
        return 0.0
    closes = sub["close"].astype(float).values
    rets = (closes[1:] / closes[:-1] - 1) * 100
    return float(pd.Series(rets).std())


def index_circuit_hit(nifty_df: Optional[pd.DataFrame], target_ts: int) -> bool:
    """True if Nifty50 had a day ≤ INDEX_CIRCUIT_PCT in last N bars."""
    if nifty_df is None or nifty_df.empty:
        return False
    sub = nifty_df[nifty_df["timestamp"] <= target_ts].tail(INDEX_CIRCUIT_LOOKBACK + 1)
    if len(sub) < 2:
        return False
    closes = sub["close"].astype(float).values
    day_rets = (closes[1:] / closes[:-1] - 1) * 100
    return any(r <= INDEX_CIRCUIT_PCT for r in day_rets)


def rank_momentum(symbols: List[str], rebalance_ts: int,
                   daily_data: Dict[str, pd.DataFrame],
                   lookback_days: int = 60) -> List[tuple]:
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


def pick_with_safety(ranks: List[tuple], target_ts: int,
                     daily_data: Dict[str, pd.DataFrame], top_n: int,
                     safety: bool) -> List[tuple]:
    """Return up to top_n picks. With safety: skip picks failing ADV or vol filter."""
    if not safety:
        return ranks[:top_n]
    picks = []
    for r in ranks:
        sym = r[0]
        adv = compute_adv_lakh(sym, target_ts, daily_data)
        if adv < MIN_ADV_LAKH:
            continue
        vol = compute_vol_daily(sym, target_ts, daily_data)
        if vol > MAX_VOL_DAILY:
            continue
        picks.append(r)
        if len(picks) >= top_n:
            break
    return picks


def generate_rebalance_dates(start_dt: datetime, end_dt: datetime,
                              frequency: str) -> List[datetime]:
    """Generate rebalance dates based on frequency."""
    dates: List[datetime] = []
    if frequency == "monthly":
        cur = start_dt
        while cur < end_dt:
            dates.append(cur)
            if cur.month == 12:
                cur = datetime(cur.year + 1, 1, 1)
            else:
                cur = datetime(cur.year, cur.month + 1, 1)
    elif frequency == "weekly":
        cur = start_dt
        if cur.weekday() != 0:
            cur = cur + timedelta(days=(7 - cur.weekday()) % 7)
            if cur.weekday() != 0:
                cur = cur + timedelta(days=(0 - cur.weekday()) % 7)
        dates.append(start_dt)
        while cur < end_dt:
            if cur != start_dt:
                dates.append(cur)
            cur = cur + timedelta(days=7)
    elif frequency == "daily":
        cur = start_dt
        while cur < end_dt:
            if cur.weekday() < 5:
                dates.append(cur)
            cur = cur + timedelta(days=1)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    return dates


def scan_catastrophic_exit(sym: str, entry_price: float,
                            from_ts: int, to_ts: int,
                            daily_data: Dict[str, pd.DataFrame]) -> Optional[Tuple[int, float]]:
    """Scan held position between rebalances. Return (ts, price) of catastrophic exit if hit."""
    df = daily_data.get(sym)
    if df is None or df.empty:
        return None
    sub = df[(df["timestamp"] > from_ts) & (df["timestamp"] <= to_ts)]
    threshold = entry_price * (1 + CATASTROPHIC_STOP_PCT / 100)
    for r in sub.itertuples():
        if float(r.close) <= threshold:
            return (int(r.timestamp), float(r.close))
    return None


def run_momentum_rotation(universe: List[str], start: str, end: str,
                           top_n: int = 3, frequency: str = "monthly",
                           safety: bool = False,
                           out_dir: Path = None) -> List[Dict]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"Loading daily bars for {len(universe)} symbols (safety={safety})")
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")
        df = read_cached(sym, "D",
                          int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    # Load Nifty50 index for circuit check
    nifty_df: Optional[pd.DataFrame] = None
    if safety:
        ndf = read_cached("^NSEI", "D",
                           int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not ndf.empty:
            nifty_df = ndf.sort_values("timestamp").reset_index(drop=True)
            log.info(f"Loaded Nifty50 {len(nifty_df)} bars for index circuit")
        else:
            log.warning("Nifty50 data unavailable - index circuit disabled")

    rebalance_dates = generate_rebalance_dates(start_dt, end_dt, frequency)
    log.info(f"Frequency={frequency} -> {len(rebalance_dates)} rebalance dates")

    per_symbol_events: Dict[str, List[Dict]] = {sym: [] for sym in universe}
    held: Dict[str, float] = {}  # symbol -> entry price
    skipped_catastrophic: List[str] = []  # symbols currently in catastrophic-cooldown (cash)

    for i, reb_dt in enumerate(rebalance_dates):
        next_reb_dt = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_dt
        reb_ts = int(reb_dt.timestamp())
        next_reb_ts = int(next_reb_dt.timestamp())

        # Safety check 4: index circuit at this rebalance date
        circuit = safety and index_circuit_hit(nifty_df, reb_ts)

        ranks = rank_momentum(universe, reb_ts, daily_data)
        if circuit:
            top = []  # no new entries this rebalance
            log.info(f"  {reb_dt.date()} index circuit hit, skipping new entries")
        else:
            top = pick_with_safety(ranks, reb_ts, daily_data, top_n, safety)
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

        # Safety check 3: scan held positions for catastrophic drop between now and next rebalance
        if safety:
            to_emergency_exit = []
            for sym, entry_price in list(held.items()):
                hit = scan_catastrophic_exit(sym, entry_price, reb_ts, next_reb_ts, daily_data)
                if hit is not None:
                    hit_ts, hit_price = hit
                    hit_dt = datetime.fromtimestamp(hit_ts)
                    per_symbol_events[sym].append({
                        "Stage": "Stop hit",
                        "ts": hit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "price": hit_price, "kind": "STOP",
                    })
                    to_emergency_exit.append(sym)
            for sym in to_emergency_exit:
                held.pop(sym)

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

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym, events in per_symbol_events.items():
            short = sym.split(":")[-1].replace("-EQ", "").lower()
            lines = [
                f"# {sym}", "",
                f"## Backtest Summary",
                f"- Strategy: {frequency.capitalize()} Momentum Rotation top-{top_n} safety={safety}",
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
    ap.add_argument("--frequency", default="monthly",
                    choices=["daily", "weekly", "monthly"])
    ap.add_argument("--safety", action="store_true",
                    help="Enable liquidity + vol + catastrophic-stop + index-circuit filters")
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
                                       args.top_n, args.frequency, args.safety,
                                       Path(args.out))
    total_events = sum(len(r["events"]) for r in results)
    log.info(f"Generated {total_events} events across {len(results)} active symbols")


if __name__ == "__main__":
    main()
