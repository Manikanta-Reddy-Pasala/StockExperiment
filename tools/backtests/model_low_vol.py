"""Model B — Low Volatility (N500).

Strategy:
  * Universe: N500 (load_nifty500_with_meta).
  * Quarterly rebalance (Jan/Apr/Jul/Oct 1st or first trading day in window).
  * Rank by 90-day historical volatility ascending; hold lowest-vol 20.
  * Equal weight. Exit when stock falls out of top-20 on rebalance, then enter new.
  * Long-only.

Outputs cap-sim-compatible per-symbol .md files.

Usage:
  python tools/backtests/model_low_vol.py --from 2023-05-13 --to 2024-05-12 --out /app/exports/backtests/model_b_2023_2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from src.services.data.nifty500_universe import load_nifty500_with_meta  # noqa: E402

log = logging.getLogger("model_b")


def load_n500() -> List[str]:
    rows = load_nifty500_with_meta()
    return [r[0].replace("NSE:", "").replace("-EQ", "") for r in rows]


def hist_vol(df: pd.DataFrame, asof_ts: int, lookback: int = 90) -> float:
    sub = df[df["timestamp"] <= asof_ts].tail(lookback + 5)
    if len(sub) < lookback // 2:
        return 0.0
    rets = np.log(sub["close"].astype(float)).diff().dropna()
    if len(rets) < 10:
        return 0.0
    return float(rets.std() * np.sqrt(252))


def get_close_at(df: pd.DataFrame, ts: int) -> float:
    valid = df[df["timestamp"] <= ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def quarterly_rebalance_dates(start: datetime, end: datetime) -> List[datetime]:
    """Generate rebalance dates: start date, then Jan/Apr/Jul/Oct 1st."""
    dates = [start]
    year = start.year
    for q in [(1, 1), (4, 1), (7, 1), (10, 1)]:
        for y in range(start.year, end.year + 2):
            try:
                d = datetime(y, q[0], q[1])
            except ValueError:
                continue
            if start < d < end:
                dates.append(d)
    dates = sorted(set(dates))
    return [d for d in dates if d <= end]


def run(symbols: List[str], start: str, end: str, hold_n: int,
        out_dir: Path) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=120)

    log.info(f"Loading bars for {len(symbols)} symbols")
    daily: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            log.info(f"  {i}/{len(symbols)}")
        df = read_cached(sym, "D",
                         int(warmup_dt.timestamp()),
                         int(end_dt.timestamp()))
        if df.empty or len(df) < 90:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        daily[sym] = df

    log.info(f"Symbols with data: {len(daily)}")
    rebal = quarterly_rebalance_dates(start_dt, end_dt)
    log.info(f"Rebalance dates: {[d.date().isoformat() for d in rebal]}")

    per_symbol: Dict[str, List[Dict]] = {s: [] for s in symbols}
    held: Dict[str, float] = {}  # symbol -> entry price

    for reb in rebal:
        ts = int(reb.timestamp())
        # Compute vol for every symbol
        vols = []
        for sym, df in daily.items():
            v = hist_vol(df, ts, 90)
            c = get_close_at(df, ts)
            if v > 0 and c > 0:
                vols.append((sym, v, c))
        vols.sort(key=lambda x: x[1])
        top = vols[:hold_n]
        top_syms = {x[0] for x in top}

        # Exits — drop those no longer in top set
        for sym in list(held.keys()):
            if sym not in top_syms:
                entry = held.pop(sym)
                exit_price = get_close_at(daily[sym], ts)
                if exit_price > 0:
                    kind = "TARGET" if exit_price > entry else "STOP"
                    stage = "Target hit" if kind == "TARGET" else "Stop hit"
                    per_symbol[sym].append({
                        "Stage": stage,
                        "ts": reb.strftime("%Y-%m-%d %H:%M:%S"),
                        "price": exit_price,
                    })

        # Entries — new top-N members
        for sym, _v, price in top:
            if sym not in held:
                held[sym] = price
                per_symbol[sym].append({
                    "Stage": "First Entry",
                    "ts": reb.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": price,
                })

    # Close at end
    end_ts = int(end_dt.timestamp())
    for sym, entry in held.items():
        exit_price = get_close_at(daily[sym], end_ts)
        if exit_price > 0:
            kind = "TARGET" if exit_price > entry else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            per_symbol[sym].append({
                "Stage": stage,
                "ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "price": exit_price,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sym, events in per_symbol.items():
        if not events:
            continue
        short = sym.split(":")[-1].replace("-EQ", "").lower()
        lines = [
            f"# {sym}", "",
            "## Backtest Summary",
            f"- Strategy: Model B Low Volatility (90d vol, top {hold_n}, quarterly)",
            f"- Entries: {sum(1 for e in events if 'Entry' in e['Stage'])}",
            "",
            "## Strategy Cycles", "",
            "| Stage | Timestamp | Price | sl | target | note |",
            "|---|---|---|---|---|---|",
        ]
        for e in events:
            lines.append(f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
        (out_dir / f"{short}.md").write_text("\n".join(lines))
        written += 1
    log.info(f"Wrote {written} per-symbol files to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    syms = load_n500()
    log.info(f"Universe N500 size: {len(syms)}")
    run(syms, args.date_from, args.date_to, args.hold, Path(args.out))


if __name__ == "__main__":
    main()
