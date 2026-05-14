"""Model D — Quality Momentum (Buy-and-Hold filtered).

Strategy:
  * Universe: N100.
  * Filter on each monthly rebalance: 1yr return > 0 AND 90d annualized vol < 30%.
  * Hold ALL qualified stocks equal-weight.
  * Rebalance monthly: exit stocks failing filter, enter newly-qualified.
  * Long-only, low turnover.

Outputs cap-sim-compatible per-symbol .md files.

Usage:
  python tools/backtests/model_quality.py --from 2023-05-13 --to 2024-05-12 --out /app/exports/backtests/model_d_2023_2024
"""
from __future__ import annotations

import argparse
import json
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
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS  # noqa: E402

log = logging.getLogger("model_d")


def load_n100() -> List[str]:
    path = ROOT / "logs" / "momrot" / "universes" / "n100_current.json"
    if path.exists():
        data = json.loads(path.read_text())
        return [s["symbol"] for s in data["stocks"]]
    return [s for s, _ in NIFTY50_SYMBOLS]


def get_close_at(df: pd.DataFrame, ts: int) -> float:
    valid = df[df["timestamp"] <= ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def ret_n_days(df: pd.DataFrame, ts: int, days: int) -> float:
    c_now = get_close_at(df, ts)
    c_then = get_close_at(df, ts - days * 86400)
    if c_now <= 0 or c_then <= 0:
        return 0.0
    return c_now / c_then - 1.0


def vol_90d(df: pd.DataFrame, ts: int) -> float:
    sub = df[df["timestamp"] <= ts].tail(95)
    if len(sub) < 50:
        return 999.0
    rets = np.log(sub["close"].astype(float)).diff().dropna()
    if len(rets) < 10:
        return 999.0
    return float(rets.std() * np.sqrt(252))


def monthly_dates(start: datetime, end: datetime) -> List[datetime]:
    out: List[datetime] = []
    cur = start
    while cur < end:
        out.append(cur)
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return out


def run(symbols: List[str], start: str, end: str, vol_max: float,
        out_dir: Path) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=400)

    log.info(f"Loading bars for {len(symbols)} N100 symbols")
    daily: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            log.info(f"  {i}/{len(symbols)}")
        df = read_cached(sym, "D",
                         int(warmup_dt.timestamp()),
                         int(end_dt.timestamp()))
        if df.empty or len(df) < 200:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        daily[sym] = df
    log.info(f"Loaded {len(daily)} symbols")

    rebal = monthly_dates(start_dt, end_dt)
    per_symbol: Dict[str, List[Dict]] = {s: [] for s in symbols}
    held: Dict[str, float] = {}

    for reb in rebal:
        ts = int(reb.timestamp())
        qualified: List[str] = []
        for sym, df in daily.items():
            r1y = ret_n_days(df, ts, 365)
            v90 = vol_90d(df, ts)
            if r1y > 0 and v90 < vol_max:
                qualified.append(sym)
        qset = set(qualified)

        # Exits
        for sym in list(held.keys()):
            if sym not in qset:
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

        # Entries
        for sym in qualified:
            if sym not in held:
                price = get_close_at(daily[sym], ts)
                if price <= 0:
                    continue
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
            f"- Strategy: Model D Quality Momentum (1y>0 & vol90<{vol_max})",
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
    ap.add_argument("--vol-max", type=float, default=0.30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    syms = load_n100()
    log.info(f"Universe N100 size: {len(syms)}")
    run(syms, args.date_from, args.date_to, args.vol_max, Path(args.out))


if __name__ == "__main__":
    main()
