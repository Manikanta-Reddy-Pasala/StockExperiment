"""Model C — Sector Rotation (proxy via N500 industry buckets).

We do NOT have sector indices in historical_data, so we approximate sector
returns by equal-weighting all N500 constituents per industry (from
load_nifty500_with_meta).

Strategy:
  * Monthly rebalance (1st of month).
  * Compute each industry's 60-day return as mean of constituent 60d returns.
  * Pick top-3 industries.
  * Allocate 33% to each industry. Within each industry, buy top-2 stocks
    by ADV from the N100 ranking (proxy for sector leaders).
  * Equal weight across 6 positions max (3 sectors x 2 stocks).
  * Rebalance: sell stocks whose industry left the top-3, or where the
    stock is no longer the leader; buy new entrants.

Outputs cap-sim-compatible per-symbol .md files.

Usage:
  python tools/backtests/model_sector_rot.py --from 2023-05-13 --to 2024-05-12 --out /app/exports/backtests/model_c_2023_2024
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
from src.services.data.nifty500_universe import load_nifty500_with_meta  # noqa: E402

log = logging.getLogger("model_c")


def load_universe_with_industry() -> Dict[str, str]:
    """Map plain ticker -> industry."""
    rows = load_nifty500_with_meta()
    out: Dict[str, str] = {}
    for fyers_sym, _name, industry in rows:
        plain = fyers_sym.replace("NSE:", "").replace("-EQ", "")
        out[plain] = industry
    return out


def load_n100_adv() -> Dict[str, float]:
    path = ROOT / "logs" / "momrot" / "universes" / "n100_current.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {s["symbol"]: float(s.get("adv_lakh", 0.0)) for s in data["stocks"]}


def get_close_at(df: pd.DataFrame, ts: int) -> float:
    valid = df[df["timestamp"] <= ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def ret_60d(df: pd.DataFrame, ts: int, lookback_days: int = 60) -> float:
    look_ts = ts - lookback_days * 86400
    c_now = get_close_at(df, ts)
    c_then = get_close_at(df, look_ts)
    if c_now <= 0 or c_then <= 0:
        return 0.0
    return c_now / c_then - 1.0


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


def run(start: str, end: str, top_sectors: int, picks_per_sector: int,
        out_dir: Path) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    sector_map = load_universe_with_industry()
    adv_map = load_n100_adv()
    symbols = list(sector_map.keys())

    log.info(f"Loading bars for {len(symbols)} N500 symbols")
    daily: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            log.info(f"  {i}/{len(symbols)}")
        df = read_cached(sym, "D",
                         int(warmup_dt.timestamp()),
                         int(end_dt.timestamp()))
        if df.empty or len(df) < 60:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        daily[sym] = df
    log.info(f"Loaded {len(daily)} symbols")

    rebal = monthly_dates(start_dt, end_dt)
    per_symbol: Dict[str, List[Dict]] = {s: [] for s in symbols}
    held: Dict[str, float] = {}

    for reb in rebal:
        ts = int(reb.timestamp())
        # 1. Compute industry returns
        ind_rets: Dict[str, List[float]] = {}
        for sym, df in daily.items():
            r = ret_60d(df, ts)
            if r == 0.0:
                continue
            ind = sector_map.get(sym)
            if not ind:
                continue
            ind_rets.setdefault(ind, []).append(r)
        ind_avg = {k: sum(v) / len(v) for k, v in ind_rets.items() if len(v) >= 3}
        top_inds = sorted(ind_avg.items(), key=lambda x: -x[1])[:top_sectors]
        top_ind_names = {x[0] for x in top_inds}

        # 2. Pick top-picks_per_sector by ADV inside each top industry
        target_positions: List[str] = []
        for ind, _r in top_inds:
            # candidates in this industry with valid price
            cands = []
            for sym, df in daily.items():
                if sector_map.get(sym) != ind:
                    continue
                px = get_close_at(df, ts)
                if px <= 0:
                    continue
                # prefer ADV from N100; otherwise use recent traded value proxy
                adv = adv_map.get(sym, 0.0)
                if adv <= 0:
                    # Use last 20-day avg traded value as proxy
                    sub = df[df["timestamp"] <= ts].tail(20)
                    if not sub.empty:
                        adv = float((sub["close"] * sub["volume"]).mean()) / 1e5
                cands.append((sym, adv, px))
            cands.sort(key=lambda x: -x[1])
            for sym, _adv, _px in cands[:picks_per_sector]:
                target_positions.append(sym)

        target_set = set(target_positions)

        # Exits
        for sym in list(held.keys()):
            if sym not in target_set:
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
        for sym in target_positions:
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

    # Close remaining at end
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
            f"- Strategy: Model C Sector Rotation (top {top_sectors} inds, top {picks_per_sector} by ADV)",
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
    ap.add_argument("--top-sectors", type=int, default=3)
    ap.add_argument("--picks-per-sector", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(args.date_from, args.date_to, args.top_sectors, args.picks_per_sector,
        Path(args.out))


if __name__ == "__main__":
    main()
