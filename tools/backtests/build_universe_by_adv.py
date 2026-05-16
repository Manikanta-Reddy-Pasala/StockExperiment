"""Build pseudo-Nifty-100 / Nifty-150 universes from N500 by ADV ranking.

Since we don't have official N100/N150 constituent lists in repo, we
approximate by:
  1. Compute 20-day ADV (avg ₹ traded daily) for every N500 stock
  2. Rank by ADV descending
  3. Top-100 = pseudo-N100 (large + liquid)
  4. Top-150 = pseudo-N150 (large + mid liquid)

Output: selector JSON compatible with signal_generator --universe-file.

Usage:
  python tools/backtests/build_universe_by_adv.py --top 100 \
    --end-date 2025-05-12 \
    --out exports/backtests/universes/nifty100_eq_2025-05-12.json
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
from tools.backtests.universes import nifty500_symbols  # noqa: E402

log = logging.getLogger("build_universe")


def compute_adv(symbol: str, end_dt: datetime, days: int = 60) -> float:
    """Return avg daily ₹ value traded over last `days` calendar days, in lakh."""
    start_dt = end_dt - timedelta(days=days)
    df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    if df.empty or len(df) < 5:
        return 0.0
    df["value"] = df["close"].astype(float) * df["volume"].astype(float)
    return float(df["value"].tail(20).mean()) / 1e5   # lakh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--end-date", default=None,
                    help="YYYY-MM-DD (default: today)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-adv-lakh", type=float, default=50.0,
                    help="Floor liquidity filter")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    end_dt = (datetime.strptime(args.end_date, "%Y-%m-%d")
              if args.end_date else datetime.now())

    universe = nifty500_symbols()
    log.info(f"Computing ADV for {len(universe)} N500 symbols as of {end_dt.date()}")

    rows = []
    for i, (sym, name) in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")
        adv = compute_adv(sym, end_dt)
        if adv >= args.min_adv_lakh:
            rows.append({"symbol": sym, "name": name, "adv_lakh": adv})

    rows.sort(key=lambda r: -r["adv_lakh"])
    top_n = rows[:args.top]

    out = {
        "generated_at": datetime.now().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "method": "ADV-ranked from Nifty 500",
        "top_n": args.top,
        "stocks": top_n,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out}")
    log.info(f"Top 10: {[r['symbol'].split(':')[-1].replace('-EQ','') for r in top_n[:10]]}")


if __name__ == "__main__":
    main()
