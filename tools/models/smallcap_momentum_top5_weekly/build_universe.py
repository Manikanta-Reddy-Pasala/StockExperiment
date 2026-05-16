"""Build pseudo-Smallcap-250 universe from N500 by ADV ranking.

Rationale: N500 includes large (top-50), mid (50-200), and smallcaps (200+).
Real NSE Smallcap-250 = market-cap-ranked rows 251-500 of Nifty 500.
We approximate by 20-day ADV ranking — skip top SKIP_LARGE (large caps),
keep next TOP_N (small + lower-mid liquidity).

ADV-based ranking correlates well with market-cap ranking on Indian
exchanges + adds liquidity floor naturally.

Usage:
  python tools/models/smallcap_momentum_top5_weekly/build_universe.py \
    --skip-large 50 --top 200 \
    --out /app/logs/momrot/universes/smallcap_current.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.universes import nifty500_symbols  # noqa: E402

log = logging.getLogger("smallcap_build_universe")


def compute_adv(symbol: str, end_dt: datetime, days: int = 60) -> float:
    """20-day avg ₹ value traded, in lakh."""
    start_dt = end_dt - timedelta(days=days)
    df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    if df.empty or len(df) < 5:
        return 0.0
    df["value"] = df["close"].astype(float) * df["volume"].astype(float)
    return float(df["value"].tail(20).mean()) / 1e5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-large", type=int, default=50,
                    help="Skip top-N by ADV (large caps). Default 50.")
    ap.add_argument("--top", type=int, default=200,
                    help="Pick next N after skip-large. Default 200.")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-adv-lakh", type=float, default=20.0,
                    help="Liquidity floor (₹ lakh/day). Default 20.")
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
    # Skip top large caps, take the smallcap slice
    smallcap_rows = rows[args.skip_large:args.skip_large + args.top]

    out = {
        "generated_at": datetime.now().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "method": f"ADV-ranked, skip top-{args.skip_large} large caps, "
                  f"keep next {args.top}",
        "skip_large": args.skip_large,
        "top_n": args.top,
        "stocks": smallcap_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out} ({len(smallcap_rows)} symbols)")
    log.info(f"Top 10 smallcap by ADV: "
             f"{[r['symbol'] for r in smallcap_rows[:10]]}")


if __name__ == "__main__":
    main()
