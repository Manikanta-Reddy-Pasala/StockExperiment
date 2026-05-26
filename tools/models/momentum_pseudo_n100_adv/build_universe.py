"""Build PSEUDO-Nifty-100 universe via ADV ranking from N500.

This is the V1 / "aggressive" universe used by momentum_pseudo_n100_adv model:
top-100 stocks by 20-day Average Daily ₹ Value traded from the Nifty 500 list.

Difference vs `momentum_n100_top5_max1/build_universe.py`:
  - That model uses REAL NSE Nifty 100 from `src/data/symbols/nifty100.csv`
  - This one ranks dynamically by ADV — captures retail-volume mid-caps that
    real N100 excludes (BSE, MAZDOCK, NETWEB, GRSE, IRFC, IDEA, ITI, NBCC,
    PAYTM, COFORGE, COHANCE, HFCL, etc.)

Pipeline position (data_pull -> build_universe -> live_signal -> cron -> backtest):
  This is the universe-construction stage. It reads the daily OHLCV that
  data_pull.pull_daily_ohlcv() already cached for the N500 list, ranks every
  N500 symbol by 20-day ADV at a given point-in-time (`--end-date`), keeps the
  top-N, and writes a snapshot JSON. data_pull.refresh_universe() invokes this
  once per year (the May rebalance) and merges the resulting symbol list into
  yearly_universes.json, which live_signal then consults to pick the universe
  for a trading date. The Smallcap-250 subtraction and uptrend / MAX_PRICE
  filters described in the model spec are applied downstream (live_signal /
  the consumer of this snapshot), not here — this stage only does the ADV
  ranking against N500.

Usage:
  python tools/models/momentum_pseudo_n100_adv/build_universe.py --top 100 \
    --end-date 2025-05-13 \
    --out exports/backtests/pseudo_n100_2025-05-13.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.universes import nifty500_symbols  # noqa: E402

log = logging.getLogger("build_pseudo_n100")


def compute_adv(symbol: str, end_dt: datetime, days: int = 60) -> float:
    """Average daily traded value (ADV) for one symbol, expressed in ₹ lakh.

    Args:
        symbol: NSE symbol to score.
        end_dt: Point-in-time as-of date; only cached bars on/before this date
            are read, so the ranking is PIT-safe for the rebalance it feeds.
        days: Calendar-day lookback window used to fetch candles. Wider than
            the 20 trading days actually averaged so weekends/holidays still
            leave >=20 trading bars in the slice.

    Returns:
        Mean of (close * volume) over the last 20 trading days, divided by 1e5
        to convert ₹ into lakh. Returns 0.0 if the symbol has no cached data,
        the read fails, or fewer than 5 bars exist (treated as unrankable).

    Non-obvious logic:
        - Reads only from the local OHLCV cache (no network) using a Unix-epoch
          [start, end] window derived from `end_dt - days`.
        - `.tail(20)` takes the most recent 20 rows in the window (the 20-day
          ADV), independent of how many calendar days `days` spanned.
    """
    # Pull a calendar window wide enough to guarantee >=20 trading bars.
    start_dt = end_dt - timedelta(days=days)
    try:
        # Cache lookup keyed by epoch seconds; "D" = daily interval.
        df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    except Exception:
        return 0.0
    # Too little history to form a meaningful ADV -> unrankable.
    if df.empty or len(df) < 5:
        return 0.0
    # ₹ value traded per bar = close price * shares traded.
    df["value"] = df["close"].astype(float) * df["volume"].astype(float)
    # 20-day ADV; /1e5 converts rupees to lakh for compact thresholds.
    return float(df["value"].tail(20).mean()) / 1e5


def main():
    """CLI entrypoint: rank N500 by 20-day ADV and write the top-N snapshot.

    Args (parsed from argv):
        --top: How many top-ADV symbols to keep (default 100 = pseudo-N100).
        --end-date: PIT as-of date "YYYY-MM-DD" for the ranking (default today).
        --out: Destination JSON path for the universe snapshot (required).
        --min-adv-lakh: Minimum ADV (in ₹ lakh) a symbol must clear to be
            eligible at all; floors out thinly traded names before ranking.

    Returns:
        None. Side effect: writes a JSON snapshot to `--out` with metadata
        (generated_at, end_date, method, top_n) plus the ranked `stocks` list,
        each entry {symbol, name, adv_lakh}. This file is what
        data_pull.refresh_universe() reads and merges into yearly_universes.json.

    Non-obvious logic:
        - Universe source is the full Nifty 500 list; the "pseudo-N100" is the
          top-100 slice of that ranked by liquidity (ADV), not an index file.
        - Liquidity floor (`min_adv_lakh`) is applied per-symbol before sorting,
          so the final list can be shorter than `--top` if few names qualify.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-adv-lakh", type=float, default=50.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # PIT as-of date: parse the flag, else fall back to "now" for ad-hoc runs.
    end_dt = (datetime.strptime(args.end_date, "%Y-%m-%d")
              if args.end_date else datetime.now())

    # Candidate pool = full Nifty 500; the pseudo-N100 is its top-ADV slice.
    universe = nifty500_symbols()
    log.info(f"Ranking {len(universe)} N500 symbols by ADV @ {end_dt.date()}")

    rows = []
    for i, (sym, name) in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")  # periodic progress heartbeat
        adv = compute_adv(sym, end_dt)
        # Drop thinly traded names below the liquidity floor before ranking.
        if adv >= args.min_adv_lakh:
            rows.append({"symbol": sym, "name": name, "adv_lakh": adv})

    rows.sort(key=lambda r: -r["adv_lakh"])  # descending ADV -> most liquid first
    top_n = rows[:args.top]  # keep the top-N (pseudo-N100)

    out = {
        "generated_at": datetime.now().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "method": "Pseudo-N100: top-100 by 20-day ADV from NIFTY 500",
        "top_n": args.top,
        "stocks": top_n,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out}")
    log.info(f"Top 10: {[r['symbol'] for r in top_n[:10]]}")


if __name__ == "__main__":
    main()
