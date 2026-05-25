"""Build midcap_narrow universe from N500 by ADV ranking.

Method (MUST match backtest.py's inline universe selection, else live trades a
different universe than was backtested — the exact drift that put large caps
like TITAN/MARUTI/ITC into a "midcap" book; see exports/models/
midcap_narrow_60d_breakout/SUMMARY.md "Top-100 ADV MINUS Large"):
  1. Compute 20-day ADV for every N500 stock.
  2. Sort by ADV descending.
  3. Take the top `--top` (default 100) by ADV.
  4. EXCLUDE Nifty-100 members (large caps) via the same nifty100.csv the
     backtest uses → leaves ~40 genuine midcaps.

Output: selector JSON compatible with backtest.py --universe-file.

Usage:
  python tools/models/midcap_narrow_60d_breakout/build_universe.py \
    --out logs/momrot/universes/midcap_narrow.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.universes import nifty500_symbols  # noqa: E402

log = logging.getLogger("build_midcap_narrow")

# Same large-cap exclusion list the backtest uses (backtest.py N100_CSV).
NIFTY100_CSV = ROOT / "src" / "data" / "symbols" / "nifty100.csv"


def load_nifty100(csv_path: Path) -> set:
    """Plain symbols of Nifty-100 EQ members (large caps to exclude)."""
    out = set()
    try:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out.add(r["Symbol"].strip())
    except FileNotFoundError:
        log.warning(f"nifty100 csv not found at {csv_path} — NOT excluding large caps!")
    return out


def _plain(sym: str) -> str:
    return sym.replace("NSE:", "").replace("-EQ", "")


def compute_adv(symbol: str, end_dt: datetime, days: int = 60) -> float:
    """Avg daily ₹ value traded over last `days` calendar days, in lakh."""
    start_dt = end_dt - timedelta(days=days)
    df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    if df.empty or len(df) < 5:
        return 0.0
    df["value"] = df["close"].astype(float) * df["volume"].astype(float)
    return float(df["value"].tail(20).mean()) / 1e5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-top", type=int, default=0,
                    help="Skip top-N by ADV before taking --top (0 = matches "
                         "backtest; large caps are removed via Nifty-100 exclusion)")
    ap.add_argument("--top", type=int, default=100,
                    help="Take top-N by ADV, THEN exclude Nifty-100 members")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-adv-lakh", type=float, default=20.0)
    ap.add_argument("--no-exclude-nifty100", dest="exclude_n100",
                    action="store_false", default=True,
                    help="Disable large-cap (Nifty-100) exclusion (NOT recommended "
                         "— backtest excludes them)")
    ap.add_argument("--nifty100-csv", default=str(NIFTY100_CSV))
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

    # Take top-N by ADV, then exclude Nifty-100 large caps — mirrors backtest.py.
    top_pool = rows[args.skip_top:args.skip_top + args.top]
    if args.exclude_n100:
        n100 = load_nifty100(Path(args.nifty100_csv))
        before = len(top_pool)
        midcap_band = [r for r in top_pool if _plain(r["symbol"]) not in n100]
        log.info(f"Excluded {before - len(midcap_band)} Nifty-100 large caps "
                 f"({before} -> {len(midcap_band)})")
    else:
        midcap_band = top_pool

    out = {
        "generated_at": datetime.now().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "method": (f"ADV-ranked top-{args.top} from N500"
                   + (" MINUS Nifty-100 large caps" if args.exclude_n100 else "")
                   + (f", skip-top-{args.skip_top}" if args.skip_top else "")),
        "skip_large": args.skip_top,
        "top_n": args.top,
        "excluded_nifty100": args.exclude_n100,
        "size": len(midcap_band),
        "stocks": midcap_band,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out}")
    log.info(
        f"Top 5: "
        f"{[r['symbol'].split(':')[-1].replace('-EQ', '') for r in midcap_band[:5]]}"
    )


if __name__ == "__main__":
    main()
