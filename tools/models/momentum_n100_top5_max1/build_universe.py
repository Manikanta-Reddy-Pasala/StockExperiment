"""Build NIFTY 100 universe file from REAL NSE constituents.

Source: ``src/data/symbols/nifty100.csv`` (refresh via ``tools/refresh_nifty100.py``).

Previously this script ADV-ranked top-100 from Nifty 500 — that produced
"pseudo-N100" with 47/100 stocks NOT in the real index (HFCL, GROWW, BSE,
COHANCE etc.). Replaced with curated NSE list to match index methodology.

Output JSON is compatible with momentum_n100_top5_max1/live_signal.py.

Usage:
  python tools/models/momentum_n100_top5_max1/build_universe.py \
    --out /app/logs/momrot/universes/n100_current.json

  # Optional: include ADV for ranking visibility (not used for selection)
  python tools/models/momentum_n100_top5_max1/build_universe.py \
    --out /app/logs/momrot/universes/n100_current.json --include-adv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.universes import nifty100_symbols  # noqa: E402

log = logging.getLogger("build_universe")


def compute_adv(symbol: str, end_dt: datetime, days: int = 60) -> float:
    """Return avg daily ₹ value traded over last `days` calendar days, in lakh.

    Args:
        symbol: NSE symbol in cache format (e.g. "NSE:RELIANCE-EQ").
        end_dt: as-of date; ADV is measured looking back from here.
        days: calendar-day lookback window for the OHLCV fetch (default 60;
            ~40 trading sessions, enough to land 20 valid bars after holidays).

    Returns:
        Average daily turnover in lakh rupees, or 0.0 if data is missing.

    Gotcha: this value is INFORMATIONAL ONLY (shown when --include-adv is set).
    Selection no longer narrows by ADV — the universe is the official NSE
    Nifty 100 list. See the module docstring on why the old ADV-ranking was
    dropped. Any fetch error or thin history is swallowed and returns 0.0 so a
    single bad symbol can't abort the whole build.
    """
    start_dt = end_dt - timedelta(days=days)
    try:
        df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    except Exception:
        return 0.0
    if df.empty or len(df) < 5:
        return 0.0
    # Per-bar turnover = close * volume; ADV = mean of the last 20 sessions...
    df["value"] = df["close"].astype(float) * df["volume"].astype(float)
    # ...divided by 1e5 to convert rupees -> lakh for display.
    return float(df["value"].tail(20).mean()) / 1e5


def main():
    """CLI entrypoint: write the Nifty 100 universe JSON consumed by the model.

    Reads the real NSE constituent list (tools.shared.universes.nifty100_symbols,
    backed by src/data/symbols/nifty100.csv) and serialises it to --out as a
    {"stocks": [{symbol, name[, adv_lakh]}, ...]} payload that live_signal.py /
    backtest.py rank. The file format must stay in sync with
    live_signal.load_universe (which reads payload["stocks"]).

    Returns:
        int exit code — 0 on success, 1 if the constituent list is empty (a
        signal that tools/refresh_nifty100.py needs to run first).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD (for ADV calc only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-adv", action="store_true",
                    help="Compute & include ADV per stock (slower, informational)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # As-of date only matters for the optional ADV calc; defaults to now.
    end_dt = (datetime.strptime(args.end_date, "%Y-%m-%d")
              if args.end_date else datetime.now())

    n100 = nifty100_symbols()
    if not n100:
        # Empty list => the source CSV is missing/stale. Fail loudly so the
        # caller refreshes it instead of writing an empty universe.
        log.error("Real Nifty 100 list empty. Run tools/analysis/download_niftyindices.py first.")
        return 1

    log.info(f"Loaded real NIFTY 100 ({len(n100)} stocks) from NSE CSV")

    stocks: List[dict] = []
    for i, (sym, name) in enumerate(n100):
        entry = {"symbol": sym, "name": name}
        if args.include_adv:
            # Progress log every 20 symbols since per-symbol ADV is a slow DB read.
            if i % 20 == 0:
                log.info(f"  ADV {i}/{len(n100)}")
            entry["adv_lakh"] = compute_adv(sym, end_dt)
        stocks.append(entry)

    out = {
        "generated_at": datetime.now().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "method": "Real NIFTY 100 constituents (NSE archives)",
        "source_csv": "src/data/symbols/nifty100.csv",
        "stocks": stocks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out}")
    log.info(f"First 10: {[s['symbol'] for s in stocks[:10]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
