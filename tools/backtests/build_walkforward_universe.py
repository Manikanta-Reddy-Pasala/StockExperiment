"""Walk-forward universe builder — no lookahead bias.

For year Y, builds universe = N50 minus N worst stocks from year Y-1 backtest.
For year 1 (no prior data), uses ADV-ranked top-N.

Reads per-stock _summary*.md from a prior-year backtest dir, identifies
worst-N by total Sum%, generates universe JSON.

Usage:
  python tools/backtests/build_walkforward_universe.py \
    --prior-year-dir exports/backtests/multiyear_n50/nifty50_ema_200_400_2024_2025 \
    --drop-bottom 15 \
    --out exports/backtests/universes/n50_walkforward_2025-05-12.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple


log = logging.getLogger("walkforward")


N50_BASE = [
    ("NSE:ADANIENT-EQ", "Adani Enterprises"), ("NSE:ADANIPORTS-EQ", "Adani Ports"),
    ("NSE:APOLLOHOSP-EQ", "Apollo Hospitals"), ("NSE:ASIANPAINT-EQ", "Asian Paints"),
    ("NSE:AXISBANK-EQ", "Axis Bank"), ("NSE:BAJAJ-AUTO-EQ", "Bajaj Auto"),
    ("NSE:BAJAJFINSV-EQ", "Bajaj Finserv"), ("NSE:BAJFINANCE-EQ", "Bajaj Finance"),
    ("NSE:BEL-EQ", "BEL"), ("NSE:BHARTIARTL-EQ", "Bharti Airtel"),
    ("NSE:BPCL-EQ", "BPCL"), ("NSE:BRITANNIA-EQ", "Britannia"),
    ("NSE:CIPLA-EQ", "Cipla"), ("NSE:COALINDIA-EQ", "Coal India"),
    ("NSE:DIVISLAB-EQ", "Divis Labs"), ("NSE:DRREDDY-EQ", "Dr Reddys"),
    ("NSE:EICHERMOT-EQ", "Eicher Motors"), ("NSE:ETERNAL-EQ", "Eternal"),
    ("NSE:GRASIM-EQ", "Grasim"), ("NSE:HCLTECH-EQ", "HCL Tech"),
    ("NSE:HDFCBANK-EQ", "HDFC Bank"), ("NSE:HDFCLIFE-EQ", "HDFC Life"),
    ("NSE:HEROMOTOCO-EQ", "Hero MotoCorp"), ("NSE:HINDALCO-EQ", "Hindalco"),
    ("NSE:HINDUNILVR-EQ", "Hindustan Unilever"), ("NSE:ICICIBANK-EQ", "ICICI Bank"),
    ("NSE:INDUSINDBK-EQ", "IndusInd Bank"), ("NSE:INFY-EQ", "Infosys"),
    ("NSE:ITC-EQ", "ITC"), ("NSE:JIOFIN-EQ", "Jio Financial"),
    ("NSE:JSWSTEEL-EQ", "JSW Steel"), ("NSE:KOTAKBANK-EQ", "Kotak Bank"),
    ("NSE:LT-EQ", "Larsen"), ("NSE:M&M-EQ", "M&M"),
    ("NSE:MARUTI-EQ", "Maruti"), ("NSE:NESTLEIND-EQ", "Nestle"),
    ("NSE:NTPC-EQ", "NTPC"), ("NSE:ONGC-EQ", "ONGC"),
    ("NSE:POWERGRID-EQ", "Power Grid"), ("NSE:RELIANCE-EQ", "Reliance"),
    ("NSE:SBILIFE-EQ", "SBI Life"), ("NSE:SBIN-EQ", "SBI"),
    ("NSE:SHRIRAMFIN-EQ", "Shriram Finance"), ("NSE:SUNPHARMA-EQ", "Sun Pharma"),
    ("NSE:TATACONSUM-EQ", "Tata Consumer"), ("NSE:TATAMOTORS-EQ", "Tata Motors"),
    ("NSE:TATASTEEL-EQ", "Tata Steel"), ("NSE:TCS-EQ", "TCS"),
    ("NSE:TECHM-EQ", "Tech Mahindra"), ("NSE:TITAN-EQ", "Titan"),
    ("NSE:TRENT-EQ", "Trent"), ("NSE:ULTRACEMCO-EQ", "UltraTech"),
    ("NSE:WIPRO-EQ", "Wipro"),
]


def parse_prior_year_losers(prior_dir: Path, drop_n: int = 15) -> List[str]:
    """Parse per-stock .md files. Returns symbols with worst total sum%."""
    if not prior_dir.exists():
        log.warning(f"Prior dir not found: {prior_dir}")
        return []
    sum_pct_re = re.compile(r"\| BUY \(all\) \|\s*\d+\s*\|\s*\d+\s*\|\s*[\d.]+%\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([+-]?[\d.]+)%\s*\|\s*([+-]?[\d.]+)%")
    # Actually parse per-stock sum from the markdown table format
    # The simpler approach: look at "Sum % (uncompounded)" in summary
    pattern = re.compile(r"Sum % \(uncompounded\):\s*([+-]?[\d.]+)%")

    sums = []
    for f in prior_dir.glob("*.md"):
        if f.name.startswith("_"): continue
        content = f.read_text()
        m = pattern.search(content)
        sym = f.stem.upper()
        # Convert lowercase to NSE format if needed
        if sym in {s.split(":")[1].replace("-EQ", "") for s, _ in N50_BASE}:
            full_sym = f"NSE:{sym}-EQ"
        elif sym == "M_M":  # M&M filename quirk
            full_sym = "NSE:M&M-EQ"
        else:
            full_sym = f"NSE:{sym}-EQ"
        if m:
            sums.append((full_sym, float(m.group(1))))
        else:
            sums.append((full_sym, 0.0))

    # Sort ascending (worst first), drop the worst N
    sums.sort(key=lambda x: x[1])
    losers = [s for s, _ in sums[:drop_n]]
    log.info(f"Worst {drop_n} from {prior_dir.name}: {losers}")
    return losers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-year-dir", default=None,
                    help="Backtest dir for prior year. Omit for first-year fallback.")
    ap.add_argument("--drop-bottom", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    drop_list = []
    if args.prior_year_dir:
        drop_list = parse_prior_year_losers(Path(args.prior_year_dir),
                                              args.drop_bottom)

    if not drop_list:
        log.info("No prior-year data — using empty drop list (full N50)")

    keep = [(s, n) for s, n in N50_BASE if s not in drop_list]
    log.info(f"Kept {len(keep)} stocks (dropped {len(drop_list)})")

    out_data = {
        "generated_at": "2026-05-13",
        "method": "walk-forward N50 minus prior-year worst-N",
        "prior_year_dir": str(args.prior_year_dir) if args.prior_year_dir else None,
        "drop_bottom_n": args.drop_bottom,
        "drops": drop_list,
        "stocks": [{"symbol": s, "name": n} for s, n in keep],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
