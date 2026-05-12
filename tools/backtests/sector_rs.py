"""Sector Relative Strength (RS) — Indian market sector rotation filter.

Computes daily RS of NSE sectoral indices vs Nifty 50. Top-2 sectors
by rolling 60-day RS are flagged as "leadership"; bottom-2 as
"laggards". Stocks in laggard sectors filtered out.

Sectors covered: IT, BANK, PHARMA, AUTO, FMCG, METAL, REALTY, ENERGY,
INFRA, PSE (PSU Enterprise).

No LLM. No news. Free Fyers API only.

Usage:
  python tools/backtests/sector_rs.py --start 2025-05-12 --end 2026-05-12 \
    --out exports/backtests/SECTOR_RS_2025-2026.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("sector_rs")

SECTORS = {
    "IT":      "NSE:NIFTYIT-INDEX",
    "BANK":    "NSE:NIFTYBANK-INDEX",
    "PHARMA":  "NSE:NIFTYPHARMA-INDEX",
    "AUTO":    "NSE:NIFTYAUTO-INDEX",
    "FMCG":    "NSE:NIFTYFMCG-INDEX",
    "METAL":   "NSE:NIFTYMETAL-INDEX",
    "REALTY":  "NSE:NIFTYREALTY-INDEX",
    "ENERGY":  "NSE:NIFTYENERGY-INDEX",
    "INFRA":   "NSE:NIFTYINFRA-INDEX",
    "PSE":     "NSE:NIFTYPSE-INDEX",
}
NIFTY50 = "NSE:NIFTY50-INDEX"


def fetch_index_history(user_id: int, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Chunked fetch (Fyers limits ~360 days/call)."""
    from src.services.brokers.fyers_service import FyersService
    svc = FyersService()
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    all_rows = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + pd.Timedelta(days=360), end_dt)
        r = svc.history(user_id=user_id, symbol=symbol, exchange="NSE",
                        interval="D",
                        start_date=cur.strftime("%Y-%m-%d"),
                        end_date=chunk_end.strftime("%Y-%m-%d"))
        if r and r.get("status") == "success":
            all_rows.extend(r["data"]["candles"])
        cur = chunk_end + pd.Timedelta(days=1)
    if not all_rows:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(all_rows)
    df["timestamp"] = df["timestamp"].astype(int)
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


def compute_rs(user_id: int, start: str, end: str,
                lookback_days: int = 60) -> pd.DataFrame:
    """Returns DataFrame: index=date, columns=[sector_RS values].
    RS = sector_60d_return / nifty50_60d_return.
    """
    log.info(f"Fetching Nifty 50 + 10 sectoral indices {start} → {end}")
    # Fetch with extra lookback for RS warmup
    fetch_start = (datetime.strptime(start, "%Y-%m-%d") -
                   pd.Timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
    nifty = fetch_index_history(user_id, NIFTY50, fetch_start, end).set_index("date")
    nifty = nifty.rename(columns={"close": "NIFTY50"})

    sectors = {}
    for name, sym in SECTORS.items():
        try:
            df = fetch_index_history(user_id, sym, fetch_start, end).set_index("date")
            sectors[name] = df["close"]
        except Exception as e:
            log.warning(f"Skip {name}: {e}")

    all_data = pd.concat([nifty["NIFTY50"]] + [s.rename(n) for n, s in sectors.items()],
                          axis=1).ffill()
    # 60d rolling return
    returns = all_data.pct_change(lookback_days)
    # RS = sector_return - nifty50_return (additive, simpler than ratio)
    rs = pd.DataFrame(index=returns.index)
    for sec in sectors.keys():
        rs[sec] = returns[sec] - returns["NIFTY50"]
    return rs.dropna()


def rank_leadership(rs: pd.DataFrame, top_n: int = 2,
                     bottom_n: int = 2) -> pd.DataFrame:
    """For each date, rank sectors. Return dict per date with top + bottom."""
    out = []
    for date, row in rs.iterrows():
        ranked = row.sort_values(ascending=False)
        out.append({
            "date": date,
            "top": list(ranked.head(top_n).index),
            "top_rs": ranked.head(top_n).round(4).to_dict(),
            "bottom": list(ranked.tail(bottom_n).index),
            "bottom_rs": ranked.tail(bottom_n).round(4).to_dict(),
        })
    return pd.DataFrame(out)


# Symbol-to-sector mapping (manually curated for selector top-10)
# Extend as needed
SYMBOL_SECTOR = {
    "SWIGGY": "INFRA",      # logistics/internet
    "VMM": "INFRA",         # placeholder
    "AEGISLOG": "ENERGY",   # logistics-energy
    "ANGELONE": "BANK",     # broker (BFSI)
    "SAILIFE": "PHARMA",
    "ITI": "INFRA",         # telecom equipment
    "IKS": "PHARMA",        # IKS Health
    "AMBER": "AUTO",        # consumer durables (close enough)
    "NTPCGREEN": "ENERGY",
    "BSE": "BANK",          # exchange (BFSI)
    # N50 large caps
    "RELIANCE": "ENERGY",
    "HCLTECH": "IT",
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "TECHM": "IT",
    "SBIN": "BANK",
    "ICICIBANK": "BANK",
    "HDFCBANK": "BANK",
    "AXISBANK": "BANK",
    "KOTAKBANK": "BANK",
    "INDUSINDBK": "BANK",
    "HINDALCO": "METAL",
    "TATASTEEL": "METAL",
    "JSWSTEEL": "METAL",
    "COALINDIA": "ENERGY",
    "BPCL": "ENERGY",
    "ONGC": "ENERGY",
    "POWERGRID": "ENERGY",
    "NTPC": "ENERGY",
    "BHARTIARTL": "INFRA",
    "MARUTI": "AUTO",
    "TATAMOTORS": "AUTO",
    "TMPV": "AUTO",
    "HEROMOTOCO": "AUTO",
    "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO",
    "M&M": "AUTO",
    "ITC": "FMCG",
    "HINDUNILVR": "FMCG",
    "BRITANNIA": "FMCG",
    "NESTLEIND": "FMCG",
    "TATACONSUM": "FMCG",
    "ASIANPAINT": "FMCG",
    "CIPLA": "PHARMA",
    "DRREDDY": "PHARMA",
    "SUNPHARMA": "PHARMA",
    "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "PHARMA",
    "ULTRACEMCO": "INFRA",
    "GRASIM": "INFRA",
    "LT": "INFRA",
    "ADANIPORTS": "INFRA",
    "ADANIENT": "INFRA",
    "JIOFIN": "BANK",       # NBFC/BFSI
    "BAJFINANCE": "BANK",
    "BAJAJFINSV": "BANK",
    "SBILIFE": "BANK",
    "HDFCLIFE": "BANK",
    "TITAN": "FMCG",
    "TRENT": "FMCG",
    "BEL": "PSE",
    "ZOMATO": "INFRA",
    "TATAINVEST": "BANK",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-12")
    ap.add_argument("--end", default="2026-05-12")
    ap.add_argument("--lookback", type=int, default=60,
                    help="RS lookback in trading days")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--bottom-n", type=int, default=2)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--out", default="exports/backtests/SECTOR_RS.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    rs = compute_rs(args.user_id, args.start, args.end, args.lookback)
    log.info(f"RS computed for {len(rs)} days, sectors: {list(rs.columns)}")

    # Window-clip
    win_start = datetime.strptime(args.start, "%Y-%m-%d").date()
    win_end = datetime.strptime(args.end, "%Y-%m-%d").date()
    rs_win = rs.loc[win_start:win_end]

    leadership = rank_leadership(rs_win, args.top_n, args.bottom_n)
    log.info(f"Leadership rankings: {len(leadership)} days")

    # Summary: % time each sector spent in top-N / bottom-N
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    for _, row in leadership.iterrows():
        for s in row["top"]: top_counts[s] = top_counts.get(s, 0) + 1
        for s in row["bottom"]: bot_counts[s] = bot_counts.get(s, 0) + 1

    print("\n=== Sector RS leadership over window ===")
    print(f"Days: {len(leadership)}")
    print(f"\n% time in TOP-{args.top_n}:")
    for s in sorted(top_counts, key=lambda x: -top_counts[x]):
        print(f"  {s:8s} {top_counts[s]:3d} ({top_counts[s] * 100 // len(leadership)}%)")
    print(f"\n% time in BOTTOM-{args.bottom_n}:")
    for s in sorted(bot_counts, key=lambda x: -bot_counts[x]):
        print(f"  {s:8s} {bot_counts[s]:3d} ({bot_counts[s] * 100 // len(leadership)}%)")

    out_data = {
        "generated_at": datetime.now().isoformat(),
        "start": args.start,
        "end": args.end,
        "lookback_days": args.lookback,
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
        "summary_top_pct": {s: top_counts[s] * 100 // len(leadership)
                             for s in top_counts},
        "summary_bottom_pct": {s: bot_counts[s] * 100 // len(leadership)
                                for s in bot_counts},
        "daily": leadership.to_dict(orient="records"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_data, indent=2, default=str))
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
