"""Multi-param stock selector. Picks live trading candidates using
volatility × momentum × volume × liquidity, not just historical backtest sum%.

Pure Python. Reads cached daily bars from Postgres. No LLM. No news.

Score formula (all weights configurable):
  composite = (
      w_atr  * normalized(ATR% over 20d) +     # volatility = bigger moves
      w_mom  * normalized(return over 60d) +   # momentum
      w_vol  * normalized(volume_spike) +      # 20d-avg breakout
      w_liq  * normalized(ADV / lakh) +        # liquidity
      w_dist * normalized(distance from 52W high or low, abs)
  )

Output: ranked CSV + top-N selection JSON.

Usage:
  python tools/backtests/stock_selector.py --universe nifty500 --top 30 \
    --out exports/backtests/SELECTOR_TOP30.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS, nifty500_symbols,
)

log = logging.getLogger("stock_selector")


def load_universe(name: str):
    if name == "nifty50": return NIFTY50_SYMBOLS
    if name == "nifty500": return nifty500_symbols()
    raise ValueError(name)


def compute_features(symbol: str, end_dt: datetime) -> Optional[Dict]:
    """Load daily bars for symbol, compute features."""
    start_dt = end_dt - timedelta(days=400)  # need >252 for 52W
    df = read_cached(symbol, "D", int(start_dt.timestamp()), int(end_dt.timestamp()))
    if df.empty or len(df) < 60:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # ATR% over 20d
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_20 = tr.rolling(20).mean().iloc[-1]
    atr_pct = (atr_20 / df["close"].iloc[-1]) * 100 if df["close"].iloc[-1] > 0 else 0

    # 60d return
    if len(df) >= 60:
        ret_60d = ((df["close"].iloc[-1] / df["close"].iloc[-60]) - 1) * 100
    else:
        ret_60d = 0

    # 20d-avg volume + recent spike
    vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    vol_5 = df["volume"].rolling(5).mean().iloc[-1]
    vol_spike = (vol_5 / vol_20) if vol_20 > 0 else 1.0

    # Avg daily value (lakh)
    adv = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1] / 1e5

    # Distance from 52-week high
    h_52w = df["high"].iloc[-252:].max() if len(df) >= 252 else df["high"].max()
    l_52w = df["low"].iloc[-252:].min() if len(df) >= 252 else df["low"].min()
    dist_high_pct = ((df["close"].iloc[-1] - h_52w) / h_52w) * 100
    dist_low_pct = ((df["close"].iloc[-1] - l_52w) / l_52w) * 100

    return {
        "symbol": symbol,
        "close": float(df["close"].iloc[-1]),
        "atr_pct": float(atr_pct),
        "ret_60d_pct": float(ret_60d),
        "vol_spike": float(vol_spike),
        "adv_lakh": float(adv),
        "dist_52w_high_pct": float(dist_high_pct),
        "dist_52w_low_pct": float(dist_low_pct),
        "bars": len(df),
    }


def normalize_col(df: pd.DataFrame, col: str, abs_val: bool = False) -> pd.Series:
    """Min-max normalize a column. Optionally use abs values first."""
    s = df[col].abs() if abs_val else df[col]
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.5] * len(df), index=df.index)
    return (s - mn) / (mx - mn)


def score_stocks(
    features: List[Dict],
    w_atr: float = 0.25,
    w_mom: float = 0.25,
    w_vol: float = 0.15,
    w_liq: float = 0.20,
    w_dist: float = 0.15,
    min_price: float = 50.0,
    min_adv_lakh: float = 100.0,
) -> pd.DataFrame:
    df = pd.DataFrame(features)
    df = df[df["close"] >= min_price].copy()
    df = df[df["adv_lakh"] >= min_adv_lakh].copy()
    if df.empty:
        return df

    df["score_atr"] = normalize_col(df, "atr_pct")
    df["score_mom"] = normalize_col(df, "ret_60d_pct")
    df["score_vol"] = normalize_col(df, "vol_spike")
    df["score_liq"] = normalize_col(df, "adv_lakh")
    df["score_dist"] = normalize_col(df, "dist_52w_high_pct", abs_val=True)

    df["composite"] = (
        w_atr * df["score_atr"] +
        w_mom * df["score_mom"] +
        w_vol * df["score_vol"] +
        w_liq * df["score_liq"] +
        w_dist * df["score_dist"]
    )
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty500", choices=["nifty50", "nifty500"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="exports/backtests/SELECTOR_TOP.json")
    ap.add_argument("--min-price", type=float, default=50.0)
    ap.add_argument("--min-adv-lakh", type=float, default=100.0,
                    help="Min 20d avg daily value in lakh (₹100L = ₹1cr)")
    ap.add_argument("--w-atr", type=float, default=0.25)
    ap.add_argument("--w-mom", type=float, default=0.25)
    ap.add_argument("--w-vol", type=float, default=0.15)
    ap.add_argument("--w-liq", type=float, default=0.20)
    ap.add_argument("--w-dist", type=float, default=0.15)
    ap.add_argument("--end-date", default=None,
                    help="Reference date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    universe = load_universe(args.universe)
    log.info(f"Computing features for {len(universe)} symbols (universe={args.universe}) as of {end_dt.date()}")

    feats = []
    for i, (sym, name) in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")
        f = compute_features(sym, end_dt)
        if f:
            f["name"] = name
            feats.append(f)
    log.info(f"Loaded {len(feats)}/{len(universe)} symbols with sufficient data")

    df = score_stocks(
        feats,
        w_atr=args.w_atr, w_mom=args.w_mom, w_vol=args.w_vol,
        w_liq=args.w_liq, w_dist=args.w_dist,
        min_price=args.min_price, min_adv_lakh=args.min_adv_lakh,
    )
    log.info(f"After price+ADV filter: {len(df)} candidates")

    top_n = df.head(args.top)
    out = {
        "generated_at": datetime.now().isoformat(),
        "universe": args.universe,
        "end_date": end_dt.date().isoformat(),
        "weights": {
            "atr": args.w_atr, "mom": args.w_mom, "vol": args.w_vol,
            "liq": args.w_liq, "dist": args.w_dist,
        },
        "filters": {"min_price": args.min_price, "min_adv_lakh": args.min_adv_lakh},
        "top_n": args.top,
        "stocks": top_n.to_dict(orient="records"),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Wrote {args.out}")

    print("\n=== Top stocks ===")
    print(top_n[["rank", "symbol", "close", "atr_pct", "ret_60d_pct",
                 "vol_spike", "adv_lakh", "composite"]].to_string(index=False))


if __name__ == "__main__":
    main()
