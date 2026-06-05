"""5-minute bar data layer for orb_momentum_intraday.

Fyers history (resolution="5") is the only source (per project policy: Fyers
only, never yfinance). Bars are cached as pickles in CACHE_DIR so a backtest
re-run is instant. The DB historical_data table is DAILY only, so intraday must
be pulled from Fyers and cached here (not stored in Postgres).
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine

# Cache dir for 5-min pickles (gitignored — can be MB-scale). Override with env.
CACHE_DIR = Path(os.environ.get(
    "ORB5MIN_CACHE",
    str(Path(__file__).resolve().parent / "cache5min")))


def _fyers():
    """Build a Fyers SDK client from the active broker token in the DB."""
    from fyers_apiv3 import fyersModel
    eng = _get_engine()
    with eng.connect() as c:
        r = list(c.execute(text(
            "SELECT client_id, access_token FROM broker_configurations "
            "WHERE user_id=1 AND broker_name='fyers' AND is_active=true LIMIT 1")))[0]
    return fyersModel.FyersModel(client_id=r[0], token=r[1], is_async=False, log_path="/tmp")


def fetch_5min(symbol: str, start: dt.date, end: dt.date,
               fy=None, chunk_days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch 5-min OHLCV for one symbol from Fyers (chunked; ~100-day intraday cap).

    `symbol` is plain (e.g. "RELIANCE"); we wrap to NSE:<sym>-EQ. Returns a
    DataFrame [ts,o,h,l,c,v,dt,day] in IST, or None if no data.
    """
    fy = fy or _fyers()
    fsym = f"NSE:{symbol}-EQ"
    out: List[list] = []
    cur = start
    # NOTE: `<=` not `<`. The LIVE intraday path calls fetch_5min(today, today)
    # (start == end); `cur < end` skipped the loop entirely → None → ORB never
    # got today's bars → never bought. `<=` fetches the single (today) chunk;
    # multi-day backtests are unaffected (the last chunk still ends at `end`).
    while cur <= end:
        ce = min(cur + dt.timedelta(days=chunk_days), end)
        try:
            resp = fy.history({"symbol": fsym, "resolution": "5", "date_format": "1",
                               "range_from": cur.isoformat(), "range_to": ce.isoformat(),
                               "cont_flag": "1"})
            if resp.get("s") == "ok":
                out += resp.get("candles", [])
        except Exception:
            pass
        cur = ce + dt.timedelta(days=1)
        time.sleep(0.2)
    if not out:
        return None
    df = pd.DataFrame(out, columns=["ts", "o", "h", "l", "c", "v"]).drop_duplicates("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df["day"] = df["dt"].dt.date
    return df.sort_values("ts").reset_index(drop=True)


def get_5min(symbol: str, start: dt.date, end: dt.date, fy=None) -> Optional[pd.DataFrame]:
    """Cached 5-min: read the pickle if present, else fetch + cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{symbol}.pkl"
    if p.exists():
        try:
            return pd.read_pickle(p)
        except Exception:
            pass
    df = fetch_5min(symbol, start, end, fy=fy)
    if df is not None and len(df) >= 300:
        df.to_pickle(p)
        return df
    return None


def load_all_cached() -> dict:
    """Load every cached 5-min pickle -> {symbol: DataFrame}."""
    out = {}
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.pkl"):
            try:
                out[p.stem] = pd.read_pickle(p)
            except Exception:
                pass
    return out
