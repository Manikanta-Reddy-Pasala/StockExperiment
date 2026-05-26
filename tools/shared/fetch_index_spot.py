"""Bulk-fetch index spot daily OHLC into historical_data (Fyers API).

Required by FinNifty IC backtest pipeline. FinNifty/Nifty50/BankNifty
all share same fetch logic — pass --symbol.

Idempotent (ON CONFLICT DO NOTHING on (symbol, date)).

Usage:
    python tools/shared/fetch_index_spot.py \
        --symbol NSE:FINNIFTY-INDEX --from 2023-01-01 --to 2026-05-15
    python tools/shared/fetch_index_spot.py \
        --symbol NSE:NIFTY50-INDEX --from 2023-01-01 --to 2026-05-15
    python tools/shared/fetch_index_spot.py \
        --symbol NSE:NIFTYBANK-INDEX --from 2023-01-01 --to 2026-05-15
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily index spot OHLC from Fyers over the [start, end] window.

    Slices the request window into 360-day chunks (under Fyers's per-request
    cap), concatenates all candles, then cleans them into a daily DataFrame
    keyed by trading date.

    Args:
        symbol: Fyers index symbol, e.g. ``NSE:FINNIFTY-INDEX``.
        start: Inclusive start date, ``YYYY-MM-DD``.
        end: Inclusive end date, ``YYYY-MM-DD``.

    Returns:
        DataFrame with [timestamp, open, high, low, close, volume, date],
        de-duplicated and sorted ascending by timestamp. Empty DataFrame if no
        candles were returned.
    """
    from src.services.brokers.fyers_service import FyersService
    svc = FyersService()
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    ed = datetime.strptime(end, "%Y-%m-%d").date()
    chunks = []
    cursor = sd
    # Walk the window in 360-day slices to stay under Fyers's daily request cap.
    while cursor <= ed:
        chunk_end = min(cursor + timedelta(days=360), ed)
        res = svc.history(user_id=1, symbol=symbol, exchange="NSE",
                          interval="D",
                          start_date=cursor.strftime("%Y-%m-%d"),
                          end_date=chunk_end.strftime("%Y-%m-%d"))
        if res and res.get("status") == "success":
            cs = (res.get("data") or {}).get("candles") or []
            chunks.extend(cs)
            print(f"  {cursor}..{chunk_end}: {len(cs)} bars")
        else:
            print(f"  FAIL {cursor}..{chunk_end}: {(res or {}).get('message')}")
        # +1 day so the next chunk starts after this chunk's inclusive end.
        cursor = chunk_end + timedelta(days=1)
    if not chunks:
        return pd.DataFrame()
    # Fyers may return candles as dicts or positional [ts,o,h,l,c,v] arrays.
    if isinstance(chunks[0], dict):
        df = pd.DataFrame(chunks)
    else:
        df = pd.DataFrame(chunks, columns=["timestamp", "open", "high",
                                            "low", "close", "volume"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"])
    # Derive the IST calendar date (the natural-key column for historical_data).
    df["date"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                  .dt.tz_convert("Asia/Kolkata").dt.date)
    return df.sort_values("timestamp").reset_index(drop=True)


def insert(symbol: str, df: pd.DataFrame) -> int:
    """Upsert daily index bars into the ``historical_data`` table.

    Idempotent: ``ON CONFLICT (symbol, date) DO NOTHING`` means re-running only
    adds dates not already present, so the rowcount reflects genuinely new rows.

    Args:
        symbol: Fyers index symbol used as the row's ``symbol`` key.
        df: DataFrame from :func:`fetch` (must include a ``date`` column).

    Returns:
        Count of newly inserted rows (existing rows are skipped, not counted).
    """
    if df.empty:
        return 0
    eng = _get_engine()
    sql = text(
        "INSERT INTO historical_data "
        "(symbol, date, timestamp, open, high, low, close, volume, data_source) "
        "VALUES (:sym, :date, :ts, :o, :h, :l, :c, :v, 'fyers') "
        "ON CONFLICT (symbol, date) DO NOTHING"
    )
    n = 0
    with eng.begin() as conn:
        for r in df.itertuples():
            # rowcount is 1 for a real insert, 0 when the (symbol,date) already exists.
            n += conn.execute(sql, {
                "sym": symbol, "date": r.date, "ts": int(r.timestamp),
                "o": float(r.open), "h": float(r.high), "l": float(r.low),
                "c": float(r.close), "v": int(r.volume),
            }).rowcount or 0
    return n


def main():
    """CLI entry point: fetch one index's daily spot and upsert into the DB.

    Parses ``--symbol``, ``--from`` and ``--to``, runs :func:`fetch` then
    :func:`insert`, and prints how many bars were fetched vs newly inserted.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True,
                    help="Fyers symbol, e.g. NSE:FINNIFTY-INDEX")
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    args = ap.parse_args()
    df = fetch(args.symbol, args.frm, args.to)
    print(f"Fetched {len(df)} bars total")
    n = insert(args.symbol, df)
    print(f"Inserted {n} new rows ({len(df) - n} already existed)")


if __name__ == "__main__":
    main()
