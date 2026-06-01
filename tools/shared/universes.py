"""Shared universe data + Fyers fetcher helpers.

Single source of truth for stock universes (NIFTY 50, NIFTY 500) and
the Fyers history fetch used by momentum rotation backtest + live signal.

Previously embedded inside run_ema_200_400_backtest.py — extracted here
when the EMA backtest was removed (rejected model).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.data.nifty500_universe import (  # noqa: E402
    load_nifty500_with_meta,
)


# NIFTY 50 constituents (plain NSE tickers, post 2024-2025 reconstitution).
NIFTY50_BASE = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BPCL',
    'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB',
    'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK',
    'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
    'ITC', 'INDUSINDBK', 'INFY', 'JIOFIN', 'JSWSTEEL',
    'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC',
    'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE',
    'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TCS', 'TATACONSUM',
    'TMPV', 'TATASTEEL', 'TECHM', 'TITAN', 'TRENT',
    'ULTRACEMCO', 'UPL', 'WIPRO',
]
NIFTY50_SYMBOLS = [(s, s) for s in NIFTY50_BASE]


def nifty500_symbols(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Return [(symbol, company_name)] for the Nifty 500.

    Delegates to ``src.services.data.nifty500_universe.load_nifty500_with_meta``
    (the CSV-backed Nifty 500 loader) and strips each entry down to a plain
    NSE ticker + display name.

    Args:
        limit: If given, return only the first ``limit`` rows; otherwise all.

    Returns:
        List of (plain_ticker, company_name) tuples. Symbols are plain NSE
        tickers (no ``NSE:``/``-EQ`` wrapper); the fetcher converts to Fyers
        form on demand.
    """
    rows = load_nifty500_with_meta()  # [(fyers_sym, name, industry), ...]
    out = []
    for fyers_sym, name, _industry in rows:
        # Strip the Fyers wrapper so callers see bare tickers (matches NIFTY50_BASE).
        plain = fyers_sym.replace("NSE:", "").replace("-EQ", "")
        out.append((plain, name))
    return out[:limit] if limit else out


_NIFTY100_CACHE = ROOT / "src" / "data" / "symbols" / "nifty100.csv"


def nifty100_symbols(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Return [(symbol, company_name)] for the real NIFTY 100 (free-float mcap).

    Reads the cached NSE constituent CSV at ``src/data/symbols/nifty100.csv``
    (the official ``ind_nifty100list.csv`` layout, with ``Symbol``, ``Series``
    and ``Company Name`` columns). Refresh the CSV via
    ``tools/refresh_nifty100.py`` (NSE rebalances Mar/Sep).

    Args:
        limit: If given, return only the first ``limit`` rows; otherwise all.

    Returns:
        List of (plain_ticker, company_name) tuples. Empty list if the cache
        CSV is missing.
    """
    import csv
    # No cached constituent file → caller gets an empty universe (not an error).
    if not _NIFTY100_CACHE.exists():
        return []
    out: List[Tuple[str, str]] = []
    with _NIFTY100_CACHE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = (row.get("Symbol") or "").strip()
            series = (row.get("Series") or "EQ").strip()
            # Keep only cash-market equity rows; skip blanks and non-EQ series.
            if not sym or series.upper() != "EQ":
                continue
            # Skip NSE corporate-action PLACEHOLDER scrips (e.g. DUMMYVEDL1-4
            # from the Vedanta demerger). Not tradable, never have price data —
            # they only showed up as "symbols completely missing" in the audit.
            if sym.upper().startswith("DUMMY"):
                continue
            out.append((sym, (row.get("Company Name") or "").strip()))
    return out[:limit] if limit else out


# ---- Fyers symbol normalization ----

def to_fyers_symbol(sym: str) -> str:
    """Normalize an input ticker into the Fyers symbol format.

    Handles the four ticker shapes that flow through the pipeline:

    Plain NSE  ``RELIANCE``     -> ``NSE:RELIANCE-EQ``
    Yahoo      ``RELIANCE.NS``  -> ``NSE:RELIANCE-EQ``
    Index      ``^NSEI``        -> ``NSE:NIFTY50-INDEX``
    Already    ``NSE:RELIANCE-EQ`` (passthrough)

    Args:
        sym: Ticker in any of the supported forms (case-insensitive).

    Returns:
        The Fyers-formatted symbol string. Unknown ``^`` index codes are
        returned unchanged.
    """
    s = sym.upper()
    # Already Fyers-formatted → leave untouched.
    if s.startswith("NSE:"):
        return s
    # Yahoo-style index codes (^...) map to named Fyers index symbols.
    if s.startswith("^"):
        idx_map = {
            "^NSEI": "NSE:NIFTY50-INDEX",
            "^NSEBANK": "NSE:NIFTYBANK-INDEX",
            "^CNXFIN": "NSE:FINNIFTY-INDEX",
            "^CNXIT": "NSE:NIFTYIT-INDEX",
            "^CNXAUTO": "NSE:NIFTYAUTO-INDEX",
        }
        return idx_map.get(s, s)
    # Plain or Yahoo equity ticker: drop any ``.NS`` suffix, wrap as NSE cash equity.
    return f"NSE:{s.replace('.NS', '')}-EQ"


# ---- Fyers fetcher (lazy-cached) ----

_FYERS_CACHE = {"service": None, "init_failed": False, "user_id": 1}


def _fyers_service():
    """Lazy-initialize the Fyers broker service and memoize it in the module cache.

    Builds a ``FyersService`` once and stores it in ``_FYERS_CACHE`` so repeated
    fetches don't re-auth. A failure (import error, missing access token) is
    "sticky": ``init_failed`` is set so subsequent calls short-circuit to None
    instead of retrying the broken init on every symbol.

    Returns:
        The cached ``FyersService`` instance, or None if init failed (no token
        or exception).
    """
    # Sticky failure: once init fails, don't keep retrying for every symbol.
    if _FYERS_CACHE["init_failed"]:
        return None
    # Reuse the already-built service if present.
    if _FYERS_CACHE["service"] is not None:
        return _FYERS_CACHE["service"]
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        # No usable access token → treat as a hard (sticky) init failure.
        cfg = svc.get_broker_config(_FYERS_CACHE["user_id"])
        if not cfg or not cfg.get("access_token"):
            print(f"  fyers: no token for user_id={_FYERS_CACHE['user_id']}")
            _FYERS_CACHE["init_failed"] = True
            return None
        _FYERS_CACHE["service"] = svc
        return svc
    except Exception as e:
        print(f"  fyers init failed: {e}")
        _FYERS_CACHE["init_failed"] = True
        return None


def _history_with_retry(svc, fyers_sym: str, user_id: int, interval: str,
                         start_str: str, end_str: str,
                         max_retries: int = 3) -> dict | None:
    """Call ``svc.history()`` with exponential-backoff retry on failure.

    Without this, a single network blip or transient Fyers 429/503 drops the
    whole chunk silently → data gap. Backoff sleeps: 2s, 4s, 8s.

    Args:
        svc: Initialized ``FyersService``.
        fyers_sym: Fyers-formatted symbol (e.g. ``NSE:RELIANCE-EQ``).
        user_id: Broker account user id whose token authorizes the call.
        interval: Fyers interval code ("1h", "15m", "D", ...).
        start_str: Window start, ``YYYY-MM-DD``.
        end_str: Window end, ``YYYY-MM-DD``.
        max_retries: Total attempts before giving up.

    Returns:
        The Fyers response dict on the first ``status == "success"`` attempt,
        or None if every attempt failed.
    """
    import time as _time
    last_err = None
    for attempt in range(max_retries):
        try:
            res = svc.history(
                user_id=user_id, symbol=fyers_sym, exchange="NSE",
                interval=interval, start_date=start_str, end_date=end_str,
            )
            if res and res.get("status") == "success":
                return res
            # Non-success response: remember the message for the final log line.
            last_err = (res or {}).get("message", "no response")
        except Exception as e:
            last_err = str(e)
        # Back off before the next attempt (skip the sleep after the last try).
        if attempt < max_retries - 1:
            _time.sleep(2 ** (attempt + 1))
    print(f"  fyers chunk fail {fyers_sym} {start_str}..{end_str} "
          f"after {max_retries} retries: {last_err}")
    return None


def _fetch_fyers_interval(symbol: str, days: int, user_id: int,
                          interval: str, chunk_days: int = 95) -> pd.DataFrame:
    """Generic Fyers history fetcher for a single interval.

    Walks the [now - days, now] window in ``chunk_days`` slices (Fyers caps the
    span per request, and the cap is tighter for intraday intervals), concatenates
    all candles, then cleans and normalizes them into a tidy OHLCV DataFrame.

    Args:
        symbol: Plain/Yahoo/Fyers ticker (normalized internally).
        days: Calendar days of history to pull, counting back from now.
        user_id: Broker account user id authorizing the Fyers call.
        interval: Fyers interval code ("1h", "15m", ...).
        chunk_days: Max span per Fyers request; window is sliced into these.

    Returns:
        DataFrame with columns [timestamp, candle_time, open, high, low, close,
        volume], de-duplicated and sorted ascending by timestamp. Empty
        DataFrame if the service is unavailable or no candles came back.
    """
    svc = _fyers_service()
    if svc is None:
        return pd.DataFrame()
    fyers_sym = to_fyers_symbol(symbol)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    cursor = start_dt
    all_candles: List = []
    # Slice the window so each request stays within Fyers's per-call day cap.
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        res = _history_with_retry(
            svc, fyers_sym, user_id, interval,
            cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
        )
        if res:
            all_candles += res.get("data", {}).get("candles", []) or []
        cursor = chunk_end
    if not all_candles:
        return pd.DataFrame()
    # Fyers may return candles as dicts or as positional [ts,o,h,l,c,v] arrays.
    if isinstance(all_candles[0], dict):
        df = pd.DataFrame(all_candles)
    else:
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high",
                                                 "low", "close", "volume"])
    # Coerce everything numeric; bad cells become NaN/NA for the dropna below.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    # Overlapping chunk boundaries can repeat a bar → de-dupe on timestamp.
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Fyers epoch seconds are UTC; convert to naive IST wall-clock for storage.
    df["candle_time"] = pd.to_datetime(df["timestamp"].astype("int64"),
                                       unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low",
               "close", "volume"]]


def _fetch_daily_fyers_raw(symbol: str, days: int, user_id: int = 1,
                            chunk_days: int = 360) -> pd.DataFrame:
    """Fetch daily candles directly from Fyers (no cache). Used by prefetch.

    ``chunk_days=360`` stays safely under Fyers's 366-day max-per-request limit
    (cursor → cursor+chunk_days = 361-day span). chunk_days=365 was hitting
    the limit intermittently with -50 'Invalid input' errors.

    Robustness: each chunk goes through ``_history_with_retry`` and the cursor
    always advances; so recent IPOs that have no data for early chunks still get
    later chunks pulled successfully (one bad chunk never aborts the symbol).

    Args:
        symbol: Plain/Yahoo/Fyers ticker (normalized internally).
        days: Calendar days of daily history to pull, counting back from now.
        user_id: Broker account user id authorizing the Fyers call.
        chunk_days: Max span per request; window is sliced into these (<=360).

    Returns:
        DataFrame with columns [timestamp, candle_time, open, high, low, close,
        volume], de-duplicated and sorted ascending. Empty DataFrame if the
        service is unavailable or no candles came back.
    """
    svc = _fyers_service()
    if svc is None:
        return pd.DataFrame()
    fyers_sym = to_fyers_symbol(symbol)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    cursor = start_dt
    all_candles: List = []
    # Slice into <=360-day spans to stay under Fyers's 366-day daily-request cap.
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        res = _history_with_retry(
            svc, fyers_sym, user_id, "D",
            cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
        )
        if res:
            all_candles += res.get("data", {}).get("candles", []) or []
        cursor = chunk_end
    if not all_candles:
        return pd.DataFrame()
    # Fyers may return candles as dicts or as positional [ts,o,h,l,c,v] arrays.
    if isinstance(all_candles[0], dict):
        df = pd.DataFrame(all_candles)
    else:
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high",
                                                 "low", "close", "volume"])
    # Coerce everything numeric; bad cells become NaN/NA for the dropna below.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    # Overlapping chunk boundaries can repeat a bar → de-dupe on timestamp.
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Fyers epoch seconds are UTC; convert to naive IST wall-clock for storage.
    df["candle_time"] = pd.to_datetime(df["timestamp"].astype("int64"),
                                       unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low", "close", "volume"]]
