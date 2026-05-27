"""OHLCV cache using EXISTING Postgres tables (no new tables created).

Maps backtest fetcher requests to the prod app's historical_data tables:
    "1h"  -> historical_data_1h     (PK: symbol, timestamp)
    "15m" -> historical_data_15m    (PK: symbol, timestamp)
    "D"   -> historical_data        (PK: symbol, date)
    "5m"  -> not cached (no table); always hits Fyers

Symbol format inside the tables is the full Fyers form
``NSE:<TICKER>-EQ`` (e.g. ``NSE:RELIANCE-EQ``). Backtest harness passes
plain tickers (``RELIANCE``); we normalize via ``to_fyers_symbol`` from
the EMA harness before any DB read/write.

Usage from a fetcher:
    from tools.shared.ohlcv_cache import get_or_fetch
    df = get_or_fetch("RELIANCE", "1h", days=365,
                      lambda sym, d: raw_fyers_fetch(sym, d, user_id=1))
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Maps backtest interval string -> (table_name, time_col_kind)
# time_col_kind: "ts"   = unique by (symbol, timestamp), candle_time present
#                "date" = unique by (symbol, date),     date col present
_TABLE_MAP = {
    "1h":  ("historical_data_1h",  "ts"),
    "15m": ("historical_data_15m", "ts"),
    "D":   ("historical_data",     "date"),
}

_engine = None


def _get_engine():
    """Return the shared SQLAlchemy engine for the prod ``historical_data`` DB.

    Lazily resolves the engine from the prod app's DatabaseManager
    (``src.models.database``) on first call and memoizes it in the module-level
    ``_engine`` global, so repeated cache reads/writes reuse one connection pool.

    Returns:
        A SQLAlchemy ``Engine``, or ``None`` if the prod DB layer can't be
        imported / initialized (e.g. running outside the app env). Callers
        treat ``None`` as "DB unreachable" and degrade gracefully (empty df /
        no-op write) rather than raising — backtests must not crash on a missing DB.
    """
    global _engine
    if _engine is not None:
        return _engine  # memoized: only resolve the engine once per process
    try:
        # Reuse the prod app's configured engine rather than building our own,
        # so this module always talks to the same Postgres as the live system.
        from src.models.database import get_database_manager
        dbm = get_database_manager()
        _engine = dbm.engine
        return _engine
    except Exception as e:
        logger.warning(f"ohlcv_cache: no DB engine ({e})")
        return None


def _to_fyers_sym(symbol: str) -> str:
    """Normalize a plain ticker to the Fyers symbol form stored in the DB.

    Every row in the ``historical_data*`` tables keys on the full Fyers form
    (``NSE:<TICKER>-EQ``). The backtest harness passes plain tickers, so this
    must run before any DB read/write or the WHERE clause won't match.

    Args:
        symbol: Plain ticker (``RELIANCE``), Yahoo form (``RELIANCE.NS``),
            an already-Fyers string (``NSE:RELIANCE-EQ`` / ``BSE:...``), or an
            index pseudo-ticker (``^NSEI``).

    Returns:
        The Fyers-form symbol string. Already-Fyers and index pseudo-tickers
        pass through unchanged; Yahoo's ``.NS`` suffix is stripped.
    """
    s = symbol.upper()
    if s.startswith("NSE:") or s.startswith("BSE:"):
        return s  # already in exchange:symbol form — leave as-is
    if s.startswith("^"):
        return s   # index pseudo-tickers (e.g. ^NSEI) — fetcher handles these
    # Plain/Yahoo ticker: drop a trailing .NS and wrap in NSE:...-EQ.
    return f"NSE:{s.replace('.NS', '')}-EQ"


def read_cached(symbol: str, interval: str,
                from_ts: int, to_ts: int) -> pd.DataFrame:
    """Read cached OHLCV rows for one symbol/interval within a time window.

    Routes the interval to its backing table via ``_TABLE_MAP`` and queries the
    right key column: intraday tables (1h/15m) range on the unix ``timestamp``
    column, the daily table ranges on the ``date`` column (the unix bounds are
    converted to UTC dates for that query).

    Args:
        symbol: Ticker in any accepted form (normalized to Fyers form internally).
        interval: ``"1h"``, ``"15m"`` or ``"D"``. Any other value (e.g. ``"5m"``,
            which has no table) yields an empty frame.
        from_ts: Inclusive window start as a unix timestamp (seconds).
        to_ts: Inclusive window end as a unix timestamp (seconds).

    Returns:
        DataFrame with columns ``[timestamp, candle_time, open, high, low,
        close, volume]`` sorted ascending by time. Empty DataFrame if the
        interval has no table, the DB is unreachable, the read fails, or no
        rows match. For the daily table, ``candle_time`` is synthesized from the
        ``date`` column so the schema matches the intraday tables.
    """
    if interval not in _TABLE_MAP:
        return pd.DataFrame()  # unmapped interval (e.g. 5m) — never cached
    table, kind = _TABLE_MAP[interval]
    eng = _get_engine()
    if eng is None:
        return pd.DataFrame()  # DB down — degrade to "no cache" rather than raise
    sym = _to_fyers_sym(symbol)  # WHERE clause keys on the Fyers-form symbol
    try:
        if kind == "ts":
            q = text(
                f"SELECT timestamp, candle_time, open, high, low, close, volume "
                f"FROM {table} "
                f"WHERE symbol = :sym AND timestamp BETWEEN :a AND :b "
                f"ORDER BY timestamp"
            )
            with eng.connect() as conn:
                df = pd.read_sql(q, conn, params={"sym": sym, "a": from_ts, "b": to_ts})
        else:  # daily table keys on a date column, not the unix timestamp
            # Daily 'date' rows are keyed on the IST trading date. Convert the
            # ts bounds with LOCAL time (container TZ=Asia/Kolkata), NOT UTC:
            # utcfromtimestamp().date() yields yesterday for any read before
            # 05:30 IST, silently dropping today's bar (stale ranking).
            from_d = datetime.fromtimestamp(from_ts).date()
            to_d   = datetime.fromtimestamp(to_ts).date()
            q = text(
                f"SELECT timestamp, date, open, high, low, close, volume "
                f"FROM {table} "
                f"WHERE symbol = :sym AND date BETWEEN :a AND :b "
                f"ORDER BY date"
            )
            with eng.connect() as conn:
                df = pd.read_sql(q, conn, params={"sym": sym, "a": from_d, "b": to_d})
            if not df.empty:
                # Synthesize candle_time from date so daily rows share the
                # intraday schema (callers expect a candle_time column).
                df["candle_time"] = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])
    except Exception as e:
        logger.warning(f"ohlcv_cache: read fail {sym}/{interval}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df
    # DB numerics may arrive as Decimal/str — coerce to float/int for the engine.
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low", "close", "volume"]]


def write_rows(symbol: str, interval: str, df: pd.DataFrame) -> int:
    """Upsert fetched OHLCV rows into the matching ``historical_data*`` table.

    Inserts every row in ``df``, stamping ``data_source='fyers'``, with
    ``ON CONFLICT DO NOTHING`` on the table's natural key — so re-running a
    prefetch only adds bars that aren't already cached (idempotent). Intraday
    tables conflict on ``(symbol, timestamp)``; the daily table on
    ``(symbol, date)`` (date derived from ``candle_time``).

    Args:
        symbol: Ticker in any accepted form (normalized to Fyers form internally).
        interval: ``"1h"``, ``"15m"`` or ``"D"``. Unmapped intervals are a no-op.
        df: OHLCV frame with at least ``timestamp, candle_time, open, high, low,
            close, volume`` (volume optional, defaults to 0).

    Returns:
        Count of rows submitted to the INSERT (note: this is the attempted row
        count, not the number actually inserted after conflict suppression).
        0 if the interval is unmapped, ``df`` is empty/None, the DB is
        unreachable, or the write fails.
    """
    if interval not in _TABLE_MAP or df is None or df.empty:
        return 0
    table, kind = _TABLE_MAP[interval]
    eng = _get_engine()
    if eng is None:
        return 0  # DB down — silently skip the write (backtest still proceeds)
    sym = _to_fyers_sym(symbol)
    try:
        if kind == "ts":
            q = text(
                f"INSERT INTO {table} "
                f"(symbol, timestamp, candle_time, open, high, low, close, volume, data_source) "
                f"VALUES (:sym, :ts, :ct, :o, :h, :l, :c, :v, 'fyers') "
                f"ON CONFLICT (symbol, timestamp) DO NOTHING"
            )
            params = [{
                "sym": sym, "ts": int(r.timestamp), "ct": r.candle_time,
                "o": float(r.open), "h": float(r.high),
                "l": float(r.low), "c": float(r.close),
                "v": int(getattr(r, "volume", 0) or 0),
            } for r in df.itertuples()]
        else:  # daily table — derive the date PK component from candle_time
            q = text(
                f"INSERT INTO {table} "
                f"(symbol, date, timestamp, open, high, low, close, volume, data_source) "
                f"VALUES (:sym, :dt, :ts, :o, :h, :l, :c, :v, 'fyers') "
                f"ON CONFLICT (symbol, date) DO NOTHING"
            )
            params = [{
                "sym": sym, "dt": pd.to_datetime(r.candle_time).date(),
                "ts": int(r.timestamp),
                "o": float(r.open), "h": float(r.high),
                "l": float(r.low), "c": float(r.close),
                "v": int(getattr(r, "volume", 0) or 0),  # volume may be absent
            } for r in df.itertuples()]
        # begin() opens a transaction and commits on clean exit; one batched
        # executemany handles all rows in a single round-trip.
        with eng.begin() as conn:
            conn.execute(q, params)
        return len(params)
    except Exception as e:
        logger.warning(f"ohlcv_cache: write fail {sym}/{interval}: {e}")
        return 0


def _expected_bars(interval: str, days: int) -> int:
    """Estimate how many bars a fully-populated window should contain.

    Used only by ``get_or_fetch`` to judge cache coverage before deciding
    whether to fall back to Fyers. Approximates ~250 trading days per 365
    calendar days, then multiplies by the typical bars-per-day for the
    interval (NSE session ~6h => 6 hourly bars, 25 fifteen-minute bars,
    75 five-minute bars; daily = 1).

    Args:
        interval: ``"1h"``, ``"15m"``, ``"5m"`` or ``"D"`` (unknown intervals
            fall back to one bar per trading day).
        days: Calendar-day length of the window.

    Returns:
        Approximate expected bar count (always >= 1).
    """
    trading_days = max(1, int(days * 250 / 365))  # ~250 trading days / year
    return {
        "1h":  trading_days * 6,    # ~6 hourly bars per NSE session
        "15m": trading_days * 25,   # ~25 fifteen-minute bars per session
        "5m":  trading_days * 75,   # ~75 five-minute bars per session
        "D":   trading_days,        # one daily bar per trading day
    }.get(interval, trading_days)


def get_or_fetch(symbol: str, interval: str, days: int,
                 fyers_fetcher: Callable[[str, int], pd.DataFrame],
                 min_coverage_frac: float = 0.0) -> pd.DataFrame:
    """Return OHLCV for the last ``days`` days.

    Cache-only mode (default min_coverage_frac=0): return whatever the
    cache has. Never call Fyers during backtest — accept that recent
    IPOs / sparse symbols have less data, strategy will skip them
    naturally via len(candles) check.

    Set ``min_coverage_frac=0.50`` to re-enable fallback Fyers fetch on
    low coverage.

    Args:
        symbol: Ticker in any accepted form.
        interval: ``"1h"``, ``"15m"`` or ``"D"``.
        days: Calendar-day lookback window ending now.
        fyers_fetcher: Callback ``(symbol, days) -> DataFrame`` invoked only on
            cache miss when fallback is enabled. Lets callers inject their own
            Fyers fetch + retry logic without this module depending on it.
        min_coverage_frac: Fraction of ``_expected_bars`` the cache must hold to
            be accepted without re-fetching. ``0`` (default) = pure cache mode.

    Returns:
        OHLCV DataFrame for the window — from cache, or freshly fetched (and
        written back) when fallback fires.
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    from_ts, to_ts = int(start_dt.timestamp()), int(end_dt.timestamp())

    cached = read_cached(symbol, interval, from_ts, to_ts)
    if min_coverage_frac <= 0:
        # Pure cache mode — no Fyers fallback (the backtest default).
        return cached
    expected = _expected_bars(interval, days)
    have = len(cached)
    # Cache is "good enough" only if it has both some rows and the required
    # coverage fraction; otherwise fall through to a live fetch.
    if have >= int(expected * min_coverage_frac) and have > 0:
        return cached
    fresh = fyers_fetcher(symbol, days)
    if fresh is None or fresh.empty:
        return cached  # fetch yielded nothing — keep whatever cache had
    write_rows(symbol, interval, fresh)  # persist for the next caller
    return fresh


def warm_cache_summary() -> dict:
    """Return per-(symbol, interval) cache footprint across all 3 tables."""
    eng = _get_engine()
    if eng is None:
        return {}
    out = {}
    for interval, (table, kind) in _TABLE_MAP.items():
        try:
            with eng.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT symbol, COUNT(*) AS n FROM {table} "
                    f"GROUP BY symbol ORDER BY n DESC LIMIT 30"
                )).fetchall()
            for r in rows:
                out[(r.symbol, interval)] = r.n
        except Exception as e:
            logger.warning(f"ohlcv_cache: summary fail {table}: {e}")
    return out


if __name__ == "__main__":
    summary = warm_cache_summary()
    if not summary:
        print("Cache empty (or DB unreachable).")
    else:
        print(f"{len(summary)} (symbol, interval) entries (top 30 per table):")
        for (sym, iv), n in sorted(summary.items(), key=lambda x: -x[1])[:30]:
            print(f"  {sym:25s} {iv:4s} n={n}")
