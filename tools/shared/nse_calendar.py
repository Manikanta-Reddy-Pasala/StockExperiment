"""NSE trading-calendar helpers (single source of truth for "is the market open").

Separation of concerns: ALL holiday / first-trading-day logic lives here.
Callers (notification_service, the momentum live_signal rebalance gate, the
executor holiday guard, the picks UI via /api/nse-holidays) import from this
module rather than re-deriving the weekday/holiday rule themselves.

Holiday source priority (fail-safe):
  1. PRIMARY  — the NSE official API (``/api/holiday-master?type=trading``),
     fetched on a schedule and cached to ``HOLIDAY_CACHE_PATH``. Picked up
     automatically by ``_load_holidays`` (memoized on cache mtime, so a
     long-running scheduler sees a refreshed cache WITHOUT a restart).
  2. FALLBACK — the hardcoded ``_FALLBACK_HOLIDAYS`` frozenset below. Used
     offline / when the API is unreachable / when the cache is missing or
     corrupt. The loaded set is always the UNION of cache + fallback, so a
     stale or partial cache can never make us treat a known holiday as a
     trading day.

A network/parse failure NEVER raises and NEVER blocks trading — it just keeps
the fallback list. ``_FALLBACK_HOLIDAYS`` should still be refreshed yearly as a
belt-and-suspenders safety net (NSE publishes the next year's list in December).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Path where the scheduled fetch persists the NSE-sourced holiday list. Lives in
# the shared /app/logs volume so the scheduler (writer) and the web app (reader
# via /api/nse-holidays) and the live_signal/executor (readers) all see it.
HOLIDAY_CACHE_PATH = "/app/logs/nse_holidays.json"

# NSE official trading-holiday API. CM = capital market (equity) segment.
_NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
_NSE_HOME_URL = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
# NSE returns tradingDate like "15-Jan-2026".
_NSE_DATE_FMT = "%d-%b-%Y"

# OFFLINE FALLBACK NSE equity-segment trading holidays. Refresh yearly.
# 2025 + 2026 weekday holidays. Saturdays/Sundays are handled by the weekday
# rule, so a holiday that lands on a weekend need not be listed.
_FALLBACK_HOLIDAYS = frozenset({
    # ---- 2025 ----
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr (Ramadan Eid)
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali Laxmi Pujan (Muhurat session aside)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas

    # ---- 2026 (per NSE official trading-holiday circular — weekday closures.
    # Weekend-only holidays (Mahashivratri 02-15 Sun, Id-ul-Fitr 03-21 Sat,
    # Independence 08-15 Sat, Diwali-Laxmi-Pujan 11-08 Sun) are covered by the
    # weekday rule and not listed.) ----
    date(2026, 1, 15),   # Maharashtra municipal elections (Thu)
    date(2026, 1, 26),   # Republic Day (Mon)
    date(2026, 3, 3),    # Holi (Tue)
    date(2026, 3, 26),   # Shri Ram Navami (Thu)
    date(2026, 3, 31),   # Mahavir Jayanti (Tue)
    date(2026, 4, 3),    # Good Friday (Fri)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti (Tue)
    date(2026, 5, 1),    # Maharashtra Day (Fri)
    date(2026, 5, 28),   # Bakri Id (Thu)
    date(2026, 6, 26),   # Muharram (Fri)
    date(2026, 9, 14),   # Ganesh Chaturthi (Mon)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti (Fri)
    date(2026, 10, 20),  # Dussehra (Tue)
    date(2026, 11, 10),  # Diwali Balipratipada (Tue)
    date(2026, 11, 24),  # Guru Nanak Jayanti (Tue)
    date(2026, 12, 25),  # Christmas (Fri)
})

# Backward-compat alias. Older callers import ``NSE_HOLIDAYS`` directly. It now
# points at the offline fallback set (the merged/live set is via _load_holidays).
NSE_HOLIDAYS = _FALLBACK_HOLIDAYS

# Memoization of the merged (cache ∪ fallback) set, keyed by the cache file's
# mtime. A long-running scheduler picks up a refreshed cache automatically (the
# mtime changes) WITHOUT a restart, but we never re-read/re-parse on every call.
# Key sentinel ``None`` = cache file absent on the last load.
_CACHE_KEY: Optional[float] = object()  # type: ignore[assignment]  # impossible mtime → forces first load
_CACHED_SET: frozenset = _FALLBACK_HOLIDAYS


def fetch_nse_holidays(timeout: int = 12) -> Optional[set]:
    """Fetch NSE equity (CM) trading holidays from the official API and cache them.

    Fail-safe: returns ``None`` (and logs a warning) on ANY error — network,
    non-200, JSON parse, missing CM key, or date-parse. The caller keeps the
    offline fallback. NEVER raises.

    On success: parses every CM entry's ``tradingDate`` ("%d-%b-%Y"), writes
    ``HOLIDAY_CACHE_PATH`` as JSON ``{"fetched_at": iso, "holidays": [iso,...]}``
    and returns the set of ``date`` objects.

    NSE blocks bare requests, so we use a Session with browser headers and
    prime cookies by GETting the homepage first.
    """
    try:
        import os
        import requests
    except Exception as e:  # requests not installed — keep fallback
        logger.warning("fetch_nse_holidays: requests import failed: %s", e)
        return None

    try:
        s = requests.Session()
        # Prime cookies — NSE rejects the API call without a homepage cookie set.
        s.get(_NSE_HOME_URL, headers=_NSE_HEADERS, timeout=timeout)
        r = s.get(_NSE_HOLIDAY_URL, headers=_NSE_HEADERS, timeout=timeout)
        if r.status_code != 200:
            logger.warning("fetch_nse_holidays: non-200 from NSE (%s)", r.status_code)
            return None
        data = r.json()
        cm = data.get("CM")
        if not isinstance(cm, list) or not cm:
            logger.warning("fetch_nse_holidays: CM segment missing/empty in response")
            return None

        holidays: set = set()
        for entry in cm:
            raw = (entry or {}).get("tradingDate")
            if not raw:
                continue
            try:
                holidays.add(datetime.strptime(raw.strip(), _NSE_DATE_FMT).date())
            except (ValueError, AttributeError):
                # Skip a single malformed date rather than failing the whole fetch.
                logger.warning("fetch_nse_holidays: unparseable tradingDate %r", raw)
                continue

        if not holidays:
            logger.warning("fetch_nse_holidays: parsed 0 dates from CM segment")
            return None

        # Persist cache (best-effort — a write failure should not lose the result).
        try:
            os.makedirs(os.path.dirname(HOLIDAY_CACHE_PATH), exist_ok=True)
            payload = {
                "fetched_at": datetime.now().isoformat(),
                "holidays": sorted(d.isoformat() for d in holidays),
            }
            with open(HOLIDAY_CACHE_PATH, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning("fetch_nse_holidays: cache write failed: %s", e)

        return holidays
    except Exception as e:
        logger.warning("fetch_nse_holidays: fetch/parse failed: %s", e)
        return None


def _load_holidays() -> frozenset:
    """Return the live holiday set: cache (if present/valid) UNION fallback.

    Memoized on the cache file's mtime so a refreshed cache is picked up without
    a restart, but we don't re-read/re-parse on every call. On a missing or
    corrupt cache, returns ``_FALLBACK_HOLIDAYS`` (fail-safe).
    """
    global _CACHE_KEY, _CACHED_SET

    import os
    try:
        mtime: Optional[float] = os.path.getmtime(HOLIDAY_CACHE_PATH)
    except OSError:
        mtime = None  # cache absent

    if mtime == _CACHE_KEY:
        return _CACHED_SET

    # Cache state changed (or first call) — (re)build the merged set.
    merged: frozenset = _FALLBACK_HOLIDAYS
    if mtime is not None:
        try:
            with open(HOLIDAY_CACHE_PATH) as f:
                payload = json.load(f)
            parsed = set()
            for iso in payload.get("holidays", []):
                try:
                    parsed.add(date.fromisoformat(iso))
                except (ValueError, TypeError):
                    continue
            if parsed:
                merged = frozenset(parsed | _FALLBACK_HOLIDAYS)
            else:
                logger.warning("_load_holidays: cache had 0 valid dates, using fallback")
        except Exception as e:
            logger.warning("_load_holidays: corrupt cache (%s), using fallback", e)
            merged = _FALLBACK_HOLIDAYS

    _CACHE_KEY = mtime
    _CACHED_SET = merged
    return merged


def _as_date(d: Union[date, datetime, None]) -> date:
    """Coerce a datetime/date/None to a plain ``date`` (None -> today)."""
    if d is None:
        return datetime.now().date()
    if isinstance(d, datetime):
        return d.date()
    return d


def is_trading_day(d: Union[date, datetime, None] = None) -> bool:
    """True if ``d`` is an NSE trading day: a weekday (Mon-Fri) not a holiday.

    Uses the live merged holiday set (NSE cache ∪ offline fallback).

    Args:
        d: date/datetime to test (defaults to today).

    Returns:
        bool: True only on Mon-Fri that are not listed NSE holidays.
    """
    dd = _as_date(d)
    return dd.weekday() < 5 and dd not in _load_holidays()


def first_trading_day_of_month(d: Union[date, datetime, None] = None) -> date:
    """Return the first NSE trading day on/after the 1st of ``d``'s month.

    Walks forward from the 1st until a trading day is found (skips a 1st that
    falls on a weekend or holiday).
    """
    dd = _as_date(d)
    cur = date(dd.year, dd.month, 1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def is_first_trading_day_of_month(d: Union[date, datetime, None] = None) -> bool:
    """True if ``d`` is the first NSE trading day of its month."""
    dd = _as_date(d)
    return dd == first_trading_day_of_month(dd)


if __name__ == "__main__":
    # Manual refresh: `python3 tools/shared/nse_calendar.py`
    # Used by the scheduler job + manual testing. Exits 0 on success, 1 on
    # failure (so a wrapping subprocess sees a non-zero code and can retry/alert).
    import sys as _sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    result = fetch_nse_holidays()
    if result is None:
        print("FAILED to fetch NSE holidays — fallback list remains in effect.")
        _sys.exit(1)
    print(f"Fetched {len(result)} NSE trading holidays → {HOLIDAY_CACHE_PATH}")
    _sys.exit(0)
