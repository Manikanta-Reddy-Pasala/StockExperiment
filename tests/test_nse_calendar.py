"""Unit tests for the NSE trading-calendar helpers (FIX 4).

Pure-logic, no DB / no network. Guards the holiday set and the
first-trading-day-of-month rule that the rebalance gate + executor depend on.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.shared.nse_calendar as cal
from tools.shared.nse_calendar import (
    NSE_HOLIDAYS,
    is_trading_day,
    first_trading_day_of_month,
    is_first_trading_day_of_month,
)


def test_known_holiday_is_not_trading_day():
    # Republic Day 2026 (a Monday) is an NSE holiday.
    assert date(2026, 1, 26) in NSE_HOLIDAYS
    assert not is_trading_day(date(2026, 1, 26))


def test_normal_weekday_is_trading_day():
    # 2026-01-27 (Tuesday) is a plain trading day.
    assert is_trading_day(date(2026, 1, 27))


def test_weekend_is_not_trading_day():
    # 2026-01-24 is a Saturday.
    assert not is_trading_day(date(2026, 1, 24))


def test_first_trading_day_skips_holiday_at_month_start():
    # 2026-01-01 (New Year) is a Thursday but a trading holiday is NOT listed
    # for it here; verify the generic mechanic with January 2026 where the 1st
    # falls on a weekday holiday-free day -> first trading day is the 1st itself
    # only if it's a trading day. Use a month whose 1st is a weekend instead:
    # 2026-02-01 is a Sunday -> first trading day is Monday 2026-02-02.
    assert first_trading_day_of_month(date(2026, 2, 15)) == date(2026, 2, 2)
    assert is_first_trading_day_of_month(date(2026, 2, 2))
    assert not is_first_trading_day_of_month(date(2026, 2, 1))   # Sunday
    assert not is_first_trading_day_of_month(date(2026, 2, 3))   # 2nd trading day


def test_first_trading_day_when_first_is_holiday():
    # 2026-05-01 (Maharashtra Day) is a Friday holiday -> first trading day of
    # May 2026 is Monday 2026-05-04 (5/2 + 5/3 are the weekend).
    assert date(2026, 5, 1) in NSE_HOLIDAYS
    assert first_trading_day_of_month(date(2026, 5, 20)) == date(2026, 5, 4)
    assert is_first_trading_day_of_month(date(2026, 5, 4))
    assert not is_first_trading_day_of_month(date(2026, 5, 1))


# --- Auto-sourced holiday cache (PRIMARY source) ---------------------------
# These tests must NEVER hit the live network — requests is monkeypatched.

def test_backward_compat_alias_points_at_fallback():
    # Anything importing the old NSE_HOLIDAYS name still works; it now aliases
    # the offline fallback set.
    assert NSE_HOLIDAYS is cal._FALLBACK_HOLIDAYS


def test_fetch_returns_none_on_network_error(monkeypatch):
    """A network failure must return None gracefully (caller keeps fallback)."""
    class _BoomSession:
        def get(self, *a, **k):
            raise RuntimeError("network down")

    import requests
    monkeypatch.setattr(requests, "Session", lambda: _BoomSession())
    assert cal.fetch_nse_holidays(timeout=1) is None


def test_load_holidays_falls_back_when_cache_absent(monkeypatch, tmp_path):
    """No cache file -> _load_holidays() returns the fallback set verbatim."""
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(cal, "HOLIDAY_CACHE_PATH", str(missing))
    # Reset memoization so the patched path is honoured.
    monkeypatch.setattr(cal, "_CACHE_KEY", object())
    monkeypatch.setattr(cal, "_CACHED_SET", cal._FALLBACK_HOLIDAYS)
    assert cal._load_holidays() == cal._FALLBACK_HOLIDAYS


def test_load_holidays_unions_cache_with_fallback(monkeypatch, tmp_path):
    """A valid cache is UNIONed with the fallback (cache can add, never drop)."""
    import json
    extra = date(2099, 7, 7)  # a date NOT in the fallback set
    cache = tmp_path / "nse_holidays.json"
    cache.write_text(json.dumps({
        "fetched_at": "2026-01-01T00:00:00",
        "holidays": [extra.isoformat()],
    }))
    monkeypatch.setattr(cal, "HOLIDAY_CACHE_PATH", str(cache))
    monkeypatch.setattr(cal, "_CACHE_KEY", object())
    monkeypatch.setattr(cal, "_CACHED_SET", cal._FALLBACK_HOLIDAYS)
    merged = cal._load_holidays()
    assert extra in merged                       # cache date present
    assert date(2026, 1, 26) in merged           # fallback date still present
    assert not is_trading_day(extra)             # uses the merged set
