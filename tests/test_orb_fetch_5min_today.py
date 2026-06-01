"""Lock the ORB intraday single-day fetch (2026-06-01).

Bug: fetch_5min looped `while cur < end`, so the LIVE path fetch_5min(today,
today) (start == end) ran ZERO iterations and returned None — ORB never got
today's 5-min bars and never bought. Fixed to `cur <= end`. These tests assert
the single-day (today) range issues at least one history call and returns bars,
and that a multi-day range still works.
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.models.orb_momentum_intraday.data import fetch_5min


class _FakeFy:
    def __init__(self):
        self.calls = []

    def history(self, params):
        self.calls.append(params)
        # one 5-min candle: [ts, o, h, l, c, v]
        return {"s": "ok", "candles": [[1717200000, 100.0, 101.0, 99.5, 100.5, 1000]]}


def test_single_day_today_fetches():
    fy = _FakeFy()
    today = dt.date(2026, 6, 1)
    df = fetch_5min("RELIANCE", today, today, fy=fy)
    assert len(fy.calls) == 1, "single-day (today) range must issue exactly one history call"
    assert df is not None and len(df) == 1
    # symbol wrapped to Fyers form
    assert fy.calls[0]["symbol"] == "NSE:RELIANCE-EQ"
    assert fy.calls[0]["resolution"] == "5"
    assert fy.calls[0]["range_from"] == "2026-06-01"
    assert fy.calls[0]["range_to"] == "2026-06-01"


def test_multi_day_still_fetches():
    fy = _FakeFy()
    df = fetch_5min("RELIANCE", dt.date(2026, 5, 1), dt.date(2026, 6, 1), fy=fy)
    assert len(fy.calls) >= 1
    assert df is not None


def test_no_data_returns_none():
    class _Empty(_FakeFy):
        def history(self, params):
            self.calls.append(params)
            return {"s": "no_data", "candles": []}
    fy = _Empty()
    today = dt.date(2026, 6, 1)
    assert fetch_5min("RELIANCE", today, today, fy=fy) is None
