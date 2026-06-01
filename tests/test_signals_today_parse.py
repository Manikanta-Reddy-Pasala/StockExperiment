"""Lock _normalize_signal_payload — the Today's Signals payload coercer (2026-06-01).

Bug it fixes: Today's Signals only handled list / {"signals":[...]} shapes and
_SIGNAL_PATHS omitted emerging + retest, so neither showed up. retest also
writes a non-dated latest.json with a {buys,sells} dict — this normalizer turns
that into signal rows AND guards against showing a stale (different-day) file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web.admin_routes import _normalize_signal_payload as norm


def test_list_passthrough():
    assert norm([{"symbol": "X", "side": "BUY"}], "2026-06-01") == [{"symbol": "X", "side": "BUY"}]


def test_legacy_signals_wrapper():
    assert norm({"signals": [{"a": 1}]}, "2026-06-01") == [{"a": 1}]


def test_multi_buys_sells_today():
    r = norm({"date": "2026-06-01",
              "sells": [{"symbol": "A", "reason": "RANK_DROP"}],
              "buys": [{"symbol": "B"}, {"symbol": "C"}]}, "2026-06-01")
    assert r == [
        {"symbol": "A", "side": "SELL", "reason": "RANK_DROP"},
        {"symbol": "B", "side": "BUY"},
        {"symbol": "C", "side": "BUY"},
    ]


def test_multi_stale_day_returns_empty():
    # latest.json from a previous day must NOT show as today's signals
    assert norm({"date": "2026-05-30", "buys": [{"symbol": "B"}]}, "2026-06-01") == []


def test_multi_no_date_field_still_parses():
    # no date field -> trust it (can't prove stale)
    assert norm({"buys": [{"symbol": "B"}]}, "2026-06-01") == [{"symbol": "B", "side": "BUY"}]


def test_garbage_returns_empty():
    assert norm(None, "2026-06-01") == []
    assert norm(42, "2026-06-01") == []
