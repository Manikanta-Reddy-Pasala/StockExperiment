"""Unit tests for the single notification funnel.

Pure routing/format/dedupe logic — the DB layer (`notify()` persistence) and
Telegram send are monkeypatched, so no Postgres / network needed. Run:
    python3 -m pytest tests/test_notification_service.py -q
"""
from datetime import datetime

import pytest

from src.services import notification_service as ns


# ----- pure helpers ---------------------------------------------------------

def test_signature_holds_are_stable_and_dedupe():
    a = ns._decision_signature([], "BSE")
    b = ns._decision_signature([], "BSE")
    assert a == b == "HOLD:BSE"


def test_signature_flat_vs_held_differ():
    assert ns._decision_signature([], None) == "HOLD:FLAT"
    assert ns._decision_signature([], "BSE") != ns._decision_signature([], None)


def test_signature_changes_when_action_appears():
    hold = ns._decision_signature([], "BSE")
    act = ns._decision_signature(
        [{"signal": "ENTRY1", "symbol": "VEDL", "side": "BUY"}], "BSE")
    assert hold != act


def test_decision_message_no_change_with_return():
    title, body = ns._decision_message("m", [], "BSE", 12.34, None)
    assert title == "m: no change"
    assert "BSE" in body and "+12.34%" in body


def test_decision_message_flat():
    _, body = ns._decision_message("m", [], None, None, None)
    assert "flat" in body.lower()


def test_decision_message_action_lists_signal():
    _, body = ns._decision_message(
        "m", [{"signal": "ROTATE", "symbol": "BSE", "side": "BUY",
               "price": 2345.0}], None, None, None)
    assert "ROTATE" in body and "BSE" in body and "2,345" in body


def test_is_trading_day():
    assert ns.is_trading_day(datetime(2026, 5, 25))   # Monday
    assert not ns.is_trading_day(datetime(2026, 5, 23))  # Saturday
    assert not ns.is_trading_day(datetime(2026, 5, 24))  # Sunday


# ----- routing (notify monkeypatched) ---------------------------------------

@pytest.fixture
def captured(monkeypatch):
    """Capture notify() calls without touching DB/Telegram."""
    calls = []

    def fake_notify(event_type, **kw):
        calls.append({"event": event_type, **kw})
        return {"ok": True, "id": len(calls)}

    monkeypatch.setattr(ns, "notify", fake_notify)
    # Verdict pings (plan / no-change / skip) are env-gated so ONLY the
    # scheduled cron emit notifies — diagnostic/preview runs stay silent.
    # The routing tests emulate that cron context.
    monkeypatch.setenv("MOMROT_TG_NOTIFY", "1")
    return calls


def test_verdict_notifications_gated_off_without_env(monkeypatch):
    """No MOMROT_TG_NOTIFY => verdict producers are silent (leak guard)."""
    monkeypatch.delenv("MOMROT_TG_NOTIFY", raising=False)
    calls = []
    monkeypatch.setattr(ns, "notify", lambda *a, **k: calls.append(1))
    r1 = ns.notify_model_decision("m", [], held_symbol="BSE",
                                  today=datetime(2026, 5, 25))
    r2 = ns.notify_skip("m", "x", today=datetime(2026, 5, 25))
    assert r1["skipped"] and r1["reason"] == "notify_gated"
    assert r2["skipped"] and r2["reason"] == "notify_gated"
    assert calls == []


def test_decision_weekend_is_silent(captured):
    res = ns.notify_model_decision("m", [], held_symbol="BSE",
                                   today=datetime(2026, 5, 23))  # Sat
    assert res["skipped"] and res["reason"] == "weekend"
    assert captured == []


def test_decision_no_change_routes_no_change_event(captured):
    ns.notify_model_decision("m", [], held_symbol="BSE",
                             today=datetime(2026, 5, 25))  # Mon
    assert len(captured) == 1
    c = captured[0]
    assert c["event"] == ns.NO_CHANGE
    assert c["telegram"] is True
    assert c["dedupe_key"] == "CRON:HOLD:BSE"


def test_decision_action_routes_signal_event(captured):
    ns.notify_model_decision(
        "m", [{"signal": "ENTRY1", "symbol": "VEDL", "side": "BUY"}],
        today=datetime(2026, 5, 25))
    c = captured[0]
    assert c["event"] == ns.SIGNAL
    assert c["level"] == "success"
    assert "ENTRY1:VEDL:BUY" in c["dedupe_key"]


def test_decision_trigger_in_dedupe_key(captured):
    ns.notify_model_decision("m", [], held_symbol="BSE", trigger="MID_MONTH",
                             today=datetime(2026, 5, 25))
    assert captured[0]["dedupe_key"] == "MID_MONTH:HOLD:BSE"


def test_skip_weekend_silent(captured):
    res = ns.notify_skip("m", "not rebalance day",
                         today=datetime(2026, 5, 24))  # Sun
    assert res["skipped"]
    assert captured == []


def test_skip_records_db_only(captured):
    ns.notify_skip("m", "not rebalance day", today=datetime(2026, 5, 25))
    c = captured[0]
    assert c["event"] == ns.SKIP
    assert c["telegram"] is False
    assert c["dedupe_key"] == "SKIP:not rebalance day"


def test_order_failed_inferred_from_emoji(captured):
    ns.notify_order("⚠️ BUY m VEDL placed but ledger write FAILED: boom")
    c = captured[0]
    assert c["event"] == ns.ORDER_FAILED
    assert c["level"] == "error"
    assert c["telegram"] is True


def test_order_placed_inferred(captured):
    ns.notify_order("✅ *BUY m*\n`VEDL` x90 @ ₹329.95")
    c = captured[0]
    assert c["event"] == ns.ORDER_PLACED
    assert c["level"] == "success"
    # title cleaned of emoji + markdown
    assert c["title"].startswith("BUY m")
