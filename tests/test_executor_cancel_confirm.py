"""Tests for the executor's cancel-before-replace confirmation.

Locks the fix for the 2026-05-25 VEDL "Margin Shortfall" failure: the
LIMIT-walk must confirm a cancel actually released margin before placing a
replacement / MARKET order. A fire-and-forget cancel left the prior LIMIT
working, so the account held two reservations and Fyers rejected the second
order for ~2x margin.

Uses a fake broker (no network). Run:
    python3 -m pytest tests/test_executor_cancel_confirm.py -q
"""
from tools.live.fyers_executor import _cancel_and_confirm


class FakeSvc:
    """Returns a scripted sequence of order-book rows on successive polls.

    `rows` is a list; each orderbook() call yields the next one (last one
    sticks). `None` means "order not in book".
    """
    def __init__(self, rows):
        self._rows = rows
        self._i = 0
        self.cancel_calls = 0

    def cancelorder(self, user_id, orderid):
        self.cancel_calls += 1
        return {"s": "ok"}

    def orderbook(self, user_id):
        row = self._rows[min(self._i, len(self._rows) - 1)]
        self._i += 1
        return {"data": [row] if row else []}


def _row(status, qty=90, filled=0, oid="X1"):
    return {"orderid": oid, "status": status,
            "quantity": str(qty), "filled_quantity": str(filled)}


def test_cancel_confirmed_returns_cancelled():
    # pending, then cancelled
    svc = FakeSvc([_row(6), _row(1)])
    assert _cancel_and_confirm(svc, 1, "X1", timeout_s=2, poll_s=0.05) == "cancelled"
    assert svc.cancel_calls >= 1


def test_cancel_confirmed_via_mapped_string():
    svc = FakeSvc([_row("PENDING"), _row("CANCELLED")])
    assert _cancel_and_confirm(svc, 1, "X1", timeout_s=2, poll_s=0.05) == "cancelled"


def test_order_fills_during_cancel_returns_filled():
    # Race: order fills (filled_quantity == quantity) before cancel lands.
    svc = FakeSvc([_row(6, filled=0), _row(2, filled=90)])
    assert _cancel_and_confirm(svc, 1, "X1", timeout_s=2, poll_s=0.05) == "filled"


def test_cancel_never_confirms_returns_still_live():
    # Order stays PENDING the whole window -> must NOT be treated as cancelled,
    # so the caller will refuse to place a replacement (no double-margin).
    svc = FakeSvc([_row(6)])
    assert _cancel_and_confirm(svc, 1, "X1", timeout_s=1, poll_s=0.05) == "still_live"
    # Cancel is re-sent at least once before giving up.
    assert svc.cancel_calls >= 2


def test_empty_order_id_is_noop_cancelled():
    svc = FakeSvc([_row(6)])
    assert _cancel_and_confirm(svc, 1, "", timeout_s=1, poll_s=0.05) == "cancelled"
    assert svc.cancel_calls == 0
