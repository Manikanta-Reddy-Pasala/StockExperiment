"""_placeorder product-safety guard — an INTRADAY model must place MIS, a swing
model must place CNC, and the two can never disagree.

2026-06-02 SPARC phantom-short: the morning ORB entry booked CNC (product
resolved from a then-unset global _CURRENT_MODEL → None → CNC), but the 15:10
square-off sold INTRADAY. Selling intraday against a delivery long opened a
phantom short that the broker auto-covered. These tests lock the guard that
re-resolves product from the known model and force-corrects any mismatch.
"""
import tools.live.fyers_executor as FE


class _FakeSvc:
    """Captures the product passed to placeorder."""
    def __init__(self):
        self.captured = {}

    def placeorder(self, **kw):
        self.captured = kw
        return {"status": "ok", "id": "OID1"}


def _place(monkeypatch, model, product=None):
    monkeypatch.setattr(FE, "_CURRENT_MODEL", model)
    monkeypatch.setattr(FE, "_tg_safe", lambda *a, **k: None)
    # neutralise the audit write (DB not available / irrelevant here)
    import src.services.audit_service as AS
    monkeypatch.setattr(AS, "write_order", lambda *a, **k: 1)
    svc = _FakeSvc()
    FE._placeorder(svc, 1, "NSE:SPARC-EQ", 10, "BUY",
                   pricetype="LIMIT", price=100.0, product=product)
    return svc.captured.get("product")


def test_swing_model_places_cnc(monkeypatch):
    assert _place(monkeypatch, "momentum_retest_n500") == "CNC"


def test_guard_corrects_mis_forced_on_swing_model(monkeypatch):
    assert _place(monkeypatch, "momentum_retest_n500", product="INTRADAY") == "CNC"


def test_no_model_defaults_cnc(monkeypatch):
    # Model unknown → safe default CNC (and the guard logs loudly, not tested here).
    assert _place(monkeypatch, None) == "CNC"
