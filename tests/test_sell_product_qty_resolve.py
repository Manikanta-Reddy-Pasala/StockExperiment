"""Sell-time product + qty resolution against the BROKER (the user's directive:
"check stock type intra/delivery and qty for that model, then sell").

2026-06-02 SPARC phantom short: the ORB square-off sold INTRADAY (model policy)
148 shares (ledger qty) while the position was actually held as CNC at the
broker → the intraday sell opened a short instead of closing the delivery long.
resolve_sell_product_qty fixes both halves: sell the product the shares are
REALLY in, and never more than the broker holds.
"""
import tools.live.fyers_executor as FE


class _Svc:
    """Fake broker: holdings() = CNC/delivery, positions = same-day legs."""
    def __init__(self, holdings=None, positions=None, raise_holdings=False,
                 raise_positions=False):
        self._h = holdings or []
        self._p = positions or []
        self._rh = raise_holdings
        self._rp = raise_positions

    def holdings(self, user_id):
        if self._rh:
            raise RuntimeError("holdings api down")
        return {"data": self._h}

    def _get_api_instance(self, user_id):
        svc = self
        class _Api:
            def _make_request(self, _m, _ep):
                if svc._rp:
                    raise RuntimeError("positions api down")
                return {"data": svc._p}
        return _Api()


def test_cnc_delivery_holding_resolves_cnc():
    # 148 held as delivery (holdings) → sell CNC, capped to 148.
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "148"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148)
    assert (prod, qty, src) == ("CNC", 148, "broker")


def test_cnc_position_same_day_resolves_cnc():
    # Same-day CNC buy shows in positions with productType CNC (not yet in
    # holdings) — this is the exact SPARC case. Must sell CNC, not INTRADAY.
    svc = _Svc(positions=[{"symbol": "NSE:SPARC-EQ", "netQty": "148", "productType": "CNC"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148)
    assert (prod, qty, src) == ("CNC", 148, "broker")


def test_intraday_position_resolves_intraday():
    svc = _Svc(positions=[{"symbol": "NSE:HFCL-EQ", "netQty": "500", "productType": "INTRADAY"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:HFCL-EQ", 500)
    assert (prod, qty, src) == ("INTRADAY", 500, "broker")


def test_caps_qty_to_broker_available():
    # Ledger thinks 148, broker only has 100 (partial manual sale) → sell 100,
    # never 148 (selling 148 would short 48).
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "100"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148)
    assert (prod, qty, src) == ("CNC", 100, "broker")


def test_broker_flat_returns_flat():
    # Nothing held at the broker (already sold manually) → flat → caller SKIPS.
    svc = _Svc(holdings=[], positions=[])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148)
    assert (prod, qty, src) == (None, 0, "flat")


def test_short_position_not_sellable():
    # A net SHORT (negative) is not a sellable long → flat (don't sell into a short).
    svc = _Svc(positions=[{"symbol": "NSE:X-EQ", "netQty": "-50", "productType": "INTRADAY"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:X-EQ", 50)
    assert src == "flat"


def test_api_failure_falls_back():
    # Both broker calls fail → fall back to model-policy product + ledger qty so
    # an API hiccup never blocks a real exit (over-sell guard is the backstop).
    svc = _Svc(raise_holdings=True, raise_positions=True)
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148)
    assert (prod, qty, src) == (None, 148, "fallback")


def test_explicit_product_bypasses_policy_guard(monkeypatch):
    # An EXIT passing product="CNC" for the intraday ORB model must NOT be
    # force-corrected to INTRADAY by the _placeorder guard (that override is the
    # whole point — close the CNC leg the shares are really in).
    monkeypatch.setattr(FE, "_CURRENT_MODEL", "orb_momentum_intraday")
    monkeypatch.setattr(FE, "_tg_safe", lambda *a, **k: None)
    import src.services.audit_service as AS
    monkeypatch.setattr(AS, "write_order", lambda *a, **k: 1)

    captured = {}
    class _S:
        def placeorder(self, **kw):
            captured["product"] = kw.get("product")
            return {"status": "ok", "id": "OID"}
    FE._placeorder(_S(), 1, "NSE:SPARC-EQ", 148, "SELL",
                   pricetype="MARKET", product="CNC")
    assert captured["product"] == "CNC", "explicit exit product must be honored"
