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
    # Model qty 148 in CNC; account holds 148 CNC → sell CNC 148.
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "148"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 148, "broker")


def test_cnc_position_same_day_resolves_cnc():
    # Same-day CNC buy shows in positions with productType CNC (the SPARC case).
    svc = _Svc(positions=[{"symbol": "NSE:SPARC-EQ", "netQty": "148", "productType": "CNC"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 148, "broker")


def test_intraday_model_sells_intraday():
    svc = _Svc(positions=[{"symbol": "NSE:HFCL-EQ", "netQty": "500", "productType": "INTRADAY"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:HFCL-EQ", 500, "INTRADAY")
    assert (prod, qty, src) == ("INTRADAY", 500, "broker")


def test_qty_comes_from_model_not_account_when_shared():
    # CRITICAL: account holds 248 CNC (two models share SPARC), but THIS model
    # only owns 148. Must sell the MODEL's 148, never the account-wide 248.
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "248"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 148, "broker")


def test_caps_to_account_available_to_avoid_account_short():
    # Model thinks 148 but the account only holds 100 of that product (something
    # already squared) → cap to 100 so we never short the account.
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "100"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 100, "broker")


def test_account_flat_for_product_returns_flat():
    # Account holds 0 of the model's product for the symbol → flat → caller SKIPS.
    svc = _Svc(holdings=[], positions=[])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 0, "flat")


def test_only_other_product_held_is_flat_for_this_product():
    # Model is INTRADAY but the account only holds the name in CNC (another
    # model's delivery). This INTRADAY model has nothing to sell → flat (it must
    # NOT sell the other model's CNC shares).
    svc = _Svc(holdings=[{"symbol": "NSE:SPARC-EQ", "quantity": "100"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 50, "INTRADAY")
    assert (prod, qty, src) == ("INTRADAY", 0, "flat")


def test_short_position_not_sellable():
    svc = _Svc(positions=[{"symbol": "NSE:X-EQ", "netQty": "-50", "productType": "INTRADAY"}])
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:X-EQ", 50, "INTRADAY")
    assert src == "flat"


def test_api_failure_falls_back_to_model_qty():
    # Both broker calls fail → sell the full MODEL qty in the policy product
    # (old behaviour; in-call over-sell guard backstops).
    svc = _Svc(raise_holdings=True, raise_positions=True)
    prod, qty, src = FE.resolve_sell_product_qty(svc, 1, "NSE:SPARC-EQ", 148, "CNC")
    assert (prod, qty, src) == ("CNC", 148, "fallback")


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
