"""Funds-card P&L: symbol present in BOTH settled holdings AND today's CNC
positions must contribute BOTH lots to account totals — exactly once each.

Bug fixed 2026-06-13 (src/web/momrot_routes.py _portfolio_state): the CNC
positions loop did `if sym in seen_syms: continue`, silently dropping the
today-bought lot's P&L/cost/value whenever the same symbol also had a settled
holdings lot (real case: HFCL 203 settled + 502 bought today = 705 shares).

Fyers overlap model (what the broker actually returns):
  - holdings()            → SETTLED shares only (quantity + average_price)
  - positions() CNC netQty>0 → shares bought TODAY (netQty + buyAvg)
  These are DISJOINT lots; summing both counts each share exactly once.

Day MTM convention asserted here:
  - settled lot:      qty × (LTP − prev_close)
  - today-bought lot: qty × (LTP − buyAvg)   (not owned at prev_close)

All Fyers/DB/admin touchpoints are monkeypatched — no network, no DB.
Fyers field values are STRINGS (matches the real API) to pin the coercion.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

LTP_HFCL, PC_HFCL = 115.0, 108.0
LTP_TCS, PC_TCS = 50.0, 49.0


class _FakeApi:
    def _make_request(self, method, endpoint):
        assert endpoint == "positions"
        return {
            "data": [
                {   # today's CNC buy of a symbol ALSO held settled (the overlap)
                    "symbol": "NSE:HFCL-EQ",
                    "productType": "CNC",
                    "netQty": "502",
                    "buyAvg": "112.0",
                    "netAvg": "112.0",
                    "ltp": "114.0",          # stale; fresh quote overrides
                },
                {   # non-CNC noise — must be ignored
                    "symbol": "NSE:HFCL-EQ",
                    "productType": "INTRADAY",
                    "netQty": "100",
                    "buyAvg": "1.0",
                    "ltp": "1.0",
                },
            ]
        }


class _FakeFyersService:
    def funds(self, uid):
        return {"data": {"available_cash": "1000.0",
                         "total_margin": "0", "utilized_margin": "0"}}

    def holdings(self, uid):
        # STRING fields, like the real standardized holdings payload
        return {"data": [
            {"symbol": "HFCL", "quantity": "203",
             "average_price": "100.0", "last_price": "109.0"},
            {"symbol": "TCS", "quantity": "10",
             "average_price": "40.0", "last_price": "49.5"},
        ]}

    def quotes_multiple(self, uid, syms):
        return {"data": {
            "NSE:HFCL-EQ": {"ltp": str(LTP_HFCL), "prev_close": str(PC_HFCL)},
            "NSE:TCS-EQ": {"ltp": str(LTP_TCS), "prev_close": str(PC_TCS)},
        }}

    def _get_api_instance(self, uid):
        return _FakeApi()


@pytest.fixture
def state(monkeypatch):
    import src.services.brokers.fyers_service as FY
    import src.web.admin_routes as AR
    from src.web import momrot_routes as MR

    monkeypatch.setattr(FY, "FyersService", _FakeFyersService)
    monkeypatch.setattr(MR, "_live_price", lambda s: 0.0)      # no DB
    monkeypatch.setattr(MR, "_read_history", lambda: [])       # no ledger file
    monkeypatch.setattr(AR, "_fyers_account_txn_charges",
                        lambda uid: {"total": 0.0, "holdings": 0.0, "today": 0.0})
    return MR._portfolio_state()


def test_total_pnl_includes_both_lots_exactly_once(state):
    # holdings lot: (115-100)*203 = 3045 ; today lot: (115-112)*502 = 1506
    # TCS settled:  (50-40)*10    = 100
    assert state["account_total_pnl"] == pytest.approx(3045 + 1506 + 100, abs=0.01)
    # Regression guards: neither the dropped-lot value nor a double count
    assert state["account_total_pnl"] != pytest.approx(3045 + 100, abs=0.01)
    assert state["account_total_pnl"] != pytest.approx(3045 + 2 * 1506 + 100, abs=0.01)


def test_today_pnl_includes_both_lots_with_correct_baselines(state):
    # settled lots vs prev_close: HFCL (115-108)*203 = 1421 ; TCS (50-49)*10 = 10
    # today-bought lot vs buyAvg: HFCL (115-112)*502 = 1506
    assert state["account_today_pnl"] == pytest.approx(1421 + 1506 + 10, abs=0.01)


def test_invested_and_market_value_sum_both_lots(state):
    cost = 100.0 * 203 + 112.0 * 502 + 40.0 * 10
    mv = LTP_HFCL * (203 + 502) + LTP_TCS * 10
    assert state["account_invested"] == pytest.approx(cost, abs=0.01)
    assert state["market_value"] == pytest.approx(mv, abs=0.01)
    # gross account total pass must agree with the per-lot accumulation
    assert state["holdings_value"] == pytest.approx(mv, abs=0.01)
    assert state["account_total"] == pytest.approx(1000.0 + mv, abs=0.01)


def test_display_row_merges_overlap_into_705_shares(state):
    rows = {r["symbol"]: r for r in state["open_positions"]}
    assert set(rows) == {"HFCL", "TCS"}          # one row per symbol (no dupes)
    h = rows["HFCL"]
    assert h["qty"] == 705                       # 203 settled + 502 today
    assert h["source"] == "holding+position"
    assert h["cost"] == pytest.approx(100.0 * 203 + 112.0 * 502, abs=0.01)
    assert h["unrealized_pnl"] == pytest.approx(3045 + 1506, abs=0.01)
    assert h["entry_price"] == pytest.approx(h["cost"] / 705, abs=0.0001)
    assert rows["TCS"]["source"] == "holding"


def test_position_only_symbol_still_counted_vs_buyavg(monkeypatch):
    """No-overlap CNC lot: counted once, day MTM vs buyAvg."""
    import src.services.brokers.fyers_service as FY
    import src.web.admin_routes as AR
    from src.web import momrot_routes as MR

    class _OnlyPosApi:
        def _make_request(self, method, endpoint):
            return {"data": [{"symbol": "NSE:IDEA-EQ", "productType": "CNC",
                              "netQty": "50", "buyAvg": "10.0", "ltp": "11.0"}]}

    class _Svc(_FakeFyersService):
        def holdings(self, uid):
            return {"data": []}

        def quotes_multiple(self, uid, syms):
            return {"data": {"NSE:IDEA-EQ": {"ltp": "12.0", "prev_close": "9.0"}}}

        def _get_api_instance(self, uid):
            return _OnlyPosApi()

    monkeypatch.setattr(FY, "FyersService", _Svc)
    monkeypatch.setattr(MR, "_live_price", lambda s: 0.0)
    monkeypatch.setattr(MR, "_read_history", lambda: [])
    monkeypatch.setattr(AR, "_fyers_account_txn_charges",
                        lambda uid: {"total": 0.0, "holdings": 0.0, "today": 0.0})
    st = MR._portfolio_state()
    assert st["account_total_pnl"] == pytest.approx((12.0 - 10.0) * 50, abs=0.01)
    # bought today → day MTM vs buyAvg, NOT vs prev_close (9.0)
    assert st["account_today_pnl"] == pytest.approx((12.0 - 10.0) * 50, abs=0.01)
    assert st["open_positions"][0]["source"] == "position"
