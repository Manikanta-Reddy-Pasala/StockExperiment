"""Broker charge calculator — rates verified against fyers.in/charges-list
(2026-06-02). Covers intraday + delivery, buy + sell, brokerage cap, and the
delivery-only DP / both-sides STT rules.
"""
from tools.live.broker_charges import compute_charges, estimate_roundtrip


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


# ── DELIVERY (CNC) ──────────────────────────────────────────────────────────

def test_cnc_buy_100_at_200():
    c = compute_charges("BUY", 100, 200.0, "CNC")        # ₹20,000 turnover
    assert c["brokerage"] == 20.0                         # min(20, 0.3%*20000=60)
    assert approx(c["stt"], 20.0)                         # 0.1% BOTH sides → buy charged
    assert approx(c["stamp"], 3.0)                        # 0.015% buy
    assert c["dp"] == 0.0                                 # no DP on buy
    assert approx(c["exchange"], 0.6140)                  # 0.0030699%
    assert approx(c["total"], 47.35, 0.05)


def test_cnc_sell_100_at_200_has_dp_and_no_stamp():
    c = compute_charges("SELL", 100, 200.0, "CNC")
    assert approx(c["stt"], 20.0)                         # 0.1% sell side
    assert c["stamp"] == 0.0                              # stamp is buy-only
    assert approx(c["dp"], 14.75)                         # ₹12.5 + 18% GST
    assert approx(c["total"], 59.10, 0.05)


def test_cnc_brokerage_caps_at_20_for_large_order():
    # 0.3% of a large delivery order exceeds ₹20 → capped at ₹20.
    c = compute_charges("BUY", 1000, 500.0, "CNC")       # ₹5,00,000
    assert c["brokerage"] == 20.0


def test_cnc_small_order_brokerage_is_percentage():
    # Small delivery order: 0.3% < ₹20 → charged the percentage, not flat ₹20.
    c = compute_charges("BUY", 1, 100.0, "CNC")          # ₹100 turnover
    assert approx(c["brokerage"], 0.30)                  # 0.3% of 100


# ── INTRADAY (MIS) ──────────────────────────────────────────────────────────

def test_intraday_buy_no_stt_lower_brokerage_and_stamp():
    c = compute_charges("BUY", 100, 200.0, "INTRADAY")
    assert approx(c["brokerage"], 6.0)                   # 0.03% * 20000 = 6 (< 20)
    assert c["stt"] == 0.0                               # intraday STT sell-only
    assert approx(c["stamp"], 0.60)                      # 0.003% buy
    assert c["dp"] == 0.0
    assert approx(c["total"], 8.43, 0.05)


def test_intraday_sell_has_stt_sell_side_no_dp():
    c = compute_charges("SELL", 100, 200.0, "INTRADAY")
    assert approx(c["stt"], 5.0)                         # 0.025% * 20000
    assert c["dp"] == 0.0                                # never DP on intraday
    assert approx(c["total"], 12.83, 0.05)


def test_intraday_brokerage_caps_at_20():
    c = compute_charges("SELL", 1000, 500.0, "INTRADAY")  # 0.03% of 5L = 150 → cap 20
    assert c["brokerage"] == 20.0


# ── general ─────────────────────────────────────────────────────────────────

def test_intraday_cheaper_than_delivery_same_trade():
    cnc = compute_charges("SELL", 100, 200.0, "CNC")["total"]
    mis = compute_charges("SELL", 100, 200.0, "INTRADAY")["total"]
    assert mis < cnc


def test_unsupported_product_zero():
    c = compute_charges("BUY", 10, 100.0, "FUTURES")
    assert c["total"] == 0.0 and "note" in c


def test_zero_inputs_safe():
    assert compute_charges("BUY", 0, 100.0)["total"] == 0.0
    assert compute_charges("SELL", 10, 0.0)["total"] == 0.0


def test_roundtrip_net_pnl_after_charges():
    rt = estimate_roundtrip(100, 200.0, 204.0, "INTRADAY")
    assert approx(rt["gross_pnl"], 400.0)
    assert rt["net_pnl"] < rt["gross_pnl"]               # charges reduce P&L
    assert approx(rt["net_pnl"], rt["gross_pnl"] - rt["total_charges"], 0.01)
    assert rt["breakeven_move_pct"] > 0
