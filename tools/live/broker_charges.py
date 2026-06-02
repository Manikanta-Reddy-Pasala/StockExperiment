"""Indian equity broker charge calculator (Fyers schedule).

Computes APPROX per-trade charges (brokerage + STT + exchange txn + SEBI + IPFT
+ stamp + GST + DP) for NSE equity, separately for INTRADAY (MIS) and DELIVERY
(CNC). Buy vs sell differ (stamp = buy only; DP = delivery sell only; STT side
differs by product).

Rates verified against https://fyers.in/charges-list (2026-06-02):

  ── INTRADAY (NSE equity) ───────────────────────────────────────────────
    Brokerage        : min(₹20, 0.03% × turnover) per executed order
    STT              : 0.025% on SELL value only
    Exchange txn(NSE): 0.0030699% (both sides)
    SEBI turnover    : ₹10 per crore  (0.0001%, both sides)
    NSE IPFT         : ₹0.01 per crore (both sides)
    Stamp duty       : 0.003% on BUY value only
    GST              : 18% on (brokerage + exchange + SEBI + IPFT)
    DP charges       : none (no demat movement intraday)

  ── DELIVERY / CNC (NSE equity) ─────────────────────────────────────────
    Brokerage        : min(₹20, 0.3% × turnover) per executed order
    STT              : 0.1% on BOTH buy and sell value
    Exchange txn(NSE): 0.0030699% (both sides)
    SEBI turnover    : ₹10 per crore  (0.0001%, both sides)
    NSE IPFT         : ₹0.01 per crore (both sides)
    Stamp duty       : 0.015% on BUY value only
    GST              : 18% on (brokerage + exchange + SEBI + IPFT)
    DP charges (SELL): ₹12.5 + 18% GST = ₹14.75 per scrip
                       (₹3.5 CDSL + ₹9 FYERS, +GST)

APPROX, not exact: brokerage is per EXECUTED ORDER, so a single order filled in
N partials is charged N× the per-fill brokerage at the broker — pass per-fill
qty/price for an exact match, or accept a small under-estimate on the aggregate.
Stamp duty also has tiny daily-rounding the broker applies. Rates change when
SEBI/exchanges revise schedules — update the constants here and every downstream
audit row + UI estimate follows.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict

CRORE = Decimal("10000000")  # 1e7

# ── Rate constants (Decimal for precision) ──────────────────────────────────
BROKERAGE_CAP = Decimal("20.00")          # ₹20 per executed order cap (both)
BROKERAGE_MIS_PCT = Decimal("0.0003")     # 0.03% intraday
BROKERAGE_CNC_PCT = Decimal("0.003")      # 0.3%  delivery

STT_CNC_PCT = Decimal("0.001")            # 0.1%   delivery, BOTH sides
STT_MIS_SELL_PCT = Decimal("0.00025")     # 0.025% intraday, SELL only

EXCHANGE_TXN_PCT = Decimal("0.000030699")  # 0.0030699% NSE, both sides
SEBI_PER_CRORE = Decimal("10")            # ₹10 / crore, both sides
IPFT_PER_CRORE = Decimal("0.01")          # ₹0.01 / crore (NSE), both sides

STAMP_BUY_CNC_PCT = Decimal("0.00015")    # 0.015% delivery buy
STAMP_BUY_MIS_PCT = Decimal("0.00003")    # 0.003% intraday buy

GST_PCT = Decimal("0.18")                 # 18%

# DP charges (delivery SELL only): ₹3.5 CDSL + ₹9 FYERS = ₹12.5, +18% GST.
DP_BASE = Decimal("12.5")
DP_CHARGES_INCL_GST = (DP_BASE * (Decimal("1") + GST_PCT))  # ₹14.75


def compute_charges(side: str, qty: int, price: float,
                    product: str = "CNC") -> Dict[str, float]:
    """All charge components + total for a single NSE equity trade (approx).

    side    : "BUY" or "SELL"
    qty     : shares (int)
    price   : average fill price per share
    product : "CNC"/"DELIVERY"/"MARGIN" (delivery) or "INTRADAY"/"MIS" (intraday)

    Returns a dict of rupee components + 'total', cast to float for JSON.
    """
    if not qty or not price or qty < 0 or price <= 0:
        return _zero_breakdown()

    prod_u = (product or "").upper()
    is_intraday = prod_u in ("INTRADAY", "MIS")
    is_cnc = prod_u in ("CNC", "DELIVERY", "MARGIN")
    if not (is_intraday or is_cnc):
        return _zero_breakdown(note=f"unsupported product: {product}")

    side_u = (side or "").upper()
    turnover = Decimal(str(qty)) * Decimal(str(price))

    # Brokerage — min(₹20, pct × turnover) per executed order. Delivery 0.3%,
    # intraday 0.03%. Small orders pay the percentage; large orders cap at ₹20.
    brk_pct = BROKERAGE_MIS_PCT if is_intraday else BROKERAGE_CNC_PCT
    brokerage = min(BROKERAGE_CAP, turnover * brk_pct)

    # STT — delivery 0.1% on BOTH sides; intraday 0.025% on SELL only.
    if is_cnc:
        stt = turnover * STT_CNC_PCT
    else:  # intraday
        stt = turnover * STT_MIS_SELL_PCT if side_u == "SELL" else Decimal("0")

    # Exchange transaction charges (both sides, same rate CNC/MIS).
    exchange = turnover * EXCHANGE_TXN_PCT

    # SEBI turnover fee + NSE IPFT (both per-crore, both sides).
    sebi = turnover / CRORE * SEBI_PER_CRORE
    ipft = turnover / CRORE * IPFT_PER_CRORE

    # Stamp duty — BUY only. Delivery 0.015%, intraday 0.003%.
    if side_u == "BUY":
        stamp_rate = STAMP_BUY_MIS_PCT if is_intraday else STAMP_BUY_CNC_PCT
        stamp = turnover * stamp_rate
    else:
        stamp = Decimal("0")

    # GST 18% on (brokerage + exchange + SEBI + IPFT).
    gst = (brokerage + exchange + sebi + ipft) * GST_PCT

    # DP charges — delivery SELL only (demat debit); none for intraday.
    dp = DP_CHARGES_INCL_GST if (side_u == "SELL" and is_cnc) else Decimal("0")

    total = brokerage + stt + exchange + sebi + ipft + stamp + gst + dp

    return {
        "brokerage": _r(brokerage),
        "stt": _r(stt),
        "exchange": _r(exchange),
        "sebi": _r(sebi),
        "ipft": _r(ipft),
        "stamp": _r(stamp),
        "gst": _r(gst),
        "dp": _r(dp),
        "total": _r(total),
        "turnover": _r(turnover),
        "rate_total_pct": _r((total / turnover) * 100) if turnover > 0 else 0.0,
        "side": side_u,
        "product": prod_u,
    }


def estimate_roundtrip(qty: int, buy_price: float, sell_price: float,
                       product: str = "CNC") -> Dict[str, float]:
    """Approx TOTAL charges for a full buy+sell round trip, + net P&L after
    charges. Handy for a 'what will this trade cost me' UI estimate."""
    b = compute_charges("BUY", qty, buy_price, product)
    s = compute_charges("SELL", qty, sell_price, product)
    charges = b["total"] + s["total"]
    gross = (float(sell_price) - float(buy_price)) * int(qty)
    return {
        "buy_charges": b["total"],
        "sell_charges": s["total"],
        "total_charges": round(charges, 4),
        "gross_pnl": round(gross, 4),
        "net_pnl": round(gross - charges, 4),
        "breakeven_move_pct": (round(charges / (float(buy_price) * int(qty)) * 100, 4)
                               if buy_price and qty else 0.0),
        "product": (product or "").upper(),
    }


def _r(d: Decimal) -> float:
    return float(round(d, 4))


def _zero_breakdown(note: str = "") -> Dict[str, float]:
    out = {k: 0.0 for k in ["brokerage", "stt", "exchange", "sebi", "ipft",
                            "stamp", "gst", "dp", "total", "turnover",
                            "rate_total_pct"]}
    out["side"] = ""
    out["product"] = ""
    if note:
        out["note"] = note
    return out


if __name__ == "__main__":
    import json
    for prod in ("CNC", "INTRADAY"):
        print(f"\n===== {prod} — 100 @ ₹200 (₹20,000 turnover) =====")
        print("BUY :", json.dumps(compute_charges("BUY", 100, 200.0, prod)))
        print("SELL:", json.dumps(compute_charges("SELL", 100, 200.0, prod)))
        print("ROUNDTRIP:", json.dumps(estimate_roundtrip(100, 200.0, 204.0, prod)))
