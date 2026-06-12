"""Lock the trade-history net-P&L charges accounting (2026-06-13 fix).

Bugs it fixes (admin_routes.model_trade_history_full / _trade_history_rollup):
  1. Per-sell net_pnl used to subtract the sell row's charges AGAIN even though
     the stored pnl is already net of them (record_sell: pnl =
     (proceeds - sell_charges) - qty*entry_px), and approximated the BUY leg's
     charges at the SELL price instead of using the buy row's STORED
     model_trades.charges_inr. New rule: net_pnl = pnl - buy_leg_stored_charges
     (FIFO-matched per symbol; approximation only when the buy row is missing).
  2. Summary used net_pnl = total_pnl(ALL sells, incl. paper) -
     total_charges(real trades only) — inconsistent numerator/denominator plus
     the same sell-charge double-count. New rule: net_pnl =
     total_pnl_real(real sells) - buy_charges_real(their matched buy legs);
     total_pnl/wins/losses still cover all rows (performance stats).

No DB / Fyers needed — feeds synthetic _trade_dict-shaped rows straight into
the pure roll-up helper.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web.admin_routes import _trade_history_rollup
from tools.live.broker_charges import compute_charges

PRODUCT = "CNC"


def _row(rid, side, symbol, qty, price, charges, pnl=None, value=None,
         fyers_order_id=None, reason="x", trade_at=None):
    """Synthetic row in the exact shape model_ledger_service._trade_dict emits."""
    return {
        "id": rid,
        "model_name": "emerging_momentum",
        "side": side,
        "symbol": symbol,
        "qty": qty,
        "price": float(price),
        "value": float(value if value is not None else qty * price),
        "pnl": float(pnl) if pnl is not None else None,
        "charges": round(float(charges), 2),
        "reason": reason,
        "fyers_order_id": fyers_order_id,
        "trade_at": trade_at or f"2026-06-{rid:02d}T10:00:00",
    }


def test_net_pnl_uses_buy_leg_stored_charges_no_sell_double_count():
    """Round trip with STORED charges on both legs.

    BUY 10 @100, stored buy charges 5.0.
    SELL 10 @110, stored sell charges 7.0; stored pnl is net of the SELL
    charges only (as record_sell computes it): 10*(110-100) - 7 = 93.
    Correct net take-home = 93 - 5 (buy leg) = 88. The old code returned
    93 - 7 - buy_chg@sell_price (sell charge subtracted twice + wrong buy
    approximation).
    """
    buy = _row(1, "BUY", "HFCL", 10, 100.0, charges=5.0, fyers_order_id="F1")
    sell = _row(2, "SELL", "HFCL", 10, 110.0, charges=7.0, pnl=93.0,
                value=10 * 110.0 - 7.0, fyers_order_id="F2")
    trades = [sell, buy]  # newest-first, like get_trades

    summary = _trade_history_rollup(trades, PRODUCT)

    assert sell["net_pnl"] == pytest.approx(93.0 - 5.0)          # = 88.0
    assert summary["total_pnl"] == pytest.approx(93.0)
    assert summary["total_pnl_real"] == pytest.approx(93.0)
    assert summary["total_charges"] == pytest.approx(5.0 + 7.0)  # both legs, real
    assert summary["buy_charges_real"] == pytest.approx(5.0)
    assert summary["net_pnl"] == pytest.approx(88.0)
    # Consistency identity: net = real pnl - real buy-leg charges
    assert summary["net_pnl"] == pytest.approx(
        summary["total_pnl_real"] - summary["buy_charges_real"])
    assert summary["wins"] == 1 and summary["losses"] == 0


def test_signals_only_rows_excluded_from_net_but_counted_in_total_pnl():
    """Paper round-trip (no fyers_order_id) shows in total_pnl/wins but must
    not leak into net_pnl or total_charges (same predicate both sides)."""
    real_buy = _row(1, "BUY", "HFCL", 10, 100.0, charges=5.0, fyers_order_id="F1")
    real_sell = _row(2, "SELL", "HFCL", 10, 110.0, charges=7.0, pnl=93.0,
                     fyers_order_id="F2")
    paper_buy = _row(3, "BUY", "IDEA", 100, 10.0, charges=1.0)
    paper_sell = _row(4, "SELL", "IDEA", 100, 11.0, charges=1.5, pnl=98.5)

    summary = _trade_history_rollup(
        [paper_sell, paper_buy, real_sell, real_buy], PRODUCT)

    # Paper sell still gets an informational net_pnl from ITS buy leg
    assert paper_sell["net_pnl"] == pytest.approx(98.5 - 1.0)
    # ...but totals stay real-only and internally consistent
    assert summary["total_pnl"] == pytest.approx(93.0 + 98.5)    # all rows
    assert summary["total_pnl_real"] == pytest.approx(93.0)      # real only
    assert summary["total_charges"] == pytest.approx(12.0)       # real legs only
    assert summary["buy_charges_real"] == pytest.approx(5.0)     # real buy leg only
    assert summary["net_pnl"] == pytest.approx(88.0)
    assert summary["wins"] == 2                                  # perf stats: all rows


def test_fallback_approximation_only_when_buy_row_missing():
    """SELL with no visible BUY (truncated/legacy history) falls back to
    approximating the buy charges at the sell price — the old behavior,
    now reserved for this case only."""
    sell = _row(2, "SELL", "HFCL", 10, 110.0, charges=7.0, pnl=93.0,
                fyers_order_id="F2")
    summary = _trade_history_rollup([sell], PRODUCT)

    approx_buy = compute_charges("BUY", 10, 110.0, PRODUCT)["total"]
    assert approx_buy > 0
    assert sell["net_pnl"] == pytest.approx(round(93.0 - approx_buy, 2))
    assert summary["net_pnl"] == pytest.approx(
        round(summary["total_pnl_real"] - summary["buy_charges_real"], 2))


def test_scale_in_fifo_prorata_buy_charge_attribution():
    """Two buy lots, then a partial and a final sell: buy charges attribute
    FIFO pro-rata, and across the full position everything is consumed."""
    buy1 = _row(1, "BUY", "ADANIPOWER", 5, 100.0, charges=3.0, fyers_order_id="F1")
    buy2 = _row(2, "BUY", "ADANIPOWER", 5, 102.0, charges=4.0, fyers_order_id="F2")
    # Partial sell of 5 -> consumes lot 1 entirely -> buy_chg 3.0
    sell1 = _row(3, "SELL", "ADANIPOWER", 5, 110.0, charges=2.0, pnl=48.0,
                 fyers_order_id="F3")
    # Final sell of 5 -> consumes lot 2 -> buy_chg 4.0
    sell2 = _row(4, "SELL", "ADANIPOWER", 5, 111.0, charges=2.1, pnl=42.9,
                 fyers_order_id="F4")

    summary = _trade_history_rollup([sell2, sell1, buy2, buy1], PRODUCT)

    assert sell1["net_pnl"] == pytest.approx(48.0 - 3.0)
    assert sell2["net_pnl"] == pytest.approx(42.9 - 4.0)
    assert summary["buy_charges_real"] == pytest.approx(7.0)     # all buy charges used
    assert summary["net_pnl"] == pytest.approx((48.0 + 42.9) - 7.0)
    assert summary["total_charges"] == pytest.approx(3.0 + 4.0 + 2.0 + 2.1)


def test_summary_schema_keys_preserved():
    """UI depends on these keys — guard the response schema."""
    summary = _trade_history_rollup([], PRODUCT)
    for k in ("total_buys", "total_sells", "total_deposits", "total_withdrawals",
              "total_buy_value", "total_sell_value", "total_pnl", "total_charges",
              "net_pnl", "wins", "losses", "win_rate_pct",
              "total_pnl_real", "buy_charges_real"):
        assert k in summary, f"missing summary key {k}"
    assert summary["net_pnl"] == 0.0 and summary["win_rate_pct"] == 0.0
