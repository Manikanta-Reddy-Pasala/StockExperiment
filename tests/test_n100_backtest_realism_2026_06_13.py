"""Backtest realism convention (2026-06-13) — momentum_n100_top5_max1.

The backtest previously filled rotation/stop/profit-take legs at the SAME
bar's close with ZERO charges. The realism convention (shared across all 5
live models so numbers stay comparable):

  1. real Fyers CNC charges (tools/live/broker_charges.compute_charges) are
     deducted from cash at EVERY fill;
  2. decisions on bar d's close fill at bar d+1's OPEN (live parity);
  3. stop fills also move to the next open (detection unchanged);
  4. NAV marking stays close-based; decision logic untouched.

No DB / Fyers needed: importing the module is side-effect free (the engine is
lazy); we unit-test the module-level charge/sizing helpers and assert the
load-bearing wiring (opens in the SQL panel, next-open fill plumbing) at
source level, like tests/test_model_paths_restored_2026_06_05.py does.
"""
import re
from pathlib import Path

from tools.live.broker_charges import compute_charges
from tools.models.momentum_n100_top5_max1 import backtest as bt

SRC = Path(bt.__file__).read_text()


# ── Convention flags ─────────────────────────────────────────────────────────

def test_convention_flags_declared():
    assert bt.FILL_AT_NEXT_OPEN is True
    assert bt.CHARGES == "fyers_cnc"


# ── Charges helper: must be the live ledger's engine, CNC, both sides ───────

def test_charge_total_matches_broker_charges_cnc():
    for side in ("BUY", "SELL"):
        expected = compute_charges(side, 100, 200.0, "CNC")["total"]
        assert bt._charge_total(side, 100, 200.0) == float(expected)
        assert bt._charge_total(side, 100, 200.0) > 0  # never a free fill


def test_sell_charges_exceed_buy_for_same_fill():
    # CNC sell adds DP charges (₹14.75) on top of symmetric components.
    assert bt._charge_total("SELL", 50, 500.0) > bt._charge_total("BUY", 50, 500.0)


# ── Buy sizing: qty*px + buy charges must fit in cash, maximally ─────────────

def test_max_affordable_qty_fits_cash_including_charges():
    for cash, px in [(100_000.0, 333.33), (67_530.0, 2_456.0), (1_000_000.0, 99.95)]:
        n = bt._max_affordable_qty(cash, px)
        assert n >= 1
        assert n * px + bt._charge_total("BUY", n, px) <= cash
        # maximal: one more share would not fit
        n1 = n + 1
        assert n1 * px + bt._charge_total("BUY", n1, px) > cash


def test_max_affordable_qty_naive_sizing_would_overdraw():
    # A price that exactly divides cash: the OLD int(cash/px) sizing spends
    # every rupee and the buy charges would overdraw — new helper sizes down.
    cash, px = 10_000.0, 100.0
    assert bt._max_affordable_qty(cash, px) < int(cash // px)


def test_max_affordable_qty_edge_cases():
    assert bt._max_affordable_qty(0.0, 100.0) == 0
    assert bt._max_affordable_qty(50.0, 100.0) == 0   # can't afford 1 share
    assert bt._max_affordable_qty(100_000.0, 0.0) == 0
    assert bt._max_affordable_qty(100_000.0, -5.0) == 0


# ── Source-level wiring guards (no DB to run the walk itself) ────────────────

def test_panel_query_loads_opens():
    assert re.search(r"SELECT symbol,date,open,low,close,volume FROM historical_data", SRC), \
        "daily panel must load the open column for next-open fills"
    assert 'values="open"' in SRC, "open pivot (opn) missing"


def test_next_open_fill_plumbing_present():
    # pending-fill queue executed at the NEXT bar's open via _fill_px
    for needle in ("pend_exit", "pend_pt", "pend_buy", "_fill_px", "_book_sell"):
        assert needle in SRC, f"missing next-open fill plumbing: {needle}"
    # no same-bar fills left: every cash mutation goes through the helpers
    assert "cash += q * lvl" not in SRC, "stop must not fill at the exact level same-day"
    assert "cash += q * sx" not in SRC, "rotation sell must not fill at same-day close"
    assert "cash -= n * bx" not in SRC or "cash -= n * bx + ch" in SRC, \
        "buy must deduct charges with the fill"


def test_charges_flow_into_summary_and_ledger():
    assert "total_charges_inr" in SRC
    assert '"charges": round(ch, 2)' in SRC, "per-trade sell-leg charges missing from ledger"


def test_decision_logic_untouched():
    # The shared decision helpers must still drive selection (logic unchanged).
    for needle in ("decide_rotation(hold, ranked, retain_top_n=retain_top_n)",
                   "_midok(hold, midret_at(di), mid_month_lead_pct)",
                   "_fix_hit(entry, dlow, _STOP)"):
        assert needle in SRC, f"decision logic changed: {needle}"
