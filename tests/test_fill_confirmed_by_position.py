"""Lock the cancel/fill-race double-fill guard for BOTH sides (2026-06-01).

place_limit_with_fallback re-checks the account NET position before placing a
retry / MARKET order, so an order that 'cancelled' but actually FILLED doesn't
get duplicated. This was BUY-only (could over-SELL on the sell side); now it is
symmetric:
  BUY  filled  -> net ROSE by >= qty
  SELL filled  -> net DROPPED by >= qty
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.fyers_executor import fill_confirmed_by_position as f


# ---- BUY ----
def test_buy_filled_when_net_rose_by_qty():
    assert f("BUY", base_net=0, cur_net=100, qty=100) is True

def test_buy_not_filled_when_net_unchanged():
    assert f("BUY", base_net=0, cur_net=0, qty=100) is False

def test_buy_filled_when_net_rose_more_than_qty():
    assert f("BUY", base_net=50, cur_net=200, qty=100) is True   # +150 >= 100


# ---- SELL (the new side) ----
def test_sell_filled_when_net_dropped_by_qty():
    # held 203, after sell net is 0 -> dropped 203 >= 203 -> filled
    assert f("SELL", base_net=203, cur_net=0, qty=203) is True

def test_sell_not_filled_when_net_unchanged():
    # sell reported cancelled AND net still 203 -> NOT filled, safe to retry
    assert f("SELL", base_net=203, cur_net=203, qty=203) is False

def test_sell_partial_drop_below_qty_not_confirmed():
    # only 50 of 203 left the book -> not a full fill of qty
    assert f("SELL", base_net=203, cur_net=153, qty=203) is False  # dropped 50 < 203

def test_sell_filled_into_short():
    # net went 0 -> -100 (intraday short sell of 100) -> dropped 100 >= 100
    assert f("SELL", base_net=0, cur_net=-100, qty=100) is True
