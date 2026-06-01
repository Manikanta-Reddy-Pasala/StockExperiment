"""Tests for multi-holding executor robustness helpers (2026-06-01).

These lock the fixes for the ADANIENT silent-drop incident on
momentum_retest_n500: a brand-new BUY had no price to start with, the single
live-LTP fetch missed (transient quote hiccup at the 09:34 open), and the multi
buy loop's `if not ltp: skip` dropped the order with NO order placed and NO
record anywhere (the multi executor wrote no audit_orders).

  * pick_buy_price(live, signal, last) — choose the first POSITIVE price among
    live LTP -> signal-file price -> last-known (held entry / close). So a
    transient live miss falls back instead of silently dropping the buy.

  * model_max_holdings(name) — concurrent-position cap for a model. Multi-
    holding models (momentum_retest_n500 = K4) return their K; single-position
    models return None (caller keeps the env default). Fixes RiskManager
    treating a K=4 model as max_concurrent=1 under MAX_CONCURRENT=1.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------- pick_buy_price (pure) ----------------

def test_pick_buy_price_prefers_live():
    from tools.live.fyers_executor_multi import pick_buy_price
    assert pick_buy_price(101.5, 100.0, 99.0) == 101.5


def test_pick_buy_price_falls_back_to_signal_when_live_missing():
    from tools.live.fyers_executor_multi import pick_buy_price
    # transient live miss (None) -> use signal-file price (the ADANIENT case)
    assert pick_buy_price(None, 2977.0, None) == 2977.0


def test_pick_buy_price_falls_back_to_last_when_live_and_signal_missing():
    from tools.live.fyers_executor_multi import pick_buy_price
    assert pick_buy_price(None, None, 145.46) == 145.46


def test_pick_buy_price_skips_nonpositive():
    from tools.live.fyers_executor_multi import pick_buy_price
    # 0 / negative are not valid prices -> keep falling back
    assert pick_buy_price(0, -5, 50.0) == 50.0


def test_pick_buy_price_none_when_all_missing():
    from tools.live.fyers_executor_multi import pick_buy_price
    assert pick_buy_price(None, None, None) is None
    assert pick_buy_price(0, 0, 0) is None


# ---------------- model_max_holdings (pure-ish registry) ----------------

def test_model_max_holdings_multi_returns_k():
    from src.services.trading.model_ledger_service import model_max_holdings
    assert model_max_holdings("momentum_retest_n500") == 4


def test_model_max_holdings_single_returns_none():
    from src.services.trading.model_ledger_service import model_max_holdings
    assert model_max_holdings("momentum_n100_top5_max1") is None
    assert model_max_holdings("unknown_model") is None
