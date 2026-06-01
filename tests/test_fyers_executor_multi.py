"""Tests for multi-holding executor robustness helpers (2026-06-01).

Locks fixes from the ADANIENT silent-drop incident on momentum_retest_n500:
the multi buy loop dropped an order with NO order placed and NO record (the
multi executor wrote no audit_orders), and RiskManager treated the K=4 model
as max_concurrent=1.

  * model_max_holdings(name) — concurrent-position cap for a model. Multi-
    holding models (momentum_retest_n500 = K4) return their K; single-position
    models return None (caller keeps the env default). Fixes RiskManager
    treating a K=4 model as max_concurrent=1 under MAX_CONCURRENT=1.

  (Buy pricing is LIVE-LTP-ONLY — no signal-price fallback: a persistent live
  miss is skipped + audited + alerted, never priced off a stale signal.)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------- model_max_holdings (pure-ish registry) ----------------

def test_model_max_holdings_multi_returns_k():
    from src.services.trading.model_ledger_service import model_max_holdings
    assert model_max_holdings("momentum_retest_n500") == 4


def test_model_max_holdings_single_returns_none():
    from src.services.trading.model_ledger_service import model_max_holdings
    assert model_max_holdings("momentum_n100_top5_max1") is None
    assert model_max_holdings("unknown_model") is None


# ---------------- pick_tick (pure) — per-symbol LIMIT tick (Q2) ----------------

def test_pick_tick_uses_valid_db_tick():
    from tools.live.fyers_executor import pick_tick
    assert pick_tick(0.01) == 0.01      # IDEA
    assert pick_tick(0.05) == 0.05      # IDEAFORGE
    assert pick_tick(0.1) == 0.1        # ENRIN / RELIANCE


def test_pick_tick_falls_back_to_010_on_junk():
    from tools.live.fyers_executor import pick_tick
    assert pick_tick(None) == 0.10
    assert pick_tick(0) == 0.10
    assert pick_tick(-1) == 0.10
    assert pick_tick("nan-ish") == 0.10


def test_snap_tick_respects_symbol_tick():
    from tools.live.fyers_executor import _snap_tick
    # IDEA 0.01 tick: keep cents; ENRIN 0.10 tick: snap to dime.
    assert _snap_tick(14.237, 0.01) == 14.24
    assert _snap_tick(3872.13, 0.10) == 3872.10
    assert _snap_tick(3872.16, 0.10) == 3872.20
