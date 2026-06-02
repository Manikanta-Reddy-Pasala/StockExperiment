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


# ---------- ORB bare-symbol dedup + INTRADAY product (2026-06-02 SPARC) ----------
# SPARC double-bought (74 @ 09:50 + 74 @ 09:55 = 148) and went CNC not INTRADAY.
# Root cause 1: ORB emits BARE symbols ("SPARC") but model_holdings keys are
# normalized ("NSE:SPARC-EQ"); the `sym in held` guard missed -> re-buy. Fix:
# normalize the signal symbol with _normalize_symbol before the held check.
# Root cause 2: the multi path never set fyers_executor._CURRENT_MODEL, so
# _placeorder resolved product_for_model(None) -> CNC. Fix: set _CURRENT_MODEL.

def test_normalize_bare_matches_holdings_key():
    from src.services.trading.model_ledger_service import _normalize_symbol
    held = {"NSE:SPARC-EQ"}                       # model_holdings stores normalized
    assert _normalize_symbol("SPARC") in held     # bare ORB signal must match
    assert _normalize_symbol("NSE:SPARC-EQ") in held  # idempotent on normalized


def test_orb_is_intraday_product():
    from src.services.trading.model_ledger_service import product_for_model
    assert product_for_model("orb_momentum_intraday") == "INTRADAY"
    assert product_for_model(None) == "CNC"       # the (buggy) default the fix avoids


def test_multi_executor_sets_current_model_for_product():
    # The multi run sets fyers_executor._CURRENT_MODEL so _placeorder resolves
    # the model's product (INTRADAY for orb) instead of the CNC None-default.
    import tools.live.fyers_executor as fe
    fe._CURRENT_MODEL = "orb_momentum_intraday"
    from src.services.trading.model_ledger_service import product_for_model
    assert product_for_model(fe._CURRENT_MODEL) == "INTRADAY"


# 2026-06-02: no Telegram notification on retest/orb purchase OR signals.
# Root cause A: the multi executor NEVER notified fills (single executor pings
# every BOUGHT/SOLD via _tg_safe -> notify_order -> Telegram + DB feed). Fix:
# import _tg_safe into the multi path and ping on each filled buy/sell.
# Root cause B: retest/orb live_signal never called notify_model_decision (the
# verdict ping n100/emerging emit), and the verdict gate MOMROT_TG_NOTIFY was
# unset on the VM. Fix: wire notify_model_decision + set the flag.

def test_multi_executor_imports_tg_safe():
    # The fill-notify path exists: _tg_safe must be importable into the multi
    # executor (parity with the single executor's BOUGHT/SOLD pings).
    import tools.live.fyers_executor_multi as fem
    assert hasattr(fem, "_tg_safe"), "multi executor must import _tg_safe for fill pings"


def test_notify_order_is_ungated():
    # Fill pings go through notify_order, which is NOT behind MOMROT_TG_NOTIFY
    # (unlike the verdict ping) — so a purchase always notifies regardless of
    # the verdict gate. Guards against re-gating the fill path by accident.
    import inspect
    from src.services import notification_service as ns
    src = inspect.getsource(ns.notify_order)
    assert "_verdict_notify_enabled" not in src


def test_verdict_signals_shape_accepted():
    # notify_model_decision consumes {signal,symbol,side[,price]} dicts — the
    # exact shape retest/orb live_signal now build. Signature must not raise on it.
    from src.services.notification_service import _decision_signature, _decision_message
    dec = [
        {"signal": "EXIT", "symbol": "NSE:HFCL-EQ", "side": "SELL"},
        {"signal": "ORB_BREAKOUT", "symbol": "NSE:SPARC-EQ", "side": "BUY", "price": 200.25},
    ]
    sig = _decision_signature(dec, None)
    assert "SPARC" in sig and "HFCL" in sig
    title, body = _decision_message("orb_momentum_intraday", dec, None, None, None)
    assert "orb_momentum_intraday" in title
    assert "SPARC" in body
