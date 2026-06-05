"""Tests for 2026-05-31 execution/signal hardening fixes.

Covers three confirmed bugs found in the order-execution + signal audit:

  1. `_resolve_fill_price(res, fallback)` (fyers_executor) — the multi-holding
     executor recorded fills off a key (`traded_px`) that `place_limit_with_fallback`
     NEVER returns (real key is `fill_price`), so every multi fill was booked at the
     pre-trade LTP. This pure helper centralises "use the real fill price, fall back
     to the quote only when the broker omits a positive traded price".

  2. `bar_is_stale(last_ts, target_ts)` (ohlcv_cache) — the n100/pseudo/emerging
     live signal read the last cached daily close with NO freshness check, so a
     failed nightly OHLCV pull would silently rank/hold on a days-old price. This
     pure helper is the gate; get_close_at returns 0.0 (drops the name) when stale.

  3. `multi_trading_blocked(enabled, is_trading_day)` (fyers_executor_multi) — the
     multi executor had none of the single executor's pre-flight guards (enabled
     backstop, holiday guard). This pure helper returns the abort reason or None.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --- 1. fill-price resolution -------------------------------------------------

def test_resolve_fill_price_uses_real_fill_when_present():
    from tools.live.fyers_executor import _resolve_fill_price
    res = {"filled": True, "fill_price": 105.5, "fill_qty": 10}
    assert _resolve_fill_price(res, fallback=100.0) == 105.5


def test_resolve_fill_price_falls_back_when_missing():
    from tools.live.fyers_executor import _resolve_fill_price
    assert _resolve_fill_price({"filled": True}, fallback=100.0) == 100.0


def test_resolve_fill_price_falls_back_when_none_or_zero():
    from tools.live.fyers_executor import _resolve_fill_price
    assert _resolve_fill_price({"fill_price": None}, fallback=100.0) == 100.0
    assert _resolve_fill_price({"fill_price": 0}, fallback=100.0) == 100.0
    assert _resolve_fill_price({"fill_price": -3}, fallback=100.0) == 100.0


def test_resolve_fill_price_never_reads_nonexistent_traded_px_key():
    # Regression: the multi executor used res.get("traded_px") which is never
    # a key place_limit_with_fallback returns -> always fell through to ltp.
    from tools.live.fyers_executor import _resolve_fill_price
    res = {"filled": True, "fill_price": 250.0, "traded_px": 999.0}
    # Must trust fill_price (the real key), not the bogus traded_px.
    assert _resolve_fill_price(res, fallback=100.0) == 250.0


# --- 2. bar staleness ---------------------------------------------------------

DAY = 86400


def test_bar_is_stale_fresh_bar_not_stale():
    from tools.shared.ohlcv_cache import bar_is_stale
    target = 1_000_000_000
    assert bar_is_stale(target - 1 * DAY, target) is False


def test_bar_is_stale_across_long_weekend_not_stale():
    from tools.shared.ohlcv_cache import bar_is_stale
    target = 1_000_000_000
    # 4-day gap (e.g. Fri bar, signal runs Tue after a Mon holiday) — OK.
    assert bar_is_stale(target - 4 * DAY, target) is False


def test_bar_is_stale_old_bar_is_stale():
    from tools.shared.ohlcv_cache import bar_is_stale
    target = 1_000_000_000
    # 10-day gap = nightly pull has been broken for over a week.
    assert bar_is_stale(target - 10 * DAY, target) is True


def test_get_close_at_returns_zero_on_stale_bar(monkeypatch):
    """get_close_at must drop a name whose last cached bar is stale."""
    import pandas as pd
    import tools.models.momentum_n100_top5_max1.live_signal as ls

    target = 1_000_000_000
    stale_df = pd.DataFrame({
        "timestamp": [target - 30 * DAY],
        "close": [123.0],
    })
    monkeypatch.setattr(ls, "read_cached", lambda *a, **k: stale_df)
    assert ls.get_close_at("NSE:FOO-EQ", target) == 0.0


def test_get_close_at_returns_close_on_fresh_bar(monkeypatch):
    import pandas as pd
    import tools.models.momentum_n100_top5_max1.live_signal as ls

    target = 1_000_000_000
    fresh_df = pd.DataFrame({
        "timestamp": [target - 1 * DAY],
        "close": [123.0],
    })
    monkeypatch.setattr(ls, "read_cached", lambda *a, **k: fresh_df)
    assert ls.get_close_at("NSE:FOO-EQ", target) == 123.0


# --- 3. multi-executor pre-flight guard --------------------------------------

def test_multi_trading_blocked_when_disabled():
    from tools.live.fyers_executor_multi import multi_trading_blocked
    reason = multi_trading_blocked(enabled=False, is_trading_day=True)
    assert reason and "enabled" in reason.lower()


def test_multi_trading_blocked_on_holiday():
    from tools.live.fyers_executor_multi import multi_trading_blocked
    reason = multi_trading_blocked(enabled=True, is_trading_day=False)
    assert reason and ("holiday" in reason.lower() or "trading day" in reason.lower())


def test_multi_trading_allowed_when_enabled_and_trading_day():
    from tools.live.fyers_executor_multi import multi_trading_blocked
    assert multi_trading_blocked(enabled=True, is_trading_day=True) is None


# --- 4. ledger idempotency (no double-record on retry) ------------------------

def test_order_already_recorded_true_when_id_present():
    from src.services.trading.model_ledger_service import _order_already_recorded
    assert _order_already_recorded("OID123", ["OID123", "OID999"]) is True


def test_order_already_recorded_false_when_new():
    from src.services.trading.model_ledger_service import _order_already_recorded
    assert _order_already_recorded("OIDNEW", ["OID123"]) is False


def test_order_already_recorded_ignores_blank_and_dry():
    from src.services.trading.model_ledger_service import _order_already_recorded
    # Blank / DRY / None order ids are never dedupable (can't identify a fill).
    assert _order_already_recorded("", ["", "x"]) is False
    assert _order_already_recorded("DRY", ["DRY"]) is False
    assert _order_already_recorded(None, ["x"]) is False


# --- 5. emerging panel staleness ---------------------------------------------

def test_panel_is_stale_fresh_not_stale():
    from datetime import date
    from tools.models.emerging_momentum.live_signal import panel_is_stale
    assert panel_is_stale(date(2026, 5, 29), date(2026, 5, 31)) is False


def test_panel_is_stale_old_is_stale():
    from datetime import date
    from tools.models.emerging_momentum.live_signal import panel_is_stale
    assert panel_is_stale(date(2026, 5, 1), date(2026, 5, 31)) is True


def test_panel_is_stale_accepts_pandas_timestamp():
    import pandas as pd
    from tools.models.emerging_momentum.live_signal import panel_is_stale
    assert panel_is_stale(pd.Timestamp("2026-05-29"), pd.Timestamp("2026-05-31")) is False
    assert panel_is_stale(pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-31")) is True


# --- 6. cross-process trading lock decision ----------------------------------

def test_lock_proceed_when_acquired():
    from src.services.trading.trade_lock import lock_proceed_decision
    assert lock_proceed_decision(acquired=True, infra_error=False) is True


def test_lock_abort_when_held_by_other():
    # Lock held by another process (cron/other worker) and no infra error ->
    # do NOT proceed (prevents the double-placement race).
    from src.services.trading.trade_lock import lock_proceed_decision
    assert lock_proceed_decision(acquired=False, infra_error=False) is False


def test_lock_failopen_on_infra_error():
    # Lock infra unreachable -> proceed (fail-open) rather than halt the account;
    # the executor's own duplicate-buy guard remains the backstop.
    from src.services.trading.trade_lock import lock_proceed_decision
    assert lock_proceed_decision(acquired=False, infra_error=True) is True


# --- 7. partial-sell outcome (don't book full position on a partial fill) -----

def test_partial_sell_none_qty_is_full_close():
    from src.services.trading.model_ledger_service import partial_sell_outcome
    assert partial_sell_outcome(100, None) == (100, 0, True)


def test_partial_sell_requested_ge_open_is_full():
    from src.services.trading.model_ledger_service import partial_sell_outcome
    assert partial_sell_outcome(100, 100) == (100, 0, True)
    assert partial_sell_outcome(100, 150) == (100, 0, True)  # never sell more than held


def test_partial_sell_partial_keeps_residual():
    from src.services.trading.model_ledger_service import partial_sell_outcome
    # filled 40 of 100 -> sell 40, keep 60 open, NOT a full close.
    assert partial_sell_outcome(100, 40) == (40, 60, False)


def test_partial_sell_nonpositive_qty_falls_back_to_full():
    from src.services.trading.model_ledger_service import partial_sell_outcome
    # defensive: 0/negative requested -> treat as full close (never strand).
    assert partial_sell_outcome(100, 0) == (100, 0, True)
    assert partial_sell_outcome(100, -5) == (100, 0, True)


# --- 8. mid-month retain parity (live == backtest) ---------------------------

def test_mid_month_retain_is_top1_on_mid_month():
    # Mid-month leg always rotates on top-1, regardless of the model's band.
    from tools.shared.rotation_strategy import mid_month_retain
    assert mid_month_retain(True, 3) == 1
    assert mid_month_retain(True, 1) == 1


def test_mid_month_retain_uses_full_band_on_regular():
    from tools.shared.rotation_strategy import mid_month_retain
    assert mid_month_retain(False, 3) == 3
    assert mid_month_retain(False, 1) == 1


# --- 9. multi executor passes rm_cfg to place_limit_with_fallback ------------

def test_multi_run_orders_passes_rm_cfg(monkeypatch):
    """Regression: the multi executor must pass rm_cfg (7th positional) to
    place_limit_with_fallback — omitting it crashes every live multi order."""
    import tools.live.fyers_executor_multi as m
    import tools.live.risk_manager as RM

    captured = {}

    def fake_place(svc, user_id, symbol, qty, side, last_price, rm_cfg, tag=""):
        captured["rm_cfg"] = rm_cfg
        return {"filled": True, "fill_price": 100.0, "fill_qty": qty, "order_id": "OID"}

    monkeypatch.setattr(m, "place_limit_with_fallback", fake_place)
    # Neutralise the central universe guard: this test uses a placeholder symbol
    # ("X") that is not a real index member, which the guard would (correctly)
    # block. None = "no opinion, allow" so the order path under test runs.
    monkeypatch.setattr(m, "is_in_universe", lambda *a, **k: None)
    monkeypatch.setattr(m, "_fetch_live_ltp", lambda *a, **k: 100.0)
    monkeypatch.setattr(m, "record_buy_multi", lambda *a, **k: {})
    monkeypatch.setattr(m, "record_sell_multi", lambda *a, **k: {})
    monkeypatch.setattr(m.MLS, "get_ledger", lambda model: {"cash": 1000.0})
    monkeypatch.setattr(
        RM.RiskManager, "from_model",
        classmethod(lambda cls, mn: RM.RiskManager(RM.RiskConfig())),
    )

    class _A:
        dry_run = False
        user_id = 1

    rc = m._run_orders(_A(), "momentum_retest_n500", [],
                       [{"symbol": "NSE:X-EQ"}], {}, object())
    assert rc == 0
    assert captured.get("rm_cfg") is not None  # rm_cfg was forwarded


# --- 10. cross-model BUY drift decision (model-scoped, sibling-aware) ---------

def test_buy_drift_skip_when_this_model_already_holds():
    from tools.live.fyers_executor import buy_drift_decision
    skip, _ = buy_drift_decision(fyers_qty=10, mine_qty=10, sibling_qty=0)
    assert skip is True


def test_buy_allow_when_position_belongs_to_sibling_model():
    # Shared account: another model holds the symbol; THIS model is flat on it
    # and must be allowed to enter (the old account-wide guard wrongly skipped).
    from tools.live.fyers_executor import buy_drift_decision
    skip, _ = buy_drift_decision(fyers_qty=10, mine_qty=0, sibling_qty=10)
    assert skip is False


def test_buy_skip_on_unaccounted_drift():
    # Fyers holds more than any model's ledger claims -> genuine drift, skip.
    from tools.live.fyers_executor import buy_drift_decision
    skip, _ = buy_drift_decision(fyers_qty=15, mine_qty=0, sibling_qty=10)
    assert skip is True


def test_buy_allow_when_flat_everywhere():
    from tools.live.fyers_executor import buy_drift_decision
    skip, _ = buy_drift_decision(fyers_qty=0, mine_qty=0, sibling_qty=0)
    assert skip is False


# --- 11. emerging ATR-from-entry hard stop (2.5x) ----------------------------

def test_atr_stop_level_from_entry():
    from tools.models.emerging_momentum.strategy import atr_stop_level
    # entry 1000, ATR 40, 2.5x -> level 900
    assert atr_stop_level(1000.0, 40.0, 2.5) == 900.0


def test_atr_stop_level_none_on_bad_inputs():
    from tools.models.emerging_momentum.strategy import atr_stop_level
    assert atr_stop_level(1000.0, 0, 2.5) is None       # no ATR
    assert atr_stop_level(0, 40.0, 2.5) is None          # no entry
    assert atr_stop_level(50.0, 40.0, 2.5) is None       # level would be <=0


def test_atr_stop_hit_when_low_pierces_level():
    from tools.models.emerging_momentum.strategy import atr_stop_hit
    hit, lvl = atr_stop_hit(1000.0, 40.0, day_low=895.0, mult=2.5)
    assert hit is True and lvl == 900.0


def test_atr_stop_not_hit_above_level():
    from tools.models.emerging_momentum.strategy import atr_stop_hit
    hit, lvl = atr_stop_hit(1000.0, 40.0, day_low=905.0, mult=2.5)
    assert hit is False and lvl == 900.0


def test_atr_latest_simple():
    import pandas as pd
    from tools.models.emerging_momentum.strategy import atr_latest
    n = 20
    high = pd.Series([102.0] * n)
    low = pd.Series([98.0] * n)
    close = pd.Series([100.0] * n)
    a = atr_latest(high, low, close, win=14)
    # constant 4-wide bars, no gaps -> TR=4 -> ATR=4
    assert a is not None and abs(a - 4.0) < 1e-6


# (ORB per-slot sizing tests removed — ORB archived 2026-06-05.)
