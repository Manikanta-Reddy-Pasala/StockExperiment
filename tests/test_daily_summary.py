"""Daily heartbeat summary composer — one consolidated ping for all models.

Covers the message shape the gated models can't send themselves: on a
non-rebalance day they notify_skip(telegram=False), so this single summary is
the user's daily confirmation that every model evaluated + what it holds.
"""
from tools.daily_summary import compose_summary, _model_display_order, MODEL_ORDER


def test_flat_and_held_models_render():
    models = [
        {"model": "momentum_n100_top5_max1",
         "holdings": [("NSE:ENRIN-EQ", 11)],
         "action": "evaluated — no rebalance today", "today_pnl": None},
        {"model": "orb_momentum_intraday",
         "holdings": [], "action": "flat — no breakout", "today_pnl": 0.0},
    ]
    title, body = compose_summary("2026-06-03", models)
    assert "2026-06-03" in title
    assert "n100" in body and "ENRIN×11" in body
    assert "no rebalance today" in body
    assert "orb" in body and "flat" in body


def test_today_pnl_signs():
    title, body = compose_summary("2026-06-03", [
        {"model": "orb_momentum_intraday", "holdings": [], "action": "traded",
         "today_pnl": 335.0},
        {"model": "n20_daily_large_only", "holdings": [], "action": "hold",
         "today_pnl": -42.0},
    ])
    assert "+₹335" in body
    assert "-₹42" in body


def test_no_pnl_omits_paren():
    _, body = compose_summary("2026-06-03", [
        {"model": "emerging_momentum", "holdings": [], "action": "hold",
         "today_pnl": None},
    ])
    assert "today" not in body  # no "(today …)" clause when pnl unknown


def test_retired_models_excluded_from_heartbeat():
    """Retired models keep model_ledger rows when they have trade history
    (ensure_models_seeded leaves them for forensics), so the heartbeat must
    filter on the canonical RETIRED_MODELS set — one source of truth in
    model_ledger_service — or orb/midcap reappear in the daily ping."""
    from src.services.trading.model_ledger_service import RETIRED_MODELS
    assert "orb_momentum_intraday" in RETIRED_MODELS
    assert "midcap_narrow_60d_breakout" in RETIRED_MODELS

    ledger_names = list(MODEL_ORDER) + list(RETIRED_MODELS) + ["some_future_model"]
    names = _model_display_order(ledger_names)

    for retired in RETIRED_MODELS:
        assert retired not in names
    # Live models + unknown-but-active extras stay, in canonical order first.
    assert names[: len(MODEL_ORDER)] == list(MODEL_ORDER)
    assert "some_future_model" in names


def test_multi_holding_lists_all_names():
    _, body = compose_summary("2026-06-03", [
        {"model": "momentum_retest_n500",
         "holdings": [("NSE:IDEA-EQ", 1053), ("NSE:BHEL-EQ", 36)],
         "action": "hold", "today_pnl": None},
    ])
    assert "IDEA×1053" in body and "BHEL×36" in body
