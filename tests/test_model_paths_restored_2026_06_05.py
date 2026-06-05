"""Regression: the ORB-archive commit (ac7c6e6e) accidentally deleted the
MODEL_PATHS dict in admin_routes, leaving 9 dangling references that raised
NameError -> every Today's Picks "Re-calculate", triggers/status and rebalance
call 500'd. Guard the dict's presence + shape, and that retired models stay
absent.
"""
import re
from pathlib import Path

import src.web.admin_routes as ar
from src.services.trading.model_ledger_service import RETIRED_MODELS

ACTIVE_6 = {
    "momentum_n100_top5_max1",
    "momentum_pseudo_n100_adv",
    "midcap_narrow_60d_breakout",
    "n20_daily_large_only",
    "emerging_momentum",
    "momentum_retest_n500",
}


def test_model_paths_defined_with_six_active_models():
    assert isinstance(ar.MODEL_PATHS, dict)
    assert set(ar.MODEL_PATHS) == ACTIVE_6


def test_retired_models_absent_from_model_paths():
    for name in RETIRED_MODELS:
        assert name not in ar.MODEL_PATHS


def test_every_model_path_entry_has_required_keys():
    for name, p in ar.MODEL_PATHS.items():
        for k in ("signals_dir", "ranking_dir", "live_signal", "extra_args"):
            assert k in p, f"{name} missing {k}"


def test_picks_page_models_all_wired_in_model_paths():
    """Every model card the Today's Picks page renders must resolve in
    MODEL_PATHS, else its ranking/recalc fetch 400s/500s."""
    html = Path("src/web/templates/v2/picks.html").read_text()
    keys = set(re.findall(r'key:\s*"([^"]+)"', html))
    assert keys, "no MODELS keys parsed from picks.html"
    assert keys <= set(ar.MODEL_PATHS), keys - set(ar.MODEL_PATHS)
