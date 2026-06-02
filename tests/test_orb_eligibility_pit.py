"""ORB live leader selection must use POINT-IN-TIME index membership.

2026-06-02 bug: tools/models/orb_momentum_intraday/live_signal._today_leaders
ranked over universe_union(INDEX) — the survivorship-biased UNION of every name
that was EVER in the index (703 names) — instead of eligible_at(INDEX, date)
(the 500 current members). A non-current member (SPARC, in the union but not the
current N500) ranked top-3 and was BOUGHT live, a trade the backtest would never
make and which mismatched the displayed watchlist (HFCL/WOCKPHARMA/IDEA).

backtest.py:83 and admin_routes._orb_today_watchlist both rank over eligible_at.
The live path must agree. rank_momentum filters strictly to the `eligible` set,
so passing eligible_at excludes union-only names.
"""
import pandas as pd
from tools.models.orb_momentum_intraday import strategy as S


def test_rank_momentum_excludes_non_eligible_names():
    # SPARC has the strongest momentum but is NOT eligible -> must be excluded.
    # HFCL/WOCKPHARMA/IDEA are eligible -> they are the top-3.
    idx = pd.date_range("2026-04-01", periods=S.LOOKBACK + 2, freq="D")
    base = {
        "NSE:SPARC-EQ":      [100] * (S.LOOKBACK + 1) + [300],   # +200% (would win)
        "NSE:HFCL-EQ":       [100] * (S.LOOKBACK + 1) + [150],   # +50%
        "NSE:WOCKPHARMA-EQ": [100] * (S.LOOKBACK + 1) + [140],   # +40%
        "NSE:IDEA-EQ":       [100] * (S.LOOKBACK + 1) + [130],   # +30%
        "NSE:DUD-EQ":        [100] * (S.LOOKBACK + 1) + [101],   # +1%
    }
    cl = pd.DataFrame(base, index=idx)
    eligible = {"HFCL", "WOCKPHARMA", "IDEA", "DUD"}             # SPARC NOT eligible
    top = S.rank_momentum(cl, len(cl) - 1, eligible)
    assert "SPARC" not in top, "non-eligible name must never be a live leader"
    assert top[:3] == ["HFCL", "WOCKPHARMA", "IDEA"]


def test_today_leaders_uses_eligible_at_not_union():
    # Guard the exact regression: the source must rank over eligible_at, not the
    # survivorship-biased universe_union, when building the live `elig` set.
    import inspect
    from tools.models.orb_momentum_intraday import live_signal as ls
    src = inspect.getsource(ls._today_leaders)
    assert "eligible_at(" in src, "live leaders must use point-in-time eligible_at"
    # the elig= assignment must not be universe_union (the bug)
    assert "elig = set(universe_union" not in src
