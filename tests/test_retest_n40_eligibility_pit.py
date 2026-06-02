"""Source-level guards: momentum_retest_n500 and n40 live signals must select
their candidate universe with POINT-IN-TIME index membership (eligible_at), not
the current snapshot CSV.

Mirrors tests/test_orb_eligibility_pit.py — same survivorship bug class
(2026-06-02 ORB bought SPARC, a non-current N500 member). The functional
behaviour of eligible_at is covered in test_universe_guard.py; these guard the
exact regression by inspecting the live source so the filter can't be dropped.
"""
import inspect


def test_retest_live_filters_through_eligible_at():
    from tools.models.momentum_retest_n500 import live_signal as ls
    src = inspect.getsource(ls)
    # Must import + apply the point-in-time N500 filter on the ranked leaders,
    # exactly like backtest.py does per rebalance.
    assert "eligible_at" in src, "retest live must use eligible_at (point-in-time N500)"
    assert 'eligible_at("n500"' in src, "retest live must filter rk through eligible_at(\"n500\", date)"


def test_n40_live_uses_pit_n100_not_current_csv():
    from tools.models.n40 import live_signal as ls
    src = inspect.getsource(ls)
    # The N100 universe gate must come from eligible_at("n100", ...), not the
    # current ind_nifty100list.csv as the primary source.
    assert 'eligible_at("n100"' in src, "n40 live must build its N100 gate from eligible_at(\"n100\", date)"
    # The PIT set assignment must be eligible_at, with nifty100_symbols() only a
    # fallback (guard against silently reverting to the current CSV).
    assert "n100 = set(eligible_at(" in src, "n40 n100 set must be eligible_at, not the current CSV"
