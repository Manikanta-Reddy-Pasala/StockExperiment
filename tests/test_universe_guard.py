"""Central execution-layer universe guard (tools/shared/model_universe).

Belt-and-suspenders behind every model's signal layer: the order executors call
is_in_universe() and REFUSE a buy whose symbol is not a point-in-time member of
the model's index on the trade date. This guards against a FUTURE regression in
any model re-introducing the 2026-06-02 SPARC survivorship bug.

Deterministic: patches model_universe.eligible_at so the test does not depend on
the live membership CSV contents.
"""
from datetime import date

import tools.shared.model_universe as MU


def _fake_eligible(monkeypatch, mapping):
    """Patch eligible_at to return mapping[index] (a set), {} otherwise."""
    monkeypatch.setattr(MU, "eligible_at", lambda idx, d: set(mapping.get(idx, set())))


def test_blocks_off_universe_buy(monkeypatch):
    # SPARC not in n500 -> retest must report False (executor blocks).
    _fake_eligible(monkeypatch, {"n500": {"HFCL", "IDEA", "WOCKPHARMA"}})
    assert MU.is_in_universe("momentum_retest_n500", "SPARC", date(2026, 6, 2)) is False


def test_allows_member_buy(monkeypatch):
    _fake_eligible(monkeypatch, {"n500": {"HFCL", "IDEA", "WOCKPHARMA"}})
    assert MU.is_in_universe("momentum_retest_n500", "NSE:HFCL-EQ", date(2026, 6, 2)) is True


def test_n40_uses_n100_index(monkeypatch):
    # n40's universe gate is the Nifty-100, not n500.
    _fake_eligible(monkeypatch, {"n100": {"RELIANCE", "TCS"}, "n500": {"SPARC", "RELIANCE", "TCS"}})
    assert MU.is_in_universe("n20_daily_large_only", "SPARC", date(2026, 6, 2)) is False
    assert MU.is_in_universe("n20_daily_large_only", "RELIANCE", date(2026, 6, 2)) is True


def test_file_based_model_not_gated(monkeypatch):
    # File-based models (curated JSON universe) map to None -> no opinion (allow).
    _fake_eligible(monkeypatch, {"n500": set()})
    for m in ("momentum_n100_top5_max1", "momentum_pseudo_n100_adv"):
        assert MU.is_in_universe(m, "ANYTHING", date(2026, 6, 2)) is None


def test_unknown_model_not_gated(monkeypatch):
    _fake_eligible(monkeypatch, {"n500": {"HFCL"}})
    assert MU.is_in_universe("does_not_exist", "SPARC", date(2026, 6, 2)) is None


def test_fails_open_on_empty_membership(monkeypatch):
    # Empty / unreadable membership table -> None (allow), never block a real
    # trade on a data-loading hiccup. Only a positive False ever blocks.
    _fake_eligible(monkeypatch, {})  # eligible_at returns empty set
    assert MU.is_in_universe("momentum_retest_n500", "SPARC", date(2026, 6, 2)) is None


def test_fails_open_on_membership_exception(monkeypatch):
    def _boom(idx, d):
        raise FileNotFoundError("membership csv missing")
    monkeypatch.setattr(MU, "eligible_at", _boom)
    assert MU.is_in_universe("momentum_retest_n500", "SPARC", date(2026, 6, 2)) is None


def test_every_known_model_is_registered():
    # The registry must list every live model so none silently bypasses the gate
    # (a new model with no entry would be treated as unknown -> ungated).
    expected = {
        "momentum_retest_n500", "n20_daily_large_only",
        "emerging_momentum", "momentum_n100_top5_max1", "momentum_pseudo_n100_adv",
    }
    assert expected.issubset(set(MU.MODEL_INDEX)), (
        "a live model is missing from MODEL_INDEX — add it (index name or None)")
