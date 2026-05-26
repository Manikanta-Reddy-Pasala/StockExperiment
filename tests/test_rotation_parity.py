"""Parity + unit tests for the shared rotation strategy core.

Guards the fix from 2026-05-26: backtest.py and live_signal.py used to
re-implement the entry/exit rule separately and drifted (backtest top-1 vs
live top-5; n100 stateless). All three rotation models now import
tools.shared.rotation_strategy. These tests:

  1. Pin the rule (decide_rotation / midmonth_lead_ok) with unit cases.
  2. Assert each model's live emit_signals produces SELL/BUY symbols that
     match decide_rotation — so emit_signals can never silently re-diverge
     from the core the backtest also uses.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok


# ---------------- decide_rotation unit cases ----------------

def test_flat_enters_rank1():
    d = decide_rotation(None, ["A", "B", "C"], retain_top_n=1)
    assert d.sell is None and d.buy == "A"


def test_holding_rank1_is_noop():
    d = decide_rotation("A", ["A", "B", "C"], retain_top_n=1)
    assert d.is_noop


def test_top1_rotates_when_held_drops_to_rank2():
    # The core bug: top-1 rotation MUST sell at rank-2.
    d = decide_rotation("B", ["A", "B", "C"], retain_top_n=1)
    assert d.sell == "B" and d.buy == "A"


def test_top5_retention_holds_rank2():
    # Old live behaviour (top-5): rank-2 is kept, no rotation.
    d = decide_rotation("B", ["A", "B", "C"], retain_top_n=5)
    assert d.is_noop


def test_top5_rotates_only_when_out_of_band():
    d = decide_rotation("F", ["A", "B", "C", "D", "E", "F"], retain_top_n=5)
    assert d.sell == "F" and d.buy == "A"


def test_held_dropped_from_universe_sells_and_buys_rank1():
    d = decide_rotation("Z", ["A", "B"], retain_top_n=1)
    assert d.sell == "Z" and d.buy == "A"


def test_empty_ranking_is_noop():
    assert decide_rotation("A", [], retain_top_n=1).is_noop


# ---------------- midmonth_lead_ok unit cases ----------------

def test_midmonth_holding_rank1_blocks():
    assert midmonth_lead_ok("A", [("A", 10.0), ("B", 8.0)], 5.0) is False


def test_midmonth_insufficient_lead_blocks():
    # rank-1 leads held by only 3pp (< 5pp) -> no rotation.
    assert midmonth_lead_ok("B", [("A", 10.0), ("B", 7.0)], 5.0) is False


def test_midmonth_sufficient_lead_allows():
    assert midmonth_lead_ok("B", [("A", 10.0), ("B", 3.0)], 5.0) is True


def test_midmonth_held_dropped_allows():
    assert midmonth_lead_ok("Z", [("A", 10.0), ("B", 3.0)], 5.0) is True


# ---------------- live emit_signals parity with the core ----------------

def _picks(*syms):
    # (symbol, name, ret30d_pct, price) tuples as live rank_universe emits.
    return [(s, s, 10.0 - i, 100.0 + i) for i, s in enumerate(syms)]


@pytest.mark.parametrize("module_path,top_n", [
    ("tools.models.momentum_pseudo_n100_adv.live_signal", 1),
    ("tools.models.n20_daily_large_only.live_signal", 1),
])
def test_pseudo_n20_emit_matches_core(module_path, top_n, monkeypatch):
    import importlib
    mod = importlib.import_module(module_path)
    # neutralise DB price lookup used on the SELL leg
    monkeypatch.setattr(mod, "get_close_at", lambda *a, **k: 123.0, raising=False)

    picks = _picks("A", "B", "C", "D")
    # Held = rank-3 "C" -> top-1 rotation must SELL C, BUY A.
    pos = {"open_symbol": "C", "open_entry_px": 100.0}
    if "n20" in module_path:
        sigs = mod.emit_signals(picks, pos, top_n)
    else:
        sigs = mod.emit_signals(picks, pos, top_n, retain_top_n=top_n)

    core = decide_rotation("C", [p[0] for p in picks], retain_top_n=top_n)
    sells = [s["symbol"] for s in sigs if s["side"] == "SELL"]
    buys = [s["symbol"] for s in sigs if s["side"] == "BUY"]
    assert sells == ([core.sell] if core.sell else [])
    assert buys == ([core.buy] if core.buy else [])
    assert core.sell == "C" and core.buy == "A"


def test_n100_emit_matches_core(monkeypatch):
    import importlib
    mod = importlib.import_module("tools.models.momentum_n100_top5_max1.live_signal")
    monkeypatch.setattr(mod, "get_close_at", lambda *a, **k: 123.0, raising=False)

    picks = _picks("A", "B", "C", "D")
    held = [{"symbol": "C", "entry_price": 100.0}]   # n100 held is a list
    sigs = mod.emit_signals(picks, held, top_n=5, retain_top_n=1)

    core = decide_rotation("C", [p[0] for p in picks], retain_top_n=1)
    sells = [s["symbol"] for s in sigs if s["side"] == "SELL"]
    buys = [s["symbol"] for s in sigs if s["side"] == "BUY"]
    assert sells == [core.sell] and buys == [core.buy]
    assert core.sell == "C" and core.buy == "A"
