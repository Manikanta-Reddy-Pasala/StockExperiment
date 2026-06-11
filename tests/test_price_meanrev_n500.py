"""price_meanrev_n500 (2026-06-11): package wiring + strategy invariants.

The model is PAPER-ONLY by design: its edge needs LIMIT fills at the dip level
(close-fill execution drops 2025-03->now CAGR from 102.8% to 36.1%), so it must
never be wired into the market-order executor. These tests guard:
  1. strategy params = the swept winner (k1.0 / SMA50 exit / 1.5ATR / 40d / cd10 / K3)
  2. indicator math (entry level = SMA50 - k*ATR, PIT shapes)
  3. registration: MODEL_PATHS + MODEL_INDEX + KNOWN_MODELS(signals_only=True)
  4. cron registers an emit job and does NOT register any execute job
"""
from pathlib import Path

import numpy as np
import pandas as pd

import src.web.admin_routes as ar

ROOT_DIR = Path(__file__).resolve().parents[1]
from src.services.trading.model_ledger_service import KNOWN_MODELS
from tools.shared.model_universe import MODEL_INDEX
from tools.models.price_meanrev_n500 import strategy as S
from tools.models.price_meanrev_n500 import cron as C

MODEL = "price_meanrev_n500"


# ---- 1. params are the swept winner ----------------------------------------
def test_params_match_swept_winner():
    assert S.K == 3
    assert S.SMA_LEN == 50 and S.ATR_LEN == 14
    assert S.ENTRY_ATR_K == 1.0 and S.STOP_ATR == 1.5
    assert S.MAXHOLD == 40 and S.COOLDOWN == 10 and S.LOOKBACK == 60


# ---- 2. indicator math ------------------------------------------------------
def _toy_panels(n=80):
    idx = pd.bdate_range("2025-01-01", periods=n)
    base = pd.Series(np.linspace(100, 120, n), index=idx)
    cl = pd.DataFrame({"NSE:AAA-EQ": base, "NSE:BBB-EQ": base * 2})
    hi = cl * 1.01
    lo = cl * 0.99
    return cl, hi, lo


def test_entry_level_is_sma_minus_k_atr():
    cl, hi, lo = _toy_panels()
    atr14, sma50, mom60, lvl = S.indicators(cl, hi, lo)
    assert lvl.shape == cl.shape
    sub = (sma50 - S.ENTRY_ATR_K * atr14).dropna()
    pd.testing.assert_frame_equal(lvl.dropna(), sub)
    # warmup respected: no level before max(SMA_LEN, ATR_LEN) bars
    assert lvl.iloc[: S.ATR_LEN - 1].isna().all().all()
    assert lvl.iloc[: S.SMA_LEN - 1].isna().all().all()


def test_stop_price_frozen_at_entry():
    assert S.stop_price(100.0, 2.0) == 100.0 - S.STOP_ATR * 2.0


def test_rank_candidates_orders_by_momentum_desc_and_drops_nan():
    row = pd.Series({"A": 0.10, "B": 0.50, "C": np.nan, "D": 0.30})
    assert S.rank_candidates(["A", "B", "C", "D"], row) == ["B", "D", "A"]


# ---- 3. registration ----------------------------------------------------------
def test_model_paths_entry_wired():
    p = ar.MODEL_PATHS[MODEL]
    assert p["live_signal"] == "tools/models/price_meanrev_n500/live_signal.py"
    assert MODEL in p["signals_dir"] and MODEL in p["ranking_dir"]


def test_model_index_gates_on_n500():
    assert MODEL_INDEX[MODEL] == "n500"


def test_known_models_seeds_paper_only_zero_capital():
    row = next(m for m in KNOWN_MODELS if m["name"] == MODEL)
    assert row.get("signals_only") is True   # PAPER — must never default to live
    assert row["default_capital"] == 0       # no real allocation
    assert row.get("enabled", True) is True  # enabled so live_signal emits


# ---- 4. cron: emit + LIMIT executor jobs (never the market-fill executor) ---
class _FakeJob:
    def __init__(self, rec): self._rec = rec
    def at(self, t): self._rec["at"] = t; return self
    def do(self, fn): self._rec["fn"] = fn; return self


class _FakeDay:
    def __init__(self, rec): self.rec = rec
    @property
    def day(self): return _FakeJob(self.rec)
    @property
    def minutes(self): return _FakeJob(self.rec)


class _FakeSchedule:
    def __init__(self): self.jobs = []
    def every(self, n=None):
        rec = {"every": n}; self.jobs.append(rec); return _FakeDay(rec)


def test_cron_registers_limit_executor_jobs():
    sch = _FakeSchedule()
    C.register_trading_jobs(sch)
    C.register_data_jobs(sch)
    names = {j["fn"].__name__ for j in sch.jobs}
    assert names == {"emit_signal", "place_orders", "exit_checks", "reconcile_fills"}
    by_name = {j["fn"].__name__: j for j in sch.jobs}
    assert by_name["emit_signal"]["at"] == "08:55"
    assert by_name["place_orders"]["at"] == "09:16"      # post-open resting limits
    assert by_name["reconcile_fills"]["at"] == "15:20"
    assert by_name["exit_checks"]["every"] == 5          # 5-min intraday polling


def test_cron_never_routes_through_market_fill_executor():
    """The edge needs LIMIT fills (close-fill = 36% vs 103% CAGR) — this model
    must NEVER execute via fyers_executor_multi's fill-now-at-LTP path."""
    src = (ROOT_DIR / "tools/models/price_meanrev_n500/cron.py").read_text()
    assert "fyers_executor_multi" not in src.replace(
        "must NEVER route through fyers_executor_multi", "")
    assert "fyers_executor_limit" in src


# ---- 5. limit executor pure logic --------------------------------------------
from tools.live.fyers_executor_limit import decide_exit, trading_days_between


def test_decide_exit_stop_before_target_on_both_hit():
    # ltp at/below stop wins even if it also clears target ordering edge cases
    assert decide_exit(94.0, 95.0, 105.0, 0, 40) == "STOP"
    assert decide_exit(106.0, 95.0, 105.0, 0, 40) == "TARGET"
    assert decide_exit(100.0, 95.0, 105.0, 40, 40) == "TIME"
    assert decide_exit(100.0, 95.0, 105.0, 39, 40) is None
    assert decide_exit(None, 95.0, 105.0, 50, 40) is None   # no LTP -> hold


def test_trading_days_between_counts_only_trading_days():
    from datetime import date
    always = lambda dt: True
    weekdays = lambda dt: dt.weekday() < 5
    assert trading_days_between(date(2026, 6, 1), date(2026, 6, 5), always) == 4
    # Mon 2026-06-08 -> Mon 2026-06-15 spans one weekend: 5 weekdays
    assert trading_days_between(date(2026, 6, 8), date(2026, 6, 15), weekdays) == 5
