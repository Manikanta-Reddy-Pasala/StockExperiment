"""price_meanrev_n500 (2026-06-11): package wiring + strategy invariants.

The model is PAPER-ONLY by design: its edge needs LIMIT fills at the dip level
(close-fill execution drops 2025-03->now CAGR from 102.8% to 36.1%), so it must
never be wired into the market-order executor. These tests guard:
  1. strategy params = the swept winner (k1.0 / SMA50 exit / 1.5ATR / 40d / cd10 / K3)
  2. indicator math (entry level = SMA50 - k*ATR, PIT shapes)
  3. registration: MODEL_PATHS + MODEL_INDEX + KNOWN_MODELS(signals_only=True)
  4. cron registers an emit job and does NOT register any execute job
"""
import numpy as np
import pandas as pd

import src.web.admin_routes as ar
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


# ---- 4. cron: emit-only, never an execute job -------------------------------
class _FakeJob:
    def __init__(self, rec): self._rec = rec
    def at(self, t): self._rec["at"] = t; return self
    def do(self, fn): self._rec["fn"] = fn; return self


class _FakeSchedule:
    def __init__(self): self.jobs = []
    def every(self):
        rec = {}; self.jobs.append(rec); return _FakeDay(rec)


class _FakeDay:
    def __init__(self, rec): self.rec = rec
    @property
    def day(self): return _FakeJob(self.rec)


def test_cron_registers_emit_only():
    sch = _FakeSchedule()
    C.register_trading_jobs(sch)
    C.register_data_jobs(sch)
    assert len(sch.jobs) == 1                      # exactly one job: the emit
    assert sch.jobs[0]["at"] == "08:55"
    assert sch.jobs[0]["fn"].__name__ == "emit_signal"
    # the module must not even define an execute_orders entry point
    assert not hasattr(C, "execute_orders")
