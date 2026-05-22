"""finnifty_ic_otm2_w150_lots5 — live signal for monthly Iron Condor.

Promoted 2026-05-22 from prior OTM4/W300 config to the safe-tight variant
(OTM 2%, 150-pt wings, 5 lots). Backtest: +112% CAGR / 17.5% max-loss/trade.
Folder name retained for cron + path stability; runtime identifier is
MODEL_NAME below.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5 import _base_logic as base
from tools.models.finnifty_ic_otm4_w300_lots5._base_logic import main as _main

# Safe-tight FinNifty IC params (promoted 2026-05-22)
base.MODEL_NAME = "finnifty_ic_otm2_w150_lots5"
base.OTM_PCT = 2.0
base.WING_WIDTH = 150
base.LOTS = 5

if __name__ == "__main__":
    raise SystemExit(_main())
