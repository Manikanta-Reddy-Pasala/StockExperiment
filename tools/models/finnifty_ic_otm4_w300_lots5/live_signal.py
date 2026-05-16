"""finnifty_ic_otm4_w300_lots5 — live signal for monthly Iron Condor (4% OTM, 300pt wings, 5 lots)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Reuse all helpers from sister model; only the strategy params differ.
from tools.models.finnifty_ic_otm3_w500_lots4 import live_signal as base
from tools.models.finnifty_ic_otm3_w500_lots4.live_signal import main as _main

# Override params for this variant
base.MODEL_NAME = "finnifty_ic_otm4_w300_lots5"
base.OTM_PCT = 4.0
base.WING_WIDTH = 300
base.LOTS = 5

if __name__ == "__main__":
    raise SystemExit(_main())
