"""Targeted ORB checks: is CUTOFF=585 a robust plateau or an overfit spike?
Plus the remaining levers the big sweep didn't reach. Single-regime caveat applies.
Run:  python3 tools/analysis/orb_cutoff_check.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import io, contextlib
from datetime import date
import tools.models.orb_momentum_intraday.strategy as S
from tools.models.orb_momentum_intraday import backtest as BT


def run(label, **ov):
    sv = {k: getattr(S, k) for k in ov}
    for k, v in ov.items():
        setattr(S, k, v)
    try:
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            r = BT.run(date(2025, 3, 1), date(2026, 5, 29))
        print(f"{label:<16} CAGR {r['cagr_pct']:>6.0f}  DD {r['max_dd_pct']:>5.1f}  "
              f"Calmar {r['calmar']:>6.1f}  trades {r['trades']:>4}  WR {r['win_rate_pct']:.0f}", flush=True)
    except Exception as e:
        print(f"{label:<16} ERR {e}", flush=True)
    finally:
        for k, v in sv.items():
            setattr(S, k, v)


print("-- CUTOFF fine (is 585 a plateau?) --")
for c in (575, 580, 585, 590, 595, 600, 615):
    run(f"CUTOFF={c}", ENTRY_CUTOFF_MIN=c)
print("-- TARGET_MULT --")
for v in (2.5, 3.0, 5.0, 999):
    run(f"TARGET={v}", TARGET_MULT=v)
print("-- LOOKBACK --")
for v in (10, 15, 30, 40):
    run(f"LOOKBACK={v}", LOOKBACK=v)
print("-- SELECT_TOP --")
for v in (2, 4, 5):
    run(f"SELECT_TOP={v}", SELECT_TOP=v)
print("-- EOD_FLAT --")
for v in (840, 870, 925):
    run(f"EOD={v}", EOD_FLAT_MIN=v)
print("-- best combo: CUTOFF585 + variants --")
run("C585+T2.5", ENTRY_CUTOFF_MIN=585, TARGET_MULT=2.5)
run("C585+SEL5", ENTRY_CUTOFF_MIN=585, SELECT_TOP=5)
print("SWEEP2_DONE")
