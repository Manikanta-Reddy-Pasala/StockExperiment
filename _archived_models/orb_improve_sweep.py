"""orb_momentum_intraday improvement sweep — one lever at a time.

Research only (NOT wired). Monkeypatches the strategy constants and re-runs the
REAL orb backtest (tools.models.orb_momentum_intraday.backtest.run) so every
number uses the exact production sim. Goal: lift CAGR WITHOUT raising Max DD.

⚠️ HARD CAVEAT: ORB has only ~14 months of cached 5-min data (ONE bull regime,
2025-03→2026-05). Every result is single-regime — anything that "wins" here is
overfit-prone and CANNOT be regime-validated (no intraday bear in the data).
Treat as exploratory only.

Run:  python3 tools/analysis/orb_improve_sweep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.models.orb_momentum_intraday.strategy as S
from tools.models.orb_momentum_intraday import backtest as BT


def _run(label, **overrides):
    saved = {k: getattr(S, k) for k in overrides}
    for k, v in overrides.items():
        setattr(S, k, v)
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = BT.run(BT.DEFAULT_START, BT.DEFAULT_END, BT.DEFAULT_CAP)
        print(f"{label:<26}{r['cagr_pct']:>9.0f}{r['max_dd_pct']:>8.1f}{r['calmar']:>8.2f}"
              f"{r['trades']:>8}{r['win_rate_pct']:>7.0f}")
    except Exception as e:
        print(f"{label:<26}  ERROR {e}")
    finally:
        for k, v in saved.items():
            setattr(S, k, v)


def main():
    print("⚠️ single-regime (one bull, ~14mo) — overfit-prone, NOT regime-validated\n")
    print(f"{'config':<26}{'CAGR%':>9}{'DD%':>8}{'Calmar':>8}{'trades':>8}{'WR%':>7}")
    print("-" * 66)
    _run(f"baseline (cur)")
    print("-- OR_BARS (opening-range len) --")
    for v in (2, 4, 6):
        _run(f"OR_BARS={v}", OR_BARS=v)
    print("-- ENTRY_CUTOFF (min) --")
    for v in (570, 585, 630, 660):   # 9:30 / 9:45 / 10:30 / 11:00
        _run(f"CUTOFF={v}", ENTRY_CUTOFF_MIN=v)
    print("-- TARGET_MULT --")
    for v in (1.5, 2.5, 3.0, 5.0, 999):   # 999 = effectively EOD-only (no target)
        _run(f"TARGET={v}", TARGET_MULT=v)
    print("-- LOOKBACK (momentum win) --")
    for v in (10, 15, 30, 40):
        _run(f"LOOKBACK={v}", LOOKBACK=v)
    print("-- SELECT_TOP (basket) --")
    for v in (2, 4, 5):
        _run(f"SELECT_TOP={v}", SELECT_TOP=v)
    print("-- EOD_FLAT (square-off min) --")
    for v in (840, 870, 925):   # 14:00 / 14:30 / 15:25
        _run(f"EOD_FLAT={v}", EOD_FLAT_MIN=v)


if __name__ == "__main__":
    main()
