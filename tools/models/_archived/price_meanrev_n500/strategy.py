"""SHARED core logic for price_meanrev_n500 — used by BOTH backtest + live.

Single source of truth so the offline backtest and the live signal can never
drift. backtest.py and live_signal.py both import the params and the level /
ranking functions from here.

Strategy (price-level mean-reversion dip-buy, born 2026-06-11 from the
tools/research/price_formula_lab.py sweep — 176 configs x 3 families):
  ENTRY  — for each PIT Nifty-500 name, a resting LIMIT BUY at
               level = SMA50 - ENTRY_ATR_K * ATR14      (both from the PRIOR bar)
           fires when the day's LOW touches the level. When more names trigger
           than free slots, take the K best by 60-day momentum (buy dips in
           STRONG names, never falling knives).
  EXIT   — whichever hits first (levels FROZEN at entry):
               target = SMA50 (as of entry)              limit sell — mean reversion
               stop   = entry - STOP_ATR * ATR14(entry)  stop-loss
               time   = MAXHOLD trading days             market out
  COOLDOWN — a name cannot be re-entered for COOLDOWN trading days after exit
           (kills the revert->exit->re-dip->rebuy churn).

⚠ EXECUTION PARITY (verified 2026-06-11, --fam CFV): the edge LIVES IN THE
LIMIT FILL at the dip level. Filling the same signals at the CLOSE drops the
2025-03->now CAGR from 102.8% to 36.1%. Live execution MUST place limit orders
at the level — market-at-close execution does NOT reproduce the backtest.

Backtest (PIT N500, eligible_at per day, cost 0.15%/side, conservative fills):
  2025-03 -> 2026-06-10:  +102.8% CAGR / 12.2% MaxDD / Calmar 8.46 / 225 trades / WR 71%
  2021-03 -> 2026-06-10:  +38.8% CAGR / 32.8% MaxDD / Calmar 1.18 (same config =
                          full-cycle family winner too; positive every year)
"""
from __future__ import annotations

import pandas as pd

# ---- Strategy parameters (sweep winner: MR k1.0 off0 cd10 topn3, 2026-06-11) ----
K = 3                # concurrent position slots (top-3; topn5=64.8%, topn3=102.8% recent)
SMA_LEN = 50         # mean-reversion anchor (sma50 beat sma20 across the sweep)
ATR_LEN = 14         # ATR window
ENTRY_ATR_K = 1.0    # entry level = SMA50 - 1.0*ATR (deeper k1.5/2.0 = fewer/worse)
STOP_ATR = 1.5       # stop = entry - 1.5*ATR(entry)  (3.0 looser = worse Calmar)
MAXHOLD = 40         # time exit after 40 trading days
COOLDOWN = 10        # trading days a name is banned after ANY exit (cd10 beat cd0/cd5)
LOOKBACK = 60        # momentum ranking window (rank candidates by 60d return)
COST = 0.0015        # per-side cost used by the backtest (parity with other models)


def indicators(cl: pd.DataFrame, hi: pd.DataFrame, lo: pd.DataFrame):
    """Build the indicator panels the strategy needs (all rolling, PIT-safe).

    Args:
        cl: close panel (date x symbol).  hi/lo: high/low panels aligned to cl.
    Returns:
        (atr14, sma50, mom60, entry_level) panels. entry_level = sma50 - k*atr.
    """
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()]).groupby(level=0).max()
    tr = tr.reindex(cl.index)
    atr14 = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    sma50 = cl.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    mom60 = cl / cl.shift(LOOKBACK) - 1
    entry_level = sma50 - ENTRY_ATR_K * atr14
    return atr14, sma50, mom60, entry_level


def stop_price(entry_px: float, atr_at_entry: float) -> float:
    """Stop level frozen at entry: entry - STOP_ATR * ATR."""
    return float(entry_px) - STOP_ATR * float(atr_at_entry)


def rank_candidates(symbols, mom_row) -> list:
    """Order candidate symbols best-to-worst by 60d momentum (prior-bar row).

    Args:
        symbols: iterable of symbols whose limit level could fire.
        mom_row: the PRIOR bar's mom60 row (Series or dict-like).
    Returns:
        Symbols sorted by descending momentum; NaN-momentum names dropped.
    """
    out = []
    for s in symbols:
        m = mom_row.get(s)
        if m is None or pd.isna(m):
            continue
        out.append((float(m), s))
    out.sort(reverse=True)
    return [s for _, s in out]
