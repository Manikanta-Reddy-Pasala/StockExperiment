"""SHARED core logic for regime_momentum_n500 — used by BOTH backtest + live.

Single source of truth so the offline backtest and the live signal can never
drift. backtest.py and live_signal.py both import the params + the selection
(`rank_targets`), regime (`regime_healthy`) and risk-exit (`hit_stop`,
`hit_trail`) helpers from here.

Strategy ("Regime-Adaptive Momentum"):
  - Universe: top-TOPN by 20-day ADV within Nifty-500, price <= MAX_PRICE.
  - Rank: LOOKBACK-day return, momentum > 0. Hold K equal-weight; a held name is
    kept while it stays in the top-RETAIN rank (winners ride).
  - Buys happen on the 1st trading day of each month (monthly rebalance). There
    is NO take-profit — winners run.
  - REGIME SWITCH on the Nifty-50 index vs its 200-DMA:
      healthy (index > 200DMA): no stops — let trends run.
      bear   (index < 200DMA): arm a hard stop (BAD_STOP) and a trailing stop
                                (BAD_TRAIL from peak) on every holding, checked
                                DAILY, to cut losers + lock gains in down markets.

Backtest: 2023-05..2026-05 (true N500, net 0.15%/side) = +69% CAGR / 27% DD /
Calmar 2.5; Mar-2025..May-2026 window = +46.8% / +38% annualized / 20% DD.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# ---- Strategy parameters (the locked window-optimized winner) ----
TOPN = 80            # universe = top-80 by 20d ADV from N500 (liquidity/quality)
K = 5                # hold 5 positions, equal-weight
RETAIN = 6           # keep a name while it stays in the top-6 rank
LOOKBACK = 30        # momentum ranking window (trading days)
MOM_FLOOR = 0.0      # require LOOKBACK-day return > 0 (positive momentum)
MAX_PRICE = 3000.0   # skip names priced above this at entry
ADV_WIN = 20         # ADV averaging window

# ---- Regime + risk (bear-only) ----
INDEX = "NSE:NIFTY50-INDEX"
REGIME_SMA = 200     # index trend gate
BAD_STOP = 0.10      # bear-regime hard stop: sell if px <= entry*(1-0.10)
BAD_TRAIL = 0.15     # bear-regime trailing stop: sell if px <= peak*(1-0.15)
# (healthy regime: no stops, no take-profit — winners ride)


def indicators(cl: pd.DataFrame, adv_rs: pd.DataFrame):
    """Build the rolling indicator panels the strategy needs.

    Args:
        cl: close price panel (date x symbol).
        adv_rs: per-bar traded value (close*volume) panel.
    Returns:
        (adv20, idx_sma200) — 20d ADV panel + the index 200-DMA series.
    """
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    idx_sma200 = cl[INDEX].rolling(REGIME_SMA).mean() if INDEX in cl.columns else None
    return adv20, idx_sma200


def regime_healthy(cl: pd.DataFrame, idx_sma200: pd.Series, di: int) -> bool:
    """True when Nifty-50 is above its 200-DMA at row `di` (uptrend regime)."""
    if INDEX not in cl.columns or idx_sma200 is None:
        return True
    iv = cl[INDEX].iloc[di]
    sv = idx_sma200.iloc[di]
    return bool(pd.notna(iv) and pd.notna(sv) and float(iv) > float(sv))


def rank_targets(cl, adv20, di):
    """Ranked momentum leaders passing all filters, at row index `di`.

    Universe = top-TOPN by 20d ADV (excluding the index row). Filters: price in
    (0, MAX_PRICE], LOOKBACK-day return > MOM_FLOOR. Returns Fyers-style symbols
    ordered best-to-worst by LOOKBACK-day return.
    """
    a = adv20.iloc[di].dropna()
    a = a[(a > 0) & (a.index != INDEX)]
    univ = list(a.sort_values(ascending=False).head(TOPN).index)
    row, rowL = cl.iloc[di], cl.iloc[di - LOOKBACK]
    out = []
    for s in univ:
        px, pxl = row.get(s), rowL.get(s)
        if pd.isna(px) or pd.isna(pxl) or pxl <= 0:
            continue
        if float(px) <= 0 or float(px) > MAX_PRICE:
            continue
        ret = (px / pxl - 1) * 100
        if ret <= MOM_FLOOR:
            continue
        out.append((s, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out]


def hit_stop(px: float, entry_px: float, healthy: bool) -> bool:
    """Bear-regime hard stop: True if (in bear) px has fallen >= BAD_STOP below entry."""
    if healthy or px is None or entry_px is None or entry_px <= 0:
        return False
    return float(px) <= entry_px * (1 - BAD_STOP)


def hit_trail(px: float, peak_px: float, healthy: bool) -> bool:
    """Bear-regime trailing stop: True if (in bear) px has fallen >= BAD_TRAIL off peak."""
    if healthy or px is None or peak_px is None or peak_px <= 0:
        return False
    return float(px) <= peak_px * (1 - BAD_TRAIL)
