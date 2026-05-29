"""SHARED core logic for emerging_momentum — used by BOTH backtest + live.

Single source of truth so the offline backtest and the live signal can never
drift. backtest.py and live_signal.py both import the params + the universe
pool build (`build_pools` / `pool_for_date`), selection (`rank_targets`),
regime (`regime_healthy`) and risk-exit (`hit_trail`, `hit_bear_trail`,
`hit_bear_stop`) helpers from here.

Strategy ("Emerging Momentum"):
  - Universe: POINT-IN-TIME mid/small caps = top-POOL by 20-day ADV out of
    (eligible_at("n500", d) MINUS eligible_at("n100", d)), rebuilt per year-start
    so survivorship is honest. History is loaded from universe_union("n500").
  - Rank: LOOKBACK-day return, require ret > MOM_FLOOR (15%). Hold K equal-weight;
    a held name is kept while it stays in the top-RETAIN rank (winners ride).
  - Entry filter: close > 200-DMA (skip downtrending names), price <= MAX_PRICE.
  - Buys happen on the 1st trading day of each month (monthly rebalance), filled
    at the NEXT-day open. There is NO take-profit — winners run.
  - EXITS (checked DAILY, in priority order):
      1. ALWAYS-ON trailing stop: sell if px <= peak*(1-ALWAYS_TRAIL) in ALL
         regimes (cuts the healthy-regime give-back that drives DD).
      2. BEAR regime (Nifty50 < 200DMA): hard stop (px <= entry*(1-BEAR_STOP))
         OR bear trailing stop (px <= peak*(1-BEAR_TRAIL)).
      3. MONTHLY rank-drop: sell if a holding falls out of the top-RETAIN rank.

Backtest (PIT N500-minus-N100, net 0.15%/side, next-day-open entry):
  FULL 2023-05..2026-05 : ~+77% CAGR / ~26% DD
    per-year 2023 ~+129, 2024 ~+19, 2025 ~+81, 2026 ~+11 (partial)
  WINDOW Mar-2025..May-2026 : ~+84% CAGR / ~16% DD
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from tools.shared.index_membership import eligible_at

ROOT = Path(__file__).resolve().parents[3]

# ---- Strategy parameters (the locked winner: sma200 + momfloor15 + trail25) ----
POOL = 100           # universe pool = top-100 by 20d ADV from (N500 minus N100)
K = 3                # hold 3 positions, equal-weight
RETAIN = 4           # keep a name while it stays in the top-4 rank
LOOKBACK = 30        # momentum ranking window (trading days)
MOM_FLOOR = 0.15     # require LOOKBACK-day return > 15% (stronger-momentum only)
MAX_PRICE = 3000.0   # skip names priced above this at entry
ADV_WIN = 20         # ADV averaging window
ENTRY_SMA = 200      # entry gate: close must be above its 200-DMA

# Canonical anchor for the per-year PIT pool rebuild. The pool is rebuilt every
# 12 months stepping from THIS date (not the eval-window start and not calendar
# Jan-1), so any sub-window backtest uses the exact same pools the full
# 2023-05-15.. backtest used — matching tools/analysis/emerging_momentum_dd.py
# (which always builds pools over its FULL range regardless of eval window).
POOL_ANCHOR_START = date(2023, 5, 15)

# ---- Regime + risk ----
INDEX = "NSE:NIFTY50-INDEX"
REGIME_SMA = 200     # index trend gate
ALWAYS_TRAIL = 0.25  # ALL-regime trailing stop: sell if px <= peak*(1-0.25)
BEAR_STOP = 0.10     # bear-regime hard stop: sell if px <= entry*(1-0.10)
BEAR_TRAIL = 0.15    # bear-regime trailing stop: sell if px <= peak*(1-0.15)
# (healthy regime: only the always-on 25% trail; no take-profit — winners ride)


def indicators(cl: pd.DataFrame, adv_rs: pd.DataFrame):
    """Build the rolling indicator panels the strategy needs.

    Args:
        cl: close price panel (date x symbol).
        adv_rs: per-bar traded value (close*volume) panel.
    Returns:
        (adv20, sma200, idx_sma200) — 20d ADV panel, per-symbol 200-DMA panel,
        and the index 200-DMA series.
    """
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(ENTRY_SMA).mean()
    idx_sma200 = cl[INDEX].rolling(REGIME_SMA).mean() if INDEX in cl.columns else None
    return adv20, sma200, idx_sma200


def _year_anchors(end: date):
    """Year anchors stepping by 12 months from POOL_ANCHOR_START up to `end`.

    Independent of the eval-window start so sub-window backtests reuse the same
    pools the full backtest built (matches emerging_momentum_dd.py).
    """
    out = []
    y = POOL_ANCHOR_START
    while y <= end:
        out.append(pd.Timestamp(y))
        try:
            y = y.replace(year=y.year + 1)
        except ValueError:                       # Feb-29 anchor — clamp to Feb-28
            y = y.replace(year=y.year + 1, day=28)
    return out


def build_pools(adv20: pd.DataFrame, dates, end: date):
    """Per-year PIT pools of emerging (N500 minus N100) liquidity leaders.

    For each year anchor we take the first trading day >= anchor, compute the
    eligible mid/small set (eligible_at n500 MINUS eligible_at n100) AS OF that
    day, then keep the top-POOL by 20d ADV on that day. Mirrors
    emerging_momentum_dd.build_pools exactly (anchors fixed to POOL_ANCHOR_START).

    Returns (year_anchors, {anchor_ts: [fyers_symbol, ...]}).
    """
    anchors = _year_anchors(end)
    pools: dict[pd.Timestamp, list[str]] = {}
    for y in anchors:
        fut = dates[dates >= y]
        if len(fut) == 0:
            continue
        di = dates.get_loc(fut[0])
        yd = fut[0].date()
        mids = {f"NSE:{s}-EQ" for s in eligible_at("n500", yd)} \
            - {f"NSE:{s}-EQ" for s in eligible_at("n100", yd)}
        a = adv20.iloc[di].dropna().sort_values(ascending=False)
        pools[y] = a[a.index.isin(mids)].head(POOL).index.tolist()
    return anchors, pools


def pool_for_date(anchors, pools, d) -> list[str]:
    """The pool in force at date `d` = the most recent year anchor <= d."""
    chosen = anchors[0] if anchors else None
    for y in anchors:
        if d >= y:
            chosen = y
    return pools.get(chosen, []) if chosen is not None else []


def regime_healthy(cl: pd.DataFrame, idx_sma200: pd.Series, di: int) -> bool:
    """True when Nifty-50 is above its 200-DMA at row `di` (uptrend regime)."""
    if INDEX not in cl.columns or idx_sma200 is None:
        return True
    iv = cl[INDEX].iloc[di]
    sv = idx_sma200.iloc[di]
    return bool(pd.notna(iv) and pd.notna(sv) and float(iv) > float(sv))


def rank_targets(cl, sma200, pool, di):
    """Ranked momentum leaders passing all filters, at row index `di`.

    Universe = the PIT `pool` (top-POOL ADV from N500-minus-N100). Filters:
    LOOKBACK-day return > MOM_FLOOR, price in (0, MAX_PRICE], close > 200-DMA.
    Returns Fyers-style symbols ordered best-to-worst by LOOKBACK-day return.
    """
    row, rowL = cl.iloc[di], cl.iloc[di - LOOKBACK]
    out = []
    for s in pool:
        px, pxl = row.get(s), rowL.get(s)
        if pd.isna(px) or pd.isna(pxl) or pxl <= 0:
            continue
        ret = px / pxl - 1
        if ret <= MOM_FLOOR:
            continue
        if not (0 < float(px) <= MAX_PRICE):
            continue
        sm = sma200.iloc[di].get(s)
        if pd.isna(sm) or float(px) <= float(sm):
            continue
        out.append((s, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out]


def hit_trail(px: float, peak_px: float) -> bool:
    """ALWAYS-ON trailing stop: True if px has fallen >= ALWAYS_TRAIL off peak."""
    if px is None or peak_px is None or peak_px <= 0:
        return False
    return float(px) <= peak_px * (1 - ALWAYS_TRAIL)


def hit_bear_trail(px: float, peak_px: float, healthy: bool) -> bool:
    """Bear-regime trailing stop: True if (in bear) px fell >= BEAR_TRAIL off peak."""
    if healthy or px is None or peak_px is None or peak_px <= 0:
        return False
    return float(px) <= peak_px * (1 - BEAR_TRAIL)


def hit_bear_stop(px: float, entry_px: float, healthy: bool) -> bool:
    """Bear-regime hard stop: True if (in bear) px fell >= BEAR_STOP below entry."""
    if healthy or px is None or entry_px is None or entry_px <= 0:
        return False
    return float(px) <= entry_px * (1 - BEAR_STOP)
