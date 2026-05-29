"""SHARED core logic for emerging_momentum — used by BOTH backtest + live.

Single source of truth so the offline backtest and the live signal can never
drift. backtest.py and live_signal.py both import the params + the universe
pool build (`build_pools` / `pool_for_date`) and the per-date ranking
(`rank_pool`) from here.

Strategy ("Emerging Momentum") — SINGLE-POSITION rotation (Config 1):
  - Universe: POINT-IN-TIME mid/small caps = top-POOL by 20-day ADV out of
    (eligible_at("n500", d) MINUS eligible_at("n100", d)), rebuilt per year-start
    so survivorship is honest. History is loaded from universe_union("n500").
  - Rank: LOOKBACK-day return, require ret > 0. Hold ONE name (max_concurrent=1);
    a held name is kept while it stays in the top-RETAIN rank (winner rides).
  - Filter: price in (0, MAX_PRICE]. NO sma200 gate (Config 1 winner = sma OFF).
  - Rotation: monthly (1st trading day) + MID-MONTH check. The mid-month check
    rotates only if a new leader beats the held name's LOOKBACK-day return by
    >= MIDMONTH_LEAD percentage points.
  - Execution: tools.shared.backtest_engine.run_rotation_backtest (same engine
    as momentum_n100_top5_max1); live path = tools/live/fyers_executor.py via the
    single-position model_ledger.

Backtest (PIT N500-minus-N100, gross-of-fee rotation engine, same-day close):
  FULL 2023-05-15..2026-05-12 : ~+98% CAGR / ~23% DD / Calmar ~4.2
  WINDOW Mar-2025..May-2026    : ~+56% CAGR
Reproduces tools/analysis/emerging_variants.py CONFIG 1 (lb15, sma off).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from tools.shared.index_membership import eligible_at

ROOT = Path(__file__).resolve().parents[3]

# ---- Strategy parameters (Config 1: max-1, lb15, retain-3, mid-month +5%, sma OFF) ----
POOL = 100           # universe pool = top-100 by 20d ADV from (N500 minus N100)
TOPN = 100           # alias for POOL (display/compat with n100-style naming)
RETAIN = 3           # keep the held name while it stays in the top-3 rank
LOOKBACK = 15        # momentum ranking window (TRADING days). Config 1 winner.
MAX_PRICE = 3000.0   # skip names priced above this at entry
ADV_WIN = 20         # ADV averaging window
MIDMONTH_LEAD = 5.0  # rotate mid-month only if new rank-1 leads held by >= 5pp

# Canonical anchor for the per-year PIT pool rebuild. The pool is rebuilt every
# 12 months stepping from THIS date (not the eval-window start and not calendar
# Jan-1), so any sub-window backtest uses the exact same pools the full
# 2023-05-15.. backtest used — matching tools/analysis/emerging_variants.py
# (which always builds pools over its FULL range regardless of eval window).
POOL_ANCHOR_START = date(2023, 5, 15)

# Index symbol kept only for the equity-trading-day mask in load_panels (the
# index trades on days some equities don't and would poison rolling windows).
INDEX = "NSE:NIFTY50-INDEX"


def indicators(cl: pd.DataFrame, adv_rs: pd.DataFrame):
    """Build the rolling indicator panels the strategy needs.

    Args:
        cl: close price panel (date x symbol).
        adv_rs: per-bar traded value (close*volume) panel.
    Returns:
        adv20 — the 20d rolling ADV panel used to pick the per-year ADV pool.
    """
    return adv_rs.rolling(ADV_WIN).mean()


def _year_anchors(end: date):
    """Year anchors stepping by 12 months from POOL_ANCHOR_START up to `end`.

    Independent of the eval-window start so sub-window backtests reuse the same
    pools the full backtest built (matches emerging_variants.py build_pools).
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
    emerging_variants.build_pools exactly (anchors fixed to POOL_ANCHOR_START).

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


def rank_pool(cl, pool, di):
    """Ranked momentum leaders passing the filters, at row index `di`.

    Universe = the PIT `pool` (top-POOL ADV from N500-minus-N100). Filters:
    LOOKBACK-day return > 0, price in (0, MAX_PRICE]. NO sma200 gate (Config 1
    winner). Returns Fyers-style symbols ordered best-to-worst by LOOKBACK-day
    return.

    Args:
        cl: close price panel (date x symbol), ffilled.
        pool: the PIT symbol pool in force for this date.
        di: integer row index into `cl` for the rebalance day.
    """
    if di < LOOKBACK:
        return []
    row, rowL = cl.iloc[di], cl.iloc[di - LOOKBACK]
    out = []
    for s in pool:
        px, pxl = row.get(s), rowL.get(s)
        if pd.isna(px) or pd.isna(pxl) or pxl <= 0:
            continue
        ret = px / pxl - 1
        if ret <= 0:
            continue
        if not (0 < float(px) <= MAX_PRICE):
            continue
        out.append((s, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out]


def midret_pool(cl, pool, di):
    """(symbol, LOOKBACK-day return %) pairs for the mid-month lead gate.

    Restricted to the PIT `pool`, NO filters and emitted in POOL ORDER (which
    is 20d-ADV descending from build_pools), NOT sorted by return. This mirrors
    tools/analysis/emerging_variants.py `mm` EXACTLY (which is what produces the
    validated Config-1 numbers). The engine's mid-month lead gate
    (midmonth_lead_ok) compares the held name's LOOKBACK return against this
    list's first entry; keeping pool order (not return order) is load-bearing —
    sorting here changes the gate behaviour and breaks Config-1 parity.

    Returns list[tuple[str, float]], return expressed in percentage points (the
    unit MIDMONTH_LEAD compares against).
    """
    if di < LOOKBACK:
        return []
    row, rowL = cl.iloc[di], cl.iloc[di - LOOKBACK]
    out = []
    for s in pool:
        px, pxl = row.get(s), rowL.get(s)
        if pd.isna(px) or pd.isna(pxl) or pxl <= 0:
            continue
        out.append((s, (px / pxl - 1) * 100))
    return out
