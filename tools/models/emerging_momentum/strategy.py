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
  FULL 2023-05-15..2026-05-12 : ~+111% CAGR / ~23% DD / Calmar ~4.75  (climber ON)
  (climber OFF baseline = ~+98% CAGR / ~23% DD / Calmar ~4.2)
Config 1 (lb15, sma off) + MCAP CLIMBER filter (see CLIMBER_ENABLED below).
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from tools.shared.index_membership import eligible_at

ROOT = Path(__file__).resolve().parents[3]

# ---- Strategy parameters (max-1, lb30, retain-2, mid-month +5%, sma OFF) ----
# 2026-05-31 re-tune on the AUTHORITATIVE PIT membership: LB30/RET2 beats the old
# lb15/ret3 ("Config 1", tuned on the pre-rebuild buggy universe) — full-cycle
# 2021-03→2026-05 ≈ +65.6% vs +45.6% CAGR. Longer 30d momentum + tighter retain
# (hold while in top-2) ride the clean mid/small winners harder.
POOL = 100           # universe pool = top-100 by 20d ADV from (N500 minus N100)
TOPN = 100           # alias for POOL (display/compat with n100-style naming)
RETAIN = 1           # top-1 rotation (2026-05-31: RET1 + vol-adj beats RET2, +111% vs +95%)
LOOKBACK = 30        # momentum ranking window (TRADING days). 2026-05-31 re-tune.
MAX_PRICE = 3000.0   # skip names priced above this at entry
ADV_WIN = 20         # ADV averaging window
MIDMONTH_LEAD = 5.0  # rotate mid-month only if new rank-1 leads held by >= 5pp
# Vol-adjusted momentum: rank by LOOKBACK-return / VOL_WIN-day return-vol instead
# of raw return. On the high-vol mid/small universe this picks SMOOTH strong
# trends over jumpy ones and compounds much better (2026-05-31 sweep). Set
# RANK_MODE="ret" for the old raw-return ranking.
RANK_MODE = "vol_adj"
VOL_WIN = 60
_VOL_CACHE: dict = {}

# ---- MCAP CLIMBER overlay (validated +13pp CAGR at same DD, 2026-05-30) ------
# Keep only entry candidates whose free-float MARKET-CAP RANK has RISEN over the
# last CLIMB_LOOKBACK trading days (genuinely climbing toward index inclusion).
# Mechanically a cross-sectional price relative-strength filter (FF-shares are
# frozen at one NSE scrape, exports/nse_mcap.csv); it lifts emerging from ~+98%
# to ~+111% CAGR at the SAME ~23% DD. A/B across all models showed this edge
# lives ONLY in this liquid-mid tier. If the scrape CSV is absent the filter
# no-ops (falls back to pure-momentum baseline) so live never hard-fails.
CLIMBER_ENABLED = False   # 2026-05-31: climber was a pre-rebuild artifact; OFF + vol-adj
                          # ranking gives +111% vs +101% (climber-on) — and lower DD.
CLIMB_LOOKBACK = 60
MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
_MCAP_RANK_CACHE: dict = {}

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


def _load_ffmcap() -> dict:
    """Current free-float market cap (₹ Cr) per Fyers symbol from the NSE scrape."""
    out: dict = {}
    if not MCAP_CSV.exists():
        return out
    with open(MCAP_CSV) as f:
        for r in csv.DictReader(f):
            try:
                ff = float(r["ff_mcap_cr"])
                if ff > 0:
                    out[f"NSE:{r['symbol']}-EQ"] = ff
            except (ValueError, TypeError, KeyError):
                continue
    return out


def mcap_rank_panel(cl):
    """Date-indexed FF-mcap RANK panel (1 = biggest), memoized per `cl`.

    FF-shares = current FF-mcap / latest close (the scrape LTP is unreliable),
    applied to historical close -> ffmcap[t] = shares*close[t] -> daily rank.
    Returns None if the scrape CSV is missing/empty (climber then no-ops).
    """
    key = id(cl)
    if key in _MCAP_RANK_CACHE:
        return _MCAP_RANK_CACHE[key]
    ff = _load_ffmcap()
    shares = {}
    for s, v in ff.items():
        if s in cl.columns:
            last = cl[s].dropna()
            if len(last) and last.iloc[-1] > 0:
                shares[s] = v * 1e7 / last.iloc[-1]
    rank = None
    if shares:
        eq = list(shares)
        rank = cl[eq].mul(pd.Series(shares), axis=1).rank(axis=1, ascending=False, method="first")
    _MCAP_RANK_CACHE[key] = rank
    return rank


def _is_climber(rank, s, di) -> bool:
    """True if `s` IMPROVED (rose) in FF-mcap rank over CLIMB_LOOKBACK days."""
    if rank is None or s not in rank.columns:
        return False
    col = rank.columns.get_loc(s)
    a = rank.iat[di, col]
    b = rank.iat[max(0, di - CLIMB_LOOKBACK), col]
    return pd.notna(a) and pd.notna(b) and a < b


def _vol_panel(cl):
    """Cached VOL_WIN-day rolling std of daily returns (for vol-adjusted ranking)."""
    key = id(cl)
    if key not in _VOL_CACHE:
        _VOL_CACHE.clear()
        _VOL_CACHE[key] = cl.pct_change().rolling(VOL_WIN).std()
    return _VOL_CACHE[key]


def rank_pool(cl, pool, di):
    """Ranked momentum leaders passing the filters, at row index `di`.

    Universe = the PIT `pool` (top-POOL ADV from N500-minus-N100). Filters:
    LOOKBACK-day return > 0, price in (0, MAX_PRICE]. NO sma200 gate (Config 1
    winner). When CLIMBER_ENABLED + the scrape CSV is present, additionally keep
    only names whose FF-mcap rank has RISEN over CLIMB_LOOKBACK days (the
    validated +13pp edge). Returns Fyers symbols best-to-worst by LOOKBACK return.

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
        score = ret
        if RANK_MODE == "vol_adj":
            vp = _vol_panel(cl)
            v = vp[s].iloc[di] if s in vp.columns else None
            if v is None or pd.isna(v) or v <= 0:
                continue
            score = ret / v        # return per unit of volatility
        out.append((s, score))
    out.sort(key=lambda x: x[1], reverse=True)
    ranked = [s for s, _ in out]
    if CLIMBER_ENABLED:
        rank = mcap_rank_panel(cl)
        if rank is not None:
            ranked = [s for s in ranked if _is_climber(rank, s, di)]
    return ranked


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


# Rebalance calendar from the single shared source (all rotation models reuse it).
from tools.shared.rebalance_calendar import (   # noqa: E402
    build_calendar, is_mid_month_check_day, MID_MONTH_FROM_DAY)
