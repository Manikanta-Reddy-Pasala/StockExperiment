"""SHARED core for midcap_breakout_k3 — used by BOTH backtest + live.

K=3 multi-position variant of midcap_narrow_60d_breakout. Same entry/exit/
universe rules as the single-position model; the ONLY change is holding up to K
concurrent breakouts (deploys idle capital that the single-position model leaves
in cash between rare breakouts). At K=1 this reduces EXACTLY to the original
model (validation gate).

Entry  : fresh HH_WIN-day high + close>200DMA + volume>=VOL_MULT×20d avg.
         Rank competing breakouts by volume ratio; fill free slots; enter at
         NEXT day's OPEN (signal forms on close, fill tomorrow).
Exit   : SHARED breakout_exit_reason — TARGET +100% / STOP -20% / TRAIL -20% off
         peak (armed at +10%) / MAX_HOLD 120 calendar days.
Universe: PIT — top-KEEP_NEXT by 20d ADV from PIT-N500 (ever-member union),
         minus PIT-N100, intersect freshly-trading names (staleness<=5 sessions).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from tools.shared.breakout_strategy import is_breakout, breakout_exit_reason  # noqa: F401
from tools.shared.index_membership import eligible_at, universe_union

# ---- params (V2 winner of the single-position model) ----
K = 3                 # concurrent positions (the variant's defining change)
HH_WIN = 40
VOL_MULT = 2.0
SMA_LONG = 200
TRAIL_PCT = 0.20
PROFIT_TRIG = 0.10
TARGET_PCT = 1.00
STOP_PCT = 0.20
MAX_HOLD = 120        # calendar days
ADV_WIN = 20
SKIP_TOP = 0
KEEP_NEXT = 100
STALE_SESSIONS = 5
# costs
SLIP, BR, STT = 0.001, 20, 0.001


def n500_union_symbols():
    """Fyers-form symbols for every stock that was EVER an n500 member (PIT)."""
    return [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]


def fresh_symbols(df: pd.DataFrame):
    """Names whose true last bar is within STALE_SESSIONS sessions of the panel
    end (drops delisted/suspended names a ffilled close would otherwise carry)."""
    all_dates = sorted(df["date"].unique())
    cutoff = all_dates[-(STALE_SESSIONS + 1)] if len(all_dates) > STALE_SESSIONS else all_dates[0]
    last_seen = df.groupby("symbol")["date"].max()
    return set(last_seen[last_seen >= cutoff].index)


def indicators(cl, hi, vol, adv_rs):
    """Rolling panels: 200d SMA, prior-HH_WIN-day high (shifted), 20d avg vol, 20d ADV."""
    sma_long = cl.rolling(SMA_LONG).mean()
    hh = hi.rolling(HH_WIN).max().shift(1)
    vol_avg20 = vol.rolling(20).mean()
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    return sma_long, hh, vol_avg20, adv20


def build_year_pools(adv20, dates, fresh_syms, start: date, end: date):
    """PIT midcap pools per year-start: top-KEEP_NEXT by 20d ADV from PIT-N500,
    minus PIT-N100, intersect fresh names. Returns (year_starts, pools)."""
    ys_cursor = start
    year_starts = []
    while ys_cursor <= end:
        year_starts.append(pd.Timestamp(ys_cursor))
        ys_cursor = ys_cursor.replace(year=ys_cursor.year + 1)
    pools = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0:
            continue
        di = dates.get_loc(fut[0]); ysd = fut[0].date()
        n500_elig = {f"NSE:{s}-EQ" for s in eligible_at("n500", ysd)}
        a = adv20.iloc[di].dropna().sort_values(ascending=False)
        a = a[a.index.isin(n500_elig) & a.index.isin(fresh_syms)]
        pool = a.iloc[SKIP_TOP:SKIP_TOP + KEEP_NEXT].index.tolist()
        n100 = set(eligible_at("n100", ysd))
        pools[ys] = [s for s in pool if s.replace("NSE:", "").replace("-EQ", "") not in n100]
    return year_starts, pools


def band_for(year_starts, pools, d):
    """PIT midcap band in force on date `d` (latest year-start on/before d)."""
    chosen = year_starts[0]
    for ys in year_starts:
        if d >= ys:
            chosen = ys
    return pools.get(chosen, [])


def scan_breakouts(cl, sma_long, hh, vol, vol_avg20, band, di, held: set):
    """Ranked breakout candidates (best volume-ratio first) at row di, excluding
    names already held. Returns list of {sym, vr}."""
    cands = []
    for sym in band:
        if sym in held or sym not in cl.columns:
            continue
        c_v = cl[sym].iloc[di]
        sma_v = sma_long[sym].iloc[di] if sym in sma_long.columns else None
        hh_v = hh[sym].iloc[di] if sym in hh.columns else None
        va_v = vol_avg20[sym].iloc[di] if sym in vol_avg20.columns else None
        v_v = vol[sym].iloc[di] if sym in vol.columns else None
        if any(x is None or pd.isna(x) for x in (c_v, sma_v, hh_v, va_v, v_v)):
            continue
        ok, vr = is_breakout(float(c_v), float(hh_v), float(sma_v), float(v_v), float(va_v),
                             vol_mult=VOL_MULT)
        if ok:
            cands.append({"sym": sym, "vr": vr})
    cands.sort(key=lambda x: -x["vr"])
    return cands
