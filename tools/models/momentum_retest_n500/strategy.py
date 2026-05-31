"""SHARED core logic for momentum_retest_n500 — used by BOTH backtest + live.

Single source of truth so the offline backtest and the live signal can never
drift. backtest.py and live_signal.py both import the params, selection
(`rank_targets`) and entry (`is_retest`) from here.

Strategy: monthly pick the top-K momentum leaders from the top-N-ADV liquid
N500 pool (uptrend + filters); buy each on a pullback/retest to its 20-EMA;
hold while it stays in the top-`RETAIN` rank.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# ---- Strategy parameters (K2/top120/ret4/band20% — 2026-05-30 sweep winner) ----
# Both-window sweep (2023-26 + full 2021-26 cycle, real PIT engine) upgraded this
# from the old K3/ret6/band8% config. The new config DOMINATES on both windows:
#   2023-26: +146% vs +91% CAGR, 21% vs 19% DD, Calmar 7.0 vs 4.9
#   2021-26: +66%  vs +39% CAGR, 39% vs 48% DD (LOWER), Calmar 1.7 vs 0.8
# Two independent Pareto levers found:
#   - K=2 (concentrate from 3): fewer, higher-conviction names beat K3/4/5.
#   - RETEST_HI=0.20 (was 0.08): the tight pullback band was the main drag — the
#     strongest momentum leaders never pull back to within 8% of the 20-EMA, so
#     the model kept missing them. Widening to 20% lets them in; effect plateaus
#     past 0.20 (natural knee — names rarely sit >20% above EMA), so not overfit.
TOPN = 120          # universe = top-120 by 20d ADV from N500 (minus smallcap)
K = 4               # 2026-05-31 re-tune: K4 (was K2) diversifies the basket —
                    # recent 2025-03→2026-05 CAGR +38→+53, recent DD 21→15, AND
                    # full-cycle DD 57→39 (per-year DD now ≤32 every year), for
                    # only −7pt full CAGR (+64→+57). K-knee: K5/K6 decay. The
                    # old K2 note ("beat K3/4/5") was the pre-PIT survivorship era.
RETAIN = 4          # hold a name while it stays in the top-4 rank
LOOKBACK = 30       # momentum ranking window (trading days; 30 beat 15/20/40)
MOM_FLOOR = 10.0    # require LOOKBACK-day return > 10% (10 beat 0/5/15/20)
ACCEL_DAYS = 10     # require ACCEL_DAYS-day return > 0 (accelerating)
MAX_PRICE = 3000.0  # skip names priced above this at entry
RETEST_LO = 0.01    # entry: price >= 20EMA * (1 - 1%)
RETEST_HI = 0.20    # entry: price <= 20EMA * (1 + 20%) — widened from 8% (see above)
ADV_WIN = 20        # ADV averaging window
SMA_LONG = 200      # uptrend gate
EMA_FAST = 20       # retest reference EMA


def load_smallcap() -> set:
    """Nifty Smallcap-250 plain symbols (EQ) — subtracted from the universe."""
    out = set()
    p = ROOT / "src" / "data" / "symbols" / "nifty_smallcap250.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("Series", "").strip() == "EQ":
                out.add(r["Symbol"].strip())
    return out


def indicators(cl: pd.DataFrame, adv_rs: pd.DataFrame):
    """Build the rolling indicator panels the strategy needs.

    Args:
        cl: close price panel (date x symbol).
        adv_rs: per-bar traded value (close*volume) panel.
    Returns:
        (adv20, sma200, ema20) panels.
    """
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(SMA_LONG).mean()
    ema20 = cl.ewm(span=EMA_FAST).mean()
    return adv20, sma200, ema20


def rank_targets(cl, adv20, sma200, smallcap, di):
    """Ranked momentum leaders passing all filters, at row index `di`.

    Universe = top-TOPN by 20d ADV (minus smallcap). Filters: close>SMA200,
    price<=MAX_PRICE, LOOKBACK-day return>MOM_FLOOR, ACCEL_DAYS-day return>0.
    Returns Fyers-style symbols ordered best-to-worst by LOOKBACK-day return.
    """
    a = adv20.iloc[di].dropna()
    a = a[a > 0]
    univ = [s for s in a.sort_values(ascending=False).head(TOPN).index
            if s.replace("NSE:", "").replace("-EQ", "") not in smallcap]
    row, rowL, rowA = cl.iloc[di], cl.iloc[di - LOOKBACK], cl.iloc[di - ACCEL_DAYS]
    out = []
    for s in univ:
        px, pxl = row.get(s), rowL.get(s)
        if pd.isna(px) or pd.isna(pxl) or pxl <= 0:
            continue
        sv = sma200[s].iloc[di]
        if pd.isna(sv) or px < sv or float(px) > MAX_PRICE:
            continue
        ret = (px / pxl - 1) * 100
        if ret < MOM_FLOOR:
            continue
        pa = rowA.get(s)
        if pd.isna(pa) or px <= float(pa):
            continue
        out.append((s, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out]


def is_retest(px: float, ema20_val: float) -> bool:
    """Retest entry test: price within [-RETEST_LO, +RETEST_HI] of the 20-EMA."""
    if px is None or ema20_val is None or pd.isna(px) or pd.isna(ema20_val):
        return False
    return ema20_val * (1 - RETEST_LO) <= float(px) <= ema20_val * (1 + RETEST_HI)
