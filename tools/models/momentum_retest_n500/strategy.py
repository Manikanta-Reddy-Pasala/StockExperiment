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

# ---- Strategy parameters (K4/top120/ret4/band20% — 2026-05-31 re-tune) ----
# CURRENT headline (2026-06-13 realism regen — net of real Fyers CNC charges,
# next-open fills, PIT-before-ADV fix):
#   full 2021-03→2026-05: +58.7% CAGR / 34.0% DD / Calmar 1.73 / 183 trades
#   3-yr 2023-05→2026-05: +102.3% CAGR / 23.6% DD / Calmar 4.34
# Sweep history below is pre-realism (close fills, flat 0.15%/side):
# 2026-05-31: K2→K4. On both windows K4 dominates K2 on risk-adjusted return:
#   full 2021-03→2026-05: +57% CAGR / 39% DD / Calmar 1.48 (was K2 +64/57/1.12)
#   recent 2025-03→2026-05: +53% CAGR / 15% DD (was K2 +38/21)
# K2→K4 diversifies the basket: recent CAGR +38→+53, DD cut both windows
# (full 57→39, per-year DD ≤32 EVERY year). −7pt full CAGR for far better DD.
# K-knee verified: K5/K6 decay. (The old K2 "beat K3/4/5" was the pre-PIT
# survivorship-era sweep; on the corrected PIT N500 engine K4 is the winner.)
# Other lever (unchanged):
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
# Partial profit-take: book HALF a holding once it closes >= entry*(1+PCT), the
# rest rides until it leaves the retain band. Default OFF (0.0); test/enable per
# the 2026-06-06 emerging finding (helps high-vol names). Set >0 to enable.
PROFIT_TAKE_PCT = 0.0


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


def rank_targets(cl, adv20, sma200, smallcap, di, eligible=None):
    """Ranked momentum leaders passing all filters, at row index `di`.

    Universe = top-TOPN by 20d ADV (minus smallcap). Filters: close>SMA200,
    price<=MAX_PRICE, LOOKBACK-day return>MOM_FLOOR, ACCEL_DAYS-day return>0.
    Returns Fyers-style symbols ordered best-to-worst by LOOKBACK-day return.

    eligible (optional): set of PLAIN symbols (no NSE:/-EQ) that are index-
    eligible at this date — e.g. eligible_at("n500", d) from the backtest's
    PIT membership. When given, the ADV top-TOPN cut is taken over eligible
    names ONLY (PIT-before-ADV, 2026-06-13 fix): previously the backtest
    post-filtered AFTER the head(TOPN) cut, so historically-ineligible names
    displaced eligible ones at the ADV margin. LIVE passes None — its panel
    is built from the CURRENT nifty500 list, which is trivially PIT for
    today, so live behavior is unchanged.
    """
    a = adv20.iloc[di].dropna()
    a = a[a > 0]
    if eligible is not None:
        a = a[[s for s in a.index
               if s.replace("NSE:", "").replace("-EQ", "") in eligible]]
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
