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

Backtest (PIT N500-minus-N100, vol-adj rank + 2.5×ATR stop) — 2026-06-13
REALISM CONVENTION: net of real Fyers CNC charges, next-open fills:
  FULL 2021-03..2026-06 : +105.3% CAGR / 38.6% DD / Calmar 2.73 (charges ₹2.56M)
  RECENT 2023-05..2026-05 : +138.9% CAGR / 27.4% DD / Calmar 5.07
  (old close-fill zero-charge convention: +115.6%/37.9% full, +165%/26% recent)
  ALL UNLEVERED (own cash only — no borrow). Backtest == live core
  (PIT, no-lookahead). Current config = vol-adj rank, RETAIN=1, lb30.
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
# 2026-06-13 re-tune POOL 100 -> 80 (tools/research/emerging_improve.py): narrowing
# to the top-80 by 20d ADV drops the lower-liquidity / jumpier mid-small tail that
# diluted compounding + inflated DD. Beats pool-100 on ALL windows (full
# +110->+135 CAGR / 38.6->35.2 DD / Calmar 2.86->3.83; 3y +139->+171/5.07->6.11;
# since-Mar-2025 +46->+56/1.69->2.01), WF OOS both-axes win, PLATEAU-confirmed
# (pool 80 ~= 85, smooth shoulder — not a spike). (VOL_WIN 90 tested too and
# REJECTED: in-sample spike, fails the plateau check.)
POOL = 80            # universe pool = top-80 by 20d ADV from (N500 minus N100)
TOPN = 80            # alias for POOL (display/compat with n100-style naming)
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

# ---- ATR-from-ENTRY hard stop (2026-06-01, backtest-validated) --------------
# A hard stop at entry_px - ATR_STOP_MULT * ATR(ATR_WIN), checked DAILY. It is a
# FIXED level anchored at the entry day (NOT trailing) so it only cuts genuine
# breakdowns below entry and never whipsaws a winner out. 2.5x is the sweet spot
# from tools/analysis/emerging_pricefloor_atr_sweep.py (validated BOTH windows +
# every calendar year): full 2021-26 +27% net P&L / 120% CAGR / 38% DD; recent
# 2023-26 +37% / 168% CAGR / 26% DD — vs the rotation-only baseline. The ATR
# value is the CURRENT ATR at the check (recomputed daily); the LEVEL is anchored
# to entry. Set ATR_STOP_MULT=0 to disable.
ATR_STOP_MULT = 2.5
ATR_WIN = 14

# ---- Partial profit-take (2026-06-06, backtest-validated) -------------------
# Book HALF the position ONCE when price first closes >= entry*(1+PROFIT_TAKE_PCT);
# the other half keeps riding under the ATR-from-entry stop. On this high-vol
# mid/small universe the names spike then mean-revert, so banking half the spike
# cuts drawdown materially while the runner still captures the trend — a genuine
# BOTH-axes win, smooth/monotonic across 1yr/2yr/3yr (PT sweep 2026-06-06):
#   3yr  2023-05→2026-05 : Calmar 6.44 -> 7.34, DD 26.3 -> 22.3, CAGR ~flat
#   2yr                  : Calmar 2.49 -> 3.73, DD 24.6 -> 18.4, CAGR 61 -> 68
#   1yr  2025-06→2026-06 : Calmar 1.99 -> 3.53, DD 24.6 -> 17.7, CAGR 49 -> 62
# 30% is mid-plateau (15-35% all help; >35% the trigger rarely fires -> fades).
# Set PROFIT_TAKE_PCT=0 to disable. Checked DAILY on the close (live mirrors via
# the --stop-check path so backtest/live can't drift).
#
# 2026-06-10 — DISABLED (set to 0) by request, to "let winners run" for max CAGR.
# Full-window evidence (2021-03->2026-06): PT 0 vs 0.30 = CAGR 109->116 at the SAME
# DD (38) -> Calmar 2.88->3.05, because 2022/2023 mega-trends run further
# (2023 +269%->+358%). TRADE-OFF (recorded, eyes-open): in the recent chop regime
# the half-bank HELPED, so disabling HURTS there -> 2025 CAGR 62->51, DD 18->25,
# Calmar 3.48->2.08. i.e. this optimises the long bull pattern at the cost of the
# current regime. Revert to 0.30 if the recent-regime drawdown proves worse live.
PROFIT_TAKE_PCT = 0.0


def atr_latest(high: pd.Series, low: pd.Series, close: pd.Series,
               win: int = ATR_WIN):
    """Latest ATR (simple mean of True Range over `win`) for one symbol's
    high/low/close series. Returns None if insufficient / invalid data."""
    try:
        df = pd.DataFrame({"h": high, "l": low, "c": close}).dropna()
        if len(df) < win + 1:
            return None
        prev_c = df["c"].shift(1)
        tr = pd.concat([
            (df["h"] - df["l"]).abs(),
            (df["h"] - prev_c).abs(),
            (df["l"] - prev_c).abs(),
        ], axis=1).max(axis=1)
        a = float(tr.rolling(win).mean().iloc[-1])
        return a if a > 0 else None
    except Exception:
        return None


def atr_stop_level(entry_px: float, atr_val: float,
                   mult: float = ATR_STOP_MULT):
    """The hard-stop price = entry_px - mult*ATR. <=0 / bad inputs -> None."""
    try:
        if entry_px and atr_val and atr_val > 0 and mult and mult > 0:
            lvl = float(entry_px) - float(mult) * float(atr_val)
            return lvl if lvl > 0 else None
    except (TypeError, ValueError):
        pass
    return None


def atr_stop_hit(entry_px: float, atr_val: float, day_low: float,
                 mult: float = ATR_STOP_MULT):
    """(hit, level): True if the day's LOW pierced entry_px - mult*ATR."""
    lvl = atr_stop_level(entry_px, atr_val, mult)
    if lvl is None or day_low is None:
        return False, lvl
    try:
        return float(day_low) <= lvl, lvl
    except (TypeError, ValueError):
        return False, lvl

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
    return [s for s, _ in rank_pool_scored(cl, pool, di)]


def rank_pool_scored(cl, pool, di):
    """Like rank_pool but returns [(symbol, rank score)] best-to-worst.

    The score is the actual rank key (LOOKBACK return, or return ÷ VOL_WIN-day
    return-vol when RANK_MODE == 'vol_adj') so the UI can show WHY the order
    differs from raw 30d return. Same filters and order as rank_pool.
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
            score = ret / v
        out.append((s, score))
    out.sort(key=lambda x: x[1], reverse=True)
    if CLIMBER_ENABLED:
        rank = mcap_rank_panel(cl)
        if rank is not None:
            out = [(s, sc) for s, sc in out if _is_climber(rank, s, di)]
    return out


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
