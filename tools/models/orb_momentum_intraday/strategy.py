"""SHARED core for orb_momentum_intraday — params + selection + ORB rules,
imported by BOTH backtest.py and live_signal.py so they can never drift.

================================ HOW IT WORKS ================================
A DAY-TRADING model (the only intraday model in the book; everything else is a
multi-day swing/rotation strategy). It combines a daily MOMENTUM selection with
an intraday OPENING-RANGE-BREAKOUT (ORB) execution, long-only, flat by EOD.

Each trading day:
  1. SELECT — at the open, rank the Nifty-500 (point-in-time membership) by the
     trailing LOOKBACK-day (20) return and take the top SELECT_TOP (3) momentum
     leaders. (Momentum stocks trend; their intraday breakouts continue. Raw ORB
     on random names just whipsaws — the momentum filter is the edge.)
  2. OPENING RANGE — for each leader, the first OR_BARS (3 × 5-min = 15 min,
     09:15-09:30) define the opening range: ORH = high, ORL = low, width = ORH-ORL.
  3. ENTRY — watch the top SELECT_TOP leaders; go ALL-IN (full capital, ONE
     position) on the FIRST to break ABOVE its ORH, but ONLY if that breakout is
     before ENTRY_CUTOFF (10:00). The pick is the earliest breakout (rank as the
     tiebreak) — achievable live, we commit the moment the first one breaks
     without needing to know which others break later. Long-only. One trade per
     day: after the position exits there is no re-entry.
  4. STOP / TARGET — stop at ORL (one range-width below entry); target at
     entry-side ORH + TARGET_MULT (2.0) × width.
  5. EXIT — whichever comes first: stop, target, or a forced square-off at
     EOD_FLAT (15:15). Most trades exit flat at EOD (riding the intraday
     trend); the rest stop/target. ZERO overnight risk.

Position sizing (live): ALL-IN, single position. The full model capital
(invested_amount) goes into the one best-momentum leader that breaks out
(strategy.full_qty + pick_leader). The earlier 1/SELECT_TOP-per-slot scheme left
~45% of capital idle (≈49% of days only one of three leaders fires), roughly a
third of the achievable return — so ORB now concentrates into the single
breakout instead of reserving slots for leaders that may never break.

================================ WHY THESE PARAMS ============================
6-yr daily momentum work established that momentum is the only ≥60% edge on NSE.
This model ports that edge to intraday execution. A 2025-03→2026-05 sweep on the
full PIT Nifty-500 (5-min bars, realistic 0.15% slippage + 0.15% round-trip cost)
found:
  - SELECT_TOP=3 (concentrated leaders) beat 5/8.
  - ENTRY_CUTOFF=10:00 (morning only) ~doubled annualized vs all-day (+251 vs +124).
  - TARGET_MULT=2.0 + EOD-flat captured the trend better than tighter targets.
Result at realistic slippage: +216% total / +251% ann / 17% max DD / Sharpe ~3.45
over 15 months, 13/15 months green.

================================ HONEST CAVEATS =============================
  - Validated on ONE bull regime (15 months). Feb-2026 (-11.9%) shows it bleeds
    in chop; no 2022-type bear has been tested intraday.
  - HIGHLY slippage-sensitive: +251% at 0.15% slip degrades to ~+46-90% at 0.25%.
    The real number is only knowable from live fills — PAPER-TRADE before trusting.
  - The fill model is bar-level (entry at ORH, intra-bar stop/target on 5-min
    bars). Tick reality differs; treat magnitudes as optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

# ---- Strategy parameters (single source of truth, shared backtest<->live) ----
INDEX = "n500"          # PIT universe for the daily momentum selection
LOOKBACK = 20           # momentum ranking window (trading days)
SELECT_TOP = 3          # trade the top-3 momentum leaders each day
OR_BARS = 3             # opening-range = first 3 × 5-min bars (09:15-09:30 = 15 min)
ENTRY_CUTOFF_MIN = 600  # only enter if the breakout fires before 10:00 (=10*60 min)
TARGET_MULT = 2.0       # target = entry + TARGET_MULT × opening-range width
EOD_FLAT_MIN = 915      # force square-off at/after 15:15 (=15*60+15); intraday only.
                        # SHARED by backtest (orb_trade EOD exit) AND live (emit
                        # SELLS + cron square-off) so live and backtest flatten at
                        # the SAME time — no CAGR drift from a mismatched exit.
                        # 15:15 = robust peak of the EOD-time sweep (CAGR +323% /
                        # Calmar 22.92); holding to the 15:25/close bar fades hard
                        # (+235%). Before broker MIS auto-square-off (~15:20).
MAX_PRICE = 1e9         # no price cap (liquid N500 names)
MIN_PRICE = 100.0       # drop sub-₹100 penny names: their tiny opening ranges (a
                        # ₹0.10 tick = a huge %) whipsaw the ORB into fake breakouts
                        # (e.g. IDEA ~₹8) — noise, not the momentum-trend edge.
# Data-freshness guards (live): refuse to act on stale data.
STALE_BAR_MAX_MIN = 15  # latest 5-min bar must be within this many minutes of now
DAILY_STALE_MAX_DAYS = 7  # daily ranking panel's last close must be within this

# Cost model (per side baked into round-trip). Realistic default; raise for a
# conservative read. The headline backtest uses SLIPPAGE=0.0015, ROUND_TRIP=0.0015.
SLIPPAGE = 0.0015       # fraction added to entry / subtracted from stop fills
ROUND_TRIP_COST = 0.0015  # brokerage + STT + exchange, round-trip


@dataclass
class OrbTrade:
    symbol: str
    day: str
    entry_time: str
    exit_time: str
    entry_px: float
    exit_px: float
    ret_pct: float          # net of slippage + round-trip cost
    reason: str             # "stop" | "target" | "eod"


def slot_qty(invested: float, select_top: int, entry_px: float) -> int:
    """Shares for ONE breakout slot = floor((invested / select_top) / entry_px).

    Per-slot reserve so each of the up-to-SELECT_TOP leaders gets its own equal
    cash slice (invested/SELECT_TOP) regardless of how many fire that cycle — a
    later breakout still has its slot's cash. Returns 0 on bad inputs.
    """
    try:
        if invested > 0 and select_top > 0 and entry_px > 0:
            return int((float(invested) / int(select_top)) / float(entry_px))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 0


def rank_momentum(daily_close: pd.DataFrame, di: int, eligible: set) -> List[str]:
    """Top-SELECT_TOP momentum leaders at daily row `di`.

    Ranks `eligible` symbols by LOOKBACK-trading-day return (close[di]/close[di-LB]-1),
    descending. `eligible` is the PIT index membership (plain symbols, no NSE: wrap).
    Returns plain symbols best-first. Shared by backtest and live (live passes the
    last daily row + today's official Nifty-500 list).

    Sub-MIN_PRICE names are dropped (penny stocks whipsaw the intraday ORB — see
    MIN_PRICE), using the ranking-day close as the price reference.
    """
    if di < LOOKBACK:
        return []
    now = daily_close.iloc[di]
    then = daily_close.iloc[di - LOOKBACK]
    rets = {}
    for s in daily_close.columns:
        bare = s.replace("NSE:", "").replace("-EQ", "")
        if bare not in eligible:
            continue
        a, b = now.get(s), then.get(s)
        if pd.notna(a) and pd.notna(b) and float(b) > 0 and float(a) >= MIN_PRICE:
            rets[bare] = float(a) / float(b) - 1.0
    return [s for s, _ in sorted(rets.items(), key=lambda kv: -kv[1])][:SELECT_TOP]


def opening_range(day_bars: pd.DataFrame,
                  min_post_bars: int = 5) -> Optional[Tuple[float, float]]:
    """(ORH, ORL) from the first OR_BARS 5-min bars, or None if not enough data.

    `min_post_bars` = how many bars beyond the opening range must already exist
    before a range is returned. The backtest keeps the historical default (5),
    so full-day results stay byte-identical. The LIVE scanner passes 1: it only
    needs a single post-range bar to test a breakout, so it can fire from the
    first scan after the opening range (~09:35) instead of waiting for 8 bars
    (~09:50) and missing early breakouts the backtest would take.
    """
    if len(day_bars) < OR_BARS + min_post_bars:
        return None
    orh = float(day_bars["h"].iloc[:OR_BARS].max())
    orl = float(day_bars["l"].iloc[:OR_BARS].min())
    if orh - orl <= 0:
        return None
    return orh, orl


def live_breakout(day_bars: pd.DataFrame, orh: float) -> bool:
    """True if any bar AFTER the opening range printed a high >= orh.

    Mirrors the backtest's intrabar breakout test (orb_trade scans
    ``g["h"].iloc[i] >= orh`` over range(OR_BARS, len)). The live scanner used
    to test only the LAST bar's CLOSE >= orh, so it missed intrabar breakouts
    that spiked above the range and closed back under — a divergence that made
    live take fewer entries than the backtested edge. Using the same high-based
    rule keeps live and backtest detecting the same breakouts.
    """
    for i in range(OR_BARS, len(day_bars)):
        if float(day_bars["h"].iloc[i]) >= orh:
            return True
    return False


def live_exit_reason(day_bars: pd.DataFrame, now_mins: int) -> Optional[str]:
    """The exit a HELD ORB position should take RIGHT NOW given today's bars-so-far.

    Returns "STOP" / "TARGET" / "EOD_FLAT" / None(hold), using the SAME rules and
    the SAME priority as orb_trade's exit loop so live and backtest exit on the
    same condition (test-locked: test_live_exit_reason_matches_orb_trade):
      - entry bar = first post-opening-range bar whose high >= ORH (the breakout);
      - then, in order, the first later bar with low <= ORL (STOP) or high >=
        ORH + TARGET_MULT*width (TARGET) — stop checked before target per bar,
        exactly like orb_trade;
      - else, if the clock has reached EOD_FLAT_MIN, EOD_FLAT (force square-off);
      - else None (still in the trade).
    `now_mins` is the wall-clock minute-of-day of this scan (for the EOD gate);
    the stop/target decision is purely from the bars, so it matches the backtest.
    """
    g = day_bars.reset_index(drop=True)
    rng = opening_range(g, min_post_bars=1)
    if rng is None:
        return "EOD_FLAT" if now_mins >= EOD_FLAT_MIN else None
    orh, orl = rng
    target = orh + TARGET_MULT * (orh - orl)
    entry_i = None
    for i in range(OR_BARS, len(g)):
        if float(g["h"].iloc[i]) >= orh:
            entry_i = i
            break
    if entry_i is not None:
        for j in range(entry_i + 1, len(g)):
            if float(g["l"].iloc[j]) <= orl:     # stop (same priority as orb_trade)
                return "STOP"
            if float(g["h"].iloc[j]) >= target:  # target
                return "TARGET"
    return "EOD_FLAT" if now_mins >= EOD_FLAT_MIN else None


def breakout_bar_index(day_bars: pd.DataFrame) -> Optional[int]:
    """Index of the FIRST post-opening-range bar whose HIGH breaks ORH before the
    entry cutoff (the bar a long would enter on), or None if no qualifying
    breakout. Same breakout + cutoff test as orb_trade's entry scan."""
    g = day_bars.reset_index(drop=True)
    rng = opening_range(g, min_post_bars=1)
    if rng is None:
        return None
    orh, _ = rng
    tmin = g["dt"].dt.hour * 60 + g["dt"].dt.minute
    for i in range(OR_BARS, len(g)):
        if tmin.iloc[i] >= ENTRY_CUTOFF_MIN:
            return None
        if float(g["h"].iloc[i]) >= orh:
            return i
    return None


def pick_leader(leaders_bars: List[Optional[pd.DataFrame]]) -> Optional[int]:
    """Choose the ONE leader to go ALL-IN on (single full-capital position).

    `leaders_bars` is ordered by momentum rank (index 0 = strongest); each entry
    is that leader's day bars (or None if no data). Returns the chosen index, or
    None if no leader broke out before the cutoff.

    Pick = the EARLIEST breakout (so it is achievable LIVE — we commit the moment
    the first leader breaks, never needing to know which others break later),
    with momentum RANK as the tiebreak when two break on the same bar. Shared by
    backtest and live so both pick the same name (no lookahead, no drift). ORB
    holds ONE position with the full model capital, not a 1/SELECT_TOP slice —
    reserving slots for leaders that may never break wastes ~45% of capital
    (≈49% of days only one of three fires).
    """
    best_key = None
    best_idx = None
    for rank, bars in enumerate(leaders_bars):
        if bars is None:
            continue
        bi = breakout_bar_index(bars)
        if bi is None:
            continue
        key = (bi, rank)          # earliest bar first, then strongest (lowest) rank
        if best_key is None or key < best_key:
            best_key, best_idx = key, rank
    return best_idx


def full_qty(invested: float, entry_px: float) -> int:
    """Shares for an ALL-IN single position = floor(invested / entry_px).
    ORB commits the full model capital to the one best-momentum leader that
    breaks out (single position), so there is no per-slot split. Returns 0 on
    bad inputs."""
    try:
        if invested > 0 and entry_px > 0:
            return int(float(invested) / float(entry_px))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 0


def orb_trade(day_bars: pd.DataFrame, symbol: str) -> Optional[OrbTrade]:
    """Run the long-only morning ORB on one stock's 5-min bars for one day.

    Bars: DataFrame with columns o/h/l/c and a `dt` (tz-aware) per 5-min bar,
    chronological. Returns an OrbTrade (net of slippage + cost) or None if no
    qualifying breakout before ENTRY_CUTOFF_MIN.
    """
    g = day_bars.reset_index(drop=True)
    rng = opening_range(g)
    if rng is None:
        return None
    orh, orl = rng
    width = orh - orl
    stop = orl
    target = orh + TARGET_MULT * width
    tmin = g["dt"].dt.hour * 60 + g["dt"].dt.minute
    times = g["dt"].dt.strftime("%H:%M").tolist()
    for i in range(OR_BARS, len(g)):
        if tmin.iloc[i] >= ENTRY_CUTOFF_MIN:
            return None  # past the morning cutoff with no breakout -> skip
        if float(g["h"].iloc[i]) >= orh:
            entry = orh * (1 + SLIPPAGE)            # chase the breakout up
            for j in range(i + 1, len(g)):
                if float(g["l"].iloc[j]) <= stop:   # stop (gap-through slip)
                    fill = stop * (1 - SLIPPAGE)
                    return OrbTrade(symbol, str(g["dt"].iloc[0].date()), times[i],
                                    times[j], round(entry, 2), round(fill, 2),
                                    round((fill / entry - 1 - ROUND_TRIP_COST) * 100, 3), "stop")
                if float(g["h"].iloc[j]) >= target:  # target (limit fill)
                    return OrbTrade(symbol, str(g["dt"].iloc[0].date()), times[i],
                                    times[j], round(entry, 2), round(target, 2),
                                    round((target / entry - 1 - ROUND_TRIP_COST) * 100, 3), "target")
                if tmin.iloc[j] >= EOD_FLAT_MIN:    # EOD square-off at 15:15 — LIVE
                    cpx = float(g["c"].iloc[j])     # PARITY: the live cron flattens
                    return OrbTrade(symbol, str(g["dt"].iloc[0].date()), times[i],  # at EOD_FLAT_MIN, so
                                    times[j], round(entry, 2), round(cpx, 2),       # the backtest must
                                    round((cpx / entry - 1 - ROUND_TRIP_COST) * 100, 3), "eod")  # exit at the same bar
            cpx = float(g["c"].iloc[-1])            # day ended before EOD_FLAT (half-session) -> last close
            return OrbTrade(symbol, str(g["dt"].iloc[0].date()), times[i],
                            times[-1], round(entry, 2), round(cpx, 2),
                            round((cpx / entry - 1 - ROUND_TRIP_COST) * 100, 3), "eod")
    return None
