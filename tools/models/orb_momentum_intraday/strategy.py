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
  3. ENTRY — go LONG when price breaks ABOVE ORH, but ONLY if that breakout
     happens before ENTRY_CUTOFF (10:00). Late breakouts are skipped (they have
     less room to run and a worse edge — morning-only nearly doubled the return).
     Long-only: we never short, because on up-momentum names the downside break
     mostly fails.
  4. STOP / TARGET — stop at ORL (one range-width below entry); target at
     entry-side ORH + TARGET_MULT (2.0) × width.
  5. EXIT — whichever comes first: stop, target, or a forced square-off at
     EOD_FLAT (15:10). Most trades exit flat at EOD (riding the intraday
     trend); the rest stop/target. ZERO overnight risk.

Position sizing (live): per-slot reserve = invested_amount / SELECT_TOP (e.g.
₹30k/3 = ₹10k per leader). Each breakout takes ONE slot, so a single morning
breakout deploys only its ₹10k and leaves the other slots' cash free for later
breakouts (the incremental nature of ORB entries). Already-held names are not
re-bought.

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
EOD_FLAT_MIN = 910      # force square-off at/after 15:10 (=15*60+10); intraday only
MAX_PRICE = 1e9         # no price cap (liquid N500 names)
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
        if pd.notna(a) and pd.notna(b) and float(b) > 0:
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
            cpx = float(g["c"].iloc[-1])            # EOD square-off at the close
            return OrbTrade(symbol, str(g["dt"].iloc[0].date()), times[i],
                            times[-1], round(entry, 2), round(cpx, 2),
                            round((cpx / entry - 1 - ROUND_TRIP_COST) * 100, 3), "eod")
    return None
