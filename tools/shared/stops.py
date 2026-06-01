"""Shared hard-stop helpers — used by BOTH the backtests and the live
stop-check paths so live and backtest can never drift.

Two from-ENTRY hard stops (fixed level anchored at entry; cut genuine
breakdowns below entry, never whipsaw a winner out):

  * ATR-from-entry  : level = entry_px - mult * ATR(win)   (pseudo, emerging)
  * fixed-% from-entry: level = entry_px * (1 - pct)        (n100)

A stop is HIT when the day's LOW pierces the level (intraday fill at the level).
All helpers are pure (no IO) and fail-safe (None / False on bad inputs) so they
are trivially unit-testable and safe to call from a live cron.
"""
from __future__ import annotations

import pandas as pd


def atr_latest(high: pd.Series, low: pd.Series, close: pd.Series, win: int = 14):
    """Latest ATR (simple mean of True Range over `win`). None if insufficient."""
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


def atr_stop_level(entry_px, atr_val, mult):
    """Hard-stop price = entry_px - mult*ATR. None on bad/<=0 inputs."""
    try:
        if entry_px and atr_val and atr_val > 0 and mult and mult > 0:
            lvl = float(entry_px) - float(mult) * float(atr_val)
            return lvl if lvl > 0 else None
    except (TypeError, ValueError):
        pass
    return None


def atr_stop_hit(entry_px, atr_val, day_low, mult):
    """(hit, level): True if day_low pierced entry_px - mult*ATR."""
    lvl = atr_stop_level(entry_px, atr_val, mult)
    if lvl is None or day_low is None:
        return False, lvl
    try:
        return (float(day_low) <= lvl), lvl
    except (TypeError, ValueError):
        return False, lvl


def fixed_stop_level(entry_px, pct):
    """Hard-stop price = entry_px * (1 - pct). None on bad/<=0 inputs."""
    try:
        if entry_px and entry_px > 0 and pct and pct > 0:
            lvl = float(entry_px) * (1.0 - float(pct))
            return lvl if lvl > 0 else None
    except (TypeError, ValueError):
        pass
    return None


def fixed_stop_hit(entry_px, day_low, pct):
    """(hit, level): True if day_low pierced entry_px * (1 - pct)."""
    lvl = fixed_stop_level(entry_px, pct)
    if lvl is None or day_low is None:
        return False, lvl
    try:
        return (float(day_low) <= lvl), lvl
    except (TypeError, ValueError):
        return False, lvl
