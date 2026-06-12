"""Shared PURE decision core for the breakout-swing model.

Used by: midcap_narrow_60d_breakout (backtest.py + live_signal.py).

Same motivation as rotation_strategy.py: the entry-qualification and the
exit rule were implemented twice (vectorized backtest loop vs single-day live
emit) and could drift. These pure functions are the single source of truth.
No I/O, no pandas — just the rule. Both paths import them.
"""
from __future__ import annotations

from typing import Optional, Tuple


def is_breakout(close: float, prior_high: float, sma_long: float,
                volume: float, vol_avg20: float, *, vol_mult: float) -> Tuple[bool, float]:
    """Entry qualification for one stock on one day.

    Qualifies when (ALL must hold):
      - close > prior N-day high (fresh breakout)
      - close > long SMA (Stage-2 uptrend)
      - volume >= vol_mult * 20-day average volume (volume surge)

    Returns (qualifies, vol_ratio). vol_ratio is volume/vol_avg20 (0 if no
    volume baseline) and is used by the caller to rank competing breakouts —
    highest vol_ratio wins.
    """
    if vol_avg20 <= 0:
        return False, 0.0
    vol_ratio = volume / vol_avg20
    qualifies = (close > prior_high and close > sma_long and vol_ratio >= vol_mult)
    return qualifies, vol_ratio


def breakout_exit_reason(entry_px: float, close: float, peak: float,
                         age_days: int, *, target_pct: float, stop_pct: float,
                         trail_pct: float, profit_trigger: float,
                         max_hold_days: int) -> Optional[str]:
    """Exit reason for an open breakout position, or None to keep holding.

    Precedence (first match wins):
      TARGET   — up >= target_pct from entry
      STOP     — down >= stop_pct from entry (catastrophe stop; skip if stop_pct<=0)
      TRAIL    — in profit >= profit_trigger AND down >= trail_pct from PEAK price
      MAX_HOLD — held >= max_hold_days

    `peak` is the highest close seen since entry. Trail measures the drop from
    that peak price (NOT a drop in the gain-number): e.g. peak +40% exits at
    +12%, not +30%. SMA20 exit is intentionally not modelled here (disabled).
    """
    ret_e = (close - entry_px) / entry_px if entry_px else 0.0
    ret_pk = (peak - close) / peak if peak > 0 else 0.0
    if ret_e >= target_pct:
        return "TARGET"
    if stop_pct > 0 and ret_e <= -stop_pct:
        return "STOP"
    if ret_e >= profit_trigger and ret_pk >= trail_pct:
        return "TRAIL"
    if age_days >= max_hold_days:
        return "MAX_HOLD"
    return None
