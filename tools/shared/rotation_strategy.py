"""Shared PURE decision core for the momentum-rotation models.

Used by: momentum_n100_top5_max1, momentum_pseudo_n100_adv, n20_daily_large_only.

WHY THIS EXISTS
---------------
Each model used to implement its entry/exit rule TWICE — once in backtest.py
(vectorized history loop) and once in live_signal.py (single-day signal emit).
The two copies drifted: the backtests rotated on top-1 while live held through
top-5, and n100 live was stateless and never rotated at all. The published
backtest CAGRs therefore did NOT describe live behaviour.

These pure functions are the single source of truth for the rotation rule.
Both backtest.py and live_signal.py import and call them, so the decision can
no longer diverge. Pure: no I/O, no DB, no pandas — just the rule. Easy to
unit-test and to assert parity between the two execution paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class RotationDecision:
    """What to do this rebalance. sell/buy are symbols (or None)."""
    sell: Optional[str]
    buy: Optional[str]
    reason: str

    @property
    def is_noop(self) -> bool:
        return self.sell is None and self.buy is None


def mid_month_retain(is_mid_month: bool, full_retain: int) -> int:
    """Retention band to use for a rotation decision.

    SINGLE source of truth so live and the backtest engine can't drift: the
    mid-month leg always rotates on a top-1 band (retain=1), the full-month
    (1st-trading-day) leg uses the model's configured band. Used by both
    backtest_engine.run_rotation_backtest and each model's live_signal.
    """
    return 1 if is_mid_month else int(full_retain)


def decide_rotation(held: Optional[str], ranked: Sequence[str],
                    retain_top_n: int = 1) -> RotationDecision:
    """Single-position top-N retention rotation (max_concurrent=1).

    Args:
        held: currently-held symbol, or None if flat.
        ranked: symbols sorted by 30-day return DESC, AFTER all universe
            filters (uptrend / max-price / cap), best first.
        retain_top_n: keep holding while `held` stays within the top-N of
            `ranked`; rotate out when it drops below rank-N. 1 == pure top-1
            rotation (sell the moment held is no longer rank-1). This is the
            knob that must match between backtest and live.

    Returns:
        RotationDecision. Entry always targets rank-1 (`ranked[0]`).
    """
    if not ranked:
        return RotationDecision(None, None, "no candidates")
    top = ranked[0]
    retain = set(ranked[:max(1, retain_top_n)])

    # Held still inside the retention band -> keep it, do nothing.
    if held is not None and held in retain:
        return RotationDecision(None, None, f"hold (in top-{retain_top_n})")

    sell = held if (held is not None and held not in retain) else None
    buy = top if held != top else None
    if sell and buy:
        return RotationDecision(sell, buy, "rotate to rank-1")
    if buy:
        return RotationDecision(None, buy, "enter rank-1")
    return RotationDecision(sell, None, "exit (no replacement)")


def midmonth_lead_ok(held: Optional[str],
                     ranked_with_ret: Sequence[Tuple[str, float]],
                     lead_pct: float) -> bool:
    """n100 mid-month gate. True if a mid-cycle rotation is allowed.

    Args:
        held: currently-held symbol, or None.
        ranked_with_ret: [(symbol, ret30d_pct), ...] sorted DESC.
        lead_pct: minimum lead (percentage points) the new rank-1 must have
            over the held stock's 30d return to justify rotating mid-cycle.

    Rules: already holding rank-1 -> no rotation. Held dropped out of the
    ranking -> allow rotation. Otherwise rotate only if rank-1 leads held by
    >= lead_pct.
    """
    if not ranked_with_ret:
        return False
    top_sym, top_ret = ranked_with_ret[0]
    if held == top_sym:
        return False
    held_ret = next((r for s, r in ranked_with_ret if s == held), None)
    if held_ret is None:
        return True
    return (top_ret - held_ret) >= lead_pct
