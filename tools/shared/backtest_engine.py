"""Shared backtest EXECUTION engine for the rotation models.

Separation of concerns (KISS):
  SELECTION  — each model supplies a `rank_at(di) -> [symbol, ...]` callback
               (its own universe + filters + 30d ranking). Model-specific.
  RULE       — `tools.shared.rotation_strategy.decide_rotation` decides hold vs
               rotate. Shared, pure. (Same function live_signal.py calls.)
  EXECUTION  — THIS engine: walks the rebalance calendar, applies the decision,
               tracks cash/position/trades, computes CAGR / MaxDD / Calmar.
               Shared across momentum_n100 / momentum_pseudo / n20_daily.

The engine owns NO strategy logic and NO data access — it only executes the
decision the shared rule produces against a price panel the model supplies.
This is the single execution path the 3 rotation backtests share, so their
accounting + metrics can no longer diverge from each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok


def _label(sym: str) -> str:
    return sym.replace("NSE:", "").replace("-EQ", "")


@dataclass
class BacktestResult:
    final_nav: float
    cagr_pct: float
    max_dd_pct: float          # realized: drawdown of rebal-day cap_after (n100/pseudo)
    calmar: float              # cagr / max_dd_pct (realized)
    trades: List[Dict]
    wins: int
    losses: int
    win_rate_pct: float
    years: float
    open_position: Optional[Dict] = field(default=None)
    max_dd_mtm_pct: float = 0.0   # daily mark-to-market drawdown (daily models, e.g. n20)
    nav_dates: List = field(default_factory=list)    # calendar-step timestamps
    nav_values: List = field(default_factory=list)   # MTM NAV at each step


def run_rotation_backtest(
    *,
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,                       # date x symbol close panel (ffilled)
    calendar: Sequence[Tuple[pd.Timestamp, str]],  # [(date, "full"|"mid"), ...]
    rank_at: Callable[[int], List[str]],       # di -> ranked symbols (best first)
    capital: float,
    start: date,
    end: date,
    retain_top_n: int = 1,
    midmonth_ret_at: Optional[Callable[[int], List[Tuple[str, float]]]] = None,
    midmonth_lead_pct: float = 5.0,
) -> BacktestResult:
    """Single-position rotation backtest. Entry/exit at same-day close, no fees
    (matches the existing rotation backtests). `calendar` kind "mid" applies the
    n100 mid-month lead gate; everything else is a normal "full" rebalance.
    """
    cap = capital
    hold: Optional[str] = None
    qty = 0
    entry_px = 0.0
    entry_date: Optional[str] = None
    trades: List[Dict] = []
    nav_marks: List[float] = [capital]   # MTM NAV going into each calendar step

    for d, kind in calendar:
        di = dates.get_loc(d)
        px_h = close[hold].iloc[di] if hold else None
        nav_marks.append(cap + (qty * float(px_h) if hold and pd.notna(px_h) else 0.0))
        ranked = rank_at(di)
        if not ranked:
            continue
        top = ranked[0]

        if kind == "mid":
            ret_list = midmonth_ret_at(di) if midmonth_ret_at else []
            if not midmonth_lead_ok(hold, ret_list, midmonth_lead_pct):
                continue
            if decide_rotation(hold, ranked, retain_top_n=1).is_noop:
                continue
            reason = "MIDCHECK"
        else:
            if decide_rotation(hold, ranked, retain_top_n).is_noop:
                continue
            reason = "ROTATE"

        # SELL leg
        if hold and qty > 0:
            sx = close[hold].iloc[di]
            if pd.notna(sx):
                sx = float(sx)
                proc = qty * sx
                cap += proc
                pnl = proc - qty * entry_px
                trades.append({
                    "sym": _label(hold),
                    "entry_date": entry_date,
                    "exit_date": d.date().isoformat(),
                    "qty": qty,
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(sx, 2),
                    "pnl": round(pnl, 0),
                    "ret_pct": round((sx / entry_px - 1) * 100, 2),
                    "cap_after": round(cap, 0),
                    "exit_reason": reason,
                })
                hold = None
                qty = 0

        # BUY leg (rank-1)
        bx = close[top].iloc[di]
        if pd.notna(bx):
            bx = float(bx)
            q = int(cap / bx)
            if q >= 1 and q * bx <= cap:
                cap -= q * bx
                qty = q
                hold = top
                entry_px = bx
                entry_date = d.date().isoformat()

    final = cap
    open_pos = None
    if hold:
        last = float(close[hold].iloc[-1])
        final = cap + qty * last
        open_pos = {
            "sym": _label(hold), "qty": qty, "entry_px": round(entry_px, 2),
            "entry_date": entry_date, "last_px": round(last, 2),
            "mtm_value": round(qty * last, 0),
            "unrealized_pnl": round(qty * (last - entry_px), 0),
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100

    peak = capital
    mdd = 0.0
    for t in trades:
        peak = max(peak, t["cap_after"])
        dd = (peak - t["cap_after"]) / peak * 100
        mdd = max(mdd, dd)
    calmar = cagr / max(0.01, mdd)

    nav_s = pd.Series(nav_marks)
    roll = nav_s.cummax()
    mtm_dd = float(((roll - nav_s) / roll).max()) * 100 if len(nav_s) > 1 else 0.0

    nav_dates = [pd.Timestamp(start)] + [d for d, _ in calendar]
    return BacktestResult(
        final_nav=final, cagr_pct=cagr, max_dd_pct=mdd, calmar=calmar,
        trades=trades, wins=wins, losses=losses,
        win_rate_pct=wins / max(1, wins + losses) * 100,
        years=yrs, open_position=open_pos, max_dd_mtm_pct=mtm_dd,
        nav_dates=nav_dates[:len(nav_marks)], nav_values=nav_marks,
    )
