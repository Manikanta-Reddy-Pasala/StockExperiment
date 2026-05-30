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


def _cap_at(sym: str, entry_date: Optional[str]) -> str:
    """Point-in-time cap class (large/mid/small) of `sym` at its entry date.
    PIT so a name that was large-cap when traded (in Nifty 100) reads 'large'
    even if it has since been demoted. Fail-soft to 'unknown'."""
    try:
        from tools.shared.market_cap import classify_pit
        return classify_pit(sym, date.fromisoformat(entry_date)) if entry_date else "unknown"
    except Exception:
        return "unknown"


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
    per_year: Dict = field(default_factory=dict)     # {year: {"ret_pct":..,"dd_pct":..}}


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
    stop_loss_pct: Optional[float] = None,
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
        # Per-position STOP-LOSS: if the held name has fallen >= stop_loss_pct
        # from entry, sell to cash and step aside this bar (caps tail losses).
        if stop_loss_pct and hold and qty > 0 and pd.notna(px_h):
            if float(px_h) <= entry_px * (1 - stop_loss_pct):
                sx = float(px_h); cap += qty * sx
                trades.append({
                    "sym": _label(hold), "entry_date": entry_date,
                    "exit_date": d.date().isoformat(), "qty": qty,
                    "entry_px": round(entry_px, 2), "exit_px": round(sx, 2),
                    "pnl": round(qty * sx - qty * entry_px, 0),
                    "ret_pct": round((sx / entry_px - 1) * 100, 2),
                    "cap_after": round(cap, 0), "exit_reason": "STOP",
                    "cap": _cap_at(hold, entry_date),
                })
                hold = None; qty = 0
                continue
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
                    "cap": _cap_at(hold, entry_date),
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
            "cap": _cap_at(hold, entry_date),
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
    nav_dates = nav_dates[:len(nav_marks)]
    # Year-wise breakdown: % return + intra-year MTM drawdown per calendar year,
    # from the MTM NAV marks (one per calendar step). Year ret = last/first - 1.
    per_year: Dict = {}
    if nav_dates:
        nser = pd.Series(nav_marks, index=pd.DatetimeIndex(nav_dates))
        for yy, g in nser.groupby(nser.index.year):
            if len(g) < 2:
                continue
            rl = g.cummax()
            per_year[int(yy)] = {
                "ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1),
            }
    return BacktestResult(
        final_nav=final, cagr_pct=cagr, max_dd_pct=mdd, calmar=calmar,
        trades=trades, wins=wins, losses=losses,
        win_rate_pct=wins / max(1, wins + losses) * 100,
        years=yrs, open_position=open_pos, max_dd_mtm_pct=mtm_dd,
        nav_dates=nav_dates, nav_values=nav_marks, per_year=per_year,
    )
