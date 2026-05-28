"""Per-calendar-year breakdown with TRUE mark-to-market at each year-end.

Background — the original tools/analysis/yearly_breakdown.py used the
cash-only `cap_after` of the last closed trade in each year as that year's
ending NAV. That breaks for any position that spans the year boundary (held
on Dec 31 with no closing trade in that year), because the open position's
MTM gain isn't in cash yet. It's especially wrong for low-churn models
(midcap holds 60-120 days; trades often straddle Dec 31) and for the final
partial year of a backtest that still holds an open position.

This analyzer walks the trade ledger chronologically by entry_date,
reconstructs the day-by-day position state (cash + held qty), and at each
year-end (or backtest end date) queries `historical_data` for the close
price of the held symbol on the year-end day. NAV_at_year_end = cash + qty *
close_on_that_day. Then the year return is NAV_end / NAV_start - 1.

Needs DB access — run inside the trading_system_app container, or anywhere
the prod Postgres is reachable through `src.models.database`.

Schema notes:
  - rotation model trade_ledger entries: {sym, entry_date, exit_date, qty,
    entry_px, exit_px, pnl, cap_after, ...}
  - midcap rotation entries: same shape (verified 2026-05-28).
  - summary.open_position may use "sym" (rotation models) OR "symbol"
    (midcap). Both keys are tolerated.

Usage:
    python tools/analysis/yearly_breakdown_mtm.py \\
        --ledger exports/bt10yr/n100/trade_ledger.json \\
        --summary exports/bt10yr/n100/summary.json
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _close_on_or_before(eng, sym, ymd):
    """SELECT close FROM historical_data on or before ymd for `sym`.

    `sym` is the plain ticker (e.g. "ADANIPOWER"); the table stores the
    full Fyers form ("NSE:ADANIPOWER-EQ"). Walks back to the most recent
    available bar so a year-end falling on a weekend / holiday still finds
    Friday's close.
    """
    from sqlalchemy import text
    fyers_sym = f"NSE:{sym}-EQ" if not sym.startswith("NSE:") else sym
    with eng.connect() as c:
        r = c.execute(text(
            "SELECT close FROM historical_data WHERE symbol=:s AND date<=:d "
            "ORDER BY date DESC LIMIT 1"
        ), {"s": fyers_sym, "d": ymd}).fetchone()
    return float(r[0]) if r else None


def _open_sym_qty_entry(open_pos):
    """summary.open_position may use 'sym' (rotation) or 'symbol' (midcap)."""
    if not open_pos:
        return None, 0, 0.0, None
    sym = open_pos.get("sym") or open_pos.get("symbol")
    qty = int(open_pos.get("qty") or 0)
    entry_px = float(open_pos.get("entry_px") or 0)
    entry_date = open_pos.get("entry_date")
    return sym, qty, entry_px, entry_date


def compute(ledger_path, summary_path):
    trades = json.loads(Path(ledger_path).read_text())
    summary = json.loads(Path(summary_path).read_text())
    capital = float(summary["capital"])
    final_nav = float(summary["final_nav"])
    end_date = date.fromisoformat(summary["end"])
    start_date = date.fromisoformat(summary["start"])
    op_sym, op_qty, op_entry_px, op_entry_date = _open_sym_qty_entry(
        summary.get("open_position"))

    # Sort trades by entry_date so we walk state forward in time.
    trades = sorted(trades, key=lambda t: t.get("entry_date") or "")

    # Reconstruct cash + held intervals.
    # Each closed trade T: state during [T.entry_date, T.exit_date) =
    # holding T.qty of T.sym; cash = cap_after(T) - qty*exit_px.
    # Between trades: flat; cash = cap_after of the most recent closed trade
    # (or `capital` before any trade fired).
    intervals = []
    for t in trades:
        ed = date.fromisoformat(t["entry_date"])
        xd = date.fromisoformat(t["exit_date"])
        cap_after = float(t["cap_after"])
        qty = int(t["qty"])
        exit_px = float(t["exit_px"])
        cash_during = cap_after - qty * exit_px  # cash while position held
        intervals.append({
            "entry": ed, "exit": xd, "sym": t["sym"],
            "qty": qty, "cash_during": cash_during, "cap_after": cap_after,
            "entry_px": float(t["entry_px"]),
        })
    # Open position interval (still held at summary.end_date)
    if op_sym and op_qty > 0 and op_entry_date:
        # cap_before(open) = last closed trade's cap_after (or capital).
        cap_before_open = intervals[-1]["cap_after"] if intervals else capital
        intervals.append({
            "entry": date.fromisoformat(op_entry_date),
            "exit": end_date + timedelta(days=1),  # still open
            "sym": op_sym, "qty": op_qty,
            "cash_during": cap_before_open - op_qty * op_entry_px,
            "cap_after": None, "entry_px": op_entry_px, "_open": True,
        })

    # DB engine.
    from tools.shared.ohlcv_cache import _get_engine
    eng = _get_engine()
    if eng is None:
        raise SystemExit("DB unreachable — run inside the app container")

    def nav_on(target_date):
        """True MTM NAV on `target_date`: cash + held_qty * close_on_date."""
        # Find interval containing target_date (entry <= target < exit).
        for iv in intervals:
            if iv["entry"] <= target_date < iv["exit"]:
                close = _close_on_or_before(eng, iv["sym"], target_date)
                if close is None:
                    # No price data → fall back to cash_during (no MTM).
                    return iv["cash_during"]
                return iv["cash_during"] + iv["qty"] * close
        # Between intervals — find the most recent closed trade before target
        # and use its cap_after. (No held position, all in cash.)
        prior_caps = [iv["cap_after"] for iv in intervals
                      if iv["exit"] <= target_date and iv.get("cap_after") is not None]
        return prior_caps[-1] if prior_caps else capital

    # Per-year boundaries: Dec 31 of each year in [start.year, end.year-1].
    # Final boundary = summary.end (which already has final_nav).
    rows = []
    prev_end_nav = capital
    prev_end_date = start_date  # not used, just bootstrap
    for yr in range(start_date.year, end_date.year + 1):
        # Boundary = Dec 31 (or the backtest end if it's within this year).
        boundary = date(yr, 12, 31) if yr != end_date.year else end_date
        year_end_nav = (final_nav if (yr == end_date.year and boundary == end_date)
                        else nav_on(boundary))
        # Realized P&L = sum of trades CLOSING in this year (informational).
        realized = sum(float(t.get("pnl") or 0) for t in trades
                       if (t.get("exit_date") or "").startswith(f"{yr}-"))
        n_trades = sum(1 for t in trades
                       if (t.get("exit_date") or "").startswith(f"{yr}-"))
        wins = sum(1 for t in trades
                   if (t.get("exit_date") or "").startswith(f"{yr}-")
                   and float(t.get("pnl") or 0) > 0)
        losses = sum(1 for t in trades
                     if (t.get("exit_date") or "").startswith(f"{yr}-")
                     and float(t.get("pnl") or 0) < 0)
        ret_pct = ((year_end_nav / prev_end_nav) - 1) * 100 if prev_end_nav > 0 else 0
        rows.append({
            "year": yr, "trades": n_trades, "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "realized_pnl": round(realized, 0),
            "start_nav": round(prev_end_nav, 0),
            "end_nav": round(year_end_nav, 0),
            "return_pct": round(ret_pct, 2),
        })
        prev_end_nav = year_end_nav
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, summary = compute(a.ledger, a.summary)
    hdr = (f"{'year':>4s} {'trades':>6s} {'WR%':>5s} {'realized':>12s} "
           f"{'start_nav':>13s} {'end_nav':>13s} {'ret%':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['year']:>4d} {r['trades']:>6d} {r['win_rate_pct']:>5.1f} "
              f"{r['realized_pnl']:>12,.0f} {r['start_nav']:>13,.0f} "
              f"{r['end_nav']:>13,.0f} {r['return_pct']:>+8.2f}")
    op = summary.get("open_position") or {}
    op_label = op.get("sym") or op.get("symbol") or "None"
    print(f"\ninitial_cap={summary['capital']:,.0f}  "
          f"final_nav={summary['final_nav']:,.0f}  open_pos={op_label}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
