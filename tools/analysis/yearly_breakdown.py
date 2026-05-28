"""Per-calendar-year breakdown of a model's trade ledger.

For a backtest run that produced ``trade_ledger.json`` (one row per closed
trade with ``entry_date`` / ``exit_date`` / ``pnl`` / ``cap_after``) and
``summary.json`` (initial capital + final NAV), compute year-by-year:

  - trades closed in the year
  - realized P&L sum
  - year-end cash (cap_after of the last closing trade in that year)
  - year return % vs the prior year's end cash (or initial capital for year 1)
  - rough max drawdown within the year (peak-to-trough on cap_after marks
    inside that year — daily MTM unavailable from a trade ledger, so this
    underestimates true intra-trade drawdowns).

This is intentionally a POST-RUN analyzer that lives outside the model
backtests (which only emit headline metrics + the trade ledger) so the same
script can be pointed at any rotation model's output. Built for the 10yr
backtest appendix where headline 10yr CAGR is less informative than the
year-by-year breakdown.

Usage:
    python tools/analysis/yearly_breakdown.py \\
        --ledger exports/bt10yr/n100/trade_ledger.json \\
        --summary exports/bt10yr/n100/summary.json
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
from datetime import date


def load(ledger_path: Path, summary_path: Path):
    """Load ledger + summary; return (trades_sorted_by_exit, initial_capital,
    final_nav, open_position_or_None)."""
    trades = json.loads(ledger_path.read_text())
    summary = json.loads(summary_path.read_text())
    # exit_date is the closing event — sort by it so the cumulative cap_after
    # walks forward in event order.
    trades = sorted(trades, key=lambda t: t.get("exit_date") or "")
    return (trades,
            float(summary.get("capital") or 0),
            float(summary.get("final_nav") or 0),
            summary.get("open_position"))


def breakdown(trades, initial_cap, final_nav, open_pos):
    """Group trades by exit-year. Returns list of dicts, one per calendar year."""
    by_year = defaultdict(list)
    for t in trades:
        y = int((t.get("exit_date") or "0000")[:4])
        if y > 1900:
            by_year[y].append(t)
    if not by_year:
        return []

    out = []
    years = sorted(by_year.keys())
    prev_end_cap = initial_cap
    for y in years:
        ts = by_year[y]
        n_trades = len(ts)
        wins = sum(1 for t in ts if (t.get("pnl") or 0) > 0)
        losses = sum(1 for t in ts if (t.get("pnl") or 0) < 0)
        realized = sum(float(t.get("pnl") or 0) for t in ts)
        # Year-end cash = cap_after of the last trade closed in the year.
        # cap_after is the running cash post-sell, before the next BUY consumes
        # it. Last-trade-of-year cap_after is a reasonable year-end NAV proxy
        # (excluding any open position carried into the next year).
        end_cap = float(ts[-1].get("cap_after") or prev_end_cap + realized)
        # Year return = end_cap / start_cap - 1. start_cap = prior year's end
        # (or initial_capital for the first year).
        ret_pct = ((end_cap / prev_end_cap) - 1) * 100 if prev_end_cap > 0 else 0.0
        # Within-year peak-to-trough on cap_after marks. Underestimates real
        # MTM drawdowns (a held position drawing down mid-year doesn't show as
        # cap_after — only realized exits do).
        caps = [float(t.get("cap_after") or 0) for t in ts]
        peak = max([prev_end_cap] + caps) if caps else prev_end_cap
        trough = peak
        for c in caps:
            if c > peak:
                peak = c
            if c < trough:
                trough = c
            # rolling DD
        # Recompute rolling DD properly (peak resets as it climbs).
        running_peak = prev_end_cap
        max_dd_pct = 0.0
        for c in caps:
            if c > running_peak:
                running_peak = c
            dd = (running_peak - c) / running_peak * 100 if running_peak > 0 else 0
            if dd > max_dd_pct:
                max_dd_pct = dd
        out.append({
            "year": y,
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "realized_pnl": round(realized, 0),
            "start_cap": round(prev_end_cap, 0),
            "end_cap": round(end_cap, 0),
            "return_pct": round(ret_pct, 2),
            "max_dd_pct_within_year": round(max_dd_pct, 2),
        })
        prev_end_cap = end_cap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="path to trade_ledger.json")
    ap.add_argument("--summary", required=True, help="path to summary.json")
    ap.add_argument("--out", default=None, help="write JSON breakdown to this path")
    a = ap.parse_args()
    trades, cap0, final_nav, open_pos = load(Path(a.ledger), Path(a.summary))
    rows = breakdown(trades, cap0, final_nav, open_pos)
    if not rows:
        print("no trades found in ledger"); return
    # Pretty table
    hdr = (f"{'year':>4s} {'trades':>6s} {'WR%':>5s} "
           f"{'realized':>12s} {'start':>12s} {'end':>12s} "
           f"{'ret%':>8s} {'maxDD%':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['year']:>4d} {r['trades']:>6d} {r['win_rate_pct']:>5.1f} "
              f"{r['realized_pnl']:>12,.0f} {r['start_cap']:>12,.0f} "
              f"{r['end_cap']:>12,.0f} {r['return_pct']:>+8.2f} "
              f"{r['max_dd_pct_within_year']:>7.2f}")
    print(f"\ninitial_cap={cap0:,.0f}  final_nav={final_nav:,.0f}  "
          f"open_pos={open_pos.get('sym') if open_pos else 'None'}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
