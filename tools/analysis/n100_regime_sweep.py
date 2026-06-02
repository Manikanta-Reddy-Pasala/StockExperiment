"""momentum_n100_top5_max1 — INDEX-REGIME gate sweep (research only, NOT wired).

User idea (2026-06-03): NIFTY50 spent May-2025→Jan-2026 ABOVE its 200d SMA, then
fell BELOW from Feb-2026. Use the index regime to AVOID entries / sit out the
downtrend → fewer −12% stop-outs, lower DD, better CAGR.

Replicates the PRODUCTION n100 engine EXACTLY via n100_improve_sweep
(PIT NSE Nifty-100 eligible_at, LOOKBACK=15, RETAIN=3, monthly + mid-month 5pp
lead, from-entry −12% hard stop checked daily on the LOW). Adds an index gate:
  - index ABOVE its SMA (uptrend)  → behave normally.
  - index BELOW its SMA (downtrend) → ENTRY-gate: take no new entry. Optionally
    EXIT to cash while below (sit out bear phases).

Index = NSE:NIFTY50-INDEX (history from 2023-01), so the comparison window is
2023→2026. Feb-2026+ is the below-200SMA stretch the user highlighted.

Run:
  python3 tools/analysis/n100_regime_sweep.py --from 2023-02-01 --to 2026-05-22
"""
import sys, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok, mid_month_retain
from tools.models.momentum_n100_top5_max1.strategy import (
    LOOKBACK, RETAIN, MIDMONTH_LEAD, STOP_PCT, build_calendar)
from tools.analysis.n100_improve_sweep import (
    load_panels, atr_panel, make_rank, _metrics)

INDEX_SYM = "NSE:NIFTY50-INDEX"


def load_index(eng, start, end, ds="fyers"):
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT date, close FROM historical_data "
            "WHERE symbol=:s AND date BETWEEN :a AND :b AND data_source=:ds ORDER BY date"
        ), c, params={"s": INDEX_SYM, "a": start - timedelta(days=400), "b": end, "ds": ds})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float)


def regime_sim(dates, cl, hi, lo, atr, calendar, rank_at, midret_at, *,
               capital, start, end, idx_a, idx_sma_a,
               entry_stop_pct=STOP_PCT, retain=RETAIN, midmonth_lead=MIDMONTH_LEAD,
               regime_gate=False, regime_exit=False):
    """Production daily-MTM single-position walk (−12% from-entry stop + mid-month)
    PLUS an index-regime gate. regime_gate=True blocks new entries while the index
    is below its SMA; regime_exit=True also liquidates the held name to cash while
    below (and stays out until the index recovers)."""
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    nav_marks = [capital]; cash = capital
    hold = None; qty = 0; entry_px = 0.0
    trades = []

    def idx_up(di):
        a, b = idx_a.iloc[di], idx_sma_a.iloc[di]
        if pd.isna(a) or pd.isna(b):
            return True   # unknown regime → don't over-block
        return a >= b

    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_marks.append(cash + (qty * px if hold and px else 0.0))

        # production −12% from-entry hard stop (daily, on the LOW)
        if hold and qty > 0 and entry_stop_pct and entry_px > 0:
            lvl = entry_px * (1 - entry_stop_pct)
            low = float(lo[hold].iloc[di]) if pd.notna(lo[hold].iloc[di]) else px
            if lvl > 0 and low is not None and low <= lvl:
                cash += qty * lvl; trades.append({"pnl": qty * lvl - qty * entry_px})
                hold = None; qty = 0; entry_px = 0.0

        # regime EXIT: liquidate while index below its SMA (sit out the downtrend)
        if regime_exit and hold and qty > 0 and not idx_up(di) and px:
            cash += qty * px; trades.append({"pnl": qty * px - qty * entry_px})
            hold = None; qty = 0; entry_px = 0.0

        kind = cal.get(d)
        if kind is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        top = ranked[0]
        if kind == "mid":
            if not midmonth_lead_ok(hold, midret_at(di), midmonth_lead):
                continue
            if decide_rotation(hold, ranked, retain_top_n=mid_month_retain(True, retain)).is_noop:
                continue
        else:
            if decide_rotation(hold, ranked, retain_top_n=retain).is_noop:
                continue
        # ENTRY gate: in a downtrend, don't open a NEW position. If we'd rotate,
        # sell the old name (it left the retain band) but stay in cash.
        block_entry = (regime_gate or regime_exit) and not idx_up(di)
        if hold and qty > 0:
            sx = float(cl[hold].iloc[di]); cash += qty * sx
            trades.append({"pnl": qty * sx - qty * entry_px})
            hold = None; qty = 0; entry_px = 0.0
        if block_entry:
            continue
        bx = float(cl[top].iloc[di])
        if bx > 0:
            q = int(cash / bx)
            if q >= 1 and q * bx <= cash:
                cash -= q * bx; qty = q; hold = top; entry_px = bx
    final = cash + (qty * float(cl[hold].iloc[-1]) if hold else 0.0)
    return _metrics(final, capital, trades, nav_marks, start, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2023-02-01")
    ap.add_argument("--to", dest="end", default="2026-05-22")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    a = ap.parse_args()
    start = date.fromisoformat(a.start); end = date.fromisoformat(a.end)

    eng = _get_engine()
    print(f"Loading panels {start}..{end} (LOOKBACK={LOOKBACK} RETAIN={RETAIN} "
          f"stop=-{int(STOP_PCT*100)}% midlead={MIDMONTH_LEAD}) ...")
    cl, hi, lo, sma20 = load_panels(eng, start, end)
    atr = atr_panel(cl, hi, lo)
    dates = cl.index
    cal = build_calendar(dates, start, end, mid_check=True)
    rank_at, midret_at = make_rank(dates, cl)
    idx = load_index(eng, start, end)
    idx_a = idx.reindex(dates).ffill()

    def run(name, win, **kw):
        idx_sma_a = idx.rolling(win).mean().reindex(dates).ffill()
        m = regime_sim(dates, cl, hi, lo, atr, cal, rank_at, midret_at,
                       capital=a.capital, start=start, end=end,
                       idx_a=idx_a, idx_sma_a=idx_sma_a, **kw)
        calmar = (m['cagr'] / m['dd']) if m['dd'] else 0.0
        print(f"  {name:<40}{m['cagr']:>8}{m['dd']:>8}{calmar:>8.2f}{m['ret']:>9}{m['trades']:>8}{m['wr']:>6}")

    print(f"\n{'scenario':<42}{'CAGR%':>8}{'DD%':>8}{'Calmar':>8}{'ret%':>9}{'trades':>8}{'WR%':>6}")
    print("-" * 80)
    # baseline = current production: −12% stop, NO regime
    run("baseline (prod, -12% stop, no regime)", 200, regime_gate=False, regime_exit=False)
    for win in (200, 100, 50):
        run(f"regime ENTRY-gate (idxSMA{win})", win, regime_gate=True, regime_exit=False)
        run(f"regime EXIT+gate (idxSMA{win})",  win, regime_exit=True)
    # also show entry-gate / exit WITHOUT the -12% stop (does regime replace the stop?)
    print("  " + "-" * 70)
    run("regime EXIT idxSMA200, NO -12% stop", 200, regime_exit=True, entry_stop_pct=0.0)
    run("regime EXIT idxSMA100, NO -12% stop", 100, regime_exit=True, entry_stop_pct=0.0)


if __name__ == "__main__":
    main()
