"""momentum_n100_top5_max1 — NEW-lever sweep (research only, NOT wired).

Beyond the stop/floor/lookback levers already in n100_improve_sweep, test the
structural changes most likely to cut DD without killing CAGR:

  1. MULTI-POSITION (top-K equal-weight) — the classic diversification DD cut.
     Current model is max-1 (all-in one name). K=2/3/5 spread the risk.
  2. VOL-ADJUSTED momentum — rank by LOOKBACK-ret / daily-return-vol (emerging's
     edge): prefers SMOOTH trends over spiky ones, usually lower DD.
  3. ACCELERATION filter — only enter names whose recent 10d return is > 0
     (don't buy a 15d-winner that's already rolling over).

All keep n100's PIT Nifty-100 universe (eligible_at), monthly + mid-month
calendar, and the production −12% from-entry daily stop. K=1 plain = production
baseline.

Run:
  python3 tools/analysis/n100_multipos_sweep.py --from 2021-03-01 --to 2026-05-31
"""
import sys, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import eligible_at
from tools.models.momentum_n100_top5_max1.strategy import (
    LOOKBACK, STOP_PCT, INDEX_NAME, build_calendar)
from tools.analysis.n100_improve_sweep import load_panels, atr_panel, _metrics

VOL_WIN = 60
ACCEL_WIN = 10


def make_ranker(dates, cl, *, lb=LOOKBACK, vol_adj=False, accel=False):
    rets_d = cl.pct_change()
    vol = rets_d.rolling(VOL_WIN).std()

    def pit(di):
        elig = eligible_at(INDEX_NAME, dates[di].date())
        return [f"NSE:{s}-EQ" for s in elig
                if f"NSE:{s}-EQ" in cl.columns and pd.notna(cl[f"NSE:{s}-EQ"].iloc[di])]

    def rank_at(di):
        if di < max(lb, VOL_WIN):
            return []
        univ = pit(di)
        if not univ:
            return []
        mom = cl.iloc[di].reindex(univ) / cl.iloc[di - lb].reindex(univ) - 1
        score = mom.copy()
        if vol_adj:
            v = vol.iloc[di].reindex(univ)
            score = mom / v.replace(0, np.nan)
        score = score.dropna()
        if accel:
            acc = cl.iloc[di].reindex(score.index) / cl.iloc[di - ACCEL_WIN].reindex(score.index) - 1
            score = score[acc.reindex(score.index) > 0]
        return list(score.sort_values(ascending=False).index)
    return rank_at


def multi_sim(dates, cl, lo, calendar, rank_at, *, capital, start, end,
              K=1, retain_buffer=0, entry_stop_pct=STOP_PCT):
    """Top-K equal-weight momentum portfolio. Rebalance on calendar days to the
    top-K (hold a name while it stays in top-(K+retain_buffer)); per-name −12%
    from-entry daily stop on the LOW. K=1, buffer=0 == production max-1."""
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    cash = capital
    port = {}   # sym -> {"qty":, "entry":}
    nav_marks = [capital]; trades = []
    retain_n = K + retain_buffer

    def price(di, s):
        v = cl[s].iloc[di]
        return float(v) if pd.notna(v) else None

    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        mtm = cash + sum((h["qty"] * (price(di, s) or 0)) for s, h in port.items())
        nav_marks.append(mtm)

        # per-name −12% from-entry hard stop (daily, on the LOW)
        if entry_stop_pct:
            for s in list(port.keys()):
                h = port[s]; lvl = h["entry"] * (1 - entry_stop_pct)
                low = float(lo[s].iloc[di]) if pd.notna(lo[s].iloc[di]) else price(di, s)
                if lvl > 0 and low is not None and low <= lvl:
                    cash += h["qty"] * lvl
                    trades.append({"pnl": h["qty"] * lvl - h["qty"] * h["entry"]})
                    del port[s]

        kind = cal.get(d)
        if kind is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        target = ranked[:K]
        retain_set = set(ranked[:retain_n])
        # SELL holdings that fell out of the retain band
        for s in list(port.keys()):
            if s not in retain_set:
                p = price(di, s)
                if p:
                    cash += port[s]["qty"] * p
                    trades.append({"pnl": port[s]["qty"] * p - port[s]["qty"] * port[s]["entry"]})
                    del port[s]
        # BUY top-K names not yet held, equal-weight on current MTM/K
        want = [s for s in target if s not in port]
        free_slots = K - len(port)
        want = want[:max(0, free_slots)]
        if want:
            mtm = cash + sum((h["qty"] * (price(di, s) or 0)) for s, h in port.items())
            budget = mtm / K
            for s in want:
                p = price(di, s)
                if p and p > 0:
                    q = int(min(budget, cash) / p)
                    if q >= 1 and q * p <= cash:
                        cash -= q * p; port[s] = {"qty": q, "entry": p}
    final = cash + sum((h["qty"] * (price(len(dates) - 1, s) or 0)) for s, h in port.items())
    return _metrics(final, capital, trades, nav_marks, start, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2021-03-01")
    ap.add_argument("--to", dest="end", default="2026-05-31")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    a = ap.parse_args()
    start = date.fromisoformat(a.start); end = date.fromisoformat(a.end)
    eng = _get_engine()
    print(f"Loading panels {start}..{end} ...")
    cl, hi, lo, sma20 = load_panels(eng, start, end)
    dates = cl.index
    cal = build_calendar(dates, start, end, mid_check=True)

    plain = make_ranker(dates, cl)
    voladj = make_ranker(dates, cl, vol_adj=True)
    accel = make_ranker(dates, cl, accel=True)
    voladj_accel = make_ranker(dates, cl, vol_adj=True, accel=True)

    rows = [
        ("K1 plain (PRODUCTION)",      plain,         dict(K=1)),
        ("K2 plain",                   plain,         dict(K=2, retain_buffer=1)),
        ("K3 plain",                   plain,         dict(K=3, retain_buffer=2)),
        ("K5 plain",                   plain,         dict(K=5, retain_buffer=2)),
        ("K1 vol-adj",                 voladj,        dict(K=1)),
        ("K2 vol-adj",                 voladj,        dict(K=2, retain_buffer=1)),
        ("K3 vol-adj",                 voladj,        dict(K=3, retain_buffer=2)),
        ("K1 accel",                   accel,         dict(K=1)),
        ("K1 vol-adj+accel",           voladj_accel,  dict(K=1)),
        ("K3 vol-adj+accel",           voladj_accel,  dict(K=3, retain_buffer=2)),
    ]
    print(f"\n{'scenario':<26}{'CAGR%':>8}{'DD%':>8}{'Calmar':>8}{'ret%':>9}{'trades':>8}{'WR%':>6}")
    print("-" * 74)
    for name, rk, kw in rows:
        m = multi_sim(dates, cl, lo, cal, rk, capital=a.capital, start=start, end=end, **kw)
        cm = (m['cagr'] / m['dd']) if m['dd'] else 0
        print(f"{name:<26}{m['cagr']:>8}{m['dd']:>8}{cm:>8.2f}{m['ret']:>9}{m['trades']:>8}{m['wr']:>6}")


if __name__ == "__main__":
    main()
