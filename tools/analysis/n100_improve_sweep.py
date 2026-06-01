"""momentum_n100_top5_max1 improvement sweep — price floor / stops / SMA20 / cooldown.

Research only (NOT wired). Mirrors pseudo_improve_sweep but with n100's exact
selection (PIT NSE Nifty-100 via eligible_at, LOOKBACK=15, RETAIN=3, mid-month
lead check ON) on a DAILY-MTM single-position simulator so intra-month stops
can be checked daily. baseline must reproduce the production engine.

Run:
  python3 tools/analysis/n100_improve_sweep.py --from 2021-03-01 --to 2026-05-31
  python3 tools/analysis/n100_improve_sweep.py --from 2025-03-01 --to 2026-05-31
"""
import sys, argparse
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union, eligible_at
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok, mid_month_retain
from tools.models.momentum_n100_top5_max1.strategy import (
    LOOKBACK, RETAIN, MIDMONTH_LEAD, INDEX_NAME, build_calendar)

ATR_WIN = 14


def load_panels(eng, start, end, ds="fyers"):
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union(INDEX_NAME))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,high,low,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=400), "b": end, "ds": ds})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high").ffill()
    lo = df.pivot(index="date", columns="symbol", values="low").ffill()
    sma20 = cl.rolling(20).mean()
    return cl, hi, lo, sma20


def atr_panel(cl, hi, lo, win=ATR_WIN):
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()]).groupby(level=0).max()
    return tr.rolling(win).mean()


def make_rank(dates, cl):
    def pit(di):
        elig = eligible_at(INDEX_NAME, dates[di].date())
        return [f"NSE:{s}-EQ" for s in elig
                if f"NSE:{s}-EQ" in cl.columns and pd.notna(cl[f"NSE:{s}-EQ"].iloc[di])]

    def rank_at(di):
        if di < LOOKBACK:
            return []
        univ = pit(di)
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    def midret_at(di):
        univ = pit(di)
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        return [(s, float(rk[s]) * 100) for s in rk.index]
    return rank_at, midret_at


def daily_sim(dates, cl, hi, lo, atr, sma20, calendar, rank_at, midret_at, *,
              capital, start, end, retain=RETAIN, midmonth_lead=MIDMONTH_LEAD,
              price_floor=0.0, atr_from_entry=None, entry_stop_pct=None,
              sma20_trail=False, cooldown_cycles=0):
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    nav_marks = [capital]; cash = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    trades = []; cooldown = {}; cycle = 0

    def ranked_at(di):
        r = rank_at(di)
        if price_floor > 0:
            r = [s for s in r if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) >= price_floor]
        if cooldown_cycles:
            r = [s for s in r if cooldown.get(s, -1) < cycle]
        return r

    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_marks.append(cash + (qty * px if hold and px else 0.0))

        if hold and qty > 0:
            stop_level = None
            if atr_from_entry and entry_px > 0:
                a = atr[hold].iloc[di] if hold in atr.columns else None
                if a is not None and pd.notna(a) and a > 0:
                    stop_level = entry_px - atr_from_entry * float(a)
            elif entry_stop_pct and entry_px > 0:
                stop_level = entry_px * (1 - entry_stop_pct)
            exited = False
            if stop_level is not None and stop_level > 0:
                day_low = float(lo[hold].iloc[di]) if pd.notna(lo[hold].iloc[di]) else px
                if day_low is not None and day_low <= stop_level:
                    cash += qty * stop_level
                    trades.append(_t(hold, entry_px, stop_level, qty))
                    if cooldown_cycles and stop_level < entry_px:
                        cooldown[hold] = cycle + cooldown_cycles
                    hold = None; qty = 0; entry_px = 0.0; exited = True
            if not exited and sma20_trail and px is not None:
                s20 = sma20[hold].iloc[di] if hold in sma20.columns else None
                if s20 is not None and pd.notna(s20) and px < float(s20):
                    cash += qty * px; trades.append(_t(hold, entry_px, px, qty))
                    if cooldown_cycles and px < entry_px:
                        cooldown[hold] = cycle + cooldown_cycles
                    hold = None; qty = 0; entry_px = 0.0; exited = True

        kind = cal.get(d)
        if kind is None:
            continue
        cycle += 1
        ranked = ranked_at(di)
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
        if hold and qty > 0:
            sx = float(cl[hold].iloc[di]); cash += qty * sx
            trades.append(_t(hold, entry_px, sx, qty))
            if cooldown_cycles and sx < entry_px:
                cooldown[hold] = cycle + cooldown_cycles
            hold = None; qty = 0
        bx = float(cl[top].iloc[di])
        if bx > 0:
            q = int(cash / bx)
            if q >= 1 and q * bx <= cash:
                cash -= q * bx; qty = q; hold = top; entry_px = bx
    final = cash + (qty * float(cl[hold].iloc[-1]) if hold else 0.0)
    return _metrics(final, capital, trades, nav_marks, start, end)


def _t(sym, ep, xp, qty):
    return {"pnl": qty * xp - qty * ep}


def _metrics(final, capital, trades, nav_marks, start, end):
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    nav = pd.Series(nav_marks); roll = nav.cummax()
    mdd = float(((roll - nav) / roll).max()) * 100 if len(nav) > 1 else 0.0
    return {"cagr": round(cagr, 1), "dd": round(mdd, 1), "ret": round((final / capital - 1) * 100, 0),
            "trades": len(trades), "wr": round(wins / max(1, wins + losses) * 100, 0)}


SCENARIOS = [
    ("baseline",            dict()),
    ("floor_100",           dict(price_floor=100)),
    ("floor_150",           dict(price_floor=150)),
    ("atr2.0_fromentry",    dict(atr_from_entry=2.0)),
    ("atr2.5_fromentry",    dict(atr_from_entry=2.5)),
    ("atr3.0_fromentry",    dict(atr_from_entry=3.0)),
    ("stop_-10%",           dict(entry_stop_pct=0.10)),
    ("stop_-12%",           dict(entry_stop_pct=0.12)),
    ("stop_-14%",           dict(entry_stop_pct=0.14)),
    ("stop_-16%",           dict(entry_stop_pct=0.16)),
    ("sma20_trail",         dict(sma20_trail=True)),
    ("cooldown_2",          dict(cooldown_cycles=2)),
    ("atr2.5+cooldown2",    dict(atr_from_entry=2.5, cooldown_cycles=2)),
    ("atr3.0+cooldown2",    dict(atr_from_entry=3.0, cooldown_cycles=2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2021-03-01")
    ap.add_argument("--to", dest="end", default="2026-05-31")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
    eng = _get_engine()
    print(f"Loading panels {start}..{end} (LOOKBACK={LOOKBACK} RETAIN={RETAIN} midlead={MIDMONTH_LEAD}) ...")
    cl, hi, lo, sma20 = load_panels(eng, start, end)
    atr = atr_panel(cl, hi, lo)
    dates = cl.index
    rank_at, midret_at = make_rank(dates, cl)
    calendar = build_calendar(dates, start, end, mid_check=True)
    print(f"\n{'scenario':<22}{'CAGR%':>8}{'DD%':>8}{'ret%':>8}{'trades':>8}{'WR%':>6}")
    print("-" * 60)
    for name, kw in SCENARIOS:
        m = daily_sim(dates, cl, hi, lo, atr, sma20, calendar, rank_at, midret_at,
                      capital=a.capital, start=start, end=end, **kw)
        print(f"{name:<22}{m['cagr']:>8}{m['dd']:>8}{m['ret']:>8}{m['trades']:>8}{m['wr']:>6}")


if __name__ == "__main__":
    main()
