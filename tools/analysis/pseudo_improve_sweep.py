"""pseudo_n100_adv improvement sweep — price floor / stops / SMA20-trail / cooldown.

Research only (NOT wired into the model). Tests the rules the user proposed
(mirroring the emerging price-floor + ATR-stop work) against the current
momentum_pseudo_n100_adv config, on a DAILY-MTM single-position simulator so
intra-month stops/trails can be checked every trading day.

Selection is REPLICATED EXACTLY from tools/models/momentum_pseudo_n100_adv/
backtest.py (yearly fixed-anchor top-100-ADV PIT N500, drop smallcap, MAX_PRICE
<= 3000, rank by 30d return, rank-1 rotation). The "baseline" scenario must
reproduce the production run_rotation_backtest numbers (printed as a sanity
check).

Scenarios tested per window:
  - baseline (no floor, no stop)
  - price floor Rs.100 / Rs.150 at entry
  - ATR-from-entry hard stop k*ATR (k=2.0/2.5/3.0) — the emerging win
  - fixed from-entry stop -10% / -12%
  - SMA20 trailing exit (close < 20d SMA)
  - cooldown: no re-entry into a name for 1/2 cycles after a RED exit
  - a couple of combos (floor150 + atr2.5, floor150 + cooldown2)

Run:
  python3 tools/analysis/pseudo_improve_sweep.py --from 2021-03-01 --to 2026-05-31
  python3 tools/analysis/pseudo_improve_sweep.py --from 2025-03-01 --to 2026-05-31
"""
import sys, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union, eligible_at
from tools.shared.rotation_strategy import decide_rotation
from tools.models.momentum_pseudo_n100_adv import strategy as S
from tools.models.momentum_pseudo_n100_adv.strategy import build_calendar
from tools.models.momentum_pseudo_n100_adv.backtest import _SMALLCAP

ATR_WIN = 14


def load_panels(eng, start, end, data_source="fyers"):
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end, "ds": data_source})
    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high").ffill()
    lo = df.pivot(index="date", columns="symbol", values="low").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(S.ADV_WIN).mean()
    sma200 = cl.rolling(200).mean()
    sma20 = cl.rolling(20).mean()
    return cl, hi, lo, adv20, sma200, sma20


def atr_panel(cl, hi, lo, win=ATR_WIN):
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()]).groupby(level=0).max()
    return tr.rolling(win).mean()


def build_universes(dates, adv20, start, end):
    year_starts = [pd.Timestamp(yr, S.UNIVERSE_ANCHOR_MONTH, S.UNIVERSE_ANCHOR_DAY)
                   for yr in range(start.year - 1, end.year + 1)]
    yu = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0:
            continue
        di = dates.get_loc(fut[0])
        elig500 = eligible_at("n500", ys.date())
        pit = adv20.iloc[di].dropna().sort_values(ascending=False)
        pit = pit[[s for s in pit.index if s.replace("NSE:", "").replace("-EQ", "") in elig500]]
        top = pit.head(S.UNIV_SIZE).index.tolist()
        yu[ys] = [s for s in top if s.replace("NSE:", "").replace("-EQ", "") not in _SMALLCAP]
    return year_starts, yu


def make_rank_at(dates, cl, sma200, year_starts, yu):
    def pick_universe(d):
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:
                chosen = ys
        return yu.get(chosen, [])

    def rank_at(di):
        if di < max(S.LOOKBACK, 200):
            return []
        univ = pick_universe(dates[di])
        if S.SMA_GATE:
            up = sma200.iloc[di] < cl.iloc[di]
            univ = [s for s in univ if bool(up.get(s, False))]
        univ = [s for s in univ if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= S.MAX_PRICE]
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - S.LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)
    return rank_at


def daily_sim(dates, cl, hi, lo, atr, sma20, calendar, rank_at, *, capital, start, end,
              retain=1, price_floor=0.0, atr_from_entry=None, entry_stop_pct=None,
              sma20_trail=False, cooldown_cycles=0):
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    nav_marks = [capital]
    cash = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    trades = []; nav_by_day = []
    cooldown = {}        # symbol -> cycle index until which re-entry is blocked
    cycle = 0

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
        nav_today = cash + (qty * px if hold and px else 0.0)
        nav_marks.append(nav_today)
        nav_by_day.append((d, nav_today))

        # daily exits (from-entry hard stop / fixed % / SMA20 trail)
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
                    trades.append(_trade(hold, entry_date, d, qty, entry_px, stop_level, cash, "STOP"))
                    if cooldown_cycles and stop_level < entry_px:
                        cooldown[hold] = cycle + cooldown_cycles
                    hold = None; qty = 0; entry_px = 0.0; exited = True
            if not exited and sma20_trail and px is not None:
                s20 = sma20[hold].iloc[di] if hold in sma20.columns else None
                if s20 is not None and pd.notna(s20) and px < float(s20):
                    cash += qty * px
                    trades.append(_trade(hold, entry_date, d, qty, entry_px, px, cash, "SMA20"))
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
        if decide_rotation(hold, ranked, retain_top_n=retain).is_noop:
            continue
        if hold and qty > 0:
            sx = float(cl[hold].iloc[di])
            cash += qty * sx
            trades.append(_trade(hold, entry_date, d, qty, entry_px, sx, cash, "ROTATE"))
            if cooldown_cycles and sx < entry_px:
                cooldown[hold] = cycle + cooldown_cycles
            hold = None; qty = 0
        bx = float(cl[top].iloc[di])
        if bx > 0:
            q = int(cash / bx)
            if q >= 1 and q * bx <= cash:
                cash -= q * bx; qty = q; hold = top; entry_px = bx
                entry_date = d.date().isoformat()

    final = cash + (qty * float(cl[hold].iloc[-1]) if hold else 0.0)
    return _metrics(final, capital, trades, nav_marks, start, end, nav_by_day)


def _trade(sym, ed, d, qty, ep, xp, cap_after, reason):
    return {"sym": sym.replace("NSE:", "").replace("-EQ", ""), "entry_date": ed,
            "exit_date": d.date().isoformat(), "qty": qty, "entry_px": round(ep, 2),
            "exit_px": round(xp, 2), "pnl": round(qty * xp - qty * ep, 0),
            "ret_pct": round((xp / ep - 1) * 100, 2) if ep else 0.0,
            "exit_reason": reason}


def _metrics(final, capital, trades, nav_marks, start, end, nav_by_day=None):
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
    ("atr3.5_fromentry",    dict(atr_from_entry=3.5)),
    ("atr4.0_fromentry",    dict(atr_from_entry=4.0)),
    ("stop_-10%",           dict(entry_stop_pct=0.10)),
    ("stop_-12%",           dict(entry_stop_pct=0.12)),
    ("sma20_trail",         dict(sma20_trail=True)),
    ("cooldown_1",          dict(cooldown_cycles=1)),
    ("cooldown_2",          dict(cooldown_cycles=2)),
    ("floor150+atr2.5",     dict(price_floor=150, atr_from_entry=2.5)),
    ("floor150+cooldown2",  dict(price_floor=150, cooldown_cycles=2)),
    ("floor150+sma20",      dict(price_floor=150, sma20_trail=True)),
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
    print(f"Loading panels {start}..{end} ...")
    cl, hi, lo, adv20, sma200, sma20 = load_panels(eng, start, end)
    atr = atr_panel(cl, hi, lo)
    dates = cl.index
    year_starts, yu = build_universes(dates, adv20, start, end)
    rank_at = make_rank_at(dates, cl, sma200, year_starts, yu)
    calendar = build_calendar(dates, start, end, mid_check=False)

    print(f"\n{'scenario':<22}{'CAGR%':>8}{'DD%':>8}{'ret%':>8}{'trades':>8}{'WR%':>6}")
    print("-" * 60)
    for name, kw in SCENARIOS:
        m = daily_sim(dates, cl, hi, lo, atr, sma20, calendar, rank_at,
                      capital=a.capital, start=start, end=end, **kw)
        print(f"{name:<22}{m['cagr']:>8}{m['dd']:>8}{m['ret']:>8}{m['trades']:>8}{m['wr']:>6}")


if __name__ == "__main__":
    main()
