"""momentum_retest_n500 improvement sweep — ATR stop / regime filter / vol-sizing.

Research only (NOT wired). Clones retest's EXACT multi-position daily walk
(backtest.run: monthly top-K retest-entry, retain-band rotation) and layers the
three ideas the user proposed, so each can be measured on CAGR + Max DD:

  5.1 Per-holding ATR stop — trailing (peak - k*ATR) or from-entry (entry-k*ATR),
      checked DAILY (the edge is good; the problem is downside).
  5.2 Market-regime switch — block NEW entries when the breadth proxy (equal-
      weight mean close of the loaded universe) is below its 200d SMA (cut gross
      exposure / sit in cash in risk-off regimes).
  5.3 Volatility-based position sizing — size each slot inversely to the name's
      ATR% (risk parity) instead of equal cash/slots, so penny-rockets add
      return without dominating risk.

baseline must reproduce production retest backtest numbers.

Run:
  python3 tools/analysis/retest_improve_sweep.py --from 2021-03-01 --to 2026-05-31
  python3 tools/analysis/retest_improve_sweep.py --from 2022-01-01 --to 2023-12-31
  python3 tools/analysis/retest_improve_sweep.py --from 2025-03-01 --to 2026-05-31
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
from tools.models.momentum_retest_n500 import strategy as S

COST = 0.0015
ATR_WIN = 14


def load_panels(eng, start, end):
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high").ffill()
    lo = df.pivot(index="date", columns="symbol", values="low").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    adv20, sma200, ema20 = S.indicators(cl, adv_rs)
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()]).groupby(level=0).max()
    atr = tr.rolling(ATR_WIN).mean()
    proxy = cl.mean(axis=1)                       # equal-weight breadth proxy
    proxy200 = proxy.rolling(200).mean()
    return cl, hi, lo, adv20, sma200, ema20, atr, proxy, proxy200, S.load_smallcap()


def walk(dates, cl, lo, adv20, sma200, ema20, atr, proxy, proxy200, smallcap, *,
         start, end, capital, atr_trail=None, atr_entry=None,
         regime_block=False, vol_sizing=False):
    i0 = dates.searchsorted(pd.Timestamp(start)); i1 = dates.searchsorted(pd.Timestamp(end), side="right")
    rebals = set(); y, m = start.year, start.month
    while True:
        fut = dates[dates >= pd.Timestamp(y, m, 1)]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0].date() >= start:
            rebals.add(fut[0])
        m += 1
        if m > 12:
            m = 1; y += 1
    cash = capital; pos = {}; watch = []; trades = []; navs = []
    warm = max(S.LOOKBACK, S.SMA_LONG)

    def _exit(s, di, px, reason):
        nonlocal cash
        p = pos[s]
        cash += p["qty"] * px * (1 - COST)
        trades.append({"pnl": p["qty"] * px * (1 - COST) - p["qty"] * p["entry"], "reason": reason})
        del pos[s]

    for di in range(i0, i1):
        d = dates[di]; crow = cl.iloc[di]
        # --- daily per-holding ATR stop (5.1) ---
        if (atr_trail or atr_entry) and pos:
            for s in list(pos.keys()):
                p = pos[s]
                px = float(crow.get(s)) if pd.notna(crow.get(s)) else None
                if px is None:
                    continue
                p["peak"] = max(p.get("peak", p["entry"]), px)
                a = atr[s].iloc[di] if s in atr.columns else None
                if a is None or pd.isna(a) or a <= 0:
                    continue
                lvl = (p["peak"] - atr_trail * float(a)) if atr_trail else (p["entry"] - atr_entry * float(a))
                if lvl > 0:
                    dlow = float(lo[s].iloc[di]) if pd.notna(lo[s].iloc[di]) else px
                    if dlow <= lvl:
                        _exit(s, di, lvl, "STOP")

        if d in rebals and di >= warm:
            rk = S.rank_targets(cl, adv20, sma200, smallcap, di)
            elig = eligible_at("n500", d.date())
            rk = [s for s in rk if s.replace("NSE:", "").replace("-EQ", "") in elig]
            retset = set(rk[:S.RETAIN])
            for s in list(pos.keys()):
                if s not in retset:
                    _exit(s, di, float(cl[s].iloc[di]), "ROTATE")
            watch = [s for s in rk[:S.K] if s not in pos]

        # --- entries (with optional regime block 5.2 + vol sizing 5.3) ---
        regime_ok = True
        if regime_block and pd.notna(proxy200.iloc[di]):
            regime_ok = float(proxy.iloc[di]) >= float(proxy200.iloc[di])
        if watch and di >= warm and regime_ok:
            slots = S.K - len(pos)
            for s in list(watch):
                if s in pos or slots <= 0:
                    continue
                px, ev = crow.get(s), ema20[s].iloc[di]
                if S.is_retest(px, ev):
                    px = float(px)
                    budget = cash / max(1, slots)
                    if vol_sizing:
                        a = atr[s].iloc[di] if s in atr.columns else None
                        atrpct = (float(a) / px) if (a is not None and pd.notna(a) and a > 0 and px > 0) else None
                        if atrpct and atrpct > 0:
                            # scale 0.5x..1.5x by inverse vol vs a 3% reference
                            scale = max(0.5, min(1.5, 0.03 / atrpct))
                            budget *= scale
                    q = int(budget / px)
                    if q >= 1 and q * px * (1 + COST) <= cash:
                        cash -= q * px * (1 + COST)
                        pos[s] = {"qty": q, "entry": px, "in": d.date().isoformat(), "peak": px}
                        slots -= 1; watch.remove(s)
        mv = cash + sum(p["qty"] * float(crow.get(s)) for s, p in pos.items() if pd.notna(crow.get(s)))
        navs.append(mv)

    nav = pd.Series(navs)
    yrs = (end - start).days / 365.25
    final = float(nav.iloc[-1]) if len(nav) else capital
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100 if len(nav) > 1 else 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0); losses = sum(1 for t in trades if t["pnl"] < 0)
    return {"cagr": round(cagr, 1), "dd": round(mdd, 1), "ret": round((final / capital - 1) * 100, 0),
            "trades": len(trades), "wr": round(wins / max(1, wins + losses) * 100, 0)}


SCENARIOS = [
    ("baseline",              dict()),
    ("atr_trail_2.0",         dict(atr_trail=2.0)),
    ("atr_trail_3.0",         dict(atr_trail=3.0)),
    ("atr_fromentry_2.5",     dict(atr_entry=2.5)),
    ("atr_fromentry_3.0",     dict(atr_entry=3.0)),
    ("regime_block",          dict(regime_block=True)),
    ("vol_sizing",            dict(vol_sizing=True)),
    ("regime+atr_trail3",     dict(regime_block=True, atr_trail=3.0)),
    ("regime+atr_entry3",     dict(regime_block=True, atr_entry=3.0)),
    ("volsize+atr_trail3",    dict(vol_sizing=True, atr_trail=3.0)),
    ("regime+vol+atr_entry3", dict(regime_block=True, vol_sizing=True, atr_entry=3.0)),
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
    print(f"Loading panels {start}..{end} (K={S.K} RETAIN={S.RETAIN} LOOKBACK={S.LOOKBACK}) ...")
    cl, hi, lo, adv20, sma200, ema20, atr, proxy, proxy200, smallcap = load_panels(eng, start, end)
    dates = cl.index
    print(f"\n{'scenario':<24}{'CAGR%':>8}{'DD%':>8}{'ret%':>8}{'trades':>8}{'WR%':>6}")
    print("-" * 62)
    for name, kw in SCENARIOS:
        m = walk(dates, cl, lo, adv20, sma200, ema20, atr, proxy, proxy200, smallcap,
                 start=start, end=end, capital=a.capital, **kw)
        print(f"{name:<24}{m['cagr']:>8}{m['dd']:>8}{m['ret']:>8}{m['trades']:>8}{m['wr']:>6}")


if __name__ == "__main__":
    main()
