"""Standalone backtest: RETEST MOMENTUM (momentum_retest_n500).

Offline research/validation path. The SELECTION + ENTRY logic lives in the
shared core `strategy.py` (rank_targets / is_retest / params) — the SAME code
live_signal.py uses, so backtest and live cannot drift. This file only owns the
offline walk: load panels, build the monthly calendar, replay buy/sell, score.

Strategy (see strategy.py / SUMMARY.md): monthly pick top-2 momentum leaders
from the top-120-ADV liquid N500 pool; buy each within 20% of the 20-EMA; hold
while in the top-4 rank. Reproduces (2023-05..2026-05, true-PIT, net 0.15%/side):
+146% CAGR / 20.9% DD / Calmar 7.0 / positive every year (2026-05-30 K2 config).
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.models.momentum_retest_n500 import strategy as S

COST = 0.0015
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END = date(2026, 5, 12)
DEFAULT_CAP = 1_000_000.0


def load_panels(eng, start, end):
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    adv20, sma200, ema20 = S.indicators(cl, adv_rs)
    return cl, adv20, sma200, ema20, S.load_smallcap()


def run(start, end, capital, out_dir=None):
    eng = _get_engine()
    cl, adv20, sma200, ema20, smallcap = load_panels(eng, start, end)
    dates = cl.index
    print(f"N500 pool loaded: {len(cl)} days x {len(cl.columns)} symbols")
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
    cash = capital; pos = {}; watch = []; trades = []; navs = []; nd = []
    warm = max(S.LOOKBACK, S.SMA_LONG)
    for di in range(i0, i1):
        d = dates[di]; crow = cl.iloc[di]
        if d in rebals and di >= warm:
            rk = S.rank_targets(cl, adv20, sma200, smallcap, di)
            retset = set(rk[:S.RETAIN])
            for s in list(pos.keys()):                       # exit: out of retain band
                if s not in retset:
                    px = float(cl[s].iloc[di]); p = pos[s]
                    proc = p["qty"] * px * (1 - COST)
                    cash += proc
                    cap_after = cash + sum(
                        pp["qty"] * float(cl[ss].iloc[di])
                        for ss, pp in pos.items()
                        if ss != s and pd.notna(cl[ss].iloc[di]))
                    trades.append({
                        "sym": s.replace("NSE:", "").replace("-EQ", ""),
                        "entry_date": p["in"], "exit_date": d.date().isoformat(),
                        "qty": p["qty"], "entry_px": round(p["entry"], 2),
                        "exit_px": round(px, 2),
                        "pnl": round(proc - p["qty"] * p["entry"], 0),
                        "ret_pct": round((px / p["entry"] - 1) * 100, 2),
                        "cap_after": round(cap_after, 0),
                        "exit_reason": "ROTATE",
                    })
                    del pos[s]
            watch = [s for s in rk[:S.K] if s not in pos]
        if watch and di >= warm:                              # retest entry (daily)
            slots = S.K - len(pos)
            for s in list(watch):
                if s in pos or slots <= 0:
                    continue
                px, ev = crow.get(s), ema20[s].iloc[di]
                if S.is_retest(px, ev):
                    px = float(px)
                    q = int((cash / max(1, slots)) / px)
                    if q >= 1:
                        cash -= q * px * (1 + COST)
                        pos[s] = {"qty": q, "entry": px, "in": d.date().isoformat()}
                        slots -= 1; watch.remove(s)
        mv = cash + sum(p["qty"] * float(crow.get(s)) for s, p in pos.items()
                        if pd.notna(crow.get(s)))
        navs.append(mv); nd.append(d)

    nav = pd.Series(navs, index=pd.DatetimeIndex(nd))
    yrs = (end - start).days / 365.25
    final = float(nav.iloc[-1])
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100
    years = {}
    for yy, g in nav.groupby(nav.index.year):
        if len(g) < 2:
            continue
        rl = g.cummax()
        years[int(yy)] = {"ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                          "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1)}
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    last = cl.iloc[i1 - 1]
    open_positions = [{
        "sym": s.replace("NSE:", "").replace("-EQ", ""),
        "qty": p["qty"], "entry_px": round(p["entry"], 2),
        "entry_date": p["in"], "last_px": round(float(last[s]), 2),
        "mtm_value": round(p["qty"] * float(last[s]), 0),
        "unrealized_pnl": round(p["qty"] * (float(last[s]) - p["entry"]), 0),
    } for s, p in pos.items() if pd.notna(last.get(s))]
    print("\n## RESULTS (RETEST MOMENTUM, true-PIT, net of cost)")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%   Max DD: {mdd:.2f}%   Calmar: {cagr/max(0.01,mdd):.2f}")
    print(f"  Trades: {len(trades)} (WR {100*wins/max(1,len(trades)):.0f}%)")
    for yy in sorted(years):
        print(f"    {yy}: {years[yy]['ret_pct']:+.1f}%  (intra-yr DD {years[yy]['dd_pct']:.1f}%)")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        (out_dir / "summary.json").write_text(json.dumps({
            "model": "momentum_retest_n500", "start": start.isoformat(),
            "end": end.isoformat(), "years": round(yrs, 3),
            "capital": capital, "final_nav": round(final, 0),
            "total_return_pct": round((final/capital-1)*100, 2),
            "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
            "calmar": round(cagr/max(0.01, mdd), 2),
            "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "open_positions": open_positions, "per_year": years,
            "params": {"top_n": S.TOPN, "K": S.K, "retain": S.RETAIN,
                       "lookback": S.LOOKBACK, "mom_floor": S.MOM_FLOOR,
                       "max_price": S.MAX_PRICE, "retest_band": [S.RETEST_LO, S.RETEST_HI]},
        }, indent=2))
    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None)
