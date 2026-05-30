"""Two NON-momentum models, tested on 10yr prod data (real bears incl COVID),
to address momentum's 82% bear drawdown:

  1) FAST DE-RISK overlay on n100 momentum — go to cash when the large-cap basket
     proxy is > DD_STOP below its trailing peak (reacts in DAYS, unlike a 200d
     SMA). Variants 8/10/12/15%. Re-enter when the proxy climbs back to within
     RECLAIM of its peak.
  2) MEAN-REVERSION (Connors-style, long-only) — buy the most oversold n100 name
     (RSI2 < RSI_BUY) that is still in an uptrend (close > 200d SMA), exit when
     close > 5d SMA or after MAX_HOLD days. Tested full-period AND sideways-only.

Baseline = n100 single-position momentum (same engine).

Point DATABASE_URL at prod (10yr) via tunnel before running:
  DATABASE_URL=postgresql://trader:...@localhost:15432/trading_system \
    python3 tools/analysis/regime_models.py --start 2017-01-01 --end 2026-05-12
"""
from __future__ import annotations
import sys, argparse, warnings
from datetime import date, timedelta
import numpy as np, pandas as pd
from sqlalchemy import text
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import eligible_at
import tools.models.momentum_n100_top5_max1.backtest as N100
from tools.analysis.regime_research import regime_proxy, BASKET

COST = 0.0015


def _metrics(nav, cap0, start, end):
    nav = nav.dropna()
    if len(nav) < 2:
        return None
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1] / cap0) ** (365.25 / max(1, days)) - 1
    dd = ((nav.cummax() - nav) / nav.cummax()).max()
    return cagr * 100, dd * 100


def fast_derisk(start, end, proxy, dd_stop, reclaim):
    """n100 momentum, but flat-to-cash while proxy drawdown-from-peak > dd_stop."""
    peak = proxy.cummax()
    ddp = (peak - proxy) / peak
    # state machine: once dd breaches dd_stop -> risk_off until dd <= reclaim
    risk_off = pd.Series(False, index=proxy.index)
    off = False
    for d in proxy.index:
        if not off and ddp[d] > dd_stop:
            off = True
        elif off and ddp[d] <= reclaim:
            off = False
        risk_off[d] = off

    orig = N100.run_rotation_backtest; holder = {}

    def patched(*a, **kw):
        dates = kw["dates"]; base = kw["rank_at"]

        def wrapped(di):
            if bool(risk_off.get(dates[di], False)):
                return []
            return base(di)
        kw["rank_at"] = wrapped
        res = orig(*a, **kw); holder["res"] = res
        return res

    N100.run_rotation_backtest = patched
    try:
        N100.run(start, end, 1_000_000.0, mid_month_check=False, retain_top_n=3)
    finally:
        N100.run_rotation_backtest = orig
    r = holder["res"]
    return r.cagr_pct, r.max_dd_pct, r.calmar, len(r.trades), \
        r.wins / max(1, r.wins + r.losses) * 100


def mean_reversion(start, end, reg, sideways_only, rsi_buy=10, max_hold=10):
    """Connors-style long-only MR on PIT n100. Single position for comparability."""
    syms = [f"NSE:{s}-EQ" for s in
            sorted(set().union(*[eligible_at("n100", date(y, 1, 1)) for y in range(2017, 2027)]))]
    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,close FROM historical_data WHERE symbol=ANY(:s) "
            "AND date BETWEEN :a AND :b AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    op = df.pivot(index="date", columns="symbol", values="open")
    sma200 = cl.rolling(200).mean(); sma5 = cl.rolling(5).mean()
    # RSI(2)
    delta = cl.diff()
    gain = delta.clip(lower=0).rolling(2).mean(); loss = (-delta.clip(upper=0)).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    dates = cl.index
    i0 = dates.searchsorted(pd.Timestamp(start)); i1 = dates.searchsorted(pd.Timestamp(end), "right")
    cap = 1_000_000.0; pos = None; navs = []; nd = []; trades = 0; wins = 0
    for di in range(i0, i1):
        d = dates[di]
        if pos:                                            # exit check
            held = pos["sym"]; px = cl[held].iloc[di]
            held_days = di - pos["di"]
            if pd.notna(px) and (px > sma5[held].iloc[di] or held_days >= max_hold):
                cap += pos["qty"] * float(px) * (1 - COST)
                if float(px) > pos["entry"]:
                    wins += 1
                pos = None
        if pos is None:                                    # entry
            ok = (reg.get(d, "SIDEWAYS") == "SIDEWAYS") if sideways_only else True
            if ok and di >= 200:
                # eligible PIT + oversold (RSI2) + uptrend (close > 200d SMA)
                elig = {f"NSE:{x}-EQ" for x in eligible_at("n100", d.date())}
                cand = []
                for s in elig:
                    if s not in cl.columns:
                        continue
                    px, s200, r = cl[s].iloc[di], sma200[s].iloc[di], rsi2[s].iloc[di]
                    if pd.isna(px) or pd.isna(s200) or pd.isna(r):
                        continue
                    if px > s200 and r < rsi_buy:                       # dip in uptrend
                        cand.append((s, r))
                if cand:
                    cand.sort(key=lambda x: x[1])                       # most oversold
                    s = cand[0][0]
                    if di + 1 < len(dates):
                        o = op[s].iloc[di + 1] if s in op.columns else np.nan
                        if pd.notna(o) and o > 0:
                            q = int(cap / (float(o) * (1 + COST)))
                            if q >= 1:
                                cap -= q * float(o) * (1 + COST)
                                pos = {"sym": s, "qty": q, "entry": float(o) * (1 + COST), "di": di}
                                trades += 1
        mv = cap + (pos["qty"] * float(cl[pos["sym"]].iloc[di]) if pos and pd.notna(cl[pos["sym"]].iloc[di]) else 0)
        navs.append(mv); nd.append(d)
    nav = pd.Series(navs, index=pd.DatetimeIndex(nd))
    m = _metrics(nav, 1_000_000.0, start, end)
    wr = wins / max(1, trades) * 100
    return (m[0], m[1], m[0] / max(0.01, m[1]), trades, wr) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2026-05-12")
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    eng = _get_engine()
    reg, proxy = regime_proxy(eng, start, end)

    print(f"## n100 momentum + FAST DE-RISK (drawdown stop) {start}..{end}")
    print(f"  {'baseline':22s}: see regime_research (CAGR +9.3% / DD 82.5%)")
    for dd_stop, reclaim in [(0.08, 0.04), (0.10, 0.05), (0.12, 0.06), (0.15, 0.07)]:
        c, d, cal, n, wr = fast_derisk(start, end, proxy, dd_stop, reclaim)
        print(f"  dd-stop {int(dd_stop*100):2d}% reclaim {int(reclaim*100)}%   : "
              f"CAGR {c:+6.1f}% | DD {d:5.1f}% | Calmar {cal:4.2f} | trades {n:3d} | WR {wr:4.1f}%")

    print(f"\n## MEAN-REVERSION (Connors RSI2, long-only) {start}..{end}")
    for label, sideways in (("full-period", False), ("sideways-only", True)):
        r = mean_reversion(start, end, reg, sideways)
        if r:
            print(f"  {label:14s}: CAGR {r[0]:+6.1f}% | DD {r[1]:5.1f}% | "
                  f"Calmar {r[2]:4.2f} | trades {r[3]:3d} | WR {r[4]:4.1f}%")


if __name__ == "__main__":
    main()
