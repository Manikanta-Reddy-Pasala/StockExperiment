"""'Pre-inclusion' model: buy stocks whose FREE-FLOAT MARKET CAP already ranks
top-100 / top-500, but which the official Nifty index hasn't added yet (NSE
reviews semi-annually → there's a lag). When NSE promotes them, inclusion flows
+ momentum lift. We anticipate it.

Data:
  exports/nse_mcap.csv  (symbol, total_mcap_cr, ff_mcap_cr, ltp)  <- NSE scrape
  historical_data (D)   close prices
Method (PIT, no lookahead on PRICE; shares approximated from current FF/price):
  ff_shares = ff_mcap_cr*1e7 / ltp      (current free-float shares)
  ff_mcap[t] = ff_shares * close[t]     (historical FF-mcap proxy; shares ~stable)
  each month-start: rank all scraped names by ff_mcap[t].
  CANDIDATE (target=n100): ff_mcap rank <= CUTOFF (100) AND symbol NOT in
    eligible_at("n100", t)  -> mcap-qualified but not yet a member = about to enter.
  Buy top-K candidates (rank closest to the cutoff = strongest), equal weight,
  retain while still a candidate or within RETAIN band; bear-regime stop/trail.
Backtest net 0.15%/side. FULL 2023-26 + Mar25-May26.

NOTE: eligible_at membership is itself PIT historical; the ONLY approximation is
current FF-shares applied to past prices (shares move slowly vs price).

Run (after scrape): python3 tools/analysis/mcap_inclusion_model.py [--target n100|n500] [--k 5]
"""
from __future__ import annotations
import sys, csv, argparse, warnings
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
from sqlalchemy import text
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import eligible_at

MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
INDEX = "NSE:NIFTY50-INDEX"
COST = 0.0015
BEAR_STOP, BEAR_TRAIL = 0.10, 0.15
LOOKBACK = 30
FULL = (date(2023, 5, 15), date(2026, 5, 12)); WIN = (date(2025, 3, 1), date(2026, 5, 12))
CUTOFF = {"n100": 100, "n500": 500}


def load_ffmcap():
    """current free-float market cap (₹ Cr) per symbol from the scrape.

    FF-shares are derived later as ff_mcap / latest_close (from the price DB) —
    the scrape's LTP column is unreliable, and the DB close is the same series
    the backtest ranks on, so this keeps shares consistent with prices."""
    out = {}
    for r in csv.DictReader(open(MCAP_CSV)):
        try:
            ff = float(r["ff_mcap_cr"])
            if ff > 0:
                out[f"NSE:{r['symbol']}-EQ"] = ff
        except (ValueError, TypeError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["n100", "n500"], default="n100")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retain", type=int, default=None)
    a = ap.parse_args()
    cutoff = CUTOFF[a.target]; K = a.k; retain = a.retain or int(cutoff * 1.3)

    ff_mcap = load_ffmcap()
    syms = list(ff_mcap) + [INDEX]
    print(f"scraped names with FF-mcap: {len(ff_mcap)}")
    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": syms, "a": date(2022, 4, 1), "b": FULL[1]})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    op = df.pivot(index="date", columns="symbol", values="open")
    dates = cl.index
    # FF-shares = current FF-mcap / latest available close (price DB), applied to
    # historical close to reconstruct the FF-mcap panel (shares ~stable vs price).
    ff_shares = {}
    for s in ff_mcap:
        if s in cl.columns:
            last = cl[s].dropna()
            if len(last) and last.iloc[-1] > 0:
                ff_shares[s] = ff_mcap[s] * 1e7 / last.iloc[-1]
    print(f"names with FF-shares (matched to price DB): {len(ff_shares)}")
    eq = [s for s in ff_shares if s in cl.columns]
    shares_v = pd.Series({s: ff_shares[s] for s in eq})
    ffmcap = cl[eq].mul(shares_v, axis=1)
    ret = cl[eq] / cl[eq].shift(LOOKBACK) - 1
    idx = cl[INDEX]; idx_sma = idx.rolling(200).mean()

    def msature(dd):
        s = pd.Series(dd, index=dd)
        return [pd.Timestamp(x) for x in s.groupby([dd.year, dd.month]).first().values]

    def _v(p, s, d):
        try:
            v = p.at[d, s]; return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def candidates(di, d):
        """mcap rank <= cutoff AND not yet an official member -> about to enter."""
        row = ffmcap.iloc[di].dropna().sort_values(ascending=False)
        members = {f"NSE:{s}-EQ" for s in eligible_at(a.target, d.date())}
        out = []
        for rank, s in enumerate(row.index, 1):
            if rank > cutoff:
                break
            if s in members:
                continue                       # already in index — skip
            r30 = _v(ret, s, d)
            if np.isnan(r30) or r30 <= 0:       # require positive momentum
                continue
            out.append((s, rank))               # closest to top = strongest
        return [s for s, _ in out]

    def retain_set(di, d):
        row = ffmcap.iloc[di].dropna().sort_values(ascending=False)
        members = {f"NSE:{s}-EQ" for s in eligible_at(a.target, d.date())}
        keep = []
        for rank, s in enumerate(row.index, 1):
            if rank > retain:
                break
            if s not in members:
                keep.append(s)
        return set(keep)

    def sim(eval_s, eval_e):
        rebal = set(d for d in msature(dates) if eval_s <= d.date() <= eval_e)
        cap0 = 1_000_000.0; cash = cap0; pos = {}; nav = []
        for d in [x for x in dates if eval_s <= x.date() <= eval_e]:
            di = dates.get_loc(d)
            hh = bool(pd.notna(idx.get(d, np.nan)) and pd.notna(idx_sma.get(d, np.nan)) and idx[d] > idx_sma[d])
            for s in list(pos.keys()):
                c = _v(cl, s, d)
                if np.isnan(c): continue
                p = pos[s]; p["peak"] = max(p["peak"], c)
                if not hh and (c <= p["peak"]*(1-BEAR_TRAIL) or c <= p["entry_px"]*(1-BEAR_STOP)):
                    cash += p["qty"]*c*(1-COST); del pos[s]
            nav.append((d, cash+sum(p["qty"]*(_v(cl,s,d) if not np.isnan(_v(cl,s,d)) else p["entry_px"]) for s,p in pos.items())))
            if d not in rebal: continue
            keep = retain_set(di, d)
            for s in list(pos.keys()):
                if s not in keep:
                    px = _v(cl, s, d)
                    if px > 0: cash += pos[s]["qty"]*px*(1-COST)
                    del pos[s]
            cand = candidates(di, d)
            want = [s for s in cand if s not in pos][:max(0, K-len(pos))]
            if want and di+1 < len(dates):
                nd = dates[di+1]; per = cash/len(want)
                for s in want:
                    o = _v(op, s, nd)
                    if np.isnan(o) or o <= 0: continue
                    q = int(per/(o*(1+COST)))
                    if q >= 1: cash -= q*o*(1+COST); pos[s] = {"qty": q, "entry_px": o*(1+COST), "peak": o}
        ns = pd.Series({d:v for d,v in nav}).dropna()
        if len(ns) < 2: return None
        days = (ns.index[-1]-ns.index[0]).days
        cagr = (ns.iloc[-1]/cap0)**(365.25/max(1,days))-1
        dd = ((ns.cummax()-ns)/ns.cummax()).max()
        py = {int(y): round((g.iloc[-1]/g.iloc[0]-1)*100,1) for y,g in ns.groupby(ns.index.year)}
        return round(cagr*100,1), round(dd*100,1), py

    print(f"target={a.target} cutoff={cutoff} K={K} retain={retain}")
    for name,(s_,e_) in (("FULL 2023-26",FULL),("Mar25-May26",WIN)):
        r = sim(s_,e_)
        if r: print(f"  {name}: CAGR {r[0]:+.1f}% | DD {r[1]:.1f}% | Calmar {round(r[0]/max(0.01,r[1]),2)} | {r[2]}")


if __name__ == "__main__":
    main()
