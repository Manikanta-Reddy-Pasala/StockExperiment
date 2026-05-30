"""'Emerging' mechanics one tier DOWN — the SMALL-CAP tail, PIT-correct.

Universe at each rebalance = eligible_at(n500) MINUS eligible_at(n100), further
restricted to the SMALL end by FF-mcap rank (rank >= SMALL_MIN among all scraped
names) + a liquidity floor (small caps are thin — without it the fills are
fantasy). Primary signal = 15-day momentum (like emerging; pure mcap-climb on
the open universe already proved to blow up). Then A/B the climber filter.

PIT: index membership via eligible_at (historical). FF-mcap shares frozen at one
snapshot, so mcap rank/climb = cross-sectional price relative-strength proxy.

VERDICT (2026-05-30, FULL 2023-26): WEAK + REJECTED. Swept small-min 200/250/300
× minADV 2/5/10cr × retain 1/3: baseline momentum = +13% to +26% CAGR but 48-69%
DD, ~50% win-rate (Calmar 0.27-0.41). The climber overlay HURTS here in every
config (negative / higher DD) — opposite of emerging — because the biggest
rank-climbers in the small tail are pumped illiquid names that crash. The mcap
inclusion edge lives ONLY in the LIQUID-MID tier (emerging: top-100 ADV of
n500-minus-n100); going smaller degrades returns, explodes DD, and flips the
climber from helpful to harmful.

Run: python3 tools/analysis/mcap_smallcap_model.py [--small-min 250]
       [--min-adv-cr 2] [--lookback 15] [--retain 3] [--max-price 3000]
"""
from __future__ import annotations
import sys, csv, argparse, warnings
from pathlib import Path
from datetime import date, timedelta
import numpy as np, pandas as pd
from sqlalchemy import text
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import eligible_at, universe_union
from tools.shared.backtest_engine import run_rotation_backtest

MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
INDEX = "NSE:NIFTY50-INDEX"
FULL = (date(2023, 5, 15), date(2026, 5, 12))
WIN = (date(2025, 3, 1), date(2026, 5, 12))
CLIMB_LB = 60


def load_ffmcap():
    out = {}
    for r in csv.DictReader(open(MCAP_CSV)):
        try:
            ff = float(r["ff_mcap_cr"])
            if ff > 0:
                out[f"NSE:{r['symbol']}-EQ"] = ff
        except (ValueError, TypeError):
            continue
    return out


def run(overlay, small_min, min_adv, lookback, retain, max_price, win):
    ff_mcap = load_ffmcap()
    syms = sorted(set(f"NSE:{s}-EQ" for s in universe_union("n500")) | set(ff_mcap)) + [INDEX]
    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": FULL[0] - timedelta(days=200), "b": FULL[1]})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv20 = df.pivot(index="date", columns="symbol", values="adv").rolling(20).mean()
    dates = cl.index

    ff_shares = {}
    for s in ff_mcap:
        if s in cl.columns:
            last = cl[s].dropna()
            if len(last) and last.iloc[-1] > 0:
                ff_shares[s] = ff_mcap[s] * 1e7 / last.iloc[-1]
    eq = list(ff_shares)
    ffmcap = cl[eq].mul(pd.Series(ff_shares), axis=1)
    mrank = ffmcap.rank(axis=1, ascending=False, method="first")

    s = pd.Series(dates, index=dates)
    firsts = {pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).first().values}
    full = [d for d in dates if win[0] <= d.date() <= win[1] and d in firsts]
    mids, seen = [], set()
    for d in dates:
        if win[0] <= d.date() <= win[1] and 15 <= d.day <= 18 and (d.year, d.month) not in seen:
            mids.append(d); seen.add((d.year, d.month))
    fs = set(full)
    calendar = sorted([(d, "full") for d in full] + [(d, "mid") for d in mids if d not in fs],
                      key=lambda x: x[0])

    def small_universe(di):
        """PIT small tail: in n500, NOT in n100, FF-mcap rank >= small_min, liquid."""
        d = dates[di].date()
        n500 = {f"NSE:{x}-EQ" for x in eligible_at("n500", d)}
        n100 = {f"NSE:{x}-EQ" for x in eligible_at("n100", d)}
        pool = (n500 - n100)
        rnow = mrank.iloc[di]; adv = adv20.iloc[di]
        out = []
        for c in pool:
            if c not in cl.columns:
                continue
            r = rnow.get(c, np.nan)
            if np.isnan(r) or r < small_min:          # keep only the small tail
                continue
            px = cl[c].iloc[di]
            if pd.isna(px) or px <= 0 or px > max_price:
                continue
            if pd.isna(adv.get(c, np.nan)) or adv[c] < min_adv:
                continue
            out.append(c)
        return out

    def mom(c, di):
        if di < lookback:
            return np.nan
        a, b = cl[c].iloc[di], cl[c].iloc[di - lookback]
        return (a / b - 1) if (pd.notna(a) and pd.notna(b) and b > 0) else np.nan

    def mrk(c, di):
        v = mrank.iloc[di].get(c, np.nan)
        return float(v) if pd.notna(v) else np.nan

    def rank_at(di):
        univ = small_universe(di)
        scored = [(c, mom(c, di)) for c in univ]
        scored = [(c, m) for c, m in scored if not np.isnan(m) and m > 0]   # positive momentum
        scored.sort(key=lambda x: -x[1])
        base = [c for c, _ in scored]
        if overlay == "climber":
            di0 = max(0, di - CLIMB_LB)
            return [c for c in base if not np.isnan(mrk(c, di)) and not np.isnan(mrk(c, di0))
                    and mrk(c, di) < mrk(c, di0)]
        return base

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=1_000_000.0, start=win[0], end=win[1], retain_top_n=retain,
        midmonth_ret_at=None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-min", type=int, default=250)
    ap.add_argument("--min-adv-cr", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=15)
    ap.add_argument("--retain", type=int, default=3)
    ap.add_argument("--max-price", type=float, default=3000.0)
    a = ap.parse_args()
    min_adv = a.min_adv_cr * 1e7
    print(f"## SMALLCAP tail (n500-n100, mcap-rank>={a.small_min}) lb={a.lookback} "
          f"retain={a.retain} minADV=Rs.{a.min_adv_cr}cr")
    for name, w in (("FULL 2023-26", FULL), ("Mar25-May26", WIN)):
        for tag, ov in (("baseline", None), ("climber", "climber")):
            r = run(ov, a.small_min, min_adv, a.lookback, a.retain, a.max_price, w)
            wr = r.wins / max(1, r.wins + r.losses) * 100
            print(f"  {name:12s} {tag:8s}: CAGR {r.cagr_pct:+6.1f}% | DD {r.max_dd_pct:5.1f}% | "
                  f"Calmar {r.calmar:4.2f} | trades {len(r.trades):3d} | WR {wr:4.1f}%")


if __name__ == "__main__":
    main()
