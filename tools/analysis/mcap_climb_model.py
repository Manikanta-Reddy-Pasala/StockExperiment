"""Standalone 'rising market-cap' model on the FULL NSE equity universe (NOT
restricted to Nifty 500). Universe = every stock we scraped a free-float mcap
for (~2,025). Pick the names whose FF-mcap RANK is climbing the most, rotate
monthly (+ mid-month), single position — same engine as emerging_momentum.

IMPORTANT caveats baked in:
  * Liquidity floor — the full universe is full of illiquid micro-caps you
    can't actually fill. Require 20d ADV >= MIN_ADV (₹) at entry, else the
    backtest is fantasy. Tunable; default ₹5 cr/day.
  * "mcap climb" = cross-sectional price relative-strength vs the whole market
    (shares frozen at one snapshot) — a proxy, not fundamental mcap growth.

VERDICT (2026-05-30, FULL 2023-26): REJECTED as a standalone model. Sweeping
the liquidity floor + lookback gave 60-98% drawdowns and mostly NEGATIVE CAGR
(minADV 50cr = -30%/82%DD, 100cr = -36%/86%DD); only the recent bull window
flatters it. "Biggest mcap climber" buys the most parabolic name and rides it
down. The climb signal only adds value as a FILTER on a bounded mid-cap
momentum model (see emerging_momentum + tools/analysis/mcap_overlay.py), not as
the primary selector on the open universe. Kept for the future re-test once
market_cap_history holds real (not price-proxy) mcap drift.

Run: python3 tools/analysis/mcap_climb_model.py [--k 1] [--lookback 60]
       [--min-adv-cr 5] [--retain 3] [--max-price 3000]
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
from tools.shared.backtest_engine import run_rotation_backtest

MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
INDEX = "NSE:NIFTY50-INDEX"
FULL = (date(2023, 5, 15), date(2026, 5, 12))
WIN = (date(2025, 3, 1), date(2026, 5, 12))


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


def run(lookback, k, retain, min_adv, max_price, eval_window):
    ff_mcap = load_ffmcap()
    syms = list(ff_mcap) + [INDEX]
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
    op = df.pivot(index="date", columns="symbol", values="open")
    adv20 = df.pivot(index="date", columns="symbol", values="adv").rolling(20).mean()
    dates = cl.index

    # FF-shares from latest close -> reconstruct FF-mcap panel -> daily rank.
    ff_shares = {}
    for s in ff_mcap:
        if s in cl.columns:
            last = cl[s].dropna()
            if len(last) and last.iloc[-1] > 0:
                ff_shares[s] = ff_mcap[s] * 1e7 / last.iloc[-1]
    eq = list(ff_shares)
    ffmcap = cl[eq].mul(pd.Series(ff_shares), axis=1)
    mrank = ffmcap.rank(axis=1, ascending=False, method="first")  # 1 = biggest

    s = pd.Series(dates, index=dates)
    firsts = {pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).first().values}
    full = [d for d in dates if eval_window[0] <= d.date() <= eval_window[1] and d in firsts]
    mids, seen = [], set()
    for d in dates:
        if eval_window[0] <= d.date() <= eval_window[1] and 15 <= d.day <= 18 and (d.year, d.month) not in seen:
            mids.append(d); seen.add((d.year, d.month))
    fs = set(full)
    calendar = sorted([(d, "full") for d in full] + [(d, "mid") for d in mids if d not in fs],
                      key=lambda x: x[0])

    def climbers(di):
        """Names whose FF-mcap rank improved over `lookback`, liquid + priced,
        ordered by biggest climb (most rank positions gained)."""
        if di < lookback:
            return []
        di0 = di - lookback
        out = []
        rnow = mrank.iloc[di]; rthen = mrank.iloc[di0]
        adv = adv20.iloc[di]
        for col in eq:
            a, b = rnow.get(col, np.nan), rthen.get(col, np.nan)
            if np.isnan(a) or np.isnan(b) or a >= b:        # must be climbing
                continue
            px = cl[col].iloc[di]
            if pd.isna(px) or px <= 0 or px > max_price:     # priced + cap
                continue
            if pd.isna(adv.get(col, np.nan)) or adv[col] < min_adv:  # liquidity floor
                continue
            out.append((col, b - a))                         # rank positions gained
        out.sort(key=lambda x: -x[1])
        return [c for c, _ in out]

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=climbers,
        capital=1_000_000.0, start=eval_window[0], end=eval_window[1],
        retain_top_n=retain, midmonth_ret_at=None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--retain", type=int, default=3)
    ap.add_argument("--min-adv-cr", type=float, default=5.0)
    ap.add_argument("--max-price", type=float, default=3000.0)
    a = ap.parse_args()
    min_adv = a.min_adv_cr * 1e7
    print(f"## rising-mcap (FULL universe) lb={a.lookback} retain={a.retain} "
          f"minADV=Rs.{a.min_adv_cr}cr maxPx={a.max_price}")
    for name, w in (("FULL 2023-26", FULL), ("Mar25-May26", WIN)):
        r = run(a.lookback, a.k, a.retain, min_adv, a.max_price, w)
        wr = r.wins / max(1, r.wins + r.losses) * 100
        print(f"  {name:12s}: CAGR {r.cagr_pct:+6.1f}% | DD {r.max_dd_pct:5.1f}% | "
              f"Calmar {r.calmar:4.2f} | trades {len(r.trades):3d} | WR {wr:4.1f}%")


if __name__ == "__main__":
    main()
