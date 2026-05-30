"""Regime-aware research over a LONG history (real bears: 2018, COVID-2020, 2022).

The NIFTY50-INDEX symbol only has ~2023+ data, so regime is detected from a
10-year LARGE-CAP BASKET PROXY (equal-weight normalised closes of mega-caps that
have full 2016+ history). Trend direction is robust to exact constituents.

Regime (PIT, no lookahead):
  BULL     : proxy > SMA200 AND SMA50 > SMA200
  BEAR     : proxy < SMA200 AND SMA50 < SMA200
  SIDEWAYS : else (chop)

Tests n100 momentum (real Nifty100, has 10yr history) with regime routing:
  baseline / cash-in-bear / bull-only.

Point DATABASE_URL at prod (10yr data) before running:
  DATABASE_URL=postgresql://trader:...@localhost:15432/trading_system \
    python3 tools/analysis/regime_research.py --start 2017-01-01 --end 2026-05-12
"""
from __future__ import annotations
import sys, argparse, warnings
from datetime import date, timedelta
import numpy as np, pandas as pd
from sqlalchemy import text
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from tools.shared.ohlcv_cache import _get_engine
import tools.models.momentum_n100_top5_max1.backtest as N100

# mega-caps with deep history; equal-weight normalised = Nifty trend proxy
BASKET = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK", "HINDUNILVR",
          "ITC", "SBIN", "LT", "KOTAKBANK"]


def regime_proxy(eng, start, end):
    syms = [f"NSE:{s}-EQ" for s in BASKET]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data WHERE symbol=ANY(:s) "
            "AND date BETWEEN :a AND :b AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill().dropna(how="all")
    have = [s for s in syms if s in cl.columns]
    norm = cl[have] / cl[have].iloc[0]        # rebase each to 1.0
    proxy = norm.mean(axis=1)                  # equal-weight basket
    sma200 = proxy.rolling(200).mean(); sma50 = proxy.rolling(50).mean()
    reg = pd.Series("SIDEWAYS", index=proxy.index)
    reg[(proxy > sma200) & (sma50 > sma200)] = "BULL"
    reg[(proxy < sma200) & (sma50 < sma200)] = "BEAR"
    reg[sma200.isna()] = "SIDEWAYS"
    return reg, proxy


def run_n100(start, end, regime_gate, reg):
    orig = N100.run_rotation_backtest
    holder = {}

    def patched(*a, **kw):
        dates = kw["dates"]; base = kw["rank_at"]

        def wrapped(di):
            if regime_gate and not regime_gate(reg.get(dates[di], "SIDEWAYS")):
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
    return holder["res"]


def fmt(tag, r):
    wr = r.wins / max(1, r.wins + r.losses) * 100
    print(f"  {tag:11s}: CAGR {r.cagr_pct:+6.1f}% | DD {r.max_dd_pct:5.1f}% | "
          f"Calmar {r.calmar:4.2f} | trades {len(r.trades):3d} | WR {wr:4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2026-05-12")
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    eng = _get_engine()
    reg, proxy = regime_proxy(eng, start, end)
    win = (reg.index.date >= start) & (reg.index.date <= end)
    sub = reg[win]
    print(f"## Regime mix {start}..{end} ({len(sub)} trading days, basket proxy)")
    pr = proxy.pct_change()
    for r in ("BULL", "SIDEWAYS", "BEAR"):
        d = (sub == r).sum()
        cum = (1 + pr[win][sub == r].fillna(0)).prod() - 1
        print(f"  {r:9s}: {d:4d} ({d/max(1,len(sub))*100:4.1f}%) | proxy cum {cum*100:+6.1f}%")
    print(f"\n## n100 momentum — regime routing ({start}..{end})")
    fmt("baseline", run_n100(start, end, None, reg))
    fmt("bear-cash", run_n100(start, end, lambda r: r != "BEAR", reg))
    fmt("bull-only", run_n100(start, end, lambda r: r == "BULL", reg))


if __name__ == "__main__":
    main()
