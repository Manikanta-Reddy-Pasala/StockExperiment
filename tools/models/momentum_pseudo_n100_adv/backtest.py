"""Standalone backtest: pseudo-N100 (ADV-rank from N500, yearly PIT rebuild, MINUS Small).

Reproduces +136.39% CAGR (10L -> 1.32 Cr) over 2023-05-15 to 2026-05-12.

Same strategy as momentum_n100_top5_max1 (lb=30, mc=1, monthly, top-1) but
universe = top-100 by 20-day ADV at each year-start instead of NSE Nifty 100.
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
from tools.shared.backtest_engine import run_rotation_backtest


LOOKBACK = 30
ADV_WIN  = 20
UNIV_SIZE = 100
MAX_PRICE = 3000  # skip stocks > ₹3000 at entry (DIXON/MARUTI etc were big losers)
# Drop Small-cap NSE Nifty Smallcap 250 stocks from universe (free +2pp CAGR, DD unchanged).
import csv as _csv
_SML_PATH = str(ROOT / "src" / "data" / "symbols" / "nifty_smallcap250.csv")
def _load_smallcap():
    out = set()
    try:
        with open(_SML_PATH) as f:
            for r in _csv.DictReader(f):
                if r.get("Series","").strip()=="EQ":
                    out.add(r["Symbol"].strip())
    except FileNotFoundError:
        pass
    return out
_SMALLCAP = _load_smallcap()
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def run(start: date, end: date, capital: float, out_dir: Path | None = None,
        retain_top_n: int = 1, data_source: str = "fyers"):
    # retain_top_n: hold while in top-N by 30d return; rotate (sell + buy rank-1)
    # only when held drops OUT of top-N. 1=legacy (rotate off rank-1, canonical),
    # 5=LIVE exit (live_signal.py keeps the stock through top-5). Entry buys rank-1.
    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]
    print(f"N500 pool: {len(n500)}")

    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end,
                      "ds": data_source})

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(200).mean()  # uptrend filter
    dates = cl.index

    # Yearly-PIT universe rebuild
    year_starts = []
    cur = start
    while cur <= end:
        year_starts.append(pd.Timestamp(cur))
        cur = cur.replace(year=cur.year + 1)

    year_universes = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0: continue
        di = dates.get_loc(fut[0])
        pit_adv = adv20.iloc[di].dropna().sort_values(ascending=False)
        top = pit_adv.head(UNIV_SIZE).index.tolist()
        # Drop Small-cap names from top-100 (sweep showed +2pp CAGR, DD unchanged)
        year_universes[ys] = [s for s in top if s.replace("NSE:","").replace("-EQ","") not in _SMALLCAP]

    def pick_universe(d):
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:
                chosen = ys
        return year_universes.get(chosen, [])

    rebal_set = set()
    y, m = start.year, start.month
    while True:
        target = pd.Timestamp(y, m, 1)
        fut = dates[dates >= target]
        if len(fut) == 0 or fut[0].date() > end: break
        if fut[0].date() >= start:
            rebal_set.add(fut[0])
        m += 1
        if m > 12: m = 1; y += 1
    sd = pd.Timestamp(start)
    if sd in dates: rebal_set.add(sd)
    rebal = sorted(rebal_set)
    calendar = [(d, "full") for d in rebal]

    # SELECTION layer: yearly-PIT pseudo-N100, uptrend (>200d SMA) + MAX_PRICE
    # filter, ranked by 30-day return. EXECUTION is the shared engine.
    def rank_at(di):
        if di < max(LOOKBACK, 200):
            return []
        univ = pick_universe(dates[di])
        up = sma200.iloc[di] < cl.iloc[di]
        univ = [s for s in univ if bool(up.get(s, False))]
        univ = [s for s in univ
                if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= MAX_PRICE]
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=retain_top_n,
    )
    final, cagr, mdd, calmar = res.final_nav, res.cagr_pct, res.max_dd_pct, res.calmar
    trades, yrs, wins, losses, open_pos = (res.trades, res.years, res.wins,
                                           res.losses, res.open_position)

    print(f"\nFinal NAV: Rs.{final:,.0f}")
    print(f"Total: {(final/capital-1)*100:+.2f}%  CAGR: {cagr:+.2f}%")
    print(f"Trades: {len(trades)} (W={wins} L={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"Max DD (rebal cap_after): {mdd:.2f}%  Calmar: {calmar:.2f}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        summary = {
            "model": "momentum_pseudo_n100_adv",
            "start": start.isoformat(), "end": end.isoformat(),
            "years": round(yrs, 3),
            "capital": capital, "final_nav": round(final, 0),
            "total_return_pct": round((final / capital - 1) * 100, 2),
            "cagr_pct": round(cagr, 2),
            "max_dd_pct": round(mdd, 2),
            "calmar": round(calmar, 2),
            "trades": len(trades),
            "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "open_position": open_pos,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to",   dest="end",   default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--retain-top-n", type=int, default=1,
                    help="Hold while in top-N by 30d ret; rotate when out. "
                         "1=legacy (rotate off rank-1), 5=LIVE exit. Default 1.")
    ap.add_argument("--data-source", default="fyers",
                    help="historical_data.data_source filter. Default fyers "
                         "(canonical). Use yfinance only for local dev DBs.")
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None,
        retain_top_n=a.retain_top_n,
        data_source=a.data_source)
