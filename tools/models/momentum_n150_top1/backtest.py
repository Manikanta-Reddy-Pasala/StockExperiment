"""Momentum rotation backtest on N150 universe (yearly-PIT, lookback=30d).

Variant of momentum_n100_top5_max1 with universe widened to top-150 by ADV.
Reproduces 10L -> 50.78L (+71.88% CAGR) over 2023-05-15 -> 2026-05-12.

N100 winner: +136.39% CAGR. N150 cuts return ~50%. NOT recommended.
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols


UNIV_SIZE = 150
LOOKBACK  = 30
MAX_CONC  = 1
UPTREND   = True
SMA_LONG  = 200
ADV_WIN   = 20
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

    print(f"Loading N500 OHLCV {start} … {end} (pool={len(n500)})…")
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(SMA_LONG).mean()
    dates = cl.index
    print(f"Loaded {len(dates)} days × {cl.shape[1]} symbols")

    # Year-start PIT universe
    year_starts = []
    cur = start
    while cur <= end:
        year_starts.append(pd.Timestamp(cur))
        cur = cur.replace(year=cur.year + 1)
    year_universes = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0:
            continue
        di = dates.get_loc(fut[0])
        pit_adv = adv20.iloc[di].dropna().sort_values(ascending=False)
        year_universes[ys] = pit_adv.head(UNIV_SIZE).index.tolist()

    def pick_universe(d):
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:
                chosen = ys
        return year_universes.get(chosen, [])

    # Monthly rebalance dates
    rebal = []
    seen = set()
    y, m = start.year, start.month
    while True:
        target = pd.Timestamp(y, m, 1)
        fut = dates[dates >= target]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0] not in seen:
            rebal.append(fut[0])
            seen.add(fut[0])
        m += 1
        if m > 12:
            m = 1
            y += 1
    sd = pd.Timestamp(start)
    if sd in dates and sd not in seen:
        rebal.insert(0, sd)

    cap = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    trades = []

    for d in rebal:
        di = dates.get_loc(d)
        if di < max(LOOKBACK, SMA_LONG):
            continue
        univ = pick_universe(d)
        if UPTREND:
            up = sma200.iloc[di] < cl.iloc[di]
            univ = [s for s in univ if bool(up.get(s, False))]
        if not univ:
            continue
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        if rk.empty:
            continue
        top = rk.index[0]

        if top != hold:
            if hold and qty > 0:
                sx = cl[hold].iloc[di]
                if pd.notna(sx):
                    sx = float(sx)
                    proc = qty * sx
                    cap += proc
                    pnl = proc - qty * entry_px
                    pct = (sx / entry_px - 1) * 100
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date":  d.date().isoformat(),
                        "sym":        hold.replace("NSE:", "").replace("-EQ", ""),
                        "qty":        qty,
                        "entry_px":   round(entry_px, 2),
                        "exit_px":    round(sx, 2),
                        "pnl":        round(pnl, 0),
                        "ret_pct":    round(pct, 2),
                        "cap_after":  round(cap, 0),
                    })
                    hold = None; qty = 0

            bx = cl[top].iloc[di]
            if pd.notna(bx):
                bx = float(bx)
                q = int(cap / bx)
                if q >= 1 and q * bx <= cap:
                    cap -= q * bx
                    qty = q
                    hold = top
                    entry_px = bx
                    entry_date = d.date().isoformat()

    final = cap
    if hold:
        last = float(cl[hold].iloc[-1])
        final = cap + qty * last

    wins   = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs    = (end - start).days / 365.25
    cagr   = ((final / capital) ** (1 / yrs) - 1) * 100

    print(f"\n## RESULTS (N150)")
    print(f"  Final NAV:    ₹{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (wins={wins}, losses={losses})")
    print(f"  WR: {wins/max(1,wins+losses)*100:.1f}%")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        univ_dump = {
            str(k.date()): [s.replace("NSE:", "").replace("-EQ", "") for s in v]
            for k, v in year_universes.items()
        }
        (out_dir / "yearly_universes.json").write_text(json.dumps(univ_dump, indent=2))
        print(f"  Wrote: {out_dir}/{{trade_ledger,yearly_universes}}.json")

    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to",   dest="end",   default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(
        date.fromisoformat(a.start),
        date.fromisoformat(a.end),
        a.capital,
        Path(a.out) if a.out else None,
    )
