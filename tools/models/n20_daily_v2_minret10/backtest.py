"""n20_daily_v2: baseline + min 30d return >= 10% filter.

Variant of `n20_daily_30d_mc1_uptrend` with one extra hurdle: skip
rank-1 picks with weak momentum (30d return < 10%). Hold cash on
those days instead of entering.

Result: 10L -> 1.65 Cr (+154.72% CAGR, 48.04% DD, 127 trades).
Marginal improvement over v1 baseline (+157.11% CAGR, 50.61% DD).
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols


UNIV_SIZE = 20
LOOKBACK  = 30
ADV_WIN   = 20
SMA_LONG  = 200
MIN_30D_RET = 0.10  # NEW v2: skip weak momentum
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

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

    trading = [d for d in dates if start <= d.date() <= end]
    cap = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    trades = []

    for d in trading:
        di = dates.get_loc(d)
        if di < max(LOOKBACK, SMA_LONG): continue

        pit_adv = adv20.iloc[di].dropna().sort_values(ascending=False)
        pit_univ = pit_adv.head(UNIV_SIZE).index.tolist()
        up = sma200.iloc[di] < cl.iloc[di]
        pit_univ = [s for s in pit_univ if bool(up.get(s, False))]
        if not pit_univ: continue
        rets = cl.iloc[di].reindex(pit_univ) / cl.iloc[di - LOOKBACK].reindex(pit_univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        # NEW v2 filter
        rk = rk[rk >= MIN_30D_RET]
        if rk.empty:
            top = None  # sit in cash
        else:
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
            if top is not None:
                bx = cl[top].iloc[di]
                if pd.notna(bx):
                    bx = float(bx)
                    q = int(cap / bx)
                    if q >= 1 and q * bx <= cap:
                        cap -= q * bx
                        qty = q; hold = top
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

    print(f"\n## v2 RESULTS (min 30d return >= {MIN_30D_RET*100:.0f}%)")
    print(f"  Final NAV:    ₹{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (wins={wins}, losses={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))

    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to",   dest="end",   default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None)
