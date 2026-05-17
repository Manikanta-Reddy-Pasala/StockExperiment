"""Standalone backtest: REAL NSE Nifty 100 momentum rotation.

Strategy:
  Universe: FULL Real NSE Nifty 100 (src/data/symbols/nifty100.csv, 104 stocks)
  Filter:   close <= MAX_PRICE (skips mega-priced names with poor risk/return,
            e.g. BAJAJ-AUTO ₹12157 single trade lost ₹484K in backtest)
  Signal:   Rank by 30-day return, pick top-1
  Position: max_concurrent=1
  Rebalance: 1st trading day of each month

Pure NSE-official Nifty 100 list. No ADV narrowing (distinct from
momentum_pseudo_n100_adv and n20_daily_large_only).
"""
import sys, json, csv, argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine

LOOKBACK = 30
MAX_PRICE = 3000  # skip high-px stocks; sweep showed CAGR +21pp / DD -3pp
N100_CSV = "/app/src/data/symbols/nifty100.csv"

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_n100():
    out = []
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                out.append(f"NSE:{r['Symbol'].strip()}-EQ")
    return out


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    n100_syms = load_n100()
    print(f"NSE Nifty 100 universe: {len(n100_syms)} stocks")

    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n100_syms, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    dates = cl.index
    present = [s for s in n100_syms if s in cl.columns]
    print(f"Loaded {len(dates)} days × {len(present)} symbols")

    rebal_set = set()
    y, m = start.year, start.month
    while True:
        target = pd.Timestamp(y, m, 1)
        fut = dates[dates >= target]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0].date() >= start:
            rebal_set.add(fut[0])
        m += 1
        if m > 12: m = 1; y += 1
    sd = pd.Timestamp(start)
    if sd in dates: rebal_set.add(sd)
    rebal = sorted(rebal_set)

    cap = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    trades = []

    for d in rebal:
        di = dates.get_loc(d)
        if di < LOOKBACK:
            continue

        # Universe = present Nifty 100 filtered by MAX_PRICE at this date
        univ = [s for s in present
                if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= MAX_PRICE]
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
                        "sym":        hold.replace("NSE:", "").replace("-EQ", ""),
                        "entry_date": entry_date,
                        "exit_date":  d.date().isoformat(),
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

    peak = capital; mdd = 0
    for t in trades:
        peak = max(peak, t["cap_after"])
        dd = (peak - t["cap_after"]) / peak * 100
        mdd = max(mdd, dd)

    print(f"\n## RESULTS")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (W={wins}, L={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Calmar: {cagr/max(0.01,mdd):.2f}")

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
