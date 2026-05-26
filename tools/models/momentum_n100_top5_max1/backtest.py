"""Standalone backtest: REAL NSE Nifty 100 momentum rotation.

Strategy:
  Universe: FULL Real NSE Nifty 100 (src/data/symbols/nifty100.csv, 104 stocks)
  Signal:   Rank by 30-day return, pick top-1
  Position: max_concurrent=1
  Rebalance: 1st trading day of each month
  OPTIONAL: --mid-month-check adds a day-15 weekday rank check; rotates
            only if rank-1 leads current held's 30d return by >= 5pp.

Pure NSE-official Nifty 100 list. No ADV narrowing, no price filter (distinct
from momentum_pseudo_n100_adv which retains MAX_PRICE filter).
"""
import sys, json, csv, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.backtest_engine import run_rotation_backtest

LOOKBACK = 30
N100_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")

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


def run(start: date, end: date, capital: float, out_dir: Path | None = None,
        mid_month_check: bool = False, mid_month_lead_pct: float = 5.0,
        retain_top_n: int = 1, data_source: str = "fyers"):
    # retain_top_n: hold the position while it stays in the top-N by 30d return;
    # rotate (sell + buy new rank-1) only when it drops OUT of the top-N band.
    # retain_top_n=1 == legacy "rotate whenever held isn't rank-1" (canonical
    # backtest). retain_top_n=5 mirrors the LIVE exit (live_signal.py keeps the
    # stock through top-5). Entry always buys rank-1.
    n100_syms = load_n100()
    print(f"NSE Nifty 100 universe: {len(n100_syms)} stocks")

    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n100_syms, "a": start - timedelta(days=400), "b": end,
                      "ds": data_source})

    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    dates = cl.index
    present = [s for s in n100_syms if s in cl.columns]
    print(f"Loaded {len(dates)} days × {len(present)} symbols")

    rebal_set = set()
    mid_month_set = set()
    y, m = start.year, start.month
    while True:
        target = pd.Timestamp(y, m, 1)
        fut = dates[dates >= target]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0].date() >= start:
            rebal_set.add(fut[0])
        # Mid-month: first trading day on/after the 15th
        if mid_month_check:
            target_mid = pd.Timestamp(y, m, 15)
            fut_mid = dates[dates >= target_mid]
            if len(fut_mid) > 0 and fut_mid[0].date() <= end:
                mid_month_set.add(fut_mid[0])
        m += 1
        if m > 12: m = 1; y += 1
    sd = pd.Timestamp(start)
    if sd in dates: rebal_set.add(sd)
    # Combined rotation calendar with per-day flag
    calendar = sorted({(d, "full") for d in rebal_set} |
                      {(d, "mid") for d in mid_month_set if d not in rebal_set},
                      key=lambda x: x[0])
    rebal = [d for d, _ in calendar]
    rebal_kind = {d: k for d, k in calendar}

    # SELECTION layer: model supplies the per-date ranking; EXECUTION is the
    # shared engine (tools/shared/backtest_engine). Universe = full present
    # Nifty 100 (no filter), ranked by 30-day return.
    def rank_at(di):
        if di < LOOKBACK:
            return []
        univ = [s for s in present if pd.notna(cl[s].iloc[di])]
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    def midret_at(di):
        univ = [s for s in present if pd.notna(cl[s].iloc[di])]
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        return [(s, float(rk[s]) * 100) for s in rk.index]

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=retain_top_n,
        midmonth_ret_at=midret_at, midmonth_lead_pct=mid_month_lead_pct,
    )
    final, cagr, mdd, calmar = res.final_nav, res.cagr_pct, res.max_dd_pct, res.calmar
    trades, yrs, wins, losses, open_pos = (res.trades, res.years, res.wins,
                                           res.losses, res.open_position)

    print(f"\n## RESULTS")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (W={wins}, L={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Calmar: {calmar:.2f}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        summary = {
            "model": "momentum_n100_top5_max1",
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
    ap.add_argument("--mid-month-check", action="store_true",
                    help="Enable day-15 weekday rank check with lead gate")
    ap.add_argument("--mid-month-lead-pct", type=float, default=5.0,
                    help="Minimum lead (pp) for mid-month rotation. Default 5.0")
    ap.add_argument("--retain-top-n", type=int, default=1,
                    help="Hold while in top-N by 30d ret; rotate when out. "
                         "1=legacy (rotate off rank-1), 5=LIVE exit. Default 1.")
    ap.add_argument("--data-source", default="fyers",
                    help="historical_data.data_source filter. Default fyers "
                         "(canonical). Use yfinance only for local dev DBs.")
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None,
        mid_month_check=a.mid_month_check,
        mid_month_lead_pct=a.mid_month_lead_pct,
        retain_top_n=a.retain_top_n,
        data_source=a.data_source)
