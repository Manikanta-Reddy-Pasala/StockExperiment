"""n20_daily_large_only: v1 strategy + NSE Nifty 100 filter.

Halves Max DD vs v1 baseline by constraining universe to large-cap.
₹10L → ₹1.40 Cr (+140.78% CAGR, 26.92% NAV-DD, Calmar 5.23).

Same machinery as v1 (n20_daily_30d_mc1_uptrend) plus one filter:
must be in NSE Nifty 100 (src/data/symbols/nifty100.csv).
"""
import sys, json, csv, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.shared.backtest_engine import run_rotation_backtest


UNIV_SIZE = 20
LOOKBACK  = 30
ADV_WIN   = 20
SMA_LONG  = 200
N100_CSV  = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_n100():
    out = set()
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series","").strip()=="EQ":
                out.add(r["Symbol"].strip())
    return out


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    n100 = load_n100()
    print(f"NSE Nifty 100 filter: {len(n100)} stocks")

    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(SMA_LONG).mean()
    dates = cl.index

    trading = [d for d in dates if start <= d.date() <= end]
    calendar = [(d, "full") for d in trading]

    # SELECTION: daily top-20 ADV ∩ Nifty-100, uptrend (>200d SMA), 30d-ret rank.
    # EXECUTION + daily MTM drawdown come from the shared engine.
    def rank_at(di):
        if di < max(LOOKBACK, SMA_LONG):
            return []
        pit_univ = (adv20.iloc[di].dropna().sort_values(ascending=False)
                    .head(UNIV_SIZE).index.tolist())
        up = sma200.iloc[di] < cl.iloc[di]
        pit_univ = [s for s in pit_univ if bool(up.get(s, False))]
        pit_univ = [s for s in pit_univ
                    if s.replace("NSE:", "").replace("-EQ", "") in n100]
        if not pit_univ:
            return []
        rets = cl.iloc[di].reindex(pit_univ) / cl.iloc[di - LOOKBACK].reindex(pit_univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=1,
    )
    final, cagr = res.final_nav, res.cagr_pct
    trades, yrs, wins, losses, open_pos = (res.trades, res.years, res.wins,
                                           res.losses, res.open_position)
    mdd_nav = res.max_dd_mtm_pct        # daily MTM DD — primary metric for daily rebal
    mdd_realized = res.max_dd_pct       # rebal cap_after DD
    calmar = cagr / max(0.01, mdd_nav)

    print(f"\n## v2 Large-only RESULTS")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (wins={wins}, losses={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"  Max DD (NAV MTM): {mdd_nav:.2f}%  (rebal cap_after: {mdd_realized:.2f}%)")
    print(f"  Calmar: {calmar:.2f}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        summary = {
            "model": "n20_daily_large_only",
            "start": start.isoformat(), "end": end.isoformat(),
            "years": round(yrs, 3),
            "capital": capital, "final_nav": round(final, 0),
            "total_return_pct": round((final / capital - 1) * 100, 2),
            "cagr_pct": round(cagr, 2),
            "max_dd_pct": round(mdd_nav, 2),
            "max_dd_realized_pct": round(mdd_realized, 2),
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
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None)
