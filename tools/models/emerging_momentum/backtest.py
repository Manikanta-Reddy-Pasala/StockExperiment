"""Standalone backtest: EMERGING MOMENTUM — single-position rotation (Config 1).

Offline research/validation path. The UNIVERSE-POOL + per-date SELECTION lives in
the shared core `strategy.py` (build_pools / pool_for_date / rank_pool /
midret_pool / params) — the SAME code live_signal.py uses, so backtest and live
cannot drift. Position-keeping + NAV accounting are delegated to the shared
engine tools.shared.backtest_engine.run_rotation_backtest (the same engine
momentum_n100_top5_max1 uses).

Strategy:
  Universe: POINT-IN-TIME mid/small caps = top-100 by 20d ADV from
            (eligible_at n500 MINUS eligible_at n100), rebuilt per year-start.
  Signal:   rank by 15-trading-day return (ret > 0), price in (0, 3000]; no sma.
  Position: max_concurrent=1, retain_top_n=3 (hold while in top-3 rank).
  Rebalance: 1st trading day of each month ("full") + a mid-month ("mid")
             check on the first trading day with 15<=day<=18; the mid-month
             rotation fires only when a new leader beats the held name's 15d
             return by >= 5pp (MIDMONTH_LEAD).

CONFIG 1 (lb15, sma off) + MCAP-CLIMBER. Full-cycle 2021-04→2026-05 on
authoritative PIT membership ≈ +46.9% CAGR / 37.7% DD / Calmar 1.24 — see
exports/models/emerging_momentum/SUMMARY.md. (An earlier ~+98-121% figure was on
the buggy Wayback N100, which leaked large-cap winners into this mid/small model;
the correct PIT N100 exclusion since 2026-05-31 gives the honest +46.9%.)

Run: python3 tools/models/emerging_momentum/backtest.py \
       --from 2023-05-15 --to 2026-05-12 --out exports/models/emerging_momentum
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union
from tools.shared.backtest_engine import run_rotation_backtest
from tools.models.emerging_momentum import strategy as S

DEFAULT_START = date(2021, 3, 1)
DEFAULT_END = date(2026, 5, 29)
DEFAULT_CAP = 1_000_000.0


def load_panels(eng, start, end):
    """Load the close panel + ADV panel for the N500 universe (+ index mask).

    History is always loaded back to (POOL_ANCHOR_START - 420d) even for a later
    eval-window start so the per-year PIT pools (anchored to POOL_ANCHOR_START)
    and the 20d-ADV warmup match the full run exactly. Index-only date rows are
    dropped so they don't poison the rolling windows (same guard as live_signal).
    """
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    load_from = min(start, S.POOL_ANCHOR_START) - timedelta(days=420)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": load_from, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").astype(float)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20 = S.indicators(cl, adv_rs)
    return cl, adv20


def run(start, end, capital, out_dir=None):
    """Run the single-position emerging-momentum rotation backtest and report."""
    eng = _get_engine()
    cl, adv20 = load_panels(eng, start, end)
    dates = cl.index
    anchors, pools = S.build_pools(adv20, dates, end)

    # ---- Rebalance calendar from the SHARED core (same rule live mirrors) ----
    # full = month's first trading day; mid = first trading day on/after the 15th.
    calendar = S.build_calendar(dates, start, end)

    def rank_at(di):
        """Rank the PIT pool by 15d return at row `di` (ret>0, price<=MAX_PRICE)."""
        return S.rank_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    def midret_at(di):
        """(symbol, 15d_return_pct) best-first for the mid-month lead gate."""
        return S.midret_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=S.RETAIN,
        midmonth_ret_at=midret_at, midmonth_lead_pct=S.MIDMONTH_LEAD,
    )

    final, cagr, mdd, calmar = res.final_nav, res.cagr_pct, res.max_dd_pct, res.calmar
    trades, yrs, wins, losses, open_pos = (res.trades, res.years, res.wins,
                                           res.losses, res.open_position)

    print(f"\n## RESULTS (emerging_momentum, single-position)")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final / capital - 1) * 100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (W={wins}, L={losses}, "
          f"WR={wins / max(1, wins + losses) * 100:.1f}%)")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Calmar: {calmar:.2f}")

    result = {
        "model": "emerging_momentum",
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
            "per_year": res.per_year,
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", "--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--end", "--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out-dir", "--out", dest="out_dir", default=None)
    a = ap.parse_args()
    s = datetime.strptime(a.start, "%Y-%m-%d").date()
    e = datetime.strptime(a.end, "%Y-%m-%d").date()
    r = run(s, e, a.capital, a.out_dir)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
