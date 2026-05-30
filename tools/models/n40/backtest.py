"""n40 (dir: n20_daily_large_only): WEEKLY rotation + NSE Nifty 100 filter.

Universe constrained to large-cap (top-40 ADV ∩ PIT Nifty 100) to cut Max DD vs
the unconstrained v1. Current metrics: see exports/models/n40/SUMMARY.md
(full-cycle 2021-04→2026-05 ≈ +40% CAGR on authoritative PIT N100).

Same machinery as v1 plus one filter: must be in NSE Nifty 100 (PIT via
eligible_at). 2026-05-30: rebalance switched DAILY → WEEKLY (cut the whipsaw).

Role in the model flow (data_pull -> live_signal -> cron -> backtest)
---------------------------------------------------------------------
This is the HISTORICAL evaluation leg, not part of the live trading path.
It replays the exact same selection rule the live path uses against full
price history to produce the published CAGR / drawdown / Calmar numbers and
the trade ledger (trade_ledger.json / summary.json).

  - data_pull.py  : keeps the N500 daily OHLCV (the PIT ranking pool) fresh.
  - live_signal.py: emits today's SELL/ENTRY1 signal for real trading.
  - cron.py       : schedules data_pull + live_signal jobs.
  - backtest.py   : (this file) offline what-if over a date range.

The actual entry/exit decision is NOT re-implemented here. Selection (which
stocks are candidates and in what order) is computed locally in `rank_at`,
but the per-day rotation execution and the daily mark-to-market drawdown that
n20 reports are delegated to the SHARED engine
(tools/shared/backtest_engine.run_rotation_backtest), which internally calls
the SHARED rotation core decide_rotation — the same core live_signal.py uses.
This guarantees backtest and live cannot drift.

It rotates WEEKLY holding only rank-1 (still the highest-churn rotation model).
Its primary risk metric is the daily NAV mark-to-market drawdown (max_dd_mtm_pct,
marked every trading day) rather than the rebalance-day realized drawdown.
"""
import sys, json, csv, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols  # noqa: F401
from tools.shared.backtest_engine import run_rotation_backtest
from tools.shared.index_membership import eligible_at, universe_union
from tools.shared.rebalance_calendar import build_weekly_calendar


from tools.models.n40.strategy import (  # noqa: E402  shared w/ live
    UNIV_SIZE, LOOKBACK, ADV_WIN, SMA_LONG, RETAIN)
N100_CSV  = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")
DEFAULT_START = date(2021, 3, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0


def load_n100_pit(d: date) -> set[str]:
    """Point-in-time NSE Nifty 100 large-cap filter set for date `d`.

    Returns plain symbols (no NSE:/-EQ wrap) so the existing rank_at code
    can keep its symbol-stripping check unchanged.
    """
    return set(eligible_at("n100", d))


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    """Replay the n40 weekly-rotation strategy over [start, end] and report.

    Loads the N500 daily price/volume panel, builds the rolling 20d-ADV and
    200d-SMA matrices, defines a point-in-time selection closure (`rank_at`),
    and hands selection + capital to the shared rotation engine which performs
    the daily execution and computes the daily mark-to-market drawdown.

    Args:
        start: first trading date to evaluate (inclusive).
        end: last trading date to evaluate (inclusive).
        capital: starting capital in rupees.
        out_dir: optional directory; if given, trade_ledger.json and
            summary.json are written there.

    Returns:
        tuple[float, float, list]: (final_nav, cagr_pct, trades).
    """
    print("Large-cap filter source: PIT n100 (eligible_at per day)")

    eng = _get_engine()
    # PIT ranking pool: union of every symbol ever in NSE Nifty 500.
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    print(f"N500 union pool (PIT): {len(n500)}")

    with eng.connect() as c:
        # Pull 400 extra calendar days before `start` so the 200d SMA and the
        # rolling 20d ADV are fully warmed up on the first evaluated day.
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])
    # ADV proxy in rupees = close * volume (rupee turnover, not share count).
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(ADV_WIN).mean()      # 20-day average daily turnover
    sma200 = cl.rolling(SMA_LONG).mean()        # 200-day SMA (long uptrend ref)
    dates = cl.index

    # WEEKLY rebalance (the fix, 2026-05-30): re-rank only on the first trading
    # day of each ISO week, hold through the week. Daily rebalancing churned
    # (55% of trades held <=3 days = whipsaw); weekly cuts DD + lifts CAGR on
    # both 2023-26 and the full 2021-26 cycle. Shared rule (build_weekly_calendar)
    # so live mirrors via is_week_rebalance_day.
    calendar = build_weekly_calendar(dates, start, end)

    # SELECTION: weekly top-40 ADV ∩ Nifty-100, uptrend (>200d SMA), 30d-ret rank.
    # EXECUTION + daily MTM drawdown come from the shared engine.
    def rank_at(di):
        """Point-in-time selection for day index `di` (no look-ahead).

        Args:
            di: integer position into `dates`/`cl` for the evaluated day.

        Returns:
            list[str]: candidate symbols ordered best-first by 30d return,
            after the ADV / uptrend / Nifty-100 filters. Empty before the
            warm-up window or when no candidate survives the filters. The
            shared engine treats element 0 as rank-1.
        """
        # Need full SMA200 + 30d-return history before we can rank anything.
        if di < max(LOOKBACK, SMA_LONG):
            return []
        # Top-20 by 20d ADV, rebuilt fresh from this day's turnover snapshot.
        pit_univ = (adv20.iloc[di].dropna().sort_values(ascending=False)
                    .head(UNIV_SIZE).index.tolist())
        # Uptrend filter: keep only names trading above their 200d SMA.
        up = sma200.iloc[di] < cl.iloc[di]
        pit_univ = [s for s in pit_univ if bool(up.get(s, False))]
        # Large-cap filter: intersect with the POINT-IN-TIME NSE Nifty 100
        # constituents at this exact day (no survivorship).
        n100_today = load_n100_pit(dates[di].date())
        pit_univ = [s for s in pit_univ
                    if s.replace("NSE:", "").replace("-EQ", "") in n100_today]
        if not pit_univ:
            return []
        # Rank survivors by trailing 30-day return, highest first.
        rets = cl.iloc[di].reindex(pit_univ) / cl.iloc[di - LOOKBACK].reindex(pit_univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    # retain_top_n=1 -> pure top-1 daily rotation (sell as soon as held drops
    # below rank-1). Same knob the live path passes to decide_rotation.
    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=RETAIN,
    )
    final, cagr = res.final_nav, res.cagr_pct
    trades, yrs, wins, losses, open_pos = (res.trades, res.years, res.wins,
                                           res.losses, res.open_position)
    # Daily MTM DD is the headline risk metric for this high-churn daily model;
    # mdd_realized (rebal-day cap_after DD) is reported only as a secondary view.
    mdd_nav = res.max_dd_mtm_pct        # daily MTM DD — primary metric for daily rebal
    mdd_realized = res.max_dd_pct       # rebal cap_after DD
    # Calmar = CAGR / drawdown; clamp denominator so a ~0 DD can't divide-by-zero.
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
            "per_year": res.per_year,
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
