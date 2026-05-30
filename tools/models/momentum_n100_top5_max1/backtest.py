"""Standalone backtest: REAL NSE Nifty 100 momentum rotation — POINT-IN-TIME.

Strategy:
  Universe: POINT-IN-TIME NSE Nifty 100 membership per rebalance date,
            loaded from src/data/symbols/n100_membership.csv. The model
            ranks ONLY symbols that were actually in n100 on the rebalance
            date (no survivorship bias). A symbol drops out of the eligible
            set the moment it leaves the index; the model also force-sells
            a held symbol that drops out (DELIST exit).
  Signal:   Rank by 15-day return, pick top-1
  Position: max_concurrent=1
  Rebalance: 1st trading day of each month
  OPTIONAL: --mid-month-check adds a day-15 weekday rank check; rotates
            only if rank-1 leads current held's 15d return by >= 5pp.

Set 2026-05-28: switched from today's nifty100.csv to PIT membership table
(built by tools/analysis/build_membership_table.py). The old behaviour
inflated 10yr CAGR ~80pp by silently including stocks that joined the
index after the ranking date (Adani group, Zomato, NUVAMA, etc.) and
excluding stocks that had left (YESBANK, IL&FS, DHFL, etc.).
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
from tools.shared.index_membership import eligible_at, universe_union
from tools.models.momentum_n100_top5_max1.strategy import (
    LOOKBACK, RETAIN, MIDMONTH_LEAD, build_calendar)

# 15 TRADING days (~3 weeks). Set 2026-05-27 from a 6-year sweep: 15td beat 30td
# on BOTH CAGR and max DD (the relative result; absolute numbers predate the
# 2026-05-31 PIT-membership rebuild). Current metrics: exports/models/.../SUMMARY.md.
# LOOKBACK/RETAIN/MIDMONTH_LEAD now imported from strategy.py (shared with live).
# Today's published n100 list — kept for live signal compatibility only.
# THE BACKTEST DOES NOT USE THIS FILE; it pulls PIT membership via
# tools.shared.index_membership.eligible_at instead.
N100_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")

DEFAULT_START = date(2021, 3, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0


def load_n100_union(index_name: str = "n100"):
    """Load the union of every symbol that was EVER in `index_name`.

    Source: tools.shared.index_membership.universe_union(index_name) (built
    from Wayback snapshots). The backtest filters this superset down to the
    point-in-time eligible set via eligible_at(d) at every rebalance date.

    Args:
        index_name: 'n100' (default), 'n200', or 'n500'. Selects which NSE
            broad-market index drives membership at each rebalance date.

    Returns:
        list[str]: Fyers-style symbols, e.g. "NSE:RELIANCE-EQ".
    """
    return [f"NSE:{s}-EQ" for s in sorted(universe_union(index_name))]


def run(start: date, end: date, capital: float, out_dir: Path | None = None,
        mid_month_check: bool = True, mid_month_lead_pct: float = MIDMONTH_LEAD,
        retain_top_n: int = RETAIN, data_source: str = "fyers",
        index_name: str = "n100"):
    """Run the full Nifty 100 momentum-rotation backtest and report metrics.

    Loads close prices, builds the monthly (+ optional mid-month) rebalance
    calendar, defines a per-date 30-day-return ranking closure, then delegates
    position-keeping and NAV accounting to the shared run_rotation_backtest
    engine. Optionally writes a trade ledger and summary JSON.

    Args:
        start: Inclusive backtest start date (first day positions can be held).
        end: Inclusive backtest end date.
        capital: Starting NAV in rupees.
        out_dir: Directory to write trade_ledger.json and summary.json; None
            skips file output.
        mid_month_check: If True, add a day-15 weekday rank check to the
            calendar (rotates only when the lead gate passes).
        mid_month_lead_pct: Minimum lead in percentage points for a mid-month
            rotation to fire.
        retain_top_n: Exit retention band — hold while the position stays in
            the top-N by 30d return, rotate when it drops out. 1 == legacy
            "rotate off rank-1" (canonical); 5 mirrors the LIVE exit.
        data_source: historical_data.data_source filter ("fyers" canonical).

    Returns:
        tuple: (final_nav, cagr_pct, trades) — final NAV in rupees, CAGR in
        percent, and the list of trade dicts from the engine.
    """
    # retain_top_n: hold the position while it stays in the top-N by 30d return;
    # rotate (sell + buy new rank-1) only when it drops OUT of the top-N band.
    # retain_top_n=1 == legacy "rotate whenever held isn't rank-1" (canonical
    # backtest). retain_top_n=5 mirrors the LIVE exit (live_signal.py keeps the
    # stock through top-5). Entry always buys rank-1.
    n100_syms = load_n100_union(index_name)
    print(f"NSE {index_name.upper()} universe (union of all snapshots): {len(n100_syms)} symbols")

    eng = _get_engine()
    with eng.connect() as c:
        # Pull 400 extra calendar days before `start` so the 30d lookback has
        # warm-up history available for the earliest rebalance dates.
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n100_syms, "a": start - timedelta(days=400), "b": end,
                      "ds": data_source})

    df["date"] = pd.to_datetime(df["date"])
    # Wide close-price matrix: rows = trading days, cols = symbols. ffill carries
    # the last known close forward over gaps so ranking never sees NaN holes.
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    dates = cl.index
    # Keep only universe symbols that actually have price columns loaded.
    present = [s for s in n100_syms if s in cl.columns]
    print(f"Loaded {len(dates)} days × {len(present)} symbols")

    # ---- Rebalance calendar from the SHARED core (same rule live mirrors) ----
    # full = month's first trading day; mid = first trading day on/after the 15th.
    calendar = build_calendar(dates, start, end, mid_check=mid_month_check)

    # SELECTION layer: model supplies the per-date ranking; EXECUTION is the
    # shared engine (tools/shared/backtest_engine). Universe at each rebal =
    # POINT-IN-TIME n100 members (eligible_at), ranked by 15-day return.
    def _pit_universe(di):
        """Symbols actually in NSE Nifty 100 on the date at row index `di`.

        Intersects eligible_at(d) (membership at that date) with the symbols
        that have a non-null price loaded for this day. Returns Fyers-style
        symbols matching `cl.columns`.
        """
        on_date = dates[di].date()
        elig = eligible_at(index_name, on_date)
        return [f"NSE:{s}-EQ" for s in elig
                if f"NSE:{s}-EQ" in cl.columns
                and pd.notna(cl[f"NSE:{s}-EQ"].iloc[di])]

    def rank_at(di):
        """Rank POINT-IN-TIME n100 by 15-day return at day-index `di`.

        Args:
            di: Integer row index into `cl` / `dates` for the rebalance day.

        Returns:
            list[str]: Symbols ordered best-to-worst by 15d return; empty if
            there is not enough warm-up history (di < LOOKBACK) or no symbol
            has a valid price.

        A held symbol that has left the index simply will not appear here;
        decide_rotation in the shared engine will then sell it and buy rank-1
        (no special delist branch needed).
        """
        if di < LOOKBACK:
            return []  # not enough history for a 15-day lookback yet
        univ = _pit_universe(di)
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    def midret_at(di):
        """Return (symbol, 15d_return_pct) pairs sorted desc for the mid-month
        lead-gate check, restricted to POINT-IN-TIME n100.

        Args:
            di: Integer row index into `cl` / `dates` for the mid-month day.

        Returns:
            list[tuple[str, float]]: (symbol, 15d_return_percent) best-first.
        """
        univ = _pit_universe(di)
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        # Multiply by 100 to express the return as percentage points (the unit
        # the lead-pct gate compares against).
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
        # Full per-trade ledger (one record per buy/sell) for later inspection.
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        # Headline metrics consumed by the model comparison/leaderboard tooling.
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
    ap.add_argument("--mid-month-check", dest="mid_month_check",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Day-15 rank check + lead gate. Default ON = the LIVE "
                         "config (cron runs the mid-month job). --no-mid-month-check "
                         "to disable. Mid ON is a big win: +87.5% vs +43.2% CAGR.")
    ap.add_argument("--mid-month-lead-pct", type=float, default=5.0,
                    help="Minimum lead (pp) for mid-month rotation. Default 5.0")
    ap.add_argument("--retain-top-n", type=int, default=3,
                    help="Hold while in top-N by 15d ret; rotate when out. "
                         "3 = current canonical (2026-05-28 sweep): wins BOTH "
                         "3yr and 10yr vs retain=1 (3yr +245% vs +184%, 10yr "
                         "+86.8% vs +67.1%, DD 53% vs 61%). 1 = legacy. "
                         "5 = pre-2026-05-26 live exit band.")
    ap.add_argument("--data-source", default="fyers",
                    help="historical_data.data_source filter. Default fyers "
                         "(canonical). Use yfinance only for local dev DBs.")
    ap.add_argument("--index", default="n100", choices=["n100", "n200", "n500"],
                    help="PIT index membership universe. Default n100 (canon). "
                         "n200/n500 broaden the candidate pool to the wider "
                         "NSE indices using the same rotation rule.")
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None,
        mid_month_check=a.mid_month_check,
        mid_month_lead_pct=a.mid_month_lead_pct,
        retain_top_n=a.retain_top_n,
        data_source=a.data_source,
        index_name=a.index)
