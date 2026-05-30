"""Standalone backtest: pseudo-N100 (ADV-rank from N500, yearly PIT rebuild, MINUS Small).

Full-cycle 2021-04→2026-05 on PIT N500 (fixed May anchor) ≈ +72.9% CAGR / 28.6%
DD / Calmar 2.54 — see exports/models/momentum_pseudo_n100_adv/SUMMARY.md.

Single-position monthly rotation (lb=30, max-1, top-1 / RET1), but universe =
top-100 by 20-day ADV at each yearly anchor from PIT N500 instead of the real
NSE Nifty 100 — the deliberately-optimistic (ADV-biased) sibling of n100.

Model flow / where this file sits:
  data_pull.py    -> pulls N500 daily OHLCV + rebuilds yearly_universes.json
  build_universe.py -> ranks N500 by 20d ADV to produce a PIT top-100 snapshot
  live_signal.py  -> ranks today's universe, reads open position from DB,
                     emits SELL / ENTRY1 signals (production path)
  cron.py         -> schedules the data + signal + execute jobs
  backtest.py     -> (THIS FILE) the offline research/validation path. It
                     rebuilds the same selection logic over a date range and
                     defers the buy/sell mechanics to the SHARED execution
                     engine (tools/shared/backtest_engine.run_rotation_backtest),
                     which in turn calls the SHARED rotation core
                     (tools/shared/rotation_strategy.decide_rotation) — the very
                     same rule live_signal.py uses, so the two cannot drift.

Run directly with python (see argparse block at the bottom) to reproduce the
headline numbers and dump a trade ledger + summary JSON.
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.backtest_engine import run_rotation_backtest
from tools.shared.index_membership import universe_union, eligible_at


from tools.models.momentum_pseudo_n100_adv.strategy import (  # noqa: E402  shared w/ live
    LOOKBACK, ADV_WIN, UNIV_SIZE, MAX_PRICE, RETAIN, MIDMONTH_LEAD,
    UNIVERSE_ANCHOR_MONTH, UNIVERSE_ANCHOR_DAY, build_calendar)
# Drop Small-cap NSE Nifty Smallcap 250 stocks from universe (free +2pp CAGR, DD unchanged).
import csv as _csv
_SML_PATH = str(ROOT / "src" / "data" / "symbols" / "nifty_smallcap250.csv")
def _load_smallcap():
    """Load the set of Nifty Smallcap 250 plain symbols (EQ series only).

    These names are subtracted from the pseudo-N100 universe — a backtest
    sweep showed dropping them gives ~+2pp CAGR with drawdown unchanged.

    Returns:
        set[str]: plain NSE symbols (e.g. "TATAPOWER"), empty if CSV missing.
    """
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
DEFAULT_START = date(2021, 4, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0


def run(start: date, end: date, capital: float, out_dir: Path | None = None,
        retain_top_n: int = RETAIN, data_source: str = "fyers",
        mid_month_check: bool = False, mid_month_lead_pct: float = MIDMONTH_LEAD):
    """Run the full pseudo-N100 momentum-rotation backtest.

    Builds the price/ADV panels from historical_data, rebuilds the
    yearly-PIT universe, computes the monthly rebalance calendar, defines a
    per-day ranking function, then hands selection + execution off to the
    shared engine. Prints a result summary and optionally writes a trade
    ledger + summary JSON.

    Args:
        start: First date to trade (inclusive).
        end: Last date to trade (inclusive).
        capital: Starting NAV in rupees.
        out_dir: If given, write trade_ledger.json + summary.json here.
        retain_top_n: Exit retention band — hold while the held name stays in
            the top-N by 30d return; rotate (sell + buy rank-1) only when it
            drops OUT of top-N. 1 = legacy/canonical (rotate off rank-1),
            5 = the LIVE exit band live_signal.py historically used. Entry
            always buys rank-1.
        data_source: historical_data.data_source filter ("fyers" canonical).

    Returns:
        tuple[float, float, list]: (final_nav, cagr_pct, trades).
    """
    eng = _get_engine()
    # PIT N500 superset (2026-05-31): preload every symbol that was EVER in NSE
    # Nifty 500 across the authoritative snapshots, then restrict each yearly
    # universe to the members eligible AT that anchor date (eligible_at). Removes
    # the survivorship bias of the old static nifty500_symbols() current list.
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    print(f"N500 PIT union pool: {len(n500)}")

    # Pull 400 extra calendar days before `start` so the 200d SMA + 20d ADV
    # rolling windows are already warm on day one of the backtest.
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end,
                      "ds": data_source})

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)  # ₹ traded per day
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(ADV_WIN).mean()  # 20d Average Daily ₹ Value — the universe ranking metric
    sma200 = cl.rolling(200).mean()  # 200d SMA — uptrend filter baseline
    dates = cl.index

    # Yearly-PIT universe rebuild: snapshot the top-100-by-ADV list at a FIXED
    # calendar anchor (mid-May, matching live's annual rebuild) each year — NOT
    # the backtest start month. Fixed anchors make the yearly ADV snapshots
    # identical regardless of the backtest window, so absolute CAGR no longer
    # drifts with the start date and the backtest matches the live universe.
    # Range starts a year before `start` so a pre-May start still has the prior
    # year's anchor in force on day one.
    year_starts = [pd.Timestamp(yr, UNIVERSE_ANCHOR_MONTH, UNIVERSE_ANCHOR_DAY)
                   for yr in range(start.year - 1, end.year + 1)]

    year_universes = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0: continue
        di = dates.get_loc(fut[0])  # first trading-day index on/after the anchor
        # PIT N500 membership at the anchor date — only rank names actually in
        # Nifty 500 then (no survivorship bias).
        elig500 = eligible_at("n500", ys.date())
        pit_adv = adv20.iloc[di].dropna().sort_values(ascending=False)
        pit_adv = pit_adv[[s for s in pit_adv.index
                           if s.replace("NSE:", "").replace("-EQ", "") in elig500]]
        top = pit_adv.head(UNIV_SIZE).index.tolist()
        # Drop Small-cap names from top-100 (sweep showed +2pp CAGR, DD unchanged)
        year_universes[ys] = [s for s in top if s.replace("NSE:","").replace("-EQ","") not in _SMALLCAP]

    def pick_universe(d):
        """Return the PIT universe in force on date `d` (latest anchor <= d)."""
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:  # walk forward, keep the most recent anchor not after d
                chosen = ys
        return year_universes.get(chosen, [])

    # Monthly (+ optional mid-month) rebalance calendar from the SHARED core.
    # Mid-month ON is the 2026-05-30 config (matches n100): a day-15 lead check.
    calendar = build_calendar(dates, start, end, mid_check=mid_month_check)

    # SELECTION layer: yearly-PIT pseudo-N100, uptrend (>200d SMA) + MAX_PRICE
    # filter, ranked by 30-day return. EXECUTION is the shared engine.
    def rank_at(di):
        """Ranking callback invoked by the shared engine on each rebalance day.

        Args:
            di: Integer index into `dates`/`cl` for the rebalance day.

        Returns:
            list[str]: Eligible symbols ordered best-to-worst by 30d return,
            or [] when warm-up history is short or nothing passes the filters.
        """
        if di < max(LOOKBACK, 200):
            return []  # not enough history for both the 30d return and 200d SMA
        univ = pick_universe(dates[di])
        up = sma200.iloc[di] < cl.iloc[di]  # uptrend gate: close above its 200d SMA
        univ = [s for s in univ if bool(up.get(s, False))]
        # MAX_PRICE gate: skip names trading above ₹3000 (giant-loser guard).
        univ = [s for s in univ
                if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= MAX_PRICE]
        if not univ:
            return []
        # Rank by 30-day (LOOKBACK) trailing return, highest momentum first.
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    def midret_at(di):
        """(symbol, LOOKBACK-return-pct) pairs sorted desc for the mid-month
        lead gate — same eligible set (uptrend + MAX_PRICE) as rank_at."""
        univ = pick_universe(dates[di])
        up = sma200.iloc[di] < cl.iloc[di]
        univ = [s for s in univ if bool(up.get(s, False))
                and pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= MAX_PRICE]
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        rk = rets.dropna().sort_values(ascending=False)
        return [(s, float(rk[s]) * 100) for s in rk.index]

    res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at,
        capital=capital, start=start, end=end, retain_top_n=retain_top_n,
        midmonth_ret_at=midret_at if mid_month_check else None,
        midmonth_lead_pct=mid_month_lead_pct,
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
    ap.add_argument("--retain-top-n", type=int, default=RETAIN,
                    help=f"Hold while in top-N by 30d ret; rotate when out. "
                         f"Default {RETAIN} (top-1 rotation — wins on the fixed anchor).")
    ap.add_argument("--mid-month-check", dest="mid_month_check",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Day-15 rank check + lead gate. Default OFF (2026-05-31: "
                         "the mid-month 'win' was an artifact of the old start-anchored "
                         "universe; it loses on the fixed May anchor). Opt-in only.")
    ap.add_argument("--mid-month-lead-pct", type=float, default=MIDMONTH_LEAD,
                    help=f"Minimum lead (pp) for mid-month rotation. Default {MIDMONTH_LEAD}")
    ap.add_argument("--data-source", default="fyers",
                    help="historical_data.data_source filter. Default fyers "
                         "(canonical). Use yfinance only for local dev DBs.")
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None,
        retain_top_n=a.retain_top_n,
        data_source=a.data_source,
        mid_month_check=a.mid_month_check,
        mid_month_lead_pct=a.mid_month_lead_pct)
