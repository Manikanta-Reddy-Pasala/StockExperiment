"""n40 (dir: n20_daily_large_only): WEEKLY rotation + NSE Nifty 100 filter.

Universe constrained to large-cap (top-40 ADV ∩ PIT Nifty 100) to cut Max DD vs
the unconstrained v1. Current metrics (2026-06-13 realism regen — net of real
Fyers CNC charges, next-open fills): full-cycle 2021-03→2026-05 = +28.4% CAGR /
43.9% DD / Calmar 0.65 / 138 trades (charges ₹468,281); 3-yr 2023-05→2026-05 =
+48.6% CAGR / 30.9% DD / Calmar 1.58 — see exports/models/n40/SUMMARY.md.
(Old close-fill zero-charge convention showed +48.1%/37.1%/1.30 — the ~20pp
haircut is slippage+charges compounding on the churn-heaviest weekly rotation.)

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

2026-06-13: BACKTEST REALISM CONVENTION applied (next-open fills + real Fyers
CNC charges) — see FILL_AT_NEXT_OPEN / CHARGES below. Expect somewhat lower
numbers than the historical close-fill, zero-charge runs.
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
from tools.shared.rotation_strategy import decide_rotation
from tools.shared.stops import fixed_stop_hit
from tools.live.broker_charges import compute_charges


from tools.models.n40.strategy import (  # noqa: E402  shared w/ live
    UNIV_SIZE, LOOKBACK, ADV_WIN, SMA_LONG, RETAIN, STOP_PCT, PROFIT_TAKE_PCT)
N100_CSV  = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")
DEFAULT_START = date(2021, 3, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0

# ── BACKTEST REALISM CONVENTION (2026-06-13, identical across all 5 models) ──
# Decision LOGIC (lookbacks, ranks, gates, stop levels, WEEKLY cadence) is
# UNCHANGED; only fill price/timing and charges differ from the old
# close-fill version:
#   * FILL_AT_NEXT_OPEN: every decision made on bar d's close (weekly rotation
#     rank, entry, profit-take detection, fixed-stop detection on bar d's low)
#     FILLS at bar d+1's OPEN — live ranks the last completed daily bar and
#     executes next morning ~09:30-09:41. If d is the last bar of the window,
#     fill at d's close (window-end bookkeeping). NAV marking stays close-based.
#   * CHARGES: real Fyers CNC charges (tools/live/broker_charges.compute_charges,
#     the same calculator the live ledger stamps on model_trades.charges_inr)
#     are computed on every fill's actual qty*price and deducted from cash.
#     No charges on the final-day unrealized mark.
FILL_AT_NEXT_OPEN = True
CHARGES = "fyers_cnc"


def fill_charges(side, qty, price):
    """Real Fyers CNC charges (₹) for one fill — same calculator as the live
    ledger (tools/live/broker_charges.py)."""
    return float(compute_charges(side, int(qty), float(price), "CNC")["total"])


def next_open_fill(op, cl, dates, sym, di):
    """Fill price per FILL_AT_NEXT_OPEN: bar di+1's OPEN (close fallback when
    the open is missing for that bar); if di is the LAST bar of the window,
    bar di's close. Returns (price, fill_date_iso)."""
    if di + 1 < len(dates):
        v = op[sym].iloc[di + 1] if sym in op.columns else None
        if v is None or pd.isna(v) or float(v) <= 0:
            v = cl[sym].iloc[di + 1]
        if pd.notna(v) and float(v) > 0:
            return float(v), dates[di + 1].date().isoformat()
    return float(cl[sym].iloc[di]), dates[di].date().isoformat()


def size_buy_qty(cap, px):
    """Max whole-share qty such that qty*px + BUY charges <= cap (charges come
    out of cash, so sizing must leave room — no negative cash)."""
    if px <= 0:
        return 0
    q = int(cap / px)
    while q >= 1 and q * px + fill_charges("BUY", q, px) > cap:
        q -= 1
    return max(q, 0)


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
            "SELECT symbol,date,open,close,low,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])
    # ADV proxy in rupees = close * volume (rupee turnover, not share count).
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    lo = df.pivot(index="date", columns="symbol", values="low").ffill()  # for the daily stop
    # OPEN panel for FILL_AT_NEXT_OPEN — NOT ffilled: a stale open is not a
    # tradable price; next_open_fill falls back to that bar's (ffilled) close.
    op = df.pivot(index="date", columns="symbol", values="open")
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

    # DAILY-MTM single-position walk WITH the from-entry FIXED-% hard stop
    # (shared tools.shared.stops.fixed_stop_hit — the SAME helper the live
    # --stop-check uses, so backtest/live cannot drift). Selection/rotation stays
    # the production rule: decide_rotation on WEEKLY calendar days, retain_top_n=
    # RETAIN. The stop is checked EVERY trading day on the LOW. STOP_PCT=0 ->
    # reproduces the rotation-only baseline exactly. (Replaces the shared engine,
    # which only checked the stop on calendar/weekly steps — too coarse for a
    # daily from-entry stop. Validated 2026-06-04: -12% lifts CAGR 41->48 / cuts
    # DD 41->37 / Calmar 0.99->1.30.)
    # REALISM (see FILL_AT_NEXT_OPEN/CHARGES at module top): detection stays on
    # bar d (close for PT/rotation, low for the fixed stop) but every fill —
    # including the stop — happens at bar d+1's OPEN with real Fyers CNC
    # charges deducted from cash. NAV marking stays close-based.
    _STOP = float(STOP_PCT or 0.0)
    _PT = float(PROFIT_TAKE_PCT or 0.0)
    cal = {pd.Timestamp(d): k for d, k in calendar}
    cash = capital; hold = None; q = 0; entry = 0.0; entry_dt = None; took = False
    trades = []; navdays = []; cap_marks = []; charges_total = 0.0
    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        navdays.append((d, cash + (q * px if hold and px else 0.0)))
        # --- DAILY partial profit-take: book HALF once at entry*(1+PT) ---
        # Detection on bar d's CLOSE (unchanged); FILL at bar d+1's open.
        if hold and q >= 2 and _PT > 0 and not took and px is not None and px >= entry * (1 + _PT):
            sell = q // 2
            fx, fdate = next_open_fill(op, cl, dates, hold, di)
            chg = fill_charges("SELL", sell, fx)
            cash += sell * fx - chg; charges_total += chg
            trades.append({"sym": hold.replace("NSE:", "").replace("-EQ", ""),
                           "entry_date": entry_dt, "exit_date": fdate,
                           "qty": sell, "entry_px": round(entry, 2), "exit_px": round(fx, 2),
                           "pnl": round(sell * fx - sell * entry, 0),
                           "ret_pct": round((fx / entry - 1) * 100, 2) if entry else 0.0,
                           "charges": round(chg, 2),
                           "cap_after": round(cash, 0), "exit_reason": "PROFIT_TAKE"})
            q -= sell; took = True
        # --- DAILY fixed-% stop (shared helper) ---
        # Detection unchanged (bar d's low breaches the level via
        # fixed_stop_hit); FILL at bar d+1's open — live detects from
        # yesterday's completed bar and sells next morning, so the fill is
        # gap-realistic, not the exact stop level.
        if hold and q > 0 and _STOP > 0:
            dlow = float(lo[hold].iloc[di]) if hold in lo.columns and pd.notna(lo[hold].iloc[di]) else px
            hit, lvl = fixed_stop_hit(entry, dlow, _STOP)
            if hit and lvl:
                fx, fdate = next_open_fill(op, cl, dates, hold, di)
                chg = fill_charges("SELL", q, fx)
                cash += q * fx - chg; charges_total += chg
                trades.append({"sym": hold.replace("NSE:", "").replace("-EQ", ""),
                               "entry_date": entry_dt, "exit_date": fdate,
                               "qty": q, "entry_px": round(entry, 2), "exit_px": round(fx, 2),
                               "stop_lvl": round(lvl, 2),
                               "pnl": round(q * fx - q * entry, 0),
                               "ret_pct": round((fx / entry - 1) * 100, 2) if entry else 0.0,
                               "charges": round(chg, 2),
                               "cap_after": round(cash, 0), "exit_reason": "FIXED_STOP"})
                cap_marks.append((pd.Timestamp(fdate), cash)); hold = None; q = 0; entry = 0.0
        if cal.get(d) is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        if decide_rotation(hold, ranked, retain_top_n=RETAIN).is_noop:
            continue
        # Rotation legs: decided on bar d's close (rank unchanged), BOTH the
        # sell and the buy fill at bar d+1's open with CNC charges.
        if hold and q > 0:
            sx, sdate = next_open_fill(op, cl, dates, hold, di)
            chg = fill_charges("SELL", q, sx)
            cash += q * sx - chg; charges_total += chg
            trades.append({"sym": hold.replace("NSE:", "").replace("-EQ", ""),
                           "entry_date": entry_dt, "exit_date": sdate,
                           "qty": q, "entry_px": round(entry, 2), "exit_px": round(sx, 2),
                           "pnl": round(q * sx - q * entry, 0),
                           "ret_pct": round((sx / entry - 1) * 100, 2) if entry else 0.0,
                           "charges": round(chg, 2),
                           "cap_after": round(cash, 0), "exit_reason": "ROTATE"})
            cap_marks.append((pd.Timestamp(sdate), cash)); hold = None; q = 0
        top = ranked[0]
        bx, bdate = next_open_fill(op, cl, dates, top, di)
        if bx > 0:
            n = size_buy_qty(cash, bx)
            if n >= 1:
                chg = fill_charges("BUY", n, bx)
                cash -= n * bx + chg; charges_total += chg
                q = n; hold = top; entry = bx; entry_dt = bdate; took = False
    final = cash + (q * float(cl[hold].iloc[-1]) if hold else 0.0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    _nav = pd.Series([v for _, v in navdays], index=pd.DatetimeIndex([d for d, _ in navdays]))
    _roll = _nav.cummax()
    # Daily MTM DD is the headline risk metric for this high-churn model;
    # mdd_realized (exit-day cap_after DD) is the secondary view.
    mdd_nav = float(((_roll - _nav) / _roll).max()) * 100 if len(_nav) > 1 else 0.0
    if cap_marks:
        _cap = pd.Series([v for _, v in cap_marks], index=pd.DatetimeIndex([d for d, _ in cap_marks]))
        _cr = _cap.cummax(); mdd_realized = float(((_cr - _cap) / _cr).max()) * 100 if len(_cap) > 1 else 0.0
    else:
        mdd_realized = 0.0
    calmar = cagr / max(0.01, mdd_nav)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    open_pos = ({"sym": hold.replace("NSE:", "").replace("-EQ", ""), "qty": q,
                 "entry_px": round(entry, 2)} if hold else None)
    per_year = {}
    for yy, g in _nav.groupby(_nav.index.year):
        if len(g) > 1:
            rl = g.cummax()
            per_year[int(yy)] = {"ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                                 "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1)}

    print(f"\n## v2 Large-only RESULTS")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (wins={wins}, losses={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"  Max DD (NAV MTM): {mdd_nav:.2f}%  (rebal cap_after: {mdd_realized:.2f}%)")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Charges (Fyers CNC, all fills): Rs.{charges_total:,.0f}")

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
            "per_year": per_year,
            "charges_total": round(charges_total, 2),
            "fill_at_next_open": FILL_AT_NEXT_OPEN,
            "charges_model": CHARGES,
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
