"""Emerging-momentum optimisation sweep: ABSOLUTE PRICE FLOOR + ATR TRAILING STOP.

Research only (NOT wired into the model). Tests the two ideas the user proposed
against the current emerging_momentum config:

  1. Absolute price floor (e.g. >= Rs.100 / Rs.150) at ENTRY — drop low-priced
     "pump" names regardless of momentum score.
  2. Dynamic ATR-based TRAILING stop (k * ATR from the running peak) instead of /
     in addition to a fixed % stop — checked DAILY.

The production engine (tools.shared.backtest_engine.run_rotation_backtest) is
rebalance-day only (no daily mark-to-market, no intra-month trailing). So this
script reuses the EXACT emerging SELECTION (tools.models.emerging_momentum.
strategy: build_pools / pool_for_date / rank_pool / midret_pool) and the same
rebalance CALENDAR + rotation RULE (decide_rotation / midmonth_lead_ok), but
runs its OWN single-position DAILY-MTM simulator so an ATR trailing stop can be
checked every trading day.

Validation: scenario "baseline (no floor, no stop)" should reproduce the
production engine's numbers (printed side-by-side as a sanity check).

Run:
  python3 tools/analysis/emerging_pricefloor_atr_sweep.py \
      --from 2023-05-15 --to 2026-05-12 --capital 1000000
"""
import sys, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union
from tools.shared.backtest_engine import run_rotation_backtest
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok, mid_month_retain
from tools.models.emerging_momentum import strategy as S

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END = date(2026, 5, 12)
DEFAULT_CAP = 1_000_000.0
ATR_WIN = 14


def load_panels(eng, start, end):
    """close + high + low + adv panels for the N500 universe (+ index mask).

    Mirrors emerging backtest.load_panels (same load_from warmup + equity-day
    mask + ffill) but ALSO loads high/low so we can compute ATR. close/high/low
    are ffilled on equity trading days; adv20 for the per-year pool build.
    """
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    load_from = min(start, S.POOL_ANCHOR_START) - timedelta(days=420)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": load_from, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").astype(float)
    hi = df.pivot(index="date", columns="symbol", values="high").astype(float)
    lo = df.pivot(index="date", columns="symbol", values="low").astype(float)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    hi = hi.loc[equity_dates].ffill()
    lo = lo.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20 = S.indicators(cl, adv_rs)
    return cl, hi, lo, adv20


def atr_panel(cl, hi, lo, win=ATR_WIN):
    """Per-symbol ATR (simple mean of True Range over `win` days)."""
    prev_close = cl.shift(1)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ]).groupby(level=0).max()
    return tr.rolling(win).mean()


def daily_sim(dates, cl, lo, atr, anchors, pools, calendar, *, capital,
              start, end, retain, midmonth_lead,
              price_floor=0.0, atr_mult=None, fixed_stop_pct=None,
              entry_stop_pct=None, atr_from_entry=None, use_low_for_stop=True):
    """Single-position DAILY-MTM emerging simulator.

    Rotation on calendar days mirrors run_rotation_backtest EXACTLY (same
    decide_rotation / midmonth_lead_ok / mid_month_retain). Additions:
      * price_floor: an entry candidate must have close >= price_floor.
      * atr_mult: if set, a trailing stop at peak_close - atr_mult*ATR is checked
        EVERY trading day; a hit exits to cash (stays flat until the next
        calendar buy). use_low_for_stop -> stop triggers when the day's LOW
        pierces the level (fill at the stop level); else when CLOSE <= level.
    Entry/exit at close (rotation) / stop level (stop), gross of fees — same
    convention as the production engine.
    """
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    nav_marks = [capital]
    cash = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None; peak = 0.0
    trades = []

    def rank_at(di):
        r = S.rank_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)
        if price_floor > 0:
            r = [s for s in r if float(cl[s].iloc[di]) >= price_floor]
        return r

    def midret_at(di):
        return S.midret_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    # iterate every trading day from the first calendar date onward
    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        # --- daily MTM mark ---
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_marks.append(cash + (qty * px if hold and px else 0.0))

        # --- daily stop, checked every day. TRAILING (from running peak):
        #     atr_mult / fixed_stop_pct. FROM-ENTRY (fixed level, only cuts
        #     losers, lets winners run to the rotation exit): entry_stop_pct /
        #     atr_from_entry. ---
        if (atr_mult or fixed_stop_pct or entry_stop_pct or atr_from_entry) \
                and hold and qty > 0:
            if px:
                peak = max(peak, px)
            stop_level = None
            if atr_mult:
                a = atr[hold].iloc[di] if hold in atr.columns else None
                if a is not None and pd.notna(a) and a > 0 and peak > 0:
                    stop_level = peak - atr_mult * float(a)
            elif fixed_stop_pct and peak > 0:
                stop_level = peak * (1 - fixed_stop_pct)
            elif entry_stop_pct and entry_px > 0:
                stop_level = entry_px * (1 - entry_stop_pct)
            elif atr_from_entry and entry_px > 0:
                a = atr[hold].iloc[di] if hold in atr.columns else None
                if a is not None and pd.notna(a) and a > 0:
                    stop_level = entry_px - atr_from_entry * float(a)
            if stop_level is not None and stop_level > 0:
                day_low = float(lo[hold].iloc[di]) if pd.notna(lo[hold].iloc[di]) else px
                hit = (day_low is not None and day_low <= stop_level) if use_low_for_stop \
                    else (px is not None and px <= stop_level)
                if hit:
                    sx = stop_level if use_low_for_stop else px
                    cash += qty * sx
                    trades.append(_trade(hold, entry_date, d, qty, entry_px, sx, cash, "STOP"))
                    hold = None; qty = 0; entry_px = 0.0; peak = 0.0
                    # stay flat until the next calendar buy

        # --- rebalance only on calendar days ---
        kind = cal.get(d)
        if kind is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        top = ranked[0]
        if kind == "mid":
            if not midmonth_lead_ok(hold, midret_at(di), midmonth_lead):
                continue
            if decide_rotation(hold, ranked, retain_top_n=mid_month_retain(True, retain)).is_noop:
                continue
            reason = "MIDCHECK"
        else:
            if decide_rotation(hold, ranked, retain_top_n=retain).is_noop:
                continue
            reason = "ROTATE"
        # SELL leg (at close)
        if hold and qty > 0:
            sx = float(cl[hold].iloc[di])
            cash += qty * sx
            trades.append(_trade(hold, entry_date, d, qty, entry_px, sx, cash, reason))
            hold = None; qty = 0
        # BUY leg (rank-1, at close)
        bx = float(cl[top].iloc[di])
        if bx > 0:
            q = int(cash / bx)
            if q >= 1 and q * bx <= cash:
                cash -= q * bx; qty = q; hold = top; entry_px = bx
                entry_date = d.date().isoformat(); peak = bx

    final = cash
    if hold:
        final = cash + qty * float(cl[hold].iloc[-1])
    return _metrics(final, capital, trades, nav_marks, start, end)


def _trade(sym, ed, d, qty, ep, xp, cap_after, reason):
    return {"sym": sym.replace("NSE:", "").replace("-EQ", ""), "entry_date": ed,
            "exit_date": d.date().isoformat(), "qty": qty, "entry_px": round(ep, 2),
            "exit_px": round(xp, 2), "pnl": round(qty * xp - qty * ep, 0),
            "ret_pct": round((xp / ep - 1) * 100, 2) if ep else 0.0,
            "cap_after": round(cap_after, 0), "exit_reason": reason}


def _metrics(final, capital, trades, nav_marks, start, end):
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    nav = pd.Series(nav_marks)
    roll = nav.cummax()
    mdd = float(((roll - nav) / roll).max()) * 100 if len(nav) > 1 else 0.0
    return {
        "final_nav": round(final, 0), "net_pnl": round(final - capital, 0),
        "total_return_pct": round((final / capital - 1) * 100, 1),
        "cagr_pct": round(cagr, 1), "max_dd_pct": round(mdd, 1),
        "trades": len(trades), "wins": wins, "losses": losses,
        "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", "--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--end", "--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
    cap = a.capital

    eng = _get_engine()
    print(f"Loading panels {start}..{end} ...")
    cl, hi, lo, adv20 = load_panels(eng, start, end)
    dates = cl.index
    anchors, pools = S.build_pools(adv20, dates, end)
    calendar = S.build_calendar(dates, start, end)
    atr = atr_panel(cl, hi, lo)

    # --- sanity: production engine baseline ---
    def rank_at(di):
        return S.rank_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    def midret_at(di):
        return S.midret_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)
    eng_res = run_rotation_backtest(
        dates=dates, close=cl, calendar=calendar, rank_at=rank_at, capital=cap,
        start=start, end=end, retain_top_n=S.RETAIN, midmonth_ret_at=midret_at,
        midmonth_lead_pct=S.MIDMONTH_LEAD)
    print(f"\n[engine baseline] ret {eng_res.cagr_pct:+.1f}% CAGR / DD {eng_res.max_dd_pct:.1f}% "
          f"/ trades {len(eng_res.trades)} W{eng_res.wins} L{eng_res.losses}")

    scenarios = [
        ("Baseline (daily-sim)",        dict(price_floor=0,   atr_mult=None)),
        ("Price floor Rs.100",          dict(price_floor=100, atr_mult=None)),
        ("Price floor Rs.150",          dict(price_floor=150, atr_mult=None)),
        ("ATR trail 2x",                dict(price_floor=0,   atr_mult=2.0)),
        ("ATR trail 3x",                dict(price_floor=0,   atr_mult=3.0)),
        ("ATR trail 4x",                dict(price_floor=0,   atr_mult=4.0)),
        ("ATR trail 5x",                dict(price_floor=0,   atr_mult=5.0)),
        ("Fixed trail 20%",             dict(price_floor=0,   fixed_stop_pct=0.20)),
        ("ENTRY stop 8% (from-entry)",  dict(price_floor=0,   entry_stop_pct=0.08)),
        ("ENTRY stop 12%",              dict(price_floor=0,   entry_stop_pct=0.12)),
        ("ENTRY stop 20%",              dict(price_floor=0,   entry_stop_pct=0.20)),
        ("ATR-from-entry 2x",           dict(price_floor=0,   atr_from_entry=2.0)),
        ("ATR-from-entry 3x",           dict(price_floor=0,   atr_from_entry=3.0)),
        ("ENTRY 8% + ATR-entry 3x best",dict(price_floor=0,   entry_stop_pct=0.08)),
    ]
    rows = []
    base_pnl = None
    for name, kw in scenarios:
        m = daily_sim(dates, cl, lo, atr, anchors, pools, calendar,
                      capital=cap, start=start, end=end, retain=S.RETAIN,
                      midmonth_lead=S.MIDMONTH_LEAD, **kw)
        if base_pnl is None:
            base_pnl = m["net_pnl"]
        vs = "—" if m["net_pnl"] == base_pnl else \
            f"{(m['net_pnl'] / base_pnl - 1) * 100:+.0f}%" if base_pnl else "n/a"
        rows.append((name, m, vs))

    print(f"\n{'Scenario':<26}{'Trades':>7}{'Loss':>6}{'Win%':>7}"
          f"{'NetP&L':>14}{'CAGR':>8}{'MaxDD':>8}{'vsBase':>9}")
    for name, m, vs in rows:
        print(f"{name:<26}{m['trades']:>7}{m['losses']:>6}{m['win_rate_pct']:>6.0f}%"
              f"{('Rs.' + format(m['net_pnl'], ',.0f')):>14}{m['cagr_pct']:>7.1f}%"
              f"{m['max_dd_pct']:>7.1f}%{vs:>9}")
    print("\n(gross-of-fee, single-position, daily-MTM DD. ATR=14d simple. "
          "vsBase = net-P&L vs the daily-sim baseline.)")


if __name__ == "__main__":
    main()
