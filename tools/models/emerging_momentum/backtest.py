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
  Position: max_concurrent=1, retain_top_n=S.RETAIN=1 (hold while in top-1 rank).
  Rebalance: 1st trading day of each month ("full") + a mid-month ("mid")
             check on the first trading day with 15<=day<=18; the mid-month
             rotation fires only when a new leader beats the held name's 15d
             return by >= 5pp (MIDMONTH_LEAD).

CURRENT (vol-adj rank, RETAIN=1, lb30, +2.5×ATR stop) under the 2026-06-13
REALISM CONVENTION — net of real Fyers CNC charges, next-open fills (the
backtest fills every leg at the NEXT bar's OPEN and deducts charges per fill;
see FILL_AT_NEXT_OPEN / CHARGES below): full-cycle 2021-03→2026-06 on
authoritative PIT membership = +105.3% CAGR / 38.6% DD / Calmar 2.73 / 68
trades (charges ₹2.56M); 3-yr 2023-05→2026-05 = +138.9% CAGR / 27.4% DD /
Calmar 5.07 — UNLEVERED (own cash only). See
exports/models/emerging_momentum/SUMMARY.md. (The old close-fill zero-charge
convention showed +115.6%/37.9%/3.05 — a normal ~10pp charges haircut; the
much older +46.9% was the pre-vol-adj rotation-only config.) Decision logic is
byte-identical to live (PIT, no-lookahead, no borrow).

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
from tools.live.broker_charges import compute_charges
from tools.models.emerging_momentum import strategy as S

DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date(2026, 5, 29)
DEFAULT_CAP = 1_000_000.0

# ── BACKTEST REALISM CONVENTION (2026-06-13, identical across all 5 models) ──
# Decision LOGIC (lookbacks, ranks, gates, stop levels, cadence) is UNCHANGED;
# only fill price/timing and charges differ from the old close-fill version:
#   * FILL_AT_NEXT_OPEN: every decision made on bar d's close (rotation rank,
#     entry, profit-take detection, ATR-stop detection on bar d's low) FILLS at
#     bar d+1's OPEN — live ranks the last completed daily bar and executes
#     next morning ~09:30-09:41. If d is the last bar of the window, fill at
#     d's close (window-end bookkeeping). NAV marking stays close-based.
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


def load_panels(eng, start, end):
    """Load close + OPEN panels + ADV panel for the N500 universe (+ index mask).

    The open panel is needed for FILL_AT_NEXT_OPEN (decisions on bar d's close
    fill at bar d+1's open) — historical_data carries full OHLC, so opens come
    from the same source/rows as the closes. History is always loaded back to
    (POOL_ANCHOR_START - 420d) even for a later eval-window start so the
    per-year PIT pools (anchored to POOL_ANCHOR_START) and the 20d-ADV warmup
    match the full run exactly. Index-only date rows are dropped so they don't
    poison the rolling windows (same guard as live_signal).
    """
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    load_from = min(start, S.POOL_ANCHOR_START) - timedelta(days=420)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": load_from, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").astype(float)
    op = df.pivot(index="date", columns="symbol", values="open").astype(float)
    hi = df.pivot(index="date", columns="symbol", values="high").astype(float)
    lo = df.pivot(index="date", columns="symbol", values="low").astype(float)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    op = op.loc[equity_dates]  # NOT ffilled: a stale open is not a tradable
    # price — next_open_fill falls back to that bar's (ffilled) close instead.
    hi = hi.loc[equity_dates].ffill()
    lo = lo.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20 = S.indicators(cl, adv_rs)
    return cl, op, hi, lo, adv20


def _atr_panel(cl, hi, lo, win=None):
    """Per-symbol ATR (simple mean of True Range) for the daily stop check."""
    win = win or S.ATR_WIN
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()]).groupby(level=0).max()
    return tr.rolling(win).mean()


def run(start, end, capital, out_dir=None):
    """Daily-MTM single-position emerging backtest WITH the ATR-from-entry stop.

    SELECTION + rotation rule are the shared core (strategy.rank_pool /
    decide_rotation / midmonth_lead_ok) — identical to live. On TOP, a DAILY
    ATR-from-entry hard stop (strategy.atr_stop_hit, the SAME helper the live
    --stop-check uses) is checked every trading day. So live and backtest share
    both the selection AND the stop logic — no drift. Engine `run_rotation_backtest`
    is left untouched (other models unaffected).

    REALISM (see FILL_AT_NEXT_OPEN/CHARGES at module top): all fills happen at
    the NEXT bar's open with real Fyers CNC charges deducted from cash;
    detection/decision logic and close-based NAV marking are unchanged.
    """
    from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok, mid_month_retain
    eng = _get_engine()
    cl, op, hi, lo, adv20 = load_panels(eng, start, end)
    dates = cl.index
    anchors, pools = S.build_pools(adv20, dates, end)
    atr = _atr_panel(cl, hi, lo)
    calendar = S.build_calendar(dates, start, end)
    cal = {pd.Timestamp(d): k for d, k in calendar}

    def rank_at(di):
        return S.rank_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    def midret_at(di):
        return S.midret_pool(cl, S.pool_for_date(anchors, pools, dates[di]), di)

    def _lbl(s):
        return s.replace("NSE:", "").replace("-EQ", "")

    cap = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None; took = False
    _PT = float(getattr(S, "PROFIT_TAKE_PCT", 0.0) or 0.0)
    trades = []; nav_by_day = []; charges_total = 0.0
    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_by_day.append((d, cap + (qty * px if hold and px else 0.0)))
        # --- DAILY partial profit-take: book HALF once at entry*(1+PT) ---
        # Detection on bar d's CLOSE (unchanged); FILL at bar d+1's open.
        if hold and qty >= 2 and _PT > 0 and not took and px is not None \
                and px >= entry_px * (1 + _PT):
            sell = qty // 2
            fx, fdate = next_open_fill(op, cl, dates, hold, di)
            chg = fill_charges("SELL", sell, fx)
            cap += sell * fx - chg
            charges_total += chg
            trades.append({"sym": _lbl(hold), "entry_date": entry_date,
                "exit_date": fdate, "qty": sell,
                "entry_px": round(entry_px, 2), "exit_px": round(fx, 2),
                "pnl": round(sell * fx - sell * entry_px, 0),
                "ret_pct": round((fx / entry_px - 1) * 100, 2),
                "charges": round(chg, 2),
                "cap_after": round(cap, 0), "exit_reason": "PROFIT_TAKE"})
            qty -= sell; took = True
        # --- DAILY ATR-from-entry stop (shared helper) ---
        # Detection unchanged (bar d's low breaches the level); FILL at bar
        # d+1's open — live detects from yesterday's completed bar and sells
        # next morning, so the fill is gap-realistic, not the exact level.
        if hold and qty > 0:
            av = atr[hold].iloc[di] if hold in atr.columns else None
            day_low = float(lo[hold].iloc[di]) if pd.notna(lo[hold].iloc[di]) else px
            hit, lvl = S.atr_stop_hit(entry_px, float(av) if pd.notna(av) else None, day_low)
            if hit and lvl:
                fx, fdate = next_open_fill(op, cl, dates, hold, di)
                chg = fill_charges("SELL", qty, fx)
                cap += qty * fx - chg
                charges_total += chg
                trades.append({"sym": _lbl(hold), "entry_date": entry_date,
                    "exit_date": fdate, "qty": qty,
                    "entry_px": round(entry_px, 2), "exit_px": round(fx, 2),
                    "stop_lvl": round(lvl, 2),
                    "pnl": round(qty * fx - qty * entry_px, 0),
                    "ret_pct": round((fx / entry_px - 1) * 100, 2),
                    "charges": round(chg, 2),
                    "cap_after": round(cap, 0), "exit_reason": "ATR_STOP"})
                hold = None; qty = 0; entry_px = 0.0
        kind = cal.get(d)
        if kind is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        top = ranked[0]
        if kind == "mid":
            if not midmonth_lead_ok(hold, midret_at(di), S.MIDMONTH_LEAD):
                continue
            if decide_rotation(hold, ranked, retain_top_n=mid_month_retain(True, S.RETAIN)).is_noop:
                continue
            reason = "MIDCHECK"
        else:
            if decide_rotation(hold, ranked, retain_top_n=S.RETAIN).is_noop:
                continue
            reason = "ROTATE"
        # Rotation legs: decided on bar d's close (rank unchanged), BOTH the
        # sell and the buy fill at bar d+1's open with CNC charges.
        if hold and qty > 0:
            sx, sdate = next_open_fill(op, cl, dates, hold, di)
            chg = fill_charges("SELL", qty, sx)
            cap += qty * sx - chg
            charges_total += chg
            trades.append({"sym": _lbl(hold), "entry_date": entry_date,
                "exit_date": sdate, "qty": qty,
                "entry_px": round(entry_px, 2), "exit_px": round(sx, 2),
                "pnl": round(qty * sx - qty * entry_px, 0),
                "ret_pct": round((sx / entry_px - 1) * 100, 2),
                "charges": round(chg, 2),
                "cap_after": round(cap, 0), "exit_reason": reason})
            hold = None; qty = 0
        bx, bdate = next_open_fill(op, cl, dates, top, di)
        if bx > 0:
            q = size_buy_qty(cap, bx)
            if q >= 1:
                chg = fill_charges("BUY", q, bx)
                cap -= q * bx + chg
                charges_total += chg
                qty = q; hold = top; entry_px = bx
                entry_date = bdate; took = False

    final = cap
    open_pos = None
    if hold:
        last = float(cl[hold].iloc[-1])
        final = cap + qty * last
        open_pos = {"sym": _lbl(hold), "qty": qty, "entry_px": round(entry_px, 2),
                    "entry_date": entry_date, "last_px": round(last, 2)}
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    nser = pd.Series([v for _, v in nav_by_day],
                     index=pd.DatetimeIndex([dd for dd, _ in nav_by_day]))
    roll = nser.cummax()
    mdd = float(((roll - nser) / roll).max()) * 100 if len(nser) > 1 else 0.0
    calmar = cagr / max(0.01, mdd)
    res_per_year = {}
    for yy, g in nser.groupby(nser.index.year):
        if len(g) > 1:
            rl = g.cummax()
            res_per_year[int(yy)] = {"ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                                     "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1)}

    print(f"\n## RESULTS (emerging_momentum, single-position)")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final / capital - 1) * 100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (W={wins}, L={losses}, "
          f"WR={wins / max(1, wins + losses) * 100:.1f}%)")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Charges (Fyers CNC, all fills): Rs.{charges_total:,.0f}")

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
        "atr_stop_mult": S.ATR_STOP_MULT,
        "per_year": res_per_year,
        "charges_total": round(charges_total, 2),
        "fill_at_next_open": FILL_AT_NEXT_OPEN,
        "charges_model": CHARGES,
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
