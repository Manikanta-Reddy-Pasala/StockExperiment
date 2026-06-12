"""Standalone backtest: RETEST MOMENTUM (momentum_retest_n500).

Offline research/validation path. The SELECTION + ENTRY logic lives in the
shared core `strategy.py` (rank_targets / is_retest / params) — the SAME code
live_signal.py uses, so backtest and live cannot drift. This file only owns the
offline walk: load panels, build the monthly calendar, replay buy/sell, score.

Strategy (see strategy.py / SUMMARY.md): monthly pick top-K (K=4) momentum
leaders from the top-120-ADV liquid N500 pool; buy each within 20% of the
20-EMA; hold while in the top-4 rank. K=4 (2026-05-31 re-tune, was K2).

2026-06-13 REALISM CONVENTION (see FILL_AT_NEXT_OPEN / CHARGES below) +
PIT-before-ADV universe fix — net of real Fyers CNC charges, next-open fills:
full-cycle 2021-03→2026-05 = +58.7% CAGR / 34.0% DD / Calmar 1.73 / 183 trades
(charges ₹386,643); 3-yr 2023-05→2026-05 = +102.3% CAGR / 23.6% DD / Calmar
4.34. (The old flat-0.15%/side close-fill convention showed +57.3%/38.8%/1.48 —
the PIT-before-ADV fix was net-POSITIVE here, the only model to improve.)
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.market_cap import classify_pit
from tools.shared.index_membership import universe_union, eligible_at
from tools.live.broker_charges import compute_charges
from tools.models.momentum_retest_n500 import strategy as S

DEFAULT_START = date(2021, 3, 1)
DEFAULT_END = date(2026, 5, 29)
DEFAULT_CAP = 1_000_000.0

# ── BACKTEST REALISM CONVENTION (2026-06-13, identical across all 5 models) ──
# Decision LOGIC (lookbacks, ranks, gates, retest band, cadence) is UNCHANGED;
# only fill price/timing and charges differ from the old close-fill version:
#   * FILL_AT_NEXT_OPEN: every decision made on bar d's close (rotation rank,
#     retest entry, profit-take detection) FILLS at bar d+1's OPEN — live ranks
#     the last completed daily bar and executes next morning ~09:30-09:41. If d
#     is the last bar of the window, fill at d's close (window-end bookkeeping).
#     NAV marking stays close-based.
#   * CHARGES: real Fyers CNC charges (tools/live/broker_charges.compute_charges,
#     the same calculator the live ledger stamps on model_trades.charges_inr)
#     are computed on every fill's actual qty*price and deducted from cash.
#     No flat percentages; no charges on the final-day unrealized mark.
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
    # PIT N500 superset (2026-05-31): preload every symbol ever in NSE Nifty 500;
    # the PIT eligible_at("n500", date) filter is applied INSIDE rank_targets
    # (BEFORE the top-120 ADV cut — see run()), removing both the survivorship
    # bias of the old static list AND the 2026-06-13 PIT gap where historically
    # ineligible names displaced eligible ones at the ADV margin.
    # The OPEN panel is needed for FILL_AT_NEXT_OPEN — historical_data carries
    # full OHLC, so opens come from the same source/rows as the closes.
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    op = df.pivot(index="date", columns="symbol", values="open").astype(float)
    op = op.loc[cl.index]  # NOT ffilled: a stale open is not a tradable price —
    # next_open_fill falls back to that bar's (ffilled) close instead.
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    adv20, sma200, ema20 = S.indicators(cl, adv_rs)
    return cl, op, adv20, sma200, ema20, S.load_smallcap()


def run(start, end, capital, out_dir=None):
    """Monthly-rotation K=4 retest backtest.

    REALISM (see FILL_AT_NEXT_OPEN/CHARGES at module top): every decision is
    made on bar d's close — exactly the old logic — but the fill executes at
    bar d+1's open with real Fyers CNC charges deducted from cash. NAV is
    recorded at the TOP of each bar (close-based marks), so a fill decided on
    bar d first shows up in bar d+1's NAV — consistent with its d+1-open fill.
    """
    eng = _get_engine()
    cl, op, adv20, sma200, ema20, smallcap = load_panels(eng, start, end)
    dates = cl.index
    print(f"N500 pool loaded: {len(cl)} days x {len(cl.columns)} symbols")
    i0 = dates.searchsorted(pd.Timestamp(start)); i1 = dates.searchsorted(pd.Timestamp(end), side="right")
    rebals = set(); y, m = start.year, start.month
    while True:
        fut = dates[dates >= pd.Timestamp(y, m, 1)]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0].date() >= start:
            rebals.add(fut[0])
        m += 1
        if m > 12:
            m = 1; y += 1
    cash = capital; pos = {}; watch = []; trades = []; navs = []; nd = []
    charges_total = 0.0
    warm = max(S.LOOKBACK, S.SMA_LONG)
    _PT = float(getattr(S, "PROFIT_TAKE_PCT", 0.0) or 0.0)
    for di in range(i0, i1):
        d = dates[di]; crow = cl.iloc[di]
        # NAV first (close-based, BEFORE this bar's decisions): fills decided on
        # bar d execute at bar d+1's open, so they belong in bar d+1's NAV.
        mv = cash + sum(p["qty"] * float(crow.get(s)) for s, p in pos.items()
                        if pd.notna(crow.get(s)))
        navs.append(mv); nd.append(d)
        # --- DAILY partial profit-take: book HALF a holding once at entry*(1+PT)
        # Detection on bar d's CLOSE (unchanged); FILL at bar d+1's open. ---
        if _PT > 0 and pos:
            for s in list(pos.keys()):
                p = pos[s]; px = crow.get(s)
                if p.get("took") or p["qty"] < 2 or pd.isna(px):
                    continue
                px = float(px)
                if px >= p["entry"] * (1 + _PT):
                    sell = p["qty"] // 2
                    fx, fdate = next_open_fill(op, cl, dates, s, di)
                    chg = fill_charges("SELL", sell, fx)
                    cash += sell * fx - chg
                    charges_total += chg
                    trades.append({
                        "sym": s.replace("NSE:", "").replace("-EQ", ""),
                        "entry_date": p["in"], "exit_date": fdate,
                        "qty": sell, "entry_px": round(p["entry"], 2),
                        "exit_px": round(fx, 2),
                        "pnl": round(sell * fx - sell * p["entry"], 0),
                        "ret_pct": round((fx / p["entry"] - 1) * 100, 2),
                        "charges": round(chg, 2),
                        "cap_after": round(cash, 0), "exit_reason": "PROFIT_TAKE",
                        "cap": classify_pit(s, date.fromisoformat(p["in"])),
                    })
                    p["qty"] -= sell; p["took"] = True
        if d in rebals and di >= warm:
            # PIT-before-ADV (2026-06-13): pass eligible_at("n500", d) INTO
            # rank_targets so the top-120 ADV cut is taken over names actually
            # in Nifty 500 on this date (the old post-filter let historically
            # ineligible names displace eligible ones at the ADV margin).
            rk = S.rank_targets(cl, adv20, sma200, smallcap, di,
                                eligible=eligible_at("n500", d.date()))
            retset = set(rk[:S.RETAIN])
            for s in list(pos.keys()):                       # exit: out of retain band
                if s not in retset:
                    p = pos[s]
                    fx, fdate = next_open_fill(op, cl, dates, s, di)
                    chg = fill_charges("SELL", p["qty"], fx)
                    proc = p["qty"] * fx - chg
                    cash += proc
                    charges_total += chg
                    cap_after = cash + sum(
                        pp["qty"] * float(cl[ss].iloc[di])
                        for ss, pp in pos.items()
                        if ss != s and pd.notna(cl[ss].iloc[di]))
                    trades.append({
                        "sym": s.replace("NSE:", "").replace("-EQ", ""),
                        "entry_date": p["in"], "exit_date": fdate,
                        "qty": p["qty"], "entry_px": round(p["entry"], 2),
                        "exit_px": round(fx, 2),
                        "pnl": round(p["qty"] * fx - p["qty"] * p["entry"], 0),
                        "ret_pct": round((fx / p["entry"] - 1) * 100, 2),
                        "charges": round(chg, 2),
                        "cap_after": round(cap_after, 0),
                        "exit_reason": "ROTATE",
                        "cap": classify_pit(s, date.fromisoformat(p["in"])),
                    })
                    del pos[s]
            watch = [s for s in rk[:S.K] if s not in pos]
        if watch and di >= warm:                              # retest entry (daily)
            # Detection on bar d's close (is_retest vs the 20-EMA, unchanged);
            # the BUY fills at bar d+1's open. Same-bar rotation sells above
            # have already credited cash, so entries are funded by the morning
            # sells — same as live (~09:30-09:41 both legs).
            slots = S.K - len(pos)
            for s in list(watch):
                if s in pos or slots <= 0:
                    continue
                px, ev = crow.get(s), ema20[s].iloc[di]
                if S.is_retest(px, ev):
                    bx, bdate = next_open_fill(op, cl, dates, s, di)
                    q = size_buy_qty(cash / max(1, slots), bx)
                    if q >= 1:
                        chg = fill_charges("BUY", q, bx)
                        cash -= q * bx + chg
                        charges_total += chg
                        pos[s] = {"qty": q, "entry": bx, "in": bdate, "took": False}
                        slots -= 1; watch.remove(s)

    last = cl.iloc[i1 - 1]
    # Final NAV recomputed AFTER the loop: last-bar decisions fill at the last
    # close (window-end bookkeeping in next_open_fill), so navs[-1] — recorded
    # before those fills — must be refreshed to include their charges.
    final = cash + sum(p["qty"] * float(last.get(s)) for s, p in pos.items()
                       if pd.notna(last.get(s)))
    if navs:
        navs[-1] = final
    nav = pd.Series(navs, index=pd.DatetimeIndex(nd))
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100
    years = {}
    for yy, g in nav.groupby(nav.index.year):
        if len(g) < 2:
            continue
        rl = g.cummax()
        years[int(yy)] = {"ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                          "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1)}
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    open_positions = [{
        "sym": s.replace("NSE:", "").replace("-EQ", ""),
        "qty": p["qty"], "entry_px": round(p["entry"], 2),
        "entry_date": p["in"], "last_px": round(float(last[s]), 2),
        "mtm_value": round(p["qty"] * float(last[s]), 0),
        "unrealized_pnl": round(p["qty"] * (float(last[s]) - p["entry"]), 0),
        "cap": classify_pit(s, date.fromisoformat(p["in"])),
    } for s, p in pos.items() if pd.notna(last.get(s))]
    print("\n## RESULTS (RETEST MOMENTUM, true-PIT, next-open fills, net of Fyers CNC charges)")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%   Max DD: {mdd:.2f}%   Calmar: {cagr/max(0.01,mdd):.2f}")
    print(f"  Trades: {len(trades)} (WR {100*wins/max(1,len(trades)):.0f}%)")
    print(f"  Charges (Fyers CNC, all fills): Rs.{charges_total:,.0f}")
    for yy in sorted(years):
        print(f"    {yy}: {years[yy]['ret_pct']:+.1f}%  (intra-yr DD {years[yy]['dd_pct']:.1f}%)")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        (out_dir / "summary.json").write_text(json.dumps({
            "model": "momentum_retest_n500", "start": start.isoformat(),
            "end": end.isoformat(), "years": round(yrs, 3),
            "capital": capital, "final_nav": round(final, 0),
            "total_return_pct": round((final/capital-1)*100, 2),
            "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
            "calmar": round(cagr/max(0.01, mdd), 2),
            "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "open_positions": open_positions, "per_year": years,
            "charges_total": round(charges_total, 2),
            "fill_at_next_open": FILL_AT_NEXT_OPEN,
            "charges_model": CHARGES,
            "params": {"top_n": S.TOPN, "K": S.K, "retain": S.RETAIN,
                       "lookback": S.LOOKBACK, "mom_floor": S.MOM_FLOOR,
                       "max_price": S.MAX_PRICE, "retest_band": [S.RETEST_LO, S.RETEST_HI]},
        }, indent=2))
    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None)
