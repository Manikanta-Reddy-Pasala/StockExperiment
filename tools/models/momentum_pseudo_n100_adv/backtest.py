"""Standalone backtest: pseudo-N100 (ADV-rank from N500, yearly PIT rebuild, NO smallcap excl).

2026-06-13 "nosml" rework + REALISM CONVENTION (next-open fills + real Fyers
CNC charges + PIT N500, see FILL_AT_NEXT_OPEN / CHARGES below) — net of charges:
full-cycle 2021-06→2026-06 = +66.5% CAGR / 44.9% DD / Calmar 1.48 / 51 trades /
72.5% win; 3-yr 2023-06→2026-06 = +109.3% / 44.9% DD / Calmar 2.43 / 80% win;
since Mar-2025 = +158.8% / 25.8% DD / Calmar 6.15 — see
exports/models/momentum_pseudo_n100_adv/SUMMARY.md.

WHY nosml (EXCLUDE_SMALLCAP=False in strategy.py): the old Smallcap-250
exclusion was SURVIVORSHIP-BIASED — it applied TODAY's Smallcap-250 list to
every historical year, deleting the ADV-rising midcap winners the model rides.
Under correct PIT snapshots it collapsed full-cycle CAGR to ~13% (the prior
published +77.4% was that bias). Dropping the exclusion restored a
walk-forward-validated edge: stitched OOS 2023→2026 +60.3% CAGR / Calmar 1.34
vs the old smallcap-excluded config +23.8% / 0.51, beating every fold
(adversarially re-verified). The snapshot machinery (smallcap_at) is retained
only to reproduce the retired biased config (flip EXCLUDE_SMALLCAP True).
NOTE: HIGH-DD sleeve (~45% full-cycle) — size accordingly in the blend.

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
from tools.shared.rotation_strategy import (
    decide_rotation, midmonth_lead_ok, mid_month_retain)
from tools.shared.index_membership import universe_union, eligible_at, _TICKER_ALIAS
from tools.live.broker_charges import compute_charges


from tools.models.momentum_pseudo_n100_adv import strategy as S  # noqa: E402
from tools.models.momentum_pseudo_n100_adv.strategy import (  # noqa: E402  shared w/ live
    LOOKBACK, ADV_WIN, UNIV_SIZE, MAX_PRICE, RETAIN, MIDMONTH_LEAD, SMA_GATE,
    EXCLUDE_SMALLCAP, UNIVERSE_ANCHOR_MONTH, UNIVERSE_ANCHOR_DAY, build_calendar)

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


# ── PIT Nifty Smallcap 250 exclusion (2026-06-13 fix) ────────────────────────
# These names are subtracted from the pseudo-N100 universe — a backtest sweep
# had showed dropping them gives ~+2pp CAGR with drawdown unchanged, but that
# finding is INVALIDATED under PIT snapshots (the +2pp was survivorship bias —
# see module docstring; full-cycle CAGR collapses 77→13% once the exclusion is
# period-correct). Filter kept for live parity pending strategy review. The OLD code
# loaded only TODAY's nifty_smallcap250.csv and applied it to every year
# 2021-2026 (NON-PIT: 2021 overlap with the current list is only ~52%). Now
# each yearly universe anchor uses the period-correct snapshot from
# src/data/symbols/index_history/smallcap250_YYYYMMDD.csv — the same
# last-known-state rule as tools/shared/index_membership.eligible_at:
# latest snapshot whose date <= anchor; before the first snapshot, fall back
# to the first snapshot (best approximation available).
import csv as _csv
_SML_DIR = ROOT / "src" / "data" / "symbols" / "index_history"
_SML_CUR_PATH = str(ROOT / "src" / "data" / "symbols" / "nifty_smallcap250.csv")


def _load_smallcap_current():
    """Fallback ONLY (no PIT snapshots on disk): today's Nifty Smallcap 250
    plain symbols (EQ series) from the current NSE list CSV."""
    out = set()
    try:
        with open(_SML_CUR_PATH) as f:
            for r in _csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out.add(r["Symbol"].strip())
    except FileNotFoundError:
        pass
    return out


def _load_smallcap_snapshots():
    """{snapshot_date: set[plain symbols]} from the PIT smallcap250_*.csv files.

    Symbols are mapped through index_membership._TICKER_ALIAS (and the raw
    period symbol kept too) so renamed names subtract correctly against the
    alias-mapped N500 universe the panel is built from.
    """
    snaps = {}
    for p in sorted(_SML_DIR.glob("smallcap250_*.csv")):
        ds = p.stem.split("_")[-1]
        try:
            snap_d = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        except (ValueError, IndexError):
            continue
        syms = set()
        with open(p) as f:
            for r in _csv.DictReader(f):
                s = (r.get("symbol") or r.get("Symbol") or "").strip()
                if s:
                    syms.add(s)
                    syms.add(_TICKER_ALIAS.get(s, s))
        if syms:
            snaps[snap_d] = syms
    return snaps


_SML_SNAPS = _load_smallcap_snapshots()


def smallcap_at(d):
    """PIT Nifty Smallcap 250 set in force on date `d` (latest snapshot <= d;
    earliest snapshot if d precedes all; current-list fallback if none)."""
    if not _SML_SNAPS:
        return _load_smallcap_current()
    snap_dates = sorted(_SML_SNAPS)
    chosen = snap_dates[0]
    for sd in snap_dates:
        if sd <= d:
            chosen = sd
        else:
            break
    return _SML_SNAPS[chosen]


DEFAULT_START = date(2021, 3, 1)
DEFAULT_END   = date(2026, 6, 12)
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
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end,
                      "ds": data_source})

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)  # ₹ traded per day
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    # Opens for FILL_AT_NEXT_OPEN — same source rows as the closes. NOT ffilled:
    # a stale open is not a tradable price — next_open_fill falls back to that
    # bar's (ffilled) close instead.
    op = df.pivot(index="date", columns="symbol", values="open").astype(float)
    _hi = df.pivot(index="date", columns="symbol", values="high").ffill()
    _lo = df.pivot(index="date", columns="symbol", values="low").ffill()
    # ATR panel for the from-entry hard stop (shared logic with live --stop-check).
    _prevc = cl.shift(1)
    _tr = pd.concat([(_hi - _lo).abs(), (_hi - _prevc).abs(), (_lo - _prevc).abs()]).groupby(level=0).max()
    atr = _tr.rolling(S.ATR_WIN).mean()
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
        # Smallcap-250 exclusion — DROPPED by default ("nosml", 2026-06-13,
        # EXCLUDE_SMALLCAP=False in strategy.py). The PIT exclusion deleted the
        # ADV-rising midcap winners the model rides (collapsed CAGR to ~11%);
        # removing it = +69.8% CAGR / 44.9% DD, walk-forward-validated. The
        # snapshot machinery (smallcap_at) is retained for reproducing the
        # retired biased config (flip the flag True).
        if EXCLUDE_SMALLCAP:
            sml = smallcap_at(ys.date())
            top = [s for s in top if s.replace("NSE:","").replace("-EQ","") not in sml]
        year_universes[ys] = top

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
        if SMA_GATE:
            up = sma200.iloc[di] < cl.iloc[di]  # uptrend gate: close above 200d SMA
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

    # DAILY-MTM single-position walk WITH the from-entry ATR hard stop (shared
    # tools.shared.stops.atr_stop_hit — the SAME helper live --stop-check uses,
    # so backtest and live cannot drift). Selection/rotation identical to the
    # production rule (decide_rotation / midmonth gate). ATR_STOP_MULT=0 -> the
    # walk reproduces the rotation-only baseline exactly.
    from tools.shared.stops import atr_stop_hit as _atr_hit
    cal = {pd.Timestamp(d): k for d, k in calendar}
    cash = capital; hold = None; q = 0; entry = 0.0; entry_dt = None; took = False
    _PT = float(getattr(S, "PROFIT_TAKE_PCT", 0.0) or 0.0)
    trades = []; navs = []; navdays = []; charges_total = 0.0
    first_di = dates.get_loc(min(cal)) if cal else 0
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_t = cash + (q * px if hold and px else 0.0)
        navs.append(nav_t); navdays.append((d, nav_t))
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
        # daily from-entry ATR stop — detection unchanged (bar d's low breaches
        # the level, shared helper); FILL at bar d+1's open — live detects from
        # yesterday's completed bar and sells next morning, so the fill is
        # gap-realistic, not the exact level.
        if hold and q > 0 and S.ATR_STOP_MULT and S.ATR_STOP_MULT > 0:
            av = atr[hold].iloc[di] if hold in atr.columns else None
            dlow = float(_lo[hold].iloc[di]) if pd.notna(_lo[hold].iloc[di]) else px
            hit, lvl = _atr_hit(entry, float(av) if av is not None and pd.notna(av) else None,
                                dlow, S.ATR_STOP_MULT)
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
                               "cap_after": round(cash, 0), "exit_reason": "ATR_STOP"})
                hold = None; q = 0; entry = 0.0
        kind = cal.get(d)
        if kind is None:
            continue
        ranked = rank_at(di)
        if not ranked:
            continue
        top = ranked[0]
        if kind == "mid" and mid_month_check:
            if not midmonth_lead_ok(hold, midret_at(di), mid_month_lead_pct):
                continue
            if decide_rotation(hold, ranked, retain_top_n=mid_month_retain(True, retain_top_n)).is_noop:
                continue
        else:
            if decide_rotation(hold, ranked, retain_top_n=retain_top_n).is_noop:
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
            hold = None; q = 0
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
    _roll = _nav.cummax(); mdd = float(((_roll - _nav) / _roll).max()) * 100 if len(_nav) > 1 else 0.0
    calmar = round(cagr / mdd, 2) if mdd > 0 else 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0); losses = sum(1 for t in trades if t["pnl"] < 0)
    per_year = {}
    for yy, g in _nav.groupby(_nav.index.year):
        if len(g) > 1:
            rl = g.cummax()
            per_year[int(yy)] = {"ret_pct": round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
                                 "dd_pct": round(float(((rl - g) / rl).max()) * 100, 1)}
    open_pos = ({"symbol": hold, "qty": q, "entry_px": round(entry, 2)} if hold else None)
    class _R: pass
    res = _R(); res.per_year = per_year

    print(f"\nFinal NAV: Rs.{final:,.0f}")
    print(f"Total: {(final/capital-1)*100:+.2f}%  CAGR: {cagr:+.2f}%")
    print(f"Trades: {len(trades)} (W={wins} L={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"Max DD (rebal cap_after): {mdd:.2f}%  Calmar: {calmar:.2f}")
    print(f"Charges (Fyers CNC, all fills): Rs.{charges_total:,.0f}")

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
            "per_year": res.per_year,
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
