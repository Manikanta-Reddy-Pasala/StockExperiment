"""Point-in-time momentum backtest for n100 AND n500 — survivorship-bias free.

Critical fix vs canon: at every rebalance date d, the universe is restricted
to symbols actually IN the index on date d (via tools.shared.index_membership.
eligible_at). Today's nifty100.csv is NEVER mixed in. Stocks that left the
index (YESBANK, IL&FS, DHFL...) ARE eligible during their tenure; stocks
that joined later (ADANIENT, ZOMATO, NUVAMA...) are NOT eligible before
their first snapshot appearance.

Pre-loads the UNION of every symbol that was ever in the index into a wide
close panel (so price history is available the moment a stock joins or
gets dropped), then filters eligible_at(d) per-rebal.

Variants run on each universe (3yr / 4yr / 10yr):
  CANON        — LB=15, retain=3, mid-month gate (matches live)
  B-abs > 10%  — entry skip when top-1 LB=15 return < 10pp
  TIME 15d/-8% — exit when held >= 15 bars AND PnL <= -8%
  TRAIL 12%    — trailing stop 12% off held peak close
  REG 50%      — n100 breadth (>200d SMA) >= 50% required to stay in market

Output: per-variant table with CAGR / DD / Calmar / per-year breakdown for
both n100 and n500. Final summary tables for easy comparison.
"""
import sys, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok
from tools.shared.index_membership import universe_union, eligible_at

LOOKBACK = 15


def _label(s):
    return s.replace("NSE:", "").replace("-EQ", "")


def _per_year(nav_series, capital):
    by_year = {}
    for d_, v in nav_series:
        by_year[d_.year] = v
    prev = capital; py = {}
    for yr in sorted(by_year.keys()):
        py[yr] = (by_year[yr] / prev - 1) * 100 if prev > 0 else 0
        prev = by_year[yr]
    return py


def _monthly(start, end, dates):
    out = set()
    y, m = start.year, start.month
    while True:
        fut = dates[dates >= pd.Timestamp(y, m, 1)]
        if len(fut) == 0 or fut[0].date() > end: break
        if fut[0].date() >= start: out.add(fut[0])
        m += 1
        if m > 12: m = 1; y += 1
    sd = pd.Timestamp(start)
    if sd in dates: out.add(sd)
    return out


def _midmonth(start, end, dates):
    out = set()
    y, m = start.year, start.month
    while True:
        target = pd.Timestamp(y, m, 15)
        if target.date() > end: break
        fut = dates[dates >= target]
        if len(fut) > 0 and fut[0].date() <= end and fut[0].date() >= start:
            out.add(fut[0])
        m += 1
        if m > 12: m = 1; y += 1
    return out


def run(*, cl, breadth, dates, index_name, start, end, capital,
        trail_pct=None, time_stop_days=None, time_stop_pnl=None,
        regime_breadth_min=None, abs_floor_pp=None,
        retain_top_n=3, mid_month_check=True, mid_month_lead_pct=5.0):
    """Walk daily, rotate monthly using POINT-IN-TIME eligible universe.

    At each rebal date we compute eligible_at(d) -> filter the loaded panel
    to that subset -> rank by LB=15 return -> apply gates.
    """
    rebal = _monthly(start, end, dates)
    mid_rebal = _midmonth(start, end, dates) if mid_month_check else set()
    mid_rebal = {d for d in mid_rebal if d not in rebal}

    def rank_at(di, on_date: date):
        if di < LOOKBACK:
            return [], {}
        # POINT-IN-TIME UNIVERSE filter
        elig = eligible_at(index_name, on_date)
        univ = [f"NSE:{s}-EQ" for s in elig
                if f"NSE:{s}-EQ" in cl.columns
                and pd.notna(cl[f"NSE:{s}-EQ"].iloc[di])]
        if not univ:
            return [], {}
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - LOOKBACK].reindex(univ) - 1
        ranked = list(rets.dropna().sort_values(ascending=False).index)
        ret_map = {s: float(rets[s]) * 100 for s in ranked}
        return ranked, ret_map

    cap = capital
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    peak_close = 0.0; bars_held = 0
    cooldown_until_next_rebal = False
    trades = []; nav_series = []
    exits = {"TRAIL": 0, "TIME": 0, "REGIME": 0, "FLOOR": 0,
             "DELIST": 0}   # DELIST = held symbol drops out of index

    walk = [d for d in dates if start <= d.date() <= end]
    for d in walk:
        di = dates.get_loc(d)
        in_rebal = d in rebal
        in_mid = d in mid_rebal

        # ---- INTRA-DAY EXIT CHECKS --------------------------------------
        if hold is not None and di < len(dates):
            px = cl[hold].iloc[di]
            if pd.notna(px):
                px = float(px)
                bars_held += 1
                if px > peak_close:
                    peak_close = px
                # Trailing stop
                if trail_pct is not None and peak_close > 0:
                    if px <= peak_close * (1 - trail_pct / 100.0):
                        cap += qty * px
                        trades.append({"sym": _label(hold), "reason": "TRAIL",
                                       "pnl": qty * (px - entry_px)})
                        exits["TRAIL"] += 1
                        hold = None; qty = 0
                        cooldown_until_next_rebal = True
                # Time stop
                if (hold is not None and time_stop_days is not None
                        and time_stop_pnl is not None
                        and bars_held >= time_stop_days):
                    pnl_pct = (px / entry_px - 1) * 100 if entry_px else 0
                    if pnl_pct <= time_stop_pnl:
                        cap += qty * px
                        trades.append({"sym": _label(hold), "reason": "TIME",
                                       "pnl": qty * (px - entry_px)})
                        exits["TIME"] += 1
                        hold = None; qty = 0
                        cooldown_until_next_rebal = True

        # Delist check: held stock no longer in index on this date.
        # Force-sell at close (mid-month or month-end, doesn't wait).
        if hold is not None:
            held_sym_raw = _label(hold)
            on_d = d.date()
            if held_sym_raw not in eligible_at(index_name, on_d):
                px = cl[hold].iloc[di]
                if pd.notna(px):
                    cap += qty * float(px)
                    trades.append({"sym": held_sym_raw, "reason": "DELIST",
                                   "pnl": qty * (float(px) - entry_px)})
                    exits["DELIST"] += 1
                    hold = None; qty = 0
                    # do NOT set cooldown — delist is forced, allow rebal entry

        # Regime gate (daily)
        regime_ok = True
        if regime_breadth_min is not None:
            br = float(breadth.iloc[di]) if di < len(breadth) else np.nan
            if not np.isnan(br) and br < regime_breadth_min:
                regime_ok = False
        if not regime_ok and hold is not None:
            px = cl[hold].iloc[di]
            if pd.notna(px):
                px = float(px)
                cap += qty * px
                trades.append({"sym": _label(hold), "reason": "REGIME",
                               "pnl": qty * (px - entry_px)})
                exits["REGIME"] += 1
                hold = None; qty = 0

        # ---- REBAL DECISION --------------------------------------------
        if (in_rebal or in_mid) and di >= LOOKBACK and regime_ok:
            if in_rebal:
                cooldown_until_next_rebal = False
            if not cooldown_until_next_rebal:
                ranked, ret_map = rank_at(di, d.date())
                if ranked:
                    top = ranked[0]
                    top_ret = ret_map[top]
                    # Absolute-floor gate (entry only)
                    if abs_floor_pp is not None and top_ret <= abs_floor_pp:
                        # Floor blocks entry; sell held if any (keep it out)
                        if hold is not None:
                            px = cl[hold].iloc[di]
                            if pd.notna(px):
                                cap += qty * float(px)
                                trades.append({"sym": _label(hold),
                                               "reason": "FLOOR",
                                               "pnl": qty * (float(px) - entry_px)})
                                exits["FLOOR"] += 1
                                hold = None; qty = 0
                    else:
                        run_dec = True
                        if in_mid and not in_rebal:
                            rets_list = [(s, ret_map[s]) for s in ranked]
                            if not midmonth_lead_ok(hold, rets_list, mid_month_lead_pct):
                                run_dec = False
                        if run_dec:
                            dec = decide_rotation(hold, ranked, retain_top_n)
                            if not dec.is_noop:
                                target = top
                                if hold is not None and hold != target:
                                    px = cl[hold].iloc[di]
                                    if pd.notna(px):
                                        cap += qty * float(px)
                                        trades.append({"sym": _label(hold),
                                                       "reason": "ROTATE",
                                                       "pnl": qty * (float(px) - entry_px)})
                                        hold = None; qty = 0
                                if hold is None:
                                    bx = cl[target].iloc[di]
                                    if pd.notna(bx):
                                        bx = float(bx)
                                        q = int(cap / bx)
                                        if q >= 1 and q * bx <= cap:
                                            cap -= q * bx
                                            hold = target; qty = q
                                            entry_px = bx
                                            entry_date = d.date().isoformat()
                                            peak_close = bx; bars_held = 0

        px_h = cl[hold].iloc[di] if hold is not None else None
        nav = cap + (qty * float(px_h) if hold is not None and pd.notna(px_h) else 0)
        nav_series.append((d.date(), nav))

    last_nav = nav_series[-1][1]
    yrs = (end - start).days / 365.25
    cagr = (((last_nav / capital) ** (1 / yrs)) - 1) * 100 if yrs > 0 else 0
    s_nav = pd.Series([v for _, v in nav_series])
    roll = s_nav.cummax()
    dd = float(((roll - s_nav) / roll).max()) * 100 if len(s_nav) > 1 else 0
    return {
        "cagr_pct": cagr, "max_dd_pct": dd,
        "calmar": cagr / max(0.01, dd), "trades": len(trades),
        "exits": exits, "per_year": _per_year(nav_series, capital),
        "final_nav": last_nav,
    }


def load_panel(index_name: str, end: date):
    """Load close panel for the FULL UNION of symbols ever in the index."""
    eng = _get_engine()
    union = sorted(universe_union(index_name))
    fyers_syms = [f"NSE:{s}-EQ" for s in union]
    earliest = date(2015, 1, 1)
    print(f"Loading {index_name} union: {len(fyers_syms)} symbols, "
          f"{earliest} -> {end}")
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b "
            "AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": fyers_syms, "a": earliest, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    dates = cl.index
    present_syms = [s for s in fyers_syms if s in cl.columns]
    print(f"  {len(dates)} days x {len(present_syms)} symbols loaded "
          f"(union={len(fyers_syms)})")
    sma200 = cl.rolling(200).mean()
    above = (cl > sma200).astype(float)
    breadth = above[present_syms].mean(axis=1)
    return cl, breadth, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes", default="n100,n500",
                    help="Comma-separated: n100,n500")
    ap.add_argument("--end", default="2026-05-12")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    a = ap.parse_args()
    end = date.fromisoformat(a.end)

    variants = [
        ("CANON (retain=3 LB=15 mid-mo)", {}),
        ("B-abs top1 > 10%", {"abs_floor_pp": 10}),
        ("TIME 15d/-8%", {"time_stop_days": 15, "time_stop_pnl": -8}),
        ("TRAIL 12%", {"trail_pct": 12}),
        ("REG 50% breadth", {"regime_breadth_min": 0.50}),
    ]
    windows = (
        ("3-YEAR (2023-05-15..2026-05-12)", date(2023, 5, 15)),
        ("4-YEAR (2022-05-15..2026-05-12)", date(2022, 5, 15)),
        ("10-YEAR (2016-05-15..2026-05-12)", date(2016, 5, 15)),
    )

    for index_name in a.indexes.split(","):
        index_name = index_name.strip()
        print(f"\n{'='*100}\n  {index_name.upper()} — POINT-IN-TIME BACKTEST\n{'='*100}")
        cl, breadth, dates = load_panel(index_name, end)

        for wlabel, wstart in windows:
            print(f"\n--- {wlabel} ---")
            hdr = ("%-36s %8s %7s %7s %6s %4s %4s %4s %4s %4s | per-year"
                   % ("variant", "CAGR%", "DD%", "Calmar", "trades",
                      "Trl", "Tim", "Reg", "Flr", "Dlst"))
            print(hdr)
            print("-" * (len(hdr) + 50))
            for name, kwargs in variants:
                r = run(cl=cl, breadth=breadth, dates=dates,
                        index_name=index_name, start=wstart, end=end,
                        capital=a.capital, **kwargs)
                e = r["exits"]
                py = r["per_year"]
                yrs_str = " ".join(f"{y}={py.get(y, 0):+5.1f}"
                                   for y in sorted(py.keys()))
                print("%-36s %+8.2f %7.2f %7.2f %6d %4d %4d %4d %4d %4d | %s" % (
                    name, r["cagr_pct"], r["max_dd_pct"], r["calmar"],
                    r["trades"], e["TRAIL"], e["TIME"], e["REGIME"],
                    e["FLOOR"], e["DELIST"], yrs_str))


if __name__ == "__main__":
    main()
