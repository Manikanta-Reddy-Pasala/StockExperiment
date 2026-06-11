"""Standalone backtest: PRICE MEAN-REVERSION DIP-BUY (price_meanrev_n500).

Offline research/validation path. ALL strategy logic (params, levels, ranking)
lives in the shared core `strategy.py` — the SAME code live_signal.py uses, so
backtest and live cannot drift. This file only owns the offline walk: load
panels, replay the limit-fill day loop, score.

Mechanics (see strategy.py docstring): resting LIMIT BUY at SMA50-1*ATR fires
on a low-touch; exit at frozen SMA50 target / 1.5*ATR stop / 40d time; 10d
re-entry cooldown; top-3 slots ranked by 60d momentum; PIT N500 eligibility.

PIT/no-lookahead: every decision level comes from the PRIOR bar (di-1); the
trigger is judged against day di's OHLC; fills are conservative (limit buy:
min(open, level); stop assumed to fill before target on a both-hit day).

Validated vs tools/research/price_formula_lab.py (2026-06-11):
  2025-03-01 -> 2026-06-10: +102.8% CAGR / 12.2% DD / Calmar 8.46 / 225 trades.
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union, eligible_at
from tools.models.price_meanrev_n500 import strategy as S

DEFAULT_START = date(2025, 3, 1)
DEFAULT_END = date(2026, 6, 10)
DEFAULT_CAP = 1_000_000.0


def load_panels(eng, start, end):
    """OHLC panels over the historical N500 superset (PIT-filtered per day in run)."""
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=420), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    piv = lambda v: df.pivot(index="date", columns="symbol", values=v).sort_index()
    cl = piv("close").ffill()
    return (piv("open").reindex_like(cl), piv("high").reindex_like(cl),
            piv("low").reindex_like(cl), cl)


def run(start, end, capital, out_dir=None):
    eng = _get_engine()
    O, H, L, C = load_panels(eng, start, end)
    dates = C.index
    print(f"N500 panel: {len(C)} days x {len(C.columns)} symbols")
    atr14, sma50, mom60, entry_lvl = S.indicators(C, H, L)

    # float32 numpy views (the 1GiB container OOMs on full-window float64 frames)
    npf = lambda x: x.values.astype("float32")
    Cn, On, Hn, Ln = npf(C), npf(O), npf(H), npf(L)
    ATRn, SMAn, MOMn, LVLn = npf(atr14), npf(sma50), npf(mom60), npf(entry_lvl)
    cols = list(C.columns)
    plain = [s.replace("NSE:", "").replace("-EQ", "") for s in cols]

    i0 = max(dates.searchsorted(pd.Timestamp(start)), S.LOOKBACK + S.SMA_LEN)
    i1 = dates.searchsorted(pd.Timestamp(end), side="right")

    # per-day PIT N500 membership mask (snapshots monthly — cache by month)
    elig = np.zeros((len(dates), len(cols)), dtype=bool)
    cache = {}
    for i in range(i0, i1):
        dd = dates[i].date()
        key = (dd.year, dd.month)
        es = cache.get(key)
        if es is None:
            es = eligible_at("n500", dd); cache[key] = es
        elig[i] = [p in es for p in plain]

    cash = capital; pos = {}; trades = []; navs = []; nd = []; last_exit = {}
    for di in range(i0, i1):
        o, h, l, c = On[di], Hn[di], Ln[di], Cn[di]
        # ---- EXITS (stop assumed to fill before target on a both-hit day) ----
        for si in list(pos.keys()):
            p = pos[si]; px = None; why = None
            cp = c[si]
            if np.isnan(cp):
                continue
            if not np.isnan(p["stop"]) and l[si] <= p["stop"]:
                px = min(o[si], p["stop"]) if not np.isnan(o[si]) else p["stop"]; why = "STOP"
            elif not np.isnan(p["target"]) and h[si] >= p["target"]:
                px = max(o[si], p["target"]) if not np.isnan(o[si]) else p["target"]; why = "TARGET"
            elif (di - p["in_di"]) >= S.MAXHOLD:
                px = cp; why = "TIME"
            if px is not None:
                px = float(px)            # numpy float32 -> python float (JSON-safe)
                proc = p["qty"] * px * (1 - S.COST)
                cash += proc
                trades.append({
                    "sym": plain[si], "entry_date": p["in"],
                    "exit_date": dates[di].date().isoformat(),
                    "qty": p["qty"], "entry_px": round(p["entry"], 2),
                    "exit_px": round(float(px), 2),
                    "pnl": round(proc - p["qty"] * p["entry"], 0),
                    "ret_pct": round((float(px) / p["entry"] - 1) * 100, 2),
                    "exit_reason": why, "cap_after": round(cash, 0),
                })
                last_exit[si] = di; del pos[si]
        # ---- ENTRIES: resting limit buys at PRIOR-bar level, low-touch fires ----
        free = S.K - len(pos)
        if free > 0:
            lvl, atr_p, mom_p = LVLn[di - 1], ATRn[di - 1], MOMn[di - 1]
            cand = []
            for si in range(len(cols)):
                if si in pos or not elig[di][si]:
                    continue
                if si in last_exit and (di - last_exit[si]) < S.COOLDOWN:
                    continue
                Lv = lvl[si]
                if (np.isnan(Lv) or np.isnan(atr_p[si]) or atr_p[si] <= 0
                        or np.isnan(mom_p[si]) or np.isnan(l[si]) or l[si] > Lv):
                    continue
                fill = min(o[si], Lv) if not np.isnan(o[si]) else Lv
                if np.isnan(fill) or fill <= 0:
                    continue
                cand.append((float(mom_p[si]), si, float(fill), float(atr_p[si])))
            cand.sort(reverse=True)               # strongest 60d momentum first
            for _, si, fill, a in cand:
                if free <= 0:
                    break
                q = int((cash / max(1, free)) / fill)
                if q < 1:
                    continue
                cash -= q * fill * (1 + S.COST)
                pos[si] = {"qty": q, "entry": fill, "in_di": di,
                           "in": dates[di].date().isoformat(),
                           "stop": S.stop_price(fill, a),
                           "target": float(SMAn[di - 1][si])}
                free -= 1
        mv = cash + sum(p["qty"] * Cn[di][si] for si, p in pos.items()
                        if not np.isnan(Cn[di][si]))
        navs.append(mv); nd.append(dates[di])

    nav = pd.Series(navs, index=pd.DatetimeIndex(nd), dtype="float64")
    yrs = (end - start).days / 365.25
    final = float(nav.iloc[-1])
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
    last = Cn[i1 - 1]
    open_positions = [{
        "sym": plain[si], "qty": p["qty"], "entry_px": round(p["entry"], 2),
        "entry_date": p["in"], "last_px": round(float(last[si]), 2),
        "stop": round(p["stop"], 2), "target": round(p["target"], 2),
        "mtm_value": round(p["qty"] * float(last[si]), 0),
        "unrealized_pnl": round(p["qty"] * (float(last[si]) - p["entry"]), 0),
    } for si, p in pos.items() if not np.isnan(last[si])]

    print("\n## RESULTS (PRICE MEAN-REVERSION DIP-BUY, true-PIT, net of cost)")
    print(f"  Final NAV:    Rs.{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%   Max DD: {mdd:.2f}%   Calmar: {cagr/max(0.01,mdd):.2f}")
    print(f"  Trades: {len(trades)} (WR {100*wins/max(1,len(trades)):.0f}%)")
    for yy in sorted(years):
        print(f"    {yy}: {years[yy]['ret_pct']:+.1f}%  (intra-yr DD {years[yy]['dd_pct']:.1f}%)")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        (out_dir / "summary.json").write_text(json.dumps({
            "model": "price_meanrev_n500", "start": start.isoformat(),
            "end": end.isoformat(), "years": round(yrs, 3),
            "capital": capital, "final_nav": round(final, 0),
            "total_return_pct": round((final/capital-1)*100, 2),
            "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
            "calmar": round(cagr/max(0.01, mdd), 2),
            "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "open_positions": open_positions, "per_year": years,
            "params": {"K": S.K, "sma_len": S.SMA_LEN, "atr_len": S.ATR_LEN,
                       "entry_atr_k": S.ENTRY_ATR_K, "stop_atr": S.STOP_ATR,
                       "maxhold": S.MAXHOLD, "cooldown": S.COOLDOWN,
                       "lookback": S.LOOKBACK},
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
