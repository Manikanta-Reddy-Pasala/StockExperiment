"""Standalone backtest: EMERGING MOMENTUM (emerging_momentum).

Offline research/validation path. The UNIVERSE-POOL + SELECTION + REGIME + RISK
logic lives in the shared core `strategy.py` (build_pools / pool_for_date /
rank_targets / regime_healthy / hit_trail / hit_bear_trail / hit_bear_stop /
params) — the SAME code live_signal.py uses, so backtest and live cannot drift.
This file only owns the offline walk: load panels, build the PIT yearly pools,
build the monthly calendar, replay buy/sell (NEXT-day-open entry), score.

Reproduces emerging_momentum_dd.py winning variant
(sma_entry=True, mom_floor=0.15, always_trail=0.25), PIT N500-minus-N100,
net 0.15%/side, next-day-open entry:
  FULL 2023-05..2026-05 : ~+77% CAGR / ~26% DD
    per-year 2023 ~+129, 2024 ~+19, 2025 ~+81, 2026 ~+11 (partial)
  WINDOW Mar-2025..May-2026 : ~+84% CAGR / ~16% DD

Run: python3 tools/models/emerging_momentum/backtest.py [--start 2023-05-15] [--end 2026-05-12]
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union
from tools.models.emerging_momentum import strategy as S

COST = 0.0015
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END = date(2026, 5, 12)
DEFAULT_CAP = 1_000_000.0


def load_panels(eng, start, end):
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    # Always load history back to (POOL_ANCHOR_START - 420d) even for a later
    # eval-window start: the per-year PIT pools are anchored to POOL_ANCHOR_START
    # and the 200-DMA/ADV warmup needs ~420 calendar days of lead-in. This keeps
    # any sub-window backtest using the exact same pools as the full run.
    load_from = min(start, S.POOL_ANCHOR_START) - timedelta(days=420)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": load_from, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").astype(float)
    op = df.pivot(index="date", columns="symbol", values="open").astype(float)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    # Restrict to EQUITY trading days — index-only date rows would poison the
    # rolling ADV/return windows (same guard as live_signal).
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    op = op.loc[equity_dates]
    adv_rs = adv_rs.loc[equity_dates]
    adv20, sma200, idx_sma200 = S.indicators(cl, adv_rs)
    return cl, op, adv20, sma200, idx_sma200


def month_starts(dates):
    s = pd.Series(dates, index=dates)
    return [pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).first().values]


def _px(panel, sym, d):
    try:
        v = panel.at[d, sym]
        return float(v) if pd.notna(v) else 0.0
    except KeyError:
        return 0.0


def run(start, end, capital, out_dir=None):
    eng = _get_engine()
    cl, op, adv20, sma200, idx_sma200 = load_panels(eng, start, end)
    dates = cl.index
    anchors, pools = S.build_pools(adv20, dates, end)
    rebal = set(d for d in month_starts(dates) if start <= d.date() <= end)
    cash = capital
    pos = {}                     # sym -> {qty, entry_px, entry_date, peak}
    nav, trades = [], []

    def sell(s, d, px, reason):
        nonlocal cash
        if px <= 0:
            return
        p = pos[s]
        cash += p["qty"] * px * (1 - COST)
        trades.append({"sym": s.replace("NSE:", "").replace("-EQ", ""),
                       "entry_date": p["entry_date"].date().isoformat(),
                       "exit_date": d.date().isoformat(),
                       "ret_pct": round((px / p["entry_px"] - 1) * 100, 2),
                       "reason": reason})
        del pos[s]

    for d in dates:
        if d.date() < start or d.date() > end:
            continue
        di = dates.get_loc(d)
        healthy = S.regime_healthy(cl, idx_sma200, di)
        # DAILY risk exits (priority: always-on trail, then bear stop/trail)
        for s in list(pos.keys()):
            px = _px(cl, s, d)
            if px <= 0:
                continue
            p = pos[s]; p["peak"] = max(p["peak"], px)
            if S.hit_trail(px, p["peak"]):
                sell(s, d, px, "TRAIL"); continue
            if S.hit_bear_trail(px, p["peak"], healthy) or S.hit_bear_stop(px, p["entry_px"], healthy):
                sell(s, d, px, "BEAR"); continue
        nav.append((d, cash + sum(p["qty"] * (_px(cl, s, d) or p["entry_px"])
                                  for s, p in pos.items())))
        if d not in rebal:
            continue
        pool = S.pool_for_date(anchors, pools, d)
        ranked = S.rank_targets(cl, sma200, pool, di)
        keep = set(ranked[:S.RETAIN])
        # monthly rank-drop exits (at today's close)
        for s in list(pos.keys()):
            if s not in keep:
                sell(s, d, _px(cl, s, d), "ROTATE")
        # buy top-K not held — filled at NEXT-day open
        want = [s for s in ranked if s not in pos][: max(0, S.K - len(pos))]
        if want and di + 1 < len(dates):
            nd = dates[di + 1]
            per = cash / len(want)
            for s in want:
                o = _px(op, s, nd)
                if o <= 0:
                    continue
                buy = o * (1 + COST)
                q = int(per / buy)
                if q >= 1 and q * buy <= cash:
                    cash -= q * buy
                    pos[s] = {"qty": q, "entry_px": buy, "entry_date": nd, "peak": o}

    navs = pd.Series({d: v for d, v in nav}).dropna()
    final = navs.iloc[-1]
    days = (navs.index[-1] - navs.index[0]).days
    cagr = (final / capital) ** (365.25 / max(1, days)) - 1
    dd = ((navs.cummax() - navs) / navs.cummax()).max()
    py = {int(y): round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
          for y, g in navs.groupby(navs.index.year)}
    wins = sum(1 for t in trades if t["ret_pct"] > 0)
    result = {
        "model": "emerging_momentum",
        "start": start.isoformat(), "end": end.isoformat(),
        "final_nav": round(final, 0), "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(dd * 100, 1), "calmar": round(cagr * 100 / max(0.01, dd * 100), 2),
        "trades": len(trades), "win_rate_pct": round(wins / max(1, len(trades)) * 100, 1),
        "per_year": py,
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "summary.json").write_text(json.dumps(result, indent=2))
        (Path(out_dir) / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
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
