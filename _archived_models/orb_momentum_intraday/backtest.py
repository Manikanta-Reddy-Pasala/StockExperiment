"""Backtest for orb_momentum_intraday — daily momentum select + morning ORB.

Uses the SHARED core strategy.py (rank_momentum / orb_trade / params) so the
backtest and live signal can never drift. Daily momentum comes from the DB
(historical_data, daily); intraday execution from cached 5-min bars (data.py).

Run:
  python -m tools.models.orb_momentum_intraday.backtest --from 2025-03-01 --to 2026-05-29 \
         --out exports/models/orb_momentum_intraday
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine                       # noqa: E402
from tools.shared.index_membership import universe_union, eligible_at  # noqa: E402
from tools.models.orb_momentum_intraday import strategy as S           # noqa: E402
from tools.models.orb_momentum_intraday.data import load_all_cached    # noqa: E402

DEFAULT_START = date(2025, 3, 1)
DEFAULT_END = date(2026, 5, 29)
DEFAULT_CAP = 1_000_000.0


def _daily_close(start: date, end: date) -> pd.DataFrame:
    eng = _get_engine()
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union(S.INDEX))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=60), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values="close").ffill()


def run(start: date, end: date, capital: float = DEFAULT_CAP, out_dir=None):
    dcl = _daily_close(start, end)
    ddates = dcl.index
    # MEMORY: only the daily-SELECTED leaders ever need 5-min bars (top-SELECT_TOP
    # per day), not the whole ~700-symbol universe. Pre-pass to collect that union,
    # then lazy-load only those pickles (was load_all_cached() -> ~1.1GB OOM).
    from pathlib import Path as _P
    from tools.models.orb_momentum_intraday.data import CACHE_DIR as _CACHE
    needed = set(); _seen0 = set()
    for di in range(len(ddates)):
        d = ddates[di]
        if d.date() < start or d.date() in _seen0 or di < 1:
            continue
        _seen0.add(d.date())
        # Rank on the PRIOR daily close (di-1), NOT today's — no lookahead.
        # Live ranks on the last available close (yesterday; today's daily bar is
        # pulled ~20:30), then trades today's intraday. The backtest must match.
        pri = ddates[di - 1].date()
        needed.update(S.rank_momentum(dcl, di - 1, set(eligible_at(S.INDEX, pri))))
    five = {}
    for s in needed:
        p = _P(_CACHE) / f"{s.replace('NSE:', '').replace('-EQ', '')}.pkl"
        if p.exists():
            try:
                five[s] = pd.read_pickle(p)
            except Exception:
                pass
    # pre-index 5-min by (symbol, day) for fast lookup (only the needed subset)
    by_day = {s: {dy: g for dy, g in df.groupby("day")} for s, df in five.items()}

    trades = []
    daily_rets = []      # single all-in trade return per trading day (full capital)
    seen = set()
    for di in range(len(ddates)):
        d = ddates[di]
        if d.date() < start or d.date() in seen or di < 1:
            continue
        seen.add(d.date())
        # Rank on the PRIOR daily close (di-1) — NO lookahead, matching live
        # (_today_leaders ranks on the last available close = yesterday, then
        # trades today's intraday bars). Trade day = di's 5-min bars.
        elig = set(eligible_at(S.INDEX, ddates[di - 1].date()))
        leaders = S.rank_momentum(dcl, di - 1, elig)
        # ALL-IN single position: pick the ONE leader to commit full capital to
        # (earliest breakout, rank tiebreak — strategy.pick_leader, the SAME pick
        # live makes). One trade/day, full-capital, no re-entry. This is the real
        # live sizing; the old per-slot/mean basket left ~45% of capital idle.
        leaders_bars = [by_day.get(sym, {}).get(d.date()) for sym in leaders]
        ci = S.pick_leader(leaders_bars)
        if ci is not None:
            t = S.orb_trade(leaders_bars[ci], leaders[ci])
            if t is not None:
                trades.append(t)
                daily_rets.append((d, t.ret_pct / 100.0))   # full capital into the one pick

    # equity curve from the single all-in daily trade (full-capital)
    idx = pd.DatetimeIndex([d for d, _ in daily_rets])
    sr = pd.Series([r for _, r in daily_rets], index=idx)
    eq = (1 + sr).cumprod()
    yrs = (end - start).days / 365.25
    final = float(capital * eq.iloc[-1]) if len(eq) else capital
    total_ret = (final / capital - 1) * 100
    cagr = ((eq.iloc[-1]) ** (252 / max(len(sr), 1)) - 1) * 100 if len(sr) else 0.0
    roll = eq.cummax()
    mdd = float(((roll - eq) / roll).max()) * 100 if len(eq) else 0.0
    sharpe = float(sr.mean() / sr.std() * np.sqrt(252)) if len(sr) > 1 and sr.std() else 0.0
    a = np.array([t.ret_pct for t in trades]) / 100.0
    wins = int((a > 0).sum())
    ex = {"stop": 0, "target": 0, "eod": 0}
    for t in trades:
        ex[t.reason] += 1
    per_year, per_month = {}, {}
    for yy, g in sr.groupby(sr.index.year):
        e = (1 + g).cumprod()
        dd = ((e.cummax() - e) / e.cummax()).max() * 100
        per_year[int(yy)] = {"ret_pct": round((e.iloc[-1] - 1) * 100, 1),
                             "dd_pct": round(float(dd), 1)}
    for mo, g in sr.groupby(sr.index.to_period("M")):
        per_month[str(mo)] = {"ret_pct": round((np.prod(1 + g.values) - 1) * 100, 1),
                              "days": int(len(g)),
                              "win_rate_pct": round(float((g.values > 0).mean()) * 100, 0)}

    print("\n## RESULTS (orb_momentum_intraday, morning ORB on momentum leaders)")
    print(f"  Window: {start} -> {end}  ({len(sr)} trading days)")
    print(f"  Total return: {total_ret:+.0f}%   CAGR: {cagr:+.0f}%   Max DD: {mdd:.1f}%   Sharpe~{sharpe:.2f}")
    print(f"  Trades: {len(trades)} ({len(trades)/max(len(sr),1):.1f}/day)  "
          f"per-trade avg {a.mean()*100:+.3f}%  WR {wins/max(len(a),1)*100:.0f}%")
    print(f"  Exits: target {ex['target']} / stop {ex['stop']} / EOD {ex['eod']}")
    for mo in sorted(per_month):
        m = per_month[mo]
        print(f"    {mo}: {m['ret_pct']:+6.1f}%  ({m['days']}d WR{m['win_rate_pct']:.0f})")

    result = {
        "model": "orb_momentum_intraday",
        "start": start.isoformat(), "end": end.isoformat(), "years": round(yrs, 3),
        "capital": capital, "final_nav": round(final, 0),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
        "calmar": round(cagr / max(0.01, mdd), 2), "sharpe": round(sharpe, 2),
        "trades": len(trades), "wins": wins, "losses": len(a) - wins,
        "win_rate_pct": round(wins / max(1, len(a)) * 100, 1),
        "exits": ex, "per_year": per_year, "per_month": per_month,
        "params": {"index": S.INDEX, "lookback": S.LOOKBACK, "select_top": S.SELECT_TOP,
                   "or_bars": S.OR_BARS, "entry_cutoff_min": S.ENTRY_CUTOFF_MIN,
                   "target_mult": S.TARGET_MULT, "slippage": S.SLIPPAGE,
                   "round_trip_cost": S.ROUND_TRIP_COST},
        "open_position": None,   # intraday model is always flat overnight
    }
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ledger = [{"sym": t.symbol, "cap": "—", "entry_date": t.day, "exit_date": t.day,
                   "entry_time": t.entry_time, "exit_time": t.exit_time,
                   "entry_px": t.entry_px, "exit_px": t.exit_px,
                   "ret_pct": t.ret_pct, "exit_reason": t.reason} for t in trades]
        (out_dir / "trade_ledger.json").write_text(json.dumps(ledger, indent=2))
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital, a.out)
