"""Backtest: midcap_breakout_k3 — K-position breakout swing.

Shared core in strategy.py (params + PIT universe + breakout scan), reused by
live_signal.py. K is parametrised: --k 1 reproduces the single-position model
(midcap_narrow_60d_breakout) EXACTLY (validation gate); --k 3 is the shipped
variant that deploys idle capital across up to 3 concurrent breakouts.

Entry at NEXT day's open (signal on close, fill tomorrow); equal-weight target
(total_equity / K per slot). Exit via shared breakout_exit_reason. Costs: 10bps
slippage both legs, 0.10% STT on sells, Rs20/order.

Run: python3 tools/models/midcap_breakout_k3/backtest.py [--k 3] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.models.midcap_breakout_k3 import strategy as S

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END = date(2026, 5, 15)
DEFAULT_CAP = 1_000_000.0


def run(start, end, capital, K=S.K, out_dir=None):
    eng = _get_engine()
    syms = S.n500_union_symbols()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    fresh = S.fresh_symbols(df)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high")
    op = df.pivot(index="date", columns="symbol", values="open")
    vol = df.pivot(index="date", columns="symbol", values="volume")
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    dates = cl.index
    sma_long, hh, vol_avg20, adv20 = S.indicators(cl, hi, vol, adv_rs)
    year_starts, pools = S.build_year_pools(adv20, dates, fresh, start, end)

    trading = [d for d in dates if start <= d.date() <= end]
    cap = capital
    pos = {}                 # sym -> {qty, entry_px, entry_date, peak}
    trades = []
    slip, br, stt = S.SLIP, S.BR, S.STT

    def mtm(di):
        tot = 0.0
        for s, p in pos.items():
            c = cl[s].iloc[di] if s in cl.columns else None
            tot += p["qty"] * (float(c) if pd.notna(c) else p["entry_px"])
        return tot

    for d in trading:
        di = dates.get_loc(d)
        # ---- EXITS ----
        for s in list(pos.keys()):
            if s not in cl.columns:
                continue
            c = cl[s].iloc[di]
            if pd.isna(c):
                continue
            c = float(c); p = pos[s]; p["peak"] = max(p["peak"], c)
            age = (d.date() - p["entry_date"]).days
            reason = S.breakout_exit_reason(
                p["entry_px"], c, p["peak"], age,
                target_pct=S.TARGET_PCT, stop_pct=S.STOP_PCT, trail_pct=S.TRAIL_PCT,
                profit_trigger=S.PROFIT_TRIG, max_hold_days=S.MAX_HOLD)
            if reason:
                exit_px = c * (1 - slip); proc = p["qty"] * exit_px
                fees = proc * stt + br
                pnl = proc - fees - p["qty"] * p["entry_px"]
                cap += proc - fees
                del pos[s]
                trades.append({"sym": s.replace("NSE:", "").replace("-EQ", ""),
                               "entry_date": p["entry_date"].isoformat(),
                               "exit_date": d.date().isoformat(),
                               "ret_pct": round((exit_px / p["entry_px"] - 1) * 100, 2),
                               "pnl": round(pnl, 0), "reason": reason,
                               "cap_after": round(cap + mtm(di), 0)})
        # ---- ENTRIES (fill free slots) ----
        free = K - len(pos)
        if free > 0:
            band = S.band_for(year_starts, pools, d)
            cands = S.scan_breakouts(cl, sma_long, hh, vol, vol_avg20, band, di, set(pos.keys()))
            if cands and di + 1 < len(dates):
                eq = cap + mtm(di)
                target = eq / K                      # equal-weight slot size
                for cand in cands[:free]:
                    top = cand["sym"]
                    op_n = op[top].iloc[di + 1] if top in op.columns else None
                    if pd.isna(op_n):
                        continue
                    entry_px = float(op_n) * (1 + slip)
                    alloc = min(cap, target)
                    q = int(alloc / entry_px)
                    if q >= 1 and q * entry_px + br <= cap:
                        cap -= q * entry_px + br
                        pos[top] = {"qty": q, "entry_px": entry_px,
                                    "entry_date": dates[di + 1].date(), "peak": entry_px}

    # final NAV
    last_di = len(dates) - 1
    final = cap + mtm(last_di)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100
    peak = capital; mdd = 0.0
    for t in trades:
        peak = max(peak, t["cap_after"])
        mdd = max(mdd, (peak - t["cap_after"]) / peak * 100)
    res = {"model": "midcap_breakout_k3", "K": K,
           "start": start.isoformat(), "end": end.isoformat(),
           "final_nav": round(final, 0), "total_return_pct": round((final / capital - 1) * 100, 2),
           "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
           "calmar": round(cagr / max(0.01, mdd), 2),
           "trades": len(trades), "wins": wins, "losses": losses,
           "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
           "per_year": {}}
    return res, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=S.K)
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = datetime.strptime(a.start, "%Y-%m-%d").date()
    e = datetime.strptime(a.end, "%Y-%m-%d").date()
    res, trades = run(s, e, a.capital, K=a.k, out_dir=a.out)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(res, indent=2))
        (Path(a.out) / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
