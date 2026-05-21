"""midcap_narrow + MULTI-POSITION overlay.

Standard midcap_narrow V2 entry logic, but allow up to MAX_CONC concurrent
breakouts (not just top-1 by vol_ratio). On any day with multiple
qualifying breakouts, take the top-N most active.

Each slot gets equal capital share.

Hypothesis: V2 baseline missed setups when already in position (8 trades
in 3y is low). Multi-pos captures more = more compounding events.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
from datetime import date, timedelta
from typing import Dict

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols


HH_WIN     = 40
VOL_MULT   = 2.0
SMA_LONG   = 200
TRAIL_PCT  = 0.20
PROFIT_TRIG = 0.10
TARGET_PCT = 1.00
MAX_HOLD   = 120
ADV_WIN    = 20
SKIP_TOP   = 0
KEEP_NEXT  = 100
N100_CSV   = "/app/src/data/symbols/nifty100.csv"


def load_n100():
    out = set()
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                out.add(r["Symbol"].strip())
    return out


def run(max_conc: int,
        start=date(2023, 5, 15), end=date(2026, 5, 12),
        capital=1_000_000.0) -> tuple:
    n100 = load_n100()
    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)

    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi = df.pivot(index="date", columns="symbol", values="high").ffill()
    op_p = df.pivot(index="date", columns="symbol", values="open").ffill()
    vol = df.pivot(index="date", columns="symbol", values="volume").fillna(0)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    dates = cl.index

    sma_long = cl.rolling(SMA_LONG).mean()
    hh = hi.rolling(HH_WIN).max().shift(1)
    vol_avg20 = vol.rolling(20).mean()
    adv20 = adv_rs.rolling(ADV_WIN).mean()

    last_di = len(dates) - 1
    last_adv = adv20.iloc[last_di].dropna().sort_values(ascending=False)
    pool = last_adv.iloc[SKIP_TOP:SKIP_TOP + KEEP_NEXT].index.tolist()
    midcap = [s for s in pool
              if s.replace("NSE:", "").replace("-EQ", "") not in n100]

    trading = [d for d in dates if start <= d.date() <= end]
    cap = capital
    positions: Dict[str, Dict] = {}  # sym -> dict
    trades = []
    slip, br, stt = 0.001, 20, 0.001
    nav_history = [capital]

    for d in trading:
        di = dates.get_loc(d)

        # Exit checks
        to_close = []
        for sym, pos in positions.items():
            if sym not in cl.columns:
                continue
            c_today = cl[sym].iloc[di]
            if pd.isna(c_today):
                continue
            close = float(c_today)
            pos["peak"] = max(pos["peak"], close)
            age = (d.date() - pos["entry_date"]).days
            ret_e = (close - pos["entry_px"]) / pos["entry_px"]
            ret_pk = (pos["peak"] - close) / pos["peak"] if pos["peak"] > 0 else 0
            reason = None
            if ret_e >= TARGET_PCT:
                reason = "TARGET"
            elif ret_e >= PROFIT_TRIG and ret_pk >= TRAIL_PCT:
                reason = "TRAIL"
            if reason is None and age >= MAX_HOLD:
                reason = "MAX_HOLD"
            if reason:
                to_close.append((sym, close, reason))

        for sym, close, reason in to_close:
            pos = positions[sym]
            exit_px = close * (1 - slip)
            proc = pos["qty"] * exit_px
            fees = proc * stt + br
            pnl = proc - fees - pos["qty"] * pos["entry_px"]
            cap += proc - fees
            ret_e = (exit_px - pos["entry_px"]) / pos["entry_px"]
            trades.append({
                "entry_date": pos["entry_date"].isoformat(),
                "exit_date":  d.date().isoformat(),
                "sym":        sym.replace("NSE:", "").replace("-EQ", ""),
                "qty":        pos["qty"],
                "entry_px":   round(pos["entry_px"], 2),
                "exit_px":    round(exit_px, 2),
                "pnl":        round(pnl, 0),
                "ret_pct":    round(ret_e * 100, 2),
                "reason":     reason,
            })
            del positions[sym]

        # Entry scan
        slots = max_conc - len(positions)
        if slots > 0:
            cands = []
            for sym in midcap:
                if sym in positions:
                    continue
                if sym not in cl.columns:
                    continue
                c_v = cl[sym].iloc[di]
                sma_v = sma_long[sym].iloc[di] if sym in sma_long.columns else None
                hh_v = hh[sym].iloc[di] if sym in hh.columns else None
                va_v = vol_avg20[sym].iloc[di] if sym in vol_avg20.columns else None
                v_v = vol[sym].iloc[di] if sym in vol.columns else None
                if any(pd.isna(x) for x in [c_v, sma_v, hh_v, va_v, v_v]):
                    continue
                c_v = float(c_v); sma_v = float(sma_v); hh_v = float(hh_v)
                va_v = float(va_v); v_v = float(v_v)
                if c_v <= hh_v or c_v <= sma_v:
                    continue
                if v_v < VOL_MULT * va_v:
                    continue
                cands.append({"sym": sym, "vr": v_v / va_v})
            cands.sort(key=lambda c: -c["vr"])
            picks = cands[:slots]
            if picks and di + 1 < len(dates):
                per_slot = cap / slots if slots > 0 else 0
                for pk in picks:
                    sym = pk["sym"]
                    op_n = op_p[sym].iloc[di + 1] if sym in op_p.columns else None
                    if pd.isna(op_n):
                        continue
                    entry_px = float(op_n) * (1 + slip)
                    q = int(per_slot / entry_px)
                    if q >= 1 and q * entry_px + br <= cap:
                        cap -= q * entry_px + br
                        positions[sym] = {
                            "qty": q, "entry_px": entry_px,
                            "entry_date": dates[di + 1].date(),
                            "peak": entry_px,
                        }

        # NAV for DD
        nav_now = cap
        for sym, pos in positions.items():
            if sym in cl.columns:
                c_p = cl[sym].iloc[di]
                if pd.notna(c_p):
                    nav_now += pos["qty"] * float(c_p)
        nav_history.append(nav_now)

    # Final force-close
    final = cap
    last_d = dates[-1]
    for sym, pos in list(positions.items()):
        last = float(cl[sym].iloc[-1])
        final += pos["qty"] * last

    wins = sum(1 for t in trades if t["pnl"] > 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100
    wr = (wins / len(trades) * 100) if trades else 0

    peak = capital
    mdd = 0
    for nav in nav_history:
        peak = max(peak, nav)
        if peak > 0:
            dd = (peak - nav) / peak * 100
            mdd = max(mdd, dd)
    return cagr, mdd, wr, len(trades)


if __name__ == "__main__":
    print("=" * 90)
    print("MIDCAP_NARROW MULTI-POSITION SWEEP")
    print("=" * 90)
    print(f"{'max_conc':>9} | {'CAGR':>8} {'DD':>6} {'WR':>5} {'#tr':>4} {'Calm':>5}")
    print("-" * 60)
    for mc in [1, 2, 3, 4, 5, 7, 10]:
        cagr, dd, wr, n = run(mc)
        calm = cagr / dd if dd > 0 else 0
        print(f"{mc:>9} | {cagr:>+7.2f}% {dd:>5.2f}% {wr:>4.1f}% {n:>4} {calm:>5.2f}")
