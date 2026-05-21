"""midcap_narrow + PYRAMID overlay — compound winners harder.

Standard midcap_narrow V2 entry logic (40d HH + vol>2x + 200d SMA),
but tranches the capital deployment so winners get more allocation
as they prove themselves.

Tranching:
  Initial   = BASE_PCT × cash at entry trigger
  Pyramid 1 = PYR1_PCT × cash at +TRIG1 from initial entry
  Pyramid 2 = remaining cash at +TRIG2 from initial entry

Exit logic uses BLENDED (weighted-average) entry price for target / trail.

Caveat: if BASE_PCT < 100%, idle cash sits during the wait for pyramid
trigger. If position never reaches trigger and just exits at MAX_HOLD,
some capital was uninvested = drag on CAGR.

Goal: lift +141% V2 baseline toward +200%+ by compounding winners
without widening DD beyond ~15%.
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
from datetime import date, timedelta
from typing import List, Dict

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols


# Strategy params (V2 base)
HH_WIN     = 40
VOL_MULT   = 2.0
SMA_LONG   = 200
TRAIL_PCT  = 0.20
PROFIT_TRIG = 0.10
TARGET_PCT = 1.00
MAX_HOLD   = 120

# Universe
ADV_WIN    = 20
SKIP_TOP   = 0
KEEP_NEXT  = 100
N100_CSV   = "/app/src/data/symbols/nifty100.csv"

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_n100():
    out = set()
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                out.add(r["Symbol"].strip())
    return out


def run(base_pct: float, pyramid_levels: List[Dict],
        start=DEFAULT_START, end=DEFAULT_END, capital=DEFAULT_CAP) -> tuple:
    """
    pyramid_levels: list of {"trigger": float, "pct_of_remaining": float}
      e.g. [{"trigger": 0.10, "pct_of_remaining": 0.5},
            {"trigger": 0.25, "pct_of_remaining": 1.0}]
      Triggers measured from initial entry price (not blended avg).
    """
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
    pos = None  # {sym, qty, blended_entry_px, initial_entry_px, entry_date,
                #  peak, levels_hit}
    trades = []
    slip, br, stt = 0.001, 20, 0.001
    nav_history = [capital]

    for d in trading:
        di = dates.get_loc(d)

        if pos:
            if pos["sym"] not in cl.columns:
                continue
            c_today = cl[pos["sym"]].iloc[di]
            if pd.isna(c_today):
                continue
            close = float(c_today)
            pos["peak"] = max(pos["peak"], close)

            # Pyramid check (only if open levels remain)
            init_ret = (close - pos["initial_entry_px"]) / pos["initial_entry_px"]
            for i, lvl in enumerate(pyramid_levels):
                if i in pos["levels_hit"]:
                    continue
                if init_ret < lvl["trigger"]:
                    continue
                # Trigger met — deploy lvl.pct_of_remaining of CURRENT cash
                deploy_cash = cap * lvl["pct_of_remaining"]
                if deploy_cash <= 0:
                    continue
                px = close * (1 + slip)  # buy at close + slip
                add_qty = int(deploy_cash / px)
                if add_qty < 1 or add_qty * px + br > cap:
                    continue
                cap -= add_qty * px + br
                # Update blended entry
                tot_qty = pos["qty"] + add_qty
                blended = (pos["qty"] * pos["blended_entry_px"]
                           + add_qty * px) / tot_qty
                pos["qty"] = tot_qty
                pos["blended_entry_px"] = blended
                pos["levels_hit"].add(i)
                pos["pyramid_count"] = len(pos["levels_hit"])
                # Don't break — multiple levels could fire on same day if gap up

            # Exit using BLENDED entry for target/trail
            age = (d.date() - pos["entry_date"]).days
            ret_e = (close - pos["blended_entry_px"]) / pos["blended_entry_px"]
            ret_pk = (pos["peak"] - close) / pos["peak"] if pos["peak"] > 0 else 0
            reason = None
            if ret_e >= TARGET_PCT:
                reason = "TARGET"
            elif ret_e >= PROFIT_TRIG and ret_pk >= TRAIL_PCT:
                reason = "TRAIL"
            if reason is None and age >= MAX_HOLD:
                reason = "MAX_HOLD"
            if reason:
                exit_px = close * (1 - slip)
                proc = pos["qty"] * exit_px
                fees = proc * stt + br
                pnl = proc - fees - pos["qty"] * pos["blended_entry_px"]
                cap += proc - fees
                trades.append({
                    "entry_date": pos["entry_date"].isoformat(),
                    "exit_date":  d.date().isoformat(),
                    "sym":        pos["sym"].replace("NSE:", "").replace("-EQ", ""),
                    "qty":        pos["qty"],
                    "entry_px":   round(pos["blended_entry_px"], 2),
                    "init_px":    round(pos["initial_entry_px"], 2),
                    "exit_px":    round(exit_px, 2),
                    "pnl":        round(pnl, 0),
                    "ret_pct":    round(ret_e * 100, 2),
                    "reason":     reason,
                    "pyr_hits":   pos["pyramid_count"],
                    "cap_after":  round(cap, 0),
                })
                pos = None

        if pos is None:
            # Scan for new entry
            cands = []
            for sym in midcap:
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
            if cands:
                top = cands[0]["sym"]
                if di + 1 < len(dates):
                    op_n = op_p[top].iloc[di + 1] if top in op_p.columns else None
                    if pd.notna(op_n):
                        entry_px = float(op_n) * (1 + slip)
                        # Deploy only BASE_PCT of cash
                        deploy = cap * base_pct
                        q = int(deploy / entry_px)
                        if q >= 1 and q * entry_px + br <= cap:
                            cap -= q * entry_px + br
                            pos = {
                                "sym": top, "qty": q,
                                "blended_entry_px": entry_px,
                                "initial_entry_px": entry_px,
                                "entry_date": dates[di + 1].date(),
                                "peak": entry_px,
                                "levels_hit": set(),
                                "pyramid_count": 0,
                            }

        # NAV tracking for DD
        nav_now = cap
        if pos and pos["sym"] in cl.columns:
            c_p = cl[pos["sym"]].iloc[di]
            if pd.notna(c_p):
                nav_now += pos["qty"] * float(c_p)
        nav_history.append(nav_now)

    final = cap
    if pos:
        last = float(cl[pos["sym"]].iloc[-1])
        final = cap + pos["qty"] * last

    wins = sum(1 for t in trades if t["pnl"] > 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100
    wr = (wins / len(trades) * 100) if trades else 0

    peak = capital
    mdd = 0
    for nav in nav_history:
        peak = max(peak, nav)
        dd = (peak - nav) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)

    pyr_trades = sum(1 for t in trades if t.get("pyr_hits", 0) > 0)
    return cagr, mdd, wr, len(trades), pyr_trades


if __name__ == "__main__":
    print("=" * 100)
    print("MIDCAP_NARROW PYRAMID SWEEP — targeting >+200% CAGR (V2 base: +141%)")
    print("=" * 100)

    # (label, base_pct, pyramid_levels)
    combos = [
        ("V2 baseline (no pyramid)", 1.00, []),
        ("50% base + 50% @+10%",     0.50, [{"trigger": 0.10, "pct_of_remaining": 1.0}]),
        ("50% base + 50% @+15%",     0.50, [{"trigger": 0.15, "pct_of_remaining": 1.0}]),
        ("50% base + 50% @+20%",     0.50, [{"trigger": 0.20, "pct_of_remaining": 1.0}]),
        ("60% base + 40% @+10%",     0.60, [{"trigger": 0.10, "pct_of_remaining": 1.0}]),
        ("70% base + 30% @+10%",     0.70, [{"trigger": 0.10, "pct_of_remaining": 1.0}]),
        ("75% base + 25% @+10%",     0.75, [{"trigger": 0.10, "pct_of_remaining": 1.0}]),
        ("33% base + 33% @+10% + 34% @+25%", 0.33,
            [{"trigger": 0.10, "pct_of_remaining": 0.5},
             {"trigger": 0.25, "pct_of_remaining": 1.0}]),
        ("50% base + 25% @+15% + 25% @+30%", 0.50,
            [{"trigger": 0.15, "pct_of_remaining": 0.5},
             {"trigger": 0.30, "pct_of_remaining": 1.0}]),
        ("60% base + 20% @+10% + 20% @+25%", 0.60,
            [{"trigger": 0.10, "pct_of_remaining": 0.5},
             {"trigger": 0.25, "pct_of_remaining": 1.0}]),
        ("50% base + 50% @+5% (early)",    0.50, [{"trigger": 0.05, "pct_of_remaining": 1.0}]),
        ("40% base + 60% @+8%",            0.40, [{"trigger": 0.08, "pct_of_remaining": 1.0}]),
    ]

    print(f"{'variant':<46} | {'CAGR':>8} {'DD':>6} {'WR':>5} {'#tr':>4} "
          f"{'pyr':>4} {'Calm':>5}")
    print("-" * 95)
    results = []
    for label, base, levels in combos:
        try:
            cagr, dd, wr, n, pyr = run(base, levels)
            results.append((label, cagr, dd, wr, n, pyr))
        except Exception as e:
            print(f"  {label}: ERROR {e}")
    for label, cagr, dd, wr, n, pyr in sorted(results, key=lambda x: -x[1]):
        calm = cagr / dd if dd > 0 else 0
        print(f"{label:<46} | {cagr:>+7.2f}% {dd:>5.2f}% {wr:>4.1f}% {n:>4} "
              f"{pyr:>4} {calm:>5.2f}")
