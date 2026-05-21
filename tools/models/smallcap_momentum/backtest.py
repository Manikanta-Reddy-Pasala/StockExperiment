"""smallcap_momentum_top1 — candidate 5th model targeting >100% CAGR.

Hypothesis: existing pseudo_n100 hit +149% CAGR on top-100 ADV by 30d
return. Smallcap250 universe is more volatile = bigger momentum bursts.
Apply the same rank-by-30d-return + 200d SMA uptrend gate to smallcaps.

Sweep:
  - Universe: Smallcap250 vs Midcap150 vs (Midcap150 ∪ Smallcap250)
  - Lookback: 30d / 60d / 90d
  - Rebal: monthly / weekly
  - Top-N: 1 / 3 / 5
  - Filters: max-price 1500 / 3000 / none, uptrend on/off
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.live.broker_charges import compute_charges


SMA_LONG  = 200
SMALLCAP_CSV = "/app/src/data/symbols/nifty_smallcap250.csv"
MIDCAP_CSV   = "/app/src/data/symbols/nifty_midcap150.csv"
N100_CSV     = "/app/src/data/symbols/nifty100.csv"

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_csv(p):
    out = []
    try:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out.append(r["Symbol"].strip())
    except FileNotFoundError:
        pass
    return out


def _chg(side, qty, price):
    if qty < 1:
        return 0.0
    return float(compute_charges(side, qty, float(price), "CNC").get("total", 0))


def build_universe(spec: str) -> list:
    spec = spec.upper()
    sm = load_csv(SMALLCAP_CSV)
    mc = load_csv(MIDCAP_CSV)
    lg = set(load_csv(N100_CSV))
    if spec == "SMALL":
        syms = sm
    elif spec == "MID":
        syms = mc
    elif spec == "MIDSMALL":
        syms = list(set(sm) | set(mc))
    elif spec == "ALLEXLG":  # all minus large caps
        syms = list((set(sm) | set(mc)) - lg)
    else:
        syms = sm
    return [f"NSE:{s}-EQ" for s in syms]


def run(universe: str, top_n: int, lookback: int, rebal: str,
        uptrend_gate: bool, max_price: float | None,
        start=DEFAULT_START, end=DEFAULT_END, capital=DEFAULT_CAP) -> tuple:
    syms = build_universe(universe)
    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b "
            "AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=400), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    sma200 = cl.rolling(SMA_LONG).mean()

    dates = cl.index

    # Build rebal dates
    rebals = set()
    if rebal == "monthly":
        y, m = start.year, start.month
        while True:
            tgt = pd.Timestamp(y, m, 1)
            fut = dates[dates >= tgt]
            if len(fut) == 0 or fut[0].date() > end:
                break
            if fut[0].date() >= start:
                rebals.add(fut[0])
            m += 1
            if m > 12:
                m = 1; y += 1
    elif rebal == "weekly":
        for d in dates:
            if d.date() < start or d.date() > end:
                continue
            if d.weekday() == 0:  # Monday
                rebals.add(d)
    sd = pd.Timestamp(start)
    if sd in dates:
        rebals.add(sd)
    rebal_dates = sorted(rebals)

    cap = capital
    holds = {}   # sym -> (qty, entry_px, entry_date)
    trades = []
    peak = capital
    max_dd = 0.0

    for d in dates:
        if d.date() < start or d.date() > end:
            continue
        di = dates.get_loc(d)
        if di < max(lookback, SMA_LONG):
            continue

        # NAV MTM
        nav = cap
        for sym, (q, ep, ed) in holds.items():
            px = cl[sym].iloc[di]
            if pd.notna(px):
                nav += q * float(px)
        peak = max(peak, nav)
        dd = (peak - nav) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if d not in rebals:
            continue

        # Compute current top-N by lookback return + filters
        close_now = cl.iloc[di]
        close_past = cl.iloc[di - lookback]
        sma_now = sma200.iloc[di]
        candidates = []
        for sym in syms:
            cn = close_now.get(sym)
            cp = close_past.get(sym)
            sv = sma_now.get(sym)
            if pd.isna(cn) or pd.isna(cp) or cp <= 0:
                continue
            cn_f = float(cn); cp_f = float(cp)
            if max_price is not None and cn_f > max_price:
                continue
            if uptrend_gate and (pd.isna(sv) or cn_f <= float(sv)):
                continue
            ret = cn_f / cp_f - 1
            candidates.append((sym, ret, cn_f))
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:top_n]
        top_set = {x[0] for x in top}

        # SELL anything not in top
        for sym in list(holds.keys()):
            if sym not in top_set:
                q, ep, ed = holds[sym]
                px = cl[sym].iloc[di]
                if pd.isna(px):
                    continue
                px = float(px)
                proc = q * px - _chg("SELL", q, px)
                cap += proc
                trades.append({
                    "entry_date": ed.date().isoformat(),
                    "exit_date": d.date().isoformat(),
                    "sym": sym.replace("NSE:", "").replace("-EQ", ""),
                    "qty": q, "entry_px": round(ep, 2), "exit_px": round(px, 2),
                    "pnl": round(proc - q * ep, 2),
                    "ret_pct": round((px - ep) / ep * 100, 2),
                    "age_days": (d - ed).days,
                    "reason": "ROTATE",
                })
                del holds[sym]

        # BUY new top picks
        slots = top_n - len(holds)
        new_picks = [t for t in top if t[0] not in holds][:slots]
        if not new_picks or slots <= 0:
            continue
        per_slot = cap * 0.99 / slots if slots > 0 else 0
        for sym, _r, px in new_picks:
            if per_slot <= 0:
                break
            q = int(per_slot / px)
            while q > 0:
                cost = q * px + _chg("BUY", q, px)
                if cost <= cap and cost <= per_slot * 1.05:
                    break
                q -= 1
            if q > 0:
                cap -= q * px + _chg("BUY", q, px)
                holds[sym] = (q, px, d)

    # Force close at end
    last_d = dates[-1]
    for sym, (q, ep, ed) in list(holds.items()):
        px = float(cl[sym].iloc[-1])
        proc = q * px - _chg("SELL", q, px)
        cap += proc
        trades.append({
            "entry_date": ed.date().isoformat(),
            "exit_date": last_d.date().isoformat(),
            "sym": sym.replace("NSE:", "").replace("-EQ", ""),
            "qty": q, "entry_px": round(ep, 2), "exit_px": round(px, 2),
            "pnl": round(proc - q * ep, 2),
            "ret_pct": round((px - ep) / ep * 100, 2),
            "age_days": (last_d - ed).days,
            "reason": "FORCE_CLOSE_END",
        })

    final = cap
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / max(yrs, 0.001)) - 1) * 100
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return cagr, max_dd * 100, (wins / len(trades) * 100) if trades else 0, len(trades)


if __name__ == "__main__":
    print("=" * 100)
    print("SMALL/MID-CAP MOMENTUM SWEEP — targeting >100% CAGR")
    print("=" * 100)
    combos = [
        # universe, top_n, lookback, rebal, uptrend, max_price
        ("SMALL",    1, 30, "monthly", True,  None),
        ("SMALL",    3, 30, "monthly", True,  None),
        ("SMALL",    5, 30, "monthly", True,  None),
        ("SMALL",    1, 30, "weekly",  True,  None),
        ("SMALL",    1, 60, "monthly", True,  None),
        ("SMALL",    1, 90, "monthly", True,  None),
        ("MID",      1, 30, "monthly", True,  None),
        ("MID",      1, 30, "weekly",  True,  None),
        ("MIDSMALL", 1, 30, "monthly", True,  None),
        ("MIDSMALL", 3, 30, "monthly", True,  None),
        ("MIDSMALL", 1, 30, "weekly",  True,  None),
        ("MIDSMALL", 1, 60, "monthly", True,  None),
        ("ALLEXLG",  1, 30, "monthly", True,  None),
        ("ALLEXLG",  1, 30, "weekly",  True,  None),
        ("ALLEXLG",  1, 60, "monthly", True,  None),
        ("SMALL",    1, 30, "monthly", True,  1500),  # cheap stocks only
        ("MIDSMALL", 1, 30, "monthly", True,  1500),
    ]
    print(f"{'univ':>9} {'N':>2} {'lb':>3} {'rebal':>7} {'up':>3} {'mxP':>5} | "
          f"{'CAGR':>8} {'DD':>7} {'WR':>6} {'#tr':>4} {'Calm':>5}")
    print("-" * 90)
    results = []
    for c in combos:
        try:
            cagr, dd, wr, n = run(*c)
            results.append((c, cagr, dd, wr, n))
        except Exception as e:
            print(f"  {c} ERROR {e}")
    for c, cagr, dd, wr, n in sorted(results, key=lambda x: -x[1]):
        calm = cagr / dd if dd > 0 else 0
        print(f"{c[0]:>9} {c[1]:>2} {c[2]:>3} {c[3]:>7} {str(c[4])[0]:>3} "
              f"{str(c[5] or '-'):>5} | "
              f"{cagr:>+7.2f}% {dd:>6.2f}% {wr:>5.1f}% {n:>4} {calm:>5.2f}")
