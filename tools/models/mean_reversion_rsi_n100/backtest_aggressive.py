"""Aggressive sweeps for mean-reversion to find >100% CAGR variant.

Knobs:
  - Universe: N100 vs N500 (wider = more deep-dip opportunities)
  - Pick count: top-1 vs top-3 (concurrent positions)
  - RSI band: tighter (<25 = deeper oversold = bigger bounces)
  - Target/Stop: wider risk/reward ratios
  - Hold: shorter (faster rotation = more compounding)

Note: max_concurrent>1 splits capital equally across picks.
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
from tools.shared.universes import nifty500_symbols


RSI_WIN     = 14
SMA_LONG    = 200
N100_CSV    = "/app/src/data/symbols/nifty100.csv"

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_n100():
    out = []
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                out.append(r["Symbol"].strip())
    return out


def rsi(closes: pd.Series, window: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _chg(side, qty, price):
    if qty < 1:
        return 0.0
    return float(compute_charges(side, qty, float(price), "CNC").get("total", 0))


def run(universe: str, max_concurrent: int, rsi_low: float, rsi_exit: float,
        target_pct: float, stop_pct: float, max_hold: int,
        start=DEFAULT_START, end=DEFAULT_END, capital=DEFAULT_CAP,
        max_price: float = None) -> tuple:
    if universe == "N100":
        syms = [f"NSE:{s}-EQ" for s in load_n100()]
    else:
        syms = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

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
    rsi_df = cl.apply(lambda c: rsi(c, RSI_WIN))

    dates = cl.index
    cap = capital
    holds = {}   # sym -> (qty, entry_px, entry_date)
    trades = []
    peak = capital
    max_dd = 0.0

    for d in dates:
        if d.date() < start or d.date() > end:
            continue
        di = dates.get_loc(d)
        if di < max(RSI_WIN, SMA_LONG):
            continue

        # NAV mark-to-market
        nav = cap
        for sym, (q, ep, ed) in holds.items():
            px = cl[sym].iloc[di]
            if pd.notna(px):
                nav += q * float(px)
        peak = max(peak, nav)
        dd = (peak - nav) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        # Exits
        to_close = []
        for sym, (q, ep, ed) in holds.items():
            px = cl[sym].iloc[di]
            r = rsi_df[sym].iloc[di]
            if pd.isna(px) or pd.isna(r):
                continue
            px = float(px)
            r = float(r)
            age = (dates[di] - ed).days
            ret = (px - ep) / ep
            reason = None
            if r > rsi_exit:
                reason = "RSI_EXIT"
            elif ret >= target_pct:
                reason = "TARGET"
            elif ret <= -stop_pct:
                reason = "STOP"
            elif age >= max_hold:
                reason = "MAX_HOLD"
            if reason:
                to_close.append((sym, px, ret, reason, age))
        for sym, px, ret, reason, age in to_close:
            q, ep, ed = holds[sym]
            proc = q * px - _chg("SELL", q, px)
            cap += proc
            trades.append({
                "entry_date": ed.date().isoformat(),
                "exit_date": d.date().isoformat(),
                "sym": sym.replace("NSE:", "").replace("-EQ", ""),
                "qty": q, "entry_px": round(ep, 2), "exit_px": round(px, 2),
                "pnl": round(proc - q * ep, 2),
                "ret_pct": round(ret * 100, 2),
                "age_days": age, "reason": reason,
            })
            del holds[sym]

        # Entries: fill empty slots with most-oversold qualifying names
        slots = max_concurrent - len(holds)
        if slots <= 0:
            continue
        r_today = rsi_df.iloc[di]
        sma_today = sma200.iloc[di]
        close_today = cl.iloc[di]
        qual = []
        for sym in syms:
            if sym in holds:
                continue
            rv = r_today.get(sym)
            sv = sma_today.get(sym)
            cv = close_today.get(sym)
            if pd.isna(rv) or pd.isna(sv) or pd.isna(cv):
                continue
            cv_f = float(cv)
            if max_price is not None and cv_f > max_price:
                continue
            if cv_f > float(sv) and float(rv) < rsi_low:
                qual.append((sym, float(rv), cv_f))
        if not qual:
            continue
        qual.sort(key=lambda x: x[1])
        per_slot = cap * 0.99 / slots if slots > 0 else 0
        for sym, _r, px in qual[:slots]:
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

    # Force close residuals at end
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
            "age_days": (last_d - ed).days, "reason": "FORCE_CLOSE_END",
        })

    final = cap
    yrs = (end - start).days / 365.25
    total_ret = (final / capital - 1) * 100
    cagr = ((final / capital) ** (1 / max(yrs, 0.001)) - 1) * 100
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return cagr, max_dd * 100, (wins / len(trades) * 100) if trades else 0, len(trades)


if __name__ == "__main__":
    print("=" * 90)
    print("AGGRESSIVE MEAN-REVERSION SWEEP")
    print("=" * 90)
    combos = [
        # universe, max_conc, rsi_low, rsi_exit, target, stop, max_hold, max_price
        ("N100", 1, 30, 50, 0.06, 0.04, 10, None),    # baseline
        ("N100", 3, 30, 50, 0.06, 0.04, 10, None),    # top-3
        ("N100", 5, 30, 50, 0.06, 0.04, 10, None),    # top-5
        ("N500", 1, 30, 50, 0.06, 0.04, 10, 3000),    # N500 single, price-cap
        ("N500", 3, 30, 50, 0.06, 0.04, 10, 3000),    # N500 top-3
        ("N500", 5, 30, 50, 0.06, 0.04, 10, 3000),    # N500 top-5
        ("N500", 5, 25, 55, 0.08, 0.05, 10, 3000),    # tighter
        ("N500", 5, 25, 60, 0.10, 0.05, 15, 3000),    # bigger wins
        ("N500", 10, 30, 50, 0.06, 0.04, 10, 3000),   # top-10 diversified
        ("N500", 5, 30, 50, 0.08, 0.05, 7, 3000),     # fast rotation
        ("N500", 3, 28, 55, 0.10, 0.06, 15, 3000),    # balanced
        ("N100", 5, 30, 55, 0.08, 0.05, 10, 3000),    # N100 top-5
    ]
    print(f"{'univ':>5} {'N':>2} {'rsi<':>4} {'rsi>':>4} {'tgt':>5} {'stp':>5} {'hld':>3} {'maxP':>5} | "
          f"{'CAGR':>7} {'DD':>6} {'WR':>5} {'#tr':>4} {'Calm':>5}")
    print("-" * 100)
    results = []
    for c in combos:
        univ, n_conc, rl, re, tg, st, hd, mp = c
        cagr, dd, wr, n = run(
            universe=univ, max_concurrent=n_conc,
            rsi_low=rl, rsi_exit=re, target_pct=tg, stop_pct=st,
            max_hold=hd, max_price=mp,
        )
        results.append((c, cagr, dd, wr, n))
    for c, cagr, dd, wr, n in sorted(results, key=lambda x: -x[1]):
        calm = cagr / dd if dd > 0 else 0
        print(f"{c[0]:>5} {c[1]:>2} {c[2]:>4} {c[3]:>4} {c[4]*100:>4.1f}% "
              f"{c[5]*100:>4.1f}% {c[6]:>3} {str(c[7] or '-'):>5} | "
              f"{cagr:>+7.2f}% {dd:>5.2f}% {wr:>4.1f}% {n:>4} {calm:>5.2f}")
