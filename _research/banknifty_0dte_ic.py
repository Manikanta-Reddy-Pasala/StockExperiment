"""BankNifty MONTHLY expiry-day Iron Condor (0-DTE).

Premise: on day-of-expiry, theta crushes ATM premium to zero by 3:30pm.
Selling ATM IC at morning open captures intraday theta with defined risk
via wings.

NOTE: BankNifty WEEKLY expiries discontinued Nov 2024 per SEBI rule.
Only monthly contracts remain. So this is ~12 trades/year max.

Strategy:
  - Trigger: expiry date (last Wed of month, BN monthly)
  - Entry: open of expiry day — sell ATM CE+PE + buy wings ±width
  - Exit: hold to expiry close (intrinsic settlement)
  - Lots scaled, defined-risk via wings

RESEARCH — NOT COMMITTED.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

BN_SPOT = "NSE:NIFTYBANK-INDEX"
BN_STEP = 100


def round_strike(p, step=BN_STEP):
    return int(round(p / step) * step)


def load_spot_close_at(d: date) -> Optional[float]:
    eng = _get_engine()
    q = text("SELECT close FROM historical_data WHERE symbol=:s AND date=:d")
    with eng.connect() as conn:
        row = conn.execute(q, {"s": BN_SPOT, "d": d}).fetchone()
    return float(row[0]) if row else None


def load_spot_open_at(d: date) -> Optional[float]:
    eng = _get_engine()
    q = text("SELECT open FROM historical_data WHERE symbol=:s AND date=:d")
    with eng.connect() as conn:
        row = conn.execute(q, {"s": BN_SPOT, "d": d}).fetchone()
    return float(row[0]) if row else None


def list_monthly_expiries(start: date, end: date) -> list:
    eng = _get_engine()
    q = text(
        "SELECT DISTINCT expiry FROM option_universe "
        "WHERE underlying='BANKNIFTY' AND expiry_kind='monthly' "
        "AND expiry BETWEEN :a AND :b ORDER BY expiry"
    )
    with eng.connect() as conn:
        return [r[0] for r in conn.execute(q, {"a": start, "b": end}).fetchall()]


def pick(expiry: date, target: int, ot: str) -> Optional[str]:
    eng = _get_engine()
    q = text(
        "SELECT symbol FROM option_universe "
        "WHERE underlying='BANKNIFTY' AND expiry=:e AND opt_type=:ot "
        "ORDER BY ABS(strike - :t) LIMIT 1"
    )
    with eng.connect() as conn:
        row = conn.execute(q, {"e": expiry, "ot": ot, "t": target}).fetchone()
    return row[0] if row else None


def daily_close(sym: str, d: date) -> Optional[float]:
    eng = _get_engine()
    q = text(
        "SELECT close FROM historical_options WHERE symbol=:s "
        "AND candle_time::date=:d AND interval='D'"
    )
    with eng.connect() as conn:
        row = conn.execute(q, {"s": sym, "d": d}).fetchone()
    return float(row[0]) if row else None


def lot_size_for(d: date) -> int:
    return 30 if d >= date(2024, 9, 24) else 15


def run(start_str, end_str, otm_pct, wing_width, lots, capital, slip):
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    expiries = list_monthly_expiries(start, end)
    print(f"BankNifty monthly expiries: {len(expiries)}")

    trades = []
    for exp in expiries:
        spot_open = load_spot_open_at(exp)
        spot_close = load_spot_close_at(exp)
        if not spot_open or not spot_close:
            continue
        ce_k = round_strike(spot_open * (1 + otm_pct/100))
        pe_k = round_strike(spot_open * (1 - otm_pct/100))
        wce_k = ce_k + wing_width
        wpe_k = pe_k - wing_width
        ce = pick(exp, ce_k, "CE"); pe = pick(exp, pe_k, "PE")
        wce = pick(exp, wce_k, "CE"); wpe = pick(exp, wpe_k, "PE")
        if not all([ce, pe, wce, wpe]):
            continue
        # On expiry day, daily bar = the expiry-day OHLC.
        # Open price ≈ open of bar. Sadly we only have daily so we use
        # daily-close prices as a proxy for both entry and exit which
        # collapses the test. For a true 0-DTE test we'd need 5-min bars.
        # Approximate: entry credit = daily close × (1+IV uplift) is unreliable.
        # SKIP simulation if open ≈ close (no granularity).
        # Instead: use daily close as exit (intrinsic) and morning open of
        # spot to derive entry credit via Black-Scholes? Too complex.
        # Simplified: use a representative entry premium estimate based on
        # OTM% (synthetic).
        # ----
        # SHORTCUT: pull option DAILY close on expiry day = pre-expiry bar.
        # Pull option daily close on PRIOR trading day = entry price.
        eng = _get_engine()
        with eng.connect() as conn:
            prior = conn.execute(text(
                "SELECT MAX(candle_time::date) FROM historical_options "
                "WHERE symbol=:s AND candle_time::date < :d"
            ), {"s": ce, "d": exp}).fetchone()
        if not prior or prior[0] is None:
            continue
        prior_d = prior[0]
        ce_e = daily_close(ce, prior_d); pe_e = daily_close(pe, prior_d)
        wce_e = daily_close(wce, prior_d); wpe_e = daily_close(wpe, prior_d)
        if any(v is None for v in [ce_e, pe_e, wce_e, wpe_e]):
            continue
        net_credit = (ce_e + pe_e) - (wce_e + wpe_e)
        if net_credit <= 0:
            continue
        # Settlement: intrinsic on expiry day
        i_ce = max(0.0, spot_close - ce_k); i_pe = max(0.0, pe_k - spot_close)
        wc = max(0.0, spot_close - wce_k); wp = max(0.0, wpe_k - spot_close)
        net_settle = max(0.0, (i_ce + i_pe) - (wc + wp))
        pnl_unit = net_credit - net_settle * (1 + slip)
        lot = lot_size_for(exp)
        trades.append({
            "expiry": exp.isoformat(),
            "spot_open": round(spot_open, 1),
            "spot_close": round(spot_close, 1),
            "ce_k": ce_k, "pe_k": pe_k,
            "credit": round(net_credit, 2),
            "settle": round(net_settle, 2),
            "pnl_unit": round(pnl_unit, 2),
            "pnl_total": round(pnl_unit * lots * lot, 2),
            "lot": lot, "lots": lots,
        })

    if not trades:
        print("No trades")
        return
    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["expiry"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["expiry"]).dt.year
    m = df.groupby("month")["pnl_total"].sum() / capital * 100
    y = df.groupby("year")["pnl_total"].sum() / capital * 100
    wins = (df["pnl_total"] > 0).sum()
    print(f"\nTrades: {len(df)}  WR: {wins/len(df)*100:.1f}%")
    print(f"Total PnL: ₹{df['pnl_total'].sum():,.0f}")
    print(f"Avg/mo: {m.mean():+.2f}%  Avg/yr: {y.mean():+.2f}%")
    print(f"Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%")
    print(f"Per year: {dict(y.round(2))}")
    print("\nAll trades:")
    print(df[["expiry", "spot_open", "spot_close", "ce_k", "pe_k",
              "credit", "settle", "pnl_unit", "pnl_total"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--otm-pct", type=float, default=1.0)
    ap.add_argument("--wing-width", type=int, default=300)
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args()
    run(args.frm, args.to, args.otm_pct, args.wing_width, args.lots,
        args.capital, args.slip)


if __name__ == "__main__":
    main()
