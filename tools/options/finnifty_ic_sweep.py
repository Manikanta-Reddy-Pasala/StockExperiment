"""FinNifty + BankNifty MONTHLY Iron Condor sweep with scaled lots.

Iron Condor max loss = (wing_width - net_credit) × lot × lots.
Capital required = max_loss + small buffer ≈ wing_width × lot × lots.
So defined risk allows 3-7x lot scaling at fixed capital.

Test: scale lots up to find where avg/mo hits 20%.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import _get_engine  # noqa: E402

SPOT_MAP = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
}
LOT_HISTORY = {
    "NIFTY":     {date(2024, 9, 24): 75, date(1, 1, 1): 50},
    "BANKNIFTY": {date(2024, 9, 24): 30, date(1, 1, 1): 15},
    "FINNIFTY":  {date(2024, 9, 24): 65, date(1, 1, 1): 40},
}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50}


def lot_size_for(u: str, d: date) -> int:
    for cut in sorted(LOT_HISTORY[u].keys(), reverse=True):
        if d >= cut:
            return LOT_HISTORY[u][cut]
    return list(LOT_HISTORY[u].values())[-1]


def load_spot(u: str, a: str, b: str) -> pd.DataFrame:
    eng = _get_engine()
    q = text(
        "SELECT date, close FROM historical_data "
        "WHERE symbol=:s AND date BETWEEN :a AND :b ORDER BY date"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"s": SPOT_MAP[u], "a": a, "b": b})
    if df.empty:
        raise RuntimeError(f"No spot for {u}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = pd.to_numeric(df["close"])
    return df


def opt_daily(symbol: str) -> pd.DataFrame:
    eng = _get_engine()
    q = text(
        "SELECT candle_time::date AS date, close FROM historical_options "
        "WHERE symbol=:s AND interval='D' ORDER BY candle_time"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"s": symbol})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def near_monthly_exp(u: str, after: date) -> Optional[date]:
    eng = _get_engine()
    q = text(
        "SELECT MIN(expiry) FROM option_universe "
        "WHERE underlying=:u AND expiry > :d AND expiry_kind='monthly'"
    )
    with eng.connect() as conn:
        row = conn.execute(q, {"u": u, "d": after}).fetchone()
    return row[0] if row and row[0] else None


def pick(u: str, exp: date, target: int, ot: str) -> Optional[str]:
    eng = _get_engine()
    q = text(
        "SELECT symbol FROM option_universe "
        "WHERE underlying=:u AND expiry=:e AND opt_type=:o "
        "ORDER BY ABS(strike - :t) LIMIT 1"
    )
    with eng.connect() as conn:
        row = conn.execute(q, {"u": u, "e": exp, "o": ot, "t": target}).fetchone()
    return row[0] if row else None


def round_strike(p: float, step: int) -> int:
    return int(round(p / step) * step)


def run_ic(underlying: str, start: str, end: str,
           otm_pct: float, wing_width: int, stop_mult: float,
           slip: float, capital: float, lots: int) -> pd.DataFrame:
    spot = load_spot(underlying, start, end)
    spot["dow"] = pd.to_datetime(spot["date"]).dt.dayofweek
    cands = spot[spot["dow"] == 0]  # Mondays
    step = STRIKE_STEP[underlying]

    trades = []
    seen_exp = set()
    for r in cands.itertuples():
        sig_d = r.date
        exp = near_monthly_exp(underlying, sig_d)
        if exp is None or exp in seen_exp:
            continue
        seen_exp.add(exp)
        spot_close = float(r.close)
        ce_k = round_strike(spot_close * (1 + otm_pct/100), step)
        pe_k = round_strike(spot_close * (1 - otm_pct/100), step)
        wce_k = ce_k + wing_width
        wpe_k = pe_k - wing_width

        ce_sym = pick(underlying, exp, ce_k, "CE")
        pe_sym = pick(underlying, exp, pe_k, "PE")
        wce_sym = pick(underlying, exp, wce_k, "CE")
        wpe_sym = pick(underlying, exp, wpe_k, "PE")
        if not all([ce_sym, pe_sym, wce_sym, wpe_sym]):
            continue
        ce_b = opt_daily(ce_sym); pe_b = opt_daily(pe_sym)
        wce_b = opt_daily(wce_sym); wpe_b = opt_daily(wpe_sym)
        if any(b.empty for b in [ce_b, pe_b, wce_b, wpe_b]):
            continue

        entry_day = sig_d
        if entry_day not in set(ce_b["date"]):
            fut = ce_b[ce_b["date"] > sig_d]
            if fut.empty:
                continue
            entry_day = fut.iloc[0]["date"]

        try:
            ce_e = float(ce_b[ce_b["date"] == entry_day].iloc[0]["close"]) * (1 - slip)
            pe_e = float(pe_b[pe_b["date"] == entry_day].iloc[0]["close"]) * (1 - slip)
            wce_e = float(wce_b[wce_b["date"] == entry_day].iloc[0]["close"]) * (1 + slip)
            wpe_e = float(wpe_b[wpe_b["date"] == entry_day].iloc[0]["close"]) * (1 + slip)
        except (IndexError, KeyError):
            continue
        if min(ce_e, pe_e) <= 0.5:
            continue
        net_credit = (ce_e + pe_e) - (wce_e + wpe_e)
        if net_credit <= 0:
            continue

        # Build pair trajectory
        pair = ce_b[(ce_b["date"] >= entry_day) & (ce_b["date"] <= exp)][
            ["date", "close"]].rename(columns={"close": "ce_close"})
        for df_o, col in [(pe_b, "pe_close"), (wce_b, "wce_close"), (wpe_b, "wpe_close")]:
            pair = pair.merge(
                df_o[(df_o["date"] >= entry_day) & (df_o["date"] <= exp)][["date", "close"]]
                    .rename(columns={"close": col}),
                on="date", how="left")
        pair = pair.ffill().fillna(0)
        pair["pv"] = (pair["ce_close"] + pair["pe_close"]
                      - pair["wce_close"] - pair["wpe_close"])

        exit_d = exit_debit = exit_reason = None
        for pr in pair.itertuples():
            if pr.pv >= net_credit * stop_mult:
                exit_debit = pr.pv * (1 + slip)
                exit_d = pr.date
                exit_reason = "SL"
                break
        if exit_debit is None:
            spot_lookup = dict(zip(spot["date"], spot["close"]))
            exp_spot = float(spot_lookup.get(exp, spot_close))
            ic_ce = max(0.0, exp_spot - ce_k); ic_pe = max(0.0, pe_k - exp_spot)
            wc = max(0.0, exp_spot - wce_k); wp = max(0.0, wpe_k - exp_spot)
            exit_debit = max(0.0, (ic_ce + ic_pe) - (wc + wp)) * (1 + slip)
            exit_d = exp
            exit_reason = "EXPIRY"

        pnl_unit = net_credit - exit_debit
        lot_size = lot_size_for(underlying, entry_day)
        trades.append({
            "entry_date": entry_day, "exit_date": exit_d, "expiry": exp,
            "spot": round(spot_close, 1),
            "ce_k": ce_k, "pe_k": pe_k,
            "wce_k": wce_k, "wpe_k": wpe_k,
            "net_credit": round(net_credit, 2),
            "exit_debit": round(exit_debit, 2),
            "pnl_unit": round(pnl_unit, 2),
            "lot": lot_size, "lots": lots,
            "pnl_total": round(pnl_unit * lots * lot_size, 2),
            "max_loss_per_unit": wing_width - net_credit,
            "max_loss_total": (wing_width - net_credit) * lots * lot_size,
            "exit_reason": exit_reason,
        })
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    return df


def summarize(df: pd.DataFrame, capital: float, label: str):
    if df.empty:
        return None
    m = df.groupby("month")["pnl_total"].sum() / capital * 100
    y = df.groupby("year")["pnl_total"].sum() / capital * 100
    wr = (df["pnl_total"] > 0).mean() * 100
    max_loss_seen = df["max_loss_total"].max()
    return {
        "label": label, "trades": len(df), "wr": wr,
        "total": float(df["pnl_total"].sum()),
        "avg_mo": float(m.mean()), "median_mo": float(m.median()),
        "best_mo": float(m.max()), "worst_mo": float(m.min()),
        "thirty_plus": int((m >= 30).sum()),
        "twenty_plus": int((m >= 20).sum()),
        "below_neg10": int((m < -10).sum()),
        "months": int(m.count()),
        "avg_yr": float(y.mean()),
        "max_loss_per_trade": float(max_loss_seen),
        "max_loss_pct_capital": max_loss_seen / capital * 100,
    }


VARIANTS = [
    # Refined high-OTM variants targeting 20%/mo sustained
    {"name": "FN_IC_OTM4_w300_lots5",  "u": "FINNIFTY", "otm": 4.0, "ww": 300, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM4_w300_lots7",  "u": "FINNIFTY", "otm": 4.0, "ww": 300, "stop": 3.0, "lots": 7},
    {"name": "FN_IC_OTM4_w400_lots5",  "u": "FINNIFTY", "otm": 4.0, "ww": 400, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM4_w500_lots3",  "u": "FINNIFTY", "otm": 4.0, "ww": 500, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM4_w500_lots5_stop2",  "u": "FINNIFTY", "otm": 4.0, "ww": 500, "stop": 2.0, "lots": 5},
    {"name": "FN_IC_OTM5_w500_lots5",  "u": "FINNIFTY", "otm": 5.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM5_w300_lots5",  "u": "FINNIFTY", "otm": 5.0, "ww": 300, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM6_w500_lots5",  "u": "FINNIFTY", "otm": 6.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM3_5_w500_lots5","u": "FINNIFTY", "otm": 3.5, "ww": 500, "stop": 3.0, "lots": 5},
    # FinNifty monthly IC with scaled lots
    {"name": "FN_IC_OTM3_w200_lots1",  "u": "FINNIFTY", "otm": 3.0, "ww": 200, "stop": 3.0, "lots": 1},
    {"name": "FN_IC_OTM3_w200_lots3",  "u": "FINNIFTY", "otm": 3.0, "ww": 200, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM3_w200_lots5",  "u": "FINNIFTY", "otm": 3.0, "ww": 200, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM3_w300_lots3",  "u": "FINNIFTY", "otm": 3.0, "ww": 300, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM3_w300_lots5",  "u": "FINNIFTY", "otm": 3.0, "ww": 300, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM3_w500_lots3",  "u": "FINNIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM3_w500_lots5",  "u": "FINNIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM3_w500_lots7",  "u": "FINNIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 7},
    {"name": "FN_IC_OTM2_w300_lots3",  "u": "FINNIFTY", "otm": 2.0, "ww": 300, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM2_w300_lots5",  "u": "FINNIFTY", "otm": 2.0, "ww": 300, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM2_w500_lots3",  "u": "FINNIFTY", "otm": 2.0, "ww": 500, "stop": 3.0, "lots": 3},
    {"name": "FN_IC_OTM2_w500_lots5",  "u": "FINNIFTY", "otm": 2.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM4_w500_lots5",  "u": "FINNIFTY", "otm": 4.0, "ww": 500, "stop": 3.0, "lots": 5},
    # BankNifty
    {"name": "BN_IC_OTM3_w300_lots3",  "u": "BANKNIFTY", "otm": 3.0, "ww": 300, "stop": 3.0, "lots": 3},
    {"name": "BN_IC_OTM3_w500_lots3",  "u": "BANKNIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 3},
    {"name": "BN_IC_OTM3_w500_lots5",  "u": "BANKNIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "BN_IC_OTM3_w1000_lots3", "u": "BANKNIFTY", "otm": 3.0, "ww": 1000, "stop": 3.0, "lots": 3},
    {"name": "BN_IC_OTM2_w500_lots5",  "u": "BANKNIFTY", "otm": 2.0, "ww": 500, "stop": 3.0, "lots": 5},
    # NIFTY for comparison
    {"name": "NF_IC_OTM3_w500_lots5",  "u": "NIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 5},
    {"name": "NF_IC_OTM2_w300_lots5",  "u": "NIFTY", "otm": 2.0, "ww": 300, "stop": 3.0, "lots": 5},
    {"name": "NF_IC_OTM2_w500_lots5",  "u": "NIFTY", "otm": 2.0, "ww": 500, "stop": 3.0, "lots": 5},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--out", default="/app/logs/iron_condor_sweep.md")
    args = ap.parse_args()

    rows = []
    for v in VARIANTS:
        print(f">>> {v['name']}", flush=True)
        try:
            df = run_ic(v["u"], args.frm, args.to, v["otm"], v["ww"],
                        v["stop"], args.slip, args.capital, v["lots"])
            s = summarize(df, args.capital, v["name"])
            if s:
                rows.append(s)
                print(f"  trades={s['trades']} avg/mo={s['avg_mo']:+.2f}% "
                      f"best={s['best_mo']:+.1f}% worst={s['worst_mo']:+.1f}% "
                      f"20+={s['twenty_plus']}/{s['months']} "
                      f"30+={s['thirty_plus']}/{s['months']} "
                      f"wr={s['wr']:.1f}% yr={s['avg_yr']:+.1f}% "
                      f"max_loss_cap={s['max_loss_pct_capital']:.1f}%", flush=True)
        except Exception as e:
            print(f"  ERR: {e}", flush=True)

    rows.sort(key=lambda r: -r["avg_mo"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Iron Condor Sweep — FinNifty + BankNifty + Nifty MONTHLY scaled lots\n\n")
        f.write(f"Capital ₹{args.capital:,.0f} | Window {args.frm}..{args.to}\n")
        f.write(f"Goal: ≥20%/mo sustained.\n\n")
        f.write("| Variant | Trades | WR | Avg/mo | Best | Worst | 20%+ | 30%+ | Avg/yr | Max single loss | Total |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['trades']} | {r['wr']:.1f}% | "
                    f"{r['avg_mo']:+.2f}% | {r['best_mo']:+.1f}% | "
                    f"{r['worst_mo']:+.1f}% | {r['twenty_plus']}/{r['months']} | "
                    f"{r['thirty_plus']}/{r['months']} | "
                    f"{r['avg_yr']:+.1f}% | "
                    f"{r['max_loss_pct_capital']:.1f}% | "
                    f"₹{r['total']:,.0f} |\n")
    print(f"\nReport: {args.out}", flush=True)


if __name__ == "__main__":
    main()
