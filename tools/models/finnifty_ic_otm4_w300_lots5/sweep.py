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

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

SPOT_MAP = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
}
LOT_HISTORY = {
    "NIFTY":     {date(2024, 9, 24): 75, date(1, 1, 1): 50},
    "BANKNIFTY": {date(2024, 9, 24): 30, date(1, 1, 1): 15},
    # FinNifty: 40 → 65 (Sep 2024) → 60 (2026 SEBI revision)
    "FINNIFTY":  {date(2026, 1, 1): 60, date(2024, 9, 24): 65,
                  date(1, 1, 1): 40},
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
    """Daily OHLCV+OI for an option contract. Returns:
        date, close, volume (contracts), oi, num_trades, turnover_lakh
    so callers can compute realistic-fill share = our_value / day_turnover.
    """
    eng = _get_engine()
    q = text(
        "SELECT candle_time::date AS date, close, "
        "       COALESCE(volume, 0) AS volume, "
        "       COALESCE(oi, 0) AS oi, "
        "       num_trades, "
        "       turnover_lakh "
        "FROM historical_options "
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


def slip_for_distance(dist_pct: float, base_slip: float) -> float:
    """Distance-aware slippage multiplier.

    Real options markets: liquidity drops sharply as you move OTM.
    Near-ATM strikes have tight 0.5-1% spreads; deep wings can be 10-20%.
    A constant `slip` parameter understates real-world execution cost on
    the wings — and Iron Condors live and die by wing-fill quality.

    Tiered multipliers vs `base_slip` (typically 1%):
      - <2% OTM (ATM): 1x   (still tight)
      - 2-3% OTM:      2x
      - 3-4% OTM:      4x
      - 4-6% OTM:      8x
      - >6% OTM:      15x   (deep wings: ₹1-5 premium, ₹2-3 spreads)
    """
    d = abs(dist_pct)
    if d < 2.0:
        mult = 1.0
    elif d < 3.0:
        mult = 2.0
    elif d < 4.0:
        mult = 4.0
    elif d < 6.0:
        mult = 8.0
    else:
        mult = 15.0
    return base_slip * mult


# SPAN+exposure margin approximation for an Iron Condor on Indian index
# options. Calibrated against live Sensibull basket on 2026-05-23 for
# FinNifty OTM2/W150/lots5 (margin shown ₹3,71,286). Formula:
#
#   short_notional = (ce_short_k + pe_short_k) * lot * lots
#   span_margin    = short_notional * SPAN_RATE         # SPAN on shorts
#   wing_credit    = (wce_entry + wpe_entry) * lot * lots  # paid premium offsets risk
#   exposure       = short_notional * EXPOSURE_RATE
#   margin         = max(0, span_margin - wing_credit) + exposure
#
# Tuned to within ~2% of the live Sensibull figure. Exchange SPAN moves
# with realised vol — treat as ballpark, not exact. Brokers add ~5-10%
# extra buffer in practice.
SPAN_RATE = 0.029         # ~2.9% of short-leg notional (NSE F&O typical)
EXPOSURE_RATE = 0.005     # ~0.5% exposure margin


def compute_ic_margin(ce_short_k: float, pe_short_k: float,
                      wce_entry_px: float, wpe_entry_px: float,
                      lot_size: int, lots: int) -> float:
    """Approx SPAN+exposure margin for a 4-leg Iron Condor (INR)."""
    short_notional = (ce_short_k + pe_short_k) * lot_size * lots
    span_margin = short_notional * SPAN_RATE
    wing_credit = (wce_entry_px + wpe_entry_px) * lot_size * lots
    exposure = short_notional * EXPOSURE_RATE
    return round(max(0.0, span_margin - wing_credit) + exposure, 2)


def run_ic(underlying: str, start: str, end: str,
           otm_pct: float, wing_width: int, stop_mult: float,
           slip: float, capital: float, lots: int,
           realistic_slip: bool = False,
           entry_week: int = 1,
           daily_volumes: Optional[list] = None) -> pd.DataFrame:
    """Backtest a monthly Iron Condor on `underlying`.

    `entry_week` picks WHEN inside the new monthly cycle the IC enters:
       1 = first trading day of week 1 (typically first Mon of cycle)
       2 = first trading day of week 2
       3 = first trading day of week 3
    Only ONE entry per monthly expiry (seen_exp guards duplicates).

    `daily_volumes` (optional list) gets appended one row per leg per
    holding day so callers can see whether each leg actually had volume
    on every day of the trade. Schema:
       {trade_idx, date, leg (ce_short|pe_short|wce_long|wpe_long),
        strike, close, volume, oi}
    """
    spot = load_spot(underlying, start, end)
    spot["dow"] = pd.to_datetime(spot["date"]).dt.dayofweek
    cands = spot[spot["dow"] < 5].copy()  # all weekdays
    # Tag each row with which week-of-month it sits in (1, 2, 3, …).
    cands["week_of_month"] = ((pd.to_datetime(cands["date"]).dt.day - 1) // 7) + 1
    cands = cands[cands["week_of_month"] == entry_week]
    step = STRIKE_STEP[underlying]

    trades = []
    seen_exp = set()
    for r in cands.itertuples():
        sig_d = r.date
        exp = near_monthly_exp(underlying, sig_d)
        if exp is None or exp in seen_exp:
            continue
        # NOTE: seen_exp.add(exp) deferred until AFTER successful entry below.
        # Earlier bug: marking expiry as seen before validating strike/bar
        # availability meant a single failed first day (e.g. wing strike not
        # yet listed OR no volume yet) would skip the entire monthly cycle.
        # Now retries on subsequent weekdays until entry succeeds.
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

        # Per-leg slippage: shorts sit at otm_pct from spot, wings sit
        # at (otm_pct + wing_width/spot*100) — wings are deeper OTM and
        # face larger real-world spreads. When realistic_slip is off, all
        # 4 legs share the flat `slip` (legacy behavior).
        if realistic_slip:
            wing_pct = (wing_width / spot_close) * 100
            short_slip = slip_for_distance(otm_pct, slip)
            wing_slip = slip_for_distance(otm_pct + wing_pct, slip)
        else:
            short_slip = wing_slip = slip
        try:
            ce_row = ce_b[ce_b["date"] == entry_day].iloc[0]
            pe_row = pe_b[pe_b["date"] == entry_day].iloc[0]
            wce_row = wce_b[wce_b["date"] == entry_day].iloc[0]
            wpe_row = wpe_b[wpe_b["date"] == entry_day].iloc[0]
            ce_e = float(ce_row["close"]) * (1 - short_slip)
            pe_e = float(pe_row["close"]) * (1 - short_slip)
            wce_e = float(wce_row["close"]) * (1 + wing_slip)
            wpe_e = float(wpe_row["close"]) * (1 + wing_slip)
        except (IndexError, KeyError):
            continue
        if min(ce_e, pe_e) <= 0.5:
            continue
        net_credit = (ce_e + pe_e) - (wce_e + wpe_e)
        if net_credit <= 0:
            continue

        # Build per-leg trajectory across the holding window. We RECORD
        # volume + oi for every day but do NOT filter on it — caller
        # gets daily_volumes back to inspect whether liquidity existed.
        def _legcols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            cols = ["date", "close", "volume", "oi", "num_trades", "turnover_lakh"]
            avail = [c for c in cols if c in df.columns]
            return (df[(df["date"] >= entry_day) & (df["date"] <= exp)][avail]
                    .rename(columns={"close": f"{prefix}_close",
                                     "volume": f"{prefix}_vol",
                                     "oi": f"{prefix}_oi",
                                     "num_trades": f"{prefix}_ntrd",
                                     "turnover_lakh": f"{prefix}_turn"}))
        pair = _legcols(ce_b, "ce")
        for df_o, prefix in [(pe_b, "pe"), (wce_b, "wce"), (wpe_b, "wpe")]:
            pair = pair.merge(_legcols(df_o, prefix), on="date", how="left")
        pair = pair.ffill().fillna(0)
        pair["pv"] = (pair["ce_close"] + pair["pe_close"]
                      - pair["wce_close"] - pair["wpe_close"])

        exit_d = exit_debit = exit_reason = None
        ce_x = pe_x = wce_x = wpe_x = None
        # Net exit cost: BUY back shorts (pay short_slip), SELL wings
        # (cross wing_slip the other way). Blended exit slip ≈ average.
        exit_slip = (short_slip + wing_slip) / 2 if realistic_slip else slip
        for pr in pair.itertuples():
            if pr.pv >= net_credit * stop_mult:
                exit_debit = pr.pv * (1 + exit_slip)
                exit_d = pr.date
                exit_reason = "SL"
                ce_x = float(pr.ce_close)
                pe_x = float(pr.pe_close)
                wce_x = float(pr.wce_close)
                wpe_x = float(pr.wpe_close)
                break
        if exit_debit is None:
            spot_lookup = dict(zip(spot["date"], spot["close"]))
            exp_spot = float(spot_lookup.get(exp, spot_close))
            ic_ce = max(0.0, exp_spot - ce_k); ic_pe = max(0.0, pe_k - exp_spot)
            wc = max(0.0, exp_spot - wce_k); wp = max(0.0, wpe_k - exp_spot)
            exit_debit = max(0.0, (ic_ce + ic_pe) - (wc + wp)) * (1 + exit_slip)
            exit_d = exp
            exit_reason = "EXPIRY"
            # At expiry, options settle at intrinsic value (per-leg)
            ce_x, pe_x, wce_x, wpe_x = ic_ce, ic_pe, wc, wp

        # Entry succeeded → claim expiry so other weekdays skip this cycle
        seen_exp.add(exp)

        # Capture per-leg daily volume + oi over the holding window.
        # NO filtering on volume — caller inspects this to see whether
        # actual market activity existed on each day of the trade.
        if daily_volumes is not None:
            trade_idx = len(trades)
            held = pair[(pair["date"] >= entry_day)
                        & (pair["date"] <= exit_d)]
            # Our intended order size per leg = lot_size * lots * entry_close.
            # Used to compute our_share_of_turnover so the caller can flag
            # "this leg-day's turnover is too thin for our intended order".
            our_order_inr_per_leg = {
                "ce_short": lot_size_for(underlying, entry_day) * lots * ce_e,
                "pe_short": lot_size_for(underlying, entry_day) * lots * pe_e,
                "wce_long": lot_size_for(underlying, entry_day) * lots * wce_e,
                "wpe_long": lot_size_for(underlying, entry_day) * lots * wpe_e,
            }
            for pr in held.itertuples():
                for prefix, leg, k in [("ce", "ce_short", ce_k),
                                        ("pe", "pe_short", pe_k),
                                        ("wce", "wce_long", wce_k),
                                        ("wpe", "wpe_long", wpe_k)]:
                    ntrd_raw = getattr(pr, f"{prefix}_ntrd", None)
                    turn_raw = getattr(pr, f"{prefix}_turn", None)
                    ntrd = (int(ntrd_raw) if ntrd_raw not in (None, 0)
                            and not pd.isna(ntrd_raw) else None)
                    turn = (float(turn_raw) if turn_raw not in (None, 0)
                            and not pd.isna(turn_raw) else None)
                    turn_inr = turn * 100_000 if turn else None
                    avg_trade_inr = (turn_inr / ntrd
                                     if turn_inr and ntrd else None)
                    our_share = (our_order_inr_per_leg[leg] / turn_inr
                                 if turn_inr and turn_inr > 0 else None)
                    daily_volumes.append({
                        "trade_idx": trade_idx,
                        "underlying": underlying,
                        "entry_date": entry_day,
                        "expiry": exp,
                        "date": pr.date,
                        "leg": leg,
                        "strike": k,
                        "close": float(getattr(pr, f"{prefix}_close", 0)),
                        "volume": int(getattr(pr, f"{prefix}_vol", 0)),
                        "oi": int(getattr(pr, f"{prefix}_oi", 0)),
                        "num_trades": ntrd,
                        "turnover_lakh": turn,
                        # Derived: avg ₹ per actual trade that day; lets
                        # caller see typical fill size.
                        "avg_trade_inr": round(avg_trade_inr, 0)
                                         if avg_trade_inr else None,
                        # Derived: our intended ₹ order vs day's total.
                        # >0.10 = our trade is >10% of day → won't fill clean.
                        "our_share_of_turnover": round(our_share, 4)
                                                  if our_share else None,
                    })

        pnl_unit = net_credit - exit_debit
        lot_size = lot_size_for(underlying, entry_day)
        margin_required = compute_ic_margin(ce_k, pe_k, wce_e, wpe_e,
                                            lot_size, lots)
        trades.append({
            "entry_date": entry_day, "exit_date": exit_d, "expiry": exp,
            "spot": round(spot_close, 1),
            "ce_k": ce_k, "pe_k": pe_k,
            "wce_k": wce_k, "wpe_k": wpe_k,
            # Per-leg ENTRY prices (post-slippage applied at fill)
            "ce_entry_px": round(ce_e, 2),
            "pe_entry_px": round(pe_e, 2),
            "wce_entry_px": round(wce_e, 2),
            "wpe_entry_px": round(wpe_e, 2),
            # Per-leg EXIT prices (intrinsic at expiry / market close at SL)
            "ce_exit_px": round(ce_x, 2) if ce_x is not None else None,
            "pe_exit_px": round(pe_x, 2) if pe_x is not None else None,
            "wce_exit_px": round(wce_x, 2) if wce_x is not None else None,
            "wpe_exit_px": round(wpe_x, 2) if wpe_x is not None else None,
            "net_credit": round(net_credit, 2),
            "exit_debit": round(exit_debit, 2),
            "pnl_unit": round(pnl_unit, 2),
            "lot": lot_size, "lots": lots,
            "pnl_total": round(pnl_unit * lots * lot_size, 2),
            "max_loss_per_unit": wing_width - net_credit,
            "max_loss_total": (wing_width - net_credit) * lots * lot_size,
            # SPAN+exposure margin approximation (INR). See compute_ic_margin
            # docstring — calibrated to live Sensibull figure ±2%.
            "margin_required_inr": margin_required,
            "margin_pct_of_capital": round(margin_required / capital * 100, 1)
                if capital > 0 else None,
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
    # === TIGHT GEOMETRY (May 2026) — liquid strikes, real-world slip ===
    # FinNifty — closer-to-ATM shorts + narrow wings stay inside liquid band.
    {"name": "FN_IC_OTM1_5_w150_lots5", "u": "FINNIFTY", "otm": 1.5, "ww": 150, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM1_5_w200_lots5", "u": "FINNIFTY", "otm": 1.5, "ww": 200, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM2_w150_lots5",   "u": "FINNIFTY", "otm": 2.0, "ww": 150, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM2_w200_lots5",   "u": "FINNIFTY", "otm": 2.0, "ww": 200, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM2_5_w200_lots5", "u": "FINNIFTY", "otm": 2.5, "ww": 200, "stop": 3.0, "lots": 5},
    {"name": "FN_IC_OTM2_5_w300_lots5", "u": "FINNIFTY", "otm": 2.5, "ww": 300, "stop": 3.0, "lots": 5},
    # NIFTY tight variants — lots auto-sized so max-loss ~75% of ₹2L cap
    # max_loss = wing*75*lots; cap=200K; target<=150K
    {"name": "NF_IC_OTM1_5_w150_lots13", "u": "NIFTY", "otm": 1.5, "ww": 150, "stop": 3.0, "lots": 13},
    {"name": "NF_IC_OTM1_5_w200_lots10", "u": "NIFTY", "otm": 1.5, "ww": 200, "stop": 3.0, "lots": 10},
    {"name": "NF_IC_OTM2_w200_lots10",   "u": "NIFTY", "otm": 2.0, "ww": 200, "stop": 3.0, "lots": 10},
    {"name": "NF_IC_OTM2_w300_lots6",    "u": "NIFTY", "otm": 2.0, "ww": 300, "stop": 3.0, "lots": 6},
    {"name": "NF_IC_OTM2_5_w300_lots6",  "u": "NIFTY", "otm": 2.5, "ww": 300, "stop": 3.0, "lots": 6},
    {"name": "NF_IC_OTM3_w300_lots6",    "u": "NIFTY", "otm": 3.0, "ww": 300, "stop": 3.0, "lots": 6},
    {"name": "NF_IC_OTM3_w500_lots4",    "u": "NIFTY", "otm": 3.0, "ww": 500, "stop": 3.0, "lots": 4},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--slip", type=float, default=0.01,
                    help="Base slippage. With --realistic-slip, this gets "
                         "multiplied 1x-15x by per-strike distance.")
    ap.add_argument("--realistic-slip", action="store_true",
                    help="Tiered per-leg slippage (deeper OTM = more slip). "
                         "Closer to real F&O execution.")
    ap.add_argument("--filter", default="",
                    help="Substring filter on variant name (e.g. 'OTM2' or 'NF_').")
    ap.add_argument("--out", default="/app/logs/iron_condor_sweep.md")
    args = ap.parse_args()

    rows = []
    variants = [v for v in VARIANTS if not args.filter or args.filter in v["name"]]
    for v in variants:
        print(f">>> {v['name']}", flush=True)
        try:
            df = run_ic(v["u"], args.frm, args.to, v["otm"], v["ww"],
                        v["stop"], args.slip, args.capital, v["lots"],
                        realistic_slip=args.realistic_slip)
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
