"""Portfolio blender: simulate splitting capital across N strategies.

Each strategy gets its share of capital, runs independently, daily NAVs
summed for combined portfolio curve.

Capital splits:
  --split 0.33,0.33,0.34  → equal-weight 3 strategies
  --split 0.5,0.3,0.2     → tilted

Strategies tested:
  1. Equity breakout v2 (mc=5 lb=90 q10 trail10) on n100
  2. FinNifty monthly IC (otm3 w500 lots=N where N scales with allocation)
  3. BankNifty 0-DTE expiry IC (otm=1 w=1000 lots=N)

Outputs combined CAGR, max DD, monthly stats.

RESEARCH — NOT COMMITTED.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.models.breakout_60d_high_volume.backtest_v2 import run as run_breakout  # noqa: E402


def run_finnifty_ic(start, end, lots, capital):
    from tools.models.finnifty_ic_otm4_w300_lots5.sweep import run_ic
    r = run_ic("FINNIFTY", start, end, otm_pct=3.0, wing_width=500,
               stop_mult=3.0, slip=0.01, capital=capital, lots=lots)
    if r.empty:
        return None
    r = r.sort_values("entry_date")
    r["pnl_cum"] = r["pnl_total"].cumsum()
    # Build daily NAV: step function — each trade adds pnl at its entry date
    return r[["entry_date", "pnl_total"]].copy()


def run_bn_ic(start, end, lots, capital, otm_pct=1.0, wing_width=500):
    # Use the standalone _research script's run logic inline
    from sqlalchemy import text
    sys.path.insert(0, str(ROOT))
    from tools.shared.ohlcv_cache import _get_engine
    from datetime import date
    eng = _get_engine()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()

    q = text("SELECT DISTINCT expiry FROM option_universe "
             "WHERE underlying='BANKNIFTY' AND expiry_kind='monthly' "
             "AND expiry BETWEEN :a AND :b ORDER BY expiry")
    with eng.connect() as conn:
        expiries = [r[0] for r in conn.execute(q, {"a": start_d, "b": end_d}).fetchall()]

    def round_strike(p, step=100):
        return int(round(p / step) * step)

    def lot_size_for(d):
        return 30 if d >= date(2024, 9, 24) else 15

    trades = []
    for exp in expiries:
        with eng.connect() as conn:
            sp_open = conn.execute(text(
                "SELECT open FROM historical_data WHERE symbol=:s AND date=:d"
            ), {"s": "NSE:NIFTYBANK-INDEX", "d": exp}).fetchone()
            sp_close = conn.execute(text(
                "SELECT close FROM historical_data WHERE symbol=:s AND date=:d"
            ), {"s": "NSE:NIFTYBANK-INDEX", "d": exp}).fetchone()
        if not sp_open or not sp_close:
            continue
        spot_open = float(sp_open[0]); spot_close = float(sp_close[0])
        ce_k = round_strike(spot_open * (1 + otm_pct/100))
        pe_k = round_strike(spot_open * (1 - otm_pct/100))
        wce_k = ce_k + wing_width; wpe_k = pe_k - wing_width

        def pick(target, ot):
            with eng.connect() as conn:
                row = conn.execute(text(
                    "SELECT symbol FROM option_universe "
                    "WHERE underlying='BANKNIFTY' AND expiry=:e AND opt_type=:o "
                    "ORDER BY ABS(strike-:t) LIMIT 1"
                ), {"e": exp, "o": ot, "t": target}).fetchone()
            return row[0] if row else None

        ce = pick(ce_k, "CE"); pe = pick(pe_k, "PE")
        wce = pick(wce_k, "CE"); wpe = pick(wpe_k, "PE")
        if not all([ce, pe, wce, wpe]):
            continue
        with eng.connect() as conn:
            pr = conn.execute(text(
                "SELECT MAX(candle_time::date) FROM historical_options "
                "WHERE symbol=:s AND candle_time::date < :d"
            ), {"s": ce, "d": exp}).fetchone()
        if not pr or pr[0] is None:
            continue
        prior_d = pr[0]

        def dc(sym):
            with eng.connect() as conn:
                r = conn.execute(text(
                    "SELECT close FROM historical_options WHERE symbol=:s "
                    "AND candle_time::date=:d AND interval='D'"
                ), {"s": sym, "d": prior_d}).fetchone()
            return float(r[0]) if r else None

        ce_e = dc(ce); pe_e = dc(pe); wce_e = dc(wce); wpe_e = dc(wpe)
        if any(v is None for v in [ce_e, pe_e, wce_e, wpe_e]):
            continue
        net_credit = (ce_e + pe_e) - (wce_e + wpe_e)
        if net_credit <= 0:
            continue
        i_ce = max(0.0, spot_close - ce_k); i_pe = max(0.0, pe_k - spot_close)
        wc = max(0.0, spot_close - wce_k); wp = max(0.0, wpe_k - spot_close)
        net_settle = max(0.0, (i_ce + i_pe) - (wc + wp))
        pnl_unit = net_credit - net_settle * 1.01
        lot = lot_size_for(exp)
        trades.append({"entry_date": exp, "pnl_total": pnl_unit * lots * lot})
    if not trades:
        return None
    return pd.DataFrame(trades)


def build_nav_curve(trades_df: pd.DataFrame, start, end, start_capital):
    """trades_df has entry_date + pnl_total. Return daily NAV series."""
    dates = pd.date_range(start, end, freq="B")
    df = pd.DataFrame({"date": dates, "pnl_today": 0.0})
    if trades_df is not None and not trades_df.empty:
        td = trades_df.copy()
        td["entry_date"] = pd.to_datetime(td["entry_date"])
        td = td.groupby("entry_date")["pnl_total"].sum().reset_index()
        td.columns = ["date", "pnl_today"]
        df = df.merge(td, on="date", how="left", suffixes=("", "_y"))
        df["pnl_today"] = df["pnl_today_y"].fillna(0)
        df = df.drop(columns=["pnl_today_y"])
    df["cum_pnl"] = df["pnl_today"].cumsum()
    df["nav"] = start_capital + df["cum_pnl"]
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2023-05-15")
    ap.add_argument("--to", dest="to", default="2026-05-15")
    ap.add_argument("--total-capital", type=float, default=200_000)
    ap.add_argument("--split", default="0.33,0.33,0.34",
                    help="Capital fractions: breakout, finnifty, banknifty")
    args = ap.parse_args()

    splits = [float(x) for x in args.split.split(",")]
    assert abs(sum(splits) - 1.0) < 0.01, "splits must sum to 1.0"
    sb, sf, sn = splits
    cap_b = args.total_capital * sb
    cap_f = args.total_capital * sf
    cap_n = args.total_capital * sn
    print(f"Total ₹{args.total_capital:,.0f} | Breakout ₹{cap_b:,.0f} | "
          f"FinNifty ₹{cap_f:,.0f} | BankNifty ₹{cap_n:,.0f}")

    # 1. Breakout backtest
    universe_file = "/app/logs/momrot/universes/n100_current.json"
    print("\n[1/3] Running breakout v2...")
    br = run_breakout(
        universe_file, args.frm, args.to, max_conc=5, capital=cap_b,
        breakout_lookback=90, vol_mult=1.5, sma_exit=20,
        trail_pct=0.10, max_hold_days=120,
        slip_bps=10, brokerage=20, stt_pct=0.1,
        regime_on=True, atr_on=False, partial_on=False, quality_on=True,
        atr_mult=3.0, quality_min_90d=0.10,
        partial_trigger=0.20, partial_pct=0.50,
    )
    eq_br = br["equity"][["nav"]].copy()
    eq_br.index = pd.to_datetime(eq_br.index)

    # 2. FinNifty IC — lots = floor(allocation / 100k), min 1
    fn_lots = max(1, int(cap_f / 100_000) * 3)
    print(f"\n[2/3] Running FinNifty IC (lots={fn_lots})...")
    fn_trades = run_finnifty_ic(args.frm, args.to, fn_lots, cap_f)
    fn_nav = build_nav_curve(fn_trades, args.frm, args.to, cap_f)

    # 3. BankNifty 0-DTE IC
    bn_lots = max(1, int(cap_n / 30_000))
    print(f"\n[3/3] Running BankNifty 0-DTE IC (lots={bn_lots})...")
    bn_trades = run_bn_ic(args.frm, args.to, bn_lots, cap_n,
                           otm_pct=1.0, wing_width=500)
    bn_nav = build_nav_curve(bn_trades, args.frm, args.to, cap_n)

    # Merge daily NAVs
    eq_br_df = eq_br.copy()
    eq_br_df.index.name = "date"
    eq_br_df = eq_br_df.reset_index()
    eq_br_df["date"] = pd.to_datetime(eq_br_df["date"])

    all_dates = pd.date_range(args.frm, args.to, freq="B")
    combined = pd.DataFrame({"date": all_dates})
    combined = combined.merge(eq_br_df[["date", "nav"]].rename(columns={"nav": "br_nav"}),
                               on="date", how="left")
    combined = combined.merge(fn_nav[["date", "nav"]].rename(columns={"nav": "fn_nav"}),
                               on="date", how="left")
    combined = combined.merge(bn_nav[["date", "nav"]].rename(columns={"nav": "bn_nav"}),
                               on="date", how="left")
    for c in ["br_nav", "fn_nav", "bn_nav"]:
        combined[c] = combined[c].ffill()
    combined["br_nav"] = combined["br_nav"].fillna(cap_b)
    combined["fn_nav"] = combined["fn_nav"].fillna(cap_f)
    combined["bn_nav"] = combined["bn_nav"].fillna(cap_n)

    combined["total_nav"] = combined["br_nav"] + combined["fn_nav"] + combined["bn_nav"]
    combined["peak"] = combined["total_nav"].cummax()
    combined["dd_pct"] = (combined["total_nav"] / combined["peak"] - 1) * 100
    combined["month"] = combined["date"].dt.to_period("M").astype(str)
    combined["year"] = combined["date"].dt.year

    monthly = combined.groupby("month")["total_nav"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    yearly = combined.groupby("year")["total_nav"].agg(["first", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100

    final = combined["total_nav"].iloc[-1]
    print("\n=== COMBINED PORTFOLIO ===")
    print(f"Start ₹{args.total_capital:,.0f} → End ₹{final:,.0f} "
          f"({(final/args.total_capital-1)*100:+.1f}%)")
    print(f"Avg/yr: {yearly['ret_pct'].mean():+.2f}%")
    print(f"Avg/mo: {monthly['ret_pct'].mean():+.2f}%  "
          f"Best/mo: {monthly['ret_pct'].max():+.1f}%  "
          f"Worst/mo: {monthly['ret_pct'].min():+.1f}%")
    print(f"Max DD: {combined['dd_pct'].min():.2f}%")
    print(f"Per year: {dict(yearly['ret_pct'].round(2))}")

    print("\n--- per-strategy contribution ---")
    print(f"Breakout: ₹{eq_br_df['nav'].iloc[-1]:,.0f} "
          f"(start ₹{cap_b:,.0f})")
    print(f"FinNifty: ₹{combined['fn_nav'].iloc[-1]:,.0f} "
          f"(start ₹{cap_f:,.0f})")
    print(f"BankNifty: ₹{combined['bn_nav'].iloc[-1]:,.0f} "
          f"(start ₹{cap_n:,.0f})")


if __name__ == "__main__":
    main()
