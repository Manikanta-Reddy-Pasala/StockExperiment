"""Cash-Secured Put Wheel backtest on NSE F&O stocks — formula-driven,
strict point-in-time (no look-forward).

Universe: 19 large-cap F&O stocks that have daily options bars in our DB
(ASIANPAINT, HDFCBANK, RELIANCE, LT, TCS, ITC, WIPRO, HINDUNILVR, SBIN,
MARUTI, ICICIBANK, BAJFINANCE, KOTAKBANK, AXISBANK, INFY, SUNPHARMA,
ADANIENT, BHARTIARTL, ADANIPORTS).

Strategy:
  Each monthly cycle (one entry per expiry):
    1. On entry day, compute formula from HISTORY-ONLY for every stock:
         trend_z   = (close - SMA200) / SMA200       # bullish if >0
         iv_proxy  = realised_vol(60d) annualised    # premium-rich
         liq_z     = log10(avg option volume, 30d)   # tradeable
         score = 0.4*trend_z + 0.4*iv_proxy + 0.2*liq_z
    2. Filter: trend_z > 0 (bullish only), liq_z > log10(100)
    3. Rank by score, pick top N (default 5).
    4. Sell ~OTM_PCT %-OTM monthly put per stock.
    5. Exit: 50 % credit captured OR expiry settlement.

No look-forward enforcement:
  - SMA200, realised_vol, liq score only use bars with date < entry_day
  - Option entry price = close of entry_day (already past)
  - Exit decision walks day-by-day, only sees that day's close

Margin model (per stock short put, approximate SPAN):
  margin = strike * lot_size * SPAN_RATE  (default 18% — F&O stock SPAN
  is ~15-20% of strike notional, no offset for naked short put).

Usage (on VM):
    docker exec trading_system_app python3 -m \
        tools.models.stock_csp_wheel.backtest \
        --capital 200000 --top 5 --otm-pct 3 --profit-pct 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.shared.ohlcv_cache import _get_engine

# F&O stocks with daily options bars in the DB. Lot sizes are approximate
# NSE F&O lots as of 2025-2026 (may have changed slightly across the
# backtest window — close enough for relative ranking).
UNIVERSE: Dict[str, int] = {
    "ASIANPAINT": 200,
    "HDFCBANK": 550,
    "RELIANCE": 250,
    "LT": 300,
    "TCS": 175,
    "ITC": 1600,
    "WIPRO": 1800,
    "HINDUNILVR": 300,
    "SBIN": 1500,
    "MARUTI": 50,
    "ICICIBANK": 1375,
    "BAJFINANCE": 125,
    "KOTAKBANK": 400,
    "AXISBANK": 1250,
    "INFY": 400,
    "SUNPHARMA": 700,
    "ADANIENT": 500,
    "BHARTIARTL": 950,
    "ADANIPORTS": 1250,
}

SPAN_RATE = 0.18   # 18 % of strike notional for naked short stock put
MIN_LIQ_LOG10 = 2  # log10(100) — at least 100 contracts/day on the ATM strike


def _equity_daily(symbol: str) -> pd.DataFrame:
    """Daily equity bars for a stock. Returns date+close, sorted ascending."""
    sym = f"NSE:{symbol}-EQ"
    eng = _get_engine()
    q = text("SELECT date, close FROM historical_data "
             "WHERE symbol=:s ORDER BY date")
    with eng.connect() as c:
        df = pd.read_sql(q, c, params={"s": sym})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = df["close"].astype(float)
    return df


def _option_daily(underlying: str) -> pd.DataFrame:
    """All daily options for a stock. Returns date, expiry, strike, opt_type,
    close, volume. We pick strikes / expiries from this in-memory."""
    eng = _get_engine()
    q = text(
        "SELECT candle_time::date AS date, expiry, strike, opt_type, "
        "close, COALESCE(volume,0) AS volume, COALESCE(oi,0) AS oi "
        "FROM historical_options "
        "WHERE underlying=:u AND interval='D' "
        "ORDER BY candle_time")
    with eng.connect() as c:
        df = pd.read_sql(q, c, params={"u": underlying})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


def _monthly_expiries(opt_df: pd.DataFrame) -> List[date]:
    """Last expiry per month from options data — proxies monthly expiries."""
    if opt_df.empty:
        return []
    exps = sorted(set(opt_df["expiry"]))
    by_month: Dict[Tuple[int, int], date] = {}
    for e in exps:
        key = (e.year, e.month)
        if key not in by_month or e > by_month[key]:
            by_month[key] = e
    return sorted(by_month.values())


def _trend_z(eq: pd.DataFrame, entry_day: date) -> Optional[float]:
    hist = eq[eq["date"] < entry_day].tail(200)
    if len(hist) < 200:
        return None
    sma200 = hist["close"].mean()
    cur = eq[eq["date"] < entry_day]["close"].iloc[-1] if len(hist) else None
    if cur is None or sma200 == 0:
        return None
    return float((cur - sma200) / sma200)


def _realised_vol(eq: pd.DataFrame, entry_day: date, window: int = 60) -> Optional[float]:
    hist = eq[eq["date"] < entry_day].tail(window + 1)
    if len(hist) < window + 1:
        return None
    rets = np.log(hist["close"].values[1:] / hist["close"].values[:-1])
    if len(rets) < 2:
        return None
    return float(np.std(rets) * np.sqrt(252))


def _liq_score(opt_df: pd.DataFrame, spot: float, entry_day: date,
               window: int = 30) -> Optional[float]:
    cutoff = entry_day - timedelta(days=window)
    # ATM ± 1 strike step on PE side over the last `window` days
    near = opt_df[(opt_df["date"] >= cutoff) & (opt_df["date"] < entry_day)
                  & (opt_df["opt_type"] == "PE")]
    if near.empty:
        return None
    # find ATM strike on most recent day
    last = near["date"].max()
    last_rows = near[near["date"] == last]
    if last_rows.empty:
        return None
    atm = last_rows.iloc[(last_rows["strike"] - spot).abs().argsort()[:3]]
    strikes = set(atm["strike"])
    vol_band = near[near["strike"].isin(strikes)]
    avg_vol = vol_band["volume"].mean()
    if pd.isna(avg_vol) or avg_vol <= 0:
        return None
    return float(np.log10(avg_vol))


def _pick_target_strike(opt_df: pd.DataFrame, spot: float, expiry: date,
                        entry_day: date, otm_pct: float) -> Optional[Tuple[int, float]]:
    """Pick PE strike ≈ spot * (1 - otm_pct/100). Return (strike, entry_close).
    Only uses option close on entry_day (no look-forward)."""
    target = spot * (1 - otm_pct / 100)
    pe_today = opt_df[(opt_df["date"] == entry_day)
                      & (opt_df["expiry"] == expiry)
                      & (opt_df["opt_type"] == "PE")]
    if pe_today.empty:
        return None
    # closest strike below target (we want short put below spot)
    candidates = pe_today[pe_today["strike"] <= target]
    if candidates.empty:
        candidates = pe_today
    chosen = candidates.iloc[(candidates["strike"] - target).abs().argsort()[:1]].iloc[0]
    if chosen["close"] <= 0.5 or chosen["volume"] < 50:
        return None
    return int(chosen["strike"]), float(chosen["close"])


def _simulate_one(symbol: str, lot: int, eq: pd.DataFrame, opt_df: pd.DataFrame,
                  monthlies: List[date], entry_offset_days: int,
                  otm_pct: float, profit_pct: float,
                  min_leg_vol: int) -> List[Dict]:
    """Walk monthly cycles for ONE stock. Returns list of trade dicts."""
    trades: List[Dict] = []
    eq_dates = set(eq["date"])
    for i, exp in enumerate(monthlies):
        # Entry day = first trading day at/after (exp_prev + offset). Use
        # eq calendar so we land on a market day.
        prev_exp = monthlies[i - 1] if i > 0 else exp - timedelta(days=30)
        target_entry = prev_exp + timedelta(days=entry_offset_days)
        eq_after = sorted(d for d in eq_dates if d >= target_entry and d < exp)
        if not eq_after:
            continue
        entry_day = eq_after[0]
        spot_row = eq[eq["date"] == entry_day]
        if spot_row.empty:
            continue
        spot = float(spot_row.iloc[0]["close"])

        # Formula score (history-only)
        tz = _trend_z(eq, entry_day)
        rv = _realised_vol(eq, entry_day)
        liq = _liq_score(opt_df, spot, entry_day)
        if tz is None or rv is None or liq is None:
            continue
        if tz <= 0 or liq < MIN_LIQ_LOG10:
            continue  # bearish trend or too thin
        score = 0.4 * tz + 0.4 * rv + 0.2 * liq

        pick = _pick_target_strike(opt_df, spot, exp, entry_day, otm_pct)
        if pick is None:
            continue
        strike, entry_credit = pick

        # Walk day-by-day to exit
        post = opt_df[(opt_df["expiry"] == exp)
                      & (opt_df["opt_type"] == "PE")
                      & (opt_df["strike"] == strike)
                      & (opt_df["date"] >= entry_day)
                      & (opt_df["date"] <= exp)].sort_values("date")
        if post.empty:
            continue

        target_buyback = entry_credit * (1 - profit_pct / 100)
        exit_day = exp
        exit_price = max(0.0, strike - float(eq[eq["date"].between(exp, exp)]["close"].iloc[0]) if not eq[eq["date"] == exp].empty else 0.0)
        exit_reason = "EXPIRY"
        for r in post.itertuples():
            if r.date == entry_day:
                continue
            # Skip if too thin to exit
            if min_leg_vol > 0 and r.volume < min_leg_vol:
                continue
            if r.close <= target_buyback:
                exit_day = r.date
                exit_price = float(r.close)
                exit_reason = f"PROFIT_{int(profit_pct)}PCT"
                break
        else:
            # Held to expiry — settle at intrinsic
            spot_at_exp_rows = eq[eq["date"] == exp]
            spot_at_exp = float(spot_at_exp_rows.iloc[0]["close"]) if not spot_at_exp_rows.empty else spot
            exit_price = max(0.0, strike - spot_at_exp)
            exit_reason = "EXPIRY"

        pnl_unit = entry_credit - exit_price
        pnl_total = pnl_unit * lot
        margin = strike * lot * SPAN_RATE
        trades.append({
            "symbol": symbol,
            "entry_date": entry_day, "exit_date": exit_day, "expiry": exp,
            "spot": round(spot, 2), "strike": strike,
            "otm_pct_actual": round((spot - strike) / spot * 100, 2),
            "entry_credit": round(entry_credit, 2),
            "exit_price": round(exit_price, 2),
            "pnl_unit": round(pnl_unit, 2),
            "lot": lot,
            "pnl_total": round(pnl_total, 2),
            "margin_inr": round(margin, 2),
            "trend_z": round(tz, 4),
            "iv_proxy": round(rv, 4),
            "liq_z": round(liq, 4),
            "score": round(score, 4),
            "exit_reason": exit_reason,
        })
    return trades


def _select_top_by_score(picks_per_cycle: Dict[date, List[Dict]],
                         top_n: int) -> List[Dict]:
    """For each entry-day cycle, keep only the top-N stocks by score."""
    out: List[Dict] = []
    for d, picks in picks_per_cycle.items():
        ranked = sorted(picks, key=lambda x: x["score"], reverse=True)[:top_n]
        out.extend(ranked)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-05-15")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--top", type=int, default=5,
                    help="Top N stocks per cycle by score")
    ap.add_argument("--otm-pct", type=float, default=3.0,
                    help="Target OTM % for short put (default 3 %)")
    ap.add_argument("--profit-pct", type=float, default=50.0,
                    help="Exit at this % of credit captured (default 50 %)")
    ap.add_argument("--entry-offset", type=int, default=3,
                    help="Days after prev expiry to enter new cycle (default 3)")
    ap.add_argument("--min-leg-volume", type=int, default=50,
                    help="Min option day-volume to consider tradeable")
    args = ap.parse_args()

    print(f"=== CSP wheel — {len(UNIVERSE)} F&O stocks ===")
    print(f"  window  = {args.start} → {args.end}")
    print(f"  formula = 0.4*trend_z + 0.4*iv_proxy + 0.2*liq_z, top {args.top}")
    print(f"  otm     = {args.otm_pct}% OTM, exit at {args.profit_pct}% credit")
    print(f"  capital = ₹{int(args.capital):,}")

    all_trades: List[Dict] = []
    t0 = time.time()
    for i, (sym, lot) in enumerate(UNIVERSE.items()):
        eq = _equity_daily(sym)
        opt = _option_daily(sym)
        if eq.empty or opt.empty:
            print(f"  [skip] {sym} — no data")
            continue
        eq = eq[(eq["date"] >= pd.to_datetime(args.start).date())
                & (eq["date"] <= pd.to_datetime(args.end).date())]
        opt = opt[(opt["date"] >= pd.to_datetime(args.start).date())
                  & (opt["date"] <= pd.to_datetime(args.end).date())]
        monthlies = _monthly_expiries(opt)
        if not monthlies:
            continue
        trades = _simulate_one(sym, lot, eq, opt, monthlies,
                               args.entry_offset, args.otm_pct,
                               args.profit_pct, args.min_leg_volume)
        all_trades.extend(trades)
        print(f"  [{i+1}/{len(UNIVERSE)}] {sym}: {len(trades)} trades "
              f"({time.time()-t0:.0f}s)")

    if not all_trades:
        print("No trades — check data / formula thresholds.")
        return 1

    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Top-N selection per entry month (cycle)
    df["cycle"] = df["entry_date"].dt.to_period("M").astype(str)
    df["rank_in_cycle"] = df.groupby("cycle")["score"].rank(ascending=False, method="first")
    selected = df[df["rank_in_cycle"] <= args.top].copy()

    # Capital headroom check: cap total simultaneous margin to args.capital.
    # If sum(margin) for cycle > capital, drop lowest-score trades until fits.
    selected = selected.sort_values(["cycle", "score"], ascending=[True, False])
    keep_idx = []
    for cyc, grp in selected.groupby("cycle"):
        cum = 0.0
        for idx, row in grp.iterrows():
            if cum + row["margin_inr"] <= args.capital:
                keep_idx.append(idx)
                cum += row["margin_inr"]
    sel = selected.loc[keep_idx].sort_values("entry_date").reset_index(drop=True)

    # Equity curve
    sel["running_pnl"] = sel["pnl_total"].cumsum()
    sel["equity"] = args.capital + sel["running_pnl"]
    sel["peak"] = sel["equity"].cummax()
    sel["drawdown_pct"] = (sel["equity"] - sel["peak"]) / sel["peak"] * 100

    # Stats
    n = len(sel)
    wins = int((sel["pnl_total"] > 0).sum())
    total_pnl = float(sel["pnl_total"].sum())
    final_eq = float(sel["equity"].iloc[-1]) if n else args.capital
    n_yrs = max(1.0, (sel["exit_date"].iloc[-1] - sel["entry_date"].iloc[0]).days / 365.25) if n else 1
    cagr = ((final_eq / args.capital) ** (1 / n_yrs) - 1) * 100 if final_eq > 0 and n > 0 else 0.0
    max_dd = float(sel["drawdown_pct"].min()) if n else 0.0

    print()
    print("=== RESULTS ===")
    print(f"  trades        = {n}")
    print(f"  win rate      = {wins/n*100:.1f}%" if n else "  win rate = -")
    print(f"  total P&L     = ₹{total_pnl:,.0f}")
    print(f"  final equity  = ₹{final_eq:,.0f}")
    print(f"  total return  = {total_pnl/args.capital*100:+.1f}%")
    print(f"  CAGR          = {cagr:+.2f}%")
    print(f"  max drawdown  = {max_dd:.1f}%")
    print(f"  avg margin    = ₹{sel['margin_inr'].mean():,.0f}")
    print(f"  peak margin/cycle = ₹{sel.groupby('cycle')['margin_inr'].sum().max():,.0f}")

    # By-symbol breakdown
    by_sym = sel.groupby("symbol").agg(
        n=("pnl_total", "size"),
        wins=("pnl_total", lambda s: (s > 0).sum()),
        total_pnl=("pnl_total", "sum"),
        avg_credit=("entry_credit", "mean"),
    ).reset_index().sort_values("total_pnl", ascending=False)
    by_sym["wr"] = (by_sym["wins"] / by_sym["n"] * 100).round(1)
    print()
    print("=== Per-symbol (sorted by P&L) ===")
    print(by_sym.to_string(index=False))

    # Save trades + summary
    out_dir = REPO_ROOT / "exports" / "models" / f"stock_csp_wheel_cap{int(args.capital/1000)}k_otm{int(args.otm_pct*10)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_dir / "trades.csv", index=False)
    by_sym.to_csv(out_dir / "by_symbol.csv", index=False)
    summary = {
        "trades": n, "wins": wins, "wr_pct": wins/n*100 if n else 0,
        "total_pnl": total_pnl, "final_equity": final_eq,
        "cagr_pct": cagr, "max_dd_pct": max_dd,
        "capital": args.capital, "otm_pct": args.otm_pct,
        "profit_pct": args.profit_pct, "top_n": args.top,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
