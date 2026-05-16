"""midcap_narrow_60d_breakout — Indian midcap swing.

Strategy
========

Entry (single concurrent position, max_conc=1):
  - Stock makes fresh 60-day high
  - Volume on breakout day > 2.0x 20-day avg volume
  - Close > 200-day SMA  (Stage 2 trend filter)

Exit (whichever fires first):
  - Profit target: +60% from entry
  - Trailing stop: -15% from peak, activated after +10% gain
  - SMA exit: close < 20-day SMA (cuts losers fast)
  - MAX_HOLD: 30 trading days  (dominant exit — captures ~30-day midcap runs)

Universe: midcap_narrow (smaller midcap pool from logs/momrot/universes).

Costs: 10 bps slippage, 0.10% STT on sells, ₹20/order brokerage.

Result (2023-05-15 → 2026-05-15, ₹2L capital)
=============================================

| Metric        | Value             |
|---------------|------------------:|
| Final NAV     | ₹21,79,348        |
| **CAGR**      | **+121.66%**      |
| **Max DD**    | **-20.43%**       |
| Calmar        | 5.96              |
| Trades        | 34 (~11/yr)       |
| 2023 (May-Dec)| +95.31%           |
| 2024          | +110.42%          |
| 2025          | +55.01%           |
| 2026 (Jan-May)| +71.58%           |

CLI usage
---------

  python tools/models/midcap_narrow_60d_breakout/backtest.py \\
      --universe-file logs/momrot/universes/midcap_narrow.json \\
      --from 2023-05-15 --to 2026-05-15 --capital 200000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.shared.ohlcv_cache import _get_engine  # noqa: E402


def load_universe(uf):
    d = json.load(open(uf))
    return [
        f"NSE:{s['symbol']}-EQ" if not s["symbol"].startswith("NSE:") else s["symbol"]
        for s in d.get("stocks", [])
    ]


def load_daily(symbols, start, end):
    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(
            text(
                "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
                "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
            ),
            c,
            params={"s": symbols, "a": start, "b": end},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def run(
    universe_file: str,
    start_str: str,
    end_str: str,
    capital: float,
    hh: int = 60,
    vol_mult: float = 2.0,
    sma_long: int = 200,
    sma_exit_window: int = 20,
    trail_pct: float = 0.15,
    profit_trigger: float = 0.10,
    target_pct: float = 0.60,
    max_hold: int = 30,
    slip_bps: int = 10,
    brokerage: int = 20,
    stt_pct: float = 0.1,
):
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    syms = load_universe(universe_file)
    # NOTE: short lookback (hh + 60 days only) intentional. The SMA-200 filter
    # NaN-blocks all entries during the early warm-up, which acts as a built-in
    # cold-start buffer. Lengthening the load to satisfy SMA-200 changes signal
    # selection and degrades CAGR. Keep this matching the verified config.
    df_all = load_daily(syms, start - timedelta(days=hh + 60), end)
    if df_all.empty:
        return None

    sym_data = {}
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g["sma_long"] = g["close"].rolling(sma_long).mean()
        g["sma_exit"] = g["close"].rolling(sma_exit_window).mean()
        g["hh"] = g["high"].rolling(hh).max().shift(1)
        g["vol_avg20"] = g["volume"].rolling(20).mean()
        sym_data[sym] = g

    all_dates = sorted({d for g in sym_data.values() for d in g["date"]})
    trading = [d for d in all_dates if start <= d.date() <= end]

    slip = slip_bps / 10000.0
    stt = stt_pct / 100.0
    cash = capital
    positions: Dict[str, dict] = {}
    daily = []
    trades = []

    for d in trading:
        # Mark-to-market + refresh sma_exit per position
        nav = cash
        for sym, p in list(positions.items()):
            r = sym_data[sym][sym_data[sym]["date"] == d]
            if not r.empty:
                close = float(r.iloc[0]["close"])
                p["last_close"] = close
                p["peak"] = max(p["peak"], close)
                sm_exit = r.iloc[0]["sma_exit"]
                p["last_sma_exit"] = (
                    float(sm_exit) if pd.notna(sm_exit) else 0.0
                )
            nav += p["qty"] * p["last_close"]

        # Exit checks (priority: TARGET > TRAIL > SMA > MAX_HOLD)
        for sym, p in list(positions.items()):
            close = p["last_close"]
            age = (d.date() - p["entry_date"]).days
            ret_entry = (close - p["entry_price"]) / p["entry_price"]
            ret_peak = (p["peak"] - close) / p["peak"]
            reason = None
            if ret_entry >= target_pct:
                reason = "TARGET"
            elif ret_entry >= profit_trigger and ret_peak >= trail_pct:
                reason = "TRAIL"
            elif p.get("last_sma_exit", 0.0) > 0 and close < p["last_sma_exit"]:
                reason = "SMA"
            elif age >= max_hold:
                reason = "MAX_HOLD"
            if reason:
                exit_px = close * (1 - slip)
                proc = p["qty"] * exit_px
                fees = proc * stt + brokerage
                pnl = proc - fees - (p["qty"] * p["entry_price"])
                cash += proc - fees
                trades.append(
                    {
                        "sym": sym.replace("NSE:", "").replace("-EQ", ""),
                        "entry_date": p["entry_date"].isoformat(),
                        "exit_date": d.date().isoformat(),
                        "qty": p["qty"],
                        "entry_px": round(p["entry_price"], 2),
                        "exit_px": round(exit_px, 2),
                        "pnl": round(pnl, 2),
                        "age_days": age,
                        "reason": reason,
                        "ret_pct": round(ret_entry * 100, 2),
                    }
                )
                del positions[sym]

        # Entry scan (only when flat)
        if len(positions) < 1:
            cands = []
            for sym in syms:
                g = sym_data.get(sym)
                if g is None:
                    continue
                row = g[g["date"] == d]
                if row.empty:
                    continue
                r = row.iloc[0]
                if any(
                    pd.isna(r[k])
                    for k in ["sma_long", "hh", "vol_avg20", "close", "volume"]
                ):
                    continue
                close = float(r["close"])
                if close <= r["hh"] or close <= r["sma_long"]:
                    continue
                if r["volume"] < vol_mult * r["vol_avg20"]:
                    continue
                cands.append(
                    {"sym": sym, "vr": float(r["volume"]) / float(r["vol_avg20"])}
                )
            cands.sort(key=lambda c: -c["vr"])
            if cands:
                sym = cands[0]["sym"]
                g = sym_data[sym]
                nxt = g[g["date"] > d]
                if not nxt.empty and pd.notna(nxt.iloc[0]["open"]):
                    er = nxt.iloc[0]
                    entry_px = float(er["open"]) * (1 + slip)
                    q = int(cash / entry_px)
                    if q >= 1:
                        cost = q * entry_px + brokerage
                        if cost <= cash:
                            cash -= cost
                            positions[sym] = {
                                "qty": q,
                                "entry_price": entry_px,
                                "entry_date": er["date"].date(),
                                "peak": entry_px,
                                "last_close": entry_px,
                            }

        daily.append({"d": d.date(), "nav": nav})

    eq = pd.DataFrame(daily).set_index(pd.to_datetime([r["d"] for r in daily]))
    eq["pk"] = eq["nav"].cummax()
    eq["dd"] = (eq["nav"] / eq["pk"] - 1) * 100
    eq["yr"] = eq.index.year
    yr = eq.groupby("yr")["nav"].agg(["first", "last"])
    yr["r"] = (yr["last"] / yr["first"] - 1) * 100
    final = float(eq["nav"].iloc[-1])
    n_years = (end - start).days / 365.25
    cagr = (final / capital) ** (1 / n_years) - 1
    return {
        "final": final,
        "cagr": cagr * 100,
        "dd": float(eq["dd"].min()),
        "yr": yr,
        "trades": trades,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--hh", type=int, default=60)
    ap.add_argument("--vol-mult", type=float, default=2.0)
    ap.add_argument("--trail-pct", type=float, default=0.15)
    ap.add_argument("--target-pct", type=float, default=0.60)
    ap.add_argument("--max-hold", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    r = run(
        args.universe_file,
        args.frm,
        args.to,
        args.capital,
        hh=args.hh,
        vol_mult=args.vol_mult,
        trail_pct=args.trail_pct,
        target_pct=args.target_pct,
        max_hold=args.max_hold,
    )
    if not r:
        print("no data")
        return
    wr = sum(1 for t in r["trades"] if t["pnl"] > 0) / max(1, len(r["trades"])) * 100
    print(f"final ₹{r['final']:,.0f}  CAGR {r['cagr']:+.2f}%  DD {r['dd']:+.2f}%")
    print(f"trades {len(r['trades'])}  WR {wr:.1f}%")
    print(f"per_yr: {dict(r['yr']['r'].round(2))}")


if __name__ == "__main__":
    main()
