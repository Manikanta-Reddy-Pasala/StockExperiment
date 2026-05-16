"""RSI-2 Mean Reversion (Connors-style) — Indian equity.

Premise: in uptrending stocks, short-term pullbacks revert quickly.

Strategy:
  - Universe: any (default midcap_narrow)
  - Filter: close > 200d SMA (long-term uptrend confirmation)
  - Entry: RSI(2) < threshold (deep oversold), next-day open
  - Exit: close > N-day SMA OR max-hold OR -X% stop
  - Position sizing: equal-weight, max-conc positions

Different alpha source vs momentum:
  - Momentum: buy strength → trend continuation
  - Mean reversion: buy weakness → snapback

RESEARCH SCRIPT — NOT COMMITTED. Saves under _research/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402


def rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def load_universe(universe_file: str) -> List[str]:
    with open(universe_file) as f:
        data = json.load(f)
    plain = [s["symbol"] for s in data.get("stocks", [])]
    return [f"NSE:{s}-EQ" if not s.startswith("NSE:") else s for s in plain]


def load_daily(symbols, start, end) -> pd.DataFrame:
    eng = _get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM historical_data "
            "WHERE symbol = ANY(:syms) AND date BETWEEN :a AND :b "
            "ORDER BY symbol, date"
        ), conn, params={"syms": symbols, "a": start, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    return df


def run(universe_file, start_str, end_str, max_conc, capital,
        rsi_period, rsi_entry, sma_long, sma_exit, max_hold, stop_pct,
        slip_bps, brokerage, stt_pct):
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    syms = load_universe(universe_file)
    df_all = load_daily(syms, start - timedelta(days=sma_long + 60), end)
    if df_all.empty:
        return None

    sym_data = {}
    for s, g in df_all.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g["sma_long"] = g["close"].rolling(sma_long).mean()
        g["sma_exit"] = g["close"].rolling(sma_exit).mean()
        g["rsi"] = rsi(g["close"], rsi_period)
        sym_data[s] = g

    all_dates = sorted({d for df in sym_data.values() for d in df["date"]})
    trading = [d for d in all_dates if start <= d.date() <= end]

    slip = slip_bps / 10000.0
    stt = stt_pct / 100.0
    cash = capital
    positions: Dict[str, dict] = {}
    daily = []
    trades = []
    total_fees = 0.0

    for d in trading:
        # Mark to market
        nav = cash
        for s, p in list(positions.items()):
            row = sym_data[s][sym_data[s]["date"] == d]
            if row.empty:
                nav += p["qty"] * p["last_close"]
                continue
            close = float(row.iloc[0]["close"])
            p["last_close"] = close
            p["last_sma_exit"] = float(row.iloc[0]["sma_exit"]) \
                if pd.notna(row.iloc[0]["sma_exit"]) else 0.0
            nav += p["qty"] * close

        # Exit checks
        for s, p in list(positions.items()):
            close = p["last_close"]
            sma_exit_v = p.get("last_sma_exit", 0.0)
            age = (d.date() - p["entry_date"]).days
            reason = None
            if close >= sma_exit_v and sma_exit_v > 0:
                reason = "SMA_EXIT"
            elif close <= p["entry_price"] * (1 - stop_pct):
                reason = "STOP"
            elif age >= max_hold:
                reason = "MAX_HOLD"
            if reason:
                exit_px = close * (1 - slip)
                proceeds = p["qty"] * exit_px
                fees = proceeds * stt + brokerage
                total_fees += fees
                pnl = proceeds - fees - (p["qty"] * p["entry_price"])
                cash += proceeds - fees
                trades.append({
                    "entry": p["entry_date"].isoformat(),
                    "exit": d.date().isoformat(),
                    "symbol": s.replace("NSE:", "").replace("-EQ", ""),
                    "entry_px": round(p["entry_price"], 2),
                    "exit_px": round(exit_px, 2),
                    "qty": p["qty"], "pnl": round(pnl, 2),
                    "reason": reason, "days": age,
                })
                del positions[s]

        # Entry scan
        if len(positions) < max_conc:
            slots = max_conc - len(positions)
            cands = []
            for s in syms:
                if s in positions:
                    continue
                df = sym_data.get(s)
                if df is None:
                    continue
                row = df[df["date"] == d]
                if row.empty:
                    continue
                r = row.iloc[0]
                if pd.isna(r["sma_long"]) or pd.isna(r["rsi"]):
                    continue
                if r["close"] <= r["sma_long"]:
                    continue
                if r["rsi"] > rsi_entry:
                    continue
                cands.append({"sym": s, "rsi": float(r["rsi"]), "close": float(r["close"])})
            cands.sort(key=lambda c: c["rsi"])  # most oversold first
            for c in cands[:slots]:
                s = c["sym"]
                nxt = sym_data[s][sym_data[s]["date"] > d]
                if nxt.empty:
                    continue
                er = nxt.iloc[0]
                if pd.isna(er["open"]):
                    continue
                entry_px = float(er["open"]) * (1 + slip)
                allocate = cash / max(1, slots)
                qty = int(allocate / entry_px)
                if qty < 1:
                    continue
                cost = qty * entry_px + brokerage
                total_fees += brokerage
                cash -= cost
                positions[s] = {
                    "qty": qty, "entry_price": entry_px,
                    "entry_date": er["date"].date(),
                    "last_close": entry_px, "last_sma_exit": 0.0,
                }
                slots -= 1
                if slots <= 0:
                    break

        daily.append({"date": d.date(), "nav": nav})

    eq = pd.DataFrame(daily).set_index(pd.to_datetime([r["date"] for r in daily]))
    eq["month"] = eq.index.to_period("M").astype(str)
    eq["year"] = eq.index.year
    eq["peak"] = eq["nav"].cummax()
    eq["dd_pct"] = (eq["nav"] / eq["peak"] - 1) * 100
    monthly = eq.groupby("month")["nav"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    yearly = eq.groupby("year")["nav"].agg(["first", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100
    return {"final": float(eq["nav"].iloc[-1]), "max_dd": float(eq["dd_pct"].min()),
            "monthly": monthly, "yearly": yearly, "trades": trades, "fees": total_fees}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--max-conc", type=int, default=3)
    ap.add_argument("--rsi-period", type=int, default=2)
    ap.add_argument("--rsi-entry", type=float, default=10)
    ap.add_argument("--sma-long", type=int, default=200)
    ap.add_argument("--sma-exit", type=int, default=5)
    ap.add_argument("--max-hold", type=int, default=10)
    ap.add_argument("--stop-pct", type=float, default=0.05)
    ap.add_argument("--slip-bps", type=float, default=10)
    ap.add_argument("--brokerage", type=float, default=20)
    ap.add_argument("--stt-pct", type=float, default=0.1)
    args = ap.parse_args()
    r = run(args.universe_file, args.frm, args.to, args.max_conc, args.capital,
            args.rsi_period, args.rsi_entry, args.sma_long, args.sma_exit,
            args.max_hold, args.stop_pct,
            args.slip_bps, args.brokerage, args.stt_pct)
    if r is None:
        print("no data")
        return
    m = r["monthly"]["ret_pct"]
    y = r["yearly"]["ret_pct"]
    wins = sum(1 for t in r["trades"] if t["pnl"] > 0)
    wr = wins / max(1, len(r["trades"])) * 100
    print(f"Final ₹{r['final']:,.0f} ({(r['final']/args.capital-1)*100:+.1f}%)")
    print(f"Avg/yr {y.mean():+.2f}%  Avg/mo {m.mean():+.2f}%  "
          f"best {m.max():+.1f}%  worst {m.min():+.1f}%")
    print(f"Max DD {r['max_dd']:.2f}%  Trades {len(r['trades'])}  WR {wr:.1f}%  "
          f"fees ₹{r['fees']:,.0f}")
    print(f"Per-year: {dict(y.round(2))}")


if __name__ == "__main__":
    main()
