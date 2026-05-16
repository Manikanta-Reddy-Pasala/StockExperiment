"""Breakout backtest v2 with risk overlays for return boost + DD reduction.

Added levers vs v1:
  1. REGIME FILTER  — only enter when NIFTY50 close >= its 200d SMA
                      (skip bear regimes → cuts DD)
  2. ATR TRAIL      — chandelier exit using N×ATR(14) below peak
                      (adapts trail to per-stock volatility)
  3. PARTIAL EXIT   — sell 50% at +20% gain, trail the rest
                      (locks in profits, reduces giveback)
  4. ENTRY QUALITY  — require 90d return >= +5% to confirm prior trend

All toggleable. Default config: regime ON, ATR ON, partial ON, quality ON.

Usage:
  python tools/models/breakout_60d_high_volume/backtest_v2.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --from 2023-05-15 --to 2026-05-15 --capital 200000 --max-conc 1 \
    --regime-on --atr-on --partial-on --quality-on \
    --out /app/logs/breakout_v2.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

log = logging.getLogger("breakout_v2")

NIFTY_SPOT = "NSE:NIFTY50-INDEX"


def load_universe(universe_file: str) -> List[str]:
    with open(universe_file) as f:
        data = json.load(f)
    plain = [s["symbol"] for s in data.get("stocks", [])]
    return [f"NSE:{s}-EQ" if not s.startswith("NSE:") else s for s in plain]


def load_daily_full(symbols: List[str], start, end) -> pd.DataFrame:
    eng = _get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM historical_data "
            "WHERE symbol = ANY(:syms) AND date BETWEEN :a AND :b "
            "ORDER BY symbol, date"
        ), conn, params={"syms": symbols, "a": start, "b": end})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def run(universe_file: str, start_str: str, end_str: str,
        max_conc: int, capital: float,
        breakout_lookback: int, vol_mult: float,
        sma_exit: int, trail_pct: float, max_hold_days: int,
        slip_bps: float, brokerage: float, stt_pct: float,
        regime_on: bool, atr_on: bool, partial_on: bool, quality_on: bool,
        atr_mult: float = 3.0, quality_min_90d: float = 0.05,
        partial_trigger: float = 0.20, partial_pct: float = 0.50) -> dict:
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    syms = load_universe(universe_file)
    log.info(f"Universe: {len(syms)} symbols")
    log.info(f"Filters: regime={regime_on} atr={atr_on} partial={partial_on} "
             f"quality={quality_on}")

    fetch_start = start - timedelta(days=max(breakout_lookback, 250) + 60)
    df_all = load_daily_full(syms + [NIFTY_SPOT], fetch_start, end)
    if df_all.empty:
        raise RuntimeError("no price data")

    # Pre-compute per-symbol indicators
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym, grp in df_all.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        g["high_lookback"] = g["close"].rolling(breakout_lookback).max()
        g["vol_avg"] = g["volume"].rolling(20).mean()
        g["sma_exit"] = g["close"].rolling(sma_exit).mean()
        g["sma200"] = g["close"].rolling(200).mean()
        g["ret_90d"] = g["close"].pct_change(90)
        # ATR(14)
        prev = g["close"].shift(1)
        tr = pd.concat([
            (g["high"] - g["low"]).abs(),
            (g["high"] - prev).abs(),
            (g["low"] - prev).abs(),
        ], axis=1).max(axis=1)
        g["atr14"] = tr.rolling(14).mean()
        sym_data[sym] = g

    # NIFTY regime
    nifty_df = sym_data.get(NIFTY_SPOT)
    nifty_regime = {}
    if nifty_df is not None:
        for r in nifty_df.itertuples():
            nifty_regime[r.date.date()] = (
                pd.notna(r.sma200) and r.close >= r.sma200
            )

    all_dates = sorted({d for sym, df in sym_data.items() if sym != NIFTY_SPOT
                        for d in df["date"]})
    trading_days = [d for d in all_dates if start <= d.date() <= end]
    log.info(f"Trading days: {len(trading_days)}")

    slip = slip_bps / 10000.0
    stt = stt_pct / 100.0

    cash = capital
    positions: Dict[str, dict] = {}
    daily = []
    trades = []
    total_fees = 0.0

    for d in trading_days:
        # 1. Mark-to-market + update peaks + ATR snapshot
        nav = cash
        for sym, p in list(positions.items()):
            row = sym_data[sym][sym_data[sym]["date"] == d]
            if row.empty:
                nav += p["qty"] * p["last_close"]
                continue
            r = row.iloc[0]
            close = float(r["close"])
            p["last_close"] = close
            p["peak"] = max(p.get("peak", p["entry_price"]), close)
            p["last_sma"] = float(r["sma_exit"]) if pd.notna(r["sma_exit"]) else 0.0
            p["last_atr"] = float(r["atr14"]) if pd.notna(r["atr14"]) else 0.0
            nav += p["qty"] * close

        # 2. Exit checks
        for sym, p in list(positions.items()):
            close = p["last_close"]
            sma = p.get("last_sma", 0.0)
            atr = p.get("last_atr", 0.0)
            age = (d.date() - p["entry_date"]).days
            exit_reason = None

            # Partial profit booking (first hit only)
            if (partial_on and not p.get("partial_done")
                    and close >= p["entry_price"] * (1 + partial_trigger)):
                sell_qty = max(1, int(p["qty"] * partial_pct))
                exit_px = close * (1 - slip)
                proceeds = sell_qty * exit_px
                fees = proceeds * stt + brokerage
                total_fees += fees
                pnl = proceeds - fees - (sell_qty * p["entry_price"])
                cash += proceeds - fees
                trades.append({
                    "entry_date": p["entry_date"].isoformat(),
                    "exit_date": d.date().isoformat(),
                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                    "entry_px": round(p["entry_price"], 2),
                    "exit_px": round(exit_px, 2),
                    "qty": sell_qty, "pnl": round(pnl, 2),
                    "hold_days": age, "reason": "PARTIAL_TP",
                })
                p["qty"] -= sell_qty
                p["partial_done"] = True
                if p["qty"] < 1:
                    del positions[sym]
                    continue

            # Trailing stop
            if atr_on and atr > 0:
                stop_level = p["peak"] - atr_mult * atr
                if close <= stop_level:
                    exit_reason = "ATR_TRAIL"
            else:
                drop_from_peak = (p["peak"] - close) / p["peak"]
                if drop_from_peak >= trail_pct:
                    exit_reason = "TRAIL_STOP"

            if not exit_reason:
                if sma > 0 and close < sma:
                    exit_reason = "SMA_EXIT"
                elif age >= max_hold_days:
                    exit_reason = "MAX_HOLD"

            if exit_reason:
                exit_px = close * (1 - slip)
                proceeds = p["qty"] * exit_px
                fees = proceeds * stt + brokerage
                total_fees += fees
                pnl = proceeds - fees - (p["qty"] * p["entry_price"])
                cash += proceeds - fees
                trades.append({
                    "entry_date": p["entry_date"].isoformat(),
                    "exit_date": d.date().isoformat(),
                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                    "entry_px": round(p["entry_price"], 2),
                    "exit_px": round(exit_px, 2),
                    "qty": p["qty"], "pnl": round(pnl, 2),
                    "hold_days": age, "reason": exit_reason,
                })
                del positions[sym]

        # 3. Entry scan (gated by regime if enabled)
        regime_ok = (not regime_on) or nifty_regime.get(d.date(), False)
        if regime_ok and len(positions) < max_conc:
            slots = max_conc - len(positions)
            candidates = []
            for sym in syms:
                if sym in positions:
                    continue
                df = sym_data.get(sym)
                if df is None:
                    continue
                row = df[df["date"] == d]
                if row.empty:
                    continue
                r = row.iloc[0]
                if pd.isna(r["high_lookback"]) or pd.isna(r["vol_avg"]):
                    continue
                if pd.isna(r["close"]):
                    continue
                if r["close"] < float(r["high_lookback"]):
                    continue
                if r["volume"] < vol_mult * float(r["vol_avg"]):
                    continue
                if float(r["vol_avg"]) <= 0:
                    continue
                if quality_on:
                    if pd.isna(r["ret_90d"]) or r["ret_90d"] < quality_min_90d:
                        continue
                candidates.append({
                    "symbol": sym,
                    "close": float(r["close"]),
                    "vol_ratio": float(r["volume"]) / float(r["vol_avg"]),
                })
            candidates.sort(key=lambda c: -c["vol_ratio"])
            for c in candidates[:slots]:
                sym = c["symbol"]
                df = sym_data[sym]
                nxt = df[df["date"] > d]
                if nxt.empty:
                    continue
                entry_row = nxt.iloc[0]
                if pd.isna(entry_row["open"]):
                    continue
                entry_px = float(entry_row["open"]) * (1 + slip)
                allocate = cash / max(1, slots)
                qty = int(allocate / entry_px)
                if qty < 1:
                    continue
                cost = qty * entry_px + brokerage
                total_fees += brokerage
                cash -= cost
                positions[sym] = {
                    "qty": qty, "entry_price": entry_px,
                    "entry_date": entry_row["date"].date(),
                    "peak": entry_px, "last_close": entry_px,
                    "last_sma": 0.0, "last_atr": 0.0,
                    "partial_done": False,
                }
                slots -= 1
                if slots <= 0:
                    break

        daily.append({"date": d.date(), "nav": nav, "cash": cash,
                      "open_pos": len(positions)})

    eq = pd.DataFrame(daily).set_index(pd.to_datetime([r["date"] for r in daily]))
    eq.index.name = "date"
    eq["month"] = eq.index.to_period("M").astype(str)
    eq["year"] = eq.index.year
    eq["peak"] = eq["nav"].cummax()
    eq["dd_pct"] = (eq["nav"] / eq["peak"] - 1) * 100

    monthly = eq.groupby("month")["nav"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    yearly = eq.groupby("year")["nav"].agg(["first", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100

    return {
        "equity": eq, "monthly": monthly, "yearly": yearly,
        "final": float(eq["nav"].iloc[-1]),
        "max_dd": float(eq["dd_pct"].min()),
        "total_fees": total_fees,
        "trades": trades,
    }


def summarize_and_write(r: dict, capital: float, label: str, out_md):
    eq = r["equity"]
    m = r["monthly"]["ret_pct"]
    y = r["yearly"]["ret_pct"]
    final = r["final"]
    wins = [t for t in r["trades"] if t["pnl"] > 0]
    wr = (len(wins) / len(r["trades"]) * 100) if r["trades"] else 0

    print(f"\n=== {label} ===")
    print(f"Start ₹{capital:,.0f} -> End ₹{final:,.0f} "
          f"({(final/capital-1)*100:+.1f}%)")
    print(f"Avg/mo: {m.mean():+.2f}%  Median: {m.median():+.2f}%")
    print(f"Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%")
    print(f"20%+: {(m>=20).sum()}/{m.count()}  30%+: {(m>=30).sum()}/{m.count()}  "
          f"<-10%: {(m<-10).sum()}/{m.count()}")
    print(f"Avg yearly: {y.mean():+.2f}%")
    print(f"Max DD: {r['max_dd']:.2f}%")
    print(f"Trades: {len(r['trades'])}  WR: {wr:.1f}%  Fees: ₹{r['total_fees']:,.0f}")

    if out_md:
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        lines = [
            f"# {label}\n",
            f"- Capital: ₹{capital:,.0f}",
            f"- Final: ₹{final:,.0f} ({(final/capital-1)*100:+.2f}%)",
            f"- Avg/mo: {m.mean():+.2f}%  Median: {m.median():+.2f}%",
            f"- Best mo: {m.max():+.2f}%  Worst: {m.min():+.2f}%",
            f"- Months ≥ 20%: {(m>=20).sum()} / {m.count()}",
            f"- Months ≥ 30%: {(m>=30).sum()} / {m.count()}",
            f"- Avg yearly: {y.mean():+.2f}%",
            f"- Max DD: {r['max_dd']:.2f}%",
            f"- Trades: {len(r['trades'])}  WR: {wr:.1f}%",
            f"- Fees: ₹{r['total_fees']:,.0f}\n",
            "## Yearly\n",
            "| Year | Start | End | ROI |",
            "|---|---:|---:|---:|",
        ]
        for yr, row in r["yearly"].iterrows():
            lines.append(f"| {yr} | ₹{row['first']:,.0f} | ₹{row['last']:,.0f} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## Monthly\n")
        lines.append("| Month | ROI |")
        lines.append("|---|---:|")
        for mo, row in r["monthly"].iterrows():
            lines.append(f"| {mo} | {row['ret_pct']:+.2f}% |")
        lines.append("\n## All Trades\n")
        lines.append("| Entry | Exit | Symbol | Entry₹ | Exit₹ | Qty | PnL₹ | Days | Reason |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for t in r["trades"]:
            lines.append(f"| {t['entry_date']} | {t['exit_date']} | {t['symbol']} | "
                         f"{t['entry_px']} | {t['exit_px']} | {t['qty']} | "
                         f"{t['pnl']} | {t['hold_days']} | {t['reason']} |")
        Path(out_md).write_text("\n".join(lines))
        print(f"\nReport: {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", dest="to", required=True)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--max-conc", type=int, default=1)
    ap.add_argument("--breakout-lookback", type=int, default=60)
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--sma-exit", type=int, default=20)
    ap.add_argument("--trail-pct", type=float, default=0.08)
    ap.add_argument("--max-hold-days", type=int, default=90)
    ap.add_argument("--slip-bps", type=float, default=10.0)
    ap.add_argument("--brokerage", type=float, default=20.0)
    ap.add_argument("--stt-pct", type=float, default=0.1)
    ap.add_argument("--regime-on", action="store_true")
    ap.add_argument("--atr-on", action="store_true")
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--partial-on", action="store_true")
    ap.add_argument("--partial-trigger", type=float, default=0.20)
    ap.add_argument("--partial-pct", type=float, default=0.50)
    ap.add_argument("--quality-on", action="store_true")
    ap.add_argument("--quality-min-90d", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    r = run(args.universe_file, args.frm, args.to,
            args.max_conc, args.capital,
            args.breakout_lookback, args.vol_mult,
            args.sma_exit, args.trail_pct, args.max_hold_days,
            args.slip_bps, args.brokerage, args.stt_pct,
            args.regime_on, args.atr_on, args.partial_on, args.quality_on,
            args.atr_mult, args.quality_min_90d,
            args.partial_trigger, args.partial_pct)
    parts = [f"v2_60d_max{args.max_conc}"]
    if args.regime_on: parts.append("regime")
    if args.atr_on: parts.append(f"atr{args.atr_mult}x")
    if args.partial_on: parts.append(f"partial{int(args.partial_trigger*100)}@{int(args.partial_pct*100)}")
    if args.quality_on: parts.append(f"qual{int(args.quality_min_90d*100)}")
    label = "_".join(parts) + f" ({args.frm}..{args.to})"
    summarize_and_write(r, args.capital, label, args.out)


if __name__ == "__main__":
    main()
