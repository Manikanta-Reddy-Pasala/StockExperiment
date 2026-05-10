"""Backtest harness — 15-min Opening Range Breakout (intraday, 5m bars).

Reuses Fyers fetcher + universe loaders from run_ema_200_400_backtest.
Output dir layout matches the EMA harness so realistic_capital_sim.py
and walk_forward_sim.py work unchanged.

Usage:
  docker exec trading_system_app python tools/backtests/run_orb_intraday_backtest.py \
    --universe nifty50 --days 90 --source fyers \
    --out exports/backtests/n50_orb15_90d
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.technical.orb_15min import ORB15MinStrategy, ORBConfig  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS, SMOKE_SYMBOLS, INDEX_SYMBOLS, nifty500_symbols,
    to_fyers_symbol, _fyers_service, _FYERS_CACHE,
    df_to_candles, simulate_pnl, render_md, summarize_subset,
)


def _fetch_5m_fyers(symbol: str, days: int, user_id: int = 1,
                    chunk_days: int = 30) -> pd.DataFrame:
    svc = _fyers_service()
    if svc is None:
        return pd.DataFrame()
    fyers_sym = to_fyers_symbol(symbol)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    cursor = start_dt
    all_candles: List = []
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            res = svc.history(
                user_id=user_id, symbol=fyers_sym, exchange="NSE",
                interval="5m",
                start_date=cursor.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
            )
            if res and res.get("status") == "success":
                all_candles += res.get("data", {}).get("candles", []) or []
            else:
                msg = (res or {}).get("message", "no response")
                print(f"  fyers 5m fail {fyers_sym} "
                      f"{cursor.date()}..{chunk_end.date()}: {msg}")
        except Exception as e:
            print(f"  fyers 5m error {fyers_sym}: {e}")
        cursor = chunk_end

    if not all_candles:
        return pd.DataFrame()
    if isinstance(all_candles[0], dict):
        df = pd.DataFrame(all_candles)
    else:
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high",
                                                  "low", "close", "volume"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["candle_time"] = pd.to_datetime(df["timestamp"].astype("int64"),
                                        unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", "candle_time", "open", "high", "low", "close", "volume"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "nifty50", "nifty500", "indices"],
                        default="nifty50")
    parser.add_argument("--days", type=int, default=90,
                        help="Calendar days of 5m history. ORB needs no warmup.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "exports" / "backtests" / "orb15_intraday")
    parser.add_argument("--source", choices=["fyers"], default="fyers")
    parser.add_argument("--volume-mult", type=float, default=1.5)
    parser.add_argument("--orb-range-min-pct", type=float, default=0.003)
    parser.add_argument("--orb-range-max-pct", type=float, default=0.015)
    parser.add_argument("--tp1-r-mult", type=float, default=1.5)
    parser.add_argument("--sl-atr-mult", type=float, default=1.0)
    parser.add_argument("--no-vwap", action="store_true",
                        help="Skip VWAP confirmation (looser entries)")
    parser.add_argument("--long-only", action="store_true",
                        help="Disable SELL side (Indian retail default)")
    args = parser.parse_args()
    _FYERS_CACHE["user_id"] = args.user_id

    if args.universe == "nifty50":
        symbols_list = NIFTY50_SYMBOLS
    elif args.universe == "nifty500":
        symbols_list = nifty500_symbols(limit=args.limit)
    elif args.universe == "indices":
        symbols_list = INDEX_SYMBOLS
    else:
        symbols_list = SMOKE_SYMBOLS
    if args.limit and args.universe != "nifty500":
        symbols_list = symbols_list[:args.limit]

    config = ORBConfig(
        require_vwap=not args.no_vwap,
        volume_mult=args.volume_mult,
        orb_range_min_pct=args.orb_range_min_pct,
        orb_range_max_pct=args.orb_range_max_pct,
        sl_atr_mult=args.sl_atr_mult, tp1_r_mult=args.tp1_r_mult,
        enable_long=True, enable_short=not args.long_only,
    )
    strat = ORB15MinStrategy(config)
    print(f"Universe: {args.universe} ({len(symbols_list)})")
    print(f"Config: ORB[09:15-09:29] window[09:30-11:15] vol≥{config.volume_mult}x "
          f"VWAP={'on' if config.require_vwap else 'off'} "
          f"long={config.enable_long} short={config.enable_short}")

    aggregate = []
    for symbol, name in symbols_list:
        print(f"--- {symbol} ---", flush=True)
        df = _fetch_5m_fyers(symbol, days=args.days, user_id=args.user_id)
        if df.empty:
            print(f"  no data, skipping")
            continue
        candles = df_to_candles(df)
        signals = strat.evaluate(user_id=1, symbol=symbol, candles=candles,
                                  eval_from_ts=None)
        pnl = simulate_pnl(signals, df, symbol,
                           partial_qty_frac=config.tp1_qty_frac)
        path = render_md(symbol, name, df, signals, pnl, args.out, config=None)
        closed = pnl.get("closed") or []
        b = summarize_subset([c for c in closed if c["trend"] == "BUY"])
        s = summarize_subset([c for c in closed if c["trend"] == "SELL"])
        print(f"  bars={len(candles)} legs={pnl['trades_closed']} "
              f"BUY({b['legs']} {b['win_rate']:.0f}% {b['sum_pct']:.0f}%) "
              f"SELL({s['legs']} {s['win_rate']:.0f}% {s['sum_pct']:.0f}%) -> {path}")
        aggregate.append({
            "symbol": symbol, "name": name, "bars": len(candles),
            "signals": len(signals), "closed_legs": closed,
            "trades_closed": pnl["trades_closed"], "winners": pnl["winners"],
            "target_hits": pnl["target_hits"], "stop_hits": pnl["stop_hits"],
            "partials": pnl["partials"], "sum_pct": pnl["sum_pct"],
            "avg_pct": pnl["avg_pct"], "buy": b, "sell": s,
        })

    total_closed = sum(a['trades_closed'] for a in aggregate)
    total_winners = sum(a['winners'] for a in aggregate)
    total_sum_pct = sum(a['sum_pct'] for a in aggregate)
    avg = (total_sum_pct / total_closed) if total_closed else 0.0
    profitable = sum(1 for a in aggregate if a['sum_pct'] > 0)
    out_lines = [
        "# 15-min ORB — Backtest Summary",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Headline",
        f"- Symbols processed: {len(aggregate)} (profitable: {profitable})",
        f"- Closed legs: {total_closed}",
        f"- Win rate: {(total_winners / total_closed * 100) if total_closed else 0:.1f}%",
        f"- **Avg % per leg: {avg:.2f}%**",
        f"- **Sum % across all legs (uncompounded): {total_sum_pct:.1f}%**",
        "",
        "## Per-symbol",
        "| Symbol | Bars | Sig | Legs | BUY legs | BUY sum% | SELL legs | SELL sum% | Sum % |",
        "|--------|------|-----|------|---------|---------|----------|---------|-------|",
    ]
    for a in aggregate:
        b, s = a["buy"], a["sell"]
        out_lines.append(
            f"| {a['symbol']} | {a['bars']} | {a['signals']} | {a['trades_closed']} | "
            f"{b['legs']} | {b['sum_pct']:.1f}% | "
            f"{s['legs']} | {s['sum_pct']:.1f}% | {a['sum_pct']:.1f}% |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_summary.md").write_text("\n".join(out_lines) + "\n")
    print(f"\nSummary -> {args.out / '_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
