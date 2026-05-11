"""Backtest harness — EMA Pullback Breakout (swing, daily bars).

Reuses Fyers fetcher + universe loaders from run_ema_200_400_backtest.
Output dir layout matches the EMA harness so realistic_capital_sim.py
and walk_forward_sim.py work unchanged.

Usage:
  docker exec trading_system_app python tools/backtests/run_swing_pullback_backtest.py \
    --universe nifty50 --days 365 --warmup-days 250 --source fyers \
    --out exports/backtests/n50_swing_pullback_365d
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

from src.services.technical.ema_pullback_breakout import (  # noqa: E402
    EMAPullbackBreakoutStrategy, PullbackConfig,
)
# Reuse Fyers fetcher + universe loaders from EMA harness.
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS, SMOKE_SYMBOLS, INDEX_SYMBOLS, nifty500_symbols,
    to_fyers_symbol, _fyers_service, _FYERS_CACHE,
    df_to_candles, simulate_pnl, render_md, summarize_subset,
)


def _fetch_daily_fyers_raw(symbol: str, days: int, user_id: int = 1,
                            chunk_days: int = 365) -> pd.DataFrame:
    """Inner Fyers daily fetcher (no cache). Wrapped by _fetch_daily_fyers below."""
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
                interval="D",
                start_date=cursor.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
            )
            if res and res.get("status") == "success":
                all_candles += res.get("data", {}).get("candles", []) or []
            else:
                msg = (res or {}).get("message", "no response")
                print(f"  fyers daily fail {fyers_sym} "
                      f"{cursor.date()}..{chunk_end.date()}: {msg}")
        except Exception as e:
            print(f"  fyers daily error {fyers_sym}: {e}")
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


def _fetch_daily_fyers(symbol: str, days: int, user_id: int = 1) -> pd.DataFrame:
    """Postgres-cached daily fetcher."""
    try:
        from tools.backtests.ohlcv_cache import get_or_fetch
    except Exception:
        return _fetch_daily_fyers_raw(symbol, days, user_id)
    return get_or_fetch(symbol, "D", days,
                        lambda s, d: _fetch_daily_fyers_raw(s, d, user_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "nifty50", "nifty500", "indices"],
                        default="nifty50")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--warmup-days", type=int, default=250,
                        help="Daily bars need EMA200 warmup → ~250 trading days "
                             "= ~365 calendar days. Adjust if shorter EMAs used.")
    parser.add_argument("--from", dest="date_from", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days). Strategy "
                             "evaluation gated to >= this date.")
    parser.add_argument("--to", dest="date_to", type=str, default=None,
                        help="End date YYYY-MM-DD (defaults to today when --from given).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "exports" / "backtests" / "swing_pullback")
    parser.add_argument("--source", choices=["fyers"], default="fyers",
                        help="Daily bars: Fyers only (Yahoo daily has gaps)")
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-med",  type=int, default=50)
    parser.add_argument("--ema-slow", type=int, default=200)
    parser.add_argument("--rsi-min", type=float, default=50)
    parser.add_argument("--rsi-max", type=float, default=70)
    parser.add_argument("--volume-mult", type=float, default=1.5)
    parser.add_argument("--sl-atr-mult", type=float, default=1.5)
    parser.add_argument("--tp1-atr-mult", type=float, default=2.0)
    parser.add_argument("--time-stop-bars", type=int, default=10)
    parser.add_argument("--min-adv-inr", type=float, default=5_00_00_000,
                        help="Min avg-daily-value INR. Default ₹5cr.")
    parser.add_argument("--min-price", type=float, default=50.0,
                        help="Penny filter — skip entries on stocks priced "
                             "below this. Default ₹50. Set 0 to disable.")
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
    if args.limit and args.universe == "nifty500":
        pass  # already capped above
    elif args.limit:
        symbols_list = symbols_list[:args.limit]

    config = PullbackConfig(
        ema_fast_period=args.ema_fast,
        ema_med_period=args.ema_med,
        ema_slow_period=args.ema_slow,
        rsi_min=args.rsi_min, rsi_max=args.rsi_max,
        volume_mult=args.volume_mult,
        sl_atr_mult=args.sl_atr_mult, tp1_atr_mult=args.tp1_atr_mult,
        time_stop_bars=args.time_stop_bars,
        min_adv_inr=args.min_adv_inr,
        min_price=args.min_price,
    )
    strat = EMAPullbackBreakoutStrategy(config)
    print(f"Universe: {args.universe} ({len(symbols_list)})")
    print(f"Config: EMA{config.ema_fast_period}/{config.ema_med_period}/{config.ema_slow_period} "
          f"RSI[{config.rsi_min},{config.rsi_max}] vol≥{config.volume_mult}x "
          f"SL={config.sl_atr_mult}×ATR T1={config.tp1_atr_mult}×ATR(50%)")

    # Resolve report window. --from/--to override --days.
    if args.date_from:
        from_dt = datetime.strptime(args.date_from, "%Y-%m-%d")
        to_dt = datetime.strptime(args.date_to, "%Y-%m-%d") if args.date_to else datetime.now()
        # Fetch enough days to cover [from .. now] (strategy walks every bar)
        # and the warmup buffer; evaluation is gated by eval_from_ts.
        args.days = max(1, (datetime.now() - from_dt).days)
        cutoff_dt = from_dt
        eval_from_ts = int(from_dt.timestamp())
        eval_to_ts = int(to_dt.timestamp())
        print(f"Date range: {from_dt.date()} → {to_dt.date()} ({args.days} days)")
    else:
        cutoff_dt = datetime.now() - timedelta(days=args.days)
        eval_from_ts = int(cutoff_dt.timestamp())
        eval_to_ts = None
    fetch_days = args.days + args.warmup_days

    aggregate = []
    for symbol, name in symbols_list:
        print(f"--- {symbol} ---", flush=True)
        df = _fetch_daily_fyers(symbol, days=fetch_days, user_id=args.user_id)
        if df.empty:
            print(f"  no data, skipping")
            continue
        candles = df_to_candles(df)
        if len(candles) < config.ema_slow_period + config.atr_period + 5:
            print(f"  only {len(candles)} bars, skipping")
            continue
        signals = strat.evaluate(user_id=1, symbol=symbol, candles=candles,
                                  eval_from_ts=eval_from_ts)
        if eval_to_ts is not None:
            signals = [s for s in signals if int(s.get("candle_ts", 0)) <= eval_to_ts]
        pnl = simulate_pnl(signals, df, symbol,
                           partial_qty_frac=config.tp1_qty_frac)
        path = render_md(symbol, name, df, signals, pnl, args.out, config=None)
        closed = pnl.get("closed") or []
        buy_stats = summarize_subset([c for c in closed if c["trend"] == "BUY"])
        print(f"  bars={len(candles)} legs={pnl['trades_closed']} "
              f"win={buy_stats['win_rate']:.1f}% sum={buy_stats['sum_pct']:.1f}% -> {path}")
        aggregate.append({
            "symbol": symbol, "name": name, "bars": len(candles),
            "signals": len(signals), "closed_legs": closed,
            "trades_closed": pnl["trades_closed"], "winners": pnl["winners"],
            "target_hits": pnl["target_hits"], "stop_hits": pnl["stop_hits"],
            "partials": pnl["partials"], "sum_pct": pnl["sum_pct"],
            "avg_pct": pnl["avg_pct"], "buy": buy_stats,
            "sell": summarize_subset([]),
        })

    # Summary
    total_closed = sum(a['trades_closed'] for a in aggregate)
    total_winners = sum(a['winners'] for a in aggregate)
    total_sum_pct = sum(a['sum_pct'] for a in aggregate)
    avg = (total_sum_pct / total_closed) if total_closed else 0.0
    profitable = sum(1 for a in aggregate if a['sum_pct'] > 0)
    out_lines = [
        "# Swing Pullback Breakout — Backtest Summary",
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
        "| Symbol | Bars | Sig | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |",
        "|--------|------|-----|------|-----|------|-----|----|-----|-------|-------|",
    ]
    for a in aggregate:
        b = a["buy"]
        out_lines.append(
            f"| {a['symbol']} | {a['bars']} | {a['signals']} | {a['trades_closed']} | "
            f"{b['winners']} | {b['win_rate']:.1f}% | "
            f"{b['target_hits']} | {b['stop_hits']} | {b['partials']} | "
            f"{b['avg_pct']:.2f}% | {b['sum_pct']:.1f}% |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_summary.md").write_text("\n".join(out_lines) + "\n")
    print(f"\nSummary -> {args.out / '_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
