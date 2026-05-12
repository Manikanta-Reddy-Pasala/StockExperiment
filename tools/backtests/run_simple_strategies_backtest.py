"""Run VCP / Donchian 20-55 / 52wH+volume / Bollinger squeeze strategies
on selector top-10 stocks. Emits per-stock .md files with Strategy Cycles
table that realistic_capital_sim.py can consume.

Daily bars only (uses historical_data daily cache).

Usage:
  python tools/backtests/run_simple_strategies_backtest.py \
    --strategy donchian --top-n 10 --start 2025-05-12 --end 2026-05-12 \
    --out exports/backtests/y1_donchian/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402

log = logging.getLogger("simple_strategies")


# ============================================================
# Strategy: Donchian 20/55 Channel Breakout (Turtle Trader)
# ============================================================
def strat_donchian(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10,
                    atr_sl: float = 2.0) -> List[Dict]:
    """Long-only Donchian breakout.
    ENTRY: close > rolling max(high, entry_n)
    STOP: rolling min(low, exit_n) (Chandelier-like)
    TARGET: 20% above entry (R:R ~3-4)
    """
    events = []
    if len(df) < entry_n + 14:
        return events
    high_n = df["high"].rolling(entry_n).max().shift(1)
    low_exit = df["low"].rolling(exit_n).min().shift(1)
    # ATR for stop adjustment
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    in_position = False
    entry_price = 0.0
    entry_idx = -1
    for i in range(len(df)):
        c = df.iloc[i]
        if pd.isna(high_n.iloc[i]) or pd.isna(atr14.iloc[i]):
            continue
        if not in_position:
            if c["close"] > high_n.iloc[i]:
                entry_price = float(c["close"])
                stop = entry_price - atr_sl * float(atr14.iloc[i])
                target = entry_price * 1.20
                events.append({
                    "stage": "First Entry",
                    "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                    "price": entry_price,
                    "stop": stop, "target": target, "kind": "ENTRY",
                })
                in_position = True
                entry_idx = i
        else:
            if c["close"] >= target:
                events.append({"stage": "Target hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "TARGET"})
                in_position = False
            elif c["close"] <= stop or c["close"] < low_exit.iloc[i]:
                events.append({"stage": "Stop hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "STOP"})
                in_position = False
    return events


# ============================================================
# Strategy: 52-week high + 2x volume breakout (O'Neil)
# ============================================================
def strat_52wh_vol(df: pd.DataFrame, vol_mult: float = 2.0,
                    target_pct: float = 0.15, atr_sl: float = 2.5) -> List[Dict]:
    """Long-only 52wH breakout with volume confirmation."""
    events = []
    if len(df) < 260:
        return events
    high_52w = df["high"].rolling(252).max().shift(1)
    vol_avg = df["volume"].rolling(50).mean().shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    in_position = False
    entry_price = 0.0
    for i in range(len(df)):
        c = df.iloc[i]
        if pd.isna(high_52w.iloc[i]) or pd.isna(vol_avg.iloc[i]) or pd.isna(atr14.iloc[i]):
            continue
        if not in_position:
            if (c["close"] > high_52w.iloc[i] and
                c["volume"] > vol_mult * vol_avg.iloc[i]):
                entry_price = float(c["close"])
                stop = entry_price - atr_sl * float(atr14.iloc[i])
                target = entry_price * (1 + target_pct)
                events.append({"stage": "First Entry",
                               "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": entry_price, "kind": "ENTRY"})
                in_position = True
        else:
            if c["close"] >= target:
                events.append({"stage": "Target hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "TARGET"})
                in_position = False
            elif c["close"] <= stop:
                events.append({"stage": "Stop hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "STOP"})
                in_position = False
    return events


# ============================================================
# Strategy: Bollinger Squeeze + Expansion
# ============================================================
def strat_bb_squeeze(df: pd.DataFrame, bb_period: int = 20,
                      bb_std: float = 2.0, squeeze_pct: float = 0.10) -> List[Dict]:
    """BB squeeze (width < 10th percentile of last year) then expansion long."""
    events = []
    if len(df) < bb_period + 252:
        return events
    sma = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    width = (upper - lower) / sma
    # Squeeze threshold: 10th percentile of last 252 bars
    in_position = False
    entry_price = 0.0
    for i in range(bb_period + 252, len(df)):
        c = df.iloc[i]
        if pd.isna(width.iloc[i]):
            continue
        squeeze_threshold = width.iloc[i - 252:i].quantile(squeeze_pct)
        if not in_position:
            # Squeeze in prior 5 bars + breakout above upper today
            prior_squeeze = (width.iloc[max(i - 5, 0):i] < squeeze_threshold).any()
            if prior_squeeze and c["close"] > upper.iloc[i]:
                entry_price = float(c["close"])
                stop = entry_price * 0.94    # 6% SL
                target = entry_price * 1.15   # 15% target
                events.append({"stage": "First Entry",
                               "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": entry_price, "kind": "ENTRY"})
                in_position = True
        else:
            if c["close"] >= target:
                events.append({"stage": "Target hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "TARGET"})
                in_position = False
            elif c["close"] <= stop:
                events.append({"stage": "Stop hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "STOP"})
                in_position = False
    return events


# ============================================================
# Strategy: VCP (Minervini Volatility Contraction Pattern, simplified)
# ============================================================
def strat_vcp(df: pd.DataFrame) -> List[Dict]:
    """Simplified VCP: 3+ pullbacks of decreasing magnitude, then breakout
    above the last pivot high on rising volume.

    This is a simplified version — full VCP requires complex pattern
    recognition. We approximate by:
    1. Stock in Stage 2 (close > 200DMA, 50DMA > 200DMA)
    2. Last 60 days show declining ATR (volatility contraction)
    3. Today's close > 20-bar high on >1.5x avg volume → ENTRY
    """
    events = []
    if len(df) < 220:
        return events
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    high20 = df["high"].rolling(20).max().shift(1)
    vol_avg = df["volume"].rolling(50).mean().shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr60_first = atr14.shift(45)
    atr60_last = atr14.shift(15)
    in_position = False
    entry_price = 0.0
    for i in range(220, len(df)):
        c = df.iloc[i]
        if pd.isna(sma200.iloc[i]) or pd.isna(atr60_first.iloc[i]):
            continue
        if not in_position:
            stage2 = (c["close"] > sma200.iloc[i] and sma50.iloc[i] > sma200.iloc[i])
            contracting = atr60_last.iloc[i] < atr60_first.iloc[i] * 0.7
            breakout = c["close"] > high20.iloc[i]
            vol_spike = c["volume"] > 1.5 * vol_avg.iloc[i] if not pd.isna(vol_avg.iloc[i]) else False
            if stage2 and contracting and breakout and vol_spike:
                entry_price = float(c["close"])
                stop = entry_price - 2.0 * float(atr14.iloc[i])
                target = entry_price * 1.20
                events.append({"stage": "First Entry",
                               "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": entry_price, "kind": "ENTRY"})
                in_position = True
        else:
            if c["close"] >= target:
                events.append({"stage": "Target hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "TARGET"})
                in_position = False
            elif c["close"] <= stop:
                events.append({"stage": "Stop hit", "ts": c["candle_time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(c["candle_time"], "strftime") else str(c["candle_time"])[:19],
                               "price": float(c["close"]), "kind": "STOP"})
                in_position = False
    return events


STRATEGIES = {
    "donchian": strat_donchian,
    "52wh": strat_52wh_vol,
    "bb_squeeze": strat_bb_squeeze,
    "vcp": strat_vcp,
}


def fyers_sym_to_plain(s: str) -> str:
    return s.replace("NSE:", "").replace("-EQ", "").upper()


def load_selector_top(top_n: int = 10, selector_json: str = None) -> List[str]:
    """Load symbols from selector JSON or use hardcoded top-10."""
    if selector_json and os.path.exists(selector_json):
        data = json.loads(Path(selector_json).read_text())
        return [s["symbol"] for s in data["stocks"][:top_n]]
    return ["SWIGGY", "VMM", "AEGISLOG", "ANGELONE", "SAILIFE",
            "ITI", "IKS", "AMBER", "NTPCGREEN", "BSE"][:top_n]


def render_per_stock_md(symbol: str, events: List[Dict], strategy: str,
                         out_dir: Path) -> None:
    """Emit per-stock .md in same format as existing backtest harnesses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {symbol}\n",
        f"## Backtest Summary\n",
        f"- Strategy: {strategy}",
        f"- Closed legs: {sum(1 for e in events if e['kind'] in ('TARGET', 'STOP'))}",
        f"- Entries: {sum(1 for e in events if e['kind'] == 'ENTRY')}",
        "",
        "## Strategy Cycles",
        "",
        "| Stage | Timestamp | Price | sl | target | note |",
        "|---|---|---|---|---|---|",
    ]
    for e in events:
        lines.append(f"| {e['stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
    (out_dir / f"{symbol.lower()}.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--selector", default=None,
                    help="Selector JSON path (overrides hardcoded top-10)")
    ap.add_argument("--start", default="2025-05-12")
    ap.add_argument("--end", default="2026-05-12")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    symbols = load_selector_top(args.top_n, args.selector)
    log.info(f"Strategy={args.strategy} on {len(symbols)} symbols, "
             f"window {args.start} → {args.end}")

    start_ts = int(datetime.strptime(args.start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(args.end, "%Y-%m-%d").timestamp())
    # Load extra history for warmup
    warmup_start = int((datetime.strptime(args.start, "%Y-%m-%d") -
                         timedelta(days=400)).timestamp())

    strat_fn = STRATEGIES[args.strategy]
    out_dir = Path(args.out)
    total_entries = 0
    total_closes = 0
    for sym in symbols:
        df = read_cached(sym, "D", warmup_start, end_ts)
        if df.empty:
            log.warning(f"{sym}: no data")
            render_per_stock_md(sym, [], args.strategy, out_dir)
            continue
        # Clip strategy events to backtest window
        events = strat_fn(df)
        # Filter to events in window
        window_events = []
        for e in events:
            try:
                e_ts = int(datetime.strptime(e["ts"][:19], "%Y-%m-%d %H:%M:%S").timestamp())
                if start_ts <= e_ts <= end_ts:
                    window_events.append(e)
            except Exception:
                window_events.append(e)
        total_entries += sum(1 for e in window_events if e["kind"] == "ENTRY")
        total_closes += sum(1 for e in window_events if e["kind"] in ("TARGET", "STOP"))
        render_per_stock_md(sym, window_events, args.strategy, out_dir)
        log.info(f"{sym}: {len(window_events)} events "
                 f"({sum(1 for e in window_events if e['kind'] == 'ENTRY')} entries)")

    log.info(f"Total: {total_entries} entries, {total_closes} closes across {len(symbols)} symbols")
    log.info(f"Wrote per-stock .md files to {out_dir}")


if __name__ == "__main__":
    main()
