"""Momentum Rotation live signal generator.

Different from tools/live/signal_generator.py (state-machine strategies).
This is a RANKER: ranks universe by 60d return, picks top-N, emits
ENTRY1 / TARGET_HIT / STOP_HIT signals compatible with paper_executor.

Strategy = Model 3 (from backtest WINNERS_FOUND.md):
  - Universe: pseudo-N100 (top-100 by 20-day ADV)
  - top_n = 5
  - max_concurrent = 1
  - rebalance: 1st of month (or first trading day on/after)

Logic per run:
  1. Load current paper ledger -> currently held symbol (if any)
  2. Rank universe by 60d return; pick top-N
  3. If held NOT in top-N -> emit STOP_HIT (rotation exit)
  4. Emit ENTRY1 for rank-1 stock if not already held

Usage:
  python tools/live/momentum_rotation_signal.py \
    --universe-file paper_portfolio/universes/n100.json \
    --top-n 5 \
    --rebalance-only \
    --signals-out signals/$(date +%F)_momrot_n100.json

Flags:
  --rebalance-only       only emit signals on 1st-of-month (or after weekend)
  --force                emit regardless of date
  --ledger PATH          paper ledger to read current holdings (optional)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402

log = logging.getLogger("momrot_signal")


def is_rebalance_day(today: datetime, last_rotation: datetime = None) -> bool:
    """True if today is rebalance trigger.

    Rule: rebalance once per month. Trigger on first weekday on/after
    the 1st of month. If last rotation was this month already, skip.
    """
    if last_rotation and last_rotation.year == today.year and last_rotation.month == today.month:
        return False
    # Today is 1st-7th of month and weekday (Mon-Fri)
    if today.day <= 7 and today.weekday() < 5:
        return True
    return False


def load_universe(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)["stocks"]


def get_close_at(symbol: str, target_ts: int) -> float:
    df = read_cached(symbol, "D", target_ts - 90 * 86400, target_ts)
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


def rank_universe(stocks: List[Dict], today_ts: int,
                  lookback_days: int = 60) -> List[tuple]:
    """Return [(symbol, name, 60d_return%, current_price)] sorted desc."""
    lookback_ts = today_ts - lookback_days * 86400
    rows = []
    for s in stocks:
        sym = s["symbol"]
        c_now = get_close_at(sym, today_ts)
        c_past = get_close_at(sym, lookback_ts)
        if c_now > 0 and c_past > 0:
            ret = (c_now / c_past - 1) * 100
            rows.append((sym, s.get("name", sym), ret, c_now))
    rows.sort(key=lambda r: -r[2])
    return rows


def load_held(ledger_path: Path) -> List[Dict]:
    if not ledger_path or not ledger_path.exists():
        return []
    try:
        with open(ledger_path) as f:
            return json.load(f).get("open", [])
    except Exception as e:
        log.warning(f"ledger read fail: {e}")
        return []


def emit_signals(top_picks: List[tuple], held: List[Dict],
                  top_n: int) -> List[Dict]:
    top_syms = {p[0] for p in top_picks[:top_n]}
    held_syms = {h["symbol"] for h in held}
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals = []

    # Exits: held but no longer in top-N
    for h in held:
        if h["symbol"] not in top_syms:
            price = get_close_at(h["symbol"], int(datetime.now().timestamp()))
            kind = "TARGET_HIT" if price >= h["entry_price"] else "STOP_HIT"
            signals.append({
                "model": "momentum_rotation",
                "universe": "n100_pseudo",
                "symbol": h["symbol"],
                "company": h["symbol"],
                "ts": today_str,
                "side": "BUY",
                "signal": kind,
                "price": float(price),
                "sl": 0.0, "target": 0.0,
                "note": f"rotation exit (dropped out of top-{top_n})",
            })

    # Entries: rank-1 stock if no held position already in top-N
    # (max_concurrent=1 means take rank-1 if not held)
    if not any(h["symbol"] in top_syms for h in held) and top_picks:
        sym, name, ret, price = top_picks[0]
        signals.append({
            "model": "momentum_rotation",
            "universe": "n100_pseudo",
            "symbol": sym,
            "company": name,
            "ts": today_str,
            "side": "BUY",
            "signal": "ENTRY1",
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"60d momentum rank-1 ({ret:+.2f}%)",
        })

    return signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ledger", default=None,
                    help="Paper ledger JSON to read current holdings")
    ap.add_argument("--rebalance-only", action="store_true",
                    help="Skip if today is not rebalance trigger day")
    ap.add_argument("--force", action="store_true",
                    help="Bypass rebalance-day check (initial deploy / manual)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    log.info(f"momentum_rotation_signal run: today={today.date()} "
             f"weekday={today.strftime('%A')} day_of_month={today.day}")

    if args.rebalance_only and not args.force:
        if not is_rebalance_day(today):
            log.info(f"Not rebalance day (need day<=7 + weekday). Skipping.")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.signals_out, "w") as f:
                json.dump([], f)
            return 0

    stocks = load_universe(args.universe_file)
    log.info(f"Universe: {len(stocks)} symbols from {args.universe_file}")

    held = load_held(Path(args.ledger)) if args.ledger else []
    log.info(f"Currently held: {[h['symbol'] for h in held]}")

    today_ts = int(today.timestamp())
    ranks = rank_universe(stocks, today_ts)
    log.info(f"Ranked {len(ranks)} stocks. Top-{args.top_n}:")
    for i, (sym, name, ret, price) in enumerate(ranks[:args.top_n], 1):
        log.info(f"  {i}. {sym:<14} {ret:+7.2f}%  @ ₹{price:.2f}")

    signals = emit_signals(ranks, held, args.top_n)
    log.info(f"Emitting {len(signals)} signals")

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.signals_out, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {args.signals_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
