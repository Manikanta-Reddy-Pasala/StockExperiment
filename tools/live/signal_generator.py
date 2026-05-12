"""Daily signal generator. Pure Python, no LLM. Cron-driven.

Reads OHLCV from Postgres cache (historical_data_1h, _15m, daily).
Runs strategy state machine per symbol. Emits signals to JSON file +
optionally to a `signals` Postgres table.

Usage:
  python tools/live/signal_generator.py --model ema_200_400 --universe nifty50
  python tools/live/signal_generator.py --model swing_pullback --universe nifty500 \
    --signals-out /var/log/trading/signals_2026-05-12.json

The signal file format (JSON list):
  [
    {"ts": "2026-05-12T15:15:00", "symbol": "RELIANCE", "side": "BUY",
     "signal": "ENTRY1", "price": 1385.3, "sl": 1320.0, "target": 1525.0,
     "note": "EMA200 retest1 break sustained"},
    ...
  ]

Scheduled use:
  - Daily 09:14 cron: generate for swing_pullback (daily bars)
  - Hourly 09:30..15:30: generate for ema_200_400 / ema_9_21 (1H bars)
  - 5-min 09:30..11:15: generate for orb_15min (5m bars)

The script DOES NOT place orders. paper_executor / fyers_executor consume
the JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached, _to_fyers_sym  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS, nifty500_symbols, df_to_candles,
)
from src.services.technical.models import get_model  # noqa: E402

log = logging.getLogger("signal_generator")


def load_universe(name: str, universe_file: str = None):
    """Load universe by name OR from a selector JSON file.

    universe_file: path to selector JSON {"stocks":[{"symbol":...,"name":...}]}
    Takes precedence over name.
    """
    if universe_file:
        with open(universe_file) as f:
            data = json.load(f)
        return [(s["symbol"], s.get("name", s["symbol"])) for s in data["stocks"]]
    if name == "nifty50":
        return NIFTY50_SYMBOLS
    if name == "nifty500":
        return nifty500_symbols()
    raise ValueError(f"Unknown universe: {name}")


def get_window_days(model_key: str) -> int:
    """How many days of bars the strategy needs (window + warmup)."""
    info = get_model(model_key)
    return info.get("default_window_days", 365) + 400


def load_bars(symbol: str, interval: str, days: int):
    """Load cached bars and convert to SimpleNamespace candles."""
    from datetime import timedelta
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    df = read_cached(symbol, interval, int(start_dt.timestamp()), int(end_dt.timestamp()))
    if df.empty:
        return []
    return [
        SimpleNamespace(
            timestamp=int(r.timestamp),
            candle_time=r.candle_time,
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=int(getattr(r, "volume", 0) or 0),
        )
        for r in df.itertuples()
    ]


def generate_signals(model_key: str, universe: str,
                      universe_file: str = None) -> List[Dict]:
    info = get_model(model_key)
    StratCls = info["strategy_class"]
    Cfg = info["config_class"]
    interval = info["bars_interval"]
    days = get_window_days(model_key)

    strat = StratCls(Cfg())
    symbols = load_universe(universe, universe_file)

    today = datetime.now().date()
    today_ts = int(datetime.combine(today, datetime.min.time()).timestamp())

    out: List[Dict] = []
    for sym, name in symbols:
        candles = load_bars(sym, interval, days)
        if not candles:
            continue
        try:
            signals = strat.evaluate(user_id=1, symbol=sym, candles=candles,
                                      eval_from_ts=today_ts)
        except Exception as e:
            log.warning(f"{sym}: strategy fail {e}")
            continue
        # Keep only ENTRY/PARTIAL/EXIT signals for today
        for s in signals:
            if s["signal_type"] not in ("ENTRY1", "ENTRY2", "PARTIAL", "TARGET_HIT", "STOP_HIT"):
                continue
            out.append({
                "model": model_key,
                "universe": universe,
                "symbol": sym,
                "company": name,
                "ts": str(s["candle_time"]),
                "side": s["trend"],
                "signal": s["signal_type"],
                "price": float(s["price"]),
                "sl": float(s.get("sl", 0)),
                "target": float(s.get("target", 0)),
                "note": s.get("note", ""),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model key: ema_200_400, ema_9_21, swing_pullback, orb_15min")
    ap.add_argument("--universe", default="nifty50",
                    choices=["nifty50", "nifty500"])
    ap.add_argument("--universe-file", default=None,
                    help="Path to selector JSON (overrides --universe)")
    ap.add_argument("--signals-out", default=None,
                    help="JSON output path. Default: signals/{date}_{model}_{universe}.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    uni_label = args.universe_file or args.universe
    log.info(f"Generating signals: model={args.model} universe={uni_label}")
    signals = generate_signals(args.model, args.universe, args.universe_file)
    log.info(f"Emitted {len(signals)} signals")

    out_path = args.signals_out
    if out_path is None:
        out_dir = ROOT / "signals"
        out_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = out_dir / f"{date_str}_{args.model}_{args.universe}.json"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
