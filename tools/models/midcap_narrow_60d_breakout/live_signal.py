"""midcap_narrow_60d_breakout — daily live signal generator.

Reads midcap_narrow universe + historical_data, decides:
  - If model holds a position: check exit conditions (target/trail/SMA/max_hold)
  - Else: scan for fresh 60-day high + vol>2x + close>200d SMA, pick highest
    vol_ratio candidate as ENTRY1.

Emits signals JSON consumed by tools/live/fyers_executor.py --model-name.

Usage:
  python tools/models/midcap_narrow_60d_breakout/live_signal.py \
    --universe-file /app/logs/momrot/universes/midcap_narrow.json \
    --signals-out /app/logs/midcap_narrow/signals/$(date +%F).json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402

log = logging.getLogger("midcap_breakout_signal")

MODEL_NAME = "midcap_narrow_60d_breakout"

# Strategy params (must match backtest.py)
HH_WINDOW = 60
VOL_MULT = 2.0
SMA_LONG = 200
SMA_EXIT = 20
TRAIL_PCT = 0.15
PROFIT_TRIGGER = 0.10
TARGET_PCT = 0.60
MAX_HOLD_DAYS = 30


def load_universe(uf: str) -> List[str]:
    d = json.load(open(uf))
    return [
        f"NSE:{s['symbol']}-EQ" if not s["symbol"].startswith("NSE:") else s["symbol"]
        for s in d.get("stocks", [])
    ]


def load_daily(symbols: List[str], days_back: int = 90) -> pd.DataFrame:
    eng = _get_engine()
    end = datetime.now().date()
    start = end - timedelta(days=days_back + 60)
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


def get_current_position() -> Optional[Dict]:
    """Read model's open position from model_ledger via service."""
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        if l and l.get("open_symbol"):
            return l
    except Exception as e:
        log.warning(f"ledger read failed: {e}")
    return None


def check_exit(pos: Dict, df_sym: pd.DataFrame) -> Optional[Dict]:
    """Return exit-signal dict if any exit condition fires for this position."""
    if df_sym.empty:
        return None
    last = df_sym.iloc[-1]
    close = float(last["close"])
    entry_price = float(pos["open_entry_px"])
    entry_date = datetime.strptime(pos["open_entry_date"], "%Y-%m-%d").date()
    age = (datetime.now().date() - entry_date).days

    ret_entry = (close - entry_price) / entry_price

    # SMA20 for SMA-exit
    df_sym = df_sym.copy()
    df_sym["sma20"] = df_sym["close"].rolling(SMA_EXIT).mean()
    sma20 = (
        float(df_sym.iloc[-1]["sma20"])
        if pd.notna(df_sym.iloc[-1]["sma20"]) else 0.0
    )

    # Track peak by scanning closes since entry
    since_entry = df_sym[df_sym["date"] >= pd.Timestamp(entry_date)]
    peak = float(since_entry["close"].max()) if not since_entry.empty else close
    ret_peak = (peak - close) / peak if peak > 0 else 0

    if ret_entry >= TARGET_PCT:
        return {"reason": "TARGET", "price": close, "ret_pct": ret_entry * 100}
    if ret_entry >= PROFIT_TRIGGER and ret_peak >= TRAIL_PCT:
        return {"reason": "TRAIL", "price": close, "ret_pct": ret_entry * 100}
    if sma20 > 0 and close < sma20:
        return {"reason": "SMA", "price": close, "ret_pct": ret_entry * 100}
    if age >= MAX_HOLD_DAYS:
        return {"reason": "MAX_HOLD", "price": close, "ret_pct": ret_entry * 100}
    return None


def scan_entry_candidate(df: pd.DataFrame, symbols: List[str]) -> Optional[Dict]:
    """Find best fresh 60d-high breakout with vol surge today."""
    today = df["date"].max()
    cands = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < SMA_LONG + 5:
            continue
        g["sma_long"] = g["close"].rolling(SMA_LONG).mean()
        g["hh"] = g["high"].rolling(HH_WINDOW).max().shift(1)
        g["vol_avg20"] = g["volume"].rolling(20).mean()
        row = g[g["date"] == today]
        if row.empty:
            continue
        r = row.iloc[0]
        if any(pd.isna(r[k]) for k in ["sma_long", "hh", "vol_avg20", "close", "volume"]):
            continue
        close = float(r["close"])
        if close <= float(r["hh"]) or close <= float(r["sma_long"]):
            continue
        if float(r["volume"]) < VOL_MULT * float(r["vol_avg20"]):
            continue
        cands.append({
            "symbol": sym,
            "close": close,
            "vol_ratio": float(r["volume"]) / float(r["vol_avg20"]),
            "high_60d_prev": float(r["hh"]),
        })
    if not cands:
        return None
    cands.sort(key=lambda c: -c["vol_ratio"])
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--signals-out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    syms = load_universe(args.universe_file)
    log.info(f"Universe: {len(syms)} midcap_narrow symbols")
    df = load_daily(syms, days_back=SMA_LONG + 30)
    if df.empty:
        log.error("No historical data — cannot emit signals")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        return 1

    signals = []
    pos = get_current_position()

    if pos:
        log.info(f"Open position: {pos['open_symbol']} qty={pos['open_qty']} "
                 f"entry={pos['open_entry_px']} date={pos['open_entry_date']}")
        df_sym = df[df["symbol"] == pos["open_symbol"]].sort_values("date").reset_index(drop=True)
        exit_sig = check_exit(pos, df_sym)
        if exit_sig:
            log.info(f"EXIT signal: {exit_sig['reason']} @ ₹{exit_sig['price']:.2f} "
                     f"({exit_sig['ret_pct']:+.2f}%)")
            signals.append({
                "signal": "STOP_HIT" if exit_sig["reason"] != "TARGET" else "TARGET_HIT",
                "symbol": pos["open_symbol"],
                "side": "SELL",
                "price": exit_sig["price"],
                "reason": exit_sig["reason"],
                "model": MODEL_NAME,
            })
        else:
            log.info("Holding, no exit signal today")
    else:
        log.info("Flat, scanning for entries")
        cand = scan_entry_candidate(df, syms)
        if cand:
            log.info(f"ENTRY candidate: {cand['symbol']} close=₹{cand['close']:.2f} "
                     f"vol_ratio={cand['vol_ratio']:.2f}")
            signals.append({
                "signal": "ENTRY1",
                "symbol": cand["symbol"],
                "side": "BUY",
                "price": cand["close"],
                "model": MODEL_NAME,
            })
        else:
            log.info("No qualifying breakout candidate today")

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.signals_out).write_text(json.dumps(signals, indent=2, default=str))
    log.info(f"Wrote {len(signals)} signals -> {args.signals_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
