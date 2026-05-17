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
    """Find best fresh 60d-high breakout with vol surge today.

    Side effects (stashed on the function so the picks UI can show short-lists):
      - `last_candidates`     — stocks that fully qualify (rare on quiet days)
      - `last_near_miss`      — top-5 stocks closest to qualifying (always
                                populated), ranked by proximity-to-breakout
                                score so the UI never shows an empty card.
    """
    today = df["date"].max()
    cands = []
    near_miss = []
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
        hh = float(r["hh"])
        sma_long = float(r["sma_long"])
        vol_avg = float(r["vol_avg20"])
        vol_ratio = float(r["volume"]) / vol_avg if vol_avg > 0 else 0.0
        hh_ratio = close / hh if hh > 0 else 0.0  # >1 = above prior 60d high
        sma_ratio = close / sma_long if sma_long > 0 else 0.0  # >1 = above SMA200
        qualifies = (close > hh and close > sma_long and vol_ratio >= VOL_MULT)
        info = {
            "symbol": sym,
            "close": close,
            "vol_ratio": vol_ratio,
            "high_60d_prev": hh,
            "hh_ratio": hh_ratio,
            "sma_ratio": sma_ratio,
            "qualifies": qualifies,
        }
        if qualifies:
            cands.append(info)
        # Near-miss score = how close the stock is to ALL 3 conditions firing.
        # Higher = closer to a breakout. Below-HH stocks get penalized.
        score = (hh_ratio - 1) * 50 + (sma_ratio - 1) * 20 + (vol_ratio - 1) * 10
        info["near_miss_score"] = score
        near_miss.append(info)
    cands.sort(key=lambda c: -c["vol_ratio"])
    near_miss.sort(key=lambda c: -c["near_miss_score"])
    scan_entry_candidate.last_candidates = cands  # type: ignore[attr-defined]
    scan_entry_candidate.last_near_miss = near_miss[:5]  # type: ignore[attr-defined]
    if not cands:
        return None
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="no-op flag (midcap has no rebalance gate); "
                    "accepted for symmetry with other models")
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

    # Persist top-5 picks for UI. Priority:
    #   1. Held symbol (if any) as rank 1
    #   2. Today's qualifying breakouts (rare — needs all 3 filters firing)
    #   3. Near-miss candidates (always populated) so card is never empty
    today_str = datetime.now().strftime("%Y-%m-%d")
    qualified = getattr(scan_entry_candidate, "last_candidates", []) or []
    near_miss = getattr(scan_entry_candidate, "last_near_miss", []) or []
    top_rows = []
    if pos:
        held_sym = pos["open_symbol"]
        last_close = float(
            df[df["symbol"] == held_sym].sort_values("date").iloc[-1]["close"]
        ) if not df[df["symbol"] == held_sym].empty else float(pos["open_entry_px"])
        ret_held = (last_close / float(pos["open_entry_px"]) - 1) * 100
        top_rows.append({
            "rank": 1,
            "symbol": held_sym,
            "name": held_sym + " (HELD)",
            "ret_30d_pct": round(ret_held, 2),
            "price": round(last_close, 2),
        })
    # Use qualified first; if none, fall back to near-miss with annotation
    pool = qualified if qualified else near_miss
    note = None
    if not qualified:
        note = ("No qualifying breakouts today — showing near-miss candidates "
                "(closest to firing all 3 filters: above 60d high, above 200d "
                "SMA, volume > 2x avg). Breakout model trades infrequently.")
    for i, c in enumerate(pool[: 5 - len(top_rows)], len(top_rows) + 1):
        # ret_30d_pct here = breakout headroom above 60d HH (proxy momentum)
        # Negative = stock still below prior 60d high (near-miss territory)
        headroom = (c["close"] / c["high_60d_prev"] - 1) * 100 \
            if c.get("high_60d_prev") else 0
        row = {
            "rank": i,
            "symbol": c["symbol"],
            "name": c["symbol"] + ("" if c.get("qualifies") else " (near-miss)"),
            "ret_30d_pct": round(headroom, 2),
            "price": round(c["close"], 2),
            "vol_ratio": round(c["vol_ratio"], 2),
        }
        top_rows.append(row)
    ranking_payload = {
        "model": MODEL_NAME,
        "date": today_str,
        "universe_size": len(syms),
        "qualifying_breakouts": len(qualified),
        "top_n": top_rows,
    }
    if note:
        ranking_payload["note"] = note
    ranking_dir = Path("/app/logs/midcap_narrow/ranking")
    ranking_dir.mkdir(parents=True, exist_ok=True)
    (ranking_dir / f"{today_str}.json").write_text(
        json.dumps(ranking_payload, indent=2, default=str)
    )
    log.info(f"Wrote ranking -> {ranking_dir / (today_str + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
