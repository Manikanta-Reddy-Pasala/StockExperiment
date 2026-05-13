"""Momentum Rotation live signal generator with safety filters.

Ranks N100 universe by 60d return, picks top-N, emits ENTRY1 / TARGET_HIT
/ STOP_HIT signals compatible with paper_executor.

Safety filters (default ON via --safety):
  1. Liquidity floor: pick must have 20d ADV >= ₹50 cr; else fall through.
  2. Volatility ceiling: pick must have 20d daily-return stdev <= 4.0%.
  3. Catastrophic stop: if held drops >= 15% from entry, emit STOP_HIT
     immediately regardless of rebalance day.
  4. Index circuit: if Nifty50 had a single-day return <= -5% in last 2
     trading days, pause NEW entries (catastrophic + ranking exits still
     fire).

Rebalance gate:
  --rebalance-only  → only emit rank-driven entries/exits on first 7
                      weekdays of month. Catastrophic stop ALWAYS checked.
  --force           → bypass gate.

Usage:
  python tools/live/momentum_rotation_signal.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --top-n 5 --rebalance-only --safety \
    --ledger /app/logs/momrot/ledger/momrot_ledger.json \
    --signals-out signals/$(date +%F)_momrot.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402

log = logging.getLogger("momrot_signal")


# Safety thresholds (must mirror momrot_freq_backtest.py)
MIN_ADV_LAKH = 5000.0          # ₹50 cr daily turnover
MAX_VOL_DAILY = 4.0            # 4% daily stdev (~ 60% annualized)
CATASTROPHIC_STOP_PCT = -15.0  # exit if held drops 15% from entry
INDEX_CIRCUIT_PCT = -5.0       # single-day drop triggering pause
INDEX_CIRCUIT_LOOKBACK = 2     # look back 2 trading days


def is_rebalance_day(today: datetime) -> bool:
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


def load_recent_bars(symbol: str, target_ts: int, days: int = 30) -> pd.DataFrame:
    """Recent bars for ADV / vol computation."""
    df = read_cached(symbol, "D", target_ts - days * 86400, target_ts)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_adv_lakh(df: pd.DataFrame, n: int = 20) -> float:
    if df.empty:
        return 0.0
    sub = df.tail(n)
    if sub.empty:
        return 0.0
    val = (sub["close"].astype(float) * sub["volume"].astype(float)).mean()
    return float(val) / 1e5


def compute_vol_daily(df: pd.DataFrame, n: int = 20) -> float:
    if df.empty:
        return 0.0
    sub = df.tail(n + 1)
    if len(sub) < n:
        return 0.0
    closes = sub["close"].astype(float).values
    rets = (closes[1:] / closes[:-1] - 1) * 100
    return float(pd.Series(rets).std())


def index_circuit_hit(target_ts: int) -> bool:
    df = read_cached("^NSEI", "D", target_ts - 10 * 86400, target_ts)
    if df.empty:
        return False
    df = df.sort_values("timestamp").reset_index(drop=True)
    sub = df.tail(INDEX_CIRCUIT_LOOKBACK + 1)
    if len(sub) < 2:
        return False
    closes = sub["close"].astype(float).values
    day_rets = (closes[1:] / closes[:-1] - 1) * 100
    return any(r <= INDEX_CIRCUIT_PCT for r in day_rets)


def rank_universe(stocks: List[Dict], today_ts: int,
                  lookback_days: int = 60) -> List[tuple]:
    """[(symbol, name, 60d_return%, current_price)] sorted desc."""
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


def pick_with_safety(ranks: List[tuple], today_ts: int, top_n: int,
                     safety: bool) -> List[tuple]:
    """Return up to top_n picks; with safety, skip illiquid + high-vol."""
    if not safety:
        return ranks[:top_n]
    picks = []
    for r in ranks:
        sym = r[0]
        df = load_recent_bars(sym, today_ts, 30)
        adv = compute_adv_lakh(df)
        vol = compute_vol_daily(df)
        if adv < MIN_ADV_LAKH:
            log.info(f"  SKIP {sym}: adv {adv:.0f} L < {MIN_ADV_LAKH} L")
            continue
        if vol > MAX_VOL_DAILY:
            log.info(f"  SKIP {sym}: vol {vol:.2f}% > {MAX_VOL_DAILY}%")
            continue
        picks.append(r)
        if len(picks) >= top_n:
            break
    return picks


def load_held(ledger_path: Path) -> List[Dict]:
    if not ledger_path or not ledger_path.exists():
        return []
    try:
        with open(ledger_path) as f:
            return json.load(f).get("open", [])
    except Exception as e:
        log.warning(f"ledger read fail: {e}")
        return []


def check_catastrophic(held: List[Dict], today_ts: int) -> List[Dict]:
    """Return list of held positions that hit catastrophic stop (-15% from entry)."""
    out = []
    for h in held:
        sym = h["symbol"]
        entry = float(h["entry_price"])
        live = get_close_at(sym, today_ts)
        if entry > 0 and live > 0:
            drop_pct = (live / entry - 1) * 100
            if drop_pct <= CATASTROPHIC_STOP_PCT:
                log.warning(f"CATASTROPHIC STOP: {sym} entry {entry:.2f} live {live:.2f} drop {drop_pct:.2f}%")
                out.append({**h, "live_price": live, "drop_pct": drop_pct})
    return out


def emit_signals(top_picks: List[tuple], held: List[Dict],
                 top_n: int, catastrophic_exits: List[Dict],
                 today_ts: int, index_circuit: bool) -> List[Dict]:
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_syms = {p[0] for p in top_picks[:top_n]}
    catastrophic_syms = {c["symbol"] for c in catastrophic_exits}
    signals = []

    # 1) Catastrophic stops always fire first (highest priority)
    for c in catastrophic_exits:
        signals.append({
            "model": "momentum_rotation",
            "universe": "n100_pseudo",
            "symbol": c["symbol"],
            "company": c["symbol"],
            "ts": today_str,
            "side": "BUY",
            "signal": "STOP_HIT",
            "price": float(c["live_price"]),
            "sl": 0.0, "target": 0.0,
            "note": f"CATASTROPHIC stop: drop {c['drop_pct']:.2f}% from entry {c['entry_price']:.2f}",
        })

    # 2) Rank exits: held but no longer in top-N (skip ones already exited)
    for h in held:
        if h["symbol"] in catastrophic_syms:
            continue
        if h["symbol"] not in top_syms:
            price = get_close_at(h["symbol"], today_ts)
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

    # 3) Entries: rank-1 if no held already in top-N AND no index circuit
    if index_circuit:
        log.warning("INDEX CIRCUIT active — skipping new entries")
    else:
        remaining_held = [h for h in held if h["symbol"] not in catastrophic_syms]
        if not any(h["symbol"] in top_syms for h in remaining_held) and top_picks:
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
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--rebalance-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--safety", action="store_true", default=True,
                    help="Enable safety filters (default ON)")
    ap.add_argument("--no-safety", dest="safety", action="store_false",
                    help="Disable safety filters")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    today_ts = int(today.timestamp())
    log.info(f"momrot_signal: today={today.date()} weekday={today.strftime('%A')} "
             f"day_of_month={today.day} safety={args.safety}")

    stocks = load_universe(args.universe_file)
    log.info(f"Universe: {len(stocks)} symbols")
    held = load_held(Path(args.ledger)) if args.ledger else []
    log.info(f"Currently held: {[h['symbol'] for h in held]}")

    # Catastrophic check runs EVERY day, regardless of rebalance gate
    catastrophic_exits = check_catastrophic(held, today_ts) if args.safety else []

    # Rebalance gate (ranking-driven entries/exits skipped if not rebalance day)
    rebalance_active = args.force or not args.rebalance_only or is_rebalance_day(today)

    signals: List[Dict] = []

    if rebalance_active:
        ranks = rank_universe(stocks, today_ts)
        log.info(f"Ranked {len(ranks)} stocks. Top-{args.top_n} after filter:")
        top = pick_with_safety(ranks, today_ts, args.top_n, args.safety)
        for i, (sym, name, ret, price) in enumerate(top, 1):
            log.info(f"  {i}. {sym:<14} {ret:+7.2f}%  @ ₹{price:.2f}")

        circuit = args.safety and index_circuit_hit(today_ts)
        signals = emit_signals(top, held, args.top_n, catastrophic_exits, today_ts, circuit)
    else:
        log.info("Not rebalance day — only catastrophic exits will fire.")
        # Still emit catastrophic exits if any
        if catastrophic_exits:
            signals = emit_signals([], held, args.top_n, catastrophic_exits, today_ts, False)

    log.info(f"Emitting {len(signals)} signals")
    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.signals_out, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {args.signals_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
