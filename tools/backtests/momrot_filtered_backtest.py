"""Momentum Rotation backtest with optional filter overlays.

Long-only baseline = monthly top-N rotation (no SL/target).
This variant adds 4 filter modes on top of the baseline rotation:

  --filter none           (default — identical to monthly momentum_rotation_backtest)
  --filter trailing_sl    once a held position drops X% from its peak-since-entry,
                          force exit (regardless of rank). X via --trailing-pct.
  --filter partial_tp     sell 50% qty when position reaches +30% from entry;
                          tighten trailing SL on the remaining 50% to entry price.
  --filter rotation_speed if rank-1 stalls (top stock has < +5% over last 30d
                          and stays rank-1 for 2+ months), force rotate to rank-2.
  --filter drawdown_circuit
                          pause new entries when portfolio NAV > 15% below
                          all-time peak. Resume when NAV within 8% of peak.

All filters preserve cap-sim-compatible per-symbol .md output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS, nifty500_symbols  # noqa: E402


log = logging.getLogger("momrot_filt")


# ---------- helpers ----------

def get_close_at(symbol: str, target_ts: int, daily_data: Dict[str, pd.DataFrame]) -> float:
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    valid = df[df["timestamp"] <= target_ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def get_high_max_between(symbol: str, start_ts: int, end_ts: int,
                          daily_data: Dict[str, pd.DataFrame]) -> float:
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if sub.empty:
        return 0.0
    return float(sub["high"].astype(float).max())


def iter_trading_days_between(symbol: str, start_ts: int, end_ts: int,
                               daily_data: Dict[str, pd.DataFrame]):
    """Yield (ts, high, low, close) per trading day in the window."""
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return
    sub = df[(df["timestamp"] > start_ts) & (df["timestamp"] <= end_ts)]
    for _, row in sub.iterrows():
        yield (int(row["timestamp"]), float(row["high"]), float(row["low"]),
               float(row["close"]))


def rank_momentum(symbols: List[str], rebalance_ts: int,
                   daily_data: Dict[str, pd.DataFrame],
                   lookback_days: int = 60) -> List[tuple]:
    lookback_ts = rebalance_ts - lookback_days * 86400
    ranks = []
    for sym in symbols:
        c_now = get_close_at(sym, rebalance_ts, daily_data)
        c_60d = get_close_at(sym, lookback_ts, daily_data)
        if c_now > 0 and c_60d > 0:
            ret = (c_now / c_60d - 1) * 100
            ranks.append((sym, ret, c_now))
    ranks.sort(key=lambda x: -x[1])
    return ranks


def short_name(sym: str) -> str:
    return sym.split(":")[-1].replace("-EQ", "").lower()


# ---------- core loop ----------

def run(universe: List[str], start: str, end: str,
         top_n: int, filt: str,
         trailing_pct: float,
         out_dir: Path) -> Dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"Loading daily bars for {len(universe)} symbols")
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")
        df = read_cached(sym, "D",
                          int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    # Monthly rebalance dates (we still walk daily for intramonth filters)
    reb_dates: List[datetime] = []
    cur = start_dt
    while cur < end_dt:
        reb_dates.append(cur)
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)

    per_symbol_events: Dict[str, List[Dict]] = {sym: [] for sym in universe}

    # held[sym] -> dict(entry_price, entry_ts, peak_since_entry, partial_done, sl_tightened)
    held: Dict[str, Dict] = {}

    # rotation_speed bookkeeping
    prev_rank1: Optional[str] = None
    rank1_streak: int = 0

    # drawdown_circuit bookkeeping
    nav_peak: float = 1.0          # geometric NAV proxy (assumes equal-weight)
    nav: float = 1.0
    entries_paused: bool = False

    forced_exit_log: Dict[str, int] = {
        "trailing": 0, "partial": 0, "stale_rotate": 0, "circuit_skip": 0,
        "tightened_after_partial": 0,
    }

    for i, reb_dt in enumerate(reb_dates):
        next_reb = reb_dates[i + 1] if i + 1 < len(reb_dates) else end_dt
        reb_ts = int(reb_dt.timestamp())
        next_reb_ts = int(next_reb.timestamp())

        ranks = rank_momentum(universe, reb_ts, daily_data)
        chosen = ranks[:top_n]

        # --- rotation_speed filter: drop stalled rank-1, replace with rank-2 ---
        if filt == "rotation_speed" and ranks:
            r1_sym, r1_ret, r1_price = ranks[0]
            if prev_rank1 == r1_sym:
                rank1_streak += 1
            else:
                rank1_streak = 1
                prev_rank1 = r1_sym
            # check 30d gain of rank-1
            r1_close_30d_ago = get_close_at(r1_sym, reb_ts - 30 * 86400, daily_data)
            r1_30d_ret = ((r1_price / r1_close_30d_ago) - 1.0) * 100 if r1_close_30d_ago > 0 else 0.0
            if rank1_streak >= 2 and r1_30d_ret < 5.0 and len(ranks) > top_n:
                forced_exit_log["stale_rotate"] += 1
                # Drop rank-1, shift everyone up, use ranks[1:top_n+1]
                chosen = ranks[1:top_n + 1]

        top_syms = {r[0] for r in chosen}

        # --- intramonth daily walk: trailing_sl / partial_tp ---
        if filt in ("trailing_sl", "partial_tp") and held:
            for sym in list(held.keys()):
                hold = held[sym]
                # already exited intramonth?
                if hold.get("exited"):
                    continue
                for (ts, high, low, close) in iter_trading_days_between(
                        sym, reb_ts, next_reb_ts, daily_data):
                    # Track peak
                    if high > hold["peak"]:
                        hold["peak"] = high

                    # partial_tp: +30% from entry, book 50%, tighten SL to entry
                    if filt == "partial_tp" and not hold["partial_done"]:
                        tp_price = hold["entry_price"] * 1.30
                        if high >= tp_price:
                            per_symbol_events[sym].append({
                                "Stage": "Partial book", "ts": _ts_str(ts),
                                "price": tp_price, "kind": "PARTIAL",
                            })
                            hold["partial_done"] = True
                            hold["sl_floor"] = hold["entry_price"]
                            forced_exit_log["partial"] += 1
                            forced_exit_log["tightened_after_partial"] += 1

                    # trailing_sl: if drop X% from peak, exit
                    triggered = False
                    exit_price = None
                    if filt == "trailing_sl":
                        sl_price = hold["peak"] * (1 - trailing_pct / 100.0)
                        if low <= sl_price:
                            triggered = True
                            # assume slippage to SL price
                            exit_price = sl_price
                    elif filt == "partial_tp" and hold["partial_done"]:
                        # Tightened SL on remaining = entry price
                        if low <= hold["sl_floor"]:
                            triggered = True
                            exit_price = hold["sl_floor"]

                    if triggered and exit_price is not None:
                        kind = "TARGET" if exit_price > hold["entry_price"] else "STOP"
                        stage = "Target hit" if kind == "TARGET" else "Stop hit"
                        per_symbol_events[sym].append({
                            "Stage": stage, "ts": _ts_str(ts),
                            "price": exit_price, "kind": kind,
                        })
                        hold["exited"] = True
                        forced_exit_log["trailing"] += 1
                        break  # exit loop, move to next symbol

        # --- drawdown_circuit: update NAV using intermonth returns of held ---
        if filt == "drawdown_circuit":
            # Estimate NAV change since last rebalance: avg return of currently
            # held positions over the month (equal weight).
            if held:
                rets = []
                for sym, hold in held.items():
                    if hold.get("exited"):
                        continue
                    p_now = get_close_at(sym, reb_ts, daily_data)
                    p_prev = hold.get("last_mark", hold["entry_price"])
                    if p_prev > 0:
                        rets.append((p_now / p_prev) - 1.0)
                    hold["last_mark"] = p_now if p_now > 0 else hold.get("last_mark", hold["entry_price"])
                if rets:
                    avg_ret = sum(rets) / len(rets)
                    nav *= (1.0 + avg_ret)
            nav_peak = max(nav_peak, nav)
            dd = (nav_peak - nav) / nav_peak if nav_peak > 0 else 0.0
            if dd >= 0.15:
                entries_paused = True
            elif dd <= 0.08:
                entries_paused = False

        # --- normal rotation at rebalance: sell those out of top, buy new ---
        # First clear positions already exited via filter
        held_active = {s: h for s, h in held.items() if not h.get("exited")}

        to_sell = [s for s in held_active if s not in top_syms]
        for sym in to_sell:
            hold = held_active[sym]
            exit_price = get_close_at(sym, reb_ts, daily_data)
            if exit_price > 0:
                kind = "TARGET" if exit_price > hold["entry_price"] else "STOP"
                stage = "Target hit" if kind == "TARGET" else "Stop hit"
                per_symbol_events[sym].append({
                    "Stage": stage,
                    "ts": reb_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": exit_price, "kind": kind,
                })
            held.pop(sym, None)
        # drop exited intramonth from held
        for sym in list(held.keys()):
            if held[sym].get("exited"):
                held.pop(sym, None)

        # Buy new entrants
        for sym, ret, price in chosen:
            if sym in held:
                continue
            if filt == "drawdown_circuit" and entries_paused:
                forced_exit_log["circuit_skip"] += 1
                continue
            if price <= 0:
                continue
            held[sym] = {
                "entry_price": price,
                "entry_ts": reb_ts,
                "peak": price,
                "partial_done": False,
                "sl_floor": 0.0,
                "exited": False,
                "last_mark": price,
            }
            per_symbol_events[sym].append({
                "Stage": "First Entry",
                "ts": reb_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "price": price, "kind": "ENTRY",
            })

    # Close at end
    end_ts = int(end_dt.timestamp())
    for sym, hold in list(held.items()):
        if hold.get("exited"):
            continue
        exit_price = get_close_at(sym, end_ts, daily_data)
        if exit_price > 0:
            kind = "TARGET" if exit_price > hold["entry_price"] else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            per_symbol_events[sym].append({
                "Stage": stage,
                "ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "price": exit_price, "kind": kind,
            })

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym, events in per_symbol_events.items():
            if not events:
                continue
            lines = [
                f"# {sym}", "",
                "## Backtest Summary",
                f"- Strategy: Monthly Momentum Rotation top-{top_n} filter={filt}",
                f"- Entries: {sum(1 for e in events if e['kind'] == 'ENTRY')}",
                "",
                "## Strategy Cycles", "",
                "| Stage | Timestamp | Price | sl | target | note |",
                "|---|---|---|---|---|---|",
            ]
            for e in events:
                lines.append(f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
            (out_dir / f"{short_name(sym)}.md").write_text("\n".join(lines))

    return {
        "filter": filt,
        "rebalance_dates": len(reb_dates),
        "active_symbols": sum(1 for ev in per_symbol_events.values() if ev),
        "forced_actions": forced_exit_log,
    }


def _ts_str(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty50",
                    choices=["nifty50", "nifty500"])
    ap.add_argument("--universe-file", default=None)
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--filter", default="none",
                    choices=["none", "trailing_sl", "partial_tp",
                             "rotation_speed", "drawdown_circuit"])
    ap.add_argument("--trailing-pct", type=float, default=15.0,
                    help="Trailing SL drop %% from peak (trailing_sl filter)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.universe_file:
        with open(args.universe_file) as f:
            data = json.load(f)
        symbols = [s["symbol"] for s in data["stocks"]]
    elif args.universe == "nifty50":
        symbols = [s for s, _ in NIFTY50_SYMBOLS]
    else:
        symbols = [s for s, _ in nifty500_symbols()]

    log.info(f"Universe: {len(symbols)} symbols filter={args.filter}")
    summary = run(symbols, args.date_from, args.date_to,
                   args.top_n, args.filter, args.trailing_pct,
                   Path(args.out))
    log.info(f"Done: {summary}")


if __name__ == "__main__":
    main()
