"""Momentum Rotation SHORT-side with risk-overlay variants.

Modes:
  short_baseline      : bottom-N by 60d return, monthly rebalance only (no overlay)
  short_trail_8       : same, but exit on daily close >= entry_low * 1.08
                        (entry_low = min(low) since entry; rally stop)
  short_trail_5       : tighter trail at 5%
  short_negative_only : only short symbols with 60d return < -5%
  short_quick_cover   : exit short after 7 trading days regardless of price

For shorts, P&L = entry - exit (gain when price falls). Cap-sim uses
cash-secured collateral = shares * entry_price.

Outputs per-symbol cycle .md files with SHORT__ prefix so the existing
parse_per_symbol / simulate functions in momrot_long_short_backtest.py
can score them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.momrot_long_short_backtest import (  # noqa: E402
    rank_momentum, generate_rebalance_dates, get_close_at, short_key,
    simulate,
)

log = logging.getLogger("momrot_short_f")


# ---------------------------------------------------------------
# Daily-bar helpers for trail / quick-cover (need daily iteration)
# ---------------------------------------------------------------

def daily_bars_between(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[(df["timestamp"] > start_ts) & (df["timestamp"] <= end_ts)].copy()


def find_trail_exit(df: pd.DataFrame, entry_ts: int, end_ts: int,
                     entry_price: float, trail_pct: float) -> Optional[Dict]:
    """Walk daily bars from entry_ts to end_ts.

    For a SHORT trailing stop:
      track running_low = min(low) since entry (low for shorts is favorable).
      Exit when today's CLOSE >= running_low * (1 + trail_pct/100).

    Returns dict {ts, price, low_at_exit} or None if no trigger by end_ts.
    """
    if df is None or df.empty:
        return None
    seg = df[(df["timestamp"] > entry_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty:
        return None
    running_low = entry_price
    for _, row in seg.iterrows():
        lo = float(row["low"])
        cl = float(row["close"])
        if lo < running_low:
            running_low = lo
        trigger = running_low * (1.0 + trail_pct / 100.0)
        if cl >= trigger:
            return {
                "ts": int(row["timestamp"]),
                "price": cl,
                "low_at_exit": running_low,
            }
    return None


def find_quick_cover_exit(df: pd.DataFrame, entry_ts: int, end_ts: int,
                            n_bars: int) -> Optional[Dict]:
    """Exit after n_bars trading days. Use close of the Nth bar after entry."""
    if df is None or df.empty:
        return None
    seg = df[(df["timestamp"] > entry_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty:
        return None
    if len(seg) < n_bars:
        # universe ends; take last available
        row = seg.iloc[-1]
    else:
        row = seg.iloc[n_bars - 1]
    return {"ts": int(row["timestamp"]), "price": float(row["close"])}


# ---------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------

def run_filtered_short(universe: List[str], start: str, end: str,
                        mode: str, top_n: int,
                        out_dir: Path) -> Dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"[{mode}] loading daily bars for {len(universe)} symbols")
    daily_data: Dict[str, pd.DataFrame] = {}
    for sym in universe:
        df = read_cached(sym, "D",
                          int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    rebalance_dates = generate_rebalance_dates(start_dt, end_dt, "monthly")
    log.info(f"[{mode}] monthly rebalance dates: {len(rebalance_dates)}")

    # Mode parameters
    use_trail = mode in ("short_trail_8", "short_trail_5")
    trail_pct = 8.0 if mode == "short_trail_8" else (5.0 if mode == "short_trail_5" else None)
    use_quick_cover = mode == "short_quick_cover"
    use_neg_only = mode == "short_negative_only"
    neg_threshold = -5.0

    events: Dict[str, List[Dict]] = {}  # sym -> events
    # held_short: sym -> {"entry": price, "entry_ts": ts, "entry_str": str}
    held_short: Dict[str, Dict] = {}

    end_ts = int(end_dt.timestamp())

    for i, reb_dt in enumerate(rebalance_dates):
        reb_ts = int(reb_dt.timestamp())
        reb_str = reb_dt.strftime("%Y-%m-%d %H:%M:%S")
        next_reb_ts = (int(rebalance_dates[i + 1].timestamp())
                        if i + 1 < len(rebalance_dates) else end_ts)

        # Rank
        ranks = rank_momentum(universe, reb_ts, daily_data)
        if not ranks:
            continue

        # Bottom-N (worst momentum)
        bottom = list(reversed(ranks))  # ascending by return
        if use_neg_only:
            bottom = [r for r in bottom if r[1] < neg_threshold]
        target_short = bottom[:top_n]
        short_syms = {r[0] for r in target_short}

        # ------------------------------------------------------
        # First, force-close any held shorts that hit overlay
        # exits BEFORE this rebalance. Walk daily bars from
        # last-known entry up to reb_ts and check for trail or
        # quick-cover exit.
        # ------------------------------------------------------
        if use_trail or use_quick_cover:
            for sym in list(held_short.keys()):
                pos = held_short[sym]
                df = daily_data.get(sym)
                exit_event = None
                if use_trail:
                    exit_event = find_trail_exit(
                        df, pos["entry_ts"], reb_ts,
                        pos["entry"], trail_pct,
                    )
                elif use_quick_cover:
                    exit_event = find_quick_cover_exit(
                        df, pos["entry_ts"], reb_ts, 7,
                    )
                if exit_event:
                    exit_ts_str = datetime.fromtimestamp(exit_event["ts"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    exit_p = exit_event["price"]
                    kind = "TARGET" if exit_p < pos["entry"] else "STOP"
                    stage = "Target hit" if kind == "TARGET" else "Stop hit"
                    events.setdefault(sym, []).append({
                        "Stage": stage, "ts": exit_ts_str,
                        "price": exit_p, "kind": kind,
                    })
                    held_short.pop(sym)

        # ------------------------------------------------------
        # Now do normal rebalance: close anyone no longer in
        # target list, open new entrants.
        # ------------------------------------------------------
        for sym in list(held_short.keys()):
            if sym not in short_syms:
                pos = held_short.pop(sym)
                exit_p = get_close_at(sym, reb_ts, daily_data)
                if exit_p > 0:
                    kind = "TARGET" if exit_p < pos["entry"] else "STOP"
                    stage = "Target hit" if kind == "TARGET" else "Stop hit"
                    events.setdefault(sym, []).append({
                        "Stage": stage, "ts": reb_str,
                        "price": exit_p, "kind": kind,
                    })

        for sym, ret, price in target_short:
            if sym in held_short:
                continue
            if price <= 0:
                continue
            held_short[sym] = {
                "entry": price,
                "entry_ts": reb_ts,
                "entry_str": reb_str,
            }
            events.setdefault(sym, []).append({
                "Stage": "First Entry", "ts": reb_str,
                "price": price, "kind": "ENTRY",
            })

    # ----- end-of-period close all remaining -----
    # First check overlay exits up to end_ts
    if use_trail or use_quick_cover:
        for sym in list(held_short.keys()):
            pos = held_short[sym]
            df = daily_data.get(sym)
            exit_event = None
            if use_trail:
                exit_event = find_trail_exit(
                    df, pos["entry_ts"], end_ts, pos["entry"], trail_pct,
                )
            elif use_quick_cover:
                exit_event = find_quick_cover_exit(
                    df, pos["entry_ts"], end_ts, 7,
                )
            if exit_event:
                exit_ts_str = datetime.fromtimestamp(exit_event["ts"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                exit_p = exit_event["price"]
                kind = "TARGET" if exit_p < pos["entry"] else "STOP"
                stage = "Target hit" if kind == "TARGET" else "Stop hit"
                events.setdefault(sym, []).append({
                    "Stage": stage, "ts": exit_ts_str,
                    "price": exit_p, "kind": kind,
                })
                held_short.pop(sym)

    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    for sym, pos in held_short.items():
        exit_p = get_close_at(sym, end_ts, daily_data)
        if exit_p > 0:
            kind = "TARGET" if exit_p < pos["entry"] else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            events.setdefault(sym, []).append({
                "Stage": stage, "ts": end_str,
                "price": exit_p, "kind": kind,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sym, evs in events.items():
        key = short_key(sym)
        fname_base = key.replace(":", "_").replace("-EQ", "")
        lines = [
            f"# {sym}  (SHORT)", "",
            f"## Backtest Summary",
            f"- Strategy: Monthly Momentum Rotation bottom-{top_n} ({mode})",
            f"- Side: SHORT",
            f"- Entries: {sum(1 for e in evs if e['kind'] == 'ENTRY')}",
            "",
            "## Strategy Cycles", "",
            "| Stage | Timestamp | Price | sl | target | note |",
            "|---|---|---|---|---|---|",
        ]
        for e in evs:
            lines.append(
                f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |"
            )
        (out_dir / f"{fname_base}.md").write_text("\n".join(lines))
        written += 1

    meta = {
        "mode": mode,
        "top_n": top_n,
        "start": start, "end": end,
        "rebalance_dates": len(rebalance_dates),
        "short_symbol_count": len(events),
        "files_written": written,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"[{mode}] wrote {written} files; meta={meta}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest")
    bt.add_argument("--universe-file", required=True)
    bt.add_argument("--from", dest="date_from", required=True)
    bt.add_argument("--to", dest="date_to", required=True)
    bt.add_argument("--mode", required=True,
                    choices=["short_baseline", "short_trail_8", "short_trail_5",
                             "short_negative_only", "short_quick_cover"])
    bt.add_argument("--top-n", type=int, default=5)
    bt.add_argument("--out", required=True)

    sim = sub.add_parser("capsim")
    sim.add_argument("--case-dir", required=True)
    sim.add_argument("--capital", type=int, default=1_000_000)
    sim.add_argument("--max-concurrent", type=int, default=5)
    sim.add_argument("--out-json", default=None)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "backtest":
        with open(args.universe_file) as f:
            data = json.load(f)
        symbols = [s["symbol"] for s in data["stocks"]]
        log.info(f"Universe: {len(symbols)} symbols, mode={args.mode}")
        run_filtered_short(symbols, args.date_from, args.date_to,
                            args.mode, args.top_n, Path(args.out))
    elif args.cmd == "capsim":
        r = simulate(args.case_dir, args.capital,
                      args.max_concurrent,
                      has_long=False, has_short=True)
        print(json.dumps(r, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
