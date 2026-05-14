"""Momentum Rotation Long/Short variants.

Modes:
  long_only   : rank desc by 60d return, hold top-N (current baseline)
  short_only  : rank asc  by 60d return (worst momentum), short bottom-N
  long_short  : long top-N + short bottom-N, 50/50 capital
  turncoat    : when a previously-held long rank-1 drops OUT of top-N,
                short it for next period (anti-momentum on losers)

For shorts, the per-symbol cycle .md uses prefix "SHORT::" on the symbol
so the cap-sim layer can identify side and invert P&L. Long entries are
written normally. The cap-sim wrapper in this file is the final scorer.

Outputs:
  <out_dir>/{symbol|SHORT__symbol}.md   per-symbol cycle markdown
  <out_dir>/_meta.json                    mode, top_n, dates, counts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    NIFTY50_SYMBOLS,
    nifty500_symbols,
)


log = logging.getLogger("momrot_ls")


def get_close_at(symbol: str, target_ts: int,
                  daily_data: Dict[str, pd.DataFrame]) -> float:
    df = daily_data.get(symbol)
    if df is None or df.empty:
        return 0.0
    valid = df[df["timestamp"] <= target_ts]
    if valid.empty:
        return 0.0
    return float(valid.iloc[-1]["close"])


def rank_momentum(symbols: List[str], rebalance_ts: int,
                   daily_data: Dict[str, pd.DataFrame],
                   lookback_days: int = 60) -> List[Tuple[str, float, float]]:
    lookback_ts = rebalance_ts - lookback_days * 86400
    ranks = []
    for sym in symbols:
        c_now = get_close_at(sym, rebalance_ts, daily_data)
        c_60d = get_close_at(sym, lookback_ts, daily_data)
        if c_now > 0 and c_60d > 0:
            ret = (c_now / c_60d - 1) * 100
            ranks.append((sym, ret, c_now))
    ranks.sort(key=lambda x: -x[1])  # desc
    return ranks


def generate_rebalance_dates(start_dt: datetime, end_dt: datetime,
                              frequency: str = "monthly") -> List[datetime]:
    dates: List[datetime] = []
    if frequency == "monthly":
        cur = start_dt
        while cur < end_dt:
            dates.append(cur)
            if cur.month == 12:
                cur = datetime(cur.year + 1, 1, 1)
            else:
                cur = datetime(cur.year, cur.month + 1, 1)
    elif frequency == "weekly":
        cur = start_dt
        dates.append(start_dt)
        if cur.weekday() != 0:
            cur = cur + timedelta(days=(7 - cur.weekday()) % 7)
        while cur < end_dt:
            if cur != start_dt:
                dates.append(cur)
            cur = cur + timedelta(days=7)
    else:
        raise ValueError(f"freq {frequency} not supported here")
    return dates


def short_key(sym: str) -> str:
    """File-system-safe short-side symbol key."""
    return f"SHORT__{sym}"


def run_long_short(universe: List[str], start: str, end: str,
                    mode: str, top_n: int,
                    out_dir: Path) -> Dict:
    """Run a long/short momentum rotation backtest.

    Returns metadata dict. Writes per-symbol cycle .md files to out_dir.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"[{mode}] loading daily bars for {len(universe)} symbols")
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe):
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe)}")
        df = read_cached(sym, "D",
                          int(warmup_dt.timestamp()), int(end_dt.timestamp()))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            daily_data[sym] = df

    rebalance_dates = generate_rebalance_dates(start_dt, end_dt, "monthly")
    log.info(f"[{mode}] monthly rebalance -> {len(rebalance_dates)} dates")

    # Events keyed by (side, sym). Side = L or S
    # Each event list ends with First Entry / Target hit / Stop hit
    events: Dict[Tuple[str, str], List[Dict]] = {}
    held_long: Dict[str, float] = {}   # sym -> entry price
    held_short: Dict[str, float] = {}  # sym -> entry price

    prev_long_top: set = set()

    for reb_dt in rebalance_dates:
        reb_ts = int(reb_dt.timestamp())
        reb_str = reb_dt.strftime("%Y-%m-%d %H:%M:%S")
        ranks = rank_momentum(universe, reb_ts, daily_data)
        if not ranks:
            continue

        # Define long / short selection per mode
        target_long: List[Tuple[str, float, float]] = []
        target_short: List[Tuple[str, float, float]] = []

        if mode == "long_only":
            target_long = ranks[:top_n]
        elif mode == "short_only":
            # Worst momentum (rank ascending) = bottom-N
            target_short = list(reversed(ranks))[:top_n]
        elif mode == "long_short":
            target_long = ranks[:top_n]
            target_short = list(reversed(ranks))[:top_n]
        elif mode == "turncoat":
            target_long = ranks[:top_n]
            # SHORT the symbols that WERE in top-N last period but
            # have now dropped out (anti-momentum on fallen leaders)
            cur_top = {r[0] for r in target_long}
            dropped = prev_long_top - cur_top
            target_short = [
                (sym, 0.0, get_close_at(sym, reb_ts, daily_data))
                for sym in dropped
                if get_close_at(sym, reb_ts, daily_data) > 0
            ][:top_n]
        else:
            raise ValueError(f"unknown mode {mode}")

        long_syms = {r[0] for r in target_long}
        short_syms = {r[0] for r in target_short}

        # --- LONG side: close removed, open entrants
        for sym in list(held_long.keys()):
            if sym not in long_syms:
                entry = held_long.pop(sym)
                exit_p = get_close_at(sym, reb_ts, daily_data)
                if exit_p > 0:
                    kind = "TARGET" if exit_p > entry else "STOP"
                    stage = "Target hit" if kind == "TARGET" else "Stop hit"
                    events.setdefault(("L", sym), []).append({
                        "Stage": stage, "ts": reb_str,
                        "price": exit_p, "kind": kind,
                    })

        for sym, ret, price in target_long:
            if sym not in held_long and price > 0:
                held_long[sym] = price
                events.setdefault(("L", sym), []).append({
                    "Stage": "First Entry", "ts": reb_str,
                    "price": price, "kind": "ENTRY",
                })

        # --- SHORT side: close removed, open entrants
        for sym in list(held_short.keys()):
            if sym not in short_syms:
                entry = held_short.pop(sym)
                exit_p = get_close_at(sym, reb_ts, daily_data)
                if exit_p > 0:
                    # Short P&L sign: profit if exit < entry
                    # We invert: pretend the position rose when price fell.
                    # The cap-sim layer detects SHORT__ prefix and inverts.
                    kind = "TARGET" if exit_p < entry else "STOP"
                    stage = "Target hit" if kind == "TARGET" else "Stop hit"
                    events.setdefault(("S", sym), []).append({
                        "Stage": stage, "ts": reb_str,
                        "price": exit_p, "kind": kind,
                    })

        for sym, ret, price in target_short:
            if sym not in held_short and price > 0:
                held_short[sym] = price
                events.setdefault(("S", sym), []).append({
                    "Stage": "First Entry", "ts": reb_str,
                    "price": price, "kind": "ENTRY",
                })

        prev_long_top = long_syms

    end_ts = int(end_dt.timestamp())
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    for sym, entry in held_long.items():
        exit_p = get_close_at(sym, end_ts, daily_data)
        if exit_p > 0:
            kind = "TARGET" if exit_p > entry else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            events.setdefault(("L", sym), []).append({
                "Stage": stage, "ts": end_str,
                "price": exit_p, "kind": kind,
            })
    for sym, entry in held_short.items():
        exit_p = get_close_at(sym, end_ts, daily_data)
        if exit_p > 0:
            kind = "TARGET" if exit_p < entry else "STOP"
            stage = "Target hit" if kind == "TARGET" else "Stop hit"
            events.setdefault(("S", sym), []).append({
                "Stage": stage, "ts": end_str,
                "price": exit_p, "kind": kind,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (side, sym), evs in events.items():
        # File-system key
        if side == "L":
            key = sym
        else:
            key = short_key(sym)
        short_label = key.split(":")[-1].replace("-EQ", "").lower()
        # use the key as filename (replace ':' from raw symbol)
        fname_base = key.replace(":", "_").replace("-EQ", "")
        lines = [
            f"# {sym}  ({'LONG' if side == 'L' else 'SHORT'})", "",
            f"## Backtest Summary",
            f"- Strategy: Monthly Momentum Rotation top-{top_n} ({mode})",
            f"- Side: {'LONG' if side == 'L' else 'SHORT'}",
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
        "start": start,
        "end": end,
        "rebalance_dates": len(rebalance_dates),
        "long_symbol_count": sum(1 for k in events if k[0] == "L"),
        "short_symbol_count": sum(1 for k in events if k[0] == "S"),
        "files_written": written,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"[{mode}] wrote {written} files; meta={meta}")
    return meta


# ============================================================
# Capital simulator — long/short aware
# ============================================================

import re

_TS_RE = re.compile(
    r"\| ([A-Za-z][A-Za-z0-9 _\(\)]+?) \| "
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| ([\d.]+) \|"
)


def parse_per_symbol(path: str) -> List[Dict]:
    """Parse a per-symbol .md report.

    Returns events list each with: ts, symbol, side ('L' or 'S'), kind, price.
    """
    events: List[Dict] = []
    if not os.path.exists(path):
        return events
    base = os.path.basename(path).replace(".md", "")
    if base.startswith("SHORT__"):
        sym = base[len("SHORT__"):]
        side = "S"
    else:
        sym = base
        side = "L"
    in_cycles = False
    in_table = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Strategy Cycles"):
                in_cycles = True
                continue
            if not in_cycles:
                continue
            if s.startswith("## "):
                break
            if s.startswith("| Stage"):
                in_table = True
                continue
            if in_table and s.startswith("|") and "|---|" not in s:
                cells = [c.strip() for c in s.split("|")[1:-1]]
                if len(cells) < 6:
                    continue
                stage, ts, price = cells[0], cells[1], cells[2]
                try:
                    p = float(price)
                except ValueError:
                    continue
                kind = None
                if "First Entry" in stage:
                    kind = "ENTRY"
                elif "Target hit" in stage:
                    kind = "TARGET"
                elif "Stop hit" in stage:
                    kind = "STOP"
                if kind:
                    events.append({
                        "ts": ts, "symbol": sym, "side": side,
                        "kind": kind, "price": p,
                    })
            elif in_table and not s.startswith("|"):
                in_table = False
    return events


def _kind_rank(k: str) -> int:
    return {"ENTRY": 0, "TARGET": 1, "STOP": 1}.get(k, 9)


def collect_events(case_dir: str) -> List[Dict]:
    out: List[Dict] = []
    for fname in sorted(os.listdir(case_dir)):
        if fname.startswith("_") or not fname.endswith(".md"):
            continue
        out.extend(parse_per_symbol(os.path.join(case_dir, fname)))
    out.sort(key=lambda e: (e["ts"], e["symbol"], _kind_rank(e["kind"])))
    return out


def simulate(case_dir: str, capital: int, max_concurrent_per_side: int,
              has_long: bool, has_short: bool) -> Dict:
    """Replay events. For long_short, each side gets half capital, separate pool.

    For long_only / short_only, one pool with full capital.
    """
    events = collect_events(case_dir)
    if has_long and has_short:
        cash_long = capital / 2.0
        cash_short = capital / 2.0
    elif has_long:
        cash_long = float(capital)
        cash_short = 0.0
    else:
        cash_long = 0.0
        cash_short = float(capital)

    # Position: side, symbol -> {entry_price, shares}
    pos_long: Dict[str, Dict] = {}
    pos_short: Dict[str, Dict] = {}
    cur_open_long = 0
    cur_open_short = 0
    realized = 0.0
    taken = 0
    skipped = 0
    closed_legs = 0
    last_price: Dict[Tuple[str, str], float] = {}
    peak = float(capital)
    max_dd = 0.0

    for ev in events:
        key = (ev["side"], ev["symbol"])
        last_price[key] = ev["price"]

        # Mark-to-market equity
        equity = cash_long + cash_short
        for sym, p in pos_long.items():
            mark = last_price.get(("L", sym), p["entry"])
            equity += p["shares"] * mark
        for sym, p in pos_short.items():
            mark = last_price.get(("S", sym), p["entry"])
            # Short MTM: P&L = (entry - mark) * shares; collateral = entry*shares
            equity += p["shares"] * p["entry"] + p["shares"] * (p["entry"] - mark)
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        side = ev["side"]
        if ev["kind"] == "ENTRY":
            if side == "L":
                slots_left = max_concurrent_per_side - cur_open_long
                if slots_left <= 0:
                    skipped += 1
                    continue
                slot_alloc = cash_long / slots_left
                shares = int(slot_alloc // ev["price"])
                if shares < 1:
                    skipped += 1
                    continue
                cost = shares * ev["price"]
                cash_long -= cost
                pos_long[ev["symbol"]] = {
                    "entry": ev["price"], "shares": shares, "ts": ev["ts"],
                }
                cur_open_long += 1
                taken += 1
            else:
                slots_left = max_concurrent_per_side - cur_open_short
                if slots_left <= 0:
                    skipped += 1
                    continue
                slot_alloc = cash_short / slots_left
                shares = int(slot_alloc // ev["price"])
                if shares < 1:
                    skipped += 1
                    continue
                # Short: cash collateral locked, proceeds credited at exit
                # We model: cash_short -= shares*entry (margin lock)
                # Exit returns shares*(2*entry - exit) = collateral + P&L
                collateral = shares * ev["price"]
                cash_short -= collateral
                pos_short[ev["symbol"]] = {
                    "entry": ev["price"], "shares": shares, "ts": ev["ts"],
                }
                cur_open_short += 1
                taken += 1

        elif ev["kind"] in ("TARGET", "STOP"):
            if side == "L":
                p = pos_long.pop(ev["symbol"], None)
                if p:
                    proceeds = p["shares"] * ev["price"]
                    cash_long += proceeds
                    realized += (ev["price"] - p["entry"]) * p["shares"]
                    cur_open_long -= 1
                    closed_legs += 1
            else:
                p = pos_short.pop(ev["symbol"], None)
                if p:
                    pnl = (p["entry"] - ev["price"]) * p["shares"]
                    # Release collateral + P&L
                    cash_short += p["shares"] * p["entry"] + pnl
                    realized += pnl
                    cur_open_short -= 1
                    closed_legs += 1

    # Close at end at last seen price
    final_equity = cash_long + cash_short
    for sym, p in pos_long.items():
        m = last_price.get(("L", sym), p["entry"])
        final_equity += p["shares"] * m
    for sym, p in pos_short.items():
        m = last_price.get(("S", sym), p["entry"])
        final_equity += p["shares"] * p["entry"] + p["shares"] * (p["entry"] - m)

    return {
        "case_dir": case_dir,
        "events_total": len(events),
        "taken": taken,
        "skipped": skipped,
        "closed_legs": closed_legs,
        "starting_capital": capital,
        "final_equity": round(final_equity, 2),
        "realized_pnl": round(realized, 2),
        "roi_pct": round((final_equity - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "open_long_end": len(pos_long),
        "open_short_end": len(pos_short),
    }


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest")
    bt.add_argument("--universe-file", required=True)
    bt.add_argument("--from", dest="date_from", required=True)
    bt.add_argument("--to", dest="date_to", required=True)
    bt.add_argument("--mode", required=True,
                    choices=["long_only", "short_only",
                             "long_short", "turncoat"])
    bt.add_argument("--top-n", type=int, default=5)
    bt.add_argument("--out", required=True)

    sim = sub.add_parser("capsim")
    sim.add_argument("--case-dir", required=True)
    sim.add_argument("--capital", type=int, default=1_000_000)
    sim.add_argument("--max-concurrent", type=int, default=1,
                     help="Per-side max concurrent positions")
    sim.add_argument("--mode", required=True,
                     choices=["long_only", "short_only",
                              "long_short", "turncoat"])
    sim.add_argument("--out-json", default=None)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "backtest":
        with open(args.universe_file) as f:
            data = json.load(f)
        symbols = [s["symbol"] for s in data["stocks"]]
        log.info(f"Universe: {len(symbols)} symbols, mode={args.mode}")
        run_long_short(symbols, args.date_from, args.date_to,
                        args.mode, args.top_n, Path(args.out))
    elif args.cmd == "capsim":
        has_long = args.mode in ("long_only", "long_short", "turncoat")
        has_short = args.mode in ("short_only", "long_short", "turncoat")
        r = simulate(args.case_dir, args.capital,
                      args.max_concurrent, has_long, has_short)
        print(json.dumps(r, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
