"""Realistic capital sim v2 — adds risk overlays.

Overlays (all toggleable):
- DD throttle: at -3% account DD, halve slot allocation. At -5%, pause new entries
- Vol-sizing: slot alloc = risk_per_trade / (ATR proxy from entry-to-stop gap)
  (Only when entry has a STOP event lined up; falls back to equal-share)
- R:R floor: skip ENTRY if implied R:R < threshold (using next STOP / TARGET in event list)
- Consecutive-loss pause: after N losses, pause M bars
- Time-stop: close position if not profitable in N events

Usage:
  python tools/backtests/realistic_capital_sim_v2.py <case_dir> \
    --capital 1000000 --dd-throttle --vol-sizing --rr-floor 1.5
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from realistic_capital_sim import (  # noqa: E402
    Event, Position, collect_events, _kind_rank
)


@dataclass
class RiskOverlayConfig:
    dd_throttle: bool = False      # halve at -3%, pause at -5%
    vol_sizing: bool = False       # risk-based sizing using stop distance
    rr_floor: float = 0.0          # min R:R required pre-entry (0 = off)
    consecutive_loss_pause: int = 0  # after N losses, pause M trades
    consecutive_loss_window: int = 0  # M trades to skip
    risk_per_trade_pct: float = 1.0   # 1% of capital risked per trade in vol-sizing
    time_stop_events: int = 0      # close if no profit in N events (0 = off)


def find_stop_for_entry(events: List[Event], entry_idx: int) -> Optional[Event]:
    """Find next STOP for the same symbol after this entry index."""
    sym = events[entry_idx].symbol
    for j in range(entry_idx + 1, len(events)):
        if events[j].symbol == sym and events[j].kind == "STOP":
            return events[j]
        if events[j].symbol == sym and events[j].kind == "TARGET":
            return events[j]   # closed before stop, no R info
    return None


def find_target_for_entry(events: List[Event], entry_idx: int) -> Optional[Event]:
    sym = events[entry_idx].symbol
    for j in range(entry_idx + 1, len(events)):
        if events[j].symbol == sym and events[j].kind == "TARGET":
            return events[j]
        if events[j].symbol == sym and events[j].kind == "STOP":
            return None
    return None


def simulate(case_dir: str, max_concurrent: int, capital: int,
             cfg: RiskOverlayConfig) -> Dict:
    events = collect_events(case_dir)
    cash = float(capital)
    open_positions: Dict[str, List[Position]] = {}
    cur_open = 0
    realized = 0.0
    skipped_entries = 0
    taken_entries = 0
    closed_legs = 0
    losses_in_a_row = 0
    pause_counter = 0
    max_dd = 0.0
    peak = float(capital)
    last_price: Dict[str, float] = {}
    win_count = 0
    loss_count = 0

    for i, ev in enumerate(events):
        last_price[ev.symbol] = ev.price

        # mark-to-equity
        equity = cash
        for sym, poslist in open_positions.items():
            mark = last_price.get(sym, poslist[0].entry_price)
            for p in poslist:
                equity += p.qty_remaining * mark
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        # DD throttle multiplier
        size_mult = 1.0
        if cfg.dd_throttle:
            if dd >= 0.05:
                size_mult = 0.0   # pause
            elif dd >= 0.03:
                size_mult = 0.5   # half size

        # Consecutive-loss pause
        if pause_counter > 0:
            pause_counter -= 1
            if ev.kind == "ENTRY":
                skipped_entries += 1
                continue

        if ev.kind == "ENTRY":
            if size_mult == 0:
                skipped_entries += 1
                continue
            slots_left = max_concurrent - cur_open
            if slots_left <= 0:
                skipped_entries += 1
                continue

            # R:R floor check
            if cfg.rr_floor > 0:
                stop_ev = find_stop_for_entry(events, i)
                target_ev = find_target_for_entry(events, i)
                if stop_ev and target_ev:
                    risk = abs(ev.price - stop_ev.price)
                    reward = abs(target_ev.price - ev.price)
                    rr = reward / risk if risk > 0 else 0
                    if rr < cfg.rr_floor:
                        skipped_entries += 1
                        continue

            # Vol-based sizing
            if cfg.vol_sizing:
                stop_ev = find_stop_for_entry(events, i)
                if stop_ev:
                    risk_per_share = abs(ev.price - stop_ev.price)
                    risk_capital = capital * (cfg.risk_per_trade_pct / 100.0) * size_mult
                    if risk_per_share > 0:
                        shares = int(risk_capital / risk_per_share)
                    else:
                        shares = 0
                    # Cap at slot alloc (don't exceed equal share)
                    cap = int((cash / slots_left) // ev.price)
                    shares = min(shares, cap)
                else:
                    slot_alloc = (cash / slots_left) * size_mult
                    shares = int(slot_alloc // ev.price)
            else:
                slot_alloc = (cash / slots_left) * size_mult
                shares = int(slot_alloc // ev.price)

            if shares < 1:
                skipped_entries += 1
                continue
            cost = shares * ev.price
            cash -= cost
            pos = Position(symbol=ev.symbol, entry_ts=ev.ts, entry_price=ev.price,
                           qty=shares, qty_remaining=shares)
            open_positions.setdefault(ev.symbol, []).append(pos)
            cur_open += 1
            taken_entries += 1

        elif ev.kind == "PARTIAL":
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if not p.partial_done and p.qty_remaining > 0:
                    book_qty = p.qty // 2
                    if book_qty < 1:
                        continue
                    proceeds = book_qty * ev.price
                    cash += proceeds
                    p.qty_remaining -= book_qty
                    p.partial_done = True
                    realized += (ev.price - p.entry_price) * book_qty

        elif ev.kind in ("TARGET", "STOP"):
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if p.qty_remaining > 0:
                    proceeds = p.qty_remaining * ev.price
                    cash += proceeds
                    pnl = (ev.price - p.entry_price) * p.qty_remaining
                    realized += pnl
                    p.qty_remaining = 0
                    closed_legs += 1
                    cur_open -= 1
                    # win/loss tracking
                    if pnl > 0:
                        win_count += 1
                        losses_in_a_row = 0
                    else:
                        loss_count += 1
                        losses_in_a_row += 1
                        if (cfg.consecutive_loss_pause > 0 and
                            losses_in_a_row >= cfg.consecutive_loss_pause):
                            pause_counter = cfg.consecutive_loss_window
                            losses_in_a_row = 0
            open_positions[ev.symbol] = [p for p in poslist if p.qty_remaining > 0]
            if not open_positions[ev.symbol]:
                del open_positions[ev.symbol]

    # mark remaining open at last seen price
    final_equity = cash
    last_price_by_sym: Dict[str, float] = {}
    for ev in events:
        last_price_by_sym[ev.symbol] = ev.price
    for sym, poslist in open_positions.items():
        for p in poslist:
            final_equity += p.qty_remaining * last_price_by_sym.get(sym, p.entry_price)

    return {
        "max_concurrent": max_concurrent,
        "taken": taken_entries,
        "skipped": skipped_entries,
        "closed_legs": closed_legs,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(100 * win_count / max(win_count + loss_count, 1), 1),
        "starting_capital": capital,
        "final_equity": round(final_equity, 2),
        "realized_pnl": round(realized, 2),
        "roi_pct": round((final_equity - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "open_at_end": sum(len(v) for v in open_positions.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--capital", type=int, default=1_000_000)
    ap.add_argument("--max-concurrents", type=int, nargs="+",
                    default=[2, 3, 5, 8])
    ap.add_argument("--dd-throttle", action="store_true")
    ap.add_argument("--vol-sizing", action="store_true")
    ap.add_argument("--rr-floor", type=float, default=0.0)
    ap.add_argument("--consecutive-loss-pause", type=int, default=0)
    ap.add_argument("--consecutive-loss-window", type=int, default=0)
    ap.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    args = ap.parse_args()

    cfg = RiskOverlayConfig(
        dd_throttle=args.dd_throttle,
        vol_sizing=args.vol_sizing,
        rr_floor=args.rr_floor,
        consecutive_loss_pause=args.consecutive_loss_pause,
        consecutive_loss_window=args.consecutive_loss_window,
        risk_per_trade_pct=args.risk_per_trade_pct,
    )

    print(f"Case: {args.case_dir}")
    print(f"Capital: INR {args.capital:,}")
    overlays = []
    if cfg.dd_throttle: overlays.append("DD throttle")
    if cfg.vol_sizing: overlays.append(f"Vol sizing ({cfg.risk_per_trade_pct}% risk)")
    if cfg.rr_floor: overlays.append(f"R:R floor {cfg.rr_floor}")
    if cfg.consecutive_loss_pause: overlays.append(
        f"Loss pause {cfg.consecutive_loss_pause}→{cfg.consecutive_loss_window}")
    print(f"Overlays: {overlays or 'none (vanilla)'}")
    print()
    print(f"{'Max':>4} {'Taken':>7} {'Skip':>7} {'Wins':>5} {'Loss':>5} "
          f"{'Win%':>5} {'Final':>14} {'ROI%':>7} {'MaxDD%':>7} {'OpenEnd':>8}")
    print("-" * 90)
    for cap in args.max_concurrents:
        r = simulate(args.case_dir, cap, args.capital, cfg)
        print(f"{cap:>4} {r['taken']:>7} {r['skipped']:>7} "
              f"{r['wins']:>5} {r['losses']:>5} {r['win_rate']:>5.1f} "
              f"{r['final_equity']:>14,.0f} {r['roi_pct']:>+7.2f} "
              f"{r['max_drawdown_pct']:>7.2f} {r['open_at_end']:>8}")


if __name__ == "__main__":
    main()
