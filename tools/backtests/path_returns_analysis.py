"""Path returns analysis — produce per-path year-wise + month-wise return tables.

Takes a per-stock cycle dir + cap-sim config, replays trades with the same
slot allocation as realistic_capital_sim_v2, and emits:
  - Monthly P&L table
  - Year-wise summary (single row for single-year backtest;
    multi-row when crossing multiple calendar years)
  - Cumulative equity curve as CSV

Usage:
  python tools/backtests/path_returns_analysis.py \
    --case /tmp/ema921_sec_cal \
    --capital 1000000 --max-concurrent 2 --vol-sizing --risk-per-trade-pct 2.0 \
    --path-name "Path B" --out exports/backtests/PATH_B_RETURNS.md
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from realistic_capital_sim import collect_events, Position  # noqa: E402
from realistic_capital_sim_v2 import (   # noqa: E402
    RiskOverlayConfig, find_stop_for_entry, find_target_for_entry,
)


log = logging.getLogger("path_returns")


def simulate_with_log(case_dir: str, max_concurrent: int, capital: int,
                      cfg: RiskOverlayConfig) -> Dict:
    """Simulate AND return per-month P&L + equity curve."""
    events = collect_events(case_dir)
    cash = float(capital)
    open_positions: Dict[str, List[Position]] = {}
    cur_open = 0
    realized = 0.0
    skipped = 0
    taken = 0
    closed = 0
    losses_in_a_row = 0
    pause_counter = 0
    max_dd = 0.0
    peak = float(capital)
    last_price: Dict[str, float] = {}
    win_count = 0
    loss_count = 0

    # Tracking by month
    monthly_pnl: Dict[str, float] = defaultdict(float)
    monthly_trades: Dict[str, int] = defaultdict(int)
    monthly_wins: Dict[str, int] = defaultdict(int)
    monthly_losses: Dict[str, int] = defaultdict(int)
    equity_curve: List[Tuple[str, float]] = []

    for i, ev in enumerate(events):
        last_price[ev.symbol] = ev.price
        month = ev.ts[:7] if len(ev.ts) >= 7 else "unknown"

        # mark-to-equity
        equity = cash
        for sym, poslist in open_positions.items():
            mark = last_price.get(sym, poslist[0].entry_price)
            for p in poslist:
                equity += p.qty_remaining * mark
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        equity_curve.append((ev.ts, equity))

        size_mult = 1.0
        if cfg.dd_throttle:
            if dd >= 0.05:
                size_mult = 0.0
            elif dd >= 0.03:
                size_mult = 0.5

        if pause_counter > 0:
            pause_counter -= 1
            if ev.kind == "ENTRY":
                skipped += 1
                continue

        if ev.kind == "ENTRY":
            if size_mult == 0:
                skipped += 1
                continue
            slots_left = max_concurrent - cur_open
            if slots_left <= 0:
                skipped += 1
                continue

            if cfg.rr_floor > 0:
                stop_ev = find_stop_for_entry(events, i)
                target_ev = find_target_for_entry(events, i)
                if stop_ev and target_ev:
                    risk = abs(ev.price - stop_ev.price)
                    reward = abs(target_ev.price - ev.price)
                    rr = reward / risk if risk > 0 else 0
                    if rr < cfg.rr_floor:
                        skipped += 1
                        continue

            if cfg.vol_sizing:
                stop_ev = find_stop_for_entry(events, i)
                if stop_ev:
                    risk_per_share = abs(ev.price - stop_ev.price)
                    risk_capital = capital * (cfg.risk_per_trade_pct / 100.0) * size_mult
                    shares = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0
                    cap = int((cash / slots_left) // ev.price)
                    shares = min(shares, cap)
                else:
                    slot_alloc = (cash / slots_left) * size_mult
                    shares = int(slot_alloc // ev.price)
            else:
                slot_alloc = (cash / slots_left) * size_mult
                shares = int(slot_alloc // ev.price)

            if shares < 1:
                skipped += 1
                continue
            cost = shares * ev.price
            cash -= cost
            pos = Position(symbol=ev.symbol, entry_ts=ev.ts,
                           entry_price=ev.price, qty=shares, qty_remaining=shares)
            open_positions.setdefault(ev.symbol, []).append(pos)
            cur_open += 1
            taken += 1

        elif ev.kind == "PARTIAL":
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if not p.partial_done and p.qty_remaining > 0:
                    book_qty = p.qty // 2
                    if book_qty < 1: continue
                    proceeds = book_qty * ev.price
                    cash += proceeds
                    p.qty_remaining -= book_qty
                    p.partial_done = True
                    pnl = (ev.price - p.entry_price) * book_qty
                    realized += pnl
                    monthly_pnl[month] += pnl

        elif ev.kind in ("TARGET", "STOP"):
            poslist = open_positions.get(ev.symbol, [])
            for p in poslist:
                if p.qty_remaining > 0:
                    proceeds = p.qty_remaining * ev.price
                    cash += proceeds
                    pnl = (ev.price - p.entry_price) * p.qty_remaining
                    realized += pnl
                    monthly_pnl[month] += pnl
                    monthly_trades[month] += 1
                    p.qty_remaining = 0
                    closed += 1
                    cur_open -= 1
                    if pnl > 0:
                        win_count += 1
                        monthly_wins[month] += 1
                        losses_in_a_row = 0
                    else:
                        loss_count += 1
                        monthly_losses[month] += 1
                        losses_in_a_row += 1
                        if (cfg.consecutive_loss_pause > 0 and
                            losses_in_a_row >= cfg.consecutive_loss_pause):
                            pause_counter = cfg.consecutive_loss_window
                            losses_in_a_row = 0
            open_positions[ev.symbol] = [p for p in poslist if p.qty_remaining > 0]
            if not open_positions[ev.symbol]:
                del open_positions[ev.symbol]

    # final equity
    final_equity = cash
    last_by_sym: Dict[str, float] = {}
    for ev in events:
        last_by_sym[ev.symbol] = ev.price
    for sym, poslist in open_positions.items():
        for p in poslist:
            final_equity += p.qty_remaining * last_by_sym.get(sym, p.entry_price)

    return {
        "max_concurrent": max_concurrent,
        "capital": capital,
        "taken": taken,
        "skipped": skipped,
        "closed_legs": closed,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(100 * win_count / max(win_count + loss_count, 1), 1),
        "final_equity": round(final_equity, 2),
        "realized_pnl": round(realized, 2),
        "roi_pct": round((final_equity - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "monthly_pnl": dict(monthly_pnl),
        "monthly_trades": dict(monthly_trades),
        "monthly_wins": dict(monthly_wins),
        "monthly_losses": dict(monthly_losses),
        "equity_curve": equity_curve,
    }


def render_md(result: Dict, path_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    capital = result["capital"]
    lines = [
        f"# {path_name} — Returns Analysis\n",
        f"_Generated: {datetime.now().isoformat()}_\n",
        "## Summary\n",
        f"- Starting capital: ₹{capital:,}",
        f"- Final equity: ₹{result['final_equity']:,.0f}",
        f"- Profit: ₹{result['final_equity'] - capital:+,.0f}",
        f"- ROI: {result['roi_pct']:+.2f}%",
        f"- MaxDD: {result['max_drawdown_pct']:.2f}%",
        f"- Trades taken: {result['taken']}",
        f"- Trades closed: {result['closed_legs']}",
        f"- Wins / Losses: {result['wins']} / {result['losses']}",
        f"- Win rate: {result['win_rate']}%",
        f"- Max concurrent: {result['max_concurrent']}",
        "\n## Monthly P&L\n",
        "| Month | Trades | Wins | Losses | Win% | Profit₹ | Cumulative₹ |",
        "|-------|-------:|-----:|-------:|-----:|--------:|------------:|",
    ]
    cumulative = 0.0
    for m in sorted(result["monthly_pnl"].keys()):
        p = result["monthly_pnl"][m]
        t = result["monthly_trades"].get(m, 0)
        w = result["monthly_wins"].get(m, 0)
        l = result["monthly_losses"].get(m, 0)
        wr = (100 * w / t) if t else 0
        cumulative += p
        lines.append(
            f"| {m} | {t} | {w} | {l} | {wr:.0f}% | "
            f"₹{p:+,.0f} | ₹{cumulative:+,.0f} |"
        )

    # Yearly aggregate
    yearly: Dict[str, Dict] = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "losses": 0})
    for m, p in result["monthly_pnl"].items():
        y = m[:4]
        yearly[y]["pnl"] += p
        yearly[y]["trades"] += result["monthly_trades"].get(m, 0)
        yearly[y]["wins"] += result["monthly_wins"].get(m, 0)
        yearly[y]["losses"] += result["monthly_losses"].get(m, 0)
    lines += ["\n## Yearly Summary\n",
              "| Year | Trades | Wins | Losses | Win% | Profit₹ | YoY ROI% |",
              "|------|-------:|-----:|-------:|-----:|--------:|---------:|"]
    for y in sorted(yearly.keys()):
        v = yearly[y]
        wr = (100 * v["wins"] / v["trades"]) if v["trades"] else 0
        yoy = v["pnl"] / capital * 100
        lines.append(f"| {y} | {v['trades']} | {v['wins']} | {v['losses']} | "
                     f"{wr:.0f}% | ₹{v['pnl']:+,.0f} | {yoy:+.2f}% |")

    out_path.write_text("\n".join(lines))
    log.info(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--capital", type=int, default=1_000_000)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--dd-throttle", action="store_true")
    ap.add_argument("--vol-sizing", action="store_true")
    ap.add_argument("--rr-floor", type=float, default=0.0)
    ap.add_argument("--consecutive-loss-pause", type=int, default=0)
    ap.add_argument("--consecutive-loss-window", type=int, default=0)
    ap.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    ap.add_argument("--path-name", default="Custom Path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = RiskOverlayConfig(
        dd_throttle=args.dd_throttle, vol_sizing=args.vol_sizing,
        rr_floor=args.rr_floor,
        consecutive_loss_pause=args.consecutive_loss_pause,
        consecutive_loss_window=args.consecutive_loss_window,
        risk_per_trade_pct=args.risk_per_trade_pct,
    )
    log.info(f"Running {args.path_name}: case={args.case} cap=₹{args.capital:,} max={args.max_concurrent}")
    r = simulate_with_log(args.case, args.max_concurrent, args.capital, cfg)
    log.info(f"ROI: {r['roi_pct']:+.2f}%  DD: {r['max_drawdown_pct']:.2f}%  "
             f"trades: {r['taken']}  win: {r['win_rate']}%")
    render_md(r, args.path_name, Path(args.out))


if __name__ == "__main__":
    main()
