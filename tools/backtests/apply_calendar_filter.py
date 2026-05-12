"""Apply calendar-based filters (expiry-week, budget-day) to per-stock
backtest cycle events.

Blocks ENTRY rows on:
- Last Thursday of each month (monthly F&O expiry) and T-1
- Feb 1 (Union Budget day) and T-1, T+1

Usage:
  python tools/backtests/apply_calendar_filter.py \
    --case /tmp/selector_top10 --out /tmp/selector_top10_calfilter
"""
from __future__ import annotations

import argparse
import calendar
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Set

log = logging.getLogger("apply_calendar")


def last_thursday(year: int, month: int) -> date:
    """Last Thursday of given month."""
    days_in_month = calendar.monthrange(year, month)[1]
    last = date(year, month, days_in_month)
    # Thursday = weekday 3
    offset = (last.weekday() - 3) % 7
    return last - timedelta(days=offset)


def build_blackout_days(start: date, end: date) -> Set[date]:
    blackout: Set[date] = set()
    # Monthly expiries (last Thursday + T-1)
    year = start.year
    while year <= end.year:
        for m in range(1, 13):
            try:
                lt = last_thursday(year, m)
            except Exception:
                continue
            if start <= lt <= end:
                blackout.add(lt)
                blackout.add(lt - timedelta(days=1))
        year += 1
    # Budget day (Feb 1) +/- 1
    for y in range(start.year, end.year + 1):
        bd = date(y, 2, 1)
        if start <= bd <= end:
            blackout.add(bd)
            blackout.add(bd - timedelta(days=1))
            blackout.add(bd + timedelta(days=1))
    return blackout


def apply_filter(case_dir: Path, out_dir: Path, blackout: Set[date]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    blocked = 0
    for f in sorted(case_dir.iterdir()):
        if not f.name.endswith(".md"):
            continue
        if f.name.startswith("_"):
            (out_dir / f.name).write_text(f.read_text())
            continue
        lines_out = []
        in_cycles = False
        in_table = False
        for line in f.read_text().splitlines():
            s = line.strip()
            if s.startswith("## Strategy Cycles"):
                in_cycles = True
                in_table = False
            elif s.startswith("## "):
                in_cycles = False
                in_table = False
            if in_cycles and s.startswith("| Stage"):
                in_table = True
                lines_out.append(line)
                continue
            if in_cycles and in_table and s.startswith("|") and "|---|" not in s:
                cells = [c.strip() for c in s.split("|")[1:-1]]
                if len(cells) >= 2 and ("First Entry" in cells[0] or "Second Entry" in cells[0]):
                    ts_str = cells[1]
                    try:
                        entry_date = datetime.strptime(ts_str[:10], "%Y-%m-%d").date()
                    except ValueError:
                        lines_out.append(line)
                        continue
                    if entry_date in blackout:
                        blocked += 1
                        continue
                    kept += 1
            lines_out.append(line)
        (out_dir / f.name).write_text("\n".join(lines_out))
    log.info(f"Calendar filter: kept {kept} entries, blocked {blocked}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2025-05-12")
    ap.add_argument("--end", default="2026-05-12")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    blackout = build_blackout_days(start, end)
    log.info(f"Blackout days: {len(blackout)} ({start} to {end})")
    apply_filter(Path(args.case), Path(args.out), blackout)


if __name__ == "__main__":
    main()
