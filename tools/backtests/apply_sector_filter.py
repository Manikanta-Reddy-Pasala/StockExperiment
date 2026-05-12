"""Apply sector RS filter to per-stock backtest cycle events.

Reads SECTOR_RS_*.json + per-stock .md files. For each ENTRY in cycle table,
checks if the stock's sector is in the bottom-N on that date. If so, drops
the entry. Outputs filtered .md files to a new dir.

Usage:
  python tools/backtests/apply_sector_filter.py \
    --case /tmp/selector_top10 \
    --rs /app/exports/backtests/SECTOR_RS_2025-2026.json \
    --out /tmp/selector_top10_sectorfilter
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.sector_rs import SYMBOL_SECTOR  # noqa: E402

log = logging.getLogger("apply_sector_filter")


def load_rs(rs_path: Path) -> Dict[date, Dict]:
    data = json.loads(rs_path.read_text())
    out: Dict[date, Dict] = {}
    for d in data["daily"]:
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()
        out[dt] = d
    return out


def get_sector(symbol: str) -> str:
    s = symbol.upper()
    return SYMBOL_SECTOR.get(s, "UNKNOWN")


def apply_filter(case_dir: Path, rs_by_date: Dict[date, Dict], out_dir: Path,
                  block_bottom: bool = True, require_top: bool = False) -> None:
    """For each per-stock .md, drop ENTRY rows where sector is in bottom-N
    on entry date. Optionally require sector to be in top-N."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_blocked = 0
    per_symbol_blocked: Dict[str, int] = {}

    rs_dates = sorted(rs_by_date.keys())

    def latest_rs(dt: date) -> Dict:
        """Find latest RS record on or before dt."""
        for d in reversed(rs_dates):
            if d <= dt:
                return rs_by_date[d]
        return rs_by_date[rs_dates[0]] if rs_dates else {}

    for f in sorted(case_dir.iterdir()):
        if not f.name.endswith(".md"):
            continue
        sym = f.stem.upper()
        if f.name.startswith("_"):
            (out_dir / f.name).write_text(f.read_text())
            continue
        sector = get_sector(sym)
        if sector == "UNKNOWN":
            log.warning(f"{sym}: unknown sector — keeping all entries")
            (out_dir / f.name).write_text(f.read_text())
            total_kept += 1
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
                    rs = latest_rs(entry_date)
                    in_bottom = sector in rs.get("bottom", [])
                    in_top = sector in rs.get("top", [])
                    block = False
                    if block_bottom and in_bottom:
                        block = True
                    if require_top and not in_top:
                        block = True
                    if block:
                        per_symbol_blocked[sym] = per_symbol_blocked.get(sym, 0) + 1
                        total_blocked += 1
                        continue
                    total_kept += 1
            lines_out.append(line)
        (out_dir / f.name).write_text("\n".join(lines_out))

    log.info(f"Sector filter: kept {total_kept} entries, blocked {total_blocked}")
    if per_symbol_blocked:
        log.info("Blocks per symbol:")
        for sym, n in sorted(per_symbol_blocked.items(), key=lambda x: -x[1]):
            log.info(f"  {sym} ({get_sector(sym)}): blocked {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--rs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-bottom", action="store_true", default=True,
                    help="Block entries where sector is in bottom-N")
    ap.add_argument("--require-top", action="store_true",
                    help="Block entries unless sector is in top-N")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rs = load_rs(Path(args.rs))
    log.info(f"Loaded RS for {len(rs)} dates")
    apply_filter(Path(args.case), rs, Path(args.out),
                  block_bottom=args.block_bottom,
                  require_top=args.require_top)


if __name__ == "__main__":
    main()
