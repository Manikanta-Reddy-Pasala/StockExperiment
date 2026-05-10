"""Merge per-year `_monthly_profile.md` files into per-model 3-year tables.

Each yearly run drops a `_monthly_profile.md` inside its result dir
(e.g. `exports/backtests/yearly/nifty50_ema_200_400_2023_2024/`).
Per-model 36-month report = concatenate 3 such files in chrono order
and emit one table.

Usage:
  python merge_monthly_profiles.py --root exports/backtests/yearly \
    --universe nifty50 [--out exports/backtests/yearly/nifty50_monthly_3yr.md]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

MODELS = ["ema_200_400", "ema_9_21", "swing_pullback", "orb_15min"]


def parse_monthly(md_path: Path) -> List[Tuple[str, List[str]]]:
    """Return list of (yyyy_mm, row_cells_str). Skips the header rows."""
    rows: List[Tuple[str, List[str]]] = []
    if not md_path.exists():
        return rows
    in_table = False
    for line in md_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("| YYYY-MM"):
            in_table = True
            continue
        if in_table and s.startswith("|---"):
            continue
        if in_table and s.startswith("|"):
            cells = [c.strip() for c in s.split("|")[1:-1]]
            if cells and re.match(r"\d{4}-\d{2}", cells[0]):
                rows.append((cells[0], cells))
        elif in_table and not s.startswith("|"):
            in_table = False
    return rows


def merge_model(root: Path, universe: str, model: str) -> List[List[str]]:
    """Find all `<universe>_<model>_<from>_<to>` dirs under root, sort
    chronologically by `<from>` year, concat their monthly rows."""
    pattern = re.compile(rf"{re.escape(universe)}_{re.escape(model)}_(\d{{4}})_(\d{{4}})$")
    candidates: List[Tuple[int, Path]] = []
    if not root.exists():
        return []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            candidates.append((int(m.group(1)), child))
    candidates.sort()
    out: List[List[str]] = []
    seen_months = set()
    for _yr, d in candidates:
        for ts, cells in parse_monthly(d / "_monthly_profile.md"):
            if ts in seen_months:
                continue
            seen_months.add(ts)
            out.append(cells)
    out.sort(key=lambda r: r[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exports/backtests/yearly")
    ap.add_argument("--universe", required=True,
                    choices=["nifty50", "nifty500"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="Comma-sep list of model keys")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out) if args.out else (
        root / f"{args.universe}_monthly_3yr.md"
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    lines = [
        f"# {args.universe.upper()} — Monthly Profile (3-year merged) — ₹2L capital",
        "",
        "Per-model month-by-month P&L from yearly orchestrator output. Each model",
        "is run separately for each of the 3 trailing 12-month windows; the rows",
        "below stitch them in chronological order (oldest year first).",
        "",
    ]

    for model in models:
        rows = merge_model(root, args.universe, model)
        lines += [
            f"## {model}",
            "",
            "| YYYY-MM | Trades | Win | Loss | Win% | Avg ₹ | Sum ₹ | EndEquity ₹ | DD% |",
            "|---------|--------|-----|------|------|-------|-------|-------------|-----|",
        ]
        if not rows:
            lines.append("| _no data_ | | | | | | | | |")
        else:
            for cells in rows:
                # Pad to 9 cells if monthly_profile schema differs.
                cells = cells + [""] * max(0, 9 - len(cells))
                lines.append("| " + " | ".join(cells[:9]) + " |")
        # Aggregate
        total_trades = sum(_safe_int(r[1]) for r in rows)
        total_wins = sum(_safe_int(r[2]) for r in rows)
        total_pnl = sum(_safe_inr(r[6]) for r in rows)
        avg_per = (total_pnl / total_trades) if total_trades else 0
        win_rate = (total_wins / total_trades * 100) if total_trades else 0
        lines += [
            "",
            f"**3-year total:** {total_trades} trades, win rate {win_rate:.1f}%, "
            f"sum P&L ₹{total_pnl:,.0f}, avg/leg ₹{avg_per:,.0f}",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


def _safe_int(s: str) -> int:
    try:
        return int(re.sub(r"[^\d-]", "", s) or 0)
    except ValueError:
        return 0


def _safe_inr(s: str) -> float:
    """Parse '+1,234' / '-5,678' / '1234' to float."""
    try:
        return float(re.sub(r"[^\d.\-]", "", s) or 0)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
