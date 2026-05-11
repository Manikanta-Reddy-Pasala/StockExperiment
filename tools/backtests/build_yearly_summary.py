"""Build a unified yearly summary across ALL models for one universe.

Each per-model orchestrator writes its own `<universe>_summary.md` in
`exports/backtests/yearly/`, racing/overwriting the previous one. This
script walks every `<universe>_<model>_<from>_<to>/_capital_sim.txt`
file, picks the requested max_concurrent row, and emits one consolidated
markdown file: `<universe>_yearly_all_models.md`.

Also computes 3-year aggregate per model.

Usage:
  python build_yearly_summary.py --universe nifty50 [--max-concurrent 2]
                                 [--root exports/backtests/yearly]
                                 [--capital 200000]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

MODELS = ["ema_200_400", "ema_9_21", "swing_pullback", "orb_15min"]


def count_data_coverage(dir_path: Path) -> Dict:
    """Count per-symbol .md files (excluding _summary, _monthly) + flag
    those with 'No 1H data available' placeholder."""
    if not dir_path.exists():
        return {"with_data": 0, "no_data": 0}
    total = 0
    no_data = 0
    for f in dir_path.glob("*.md"):
        if f.name.startswith("_"):
            continue
        total += 1
        try:
            head = f.read_text()[:500]
        except Exception:
            continue
        if "No 1H data available" in head or "_No data_" in head:
            no_data += 1
    return {"with_data": total - no_data, "no_data": no_data}


def parse_capital_sim(txt_path: Path, max_concurrent: int) -> Dict:
    """Parse `_capital_sim.txt` for the row matching max_concurrent.
    Returns {} if file missing or row absent."""
    if not txt_path.exists():
        return {}
    pat = re.compile(
        rf"^\s*{max_concurrent}\s+(\d+)\s+(\d+)\s+([\d,]+)\s+([+\-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    )
    for line in txt_path.read_text().splitlines():
        m = pat.match(line)
        if m:
            return {
                "taken":      int(m.group(1)),
                "skip":       int(m.group(2)),
                "final":      int(m.group(3).replace(",", "")),
                "roi_pct":    float(m.group(4)),
                "max_dd_pct": float(m.group(5)),
                "open_end":   int(m.group(6)),
            }
    return {}


def find_year_dirs(root: Path, universe: str, model: str) -> List[Tuple[str, Path]]:
    """Return [(window_label, dir_path), ...] sorted oldest-first."""
    pat = re.compile(rf"^{re.escape(universe)}_{re.escape(model)}_(\d{{4}}_\d{{4}})$")
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if m:
            out.append((m.group(1), child))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["nifty50", "nifty500"])
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--root", default="exports/backtests/yearly")
    ap.add_argument("--capital", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out) if args.out else (
        root / f"{args.universe}_yearly_all_models.md"
    )
    cap = args.max_concurrent

    lines = [
        f"# {args.universe.upper()} — Yearly Backtest, All 4 Models — ₹{args.capital:,}",
        "",
        f"Capital: ₹{args.capital:,}, max_concurrent: {cap}",
        "",
        "## Data coverage",
        "",
        "Number of universe stocks that produced backtest output per model-year.",
        "Missing = recent IPOs, ticker reconstitutions, or insufficient bar history",
        "(e.g. Fyers 5m bars capped to ~6mo back; EMA 200/400 needs ~400d warmup).",
        "",
        "## Yearly Headlines",
        "",
        "| Model | Year | Window | WithData | Taken | Skip | Final₹ | ROI% | MaxDD% |",
        "|-------|------|--------|--------:|------:|-----:|-------:|-----:|-------:|",
    ]

    # Per-model aggregate accumulators
    agg: Dict[str, Dict] = {m: {"years": 0, "rois": [], "max_dd": 0.0,
                                  "best": None, "worst": None,
                                  "total_taken": 0, "total_pnl": 0}
                              for m in MODELS}

    for model in MODELS:
        for label, d in find_year_dirs(root, args.universe, model):
            cs = parse_capital_sim(d / "_capital_sim.txt", cap)
            cov = count_data_coverage(d)
            with_data = cov["with_data"]
            if not cs:
                lines.append(
                    f"| {model} | {label} | _missing_ | {with_data} | | | | | |"
                )
                continue
            yr_from, yr_to = label.split("_")
            window = f"{yr_from}-05-11..{yr_to}-05-11"
            lines.append(
                f"| {model} | {label} | {window} | {with_data} | {cs['taken']} | {cs['skip']} | "
                f"{cs['final']:,} | {cs['roi_pct']:+.2f} | {cs['max_dd_pct']:.2f} |"
            )
            a = agg[model]
            a["years"] += 1
            a["rois"].append(cs["roi_pct"])
            a["max_dd"] = max(a["max_dd"], cs["max_dd_pct"])
            a["total_taken"] += cs["taken"]
            a["total_pnl"]   += (cs["final"] - args.capital)
            if a["best"] is None or cs["roi_pct"] > a["best"]:
                a["best"] = cs["roi_pct"]
            if a["worst"] is None or cs["roi_pct"] < a["worst"]:
                a["worst"] = cs["roi_pct"]

    lines += [
        "",
        "## Per-model 3-year aggregate",
        "",
        "| Model | Years | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% | Total trades | Total P&L ₹ |",
        "|-------|------:|---------:|-------------:|----------:|-----------:|-------------:|------------:|",
    ]
    for model in MODELS:
        a = agg[model]
        if a["years"] == 0:
            lines.append(f"| {model} | 0 | _no data_ | | | | | |")
            continue
        avg_roi = sum(a["rois"]) / a["years"]
        lines.append(
            f"| {model} | {a['years']} | {avg_roi:+.2f} | {a['max_dd']:.2f} | "
            f"{a['best']:+.2f} | {a['worst']:+.2f} | {a['total_taken']} | "
            f"{a['total_pnl']:+,} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
