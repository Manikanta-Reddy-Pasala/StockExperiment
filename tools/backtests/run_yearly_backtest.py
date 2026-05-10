"""Yearly backtest orchestrator — runs each registered strategy model for each
of the last N years (separate, NOT merged) under a fixed capital constraint.

For each (model, year) pair:
    1. Invokes the model's harness with --from/--to/--out
    2. Runs realistic_capital_sim across multiple max-concurrent caps,
       writing _capital_sim.txt
    3. Runs monthly_profile.py, writing _monthly_profile.md
    4. Captures yearly headline ROI/MDD by parsing the realistic_capital_sim
       output at --max-concurrent=2 (the design "best-fit" for ₹2L capital)

Final aggregate: ``exports/backtests/yearly/<universe>_summary.md``.

Penny-stock filter caveat:
    - swing_pullback has ``min_adv_inr`` (default ₹5cr ADV) — a liquidity floor
    - EMA 200/400 + EMA 9/21 + ORB 15min currently have no min_price field;
      penny filter only applies on swing_pullback. (Tracked separately.)

Usage:
    venv/bin/python tools/backtests/run_yearly_backtest.py \
        --universe nifty50 [--models all|csv] [--capital 200000] \
        [--years 3] [--end-date YYYY-MM-DD] [--max-concurrent 2]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---- Model registry (with hardcoded fallback) -----------------------
HARDCODED_MODELS: Dict[str, Dict] = {
    "ema_200_400": {
        "harness_module": "tools.backtests.run_ema_200_400_backtest",
        "harness_script": "tools/backtests/run_ema_200_400_backtest.py",
        "extra_args": ["--ema-fast", "200", "--ema-slow", "400", "--warmup-days", "400"],
        "bars_interval": "1h",
    },
    "ema_9_21": {
        "harness_module": "tools.backtests.run_ema_200_400_backtest",
        "harness_script": "tools/backtests/run_ema_200_400_backtest.py",
        "extra_args": ["--ema-fast", "9", "--ema-slow", "21", "--warmup-days", "60"],
        "bars_interval": "1h",
    },
    "swing_pullback": {
        "harness_module": "tools.backtests.run_swing_pullback_backtest",
        "harness_script": "tools/backtests/run_swing_pullback_backtest.py",
        "extra_args": ["--warmup-days", "250"],
        "bars_interval": "daily",
    },
    "orb_15min": {
        "harness_module": "tools.backtests.run_orb_intraday_backtest",
        "harness_script": "tools/backtests/run_orb_intraday_backtest.py",
        "extra_args": [],
        "bars_interval": "5m",
    },
}


def _load_models() -> Dict[str, Dict]:
    """Try to merge registry metadata with hardcoded harness paths.

    Registry exposes ``harness_module``, ``bars_interval``, etc. — but not
    the per-model CLI flags (e.g. EMA fast/slow). We keep flags in the
    hardcoded table; if registry has the model, we honor its harness module.
    """
    merged = {k: dict(v) for k, v in HARDCODED_MODELS.items()}
    try:
        from src.services.technical.models import MODELS  # type: ignore
        for key, entry in MODELS.items():
            slot = merged.setdefault(key, {
                "extra_args": [],
                "bars_interval": entry.get("bars_interval", "?"),
            })
            slot["bars_interval"] = entry.get("bars_interval", slot.get("bars_interval"))
            harness_mod = entry.get("harness_module")
            if harness_mod:
                slot["harness_module"] = harness_mod
                # tools.backtests.run_xxx -> tools/backtests/run_xxx.py
                slot["harness_script"] = harness_mod.replace(".", "/") + ".py"
    except Exception as e:
        print(f"  (registry unavailable: {e}; using hardcoded models)")
    return merged


# ---- Year window computation ---------------------------------------
def year_windows(end_date: datetime, years: int, start_from: Optional[int] = None
                  ) -> List[Tuple[str, str, str]]:
    """Return list of (label, from_date, to_date) for the last N years.

    Each window is one calendar year ending at end_date - i*365d.
    If ``start_from`` is given (e.g. 2022), include only windows whose start
    year >= start_from.
    """
    out: List[Tuple[str, str, str]] = []
    for i in range(years):
        to_dt = end_date - timedelta(days=365 * i)
        from_dt = to_dt - timedelta(days=365)
        label = f"{from_dt.year}_{to_dt.year}"
        if start_from is not None and from_dt.year < start_from:
            continue
        out.append((label, from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")))
    return out


# ---- Subprocess wrappers -------------------------------------------
def run_harness(model_key: str, model_info: Dict, universe: str,
                 from_date: str, to_date: str, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(ROOT / model_info["harness_script"]),
        "--universe", universe,
        "--from", from_date,
        "--to", to_date,
        "--out", str(out_dir),
    ]
    cmd += list(model_info.get("extra_args", []))
    print(f"  $ {' '.join(cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), check=False)
        return r.returncode
    except Exception as e:
        print(f"  harness error: {e}")
        return -1


def run_capital_sim(out_dir: Path, capital: int) -> str:
    """Run realistic_capital_sim, capture stdout, write to _capital_sim.txt."""
    cmd = [
        sys.executable,
        str(ROOT / "tools/backtests/realistic_capital_sim.py"),
        str(out_dir),
        "--capital", str(capital),
        "1", "2", "3", "5", "8", "10", "20", "30", "50",
    ]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        out = r.stdout + (r.stderr if r.stderr else "")
    except Exception as e:
        out = f"capital_sim error: {e}\n"
    (out_dir / "_capital_sim.txt").write_text(out)
    return out


def run_monthly_profile(out_dir: Path, capital: int, max_concurrent: int) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "tools/backtests/monthly_profile.py"),
        str(out_dir),
        "--capital", str(capital),
        "--max-concurrent", str(max_concurrent),
    ]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        return r.stdout + (r.stderr if r.stderr else "")
    except Exception as e:
        return f"monthly_profile error: {e}\n"


# ---- Capital-sim output parsing -----------------------------------
# realistic_capital_sim prints rows like:
#   " Max  Taken  Skip   Final     ROI%   MaxDD%  OpenEnd"
#   "   2     45     0  204,567   +2.28    10.91     0"
_CAP_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d,\-]+)\s+([+\-\d\.]+)\s+([\d\.]+)\s+(\d+)\s*$"
)


def parse_cap_sim_for_concurrency(text: str, max_concurrent: int) -> Optional[Dict]:
    for line in text.splitlines():
        m = _CAP_ROW_RE.match(line)
        if not m:
            continue
        if int(m.group(1)) != max_concurrent:
            continue
        return {
            "max_concurrent": int(m.group(1)),
            "taken": int(m.group(2)),
            "skipped": int(m.group(3)),
            "final": float(m.group(4).replace(",", "")),
            "roi_pct": float(m.group(5)),
            "max_dd_pct": float(m.group(6)),
            "open_end": int(m.group(7)),
        }
    return None


# ---- Main orchestrator --------------------------------------------
@dataclass
class YearResult:
    model: str
    year_label: str
    from_date: str
    to_date: str
    out_dir: Path
    harness_rc: int = -1
    cap_summary: Optional[Dict] = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True,
                        choices=["smoke", "nifty50", "nifty500", "indices"])
    parser.add_argument("--models", default="all",
                        help="'all' or comma-separated keys (e.g. ema_200_400,swing_pullback)")
    parser.add_argument("--capital", type=int, default=200_000)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--end-date", type=str, default=None,
                        help="Anchor date for the year windows (default: today). "
                             "YYYY-MM-DD.")
    parser.add_argument("--start-from", type=int, default=None,
                        help="Earliest start year to include (e.g. 2022).")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Concurrency cap used for headline ROI/MDD parse.")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "exports" / "backtests" / "yearly")
    args = parser.parse_args()

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    windows = year_windows(end_date, args.years, args.start_from)
    if not windows:
        print("No windows produced (check --years / --start-from).")
        return 2

    models_table = _load_models()
    if args.models == "all":
        model_keys = list(models_table.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    missing = [k for k in model_keys if k not in models_table]
    if missing:
        print(f"Unknown models: {missing}. Known: {sorted(models_table)}")
        return 2

    print(f"Universe: {args.universe}")
    print(f"Capital: INR {args.capital:,}")
    print(f"Models: {model_keys}")
    print(f"Year windows: {[w[0] + ' ' + w[1] + '..' + w[2] for w in windows]}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    results: List[YearResult] = []

    for model_key in model_keys:
        info = models_table[model_key]
        for label, from_d, to_d in windows:
            tag = f"{args.universe}_{model_key}_{label}"
            out_dir = args.out_root / tag
            print(f"\n=== {tag} ===")
            print(f"  window: {from_d} .. {to_d}")
            yr = YearResult(model=model_key, year_label=label, from_date=from_d,
                             to_date=to_d, out_dir=out_dir)
            rc = run_harness(model_key, info, args.universe, from_d, to_d, out_dir)
            yr.harness_rc = rc
            if rc != 0:
                print(f"  harness exit={rc} — skipping post-processing")
                results.append(yr)
                continue
            cap_text = run_capital_sim(out_dir, args.capital)
            yr.cap_summary = parse_cap_sim_for_concurrency(cap_text, args.max_concurrent)
            run_monthly_profile(out_dir, args.capital, args.max_concurrent)
            results.append(yr)

    # ---- Aggregate summary --------------------------------------
    summary_path = args.out_root / f"{args.universe}_summary.md"
    lines: List[str] = [
        f"# Yearly Backtest Summary — {args.universe}",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"- Capital: INR {args.capital:,}",
        f"- Max concurrent (headline): {args.max_concurrent}",
        f"- Years: {args.years} ({len(windows)} windows)",
        f"- Models: {model_keys}",
        "",
        "## Penny stock filter",
        "- ``swing_pullback``: ``min_adv_inr`` liquidity floor (₹5cr ADV default).",
        "- ``ema_200_400`` / ``ema_9_21`` / ``orb_15min``: no min_price field yet — "
        "penny filter applies only at the swing_pullback strategy level.",
        "",
        "## Yearly Headlines",
        "",
        "| Model | Year | Window | RC | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |",
        "|-------|------|--------|---:|------:|-----:|-------:|-----:|-------:|---------:|",
    ]
    for yr in results:
        cap = yr.cap_summary or {}
        lines.append(
            f"| {yr.model} | {yr.year_label} | {yr.from_date}..{yr.to_date} | "
            f"{yr.harness_rc} | {cap.get('taken', '-')} | {cap.get('skipped', '-')} | "
            f"{cap.get('final', 0.0):,.0f} | "
            f"{cap.get('roi_pct', 0.0):+.2f} | {cap.get('max_dd_pct', 0.0):.2f} | "
            f"{cap.get('open_end', '-')} |"
        )

    # Per-model 3yr aggregate
    lines += [
        "",
        "## Per-model 3-year aggregate",
        "",
        "| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |",
        "|-------|----------:|---------:|-------------:|----------:|-----------:|",
    ]
    by_model: Dict[str, List[YearResult]] = {}
    for yr in results:
        by_model.setdefault(yr.model, []).append(yr)
    for m, yrs in by_model.items():
        rois = [y.cap_summary["roi_pct"] for y in yrs if y.cap_summary]
        dds = [y.cap_summary["max_dd_pct"] for y in yrs if y.cap_summary]
        if not rois:
            lines.append(f"| {m} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {m} | {len(rois)} | "
            f"{sum(rois) / len(rois):+.2f} | {max(dds):.2f} | "
            f"{max(rois):+.2f} | {min(rois):+.2f} |"
        )

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
