"""Exhaustive Iron Condor sweep across:

  OTM %        : 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
  Wing width   : 100, 150, 200, 300, 500
  Stop mult    : 3, 5, 99 (≈ no SL — hold to expiry)
  Entry day    : ANY, MON, TUE, WED, THU, FRI
  Underlying   : NIFTY, FINNIFTY, BANKNIFTY
  Capital      : ₹2L, ₹5L, ₹10L

6 × 5 × 3 × 6 × 3 = 1,620 backtest runs per capital × 3 capitals = 4,860 runs.
At ~2-3 s per run on the prod VM: ~3-4 hour wall time.

Output:
  - Top-20 ranked table per capital
  - Top-20 ranked table overall (cross-capital)
  - Full JSON dump for downstream slicing
  - Per-winner export folder

Usage (VM):
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.exhaustive_sweep \
        --min-leg-volume 100 --top 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import (
    compute_ic_margin, run_ic,
)

# Worker-globals: each forked process resets these on first use. Avoids
# inheriting parent's SQLAlchemy engine which can have stale pooled conns.
_WORKER_PID = None


def _worker_init():
    """Force fresh DB engine per worker process. SQLAlchemy global engine
    is cached in ohlcv_cache module; reset it after fork."""
    global _WORKER_PID
    _WORKER_PID = os.getpid()
    try:
        import tools.shared.ohlcv_cache as oc
        oc._engine = None  # next _get_engine() call rebuilds for this PID
    except Exception:
        pass


def _run_combo(task: Tuple) -> Optional[Dict]:
    """Single backtest task — runs in worker process."""
    otm, wing, stop, dow, u, start, end, min_vol = task
    try:
        df = run_ic(u, start, end, otm, wing, stop, 0.01,
                    capital=200_000, lots=1, realistic_slip=True,
                    min_leg_volume=min_vol, entry_dow=dow)
    except Exception as e:
        return {"otm": otm, "wing": wing, "stop": stop, "dow": dow,
                "u": u, "error": str(e), "df": None}
    return {"otm": otm, "wing": wing, "stop": stop, "dow": dow, "u": u,
            "df": df.to_dict("records") if not df.empty else []}

EXPORTS = REPO_ROOT / "exports" / "models"

OTMS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
WINGS = [100, 150, 200, 300, 500]
STOPS = [3.0, 5.0, 99.0]  # 99 ≈ "no SL" (3× credit × 99 will never trigger)
DOWS = [-1, 0, 1, 2, 3, 4]   # -1=ANY, 0..4=Mon..Fri
DOW_NAMES = {-1: "ANY", 0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI"}
UNDERLYINGS = ["NIFTY", "FINNIFTY", "BANKNIFTY"]


def _peak_safe_lots(trades: pd.DataFrame, capital: float) -> int:
    if trades.empty:
        return 0
    peak_per_lot = trades.apply(lambda r: compute_ic_margin(
        float(r["ce_k"]), float(r["pe_k"]),
        float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
        int(r["lot"]), 1), axis=1).max()
    if peak_per_lot <= 0:
        return 0
    return max(0, int(capital / peak_per_lot))


def _rescale_stats(df: pd.DataFrame, new_lots: int, capital: float) -> Dict:
    if df.empty or new_lots <= 0:
        return None
    x = df.copy()
    x["lots"] = new_lots
    x["pnl_total"] = x["pnl_unit"] * x["lot"] * new_lots
    x["margin_required_inr"] = x.apply(lambda r: compute_ic_margin(
        float(r["ce_k"]), float(r["pe_k"]),
        float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
        int(r["lot"]), new_lots), axis=1)
    x = x.sort_values("entry_date").reset_index(drop=True)
    x["running_pnl"] = x["pnl_total"].cumsum()
    x["equity"] = capital + x["running_pnl"]
    x["peak"] = x["equity"].cummax()
    x["dd"] = (x["equity"] - x["peak"]) / x["peak"] * 100
    final = float(x["equity"].iloc[-1])
    total_pnl = float(x["pnl_total"].sum())
    n = len(x)
    wins = int((x["pnl_total"] > 0).sum())
    n_yrs = max(1.0, (pd.to_datetime(x["exit_date"].iloc[-1])
                      - pd.to_datetime(x["entry_date"].iloc[0])).days / 365.25)
    cagr = ((final / capital) ** (1 / n_yrs) - 1) * 100 if final > 0 else -100.0
    return {
        "trades": n, "wr": wins / n * 100 if n else 0.0,
        "cagr": cagr, "total_return": total_pnl / capital * 100,
        "max_dd": float(x["dd"].min()),
        "avg_margin": float(x["margin_required_inr"].mean()),
        "peak_margin": float(x["margin_required_inr"].max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capitals", default="200000,500000,1000000")
    ap.add_argument("--start", default="2023-05-15")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--min-leg-volume", type=int, default=100)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel worker processes (default 6; "
                         "I/O-bound on DB reads so > nproc is fine)")
    args = ap.parse_args()
    capitals = [int(x) for x in args.capitals.split(",")]

    combos = list(itertools.product(OTMS, WINGS, STOPS, DOWS, UNDERLYINGS))
    tasks = [(otm, wing, stop, dow, u,
              args.start, args.end, args.min_leg_volume)
             for (otm, wing, stop, dow, u) in combos]

    print(f"=== Exhaustive IC sweep (parallel × {args.workers}) ===")
    print(f"  {len(combos)} unique backtests × {len(capitals)} capitals = "
          f"{len(combos)*len(capitals)} result rows")
    print(f"  min_leg_volume={args.min_leg_volume}, workers={args.workers}")

    raw_by_combo: Dict[tuple, pd.DataFrame] = {}
    t0 = time.time()
    completed = 0
    with Pool(processes=args.workers, initializer=_worker_init) as pool:
        for res in pool.imap_unordered(_run_combo, tasks, chunksize=4):
            completed += 1
            key = (res["otm"], res["wing"], res["stop"], res["dow"], res["u"])
            if res.get("error"):
                print(f"  ! {key}: {res['error']}")
                raw_by_combo[key] = pd.DataFrame()
            else:
                rows = res["df"] or []
                raw_by_combo[key] = pd.DataFrame(rows) if rows else pd.DataFrame()
            if completed % 50 == 0:
                rate = completed / max(1, time.time() - t0)
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(tasks)}] elapsed "
                      f"{time.time()-t0:.0f}s  rate {rate:.1f}/s  "
                      f"ETA {eta:.0f}s")

    print(f"  done in {time.time()-t0:.0f}s. building rescale table…")

    # Build long-form result table: one row per (combo, capital).
    rows: List[Dict] = []
    for (otm, wing, stop, dow, u), df in raw_by_combo.items():
        if df.empty:
            continue
        for cap in capitals:
            lots = _peak_safe_lots(df, cap)
            if lots <= 0:
                continue
            stats = _rescale_stats(df, lots, cap)
            if stats is None:
                continue
            rows.append({
                "underlying": u, "otm": otm, "wing": wing, "stop": stop,
                "dow": DOW_NAMES[dow], "capital": cap, "lots": lots,
                **stats,
            })

    if not rows:
        print("No tradeable rows. Try lowering --min-leg-volume.")
        return 0

    rdf = pd.DataFrame(rows)
    # JSON dump for downstream tooling.
    out = Path("/tmp") / f"ic_exhaustive_minvol{args.min_leg_volume}.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nJSON dump: {out}")
    rdf.to_csv(Path("/tmp") / "ic_exhaustive.csv", index=False)

    # Top-N per capital.
    for cap in capitals:
        sub = rdf[rdf["capital"] == cap].sort_values("cagr", ascending=False).head(args.top)
        print(f"\n=== Top {args.top} at ₹{cap:,} capital ===")
        print(f"{'rank':>4} {'underlying':10} {'otm':>4} {'wing':>4} "
              f"{'stop':>5} {'dow':>4} {'lots':>4} {'trades':>6} "
              f"{'WR%':>5} {'CAGR%':>8} {'Total%':>8} {'MaxDD%':>8} "
              f"{'avgM':>11} {'peakM':>11}")
        print("-" * 122)
        for i, r in enumerate(sub.itertuples(), 1):
            print(f"{i:>4} {r.underlying:10} {r.otm:>4.1f} {r.wing:>4} "
                  f"{r.stop:>5.1f} {r.dow:>4} {r.lots:>4} {r.trades:>6} "
                  f"{r.wr:>5.1f} {r.cagr:>+8.1f} {r.total_return:>+8.1f} "
                  f"{r.max_dd:>+8.1f} "
                  f"₹{r.avg_margin:>9,.0f} ₹{r.peak_margin:>9,.0f}")

    # Top-N overall by CAGR (across all capitals).
    print(f"\n=== Top {args.top} overall (any capital) ===")
    sub = rdf.sort_values("cagr", ascending=False).head(args.top)
    print(f"{'rank':>4} {'underlying':10} {'otm':>4} {'wing':>4} "
          f"{'stop':>5} {'dow':>4} {'cap':>5} {'lots':>4} {'trades':>6} "
          f"{'WR%':>5} {'CAGR%':>8} {'MaxDD%':>8}")
    print("-" * 100)
    for i, r in enumerate(sub.itertuples(), 1):
        print(f"{i:>4} {r.underlying:10} {r.otm:>4.1f} {r.wing:>4} "
              f"{r.stop:>5.1f} {r.dow:>4} {int(r.capital/100000):>3}L "
              f"{r.lots:>4} {r.trades:>6} "
              f"{r.wr:>5.1f} {r.cagr:>+8.1f} {r.max_dd:>+8.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
