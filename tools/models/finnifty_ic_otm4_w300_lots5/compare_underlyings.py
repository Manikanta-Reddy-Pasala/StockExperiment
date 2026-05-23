"""Compare Iron Condor variants across NIFTY / FINNIFTY / BANKNIFTY at a
fixed investment capital. Picks PEAK-SAFE lot size per variant (every
backtested entry must fit within capital) and ranks by CAGR.

Why this exists: prior FinNifty-only sweeps assumed ₹2L capital but
backtested 5 lots — peak margin actually needed ₹6L+. At realistic
₹2L capital the lot count is the binding constraint, not strike
geometry. Different underlyings have different lot sizes (NIFTY 75,
BANKNIFTY 30, FINNIFTY 60) so the peak-safe lots per ₹2L differ —
worth re-comparing.

Output: console table + JSON dump in /tmp + sibling export folders
exports/models/<underlying>_ic_otm{N}_w{W}_lots{L}_cap{K}k/ for the
top variants. Same schema as the existing FinNifty exports so the live
executor can switch with no code change.

Usage (on the VM where the options DB lives):
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.compare_underlyings \
        --capital 200000 --start 2023-05-15 --end 2026-05-15 --top 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import (
    compute_ic_margin, lot_size_for, run_ic,
)

EXPORTS = REPO_ROOT / "exports" / "models"

# Variants chosen to span the OTM × wing-width × stop space without
# blowing up runtime. Each variant produces 1 backtest per underlying.
VARIANTS = [
    # (otm_pct, wing_width, stop_mult, slip, label)
    (2.0, 150, 3.0, 0.01, "otm2_w150"),
    (2.0, 200, 3.0, 0.01, "otm2_w200"),
    (3.0, 200, 3.0, 0.01, "otm3_w200"),
    (3.0, 300, 3.0, 0.01, "otm3_w300"),
    (4.0, 300, 3.0, 0.01, "otm4_w300"),
    (4.0, 500, 3.0, 0.01, "otm4_w500"),
    (5.0, 500, 3.0, 0.01, "otm5_w500"),
]
UNDERLYINGS = ["NIFTY", "FINNIFTY", "BANKNIFTY"]


def _peak_safe_lots(trades: pd.DataFrame, capital: float) -> int:
    """Largest lot count s.t. PEAK per-trade margin ≤ capital."""
    if trades.empty:
        return 0
    peak_per_lot = trades.apply(lambda r: compute_ic_margin(
        float(r["ce_k"]), float(r["pe_k"]),
        float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
        int(r["lot"]), 1), axis=1).max()
    if peak_per_lot <= 0:
        return 0
    return max(0, int(capital / peak_per_lot))


def _rescale_and_summarise(trades_lots1: pd.DataFrame, new_lots: int,
                           capital: float) -> Dict:
    """Rescale a lots=1 trades DataFrame to new_lots and return stats."""
    if trades_lots1.empty or new_lots <= 0:
        return {"trades": 0, "wr": 0.0, "cagr": 0.0,
                "total_return_pct": 0.0, "max_dd_pct": 0.0,
                "avg_margin": 0.0, "peak_margin": 0.0,
                "tradeable": False}
    df = trades_lots1.copy()
    df["lots"] = new_lots
    df["pnl_total"] = df["pnl_unit"] * df["lot"] * new_lots
    df["max_loss_total"] = df["max_loss_per_unit"] * df["lot"] * new_lots
    df["margin_required_inr"] = df.apply(lambda r: compute_ic_margin(
        float(r["ce_k"]), float(r["pe_k"]),
        float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
        int(r["lot"]), new_lots), axis=1)
    df = df.sort_values("entry_date").reset_index(drop=True)
    df["running_pnl"] = df["pnl_total"].cumsum()
    df["equity"] = capital + df["running_pnl"]
    df["peak"] = df["equity"].cummax()
    df["drawdown_pct"] = (df["equity"] - df["peak"]) / df["peak"] * 100
    final_eq = float(df["equity"].iloc[-1])
    total_pnl = float(df["pnl_total"].sum())
    n = len(df)
    wins = int((df["pnl_total"] > 0).sum())
    n_years = max(1.0, (pd.to_datetime(df["exit_date"].iloc[-1])
                        - pd.to_datetime(df["entry_date"].iloc[0])).days / 365.25)
    cagr = ((final_eq / capital) ** (1 / n_years) - 1) * 100 if final_eq > 0 else -100.0
    return {
        "trades": n,
        "wr": wins / n * 100 if n else 0.0,
        "cagr": cagr,
        "total_return_pct": total_pnl / capital * 100,
        "max_dd_pct": float(df["drawdown_pct"].min()),
        "avg_margin": float(df["margin_required_inr"].mean()),
        "peak_margin": float(df["margin_required_inr"].max()),
        "trades_df": df,
        "tradeable": True,
    }


def _run_one(u: str, v: tuple, start: str, end: str,
             capital: float, min_leg_volume: int) -> Dict:
    otm, wing, stop, slip, label = v
    print(f"  running {u:9} {label}… (min_vol={min_leg_volume})", flush=True)
    raw_unfiltered = run_ic(u, start, end, otm, wing, stop, slip,
                            capital=capital, lots=1, realistic_slip=True,
                            min_leg_volume=0)
    raw = run_ic(u, start, end, otm, wing, stop, slip,
                 capital=capital, lots=1, realistic_slip=True,
                 min_leg_volume=min_leg_volume)
    n_unfiltered = len(raw_unfiltered)
    n_filtered = len(raw)
    rejected = n_unfiltered - n_filtered
    if raw.empty:
        return {"underlying": u, "variant": label, "skipped": True,
                "reason": f"all_trades_rejected_by_vol_filter ({n_unfiltered}→0)"}
    raw["month"] = pd.to_datetime(raw["entry_date"]).dt.to_period("M").astype(str)
    raw["year"] = pd.to_datetime(raw["entry_date"]).dt.year
    new_lots = _peak_safe_lots(raw, capital)
    if new_lots <= 0:
        return {"underlying": u, "variant": label, "lots": 0,
                "skipped": True, "reason": "peak_margin_exceeds_capital"}
    stats = _rescale_and_summarise(raw, new_lots, capital)
    return {
        "underlying": u, "variant": label, "otm": otm, "wing": wing,
        "stop": stop, "lots": new_lots,
        "n_unfiltered": n_unfiltered,
        "n_vol_rejected": rejected,
        "vol_reject_pct": (rejected / n_unfiltered * 100) if n_unfiltered else 0.0,
        **{k: v for k, v in stats.items() if k != "trades_df"},
        "_trades_df": stats.get("trades_df"),
    }


def _print_table(rows: List[Dict]) -> None:
    rows_ok = [r for r in rows if not r.get("skipped") and r.get("tradeable")]
    rows_ok.sort(key=lambda r: r["cagr"], reverse=True)
    print()
    print(f"{'rank':>4}  {'underlying':10} {'variant':12} {'lots':>4} "
          f"{'trades':>6} {'rej%':>5} {'WR%':>5} {'CAGR%':>8} {'Total%':>8} "
          f"{'MaxDD%':>8} {'avgM':>10} {'peakM':>10}")
    print("-" * 122)
    for i, r in enumerate(rows_ok, 1):
        print(f"{i:>4}  {r['underlying']:10} {r['variant']:12} {r['lots']:>4} "
              f"{r['trades']:>6} {r.get('vol_reject_pct',0):>5.1f} "
              f"{r['wr']:>5.1f} {r['cagr']:>8.1f} "
              f"{r['total_return_pct']:>8.1f} {r['max_dd_pct']:>8.1f} "
              f"₹{r['avg_margin']:>8,.0f} ₹{r['peak_margin']:>8,.0f}")
    skipped = [r for r in rows if r.get("skipped")]
    if skipped:
        print("\nSKIPPED:")
        for r in skipped:
            print(f"  {r['underlying']:9} {r['variant']:12}"
                  f" — {r.get('reason', 'no trades')}")


def _save_top_exports(rows: List[Dict], capital: float, top: int) -> None:
    rows_ok = [r for r in rows if r.get("tradeable")]
    rows_ok.sort(key=lambda r: r["cagr"], reverse=True)
    cap_label = f"{int(capital/1000)}k"
    for r in rows_ok[:top]:
        u_lower = r["underlying"].lower()
        folder = EXPORTS / f"{u_lower}_ic_{r['variant']}_lots{r['lots']}_cap{cap_label}"
        folder.mkdir(parents=True, exist_ok=True)
        df = r["_trades_df"]
        df.to_csv(folder / "trades.csv", index=False)
        content = (
            f"# {u_lower}_ic_{r['variant']}_lots{r['lots']}_cap{cap_label}\n\n"
            f"{r['underlying']} monthly Iron Condor — peak-safe at "
            f"₹{int(capital):,} investment capital.\n\n"
            f"## Result\n\n"
            f"| Metric | Value |\n|---|---:|\n"
            f"| Underlying | {r['underlying']} |\n"
            f"| OTM | {r['otm']} % |\n"
            f"| Wing width | {r['wing']} |\n"
            f"| Stop multiplier | {r['stop']}× |\n"
            f"| Lots | **{r['lots']}** |\n"
            f"| Trades | {r['trades']} |\n"
            f"| Win rate | {r['wr']:.1f} % |\n"
            f"| CAGR | **{r['cagr']:+.1f} %** |\n"
            f"| Total return | {r['total_return_pct']:+.1f} % |\n"
            f"| Max drawdown | {r['max_dd_pct']:.1f} % |\n"
            f"| Avg margin / trade | ₹{r['avg_margin']:,.0f} |\n"
            f"| Peak margin / trade | ₹{r['peak_margin']:,.0f} |\n"
            f"| Capital | ₹{int(capital):,} |\n"
            f"| Capital / peak ratio | {capital/r['peak_margin']:.2f}× |\n"
        )
        (folder / "SUMMARY.md").write_text(content)
        print(f"  wrote {folder}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--start", default="2023-05-15")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--top", type=int, default=3,
                    help="How many top-CAGR variants to write as export folders")
    ap.add_argument("--min-leg-volume", type=int, default=100,
                    help="Reject historical leg-days with volume below this "
                         "threshold (0 = no filter). Default 100 — keeps "
                         "out fantasy fills on zero-volume bars.")
    args = ap.parse_args()

    print(f"=== Comparing IC variants across underlyings ===")
    print(f"  capital  = ₹{int(args.capital):,}")
    print(f"  window   = {args.start} → {args.end}")
    print(f"  lot-mode = peak-safe (every entry openable at broker)")
    print()

    rows: List[Dict] = []
    for u in UNDERLYINGS:
        print(f"\n[{u}] lot size today = {lot_size_for(u, datetime.today().date())}")
        for v in VARIANTS:
            try:
                rows.append(_run_one(u, v, args.start, args.end,
                                     args.capital, args.min_leg_volume))
            except Exception as e:
                print(f"  ! {u} {v[4]} failed: {e}")
                rows.append({"underlying": u, "variant": v[4],
                             "skipped": True, "reason": str(e)})

    _print_table(rows)
    _save_top_exports(rows, args.capital, args.top)

    serialisable = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "_trades_df"}
        serialisable.append(d)
    out = Path("/tmp") / f"ic_underlying_compare_cap{int(args.capital)}.json"
    out.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"\nJSON dump: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
