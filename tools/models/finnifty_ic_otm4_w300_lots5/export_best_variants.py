"""Export the BEST IC variant per index (NIFTY/FINNIFTY/BANKNIFTY) from the
exhaustive sweep verdict — full trades.csv + summary.json + per-leg orders.csv
for 3-year window 2023-05-15 → 2026-05-15.

Best variants (from IC_EXHAUSTIVE_SWEEP_RESULTS.md):
  FINNIFTY  OTM2.5 W150 TUE no-SL  → +13.1 % CAGR / -4.3 % DD / 90.9 % WR
  NIFTY     OTM5.0 W500 THU no-SL  → +10.4 % CAGR / -2.3 % DD / 90.9 % WR
  BANKNIFTY OTM1.5 W500 WED no-SL  → +10.1 % CAGR / -11.2 % DD / 61.1 % WR

Three target capitals: ₹2L, ₹5L, ₹10L (peak-safe lot sizing per variant).

Usage (on VM):
    docker exec trading_system_app python3 -m \
        tools.models.finnifty_ic_otm4_w300_lots5.export_best_variants
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import (
    compute_ic_margin, run_ic,
)

EXPORTS = REPO_ROOT / "exports" / "models"

# (underlying, otm, wing, stop, dow, slug)
BEST = [
    ("FINNIFTY",  2.5, 150, 99.0, 1, "finnifty_otm2_5_w150_tue_nosl"),  # TUE=1
    ("NIFTY",     5.0, 500, 99.0, 3, "nifty_otm5_w500_thu_nosl"),       # THU=3
    ("BANKNIFTY", 1.5, 500, 99.0, 2, "banknifty_otm1_5_w500_wed_nosl"), # WED=2
]
CAPITALS = [200_000, 500_000, 1_000_000]
START, END = "2023-05-15", "2026-05-15"


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


def _rescale(raw: pd.DataFrame, new_lots: int, capital: float) -> pd.DataFrame:
    df = raw.copy()
    df["lots"] = new_lots
    df["pnl_total"] = df["pnl_unit"] * df["lot"] * new_lots
    df["max_loss_total"] = df["max_loss_per_unit"] * df["lot"] * new_lots
    df["margin_required_inr"] = df.apply(lambda r: compute_ic_margin(
        float(r["ce_k"]), float(r["pe_k"]),
        float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
        int(r["lot"]), new_lots), axis=1)
    df["margin_pct_of_capital"] = (df["margin_required_inr"] / capital * 100).round(2)
    df = df.sort_values("entry_date").reset_index(drop=True)
    df["running_pnl"] = df["pnl_total"].cumsum()
    df["equity"] = capital + df["running_pnl"]
    df["roi_pct"] = (df["running_pnl"] / capital * 100).round(4)
    df["peak"] = df["equity"].cummax()
    df["drawdown_pct"] = ((df["equity"] - df["peak"]) / df["peak"] * 100).round(4)
    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    return df


def _orders_csv(trades: pd.DataFrame, new_lots: int) -> pd.DataFrame:
    rows = []
    for idx, t in trades.iterrows():
        qty = int(t["lot"]) * new_lots
        for phase, pref in [("ENTRY", "entry"), ("EXIT", "exit")]:
            for leg, k_col, a_short, a_long in [
                ("ce_short", "ce_k", "SELL", "BUY"),
                ("pe_short", "pe_k", "SELL", "BUY"),
                ("wce_long", "wce_k", "BUY", "SELL"),
                ("wpe_long", "wpe_k", "BUY", "SELL"),
            ]:
                action = a_short if phase == "ENTRY" else a_long
                px_col = f"{leg.split('_')[0]}_{pref}_px"
                price = float(t.get(px_col, 0) or 0)
                sign = 1 if action == "BUY" else -1
                rows.append({
                    "trade_idx": idx, "entry_date": t["entry_date"],
                    "exit_date": t["exit_date"], "expiry": t["expiry"],
                    "spot_at_entry": t["spot"], "lot_size": int(t["lot"]),
                    "lots": new_lots, "qty_per_leg": qty,
                    "exit_reason": t["exit_reason"], "phase": phase,
                    "leg": leg, "action": action, "strike": int(t[k_col]),
                    "price": price, "qty": qty,
                    "value_inr": round(sign * price * qty, 2),
                    "margin_required_inr": float(t["margin_required_inr"]),
                })
    return pd.DataFrame(rows)


def _summary(trades: pd.DataFrame, capital: float) -> Dict:
    n = len(trades)
    wins = int((trades["pnl_total"] > 0).sum())
    total_pnl = float(trades["pnl_total"].sum())
    final_eq = float(trades["equity"].iloc[-1])
    days = (pd.to_datetime(trades["exit_date"].iloc[-1])
            - pd.to_datetime(trades["entry_date"].iloc[0])).days
    n_yrs = max(1.0, days / 365.25)
    cagr = ((final_eq / capital) ** (1 / n_yrs) - 1) * 100 if final_eq > 0 else -100.0
    return {
        "trades": n, "wins": wins, "wr_pct": round(wins / n * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_pnl / capital * 100, 2),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(float(trades["drawdown_pct"].min()), 2),
        "avg_margin_inr": round(float(trades["margin_required_inr"].mean()), 2),
        "peak_margin_inr": round(float(trades["margin_required_inr"].max()), 2),
        "capital_inr": capital,
        "lots": int(trades["lots"].iloc[0]),
        "n_years": round(n_yrs, 2),
    }


def main() -> int:
    summary_all: List[Dict] = []
    for u, otm, wing, stop, dow, slug in BEST:
        print(f"\n=== {u} OTM{otm} W{wing} dow={dow} ===")
        raw = run_ic(u, START, END, otm, wing, stop, 0.01,
                     capital=200_000, lots=1, realistic_slip=True,
                     min_leg_volume=100, entry_dow=dow)
        if raw.empty:
            print(f"  ! empty result")
            continue
        raw["month"] = pd.to_datetime(raw["entry_date"]).dt.to_period("M").astype(str)
        raw["year"] = pd.to_datetime(raw["entry_date"]).dt.year
        print(f"  raw trades = {len(raw)}")
        for cap in CAPITALS:
            lots = _peak_safe_lots(raw, cap)
            if lots <= 0:
                continue
            trades = _rescale(raw, lots, cap)
            orders = _orders_csv(trades, lots)
            cap_lbl = f"{int(cap/1000)}k"
            folder = EXPORTS / f"{slug}_lots{lots}_cap{cap_lbl}"
            folder.mkdir(parents=True, exist_ok=True)
            trades.to_csv(folder / "trades.csv", index=False)
            orders.to_csv(folder / "orders.csv", index=False)
            s = _summary(trades, cap)
            s["underlying"] = u
            s["variant_slug"] = slug
            s["otm_pct"] = otm
            s["wing_width"] = wing
            s["stop_mult"] = stop
            s["entry_dow"] = ["MON","TUE","WED","THU","FRI"][dow]
            s["min_leg_volume"] = 100
            (folder / "summary.json").write_text(json.dumps(s, indent=2))
            summary_all.append(s)
            print(f"  ₹{cap:,} lots={lots} CAGR={s['cagr_pct']:+.2f} %  "
                  f"DD={s['max_dd_pct']:+.2f} %  trades={s['trades']}  "
                  f"→ {folder.name}")

    # Roll-up
    if summary_all:
        roll = pd.DataFrame(summary_all)
        roll = roll.sort_values(["underlying", "capital_inr"])
        out = EXPORTS / "BEST_VARIANTS_ROLLUP.csv"
        roll.to_csv(out, index=False)
        print(f"\nRollup CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
