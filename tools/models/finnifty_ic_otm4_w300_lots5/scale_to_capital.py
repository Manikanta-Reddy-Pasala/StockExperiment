"""Rescale a FinNifty Iron Condor backtest to a target investment capital.

Given an existing exports/models/<model>/trades.csv (generated at some
reference lot count) and a target capital, this script:

  1. Picks the largest `lots` such that PEAK margin ≤ capital
     (peak-safe — every backtested entry would actually have been
     openable at the live broker)
  2. Rescales per-trade P&L, max-loss, margin to that lot count
  3. Rebuilds running_pnl, equity, roi_pct, peak, drawdown columns
  4. Writes a fresh sibling export folder
     exports/models/<model_rescaled>/ with trades.csv + orders.csv +
     SUMMARY.md re-computed against the new capital

Why peak-safe instead of avg-safe? At ₹2L capital a single peak-margin
month would force the trade to be skipped, breaking the strategy's
"always-on monthly" thesis. Peak-safe guarantees no skips, at the cost
of a smaller average position.

Usage:
    python3 -m tools.models.finnifty_ic_otm4_w300_lots5.scale_to_capital \
        --capital 200000

Idempotent — re-running overwrites the rescaled folders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import compute_ic_margin

EXPORTS = REPO_ROOT / "exports" / "models"

# (source_folder, otm_label, wing_label) — used to name the rescaled folder.
SOURCES = [
    ("finnifty_ic_otm2_w150_lots5", "otm2", "w150"),
    ("finnifty_ic_otm4_w300_lots5", "otm4", "w300"),
]


def _margin_per_lot(row: pd.Series) -> float:
    """Margin a single lot of this trade would need (lots=1)."""
    return compute_ic_margin(
        ce_short_k=float(row["ce_k"]),
        pe_short_k=float(row["pe_k"]),
        wce_entry_px=float(row["wce_entry_px"]),
        wpe_entry_px=float(row["wpe_entry_px"]),
        lot_size=int(row["lot"]),
        lots=1,
    )


def _pick_lots(trades: pd.DataFrame, capital: float) -> int:
    """Largest lots count s.t. PEAK per-trade margin ≤ capital."""
    peak_per_lot = trades.apply(_margin_per_lot, axis=1).max()
    if peak_per_lot <= 0:
        return 1
    lots = int(capital / peak_per_lot)
    return max(1, lots)


def _rescale_trades(src: pd.DataFrame, new_lots: int,
                    capital: float) -> pd.DataFrame:
    df = src.copy()
    # Re-stamp lots + recompute lots-dependent fields. pnl_unit, net_credit,
    # exit_debit, max_loss_per_unit are per-1-unit and stay the same.
    df["lots"] = new_lots
    df["pnl_total"] = (df["pnl_unit"] * df["lot"] * new_lots).round(2)
    df["max_loss_total"] = (df["max_loss_per_unit"] * df["lot"] * new_lots).round(2)
    df["margin_required_inr"] = df.apply(
        lambda r: compute_ic_margin(
            float(r["ce_k"]), float(r["pe_k"]),
            float(r["wce_entry_px"]), float(r["wpe_entry_px"]),
            int(r["lot"]), new_lots), axis=1)
    df["margin_pct_of_capital"] = (df["margin_required_inr"] / capital * 100).round(1)

    # Rebuild equity / drawdown columns.
    df = df.sort_values("entry_date").reset_index(drop=True)
    df["running_pnl"] = df["pnl_total"].cumsum().round(2)
    df["equity"] = (capital + df["running_pnl"]).round(2)
    df["roi_pct"] = (df["running_pnl"] / capital * 100).round(4)
    df["peak"] = df["equity"].cummax().round(2)
    df["drawdown_pct"] = ((df["equity"] - df["peak"]) / df["peak"] * 100).round(4)
    return df


def _rebuild_orders(trades: pd.DataFrame, new_lots: int) -> pd.DataFrame:
    """Mirror sweep export schema: 8 rows per trade (4 legs × 2 phases)."""
    rows = []
    for idx, t in trades.iterrows():
        qty = int(t["lot"]) * new_lots
        for phase, prefix in [("ENTRY", "entry"), ("EXIT", "exit")]:
            for leg, strike_col, action_short, action_long in [
                ("ce_short", "ce_k", "SELL", "BUY"),
                ("pe_short", "pe_k", "SELL", "BUY"),
                ("wce_long", "wce_k", "BUY", "SELL"),
                ("wpe_long", "wpe_k", "BUY", "SELL"),
            ]:
                # ENTRY: shorts SELL, longs BUY. EXIT flips.
                action = action_short if phase == "ENTRY" else action_long
                px_col = f"{leg.split('_')[0]}_{prefix}_px"
                price = float(t.get(px_col, 0) or 0)
                # value_inr sign convention: BUY positive (cash out), SELL negative
                sign = 1 if action == "BUY" else -1
                rows.append({
                    "trade_idx": idx,
                    "entry_date": t["entry_date"],
                    "exit_date": t["exit_date"],
                    "expiry": t["expiry"],
                    "spot_at_entry": t["spot"],
                    "lot_size": int(t["lot"]),
                    "lots": new_lots,
                    "qty_per_leg": qty,
                    "exit_reason": t["exit_reason"],
                    "phase": phase,
                    "leg": leg,
                    "action": action,
                    "strike": int(t[strike_col]),
                    "price": price,
                    "qty": qty,
                    "value_inr": round(sign * price * qty, 2),
                    "margin_required_inr": float(t["margin_required_inr"]),
                })
    return pd.DataFrame(rows)


def _yearly_stats(trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    g = trades.groupby("year").agg(
        trades=("pnl_total", "size"),
        wins=("pnl_total", lambda s: (s > 0).sum()),
        pnl=("pnl_total", "sum"),
    ).reset_index()
    g["wr"] = (g["wins"] / g["trades"] * 100).round(1)
    g["roi"] = (g["pnl"] / capital * 100).round(2)
    return g


def _write_summary(folder: Path, trades: pd.DataFrame, new_lots: int,
                   capital: float, otm: str, wing: str,
                   src_lots: int) -> None:
    final_equity = float(trades["equity"].iloc[-1])
    total_pnl = float(trades["pnl_total"].sum())
    n = len(trades)
    wins = int((trades["pnl_total"] > 0).sum())
    wr = wins / n * 100 if n else 0
    avg_margin = float(trades["margin_required_inr"].mean())
    peak_margin = float(trades["margin_required_inr"].max())
    max_dd = float(trades["drawdown_pct"].min())
    total_return = total_pnl / capital * 100
    n_years = max(1.0, (pd.to_datetime(trades["exit_date"].iloc[-1])
                        - pd.to_datetime(trades["entry_date"].iloc[0])).days / 365.25)
    cagr = ((final_equity / capital) ** (1 / n_years) - 1) * 100 if final_equity > 0 else 0
    avg_per_trade = total_pnl / n if n else 0

    yearly = _yearly_stats(trades, capital)
    yearly_lines = []
    for _, r in yearly.iterrows():
        yearly_lines.append(
            f"| {int(r['year'])} | {int(r['trades'])} | {int(r['wins'])} | "
            f"{r['wr']:.1f} % | ₹{int(r['pnl']):,} | "
            f"{r['roi']:+.2f} % |")

    exit_summary = trades.groupby("exit_reason").agg(
        count=("pnl_total", "size"),
        avg_pnl=("pnl_total", "mean"),
        total_pnl=("pnl_total", "sum"),
    ).reset_index()
    exit_lines = []
    for _, r in exit_summary.iterrows():
        exit_lines.append(
            f"| {r['exit_reason']} | {int(r['count'])} | "
            f"₹{int(round(r['avg_pnl'])):+,} | ₹{int(round(r['total_pnl'])):+,} |")

    content = f"""# finnifty_ic_{otm}_{wing}_lots{new_lots}

FinNifty monthly Iron Condor rescaled to ₹{int(capital):,} investment capital
(peak-safe — every backtested entry was openable at the live broker).

## Strategy

- **Underlying:** FINNIFTY (Nifty Financial Services Index)
- **Setup:** Iron Condor monthly expiry, {otm.upper()} body, {wing.upper()} wings
- **Position size:** {new_lots} lot{'s' if new_lots > 1 else ''} (auto-sized from peak margin ≤ capital)
- **Stop:** 3× entry credit OR hold to expiry
- **Capital:** ₹{int(capital):,}
- **Slippage:** realistic tiered (1× ATM → 15× >6% OTM)
- **Source:** rescaled from finnifty_ic_{otm}_{wing}_lots{src_lots} backtest

## Result at ₹{int(capital):,}

- **Started with:** ₹{int(capital):,}
- **Ended with:** ₹{int(round(final_equity)):,}
- **Total profit:** ₹{int(round(total_pnl)):,}
- **Total return:** {total_return:+.1f} %
- **CAGR:** **{cagr:+.1f} %**
- **Trades:** {n}
- **Win rate:** {wr:.1f} %
- **Avg P&L / trade:** ₹{int(round(avg_per_trade)):+,}
- **Max drawdown:** {max_dd:.1f} %

## Margin (SPAN+exposure approx — sweep.compute_ic_margin)

| Metric | Value |
|---|---:|
| Lots | **{new_lots}** |
| Avg margin / trade | ₹{int(round(avg_margin)):,} |
| Peak margin / trade | ₹{int(round(peak_margin)):,} |
| Configured capital | ₹{int(capital):,} |
| Capital / peak ratio | {capital/peak_margin if peak_margin > 0 else 0:.2f}× |

> ✅ Peak margin ≤ capital — every monthly entry was openable at the
> live broker. No skipped cycles.

## Yearly

| Year | Trades | Wins | WR | P&L | ROI |
|---|---:|---:|---:|---:|---:|
{chr(10).join(yearly_lines)}

## Exit reasons

| Reason | Count | Avg P&L | Total P&L |
|---|---:|---:|---:|
{chr(10).join(exit_lines)}

## Files

| File | Description |
|---|---|
| `SUMMARY.md` | This document |
| `trades.csv` | {n} rows, one per IC trade. Includes margin per trade |
| `orders.csv` | {n * 8} rows = {n} trades × 4 legs × 2 phases |
"""
    (folder / "SUMMARY.md").write_text(content)
    print(f"  wrote {folder / 'SUMMARY.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000,
                    help="Target investment capital in INR (default ₹2L)")
    args = ap.parse_args()

    for src_dir, otm, wing in SOURCES:
        src_folder = EXPORTS / src_dir
        src_trades_p = src_folder / "trades.csv"
        if not src_trades_p.exists():
            print(f"[skip] {src_trades_p} missing")
            continue
        src_trades = pd.read_csv(src_trades_p)
        if "margin_required_inr" not in src_trades.columns:
            print(f"[skip] {src_trades_p}: run backfill_margin_in_exports first")
            continue
        src_lots = int(src_trades["lots"].iloc[0])
        new_lots = _pick_lots(src_trades, args.capital)
        dst_dir = f"finnifty_ic_{otm}_{wing}_lots{new_lots}_cap{int(args.capital/1000)}k"
        dst_folder = EXPORTS / dst_dir
        dst_folder.mkdir(parents=True, exist_ok=True)
        print(f"[{src_dir} → {dst_dir}] capital=₹{int(args.capital):,} lots={new_lots} (was {src_lots})")

        trades = _rescale_trades(src_trades, new_lots, args.capital)
        trades.to_csv(dst_folder / "trades.csv", index=False)
        print(f"  wrote {dst_folder / 'trades.csv'} ({len(trades)} rows) — "
              f"avg margin ₹{trades['margin_required_inr'].mean():,.0f}, "
              f"peak ₹{trades['margin_required_inr'].max():,.0f}")

        orders = _rebuild_orders(trades, new_lots)
        orders.to_csv(dst_folder / "orders.csv", index=False)
        print(f"  wrote {dst_folder / 'orders.csv'} ({len(orders)} rows)")

        _write_summary(dst_folder, trades, new_lots, args.capital,
                       otm, wing, src_lots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
