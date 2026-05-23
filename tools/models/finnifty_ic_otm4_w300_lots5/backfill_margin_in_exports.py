"""Back-fill margin_required_inr column into existing FinNifty IC export CSVs.

The export folders under exports/models/finnifty_ic_otm{N}_w{W}_lots{L}/ were
generated before sweep.py emitted a margin column. trades.csv has all the
inputs we need (spot, strikes, per-leg entry prices, lot, lots) so we can
back-fill margin_required_inr without re-running the backtest.

Run from repo root:
    python3 -m tools.models.finnifty_ic_otm4_w300_lots5.backfill_margin_in_exports

Updates trades.csv + orders.csv in-place. Re-writes the Margin section in
SUMMARY.md (creates it if missing). Idempotent — running twice is safe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.models.finnifty_ic_otm4_w300_lots5.sweep import compute_ic_margin

EXPORTS = REPO_ROOT / "exports" / "models"
TARGETS = [
    ("finnifty_ic_otm2_w150_lots5", 200_000),
    ("finnifty_ic_otm4_w300_lots5", 200_000),
]


def _compute_for_row(row: pd.Series) -> float:
    return compute_ic_margin(
        ce_short_k=float(row["ce_k"]),
        pe_short_k=float(row["pe_k"]),
        wce_entry_px=float(row["wce_entry_px"]),
        wpe_entry_px=float(row["wpe_entry_px"]),
        lot_size=int(row["lot"]),
        lots=int(row["lots"]),
    )


def _patch_trades(folder: Path, capital: float) -> pd.DataFrame:
    p = folder / "trades.csv"
    if not p.exists():
        print(f"  [skip] {p} missing")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["margin_required_inr"] = df.apply(_compute_for_row, axis=1).round(2)
    df["margin_pct_of_capital"] = (df["margin_required_inr"] / capital * 100).round(1)
    df.to_csv(p, index=False)
    print(f"  wrote {p} ({len(df)} rows) — "
          f"avg margin ₹{df['margin_required_inr'].mean():,.0f}, "
          f"peak ₹{df['margin_required_inr'].max():,.0f}")
    return df


def _patch_orders(folder: Path, trades: pd.DataFrame) -> None:
    p = folder / "orders.csv"
    if not p.exists() or trades.empty:
        print(f"  [skip] {p} missing or no trades")
        return
    odf = pd.read_csv(p)
    # trades.csv has no trade_idx but rows are ordered identically.
    # orders.csv has trade_idx 0..N-1; align by index position.
    margin_by_idx = {i: float(m)
                     for i, m in enumerate(trades["margin_required_inr"])}
    odf["margin_required_inr"] = odf["trade_idx"].map(margin_by_idx)
    odf.to_csv(p, index=False)
    print(f"  wrote {p} ({len(odf)} rows) — margin stamped on every leg row")


SUMMARY_MARKER_START = "<!-- MARGIN-BLOCK-START -->"
SUMMARY_MARKER_END = "<!-- MARGIN-BLOCK-END -->"


def _patch_summary(folder: Path, trades: pd.DataFrame, capital: float) -> None:
    p = folder / "SUMMARY.md"
    if not p.exists() or trades.empty:
        print(f"  [skip] {p} missing or no trades")
        return
    text = p.read_text()
    avg = float(trades["margin_required_inr"].mean())
    peak = float(trades["margin_required_inr"].max())
    headroom = capital / avg if avg > 0 else 0.0
    block = (
        f"\n{SUMMARY_MARKER_START}\n"
        f"## Margin (SPAN+exposure approx)\n\n"
        f"Approximation calibrated to live Sensibull basket 2026-05-23 ±2 %. "
        f"See `compute_ic_margin` in `sweep.py` for the formula "
        f"(SPAN 2.9 % of short notional + 0.5 % exposure − long-wing credit).\n\n"
        f"| Metric | Value |\n"
        f"|---|---:|\n"
        f"| Avg margin / trade | ₹{avg:,.0f} |\n"
        f"| Peak margin / trade | ₹{peak:,.0f} |\n"
        f"| Configured capital | ₹{int(capital):,} |\n"
        f"| Capital / avg-margin ratio | {headroom:.2f}× |\n\n"
    )
    if avg > capital:
        block += (
            f"> ⚠️ **Margin required exceeds configured capital.** Avg margin "
            f"₹{avg:,.0f} > capital ₹{int(capital):,}. The backtest assumed "
            f"{int(trades['lots'].iloc[0])} lots could always be opened on "
            f"₹{int(capital/1000)}k capital — but the live broker will block "
            f"trades when funds are insufficient. Two ways to fix the gap:\n"
            f"> 1. **Increase capital** to ≥ ₹{int(peak*1.1):,} (≈ 1.1× peak "
            f"margin) so every trade has headroom.\n"
            f"> 2. **Reduce lots** to keep avg margin ≤ ~80 % of capital.\n\n"
        )
    block += f"{SUMMARY_MARKER_END}\n"

    if SUMMARY_MARKER_START in text and SUMMARY_MARKER_END in text:
        before = text.split(SUMMARY_MARKER_START)[0]
        after = text.split(SUMMARY_MARKER_END)[1]
        new_text = before.rstrip() + "\n" + block.strip() + "\n" + after
    else:
        # Insert before "## Files in this folder" if it exists, else append.
        anchor = "## Files in this folder"
        if anchor in text:
            new_text = text.replace(anchor, block.strip() + "\n\n" + anchor)
        else:
            new_text = text.rstrip() + "\n" + block
    p.write_text(new_text)
    print(f"  wrote {p} — Margin section refreshed")


def main() -> int:
    for model_dir, capital in TARGETS:
        folder = EXPORTS / model_dir
        if not folder.exists():
            print(f"[skip] {folder} missing")
            continue
        print(f"[{model_dir}] capital=₹{int(capital):,}")
        trades = _patch_trades(folder, capital)
        _patch_orders(folder, trades)
        _patch_summary(folder, trades, capital)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
