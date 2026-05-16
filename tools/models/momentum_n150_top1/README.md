# momentum_n150_top1

N150 variant of `momentum_n100_top5_max1`. **NOT recommended for production** — N100 dominates on every metric.

## Strategy

| Knob | Value |
|---|---|
| Universe | Top-150 N500 stocks by 20d ADV, **rebuilt at year start (yearly PIT)** |
| Filter | Close > 200d SMA (uptrend gate) |
| Signal | Rank by 30-day return |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only |

Identical to `momentum_n100_top5_max1` except **universe size = 150 instead of 100**.

## Backtest result (PIT-honest, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹23,82,992 | **+138.30%** |
| Y2 (2024-05 → 2025-05) | ₹40,01,280 | **+67.91%** |
| Y3 (2025-05 → 2026-05) | ₹50,77,910 | **+26.91%** |
| **3-yr CAGR** | | **+71.88%** |
| Total return | | **+407.79%** |

32 round-trips · 81.2% WR · Max DD (cash NAV) 28.06%

## Apples-to-apples: N100 vs N150

| Metric | N100 (winner) | N150 (this) | Verdict |
|---|---:|---:|---|
| 3-yr CAGR | **+136.39%** | +71.88% | N100 +90% better |
| Final NAV | ₹1.32 Cr | ₹50.78 L | N100 2.6× higher |
| Max DD | **16.15%** | 28.06% | N100 lower risk |
| WR | 86.7% | 81.2% | N100 higher |
| Trades | 30 | 32 | Same |
| Worst loss | -16.17% (MCX) | -28.06% (RPOWER) | N100 less severe |

**Why N150 underperforms**: top-1 pick from larger universe = more noise candidates. Top 30-day return from 50 extra mid/small-caps (rank 101-150 by ADV) = noisier. N150-only traps that hurt this run:

- **RPOWER** (Jul 2025): -28.06% (₹14.5L loss). Not in N100. Volatile mid-cap.
- **BHARATFORG** (Mar 2026): -11.22% (₹6.2L loss). Not in N100.
- **TEJASNET** (Dec 2024): -12.69% (₹5.3L loss). Not in N100.

These 3 trades alone destroyed ~₹26L (about 50% of total NAV by Y3). With N100 universe, strategy avoided them entirely.

## Conclusion

**Universe-widening hypothesis falsified for this strategy.** Larger universe doesn't catch more winners — it catches more noise. The pseudo-N100 ADV-rank already includes the highest-quality mid-caps; ranks 101-150 add noise without signal.

**Stick with `momentum_n100_top5_max1` for production.**

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (N150 universe) |
| `live_signal.py` | Daily live signal (N150 universe) |
| `trade_ledger.json` | 32 trades + ledger |
| `yearly_universes.json` | N150 universes at each year start |
