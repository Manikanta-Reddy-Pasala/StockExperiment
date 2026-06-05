# Weekly Top-40 (`n40`)

**Status:** LIVE  
WEEKLY rotation (first trading day of each ISO week), single position, uptrend gate.

**Universe:** Top-40 by ADV ∩ Nifty 100

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | First trading day of each ISO week (WEEKLY). |
| **Universe & filters** | Top-40 by 20d ADV from N500, intersect PIT Nifty 100, and close > 200d SMA (uptrend). |
| **Entry** | On the weekly rebalance, BUY rank-1 by 30-day return among the filtered set (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1) at the next weekly rebalance, or when it drops out of Nifty 100 / below its 200d SMA. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` + `ind_nifty500list.csv` → nifty100.csv/nifty500.csv. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹7,847,232 |
| Total return | +684.7% |
| CAGR (annualized) | +48.1% |
| Max drawdown | 37.0% |
| Calmar | 1.3 |
| Trades | 137 (81W / 56L) · 59% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +9.8% | 26.8% |
| 2022 | +62.6% | 27.3% |
| 2023 | +12.4% | 24.0% |
| 2024 | +67.1% | 19.3% |
| 2025 | +59.6% | 15.8% |
| 2026 | +44.3% | 25.5% |

## Note

Weekly rebalance cut the daily whipsaw (55% of daily trades held ≤3d). Full-cycle 2021-03→2026-05 on the AUTHORITATIVE PIT Nifty-100 (2026-05-31 rebuild) ≈ +41.2% CAGR / 36.9% DD / Calmar 1.12 — the clean membership (no DUMMYREL/BHEL/IDEA garbage) lifted it from +25%/0.45. Recent 2025-03→2026-05 ≈ +96% CAGR / 24% DD. Per-year DD ≤27% every year — the 37% full-cycle is 2021→2022 peak-to-trough chaining.

**Open position at window end:** ADANIENT qty 2671 entry ₹2849.7 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
