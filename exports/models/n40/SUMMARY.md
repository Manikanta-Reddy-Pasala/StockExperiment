# Weekly Top-40 (`n40`)

**Status:** LIVE  
WEEKLY rotation (first trading day of each ISO week), single position, uptrend gate.

**Universe:** Top-40 by ADV ∩ Nifty 100

Backtest window: **2021-03-01 → 2026-05-31** (emerging → 2026-06-10; full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | First trading day of each ISO week (WEEKLY). |
| **Universe & filters** | Top-40 by 20d ADV from N500, intersect PIT Nifty 100, and close > 200d SMA (uptrend). |
| **Entry** | On the weekly rebalance, BUY rank-1 by 30-day return among the filtered set (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1) at the next weekly rebalance, or when it drops out of Nifty 100 / below its 200d SMA. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` + `ind_nifty500list.csv` → nifty100.csv/nifty500.csv. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹3,711,081 |
| Total return | +271.1% |
| CAGR (annualized) | +28.4% |
| Max drawdown | 43.9% |
| Calmar | 0.65 |
| Trades | 138 (80W / 58L) · 58% win |
| Total charges (real Fyers CNC, deducted) | ₹468,281 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -4.0% | 27.5% |
| 2022 | +22.1% | 35.4% |
| 2023 | +9.0% | 21.4% |
| 2024 | +55.6% | 22.5% |
| 2025 | +37.8% | 21.9% |
| 2026 | +33.3% | 30.9% |

## Note

Weekly rebalance cut the daily whipsaw (55% of daily trades held ≤3d). 2026-06-13 realism regen (net of charges, next-open fills): full-cycle 2021-03→2026-05 +28.4% CAGR / 43.9% DD / Calmar 0.65 / 138 trades (charges ₹468,281); 3-yr 2023-05→2026-05 +48.6% CAGR / 30.9% DD / Calmar 1.58. (Old close-fill zero-charge convention had shown +48.1%/37.1%/1.30 — the ~20pp haircut is the realism convention compounding on the most churn-heavy weekly single-position rotation; no bug, trades 137→138, selection unchanged.) AUTHORITATIVE PIT Nifty-100 membership (2026-05-31 rebuild); per-year DD ≤36% — the 44% full-cycle is peak-to-trough chaining.

**Open position at window end:** ADANIENT qty 1263 entry ₹2846.0 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
