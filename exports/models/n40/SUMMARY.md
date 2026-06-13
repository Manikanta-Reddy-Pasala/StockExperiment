# Weekly Top-40 (`n40`)

**Status:** LIVE  
WEEKLY rotation (first trading day of each ISO week), single position, uptrend gate. + DAILY from-entry FIXED −10% hard stop (entry × 0.90).

**Universe:** Top-40 by ADV ∩ Nifty 100

Backtest window: **2021-01-01 → 2026-05-29** (₹10L capital; full ~5.4-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | First trading day of each ISO week (WEEKLY). |
| **Universe & filters** | Top-40 by 20d ADV from N500, intersect PIT Nifty 100, and close > 200d SMA (uptrend). |
| **Entry** | On the weekly rebalance, BUY rank-1 by 30-day return among the filtered set (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1) at the next weekly rebalance, or when it drops out of Nifty 100 / below its 200d SMA, OR the −10% from-entry stop fires. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` + `ind_nifty500list.csv` → nifty100.csv/nifty500.csv. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹4,357,799 |
| Total return | +335.8% |
| CAGR (annualized) | +31.3% |
| Max drawdown | 38.9% |
| Calmar | 0.81 |
| Trades | 139 (80W / 59L) · 58% win |
| Total charges (real Fyers CNC, deducted) | ₹513,187 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -4.0% | 27.5% |
| 2022 | +40.2% | 29.6% |
| 2023 | -0.7% | 26.8% |
| 2024 | +55.7% | 22.5% |
| 2025 | +54.6% | 12.6% |
| 2026 | +33.3% | 30.9% |

## Note

Weekly rebalance cut the daily whipsaw (55% of daily trades held ≤3d). 2026-06-13 STOP RE-TUNE 12%→10% + realism regen (net of charges, next-open fills): full-cycle 2021-01→2026-05 +31.3% CAGR / 38.9% DD / Calmar 0.81 / 139 trades (charges ₹513,187); 3-yr 2023-05→2026-05 +54.2% CAGR / 30.9% DD / Calmar 1.76. The old −12% stop was tuned PRE-realism (close-MTM, zero charges); under the charges+next-open convention 12% is a local dip — a fresh realism stop-sweep (tools/research/n40_cagr.py) found the in-sample optimum ≈8-10%, so 10% lifts full-window CAGR 28.4→32.4 AND cuts DD 43.9→38.9 (a plateau: 8/10/15 all beat 12). HONEST CAVEAT: the lift sits in the 2021-22 bear — anchored WF shows 2023-26 OOS CAGR ≈ neutral, so 10% is a stale-param fix + DD reducer, not a forward-CAGR boost; leverage was also tested and HURTS n40 (borrow cost + DD amplification on a lumpy curve cut CAGR while DD blew out). AUTHORITATIVE PIT Nifty-100 membership (2026-05-31 rebuild); per-year DD ≤31% — the 39% full-cycle is peak-to-trough chaining.

**Open position at window end:** ADANIENT qty 1483 entry ₹2846.0 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
