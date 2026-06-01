# Nifty 100 Momentum (`momentum_n100_top5_max1`)

**Status:** LIVE  
Monthly rotation + mid-month check, single position (max 1), 15-trading-day return rank. + DAILY from-entry FIXED −12% hard stop (entry × 0.88).

**Universe:** Real NSE Nifty 100 (PIT membership)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Real point-in-time NSE Nifty 100 (eligible_at). No price/SMA filter — pure index membership. |
| **Entry** | BUY rank-1 by 15-day return (single position, max 1). |
| **Exit** | Hold while in the top-3 by 15d return (RETAIN=3); rotate out when it drops below rank-3, or leaves the index. Mid-month only rotates if the new rank-1 leads the held name by ≥ 5pp. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` → nifty100.csv → n100_current.json. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹11,756,416 |
| Total return | +1075.6% |
| CAGR (annualized) | +59.9% |
| Max drawdown | 46.4% |
| Calmar | 1.29 |
| Trades | 97 (54W / 43L) · 56% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -22.8% | 36.2% |
| 2022 | +41.5% | 27.8% |
| 2023 | +166.5% | 20.8% |
| 2024 | +71.2% | 19.8% |
| 2025 | +47.9% | 16.6% |
| 2026 | +58.1% | 15.0% |

## Note

True-index version — the trustworthy-clean momentum benchmark. NOW with a from-entry fixed −12% hard stop (2026-06-02, backtest-validated): full-cycle 2021-03→2026-05 +59.9% CAGR / 46.4% DD / Calmar 1.29; recent 2025-03→2026-05 +111% CAGR / 15% DD; 2022-23 crash +95.7% CAGR / 27.8% DD — the stop's big win (was 68.7/42.9). Stop = entry×(1−0.12), checked daily on the low; shared backtest+live helper tools.shared.stops (no drift). Fixed-% fits these large-caps (uniform vol); ATR was DD-only, a price-floor threshold-fragile. DD now DAILY-MTM (stricter than the old rebal-snapshot; within-basis delta 56.8→46.4).

**Open position at window end:** NSE:VEDL-EQ qty 33342 entry ₹331.05 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
