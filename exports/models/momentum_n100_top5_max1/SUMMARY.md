# Nifty 100 Momentum (`momentum_n100_top5_max1`)

**Status:** LIVE  
Monthly rotation + mid-month check, single position (max 1), 15-trading-day return rank.

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
| Final NAV (₹10L start) | ₹10,376,511 |
| Total return | +937.6% |
| CAGR (annualized) | +56.2% |
| Max drawdown | 52.2% |
| Calmar | 1.08 |
| Trades | 96 (53W / 43L) · 55% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -15.8% | 27.1% |
| 2022 | +10.6% | 38.7% |
| 2023 | +182.4% | 11.3% |
| 2024 | +89.2% | 21.3% |
| 2025 | +37.2% | 13.1% |
| 2026 | +48.4% | 8.4% |

## Note

True-index version — the trustworthy-clean momentum benchmark. Full-cycle 2021-03→2026-05 ≈ +56.2% CAGR / 52.2% DD / Calmar 1.08. Recent 2025-03→2026-05 ≈ +112% CAGR / 9.6% DD. The 52% full-cycle DD is entirely the 2022 bear (D39); 2023-26 each ≤21%.

**Open position at window end:** VEDL [large] qty 29428 entry ₹331.05 on 2026-05-15 (unrealized +634,173)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
