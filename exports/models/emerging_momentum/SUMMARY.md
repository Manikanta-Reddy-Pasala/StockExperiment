# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead).

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF. |
| **Entry** | BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1. |
| **Exit** | Rotate when held is no longer rank-1 (RET1). Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹50,667,400 |
| Total return | +4966.7% |
| CAGR (annualized) | +111.4% |
| Max drawdown | 31.2% |
| Calmar | 3.57 |
| Trades | 61 (41W / 20L) · 67% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -4.6% | 32.6% |
| 2022 | +204.4% | 14.8% |
| 2023 | +301.0% | 22.0% |
| 2024 | +136.1% | 12.0% |
| 2025 | +38.0% | 16.2% |
| 2026 | +14.8% | 11.2% |

## Note

Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe = +111.4% CAGR / 31% DD / Calmar 3.6 full-cycle 2021-03→2026-05 (2026-05-31). Dividing momentum by volatility picks smooth strong trends over jumpy ones and compounds far better (+65.6%→+111.4%); MCAP-climber turned OFF (pre-rebuild artifact). Per-year: 2021 −5 / 2022 +204 / 2023 +301 / 2024 +136 / 2025 +38 / 2026 +15. Recent 2025-03→2026-05 ≈ +48% CAGR / 16% DD. The one model that crosses 100% organically (no leverage).

**Open position at window end:** BHEL [mid] qty 121577 entry ₹377.05 on 2026-05-04 (unrealized +4,826,607)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
