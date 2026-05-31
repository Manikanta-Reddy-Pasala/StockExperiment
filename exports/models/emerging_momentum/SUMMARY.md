# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead). + DAILY ATR-from-entry hard stop (entry − 2.5× ATR(14)).

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
| Final NAV (₹10L start) | ₹60,024,422 |
| Total return | +5902.4% |
| CAGR (annualized) | +119.9% |
| Max drawdown | 37.9% |
| Calmar | 3.16 |
| Trades | 67 (43W / 24L) · 64% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +19.6% | 37.9% |
| 2022 | +150.4% | 20.9% |
| 2023 | +358.2% | 26.8% |
| 2024 | +171.4% | 26.3% |
| 2025 | +45.8% | 11.0% |
| 2026 | +14.7% | 24.6% |

## Note

Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe, PLUS a 2.5× ATR-from-entry hard stop (2026-06-01, backtest-validated both windows + every year): +119.9% CAGR / 37.9% DD / Calmar 3.16 / 64% win full-cycle 2021-03→2026-05; recent 2023-05→2026-05 +167.7% CAGR / 26.3% DD / 75% win (+27% / +37% net P&L vs the rotation-only baseline). The stop is a FIXED level at entry − 2.5×ATR (NOT trailing) so it cuts genuine breakdowns without whipsawing winners. Per-year: 2021 +20 / 2022 +150 / 2023 +358 / 2024 +171 / 2025 +46 / 2026 +15. Shared helper strategy.atr_stop_hit used by both backtest and the live --stop-check (no drift). The one model that crosses 100% organically (no leverage).

**Open position at window end:** BHEL qty 153221 entry ₹377.05 on 2026-05-04 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
