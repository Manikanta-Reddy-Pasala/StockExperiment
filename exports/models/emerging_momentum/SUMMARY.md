# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead). + DAILY ATR-from-entry hard stop (entry − 2.5× ATR(14)).

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-03-01 → 2026-05-31** (emerging → 2026-06-10; full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF. |
| **Entry** | BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1. |
| **Exit** | Rotate when held is no longer rank-1 (RET1). Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹44,462,575 |
| Total return | +4346.3% |
| CAGR (annualized) | +105.3% |
| Max drawdown | 38.6% |
| Calmar | 2.73 |
| Trades | 68 (43W / 25L) · 63% win |
| Total charges (real Fyers CNC, deducted) | ₹2,555,014 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +13.6% | 38.6% |
| 2022 | +173.7% | 22.1% |
| 2023 | +336.4% | 26.9% |
| 2024 | +140.6% | 26.1% |
| 2025 | +35.9% | 11.1% |
| 2026 | +3.4% | 27.4% |

## Note

Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe, PLUS a 2.5× ATR-from-entry hard stop. 2026-06-13 realism regen (net of real Fyers CNC charges, next-open fills): full-cycle 2021-03→2026-06 +105.3% CAGR / 38.6% DD / Calmar 2.73 / 68 trades / 63% win (charges ₹2.56M); 3-yr 2023-05→2026-05 +138.9% CAGR / 27.4% DD / Calmar 5.07 / 40 trades. (Old close-fill zero-charge convention had shown +115.6%/37.9%/3.05 — a normal ~10pp charges haircut.) ALL figures UNLEVERED (own cash only — no borrow); decision logic byte-identical to live (PIT, no-lookahead, backtest==live). The stop is a FIXED level at entry − 2.5×ATR (NOT trailing) so it cuts genuine breakdowns without whipsawing winners. Shared helper strategy.atr_stop_hit used by both backtest and the live --stop-check (no drift). The one model that crosses 100% organically (no leverage).

**Open position at window end:** HFCL qty 262921 entry ₹180.0 on 2026-06-02 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
