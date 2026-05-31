# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead). **+ DAILY ATR-from-entry hard stop: entry − 2.5× ATR(14), checked every trading day (2026-06-01).**

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF. |
| **Entry** | BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1. |
| **Stop** | DAILY hard stop at `entry − 2.5× ATR(14)`. FIXED level anchored at entry (NOT trailing) → cuts genuine breakdowns, never whipsaws a winner. Shared helper `strategy.atr_stop_hit` used by BOTH backtest and the live `--stop-check` (no drift). |
| **Exit** | Rotate when held is no longer rank-1 (RET1), OR the ATR-from-entry stop fires. Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV (high/low for ATR). |

## Results (WITH 2.5× ATR-from-entry stop; daily-MTM, gross of fees)

| Metric | Full 2021-03→2026-05 | Recent 2023-05→2026-05 |
|---|---|---|
| CAGR | **+119.9%** | **+167.7%** |
| Max drawdown (daily-MTM) | 37.9% | 26.3% |
| Calmar | 3.16 | 6.38 |
| Trades | 67 (43W / 24L) · 64% win | 40 (30W / 10L) · 75% win |
| vs rotation-only baseline | **+27% net P&L** | **+37% net P&L** |

The 2.5× ATR-from-entry stop is the backtest-validated optimisation (vs the
rotation-only baseline +110% full / +143% recent): higher return at equal-or-lower
DD, positive every calendar year, validated on BOTH windows (not curve-fit). See
`tools/analysis/emerging_pricefloor_atr_sweep.py`.

## Year-by-year breakdown (with stop, full window)

| Year | Return % |
|---|---:|
| 2021 | +20% |
| 2022 | +150% |
| 2023 | +358% |
| 2024 | +171% |
| 2025 | +46% |
| 2026 | +15% |

## Note

Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe = +111.4% CAGR / 31% DD / Calmar 3.6 full-cycle 2021-03→2026-05 (2026-05-31). Dividing momentum by volatility picks smooth strong trends over jumpy ones and compounds far better (+65.6%→+111.4%); MCAP-climber turned OFF (pre-rebuild artifact). Per-year: 2021 −5 / 2022 +204 / 2023 +301 / 2024 +136 / 2025 +38 / 2026 +15. Recent 2025-03→2026-05 ≈ +48% CAGR / 16% DD. The one model that crosses 100% organically (no leverage).

**Open position at window end:** BHEL [mid] qty 121577 entry ₹377.05 on 2026-05-04 (unrealized +4,826,607)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
