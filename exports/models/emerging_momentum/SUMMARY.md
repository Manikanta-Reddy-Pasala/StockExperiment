# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) momentum rotation: 15d return >0, price ≤₹3000 (no SMA gate); retain top-3; monthly (1st trading day) + mid-month check that rotates only on a ≥5pp lead.

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100); 15d return > 0; price ≤ ₹3000; NO SMA gate; MCAP-CLIMBER overlay (keep only names whose mcap-rank is rising over 60d). |
| **Entry** | BUY rank-1 by 15-day return (single position, max 1). |
| **Exit** | Hold while in the top-3 by 15d return (RETAIN=3); rotate out when it drops below rank-3. Mid-month only rotates if the new rank-1 leads the held name by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹14,089,017 |
| Total return | +1308.9% |
| CAGR (annualized) | +65.6% |
| Max drawdown | 42.5% |
| Calmar | 1.54 |
| Trades | 46 (28W / 18L) · 61% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +14.5% | 27.7% |
| 2022 | +284.5% | 22.6% |
| 2023 | +127.9% | 22.0% |
| 2024 | +34.6% | 33.9% |
| 2025 | -6.5% | 34.8% |
| 2026 | +5.6% | 13.3% |

## Note

Single-position max-1, emerging mid/small, run_rotation engine + MCAP-CLIMBER filter (keep only rising-mcap-rank names). Full-cycle 2021-04→2026-05 on AUTHORITATIVE PIT membership (2026-05-31) ≈ +46.1% CAGR / 37.7% DD / Calmar 1.22. ⚠ The old +121% headline was a MIRAGE — the buggy Wayback N100 was missing large-cap winners (ADANIGREEN etc), so this 'mid/small' model wrongly held them. Correct N100 exclusion ⇒ genuinely mid/small ⇒ +46%.

**Open position at window end:** BHEL [mid] qty 33806 entry ₹377.05 on 2026-05-04 (unrealized +1,342,098)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
