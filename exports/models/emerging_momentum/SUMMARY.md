# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility over the top-80-ADV mid/small pool; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead). + DAILY ATR-from-entry hard stop (entry − 2.5× ATR(14)).

**Universe:** Top-80 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-01-01 → 2026-05-29** (₹10L capital; full ~5.4-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-80 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF. |
| **Entry** | BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1. |
| **Exit** | Rotate when held is no longer rank-1 (RET1). Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹84,377,043 |
| Total return | +8337.7% |
| CAGR (annualized) | +127.2% |
| Max drawdown | 35.3% |
| Calmar | 3.6 |
| Trades | 64 (39W / 25L) · 61% win |
| Total charges (real Fyers CNC, deducted) | ₹3,660,194 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +8.4% | 35.3% |
| 2022 | +227.0% | 28.4% |
| 2023 | +557.2% | 19.5% |
| 2024 | +157.5% | 26.1% |
| 2025 | +19.3% | 24.9% |
| 2026 | +20.2% | 28.1% |

## Note

Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe, PLUS a 2.5× ATR-from-entry hard stop. 2026-06-13 POOL 100→80 re-tune (tools/research/emerging_improve.py): narrowing to the top-80 by ADV drops the lower-liquidity/jumpier tail that diluted compounding + inflated DD — beats pool-100 on EVERY window. Net of real Fyers CNC charges + next-open fills: full-cycle 2021-01→2026-05 +127.2% CAGR / 35.3% DD / Calmar 3.60 / 64 trades / 60.9% win (charges ₹3.66M, vs old pool-100 +110%/38.6%/2.86); 3-yr 2023-05→2026-05 +179.5% CAGR / 28.1% DD / Calmar 6.39 / 35 trades (vs +138.9%/5.07); since-Mar-2025 +56% / Calmar 2.01 (vs +46%/1.69). WF OOS-only beat baseline on BOTH axes; PLATEAU-confirmed (pool 80≈85, smooth shoulder — not a spike). (VOL_WIN 60→90 was tested and REJECTED: in-sample spike that fails the plateau check.) ALL figures UNLEVERED (own cash only — no borrow); decision logic byte-identical to live (PIT, no-lookahead, backtest==live). The stop is a FIXED level at entry − 2.5×ATR (NOT trailing). The one model that crosses 100% organically (no leverage).

**Open position at window end:** BHEL qty 202464 entry ₹379.9 on 2026-05-05 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
