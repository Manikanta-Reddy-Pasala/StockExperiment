# Nifty 100 Momentum (`momentum_n100_top5_max1`)

**Status:** LIVE  
Monthly rotation + mid-month check, single position (max 1), 15-trading-day return rank. + DAILY from-entry FIXED −12% hard stop (entry × 0.88).

**Universe:** Real NSE Nifty 100 (PIT membership)

Backtest window: **2021-03-01 → 2026-06-12** (₹10L capital; full ~5.3-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Real point-in-time NSE Nifty 100 (eligible_at). No price/SMA filter — pure index membership. |
| **Entry** | BUY rank-1 by 15-day return (single position, max 1). |
| **Exit** | Hold while in the top-3 by 15d return (RETAIN=3); rotate out when it drops below rank-3, or leaves the index. Mid-month only rotates if the new rank-1 leads the held name by ≥ 5pp. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` → nifty100.csv → n100_current.json. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹8,366,534 |
| Total return | +736.6% |
| CAGR (annualized) | +49.5% |
| Max drawdown | 49.1% |
| Calmar | 1.01 |
| Trades | 98 (54W / 44L) · 55% win |
| Total charges (real Fyers CNC, deducted) | ₹604,171 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -27.1% | 39.1% |
| 2022 | +44.4% | 25.6% |
| 2023 | +129.8% | 20.8% |
| 2024 | +59.6% | 20.7% |
| 2025 | +40.8% | 17.0% |
| 2026 | +52.6% | 14.6% |

## Note

True-index version — the trustworthy-clean momentum benchmark. 2026-06-13 realism regen (net of charges, next-open fills): full-cycle 2021-03→2026-06 +49.5% CAGR / 49.1% DD / Calmar 1.01 / 98 trades (charges ₹604,171); 3-yr 2023-05→2026-05 +80.5% CAGR / 20.7% DD / Calmar 3.88. (Old close-fill zero-charge convention had shown +59.9%/46.4%/1.29 — a normal ~8pp charges+slippage haircut.) From-entry fixed −12% hard stop (2026-06-02): entry×(1−0.12), checked daily on the low; shared backtest+live helper tools.shared.stops (no drift). Fixed-% fits these large-caps (uniform vol). DD is DAILY-MTM (stricter than the old rebal-snapshot basis).

**Open position at window end:** NSE:ENRIN-EQ qty 2319 entry ₹3713.1 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
