# Retest Momentum (`momentum_retest_n500`)

**Status:** DISABLED (₹0)  
Monthly top-4 (K=4), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.

**Universe:** Top-120 by 20d ADV from N500 (minus Smallcap-250)

Backtest window: **2021-01-01 → 2026-05-29** (₹10L capital; full ~5.4-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Monthly (1st trading day) re-ranks the leaders; entry is scanned DAILY. |
| **Universe & filters** | Top-120 by 20d ADV from PIT N500 minus Smallcap-250 (incl large+mid, NOT N100-excluded); close > 200d SMA; price ≤ ₹3000; 30d return > 10% (mom floor); 10d return > 0 (accelerating). |
| **Entry** | Each month, watch the top-K=4 leaders; BUY one when its price sits within the retest band of the 20-EMA — between 20EMA×(1−1%) and 20EMA×(1+20%) — checked daily. Holds up to 4 equal-weight positions. |
| **Exit** | Rotate: SELL a holding at the monthly rebalance when it drops out of the top-4 by 30d return (RETAIN=4). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → nifty500.csv. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹11,292,990 |
| Total return | +1029.3% |
| CAGR (annualized) | +56.6% |
| Max drawdown | 34.0% |
| Calmar | 1.66 |
| Trades | 183 (110W / 72L) · 60% win |
| Total charges (real Fyers CNC, deducted) | ₹386,643 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +24.5% | 25.6% |
| 2022 | +13.1% | 32.8% |
| 2023 | +121.6% | 21.9% |
| 2024 | +141.1% | 18.3% |
| 2025 | +15.5% | 22.7% |
| 2026 | +26.7% | 12.4% |

## Note

Multi-holding K=4 (2026-05-31 re-tune, was K2). 2026-06-13 realism regen (net of charges, next-open fills, PIT-before-ADV universe fix): full-cycle 2021-01→2026-05 +56.6% CAGR / 34.0% DD / Calmar 1.66 / 183 trades (charges ₹386,643); 3-yr 2023-05→2026-05 +102.3% CAGR / 23.6% DD / Calmar 4.34. (Old flat-0.15%/side close-fill convention had shown +57.3%/38.8%/1.48 — the PIT-before-ADV fix was net-positive, the only model to IMPROVE under realism.) K2→K4 diversified the basket (per-year DD ≤33 every year); K-knee: K5/K6 decay. Wide 20% entry band keeps leaders that never pull back to the EMA.

**Open position at window end:** ADANIPOWER [large] qty 14808 entry ₹154.5 on 2026-04-02 (unrealized +1,315,987)

**Open position at window end:** ADANIGREEN [large] qty 1791 entry ₹1291.0 on 2026-05-05 (unrealized +330,260)

**Open position at window end:** BHEL [mid] qty 6087 entry ₹379.9 on 2026-05-05 (unrealized +224,306)

**Open position at window end:** ADANIENSOL [large] qty 1658 entry ₹1394.6 on 2026-05-05 (unrealized +196,805)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
