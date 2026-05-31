# Retest Momentum (`momentum_retest_n500`)

**Status:** DISABLED (₹0)  
Monthly top-4 (K=4), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.

**Universe:** Top-120 by 20d ADV from N500 (minus Smallcap-250)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Monthly (1st trading day) re-ranks the leaders; entry is scanned DAILY. |
| **Universe & filters** | Top-120 by 20d ADV from PIT N500 minus Smallcap-250 (incl large+mid, NOT N100-excluded); close > 200d SMA; price ≤ ₹3000; 30d return > 10% (mom floor); 10d return > 0 (accelerating). |
| **Entry** | Each month, watch the top-K=4 leaders; BUY one when its price sits within the retest band of the 20-EMA — between 20EMA×(1−1%) and 20EMA×(1+20%) — checked daily. Holds up to 4 equal-weight positions. |
| **Exit** | Rotate: SELL a holding at the monthly rebalance when it drops out of the top-4 by 30d return (RETAIN=4). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → nifty500.csv. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹10,736,058 |
| Total return | +973.6% |
| CAGR (annualized) | +57.3% |
| Max drawdown | 38.8% |
| Calmar | 1.48 |
| Trades | 182 (108W / 74L) · 59% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +21.0% | 23.6% |
| 2022 | +7.2% | 32.5% |
| 2023 | +139.7% | 21.3% |
| 2024 | +141.3% | 18.4% |
| 2025 | +14.5% | 22.4% |
| 2026 | +23.0% | 15.2% |

## Note

Multi-holding K=4 (2026-05-31 re-tune, was K2). Full-cycle 2021-03→2026-05 ≈ +57.3% CAGR / 38.8% DD / Calmar 1.48. Recent 2025-03→2026-05 ≈ +53% CAGR / 15% DD. K2→K4 diversified the basket: recent CAGR +38→+53, recent DD 21→15, and full DD 57→39 (per-year DD now ≤32 EVERY year: 2021 D24 / 2022 D32 / 2023 D21 / 2024 D18 / 2025 D22 / 2026 D15) — for only −7pt full CAGR. K-knee: K5/K6 decay. The old K2 (+64/57) concentrated into 2 names and chained a 57% peak-to-trough; K4 is the better risk-adjusted config. Wide 20% entry band keeps leaders that never pull back to the EMA.

**Open position at window end:** ADANIPOWER [large] qty 13959 entry ₹157.11 on 2026-04-01 (unrealized +1,204,103)

**Open position at window end:** ADANIGREEN [large] qty 1709 entry ₹1290.7 on 2026-05-04 (unrealized +315,652)

**Open position at window end:** BHEL [mid] qty 5849 entry ₹377.05 on 2026-05-04 (unrealized +232,205)

**Open position at window end:** ADANIENSOL [large] qty 1574 entry ₹1398.4 on 2026-05-04 (unrealized +180,853)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
