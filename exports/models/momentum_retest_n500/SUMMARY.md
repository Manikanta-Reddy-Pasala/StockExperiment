# Retest Momentum (`momentum_retest_n500`)

**Status:** DISABLED (₹0)  
Monthly top-2 (K=2), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.

**Universe:** Top-120 by 20d ADV from N500 (minus Smallcap-250)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Monthly (1st trading day) re-ranks the leaders; entry is scanned DAILY. |
| **Universe & filters** | Top-120 by 20d ADV from PIT N500 minus Smallcap-250 (incl large+mid, NOT N100-excluded); close > 200d SMA; price ≤ ₹3000; 30d return > 10% (mom floor); 10d return > 0 (accelerating). |
| **Entry** | Each month, watch the top-K=2 leaders; BUY one when its price sits within the retest band of the 20-EMA — between 20EMA×(1−1%) and 20EMA×(1+20%) — checked daily. Holds up to 2 equal-weight positions. |
| **Exit** | Rotate: SELL a holding at the monthly rebalance when it drops out of the top-4 by 30d return (RETAIN=4). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → nifty500.csv. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹13,473,665 |
| Total return | +1247.4% |
| CAGR (annualized) | +64.2% |
| Max drawdown | 57.1% |
| Calmar | 1.12 |
| Trades | 88 (48W / 40L) · 54% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +4.0% | 34.1% |
| 2022 | -2.0% | 46.5% |
| 2023 | +176.0% | 25.3% |
| 2024 | +253.6% | 16.5% |
| 2025 | +12.6% | 20.9% |
| 2026 | +20.9% | 19.4% |

## Note

Multi-holding K=2 (2026-05-30 sweep). Now PIT N500 (2026-05-31): full-cycle 2021-03→2026-05 ≈ +64.2% CAGR / 57.1% DD / Calmar 1.12. Recent 2025-03→2026-05 ≈ +38% CAGR / 21% DD. The 57% full-cycle DD is 2021 retest-entry chop (D34) + 2022 bear (D46) chained; 2023-26 each ≤25%. The earlier +77.8%/2.08 was survivorship-inflated (static current N500); switching to PIT eligible_at deflated CAGR and roughly doubled DD — the honest number. K=2 + wide 20% entry band (vs K3/band-8%) stops missing leaders that never pull back to the EMA.

**Open position at window end:** ADANIPOWER [large] qty 31614 entry ₹157.11 on 2026-04-01 (unrealized +2,727,024)

**Open position at window end:** ADANIGREEN [large] qty 3922 entry ₹1290.7 on 2026-05-04 (unrealized +724,393)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
