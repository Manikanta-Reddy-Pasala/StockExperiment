# n20_daily_v2_large_only — SUMMARY

**Daily rotation. Top-20 ADV + uptrend + NSE Nifty 100 filter. Top-1 by 30d return.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-13 |
| Total trades | 140 |
| Trades per year | ~46.7 |
| Rebalance period | Daily |

## Headline result (CLEAN DATA, post yfinance restore)

| Metric | Value |
|---|---:|
| Final NAV | **₹14,691,291** |
| Total return | **+1369.13%** |
| **3-yr CAGR** | **+157.04%/yr** |
| Max DD (NAV-based) | 27.24% |
| Calmar | 5.76 |
| Trades | 140 |
| WR | 44.9% (62W / 76L) |

## Change vs prior result (dirty data)

| Metric | Prior (dirty) | **Clean** | Δ |
|---|---:|---:|---:|
| CAGR | +140.78% | **+157.04%** | +16.26pp |
| Max DD | 26.92% | **27.24%** | +0.32pp |

Prior result was inflated/distorted by data anomalies in KOTAKBANK, MCX, NUVAMA, VEDL (all jumped 4-5x on 2024-12-23 in raw Fyers data — incremental pull bug). Re-fetched 8 affected stocks via yfinance (split-adjusted) on 2026-05-17.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹5,989,612 | **+498.96%** | 37 |
| 2024-25 | ₹5,989,612 | ₹12,051,698 | **+101.21%** | 52 |
| 2025-26 | ₹12,051,698 | ₹14,691,291 | **+21.90%** | 51 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 140 | 62 | 76 | 45% | +13,691,287 |
