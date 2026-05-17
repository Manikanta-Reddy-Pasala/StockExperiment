# momentum_n100_top5_max1 — SUMMARY

**LIVE production. Monthly rotation on Real NSE Nifty 100, narrowed to top-20 most-liquid by ADV. Rank by 30d return, hold top-1.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 28 |
| Trades per year | ~9.3 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers** (498/504 N500 incl. all N100, cont_flag=1) |

## Stock pick logic

1. **Universe**: REAL NSE Nifty 100 (104 stocks, `src/data/symbols/nifty100.csv`)
2. **Liquidity filter**: rank by 20-day ADV, keep top-20 most-liquid
3. **Rank by 30-day return**, pick top-1
4. **Rebalance monthly** (1st trading day)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV | **₹7,466,720** |
| Total return | **+646.67%** |
| **3-yr CAGR** | **+95.45%/yr** |
| Max DD (cash NAV) | 20.17% |
| Calmar | 4.73 |
| Trades | 28 |
| WR | 71.4% (20W / 8L) |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,336,509 | **+133.65%** | 10 |
| 2024-25 | ₹2,336,509 | ₹3,693,698 | **+58.09%** | 9 |
| 2025-26 | ₹3,693,698 | ₹7,466,720 | **+102.15%** | 9 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 28 | 20 | 8 | 71% | +6,466,719 |

## Improvement vs prior baseline

| Variant | CAGR | DD | Calmar |
|---|---:|---:|---:|
| Prior baseline (full N100, lb=30) | +64.17% | 37.30% | 1.72 |
| **Current (top-20 ADV)** | **+95.45%** | **20.17%** | **4.73** |

Single change: narrow Nifty 100 (104 stocks) to top-20 most-liquid by 20d ADV. CAGR +32pp, DD -17pp, Calmar 2.8x.