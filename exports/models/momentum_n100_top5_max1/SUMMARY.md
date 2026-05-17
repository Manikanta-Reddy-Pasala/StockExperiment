# momentum_n100_top5_max1 — SUMMARY

**LIVE production model. Monthly rotation on REAL NSE Nifty 100. Top-1 by 30d return.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-01 |
| Total trades | 30 |
| Trades per year | ~10.0 |
| Rebalance period | Monthly |

## Headline result (CLEAN DATA, post yfinance restore)

| Metric | Value |
|---|---:|
| Final NAV | **₹2,809,994** |
| Total return | **+181.00%** |
| **3-yr CAGR** | **+55.35%/yr** |
| Max DD (NAV-based) | 48.99% |
| Calmar | 1.13 |
| Trades | 30 |
| WR | 70.0% (21W / 9L) |

## Change vs prior result (dirty data)

| Metric | Prior (dirty) | **Clean** | Δ |
|---|---:|---:|---:|
| CAGR | +80.38% | **+55.35%** | -25.03pp |
| Max DD | 29.71% | **48.99%** | +19.28pp |

Prior result was inflated/distorted by data anomalies in KOTAKBANK, MCX, NUVAMA, VEDL (all jumped 4-5x on 2024-12-23 in raw Fyers data — incremental pull bug). Re-fetched 8 affected stocks via yfinance (split-adjusted) on 2026-05-17.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,416,397 | **+141.64%** | 10 |
| 2024-25 | ₹2,416,397 | ₹1,854,003 | **+-23.27%** | 10 |
| 2025-26 | ₹1,854,003 | ₹2,809,994 | **+51.56%** | 10 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 30 | 21 | 9 | 70% | +1,809,993 |
