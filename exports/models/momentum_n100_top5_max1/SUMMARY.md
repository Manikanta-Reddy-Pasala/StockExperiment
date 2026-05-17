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
| Data source | yfinance (full N500 re-pull 2026-05-17, split-adjusted) |

## Headline result (CLEAN DATA - full N500 yfinance re-pull)

| Metric | Value |
|---|---:|
| Final NAV | **₹3,205,882** |
| Total return | **+220.59%** |
| **3-yr CAGR** | **+62.33%/yr** |
| Max DD (NAV-based) | 42.82% |
| Calmar | 1.46 |
| Trades | 30 |
| WR | 70.0% (21W / 9L) |

## Data quality journey

| Stage | CAGR | DD | Notes |
|---|---:|---:|---|
| Original Fyers (dirty data) | +80.38% | 29.71% | Inflated by KOTAKBANK +400% fake jump etc. |
| **Full N500 yfinance (this)** | **+62.33%** | **42.82%** | Clean, split-adjusted, honest |

Data fix: 504 N500 stocks re-pulled from yfinance with auto_adjust=True (handles splits/bonuses). 412,152 rows refreshed 2026-05-17. Only 11 anomalies remain (5 stocks: real demergers ABFRL/VEDL + yfinance Nov 2023 quirks for CGCL/GPIL/MOTILALOFS).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,437,605 | **+143.76%** | 10 |
| 2024-25 | ₹2,437,605 | ₹2,098,470 | **+-13.91%** | 10 |
| 2025-26 | ₹2,098,470 | ₹3,205,882 | **+52.77%** | 10 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 30 | 21 | 9 | 70% | +2,205,881 |
