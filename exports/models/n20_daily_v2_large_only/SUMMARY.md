# n20_daily_v2_large_only — SUMMARY

**Daily rotation. Top-20 ADV + uptrend + NSE Nifty 100 filter. Top-1 by 30d return.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-13 |
| Total trades | 138 |
| Trades per year | ~46.0 |
| Rebalance period | Daily |
| Data source | yfinance (full N500 re-pull 2026-05-17, split-adjusted) |

## Headline result (CLEAN DATA - full N500 yfinance re-pull)

| Metric | Value |
|---|---:|
| Final NAV | **₹14,164,868** |
| Total return | **+1316.49%** |
| **3-yr CAGR** | **+153.93%/yr** |
| Max DD (NAV-based) | 27.13% |
| Calmar | 5.67 |
| Trades | 138 |
| WR | 46.3% (63W / 73L) |

## Data quality journey

| Stage | CAGR | DD | Notes |
|---|---:|---:|---|
| Original Fyers (dirty data) | +140.78% | 26.92% | Inflated by KOTAKBANK +400% fake jump etc. |
| **Full N500 yfinance (this)** | **+153.93%** | **27.13%** | Clean, split-adjusted, honest |

Data fix: 504 N500 stocks re-pulled from yfinance with auto_adjust=True (handles splits/bonuses). 412,152 rows refreshed 2026-05-17. Only 11 anomalies remain (5 stocks: real demergers ABFRL/VEDL + yfinance Nov 2023 quirks for CGCL/GPIL/MOTILALOFS).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹6,044,612 | **+504.46%** | 44 |
| 2024-25 | ₹6,044,612 | ₹12,059,034 | **+99.50%** | 50 |
| 2025-26 | ₹12,059,034 | ₹14,164,868 | **+17.46%** | 44 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 138 | 63 | 73 | 46% | +13,164,872 |
