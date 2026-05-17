# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500). Monthly rotation, yearly-PIT universe.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-01 |
| Total trades | 29 |
| Trades per year | ~9.7 |
| Rebalance period | Monthly |
| Data source | yfinance (full N500 re-pull 2026-05-17, split-adjusted) |

## Headline result (CLEAN DATA - full N500 yfinance re-pull)

| Metric | Value |
|---|---:|
| Final NAV | **₹7,762,086** |
| Total return | **+676.21%** |
| **3-yr CAGR** | **+117.98%/yr** |
| Max DD (NAV-based) | 36.45% |
| Calmar | 3.24 |
| Trades | 29 |
| WR | 79.3% (23W / 6L) |

## Data quality journey

| Stage | CAGR | DD | Notes |
|---|---:|---:|---|
| Original Fyers (dirty data) | +136.39% | 16.15% | Inflated by KOTAKBANK +400% fake jump etc. |
| **Full N500 yfinance (this)** | **+117.98%** | **36.45%** | Clean, split-adjusted, honest |

Data fix: 504 N500 stocks re-pulled from yfinance with auto_adjust=True (handles splits/bonuses). 412,152 rows refreshed 2026-05-17. Only 11 anomalies remain (5 stocks: real demergers ABFRL/VEDL + yfinance Nov 2023 quirks for CGCL/GPIL/MOTILALOFS).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,728,227 | **+172.82%** | 9 |
| 2024-25 | ₹2,728,227 | ₹4,504,740 | **+65.12%** | 11 |
| 2025-26 | ₹4,504,740 | ₹7,762,086 | **+72.31%** | 9 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 13 | 12 | 1 | 92% | +4,891,801 |
| **Mid** | 13 | 9 | 4 | 69% | +2,053,512 |
| **Small** | 3 | 2 | 1 | 67% | -183,229 |
