# midcap_narrow_60d_breakout (V2) — SUMMARY

**Mid+Small cap breakout swing. Exclude Large from pseudo-midcap pool.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-17 |
| Last exit | 2026-05-11 |
| Total trades | 12 |
| Trades per year | ~4.0 |
| Rebalance period | Event-driven (daily breakout scan) |
| Data source | yfinance (full N500 re-pull 2026-05-17, split-adjusted) |

## Headline result (CLEAN DATA - full N500 yfinance re-pull)

| Metric | Value |
|---|---:|
| Final NAV | **₹4,371,284** |
| Total return | **+337.13%** |
| **3-yr CAGR** | **+63.51%/yr** |
| Max DD (NAV-based) | 32.01% |
| Calmar | 1.98 |
| Trades | 12 |
| WR | 66.7% (8W / 4L) |

## Data quality journey

| Stage | CAGR | DD | Notes |
|---|---:|---:|---|
| Original Fyers (dirty data) | +86.63% | 15.15% | Inflated by KOTAKBANK +400% fake jump etc. |
| **Full N500 yfinance (this)** | **+63.51%** | **32.01%** | Clean, split-adjusted, honest |

Data fix: 504 N500 stocks re-pulled from yfinance with auto_adjust=True (handles splits/bonuses). 412,152 rows refreshed 2026-05-17. Only 11 anomalies remain (5 stocks: real demergers ABFRL/VEDL + yfinance Nov 2023 quirks for CGCL/GPIL/MOTILALOFS).

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,705,124 | **+70.51%** | 3 |
| 2024-25 | ₹1,705,124 | ₹1,861,734 | **+9.18%** | 4 |
| 2025-26 | ₹1,861,734 | ₹4,371,284 | **+134.80%** | 5 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 6 | 4 | 2 | 67% | +1,452,057 |
| **Small** | 6 | 4 | 2 | 67% | +1,919,466 |
