# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret). No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-15 |
| Total trades | 56 |
| Trades per year | ~18.7 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Rank by 30-day return, pick top-1
3. Rebalance: 1st trading day of month
4. Exit: rotation only — sell when not rank-1

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.6,563,631** |
| Total return | **+556.36%** |
| 2.99-yr CAGR | **+87.53%** |
| Max DD | **33.98%** |
| Calmar (CAGR / Max DD) | **2.58** |
| Trades closed | 56 |
| Wins / Losses | 36 / 20 |
| Win rate | 64.3% |
| Live deployment | YES |
| Open position | **ADANIGREEN** qty 5,018 entry Rs.1,096.05 (2026-04-15) last Rs.1,308.00 unrealized +1,063,565 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 46 | 27 | 19 | 59% | +2,934,136 |
| **Mid** | 9 | 8 | 1 | 89% | +1,331,863 |
| **Small** | 1 | 1 | 0 | 100% | +234,068 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-04-15 | 157.11 | +16.75% | +789,179 |
| ATGL         | 2023-12-01 → 2024-01-01 | 701.45 | +42.70% | +571,146 |
| HINDZINC     | 2025-12-15 → 2026-01-16 | 568.05 | +12.27% | +530,905 |
| VEDL         | 2024-04-15 → 2024-05-15 | 138.78 | +18.04% | +503,955 |
| BEL          | 2025-05-15 → 2025-06-16 | 350.40 | +15.25% | +470,307 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2024-06-03 → 2024-06-18 | 174.90 | -14.45% | -514,219 |
| ADANIENSOL   | 2024-08-01 → 2024-08-16 | 1,275.20 | -14.89% | -444,556 |
| IRFC         | 2024-07-15 → 2024-08-01 | 216.32 | -12.27% | -417,633 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -320,050 |
| INDUSINDBK   | 2025-05-02 → 2025-05-15 | 853.00 | -8.50% | -286,375 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
