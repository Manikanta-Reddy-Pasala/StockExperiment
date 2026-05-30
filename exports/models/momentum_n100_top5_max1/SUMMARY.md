# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret). No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-15 |
| Total trades | 96 |
| Trades per year | ~32.0 |
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
| Final NAV (cap + open MTM) | **Rs.6,214,692** |
| Total return | **+521.47%** |
| 5.16-yr CAGR | **+42.50%** |
| Max DD | **43.53%** |
| Calmar (CAGR / Max DD) | **0.98** |
| Trades closed | 96 |
| Wins / Losses | 56 / 40 |
| Win rate | 58.3% |
| Live deployment | YES |
| Open position | **VEDL** qty 17,625 entry Rs.331.05 (2026-05-15) last Rs.352.60 unrealized +379,819 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 96 | 56 | 40 | 58% | +4,834,874 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIGREEN   | 2026-04-15 → 2026-05-15 | 1,096.05 | +25.82% | +1,197,161 |
| ADANIPOWER   | 2026-04-01 → 2026-04-15 | 157.11 | +16.75% | +665,449 |
| ATGL         | 2023-12-01 → 2024-01-01 | 701.45 | +42.70% | +481,596 |
| HINDZINC     | 2025-12-15 → 2026-01-16 | 568.05 | +12.27% | +447,683 |
| VEDL         | 2024-04-15 → 2024-05-15 | 138.78 | +18.04% | +425,079 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2024-06-03 → 2024-06-18 | 174.90 | -14.45% | -433,785 |
| ADANIENSOL   | 2024-08-01 → 2024-08-16 | 1,275.20 | -14.89% | -375,052 |
| IRFC         | 2024-07-15 → 2024-08-01 | 216.32 | -12.27% | -352,318 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -269,915 |
| INDUSINDBK   | 2025-05-02 → 2025-05-15 | 853.00 | -8.50% | -241,642 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
