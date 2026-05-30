# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret). No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-04-01 → 2026-05-29** (~5.16 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-15 |
| Total trades | 98 |
| Trades per year | ~19.0 |
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
| Final NAV (cap + open MTM) | **Rs.4,621,317** |
| Total return | **+362.13%** |
| 5.16-yr CAGR | **+34.55%** |
| Max DD | **52.18%** |
| Calmar (CAGR / Max DD) | **0.66** |
| Trades closed | 98 |
| Wins / Losses | 53 / 45 |
| Win rate | 54.1% |
| Live deployment | YES |
| Open position | **VEDL** qty 13,106 entry Rs.331.05 (2026-05-15) last Rs.352.60 unrealized +282,434 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 98 | 53 | 45 | 54% | +3,338,885 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIGREEN   | 2026-04-15 → 2026-05-15 | 1,096.05 | +25.82% | +890,161 |
| ADANIPOWER   | 2026-04-01 → 2026-04-15 | 157.11 | +16.75% | +494,842 |
| PAYTM        | 2024-08-16 → 2024-10-01 | 564.25 | +29.61% | +448,664 |
| GLAND        | 2023-08-01 → 2023-09-01 | 1,303.60 | +35.63% | +420,837 |
| HINDZINC     | 2025-12-15 → 2026-01-16 | 568.05 | +12.27% | +332,887 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIENSOL   | 2024-08-01 → 2024-08-16 | 1,275.20 | -14.89% | -264,910 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -200,725 |
| ADANIGREEN   | 2022-04-18 → 2022-05-16 | 2,970.50 | -23.19% | -185,976 |
| INDUSINDBK   | 2025-05-02 → 2025-05-15 | 853.00 | -8.50% | -181,105 |
| CGPOWER      | 2025-09-15 → 2025-10-01 | 791.35 | -6.51% | -177,074 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
