# n40 — SUMMARY

**Top-40 ADV + uptrend + Nifty 100. Weekly rotation top-1 by 30d ret. No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-25 |
| Total trades | 130 |
| Trades per year | ~43.3 |
| Rebalance | Weekly |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-40 by 20-day ADV from N500 (rebuilt each rebalance)
2. Uptrend filter: close > 200-day SMA
3. Large-cap filter: stock must be in NSE Nifty 100
4. Rank by 30-day return, pick top-1
5. Rebalance: first trading day of each ISO week (weekly cut whipsaw vs daily)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.3,156,599** |
| Total return | **+215.66%** |
| 5.16-yr CAGR | **+24.96%** |
| Max DD | **55.45%** |
| Calmar (CAGR / Max DD) | **0.45** |
| Trades closed | 130 |
| Wins / Losses | 68 / 62 |
| Win rate | 52.3% |
| Live deployment | NO |
| Open position | **ADANIENT** qty 1,074 entry Rs.2,849.70 (2026-05-25) last Rs.2,937.40 unrealized +94,190 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 130 | 68 | 62 | 52% | +2,062,411 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-06 → 2026-05-04 | 163.26 | +39.23% | +789,165 |
| HAL          | 2024-05-13 → 2024-06-03 | 3,921.75 | +34.47% | +436,664 |
| ETERNAL      | 2025-07-14 → 2025-09-08 | 270.60 | +21.82% | +408,626 |
| INDUSTOWER   | 2024-03-18 → 2024-05-13 | 248.50 | +32.11% | +308,746 |
| BEL          | 2025-05-19 → 2025-06-30 | 363.75 | +15.88% | +262,705 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-11-10 | 170.25 | -11.86% | -268,426 |
| ADANIPOWER   | 2024-06-03 → 2024-06-10 | 174.90 | -11.98% | -204,570 |
| ETERNAL      | 2024-12-09 → 2025-01-06 | 295.30 | -10.31% | -169,881 |
| IDEA         | 2021-12-06 → 2022-01-17 | 15.00 | -16.67% | -138,820 |
| BEL          | 2026-03-09 → 2026-03-16 | 457.35 | -6.09% | -127,358 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
