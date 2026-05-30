# n40 — SUMMARY

**Top-40 ADV + uptrend + Nifty 100. Weekly rotation top-1 by 30d ret. No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 71 |
| Trades per year | ~23.7 |
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
| Final NAV (cap + open MTM) | **Rs.4,030,205** |
| Total return | **+303.02%** |
| 2.99-yr CAGR | **+59.32%** |
| Max DD | **24.44%** |
| Calmar (CAGR / Max DD) | **2.43** |
| Trades closed | 71 |
| Wins / Losses | 40 / 31 |
| Win rate | 56.3% |
| Live deployment | NO |
| Open position | **ADANIGREEN** qty 3,081 entry Rs.1,290.70 (2026-05-04) last Rs.1,308.00 unrealized +53,301 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 65 | 38 | 27 | 58% | +2,954,692 |
| **Mid** | 5 | 2 | 3 | 40% | +56,229 |
| **Small** | 1 | 0 | 1 | 0% | -34,017 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-06 → 2026-05-04 | 163.26 | +39.23% | +1,120,444 |
| HAL          | 2024-05-13 → 2024-06-03 | 3,921.75 | +34.47% | +620,522 |
| ETERNAL      | 2025-07-14 → 2025-09-08 | 270.60 | +21.82% | +580,107 |
| INDUSTOWER   | 2024-03-18 → 2024-05-13 | 248.50 | +32.11% | +438,182 |
| BEL          | 2025-05-19 → 2025-06-30 | 363.75 | +15.88% | +372,950 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-11-10 | 170.25 | -11.86% | -381,127 |
| ADANIPOWER   | 2024-06-03 → 2024-06-10 | 174.90 | -11.98% | -290,422 |
| ETERNAL      | 2024-12-09 → 2025-01-06 | 295.30 | -10.31% | -241,194 |
| BEL          | 2026-03-09 → 2026-03-16 | 457.35 | -6.09% | -180,830 |
| ATGL         | 2023-12-11 → 2024-01-15 | 1,172.15 | -11.32% | -178,680 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
