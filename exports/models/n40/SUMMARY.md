# n40 — SUMMARY

**Top-40 ADV + uptrend + Nifty 100. Weekly rotation top-1 by 30d ret. No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-03-01 → 2026-05-31** (~5.25 years) |
| First entry | 2021-04-05 |
| Last exit | 2026-05-25 |
| Total trades | 137 |
| Trades per year | ~26.1 |
| Rebalance | Weekly |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-40 by 20-day ADV from N500 (rebuilt each rebalance)
2. Uptrend filter: close > 200-day SMA
3. Large-cap filter: stock must be in NSE Nifty 100
4. Rank by 30-day return, pick top-1
5. Rebalance: first trading day of each ISO week (weekly cut whipsaw vs daily)


## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | First trading day of each ISO week (WEEKLY). |
| **Universe & filters** | Top-40 by 20d ADV from N500, intersect PIT Nifty 100, and close > 200d SMA (uptrend). |
| **Entry** | On the weekly rebalance, BUY rank-1 by 30-day return among the filtered set (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1) at the next weekly rebalance, or when it drops out of Nifty 100 / below its 200d SMA. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` + `ind_nifty500list.csv` → nifty100.csv/nifty500.csv. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.7,847,232** |
| Total return | **+684.72%** |
| 5.25-yr CAGR | **+48.07%** |
| Max DD | **37.05%** |
| Calmar (CAGR / Max DD) | **1.30** |
| Trades closed | 137 |
| Wins / Losses | 81 / 56 |
| Win rate | 59.1% |
| Live deployment | NO |
| Open position | **ADANIENT** qty 2,671 entry Rs.2,849.70 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +9.8% | 26.8% |
| 2022 | +62.6% | 27.3% |
| 2023 | +12.4% | 24.0% |
| 2024 | +67.1% | 19.3% |
| 2025 | +59.6% | 15.8% |
| 2026 | +44.3% | 25.5% |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 123 | 73 | 50 | 59% | +6,589,334 |
| **Mid** | 13 | 7 | 6 | 54% | -2,169 |
| **Small** | 1 | 1 | 0 | 100% | +25,818 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-06 → 2026-05-04 | 163.26 | +39.23% | +1,961,865 |
| ETERNAL      | 2025-07-14 → 2025-09-08 | 270.60 | +21.82% | +974,325 |
| INDUSTOWER   | 2024-03-18 → 2024-05-13 | 248.50 | +32.11% | +657,951 |
| ADANIGREEN   | 2026-05-04 → 2026-05-25 | 1,290.70 | +9.33% | +649,558 |
| BEL          | 2025-05-19 → 2025-06-30 | 363.75 | +15.88% | +626,414 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-09-24 | 170.25 | -12.00% | -647,713 |
| ETERNAL      | 2024-12-09 → 2025-01-06 | 295.30 | -10.31% | -375,601 |
| BEL          | 2026-03-09 → 2026-03-16 | 457.35 | -6.09% | -316,654 |
| ATGL         | 2023-12-11 → 2023-12-13 | 1,172.15 | -12.00% | -288,208 |
| BPCL         | 2024-02-19 → 2024-03-15 | 326.27 | -12.00% | -279,392 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
