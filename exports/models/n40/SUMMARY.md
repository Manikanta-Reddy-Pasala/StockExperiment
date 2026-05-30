# n40 — SUMMARY

**Top-40 ADV + uptrend + Nifty 100. Weekly rotation top-1 by 30d ret. No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-03-01 → 2026-05-29** (~5.24 years) |
| First entry | 2021-04-05 |
| Last exit | 2026-05-25 |
| Total trades | 133 |
| Trades per year | ~25.4 |
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
| Final NAV (cap + open MTM) | **Rs.6,097,844** |
| Total return | **+509.78%** |
| 5.24-yr CAGR | **+41.17%** |
| Max DD | **36.88%** |
| Calmar (CAGR / Max DD) | **1.12** |
| Trades closed | 133 |
| Wins / Losses | 78 / 55 |
| Win rate | 58.6% |
| Live deployment | NO |
| Open position | **ADANIENT** qty 2,075 entry Rs.2,849.70 (2026-05-25) last Rs.2,937.40 unrealized +181,978 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 133 | 78 | 55 | 59% | +4,915,872 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-06 → 2026-05-04 | 163.26 | +39.23% | +1,524,536 |
| ETERNAL      | 2025-07-14 → 2025-09-08 | 270.60 | +21.82% | +773,496 |
| INDUSTOWER   | 2024-03-18 → 2024-05-13 | 248.50 | +32.11% | +522,371 |
| ADANIGREEN   | 2026-05-04 → 2026-05-25 | 1,290.70 | +9.33% | +504,717 |
| BEL          | 2025-05-19 → 2025-06-30 | 363.75 | +15.88% | +497,285 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-11-03 | 170.25 | -7.94% | -340,285 |
| ETERNAL      | 2024-12-09 → 2025-01-06 | 295.30 | -10.31% | -298,197 |
| BEL          | 2026-03-09 → 2026-03-16 | 457.35 | -6.09% | -246,055 |
| ADANIGREEN   | 2022-04-18 → 2022-05-23 | 2,970.50 | -24.31% | -242,642 |
| INDUSTOWER   | 2024-07-29 → 2024-08-05 | 443.40 | -6.71% | -218,930 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
