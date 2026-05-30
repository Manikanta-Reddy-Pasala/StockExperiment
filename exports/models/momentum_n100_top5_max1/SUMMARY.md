# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret). No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-03-01 → 2026-05-29** (~5.24 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-15 |
| Total trades | 96 |
| Trades per year | ~18.3 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Rank by 30-day return, pick top-1
3. Rebalance: 1st trading day of month
4. Exit: rotation only — sell when not rank-1


## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Real point-in-time NSE Nifty 100 (eligible_at). No price/SMA filter — pure index membership. |
| **Entry** | BUY rank-1 by 15-day return (single position, max 1). |
| **Exit** | Hold while in the top-3 by 15d return (RETAIN=3); rotate out when it drops below rank-3, or leaves the index. Mid-month only rotates if the new rank-1 leads the held name by ≥ 5pp. |
| **Source** | Live: niftyindices.com `ind_nifty100list.csv` → nifty100.csv → n100_current.json. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.10,376,511** |
| Total return | **+937.65%** |
| 5.24-yr CAGR | **+56.24%** |
| Max DD | **52.18%** |
| Calmar (CAGR / Max DD) | **1.08** |
| Trades closed | 96 |
| Wins / Losses | 53 / 43 |
| Win rate | 55.2% |
| Live deployment | YES |
| Open position | **VEDL** qty 29,428 entry Rs.331.05 (2026-05-15) last Rs.352.60 unrealized +634,173 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 96 | 53 | 43 | 55% | +8,742,337 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIGREEN   | 2026-04-15 → 2026-05-15 | 1,096.05 | +25.82% | +1,998,759 |
| ADANIPOWER   | 2026-04-01 → 2026-04-15 | 157.11 | +16.75% | +1,111,099 |
| HINDZINC     | 2025-12-15 → 2026-01-16 | 568.05 | +12.27% | +747,463 |
| ATGL         | 2023-12-01 → 2024-01-01 | 701.45 | +42.70% | +708,018 |
| BEL          | 2025-05-15 → 2025-06-16 | 350.40 | +15.25% | +667,911 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIENSOL   | 2024-08-01 → 2024-08-16 | 1,275.20 | -14.89% | -777,451 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -450,660 |
| INDUSINDBK   | 2025-05-02 → 2025-05-15 | 853.00 | -8.50% | -406,725 |
| CGPOWER      | 2025-09-15 → 2025-10-01 | 791.35 | -6.51% | -397,605 |
| HINDZINC     | 2025-10-15 → 2025-11-03 | 513.20 | -6.34% | -373,804 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
