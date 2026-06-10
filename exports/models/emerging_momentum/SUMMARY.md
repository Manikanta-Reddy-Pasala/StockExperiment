# emerging_momentum — SUMMARY

**Emerging mid/small momentum (top-100 ADV from PIT N500 − N100). VOL-ADJ rank, top-1 monthly + mid-month. 2.5×ATR stop, NO profit-take (disabled 2026-06-10).**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-03-01 → 2026-06-10** (~5.28 years) |
| First entry | 2021-03-01 |
| Last exit | 2026-05-04 |
| Total trades | 61 |
| Trades per year | ~11.6 |
| Rebalance | Monthly (1st trading day) + mid-month (day-15) lead-check |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-100 by 20-day ADV from PIT N500 minus PIT N100 (mid/small, yearly anchor)
2. Filters: 30d return > 0, close ≤ ₹3,000 (no SMA gate)
3. Rank by VOL-ADJUSTED momentum: 30d return ÷ 60d return-volatility, pick top-1 (RET1)
4. Rebalance: 1st trading day of month + mid-month check (rotate only on ≥5pp lead)
5. Stop: DAILY hard stop at entry − 2.5×ATR(14) (not trailing)
6. Profit-take: DISABLED 2026-06-10 (was book-half @+30%) — let winners run for max CAGR
7. Exit: rotation (held drops rank-1) OR ATR-from-entry stop fires


## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month + a mid-month (day-15) lead check. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF. |
| **Entry** | BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1. |
| **Exit** | Rotate when held is no longer rank-1 (RET1). Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV. |

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.57,542,640** |
| Total return | **+5654.26%** |
| 5.28-yr CAGR | **+115.57%** |
| Max DD | **37.92%** |
| Calmar (CAGR / Max DD) | **3.05** |
| Trades closed | 61 |
| Wins / Losses | 41 / 20 |
| Win rate | 67.2% |
| Live deployment | YES |
| Open position | **HFCL** qty 340,267 entry Rs.182.28 (2026-06-01) last Rs.169.11 unrealized +0 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +19.6% | 37.9% |
| 2022 | +150.4% | 20.9% |
| 2023 | +358.2% | 26.8% |
| 2024 | +171.4% | 26.3% |
| 2025 | +45.8% | 11.0% |
| 2026 | +10.0% | 24.6% |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 1 | 1 | 0 | 100% | +898,684 |
| **Mid** | 41 | 33 | 8 | 80% | +42,101,872 |
| **Small** | 16 | 5 | 11 | 31% | +1,408,495 |
| **Other** | 3 | 2 | 1 | 67% | +431,745 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| NATIONALUM   | 2026-01-01 → 2026-03-02 | 314.60 | +15.34% | +6,470,084 |
| BSE          | 2023-10-03 → 2023-12-01 | 432.62 | +92.52% | +5,591,232 |
| DATAPATTNS   | 2025-05-15 → 2025-06-02 | 2,625.00 | +12.28% | +4,412,932 |
| FEDERALBNK   | 2025-11-17 → 2026-01-01 | 239.06 | +11.37% | +4,308,174 |
| IRFC         | 2024-01-15 → 2024-02-01 | 130.10 | +30.59% | +4,080,575 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -6,634,985 |
| BHARATFORG   | 2026-03-02 → 2026-04-01 | 1,879.30 | -11.22% | -5,460,201 |
| RRKABEL      | 2025-06-02 → 2025-06-16 | 1,390.50 | -3.04% | -1,227,631 |
| ABREL        | 2024-10-01 → 2024-11-01 | 2,890.80 | -4.25% | -1,150,904 |
| VOLTAS       | 2024-05-02 → 2024-06-03 | 1,481.95 | -5.10% | -1,096,547 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
