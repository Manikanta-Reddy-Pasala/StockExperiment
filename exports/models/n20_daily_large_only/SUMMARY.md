# n20_daily_large_only — SUMMARY

**Top-20 ADV + uptrend + Nifty 100 + MAX_PRICE≤₹2,500. Daily rotation top-1 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-13 |
| Total trades | 134 |
| Trades per year | ~44.7 |
| Rebalance | Daily |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-20 by 20-day ADV from N500 (rebuilt daily)
2. Uptrend filter: close > 200-day SMA
3. Large-cap filter: stock must be in NSE Nifty 100
4. Max-price filter: close ≤ ₹2,500 at entry
5. Rank by 30-day return, pick top-1
6. Rebalance: every trading day

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.18,676,864** |
| Total return | **+1767.69%** |
| 2.99-yr CAGR | **+165.97%** |
| Max DD | **24.57%** |
| Calmar (CAGR / Max DD) | **6.76** |
| Trades closed | 134 |
| Wins / Losses | 58 / 75 |
| Win rate | 43.6% |
| Live deployment | NO |
| Open position | **ADANIPOWER** qty 89,094 entry Rs.181.35 (2026-04-13) last Rs.209.63 unrealized +2,519,578 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 134 | 58 | 75 | 44% | +15,157,289 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MAZDOCK      | 2024-05-29 → 2024-07-04 | 1,678.68 | +66.37% | +3,572,822 |
| BEL          | 2025-05-13 → 2025-07-02 | 335.75 | +27.16% | +2,730,619 |
| ETERNAL      | 2025-07-21 → 2025-09-10 | 271.70 | +19.40% | +2,200,857 |
| SBIN         | 2026-02-05 → 2026-03-05 | 1,073.50 | +8.94% | +1,432,992 |
| MAZDOCK      | 2025-04-07 → 2025-04-15 | 2,317.30 | +14.84% | +1,368,088 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ETERNAL      | 2024-12-16 → 2024-12-23 | 294.15 | -6.87% | -710,737 |
| BEL          | 2025-03-28 → 2025-04-02 | 301.32 | -6.28% | -652,872 |
| BEL          | 2026-01-30 → 2026-02-05 | 449.00 | -3.59% | -595,974 |
| BEL          | 2026-03-12 → 2026-03-13 | 453.55 | -3.12% | -534,757 |
| KOTAKBANK    | 2025-04-02 → 2025-04-07 | 430.92 | -5.42% | -528,037 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
