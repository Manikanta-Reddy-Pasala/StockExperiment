# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly + mid-month rotation, hold while in top-5 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-15 |
| Total trades | 32 |
| Trades per year | ~10.7 |
| Rebalance | Monthly (1st trading day) + mid-month (day-15) lead check |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-100 by 20-day ADV from N500 (yearly-PIT, rebuilt at year start)
2. Drop NSE Smallcap 250 members
3. Uptrend filter: close > 200-day SMA
4. Max-price filter: close ≤ ₹3,000 at entry
5. Rank by 30-day return, hold rank-1 while it stays in the top-5
6. Rebalance: 1st trading day of month + mid-month day-15 check (rotate only if rank-1 leads held by ≥3pp)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.17,507,419** |
| Total return | **+1650.74%** |
| 2.99-yr CAGR | **+160.29%** |
| Max DD | **21.64%** |
| Calmar (CAGR / Max DD) | **7.41** |
| Trades closed | 32 |
| Wins / Losses | 26 / 6 |
| Win rate | 81.2% |
| Live deployment | NO |
| Open position | **ADANIPOWER** qty 83,515 entry Rs.183.43 (2026-04-15) last Rs.209.63 unrealized +2,188,093 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 17 | 15 | 2 | 88% | +5,959,118 |
| **Mid** | 15 | 11 | 4 | 73% | +8,360,208 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| IDEA         | 2025-09-15 → 2025-11-17 | 8.14 | +34.28% | +3,762,321 |
| BSE          | 2024-09-16 → 2024-12-02 | 1,143.93 | +32.99% | +1,912,787 |
| MAZDOCK      | 2025-04-15 → 2025-05-15 | 2,661.30 | +19.58% | +1,775,909 |
| BSE          | 2025-05-15 → 2025-06-16 | 2,382.67 | +13.21% | +1,432,651 |
| PAYTM        | 2025-08-01 → 2025-09-15 | 1,076.40 | +14.08% | +1,355,001 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| SBIN         | 2026-02-16 → 2026-03-16 | 1,208.10 | -11.70% | -1,935,059 |
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -1,855,569 |
| COCHINSHIP   | 2025-06-16 → 2025-07-01 | 2,188.00 | -6.53% | -801,251 |
| BSE          | 2024-12-16 → 2025-02-01 | 1,887.93 | -4.59% | -376,969 |
| ETERNAL      | 2024-03-01 → 2024-03-15 | 166.50 | -3.87% | -108,599 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
