# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly + mid-month rotation, hold while in top-5 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-15 |
| Total trades | 62 |
| Trades per year | ~20.7 |
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
| Final NAV (cap + open MTM) | **Rs.9,237,680** |
| Total return | **+823.77%** |
| 5.16-yr CAGR | **+53.88%** |
| Max DD | **42.23%** |
| Calmar (CAGR / Max DD) | **1.28** |
| Trades closed | 62 |
| Wins / Losses | 37 / 25 |
| Win rate | 59.7% |
| Live deployment | NO |
| Open position | **ADANIGREEN** qty 6,261 entry Rs.1,379.00 (2026-05-15) last Rs.1,475.40 unrealized +603,560 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 39 | 25 | 14 | 64% | +3,126,989 |
| **Mid** | 23 | 12 | 11 | 52% | +4,507,129 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| COCHINSHIP   | 2024-05-15 → 2024-07-15 | 1,330.50 | +106.29% | +1,944,456 |
| IDEA         | 2025-10-01 → 2025-11-17 | 8.52 | +28.29% | +1,670,839 |
| ADANIPOWER   | 2026-04-15 → 2026-05-15 | 183.43 | +20.66% | +1,478,479 |
| BSE          | 2024-09-16 → 2024-12-02 | 1,143.93 | +32.99% | +1,000,355 |
| BSE          | 2025-05-15 → 2025-07-01 | 2,382.67 | +16.47% | +933,983 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -1,067,985 |
| SBIN         | 2026-02-16 → 2026-03-16 | 1,208.10 | -11.70% | -908,919 |
| OIL          | 2024-09-02 → 2024-09-16 | 726.25 | -16.34% | -592,182 |
| BSE          | 2025-11-17 → 2025-12-15 | 2,811.90 | -5.80% | -439,122 |
| CGPOWER      | 2025-09-15 → 2025-10-01 | 791.35 | -6.51% | -411,575 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
