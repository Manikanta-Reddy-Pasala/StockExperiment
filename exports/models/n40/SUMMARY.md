# n40 — SUMMARY

**Top-40 ADV + uptrend + Nifty 100. Weekly rotation top-1 by 30d ret. No price filter — honest baseline.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-04-01 → 2026-05-29** (~5.16 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-25 |
| Total trades | 127 |
| Trades per year | ~24.6 |
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
| Final NAV (cap + open MTM) | **Rs.5,738,726** |
| Total return | **+473.87%** |
| 5.16-yr CAGR | **+40.32%** |
| Max DD | **36.85%** |
| Calmar (CAGR / Max DD) | **1.09** |
| Trades closed | 127 |
| Wins / Losses | 74 / 53 |
| Win rate | 58.3% |
| Live deployment | NO |
| Open position | **ADANIENT** qty 1,953 entry Rs.2,849.70 (2026-05-25) last Rs.2,937.40 unrealized +171,278 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 127 | 74 | 53 | 58% | +4,567,450 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-06 → 2026-05-04 | 163.26 | +39.23% | +1,434,752 |
| ETERNAL      | 2025-07-14 → 2025-09-08 | 270.60 | +21.82% | +727,909 |
| INDUSTOWER   | 2024-03-18 → 2024-05-13 | 248.50 | +32.11% | +520,136 |
| ADANIGREEN   | 2026-05-04 → 2026-05-25 | 1,290.70 | +9.33% | +474,978 |
| BEL          | 2025-05-19 → 2025-06-30 | 363.75 | +15.88% | +468,006 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-11-03 | 170.25 | -7.94% | -320,235 |
| ETERNAL      | 2024-12-09 → 2025-01-06 | 295.30 | -10.31% | -280,597 |
| ADANIGREEN   | 2022-04-18 → 2022-05-23 | 2,970.50 | -24.31% | -249,142 |
| BEL          | 2026-03-09 → 2026-03-16 | 457.35 | -6.09% | -231,545 |
| ADANIENSOL   | 2021-06-07 → 2021-06-21 | 1,589.45 | -18.08% | -205,419 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
