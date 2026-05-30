# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly rotation top-1 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2021-04-01 |
| Last exit | 2026-05-04 |
| Total trades | 52 |
| Trades per year | ~17.3 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-100 by 20-day ADV from N500 (yearly-PIT, rebuilt at a FIXED mid-May anchor)
2. Drop NSE Smallcap 250 members
3. Uptrend filter: close > 200-day SMA
4. Max-price filter: close ≤ ₹3,000 at entry
5. Rank by 30-day return, pick top-1 (RET1 rotation)
6. Rebalance: 1st trading day of month (mid-month available as opt-in, default off)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.12,644,915** |
| Total return | **+1164.49%** |
| 5.16-yr CAGR | **+63.54%** |
| Max DD | **37.64%** |
| Calmar (CAGR / Max DD) | **1.69** |
| Trades closed | 52 |
| Wins / Losses | 37 / 15 |
| Win rate | 71.2% |
| Live deployment | NO |
| Open position | **ADANIGREEN** qty 8,570 entry Rs.1,290.70 (2026-05-04) last Rs.1,475.40 unrealized +1,582,879 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 18 | 13 | 5 | 72% | +6,220,053 |
| **Mid** | 34 | 24 | 10 | 71% | +3,841,984 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +3,415,937 |
| SHRIRAMFIN   | 2025-11-03 → 2026-03-02 | 796.45 | +32.15% | +1,822,308 |
| BSE          | 2025-05-02 → 2025-06-02 | 2,102.17 | +28.12% | +1,123,147 |
| PAYTM        | 2025-08-01 → 2025-09-01 | 1,076.40 | +14.81% | +637,281 |
| IDEA         | 2025-10-01 → 2025-11-03 | 8.52 | +11.97% | +606,067 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -829,776 |
| COFORGE      | 2024-12-02 → 2025-02-01 | 1,742.14 | -7.28% | -243,971 |
| IRFC         | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -222,952 |
| HCLTECH      | 2022-01-03 → 2022-02-01 | 1,326.15 | -14.58% | -138,632 |
| CANBK        | 2022-02-01 → 2022-03-02 | 51.75 | -16.46% | -133,773 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
