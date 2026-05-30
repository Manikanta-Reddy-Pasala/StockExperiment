# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly rotation top-1 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-04-01 → 2026-05-29** (~5.16 years) |
| First entry | 2021-06-01 |
| Last exit | 2026-05-04 |
| Total trades | 50 |
| Trades per year | ~9.7 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-100 by 20-day ADV from PIT N500 (eligible_at; yearly-PIT, rebuilt at a FIXED mid-May anchor)
2. Drop NSE Smallcap 250 members
3. Uptrend filter: close > 200-day SMA
4. Max-price filter: close ≤ ₹3,000 at entry
5. Rank by 30-day return, pick top-1 (RET1 rotation)
6. Rebalance: 1st trading day of month (mid-month available as opt-in, default off)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.16,828,054** |
| Total return | **+1582.81%** |
| 5.16-yr CAGR | **+72.86%** |
| Max DD | **28.63%** |
| Calmar (CAGR / Max DD) | **2.54** |
| Trades closed | 50 |
| Wins / Losses | 38 / 12 |
| Win rate | 76.0% |
| Live deployment | NO |
| Open position | **ADANIGREEN** qty 11,405 entry Rs.1,290.70 (2026-05-04) last Rs.1,475.40 unrealized +2,106,504 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 15 | 12 | 3 | 80% | +8,133,238 |
| **Mid** | 32 | 23 | 9 | 72% | +5,402,255 |
| **Other** | 3 | 3 | 0 | 100% | +186,061 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +4,545,996 |
| SHRIRAMFIN   | 2025-11-03 → 2026-03-02 | 796.45 | +32.15% | +2,425,050 |
| BSE          | 2025-05-02 → 2025-06-02 | 2,102.17 | +28.12% | +1,494,968 |
| PAYTM        | 2025-08-01 → 2025-09-01 | 1,076.40 | +14.81% | +848,008 |
| IDEA         | 2025-10-01 → 2025-11-03 | 8.52 | +11.97% | +806,575 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -1,104,610 |
| IRFC         | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -340,875 |
| COFORGE      | 2024-12-02 → 2025-02-01 | 1,742.14 | -7.28% | -324,787 |
| HCLTECH      | 2022-01-03 → 2022-02-01 | 1,326.15 | -14.58% | -183,682 |
| CANBK        | 2022-02-01 → 2022-03-02 | 51.75 | -16.46% | -177,318 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
