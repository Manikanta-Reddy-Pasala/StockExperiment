# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 MINUS NSE Smallcap 250) + uptrend filter (close > 200d SMA). Monthly rotation, yearly-PIT, top-1 by 30d return.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 29 |
| Trades per year | ~9.7 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers** (498/504 N500, 4-yr re-pull, cont_flag=1) |

## Stock pick logic

1. **Universe**: top-100 by 20-day ADV from N500 (yearly-PIT, rebuilt at year start)
2. **Drop Small**: remove NSE Smallcap 250 members from universe
3. **Uptrend filter**: keep only stocks with close > 200-day SMA
4. **Rank by 30-day return**, pick top-1
5. **Rebalance monthly** (1st trading day)

## Headline result

| Metric | Value |
|---|---:|
| Final NAV | **₹10,820,091** |
| Total return | **+982.01%** |
| **3-yr CAGR** | **+121.18%/yr** |
| Max DD (cash NAV) | 25.42% |
| Calmar | 4.77 |
| Trades | 29 |
| WR | 86.2% (25W / 4L) |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,278,215 | **+127.82%** | 9 |
| 2024-25 | ₹2,278,215 | ₹3,908,220 | **+71.55%** | 11 |
| 2025-26 | ₹3,908,220 | ₹10,820,091 | **+176.85%** | 9 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 16 | 14 | 2 | 88% | +7,173,591 |
| **Mid** | 13 | 11 | 2 | 85% | +2,646,500 |