# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret) + MAX_PRICE≤₹3,000 filter.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 29 |
| Trades per year | ~9.7 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Filter at entry: close ≤ ₹3,000 (skips mega-priced losers BAJAJ-AUTO etc.)
3. Rank by 30-day return, pick top-1
4. Rebalance: 1st trading day of month
5. Exit: rotation only — sell when not rank-1

## Headline result

| Metric | Value |
|---|---:|
| Final NAV | **₹6,305,316** |
| Total return | **+530.53%** |
| 3-yr CAGR | **+84.74%** |
| Max DD (rebal cap_after) | **33.89%** |
| Calmar (CAGR / Max DD) | **2.50** |
| Trades | 29 |
| Wins / Losses | 20 / 9 |
| Win rate | 69.0% |
| Live deployment | ✅ YES |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 29 | 20 | 9 | 69% | +5,305,314 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +1,947,071 |
| SHRIRAMFIN   | 2025-11-03 → 2026-01-01 | 796.45 | +28.03% | +996,811 |
| MAZDOCK      | 2025-04-01 → 2025-06-02 | 2,578.55 | +31.26% | +697,147 |
| RECLTD       | 2023-11-01 → 2023-12-01 | 282.85 | +32.23% | +627,568 |
| MAZDOCK      | 2023-07-03 → 2023-09-01 | 644.55 | +46.39% | +471,491 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ENRIN        | 2026-03-02 → 2026-04-01 | 2,972.70 | -12.07% | -598,120 |
| IRFC         | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -384,525 |
| BAJAJFINSV   | 2024-10-01 → 2024-11-01 | 1,975.25 | -11.17% | -340,088 |
| HINDZINC     | 2024-11-01 → 2024-12-02 | 558.25 | -9.92% | -268,579 |
| TATACONSUM   | 2025-02-01 → 2025-03-03 | 1,069.85 | -10.84% | -264,250 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
