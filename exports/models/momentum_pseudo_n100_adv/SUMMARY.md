# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly rotation top-1 by 30d ret.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2021-03-01 → 2026-05-29** (~5.24 years) |
| First entry | 2021-06-01 |
| Last exit | 2026-05-04 |
| Total trades | 50 |
| Trades per year | ~9.5 |
| Rebalance | Monthly (1st trading day) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-100 by 20-day ADV from PIT N500 (eligible_at; yearly-PIT, rebuilt at a FIXED mid-May anchor)
2. Drop NSE Smallcap 250 members
3. Uptrend filter: close > 200-day SMA
4. Max-price filter: close ≤ ₹3,000 at entry
5. Rank by 30-day return, pick top-1 (RET1 rotation)
6. Rebalance: 1st trading day of month (mid-month available as opt-in, default off)


## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month (mid-month available as opt-in, default OFF). |
| **Universe & filters** | Top-100 by 20d ADV from PIT N500 (yearly fixed mid-May anchor) minus Smallcap-250; close > 200d SMA; price ≤ ₹3000. |
| **Entry** | BUY rank-1 by 30-day return (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → yearly_universes.json. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.19,741,787** |
| Total return | **+1874.18%** |
| 5.24-yr CAGR | **+76.63%** |
| Max DD | **28.63%** |
| Calmar (CAGR / Max DD) | **2.68** |
| Trades closed | 50 |
| Wins / Losses | 38 / 12 |
| Win rate | 76.0% |
| Live deployment | NO |
| Open position | **ADANIGREEN** qty 13,380 entry Rs.1,290.70 (2026-05-04) last Rs.1,475.40 unrealized +2,471,286 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +50.4% | 5.8% |
| 2022 | -11.5% | 28.6% |
| 2023 | +125.9% | 13.1% |
| 2024 | +56.3% | 16.4% |
| 2025 | +55.4% | 22.3% |
| 2026 | +52.4% | 2.2% |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 14 | 11 | 3 | 79% | +9,288,589 |
| **Mid** | 33 | 24 | 9 | 73% | +6,551,039 |
| **Other** | 3 | 3 | 0 | 100% | +430,877 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +5,333,106 |
| SHRIRAMFIN   | 2025-11-03 → 2026-03-02 | 796.45 | +32.15% | +2,844,972 |
| BSE          | 2025-05-02 → 2025-06-02 | 2,102.17 | +28.12% | +1,753,883 |
| PAYTM        | 2025-08-01 → 2025-09-01 | 1,076.40 | +14.81% | +994,975 |
| IDEA         | 2025-10-01 → 2025-11-03 | 8.52 | +11.97% | +946,224 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| IDEA         | 2025-02-01 → 2025-03-03 | 9.60 | -22.29% | -1,500,797 |
| MCX          | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -1,295,646 |
| IRFC         | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -486,562 |
| PAYTM        | 2024-04-01 → 2024-05-02 | 406.05 | -8.34% | -294,461 |
| HCLTECH      | 2022-01-03 → 2022-02-01 | 1,326.15 | -14.58% | -225,446 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
