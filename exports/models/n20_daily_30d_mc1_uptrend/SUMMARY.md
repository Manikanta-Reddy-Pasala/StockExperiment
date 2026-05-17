# n20_daily_30d_mc1_uptrend — SUMMARY

Aggressive daily-rebalance momentum rotation. Highest raw CAGR + highest DD.

## Stock pick logic (plain English)

1. **Universe build (per day)**: take all 500 stocks in NSE Nifty 500
2. **Compute liquidity**: 20-day ADV per stock (using only past data — strictly PIT)
3. **Rank by ADV**: sort descending, **take top 20** (much smaller than pseudo-N100's 100)
4. **Uptrend filter**: keep only stocks where today's close > 200-day SMA
5. **Rank by return**: within filtered set (typically 5-15 stocks), compute 30-day return
6. **Pick top-1**: hold the best for the next trading day
7. **Rebalance daily**: every trading day, re-rank + rotate if new top-1

**Unique filter**: 2 hurdles before momentum ranking — must be in top-20 by ADV (most liquid) AND in confirmed uptrend (close > 200d SMA). Universe rebuilds DAILY (most aggressive rebal of all 5 models).

| Key knob | Value |
|---|---|
| Universe size | Top **20** by ADV (vs 100 in pseudo-N100) |
| Filter | close > 200d SMA (uptrend gate) |
| Lookback (signal) | 30 days |
| Position | top-1 (max_concurrent=1) |
| **Rebalance period** | **Daily** |
| Universe rebuild | **Daily** (strictly PIT, prior-day data only) |
| Exit | Rotation only |

## Headline result (3-year backtest, ₹10L start)

| Metric | Value |
|---|---:|
| Final NAV (cash+open MTM) | **~₹17,000,000** |
| Total return | **+1599.57%** |
| **3-yr CAGR** | **+157.27%/yr** |
| Max DD (cash NAV) | 50.61% |
| Trades | 134 |
| WR | 47.8% (64W / 70L) |

## Returns by NSE cap segment

| Cap segment | Trades | Wins | Losses | WR | Total PnL ₹ | Avg PnL/trade ₹ |
|---|---:|---:|---:|---:|---:|---:|
| **Large** | 50 | 27 | 23 | 54% | +10,263,100 | +205,262 |
| **Mid** | 46 | 25 | 21 | 54% | +5,698,998 | +123,891 |
| **Small** | 38 | 12 | 26 | 32% | -143,536 | -3,777 |

## Full trade ledger — every entry with price, invested ₹, exit, gain/loss

**134 trades total** — only top winners + losers shown for brevity (full in `TRADE_LEDGER.md`).

### Top 15 winners

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | **Invested** | Exit Date | Exit ₹ | **PnL ₹** | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | GRSE | **Small** | Nifty Smallcap 250 | 2025-05-16 | 2,481.70 | 6,294 | ₹15,619,820 | 2025-06-11 | 3,098.60 | +3,882,769 | +24.86% | rotate |
| 2 | NETWEB | **Small** | Nifty Smallcap 250 | 2025-09-18 | 3,037.70 | 4,055 | ₹12,317,874 | 2025-10-20 | 3,897.60 | +3,486,894 | +28.31% | rotate |
| 3 | ITI | **Small** | Nifty Smallcap 250 | 2024-12-31 | 387.15 | 21,357 | ₹8,268,363 | 2025-01-06 | 544.35 | +3,357,320 | +40.60% | rotate |
| 4 | HINDCOPPER | **Small** | Nifty Smallcap 250 | 2025-12-26 | 475.60 | 23,509 | ₹11,180,880 | 2026-02-11 | 603.70 | +3,011,503 | +26.93% | rotate |
| 5 | BSE | **Mid** | Nifty Midcap 150 | 2025-04-17 | 1,977.00 | 6,438 | ₹12,727,926 | 2025-05-16 | 2,426.33 | +2,892,787 | +22.73% | rotate |
| 6 | ADANIPOWER | **Large** | Nifty 100 | 2026-04-20 | 200.83 | 73,063 | ₹14,673,242 | 2026-05-07 | 230.19 | +2,145,130 | +14.62% | rotate |
| 7 | PAYTM | **Mid** | Nifty Midcap 150 | 2025-08-08 | 1,062.40 | 11,701 | ₹12,431,142 | 2025-09-01 | 1,235.80 | +2,028,953 | +16.32% | rotate |
| 8 | IRFC | **Large** | Nifty 100 | 2024-01-12 | 113.40 | 26,783 | ₹3,037,192 | 2024-02-01 | 169.90 | +1,513,240 | +49.82% | rotate |
| 9 | COCHINSHIP | **Mid** | Nifty Midcap 150 | 2024-05-21 | 1,641.00 | 2,941 | ₹4,826,181 | 2024-06-14 | 2,122.35 | +1,415,650 | +29.33% | rotate |
| 10 | PAYTM | **Mid** | Nifty Midcap 150 | 2024-11-19 | 814.25 | 8,955 | ₹7,291,609 | 2024-12-09 | 971.95 | +1,412,204 | +19.37% | rotate |
| 11 | ADANIPOWER | **Large** | Nifty 100 | 2026-04-13 | 181.35 | 76,937 | ₹13,952,525 | 2026-04-17 | 198.50 | +1,319,470 | +9.46% | rotate |
| 12 | IDEA | **Mid** | Nifty Midcap 150 | 2025-12-10 | 10.72 | 898,589 | ₹9,632,874 | 2025-12-22 | 11.85 | +1,015,406 | +10.54% | rotate |
| 13 | BSE | **Mid** | Nifty Midcap 150 | 2024-09-19 | 1,237.07 | 5,198 | ₹6,430,290 | 2024-10-29 | 1,428.02 | +992,558 | +15.44% | rotate |
| 14 | MAZDOCK | **Large** | Nifty 100 | 2024-06-14 | 1,938.78 | 3,219 | ₹6,240,933 | 2024-07-01 | 2,196.95 | +831,049 | +13.32% | rotate |
| 15 | IRFC | **Large** | Nifty 100 | 2023-09-01 | 55.75 | 35,940 | ₹2,003,655 | 2023-10-13 | 76.60 | +749,349 | +37.40% | rotate |

### Top 15 losses

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | **Invested** | Exit Date | Exit ₹ | **PnL ₹** | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | RPOWER | **Small** | Nifty Smallcap 250 | 2025-06-11 | 71.27 | 273,669 | ₹19,504,390 | 2025-06-17 | 63.63 | -2,090,831 | -10.72% | rotate |
| 2 | CHENNPETRO | **Small** | Nifty Smallcap 250 | 2025-11-17 | 1,075.80 | 11,547 | ₹12,422,263 | 2025-11-26 | 896.60 | -2,069,222 | -16.66% | rotate |
| 3 | NETWEB | **Small** | Nifty Smallcap 250 | 2025-10-28 | 4,206.80 | 3,389 | ₹14,256,845 | 2025-11-04 | 3,630.30 | -1,953,758 | -13.70% | rotate |
| 4 | MCX | **Mid** | Nifty Midcap 150 | 2025-07-01 | 9,060.50 | 1,897 | ₹17,187,768 | 2025-07-18 | 8,235.00 | -1,565,974 | -9.11% | rotate |
| 5 | JPPOWER | **Small** | Nifty Smallcap 250 | 2025-07-18 | 22.86 | 683,379 | ₹15,622,044 | 2025-07-24 | 21.11 | -1,195,913 | -7.66% | rotate |
| 6 | GRSE | **Small** | Nifty Smallcap 250 | 2024-07-11 | 2,594.75 | 3,130 | ₹8,121,568 | 2024-07-26 | 2,212.75 | -1,195,660 | -14.72% | rotate |
| 7 | NETWEB | **Small** | Nifty Smallcap 250 | 2025-09-08 | 3,106.60 | 4,480 | ₹13,917,568 | 2025-09-15 | 2,882.90 | -1,002,176 | -7.20% | rotate |
| 8 | RVNL | **Mid** | Nifty Midcap 150 | 2024-07-30 | 614.45 | 12,461 | ₹7,656,661 | 2024-08-08 | 538.45 | -947,036 | -12.37% | rotate |
| 9 | GRSE | **Small** | Nifty Smallcap 250 | 2025-06-17 | 3,173.80 | 5,486 | ₹17,411,467 | 2025-06-25 | 3,003.10 | -936,460 | -5.38% | rotate |
| 10 | BSE | **Mid** | Nifty Midcap 150 | 2025-12-08 | 2,798.80 | 3,731 | ₹10,442,323 | 2025-12-10 | 2,581.70 | -810,000 | -7.76% | rotate |
| 11 | IDEA | **Mid** | Nifty Midcap 150 | 2025-10-27 | 9.97 | 1,510,490 | ₹15,059,585 | 2025-10-28 | 9.44 | -800,560 | -5.32% | rotate |
| 12 | ITI | **Small** | Nifty Smallcap 250 | 2024-12-10 | 389.90 | 22,369 | ₹8,721,673 | 2024-12-11 | 361.25 | -640,872 | -7.35% | rotate |
| 13 | DIXON | **Mid** | Nifty Midcap 150 | 2025-08-07 | 16,663.00 | 783 | ₹13,047,129 | 2025-08-08 | 15,864.00 | -625,617 | -4.80% | rotate |
| 14 | OLAELEC | **Small** | Nifty Smallcap 250 | 2026-04-17 | 40.82 | 374,131 | ₹15,272,027 | 2026-04-20 | 39.22 | -598,610 | -3.92% | rotate |
| 15 | OLAELEC | **Small** | Nifty Smallcap 250 | 2025-09-15 | 60.51 | 213,467 | ₹12,916,888 | 2025-09-18 | 57.71 | -597,708 | -4.63% | rotate |

**Caveats**: 50% Max DD (highest of all 5). 134 trades = ~45/yr round-trip → 3-5%/yr cost drag. Y3 cooled to +34%.
