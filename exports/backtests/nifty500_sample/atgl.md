# Adani Total Gas Ltd. (ATGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 634.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Total realized P&L (per unit):** -123.70
- **Avg P&L per closed trade:** -24.74

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-04-18 13:15:00 | ALERT1 | SELL | 934.85 | 965.89 | 966.13 | Break + close below crossover candle low |
| 2024-05-21 09:15:00 | ALERT2 | SELL | 926.60 | 921.18 | 936.84 | EMA200 retest candle locked |
| 2024-06-03 11:15:00 | CROSSOVER | BUY | 1117.15 | 946.62 | 946.61 | EMA200 above EMA400 |
| 2024-06-03 12:15:00 | ALERT1 | BUY | 1122.80 | 948.38 | 947.49 | Break + close above crossover candle high |
| 2024-06-04 10:15:00 | ALERT2 | BUY | 948.20 | 954.46 | 950.60 | EMA200 retest candle locked |
| 2024-06-21 15:15:00 | CROSSOVER | SELL | 923.00 | 949.27 | 949.31 | EMA200 below EMA400 |
| 2024-06-24 09:15:00 | ALERT1 | SELL | 919.50 | 948.98 | 949.16 | Break + close below crossover candle low |
| 2024-07-29 13:15:00 | ALERT2 | SELL | 909.00 | 903.48 | 917.62 | EMA200 retest candle locked |
| 2024-07-29 14:15:00 | ENTRY1 | SELL | 893.65 | 903.38 | 917.50 | Sell entry 1 (retest1 break) |
| 2024-07-29 14:15:00 | ALERT3 | SELL | 893.65 | 903.38 | 917.50 | EMA400 retest candle locked |
| 2024-08-01 14:15:00 | EXIT | SELL | 916.80 | 902.88 | 915.82 | Close above EMA400 |
| 2025-05-15 14:15:00 | CROSSOVER | BUY | 660.00 | 620.97 | 620.80 | EMA200 above EMA400 |
| 2025-05-15 15:15:00 | ALERT1 | BUY | 661.30 | 621.37 | 621.00 | Break + close above crossover candle high |
| 2025-06-13 09:15:00 | ALERT2 | BUY | 662.45 | 667.56 | 651.24 | EMA200 retest candle locked |
| 2025-06-27 09:15:00 | ENTRY1 | BUY | 685.80 | 653.98 | 648.26 | Buy entry 1 (retest1 break) |
| 2025-07-08 10:15:00 | EXIT | BUY | 652.15 | 658.88 | 652.40 | Close below EMA400 |
| 2025-07-29 15:15:00 | CROSSOVER | SELL | 629.50 | 649.96 | 649.98 | EMA200 below EMA400 |
| 2025-07-30 14:15:00 | ALERT1 | SELL | 626.10 | 648.77 | 649.37 | Break + close below crossover candle low |
| 2025-08-18 13:15:00 | ALERT2 | SELL | 625.20 | 623.02 | 633.54 | EMA200 retest candle locked |
| 2025-08-21 14:15:00 | ENTRY1 | SELL | 618.00 | 623.93 | 632.93 | Sell entry 1 (retest1 break) |
| 2025-08-25 12:15:00 | ALERT3 | SELL | 631.30 | 623.62 | 632.24 | EMA400 retest candle locked |
| 2025-08-26 09:15:00 | EXIT | SELL | 636.95 | 623.98 | 632.25 | Close above EMA400 |
| 2025-09-23 13:15:00 | CROSSOVER | BUY | 742.50 | 628.62 | 628.15 | EMA200 above EMA400 |
| 2025-10-20 11:15:00 | CROSSOVER | SELL | 621.00 | 631.06 | 631.08 | EMA200 below EMA400 |
| 2025-11-06 09:15:00 | ALERT1 | SELL | 618.15 | 629.71 | 630.19 | Break + close below crossover candle low |
| 2025-11-12 09:15:00 | ALERT2 | SELL | 629.90 | 625.70 | 627.99 | EMA200 retest candle locked |
| 2025-11-18 10:15:00 | ENTRY1 | SELL | 613.45 | 625.00 | 627.31 | Sell entry 1 (retest1 break) |
| 2026-01-01 09:15:00 | EXIT | SELL | 613.90 | 586.16 | 599.43 | Close above EMA400 |
| 2026-04-17 09:15:00 | CROSSOVER | BUY | 617.10 | 543.73 | 543.46 | EMA200 above EMA400 |
| 2026-04-17 14:15:00 | ALERT1 | BUY | 633.00 | 547.49 | 545.37 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2024-01-23 09:15:00 | 1026.95 | 2024-03-12 11:15:00 | 979.45 | -47.50 |
| SELL | 2024-07-29 14:15:00 | 893.65 | 2024-08-01 14:15:00 | 916.80 | -23.15 |
| BUY | 2025-06-27 09:15:00 | 685.80 | 2025-07-08 10:15:00 | 652.15 | -33.65 |
| SELL | 2025-08-21 14:15:00 | 618.00 | 2025-08-26 09:15:00 | 636.95 | -18.95 |
| SELL | 2025-11-18 10:15:00 | 613.45 | 2026-01-01 09:15:00 | 613.90 | -0.45 |
