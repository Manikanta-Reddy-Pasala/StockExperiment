# AIA Engineering Ltd. (AIAENG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3949.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Total realized P&L (per unit):** 1044.70
- **Avg P&L per closed trade:** 130.59

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-06-18 15:15:00 | CROSSOVER | BUY | 4121.60 | 3786.83 | 3786.47 | EMA200 above EMA400 |
| 2024-06-19 09:15:00 | ALERT1 | BUY | 4175.00 | 3790.69 | 3788.41 | Break + close above crossover candle high |
| 2024-08-21 13:15:00 | ALERT2 | BUY | 4449.55 | 4457.60 | 4297.68 | EMA200 retest candle locked |
| 2024-10-09 14:15:00 | CROSSOVER | SELL | 4098.10 | 4302.00 | 4302.11 | EMA200 below EMA400 |
| 2024-10-10 13:15:00 | ALERT1 | SELL | 4094.25 | 4290.75 | 4296.39 | Break + close below crossover candle low |
| 2024-10-17 12:15:00 | ALERT2 | SELL | 4247.45 | 4245.16 | 4270.40 | EMA200 retest candle locked |
| 2024-10-18 09:15:00 | ENTRY1 | SELL | 4182.05 | 4244.39 | 4269.51 | Sell entry 1 (retest1 break) |
| 2025-01-20 12:15:00 | EXIT | SELL | 3560.00 | 3437.36 | 3554.19 | Close above EMA400 |
| 2025-05-30 12:15:00 | CROSSOVER | BUY | 3502.20 | 3286.28 | 3285.51 | EMA200 above EMA400 |
| 2025-07-29 12:15:00 | CROSSOVER | SELL | 3176.60 | 3344.52 | 3345.29 | EMA200 below EMA400 |
| 2025-07-31 09:15:00 | ALERT1 | SELL | 3163.00 | 3328.43 | 3336.98 | Break + close below crossover candle low |
| 2025-09-15 10:15:00 | ALERT2 | SELL | 3108.40 | 3107.19 | 3171.70 | EMA200 retest candle locked |
| 2025-09-19 14:15:00 | ENTRY1 | SELL | 3069.80 | 3110.49 | 3163.91 | Sell entry 1 (retest1 break) |
| 2025-10-01 10:15:00 | EXIT | SELL | 3148.70 | 3095.70 | 3143.24 | Close above EMA400 |
| 2025-10-17 11:15:00 | CROSSOVER | BUY | 3310.00 | 3172.91 | 3172.73 | EMA200 above EMA400 |
| 2025-10-20 11:15:00 | ALERT1 | BUY | 3315.10 | 3181.91 | 3177.32 | Break + close above crossover candle high |
| 2025-11-03 10:15:00 | ALERT2 | BUY | 3243.00 | 3243.60 | 3214.18 | EMA200 retest candle locked |
| 2025-11-03 11:15:00 | ENTRY1 | BUY | 3260.10 | 3243.76 | 3214.41 | Buy entry 1 (retest1 break) |
| 2025-11-07 09:15:00 | ALERT3 | BUY | 3228.50 | 3253.52 | 3222.18 | EMA400 retest candle locked |
| 2025-11-10 09:15:00 | ENTRY2 | BUY | 3380.30 | 3254.86 | 3223.93 | Buy entry 2 (retest2 break) |
| 2026-01-20 14:15:00 | EXIT | BUY | 3740.10 | 3889.56 | 3763.94 | Close below EMA400 |
| 2026-03-12 14:15:00 | CROSSOVER | SELL | 3712.90 | 3816.45 | 3816.53 | EMA200 below EMA400 |
| 2026-03-13 09:15:00 | ALERT1 | SELL | 3661.20 | 3813.98 | 3815.29 | Break + close below crossover candle low |
| 2026-03-17 09:15:00 | ALERT2 | SELL | 3900.00 | 3796.58 | 3806.16 | EMA200 retest candle locked |
| 2026-03-17 14:15:00 | ENTRY1 | SELL | 3721.10 | 3797.22 | 3806.25 | Sell entry 1 (retest1 break) |
| 2026-04-08 09:15:00 | ALERT3 | SELL | 3677.00 | 3666.55 | 3721.58 | EMA400 retest candle locked |
| 2026-04-09 09:15:00 | ENTRY2 | SELL | 3668.10 | 3667.26 | 3720.05 | Sell entry 2 (retest2 break) |
| 2026-04-10 09:15:00 | EXIT | SELL | 3780.90 | 3668.07 | 3718.65 | Close above EMA400 |
| 2026-04-21 14:15:00 | CROSSOVER | BUY | 4004.50 | 3757.47 | 3757.36 | EMA200 above EMA400 |
| 2026-04-22 15:15:00 | ALERT1 | BUY | 4075.00 | 3778.13 | 3767.90 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| BUY | 2023-11-29 13:15:00 | 3615.00 | 2023-12-18 09:15:00 | 3572.75 | -42.25 |
| BUY | 2024-04-24 09:15:00 | 3919.90 | 2024-04-29 09:15:00 | 3796.50 | -123.40 |
| SELL | 2024-10-18 09:15:00 | 4182.05 | 2025-01-20 12:15:00 | 3560.00 | 622.05 |
| SELL | 2025-09-19 14:15:00 | 3069.80 | 2025-10-01 10:15:00 | 3148.70 | -78.90 |
| BUY | 2025-11-03 11:15:00 | 3260.10 | 2026-01-20 14:15:00 | 3740.10 | 480.00 |
| BUY | 2025-11-10 09:15:00 | 3380.30 | 2026-01-20 14:15:00 | 3740.10 | 359.80 |
| SELL | 2026-03-17 14:15:00 | 3721.10 | 2026-04-10 09:15:00 | 3780.90 | -59.80 |
| SELL | 2026-04-09 09:15:00 | 3668.10 | 2026-04-10 09:15:00 | 3780.90 | -112.80 |
