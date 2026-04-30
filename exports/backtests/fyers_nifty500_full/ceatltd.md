# Ceat Ltd. (CEATLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3450.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -354.78
- **Avg P&L per closed trade:** -59.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 2762.00 | 2842.31 | 2842.40 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 15:15:00 | 2932.00 | 2841.42 | 2841.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 10:15:00 | 2954.75 | 2843.47 | 2842.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 3044.45 | 3053.28 | 2977.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-26 09:15:00 | 3076.00 | 3050.78 | 2982.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 3067.90 | 3106.03 | 3040.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-14 14:15:00 | 3030.60 | 3100.82 | 3042.00 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 11:15:00 | 2916.45 | 3007.07 | 3007.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 12:15:00 | 2897.35 | 3005.98 | 3006.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 3042.35 | 2989.71 | 2998.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 2933.50 | 2993.04 | 2999.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 2824.25 | 2682.04 | 2773.59 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 14:15:00 | 3024.00 | 2812.98 | 2812.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 13:15:00 | 3051.00 | 2850.51 | 2832.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 3648.90 | 3664.51 | 3457.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 09:15:00 | 3729.20 | 3637.09 | 3527.06 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 3591.50 | 3730.40 | 3610.47 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 3245.30 | 3535.01 | 3535.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 3235.10 | 3529.21 | 3532.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 3293.50 | 3273.04 | 3363.70 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 3457.40 | 3388.68 | 3388.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 3474.10 | 3391.54 | 3390.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 3874.00 | 3902.36 | 3742.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-03 09:15:00 | 3962.70 | 3890.95 | 3784.02 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 3765.00 | 3890.89 | 3798.17 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 3710.00 | 3790.93 | 3791.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 3681.00 | 3787.10 | 3789.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 3684.70 | 3766.44 | 3777.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 3787.60 | 3763.55 | 3775.87 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 3878.10 | 3787.59 | 3787.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 3920.40 | 3791.20 | 3789.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 3851.30 | 3861.66 | 3828.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 09:15:00 | 3888.10 | 3861.98 | 3829.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-19 14:15:00 | 3820.40 | 3865.04 | 3834.08 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3460.00 | 3808.15 | 3809.54 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-26 09:15:00 | 3076.00 | 2025-01-14 14:15:00 | 3030.60 | EXIT_EMA400 | -45.40 |
| SELL | 2025-02-06 09:15:00 | 2933.50 | 2025-02-11 14:15:00 | 2736.88 | TARGET | 196.62 |
| BUY | 2025-07-08 09:15:00 | 3729.20 | 2025-07-21 09:15:00 | 3591.50 | EXIT_EMA400 | -137.70 |
| BUY | 2025-12-03 09:15:00 | 3962.70 | 2025-12-09 09:15:00 | 3765.00 | EXIT_EMA400 | -197.70 |
| SELL | 2026-02-02 09:15:00 | 3684.70 | 2026-02-03 09:15:00 | 3787.60 | EXIT_EMA400 | -102.90 |
| BUY | 2026-02-17 09:15:00 | 3888.10 | 2026-02-19 14:15:00 | 3820.40 | EXIT_EMA400 | -67.70 |
