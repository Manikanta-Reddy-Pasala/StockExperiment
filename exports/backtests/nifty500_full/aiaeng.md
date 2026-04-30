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
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 766.67
- **Avg P&L per closed trade:** 95.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 15:15:00 | 3428.55 | 3486.02 | 3486.24 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 3550.10 | 3487.06 | 3486.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 10:15:00 | 3620.00 | 3491.01 | 3488.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 13:15:00 | 3559.70 | 3559.88 | 3531.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-29 13:15:00 | 3615.00 | 3540.32 | 3526.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-18 09:15:00 | 3572.75 | 3610.39 | 3574.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 3620.15 | 3764.59 | 3764.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 12:15:00 | 3615.15 | 3763.10 | 3764.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 3744.20 | 3725.85 | 3743.69 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 3962.65 | 3757.50 | 3757.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 3993.50 | 3759.85 | 3758.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 10:15:00 | 3859.85 | 3873.54 | 3824.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-24 09:15:00 | 3919.90 | 3862.26 | 3826.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-29 09:15:00 | 3796.50 | 3867.51 | 3833.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 3740.30 | 3812.89 | 3812.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 3693.95 | 3806.42 | 3809.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 15:15:00 | 3739.00 | 3734.02 | 3764.20 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 4121.60 | 3786.83 | 3786.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 4175.00 | 3790.69 | 3788.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 4449.55 | 4457.60 | 4297.68 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 4098.10 | 4302.00 | 4302.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 4094.25 | 4290.75 | 4296.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 4247.45 | 4245.16 | 4270.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-18 09:15:00 | 4182.05 | 4244.39 | 4269.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-20 12:15:00 | 3560.00 | 3437.36 | 3554.19 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 3502.20 | 3286.28 | 3285.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 3176.60 | 3344.52 | 3345.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3163.00 | 3328.43 | 3336.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 3108.40 | 3107.19 | 3171.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 3069.80 | 3110.49 | 3163.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-01 10:15:00 | 3148.70 | 3095.70 | 3143.24 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 3310.00 | 3172.91 | 3172.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 3315.10 | 3181.91 | 3177.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 10:15:00 | 3243.00 | 3243.60 | 3214.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 11:15:00 | 3260.10 | 3243.76 | 3214.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3228.50 | 3253.52 | 3222.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-10 09:15:00 | 3380.30 | 3254.86 | 3223.93 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-20 14:15:00 | 3740.10 | 3889.56 | 3763.94 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 3712.90 | 3816.45 | 3816.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 3661.20 | 3813.98 | 3815.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3900.00 | 3796.58 | 3806.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 14:15:00 | 3721.10 | 3797.22 | 3806.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3677.00 | 3666.55 | 3721.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 3668.10 | 3667.26 | 3720.05 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 3780.90 | 3668.07 | 3718.65 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 4004.50 | 3757.47 | 3757.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 15:15:00 | 4075.00 | 3778.13 | 3767.90 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-29 13:15:00 | 3615.00 | 2023-12-18 09:15:00 | 3572.75 | EXIT_EMA400 | -42.25 |
| BUY | 2024-04-24 09:15:00 | 3919.90 | 2024-04-29 09:15:00 | 3796.50 | EXIT_EMA400 | -123.40 |
| SELL | 2024-10-18 09:15:00 | 4182.05 | 2024-10-23 09:15:00 | 3919.68 | TARGET | 262.37 |
| SELL | 2025-09-19 14:15:00 | 3069.80 | 2025-10-01 10:15:00 | 3148.70 | EXIT_EMA400 | -78.90 |
| BUY | 2025-11-03 11:15:00 | 3260.10 | 2025-11-10 09:15:00 | 3397.17 | TARGET | 137.07 |
| BUY | 2025-11-10 09:15:00 | 3380.30 | 2025-11-20 15:15:00 | 3849.42 | TARGET | 469.12 |
| SELL | 2026-03-17 14:15:00 | 3721.10 | 2026-03-20 14:15:00 | 3465.64 | TARGET | 255.46 |
| SELL | 2026-04-09 09:15:00 | 3668.10 | 2026-04-10 09:15:00 | 3780.90 | EXIT_EMA400 | -112.80 |
