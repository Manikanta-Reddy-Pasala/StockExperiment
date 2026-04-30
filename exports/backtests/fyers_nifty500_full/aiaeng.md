# AIA Engineering Ltd. (AIAENG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3951.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 933.72
- **Avg P&L per closed trade:** 155.62

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 4122.05 | 4297.92 | 4298.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 4094.25 | 4290.42 | 4294.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 4249.10 | 4244.67 | 4268.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-18 09:15:00 | 4182.05 | 4243.86 | 4268.06 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-20 12:15:00 | 3560.00 | 3437.39 | 3553.74 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 3502.20 | 3286.12 | 3285.53 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 3176.60 | 3345.00 | 3345.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3163.00 | 3328.80 | 3337.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 3108.40 | 3107.12 | 3171.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 3069.40 | 3110.26 | 3163.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-01 10:15:00 | 3148.70 | 3095.54 | 3143.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 3310.00 | 3172.80 | 3172.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 15:15:00 | 3315.00 | 3178.15 | 3175.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 10:15:00 | 3243.30 | 3243.56 | 3214.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 11:15:00 | 3260.00 | 3243.73 | 3214.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3228.50 | 3253.33 | 3222.08 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-10 09:15:00 | 3380.90 | 3254.73 | 3223.85 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-20 14:15:00 | 3740.10 | 3889.42 | 3763.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 3703.80 | 3818.26 | 3818.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 3678.00 | 3814.77 | 3816.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 14:15:00 | 3721.10 | 3796.60 | 3807.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3677.00 | 3665.92 | 3721.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-08 15:15:00 | 3670.00 | 3666.61 | 3720.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 3780.90 | 3667.60 | 3718.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 4015.80 | 3759.10 | 3758.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 4044.80 | 3764.49 | 3761.29 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-18 09:15:00 | 4182.05 | 2024-10-23 09:15:00 | 3924.01 | TARGET | 258.04 |
| SELL | 2025-09-19 14:15:00 | 3069.40 | 2025-10-01 10:15:00 | 3148.70 | EXIT_EMA400 | -79.30 |
| BUY | 2025-11-03 11:15:00 | 3260.00 | 2025-11-10 09:15:00 | 3396.84 | TARGET | 136.84 |
| BUY | 2025-11-10 09:15:00 | 3380.90 | 2025-11-26 09:15:00 | 3852.04 | TARGET | 471.14 |
| SELL | 2026-03-17 14:15:00 | 3721.10 | 2026-03-20 14:15:00 | 3463.21 | TARGET | 257.89 |
| SELL | 2026-04-08 15:15:00 | 3670.00 | 2026-04-10 09:15:00 | 3780.90 | EXIT_EMA400 | -110.90 |
