# Escorts Kubota Ltd. (ESCORTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5006 bars)
- **Last close:** 3241.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -568.68
- **Avg P&L per closed trade:** -63.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 12:15:00 | 2992.85 | 3100.94 | 3101.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 13:15:00 | 2978.00 | 3099.72 | 3100.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 10:15:00 | 2971.00 | 2963.49 | 3014.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-23 09:15:00 | 2904.00 | 2963.32 | 3013.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-01-29 14:15:00 | 3023.25 | 2954.12 | 3001.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 14:15:00 | 2997.60 | 2904.80 | 2904.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 09:15:00 | 3053.00 | 2907.21 | 2905.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 4041.50 | 4050.91 | 3813.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-08 11:15:00 | 4086.80 | 4051.26 | 3814.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 3980.05 | 4025.66 | 3863.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-22 10:15:00 | 3995.95 | 4025.37 | 3864.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 3943.00 | 4068.37 | 3933.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-02 13:15:00 | 3925.00 | 4066.94 | 3933.15 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 3791.95 | 3855.00 | 3855.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 3783.45 | 3850.75 | 3853.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 11:15:00 | 3813.00 | 3811.77 | 3830.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 11:15:00 | 3774.70 | 3818.78 | 3831.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 3803.50 | 3818.39 | 3831.55 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-20 09:15:00 | 3916.70 | 3819.38 | 3831.85 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 4202.75 | 3846.26 | 3844.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 4228.20 | 3853.59 | 3848.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 4019.50 | 4023.18 | 3948.05 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 3495.75 | 3912.14 | 3912.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 3342.85 | 3544.12 | 3639.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 3403.25 | 3390.76 | 3515.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-03 13:15:00 | 3364.90 | 3390.35 | 3512.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-15 10:15:00 | 3482.40 | 3381.59 | 3478.96 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 3345.00 | 3222.23 | 3222.04 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 3130.40 | 3225.57 | 3225.89 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3358.80 | 3226.89 | 3226.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3388.30 | 3232.15 | 3229.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 3378.20 | 3389.84 | 3326.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 12:15:00 | 3416.90 | 3381.39 | 3327.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-03 11:15:00 | 3318.10 | 3380.03 | 3328.51 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 3135.80 | 3296.58 | 3296.76 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 3363.30 | 3295.62 | 3295.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 3383.90 | 3314.54 | 3306.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 3312.50 | 3322.64 | 3311.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 3329.80 | 3318.09 | 3309.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3329.80 | 3318.09 | 3309.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 10:15:00 | 3355.70 | 3318.46 | 3309.88 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3352.60 | 3381.84 | 3350.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 10:15:00 | 3331.40 | 3381.02 | 3350.81 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 3452.20 | 3680.45 | 3681.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 3428.20 | 3673.96 | 3678.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 3614.80 | 3607.77 | 3641.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 14:15:00 | 3583.70 | 3649.24 | 3656.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-25 11:15:00 | 3640.00 | 3593.59 | 3623.33 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-23 09:15:00 | 2904.00 | 2024-01-29 14:15:00 | 3023.25 | EXIT_EMA400 | -119.25 |
| BUY | 2024-07-08 11:15:00 | 4086.80 | 2024-08-02 13:15:00 | 3925.00 | EXIT_EMA400 | -161.80 |
| BUY | 2024-07-22 10:15:00 | 3995.95 | 2024-08-02 13:15:00 | 3925.00 | EXIT_EMA400 | -70.95 |
| SELL | 2024-09-19 11:15:00 | 3774.70 | 2024-09-20 09:15:00 | 3916.70 | EXIT_EMA400 | -142.00 |
| SELL | 2025-01-03 13:15:00 | 3364.90 | 2025-01-15 10:15:00 | 3482.40 | EXIT_EMA400 | -117.50 |
| BUY | 2025-06-02 12:15:00 | 3416.90 | 2025-06-03 11:15:00 | 3318.10 | EXIT_EMA400 | -98.80 |
| BUY | 2025-07-15 09:15:00 | 3329.80 | 2025-07-17 09:15:00 | 3390.26 | TARGET | 60.46 |
| BUY | 2025-07-15 10:15:00 | 3355.70 | 2025-07-24 09:15:00 | 3493.17 | TARGET | 137.47 |
| SELL | 2026-02-13 14:15:00 | 3583.70 | 2026-02-25 11:15:00 | 3640.00 | EXIT_EMA400 | -56.30 |
