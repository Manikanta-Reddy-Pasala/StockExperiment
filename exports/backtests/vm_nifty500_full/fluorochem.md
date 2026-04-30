# Gujarat Fluorochemicals Ltd. (FLUOROCHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3602.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 19 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 0 / 13
- **Target hits / EMA400 exits:** 0 / 13
- **Total realized P&L (per unit):** -1219.65
- **Avg P&L per closed trade:** -93.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 3133.70 | 2965.43 | 2965.14 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 2817.90 | 2975.15 | 2975.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 15:15:00 | 2797.25 | 2956.94 | 2966.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 11:15:00 | 2794.95 | 2792.77 | 2851.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-15 11:15:00 | 2776.45 | 2793.37 | 2849.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-24 13:15:00 | 2848.20 | 2794.03 | 2837.28 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 11:15:00 | 3144.00 | 2862.18 | 2861.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 12:15:00 | 3160.05 | 2865.14 | 2863.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 3360.05 | 3394.71 | 3201.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 3489.80 | 3394.54 | 3208.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-09 09:15:00 | 3384.20 | 3536.89 | 3389.45 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 12:15:00 | 3101.75 | 3415.89 | 3416.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 3585.05 | 3377.43 | 3377.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 3728.85 | 3380.93 | 3378.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 13:15:00 | 3492.35 | 3521.69 | 3464.29 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 3204.50 | 3419.39 | 3419.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 3195.45 | 3417.16 | 3418.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 14:15:00 | 3175.05 | 3164.54 | 3260.66 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 3384.80 | 3269.45 | 3268.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 12:15:00 | 3416.35 | 3272.01 | 3270.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 11:15:00 | 3250.95 | 3299.01 | 3284.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-14 15:15:00 | 3345.60 | 3298.37 | 3284.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 3345.60 | 3298.37 | 3284.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-16 09:15:00 | 3369.80 | 3299.08 | 3285.34 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-20 13:15:00 | 3290.00 | 3310.28 | 3292.46 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 3165.90 | 3278.51 | 3278.76 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 3761.95 | 3281.48 | 3279.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 12:15:00 | 3799.00 | 3286.63 | 3281.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 3969.55 | 3994.65 | 3753.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 4078.00 | 3993.85 | 3761.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 4073.60 | 4309.11 | 4056.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-29 13:15:00 | 4199.00 | 4297.21 | 4057.70 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 4152.15 | 4277.34 | 4101.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-11-11 10:15:00 | 4180.00 | 4276.37 | 4101.40 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-11-13 09:15:00 | 4010.00 | 4262.96 | 4105.54 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3659.00 | 4130.25 | 4130.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3633.65 | 4125.31 | 4127.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 3782.60 | 3753.23 | 3875.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 3615.30 | 3751.71 | 3873.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 3773.55 | 3677.31 | 3774.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-10 09:15:00 | 3817.00 | 3680.75 | 3774.14 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 4019.95 | 3822.72 | 3822.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 4039.00 | 3830.38 | 3825.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 3715.00 | 3845.74 | 3834.52 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 09:15:00 | 3652.70 | 3822.62 | 3823.38 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 3921.00 | 3822.47 | 3822.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 3956.60 | 3826.46 | 3824.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 3850.20 | 3875.64 | 3853.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-06 10:15:00 | 3907.00 | 3873.81 | 3854.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 3907.00 | 3873.81 | 3854.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-06 13:15:00 | 3853.10 | 3873.67 | 3854.55 | Close below EMA400 |

### Cycle 14 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3640.10 | 3860.58 | 3861.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 3632.70 | 3858.31 | 3860.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 3635.70 | 3633.55 | 3716.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 09:15:00 | 3575.00 | 3632.29 | 3714.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-22 09:15:00 | 3619.70 | 3526.54 | 3614.92 | Close above EMA400 |

### Cycle 15 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 3805.60 | 3543.38 | 3543.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 3837.00 | 3546.30 | 3544.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 3618.50 | 3647.13 | 3606.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 09:15:00 | 3683.90 | 3647.10 | 3607.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-15 14:15:00 | 3619.40 | 3660.87 | 3623.71 | Close below EMA400 |

### Cycle 16 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 3506.90 | 3625.59 | 3625.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 3490.10 | 3620.97 | 3623.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 3485.60 | 3478.31 | 3536.58 | EMA200 retest candle locked |

### Cycle 17 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 3658.10 | 3554.63 | 3554.54 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 3493.70 | 3554.73 | 3554.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 3475.00 | 3552.55 | 3553.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 3346.30 | 3302.55 | 3400.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 3240.10 | 3382.87 | 3406.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 3270.00 | 3242.37 | 3309.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-27 09:15:00 | 3210.60 | 3242.08 | 3308.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-07 11:15:00 | 3282.90 | 3211.53 | 3279.95 | Close above EMA400 |

### Cycle 19 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 3590.10 | 3315.23 | 3313.99 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-15 11:15:00 | 2776.45 | 2023-11-24 13:15:00 | 2848.20 | EXIT_EMA400 | -71.75 |
| BUY | 2024-01-11 09:15:00 | 3489.80 | 2024-02-09 09:15:00 | 3384.20 | EXIT_EMA400 | -105.60 |
| BUY | 2024-08-14 15:15:00 | 3345.60 | 2024-08-20 13:15:00 | 3290.00 | EXIT_EMA400 | -55.60 |
| BUY | 2024-08-16 09:15:00 | 3369.80 | 2024-08-20 13:15:00 | 3290.00 | EXIT_EMA400 | -79.80 |
| BUY | 2024-10-08 10:15:00 | 4078.00 | 2024-11-13 09:15:00 | 4010.00 | EXIT_EMA400 | -68.00 |
| BUY | 2024-10-29 13:15:00 | 4199.00 | 2024-11-13 09:15:00 | 4010.00 | EXIT_EMA400 | -189.00 |
| BUY | 2024-11-11 10:15:00 | 4180.00 | 2024-11-13 09:15:00 | 4010.00 | EXIT_EMA400 | -170.00 |
| SELL | 2025-02-12 09:15:00 | 3615.30 | 2025-03-10 09:15:00 | 3817.00 | EXIT_EMA400 | -201.70 |
| BUY | 2025-05-06 10:15:00 | 3907.00 | 2025-05-06 13:15:00 | 3853.10 | EXIT_EMA400 | -53.90 |
| SELL | 2025-07-01 09:15:00 | 3575.00 | 2025-07-22 09:15:00 | 3619.70 | EXIT_EMA400 | -44.70 |
| BUY | 2025-10-07 09:15:00 | 3683.90 | 2025-10-15 14:15:00 | 3619.40 | EXIT_EMA400 | -64.50 |
| SELL | 2026-03-04 09:15:00 | 3240.10 | 2026-04-07 11:15:00 | 3282.90 | EXIT_EMA400 | -42.80 |
| SELL | 2026-03-27 09:15:00 | 3210.60 | 2026-04-07 11:15:00 | 3282.90 | EXIT_EMA400 | -72.30 |
