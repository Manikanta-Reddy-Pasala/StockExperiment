# Siemens Ltd. (SIEMENS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3825.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -539.87
- **Avg P&L per closed trade:** -59.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 3401.33 | 3517.33 | 3517.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 3365.63 | 3491.49 | 3502.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 3393.02 | 3389.29 | 3434.80 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 3719.97 | 3465.18 | 3465.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 3809.10 | 3499.05 | 3483.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 3659.94 | 3660.23 | 3580.94 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 3322.58 | 3530.83 | 3531.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 3315.97 | 3526.65 | 3529.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 3571.94 | 3468.80 | 3496.64 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 3722.92 | 3520.58 | 3520.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 3753.77 | 3527.03 | 3523.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 3710.64 | 3747.76 | 3663.39 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 3252.26 | 3593.37 | 3594.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 3200.48 | 3501.97 | 3545.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 2562.25 | 2555.22 | 2756.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 2543.64 | 2556.04 | 2752.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2694.14 | 2565.82 | 2731.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-26 12:15:00 | 2736.43 | 2569.04 | 2731.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 2918.50 | 2767.91 | 2767.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 2950.90 | 2773.08 | 2770.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 3208.60 | 3220.48 | 3092.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-01 09:15:00 | 3331.60 | 3207.85 | 3112.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 09:15:00 | 3147.00 | 3234.15 | 3151.81 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 3100.50 | 3117.64 | 3117.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 3085.40 | 3117.32 | 3117.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-08 14:15:00 | 3032.40 | 3113.05 | 3115.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-11 10:15:00 | 3124.00 | 3111.72 | 3114.56 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 3176.70 | 3117.14 | 3117.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3220.60 | 3123.00 | 3120.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3091.70 | 3135.55 | 3127.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-02 09:15:00 | 3170.00 | 3122.82 | 3121.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3170.00 | 3122.82 | 3121.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-02 14:15:00 | 3190.10 | 3125.84 | 3122.82 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-05 11:15:00 | 3106.20 | 3135.33 | 3128.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 3117.10 | 3156.86 | 3157.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 3104.80 | 3156.34 | 3156.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 3165.00 | 3154.33 | 3155.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 12:15:00 | 3131.10 | 3153.95 | 3155.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 3179.60 | 3112.88 | 3131.21 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 3305.00 | 3147.23 | 3146.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 3323.70 | 3152.12 | 3149.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 3196.30 | 3221.88 | 3189.53 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 3110.90 | 3170.38 | 3170.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 3099.50 | 3167.86 | 3169.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3134.80 | 3131.15 | 3147.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 3097.80 | 3130.85 | 3147.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3055.40 | 3014.78 | 3071.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-29 13:15:00 | 3075.40 | 3015.94 | 3071.44 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 3181.00 | 3104.79 | 3104.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 3226.20 | 3115.95 | 3110.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 12:15:00 | 3192.00 | 3195.95 | 3156.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 11:15:00 | 3221.50 | 3195.34 | 3157.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3195.60 | 3201.97 | 3163.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-09 14:15:00 | 3232.60 | 3201.80 | 3163.88 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-16 09:15:00 | 3154.80 | 3221.21 | 3179.69 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 2963.70 | 3151.96 | 3152.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2937.50 | 3149.83 | 3151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 3360.20 | 3149.48 | 3148.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 3504.10 | 3161.01 | 3154.48 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-20 09:15:00 | 2543.64 | 2025-03-26 12:15:00 | 2736.43 | EXIT_EMA400 | -192.79 |
| BUY | 2025-07-01 09:15:00 | 3331.60 | 2025-07-11 09:15:00 | 3147.00 | EXIT_EMA400 | -184.60 |
| SELL | 2025-08-08 14:15:00 | 3032.40 | 2025-08-11 10:15:00 | 3124.00 | EXIT_EMA400 | -91.60 |
| BUY | 2025-09-02 09:15:00 | 3170.00 | 2025-09-05 11:15:00 | 3106.20 | EXIT_EMA400 | -63.80 |
| BUY | 2025-09-02 14:15:00 | 3190.10 | 2025-09-05 11:15:00 | 3106.20 | EXIT_EMA400 | -83.90 |
| SELL | 2025-10-30 12:15:00 | 3131.10 | 2025-11-07 09:15:00 | 3058.02 | TARGET | 73.08 |
| SELL | 2026-01-08 10:15:00 | 3097.80 | 2026-01-12 10:15:00 | 2949.57 | TARGET | 148.23 |
| BUY | 2026-03-05 11:15:00 | 3221.50 | 2026-03-16 09:15:00 | 3154.80 | EXIT_EMA400 | -66.70 |
| BUY | 2026-03-09 14:15:00 | 3232.60 | 2026-03-16 09:15:00 | 3154.80 | EXIT_EMA400 | -77.80 |
