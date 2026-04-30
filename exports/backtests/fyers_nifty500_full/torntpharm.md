# Torrent Pharmaceuticals Ltd. (TORNTPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4192.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -349.46
- **Avg P&L per closed trade:** -38.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 3172.70 | 3307.54 | 3307.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 3156.80 | 3306.04 | 3306.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 3267.65 | 3233.55 | 3264.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 09:15:00 | 3224.65 | 3233.85 | 3264.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 3224.65 | 3233.85 | 3264.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-27 09:15:00 | 3215.95 | 3234.16 | 3263.88 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 3245.30 | 3229.70 | 3259.55 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 10:15:00 | 3300.00 | 3230.40 | 3259.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 3384.75 | 3281.73 | 3281.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 3397.15 | 3302.58 | 3292.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 3354.25 | 3361.20 | 3332.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-06 15:15:00 | 3380.00 | 3361.39 | 3332.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 3336.80 | 3364.72 | 3336.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-08 13:15:00 | 3332.90 | 3364.40 | 3336.20 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 3180.00 | 3315.41 | 3315.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 3172.10 | 3313.98 | 3314.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 09:15:00 | 3333.00 | 3278.88 | 3295.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 3193.55 | 3291.24 | 3299.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 3258.85 | 3253.17 | 3276.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 10:15:00 | 3218.00 | 3252.82 | 3276.48 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 3131.10 | 3082.66 | 3142.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 3147.55 | 3084.15 | 3142.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 3214.40 | 3176.12 | 3175.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 3257.60 | 3178.70 | 3177.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 3194.50 | 3206.00 | 3192.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 09:15:00 | 3310.00 | 3208.42 | 3193.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 3261.40 | 3238.55 | 3213.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-07 10:15:00 | 3281.20 | 3238.97 | 3213.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-08 13:15:00 | 3192.10 | 3241.27 | 3216.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 3140.50 | 3208.67 | 3208.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 3126.50 | 3204.26 | 3206.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 3254.40 | 3205.80 | 3205.66 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 3169.60 | 3205.62 | 3205.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3157.00 | 3205.14 | 3205.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-26 09:15:00 | 3181.10 | 3200.65 | 3202.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 3181.10 | 3200.65 | 3202.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-26 13:15:00 | 3208.00 | 3200.45 | 3202.77 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 3293.40 | 3205.21 | 3205.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 3326.70 | 3206.42 | 3205.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 3578.20 | 3583.11 | 3489.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 14:15:00 | 3647.00 | 3588.91 | 3516.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-15 10:15:00 | 3516.50 | 3587.97 | 3524.59 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 09:15:00 | 3224.65 | 2024-11-29 10:15:00 | 3300.00 | EXIT_EMA400 | -75.35 |
| SELL | 2024-11-27 09:15:00 | 3215.95 | 2024-11-29 10:15:00 | 3300.00 | EXIT_EMA400 | -84.05 |
| BUY | 2025-01-06 15:15:00 | 3380.00 | 2025-01-08 13:15:00 | 3332.90 | EXIT_EMA400 | -47.10 |
| SELL | 2025-02-10 10:15:00 | 3218.00 | 2025-02-12 09:15:00 | 3042.56 | TARGET | 175.44 |
| SELL | 2025-02-01 12:15:00 | 3193.55 | 2025-03-19 10:15:00 | 3147.55 | EXIT_EMA400 | 46.00 |
| BUY | 2025-04-28 09:15:00 | 3310.00 | 2025-05-08 13:15:00 | 3192.10 | EXIT_EMA400 | -117.90 |
| BUY | 2025-05-07 10:15:00 | 3281.20 | 2025-05-08 13:15:00 | 3192.10 | EXIT_EMA400 | -89.10 |
| SELL | 2025-06-26 09:15:00 | 3181.10 | 2025-06-26 13:15:00 | 3208.00 | EXIT_EMA400 | -26.90 |
| BUY | 2025-09-09 14:15:00 | 3647.00 | 2025-09-15 10:15:00 | 3516.50 | EXIT_EMA400 | -130.50 |
