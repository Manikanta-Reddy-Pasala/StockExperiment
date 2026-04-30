# Torrent Pharmaceuticals Ltd. (TORNTPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 4185.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -223.52
- **Avg P&L per closed trade:** -22.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 10:15:00 | 1847.50 | 1914.54 | 1914.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-14 13:15:00 | 1840.75 | 1912.61 | 1913.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1887.15 | 1886.04 | 1898.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-04 09:15:00 | 1861.50 | 1888.53 | 1898.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 1894.05 | 1884.42 | 1895.48 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-10-09 10:15:00 | 1883.45 | 1884.50 | 1895.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-10-09 11:15:00 | 1902.00 | 1884.67 | 1895.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 1924.35 | 1900.37 | 1900.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 15:15:00 | 1930.50 | 1901.59 | 1900.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 2051.50 | 2053.30 | 2003.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-11 13:15:00 | 2069.10 | 2053.69 | 2005.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 2517.40 | 2612.81 | 2514.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-19 10:15:00 | 2506.95 | 2611.75 | 2514.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 3150.90 | 3306.94 | 3307.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 3120.45 | 3294.35 | 3300.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 3267.55 | 3235.00 | 3265.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 09:15:00 | 3215.95 | 3235.53 | 3264.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 3245.95 | 3231.01 | 3260.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 10:15:00 | 3300.00 | 3231.69 | 3260.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 3374.15 | 3283.15 | 3282.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 3397.15 | 3303.03 | 3293.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 3353.45 | 3361.16 | 3332.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-06 15:15:00 | 3380.00 | 3361.35 | 3333.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 3339.00 | 3364.72 | 3336.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-08 13:15:00 | 3333.55 | 3364.41 | 3336.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 3180.00 | 3315.21 | 3315.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 3172.10 | 3313.79 | 3314.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 09:15:00 | 3333.00 | 3278.61 | 3295.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 13:15:00 | 3259.55 | 3292.93 | 3300.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 3131.65 | 3083.52 | 3143.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 3147.55 | 3085.04 | 3143.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 3220.00 | 3177.10 | 3176.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 3239.60 | 3178.06 | 3177.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 3194.50 | 3206.08 | 3192.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 09:15:00 | 3309.50 | 3208.56 | 3194.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 3261.40 | 3238.28 | 3213.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-07 10:15:00 | 3281.20 | 3238.71 | 3213.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-08 13:15:00 | 3192.10 | 3240.98 | 3216.32 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 3140.40 | 3209.00 | 3209.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 3126.50 | 3204.57 | 3207.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 3212.70 | 3190.68 | 3199.14 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 3254.40 | 3205.82 | 3205.82 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 3169.60 | 3205.65 | 3205.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3157.00 | 3205.16 | 3205.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 3214.70 | 3199.82 | 3202.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-26 09:15:00 | 3181.10 | 3200.74 | 3203.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 3181.10 | 3200.74 | 3203.08 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-26 13:15:00 | 3208.00 | 3200.54 | 3202.93 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 3293.40 | 3205.30 | 3205.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 3326.70 | 3206.51 | 3205.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 3578.20 | 3583.21 | 3489.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 13:15:00 | 3631.10 | 3588.34 | 3516.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-15 10:15:00 | 3516.50 | 3588.12 | 3524.68 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-04 09:15:00 | 1861.50 | 2023-10-09 11:15:00 | 1902.00 | EXIT_EMA400 | -40.50 |
| SELL | 2023-10-09 10:15:00 | 1883.45 | 2023-10-09 11:15:00 | 1902.00 | EXIT_EMA400 | -18.55 |
| BUY | 2023-12-11 13:15:00 | 2069.10 | 2023-12-26 11:15:00 | 2260.23 | TARGET | 191.13 |
| SELL | 2024-11-27 09:15:00 | 3215.95 | 2024-11-29 10:15:00 | 3300.00 | EXIT_EMA400 | -84.05 |
| BUY | 2025-01-06 15:15:00 | 3380.00 | 2025-01-08 13:15:00 | 3333.55 | EXIT_EMA400 | -46.45 |
| SELL | 2025-01-31 13:15:00 | 3259.55 | 2025-02-03 09:15:00 | 3136.65 | TARGET | 122.90 |
| BUY | 2025-04-28 09:15:00 | 3309.50 | 2025-05-08 13:15:00 | 3192.10 | EXIT_EMA400 | -117.40 |
| BUY | 2025-05-07 10:15:00 | 3281.20 | 2025-05-08 13:15:00 | 3192.10 | EXIT_EMA400 | -89.10 |
| SELL | 2025-06-26 09:15:00 | 3181.10 | 2025-06-26 13:15:00 | 3208.00 | EXIT_EMA400 | -26.90 |
| BUY | 2025-09-09 13:15:00 | 3631.10 | 2025-09-15 10:15:00 | 3516.50 | EXIT_EMA400 | -114.60 |
