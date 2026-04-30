# Thermax Ltd. (THERMAX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 4077.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 7 |
| EXIT | 5 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 7 / 6
- **Target hits / EMA400 exits:** 6 / 7
- **Total realized P&L (per unit):** 1202.00
- **Avg P&L per closed trade:** 92.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 2572.70 | 2826.12 | 2827.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 14:15:00 | 2513.80 | 2812.29 | 2820.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 2777.00 | 2743.13 | 2775.21 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 10:15:00 | 3010.00 | 2802.30 | 2802.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 11:15:00 | 3036.15 | 2804.63 | 2803.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 3034.00 | 3059.00 | 2969.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-25 09:15:00 | 3111.50 | 3057.57 | 2980.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-18 09:15:00 | 4999.95 | 5238.26 | 5025.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 4383.60 | 4940.81 | 4942.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 4211.00 | 4916.61 | 4930.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 4534.65 | 4499.40 | 4638.09 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 14:15:00 | 5238.00 | 4735.83 | 4735.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 5315.10 | 4768.73 | 4752.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 4896.05 | 4896.98 | 4828.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-04 11:15:00 | 5025.00 | 4898.64 | 4829.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-30 14:15:00 | 4974.35 | 5086.13 | 4981.61 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 11:15:00 | 4403.15 | 4947.48 | 4948.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 4370.05 | 4725.34 | 4798.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 3418.25 | 3390.54 | 3680.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 13:15:00 | 3339.10 | 3389.93 | 3674.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 3575.80 | 3365.84 | 3598.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-20 14:15:00 | 3524.00 | 3367.41 | 3597.91 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-25 09:15:00 | 3596.40 | 3388.54 | 3591.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 13:15:00 | 3563.00 | 3467.87 | 3467.64 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 11:15:00 | 3443.00 | 3471.27 | 3471.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 3382.00 | 3467.39 | 3469.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3489.30 | 3458.09 | 3464.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-08 10:15:00 | 3406.60 | 3459.49 | 3464.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3452.90 | 3458.29 | 3463.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-11 09:15:00 | 3416.10 | 3454.17 | 3461.28 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-11 11:15:00 | 3462.50 | 3454.08 | 3461.16 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 3649.80 | 3466.99 | 3466.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 3811.20 | 3470.41 | 3468.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 11:15:00 | 3620.10 | 3677.65 | 3596.48 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 3272.00 | 3538.87 | 3538.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 3265.00 | 3526.10 | 3532.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 3361.00 | 3360.51 | 3425.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-08 13:15:00 | 3331.50 | 3360.22 | 3425.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3403.00 | 3358.61 | 3421.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-10 10:15:00 | 3382.50 | 3358.85 | 3421.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 3350.10 | 3342.40 | 3395.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 09:15:00 | 3282.50 | 3341.82 | 3391.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3271.70 | 3223.59 | 3287.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-27 12:15:00 | 3261.40 | 3224.90 | 3287.46 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 3285.00 | 3225.49 | 3287.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 14:15:00 | 3311.40 | 3226.35 | 3287.57 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 3187.90 | 2986.14 | 2985.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 3205.20 | 2992.24 | 2988.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 2994.20 | 3014.00 | 3000.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 09:15:00 | 3060.40 | 3014.22 | 3000.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3051.30 | 3026.55 | 3008.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-10 09:15:00 | 3115.00 | 3029.80 | 3010.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 3077.90 | 3120.40 | 3067.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-23 11:15:00 | 3102.50 | 3120.22 | 3067.80 | Buy entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-25 09:15:00 | 3111.50 | 2024-02-19 09:15:00 | 3505.67 | TARGET | 394.17 |
| BUY | 2024-10-04 11:15:00 | 5025.00 | 2024-10-28 09:15:00 | 5611.04 | TARGET | 586.04 |
| SELL | 2025-03-10 13:15:00 | 3339.10 | 2025-03-25 09:15:00 | 3596.40 | EXIT_EMA400 | -257.30 |
| SELL | 2025-03-20 14:15:00 | 3524.00 | 2025-03-25 09:15:00 | 3596.40 | EXIT_EMA400 | -72.40 |
| SELL | 2025-07-08 10:15:00 | 3406.60 | 2025-07-11 11:15:00 | 3462.50 | EXIT_EMA400 | -55.90 |
| SELL | 2025-07-11 09:15:00 | 3416.10 | 2025-07-11 11:15:00 | 3462.50 | EXIT_EMA400 | -46.40 |
| SELL | 2025-09-10 10:15:00 | 3382.50 | 2025-09-22 14:15:00 | 3266.76 | TARGET | 115.74 |
| SELL | 2025-09-08 13:15:00 | 3331.50 | 2025-10-27 14:15:00 | 3311.40 | EXIT_EMA400 | 20.10 |
| SELL | 2025-09-25 09:15:00 | 3282.50 | 2025-10-27 14:15:00 | 3311.40 | EXIT_EMA400 | -28.90 |
| SELL | 2025-10-27 12:15:00 | 3261.40 | 2025-10-27 14:15:00 | 3311.40 | EXIT_EMA400 | -50.00 |
| BUY | 2026-03-05 09:15:00 | 3060.40 | 2026-03-16 09:15:00 | 3239.16 | TARGET | 178.76 |
| BUY | 2026-03-23 11:15:00 | 3102.50 | 2026-03-24 12:15:00 | 3206.61 | TARGET | 104.11 |
| BUY | 2026-03-10 09:15:00 | 3115.00 | 2026-04-09 10:15:00 | 3428.97 | TARGET | 313.97 |
