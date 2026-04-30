# Tube Investments of India Ltd. (TIINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2947.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 795.20
- **Avg P&L per closed trade:** 99.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 11:15:00 | 3290.05 | 2990.83 | 2989.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 15:15:00 | 3313.00 | 3003.06 | 2995.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 3131.05 | 3162.88 | 3096.30 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 2951.00 | 3057.26 | 3057.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 2948.00 | 3045.33 | 3051.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 3032.05 | 3018.19 | 3035.64 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 3199.15 | 3049.77 | 3049.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 3215.10 | 3057.21 | 3053.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 11:15:00 | 3509.00 | 3509.13 | 3383.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-04 15:15:00 | 3559.65 | 3509.32 | 3390.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 3626.00 | 3774.23 | 3622.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-05 09:15:00 | 3660.60 | 3773.10 | 3622.76 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-02-05 12:15:00 | 3617.85 | 3769.14 | 3623.01 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 3493.80 | 3582.10 | 3582.17 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 3694.80 | 3582.23 | 3582.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 3748.55 | 3587.35 | 3584.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 3599.00 | 3645.28 | 3618.28 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 3516.60 | 3598.94 | 3599.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 3493.70 | 3597.90 | 3598.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 3600.15 | 3585.91 | 3592.22 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 3727.70 | 3597.67 | 3597.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 14:15:00 | 3731.85 | 3600.27 | 3598.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 3732.00 | 3741.77 | 3683.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-16 14:15:00 | 3767.00 | 3741.63 | 3684.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-21 14:15:00 | 3677.00 | 3742.95 | 3688.83 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 3484.30 | 4167.88 | 4168.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 3450.00 | 4154.06 | 4161.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 3751.00 | 3746.50 | 3875.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 12:15:00 | 3705.10 | 3745.98 | 3870.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2800.00 | 2669.07 | 2807.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-30 09:15:00 | 2857.80 | 2676.64 | 2807.62 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 3002.70 | 2878.07 | 2877.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3033.20 | 2879.61 | 2878.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 2992.70 | 2992.96 | 2949.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 3043.70 | 2951.02 | 2938.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-03 10:15:00 | 2949.00 | 2979.36 | 2955.89 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 2891.10 | 2942.23 | 2942.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 2852.10 | 2941.33 | 2941.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 2917.50 | 2913.46 | 2926.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-05 09:15:00 | 2883.00 | 2912.68 | 2925.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2878.70 | 2911.78 | 2924.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-06 11:15:00 | 2941.90 | 2911.63 | 2924.76 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 3056.30 | 2935.38 | 2934.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 3076.00 | 2944.16 | 2939.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2991.90 | 3004.23 | 2974.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-01 11:15:00 | 3072.10 | 3000.36 | 2974.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2986.00 | 3012.42 | 2984.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-05 11:15:00 | 2970.20 | 3012.00 | 2984.62 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 2972.00 | 3099.32 | 3099.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2966.00 | 3096.73 | 3098.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3083.30 | 3082.86 | 3090.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-18 09:15:00 | 3030.50 | 3083.10 | 3090.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 14:15:00 | 2639.80 | 2440.00 | 2581.10 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 12:15:00 | 2736.10 | 2574.67 | 2574.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 2763.50 | 2581.42 | 2577.75 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-04 15:15:00 | 3559.65 | 2024-01-23 10:15:00 | 4066.71 | TARGET | 507.06 |
| BUY | 2024-02-05 09:15:00 | 3660.60 | 2024-02-05 12:15:00 | 3617.85 | EXIT_EMA400 | -42.75 |
| BUY | 2024-05-16 14:15:00 | 3767.00 | 2024-05-21 14:15:00 | 3677.00 | EXIT_EMA400 | -90.00 |
| SELL | 2024-12-17 12:15:00 | 3705.10 | 2025-01-22 12:15:00 | 3209.39 | TARGET | 495.71 |
| BUY | 2025-06-27 09:15:00 | 3043.70 | 2025-07-03 10:15:00 | 2949.00 | EXIT_EMA400 | -94.70 |
| SELL | 2025-08-05 09:15:00 | 2883.00 | 2025-08-06 11:15:00 | 2941.90 | EXIT_EMA400 | -58.90 |
| BUY | 2025-09-01 11:15:00 | 3072.10 | 2025-09-05 11:15:00 | 2970.20 | EXIT_EMA400 | -101.90 |
| SELL | 2025-11-18 09:15:00 | 3030.50 | 2025-11-27 11:15:00 | 2849.82 | TARGET | 180.68 |
