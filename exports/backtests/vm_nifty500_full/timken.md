# Timken India Ltd. (TIMKEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 3428.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -102.30
- **Avg P&L per closed trade:** -14.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 13:15:00 | 3177.75 | 3065.22 | 3065.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 10:15:00 | 3198.00 | 3069.68 | 3067.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 11:15:00 | 3137.10 | 3146.55 | 3113.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-03 13:15:00 | 3153.80 | 3146.50 | 3114.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 3223.65 | 3285.36 | 3223.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-06 13:15:00 | 3200.00 | 3284.51 | 3223.14 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 2840.25 | 3172.07 | 3173.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 2770.00 | 3024.41 | 3089.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 10:15:00 | 2795.75 | 2793.94 | 2922.91 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 3169.40 | 2940.07 | 2939.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 3179.40 | 2944.72 | 2941.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 4244.35 | 4249.20 | 3973.80 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 10:15:00 | 3750.00 | 3964.97 | 3965.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 3671.05 | 3900.75 | 3930.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 3921.60 | 3879.43 | 3914.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 14:15:00 | 3794.00 | 3875.92 | 3911.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-20 11:15:00 | 3876.00 | 3815.53 | 3863.78 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3126.40 | 2693.23 | 2692.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3144.20 | 2697.72 | 2694.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 3350.00 | 3371.29 | 3243.27 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 2881.70 | 3173.97 | 3174.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 2852.10 | 3170.76 | 3172.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3092.20 | 3088.94 | 3126.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 2955.70 | 3084.51 | 3122.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 3061.30 | 3006.14 | 3056.99 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 3095.00 | 3029.52 | 3029.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 3153.40 | 3041.17 | 3035.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 3069.50 | 3079.37 | 3059.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-15 11:15:00 | 3099.30 | 3067.32 | 3056.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3077.80 | 3069.25 | 3057.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-17 13:15:00 | 3100.00 | 3070.07 | 3058.43 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-18 09:15:00 | 3024.50 | 3069.79 | 3058.46 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 2978.00 | 3050.26 | 3050.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 2948.70 | 3048.72 | 3049.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 3050.20 | 3042.58 | 3046.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 11:15:00 | 3006.80 | 3046.35 | 3047.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-13 15:15:00 | 3040.00 | 3030.40 | 3039.18 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 3249.40 | 3038.84 | 3038.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3293.50 | 3062.42 | 3050.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3089.70 | 3098.39 | 3071.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 11:15:00 | 3121.40 | 3094.07 | 3071.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-19 14:15:00 | 3184.40 | 3288.00 | 3208.80 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-03 13:15:00 | 3153.80 | 2024-01-11 09:15:00 | 3272.54 | TARGET | 118.74 |
| SELL | 2024-09-05 14:15:00 | 3794.00 | 2024-09-20 11:15:00 | 3876.00 | EXIT_EMA400 | -82.00 |
| SELL | 2025-08-26 09:15:00 | 2955.70 | 2025-09-16 09:15:00 | 3061.30 | EXIT_EMA400 | -105.60 |
| BUY | 2025-12-15 11:15:00 | 3099.30 | 2025-12-18 09:15:00 | 3024.50 | EXIT_EMA400 | -74.80 |
| BUY | 2025-12-17 13:15:00 | 3100.00 | 2025-12-18 09:15:00 | 3024.50 | EXIT_EMA400 | -75.50 |
| SELL | 2026-01-08 11:15:00 | 3006.80 | 2026-01-13 15:15:00 | 3040.00 | EXIT_EMA400 | -33.20 |
| BUY | 2026-02-17 11:15:00 | 3121.40 | 2026-02-25 09:15:00 | 3271.47 | TARGET | 150.07 |
