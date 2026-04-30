# Ceat Ltd. (CEATLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3445.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -389.63
- **Avg P&L per closed trade:** -43.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 2148.40 | 2250.26 | 2250.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 2134.00 | 2248.09 | 2249.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 2231.00 | 2158.37 | 2190.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-17 11:15:00 | 2195.00 | 2159.47 | 2190.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 2195.00 | 2159.47 | 2190.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-10-17 13:15:00 | 2182.00 | 2160.00 | 2190.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 2181.65 | 2160.22 | 2190.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-17 15:15:00 | 2226.00 | 2160.87 | 2190.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 2298.30 | 2163.34 | 2163.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 2315.00 | 2169.86 | 2166.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 11:15:00 | 2770.00 | 2784.68 | 2653.22 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 11:15:00 | 2499.25 | 2617.97 | 2618.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 2402.30 | 2586.25 | 2599.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 13:15:00 | 2438.50 | 2412.29 | 2471.75 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2851.55 | 2500.95 | 2499.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 2870.00 | 2508.16 | 2503.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 2625.25 | 2629.42 | 2579.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 2646.75 | 2624.77 | 2580.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2646.75 | 2624.77 | 2580.69 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-29 10:15:00 | 2686.00 | 2624.42 | 2585.10 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 2599.20 | 2640.07 | 2599.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 09:15:00 | 2713.00 | 2848.80 | 2849.37 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 2975.00 | 2847.65 | 2847.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 11:15:00 | 2980.00 | 2853.50 | 2850.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 3044.45 | 3053.48 | 2979.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-26 09:15:00 | 3076.00 | 3051.54 | 2984.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 3067.90 | 3106.41 | 3042.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-14 14:15:00 | 3030.60 | 3101.20 | 3043.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 2937.90 | 3008.34 | 3008.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 11:15:00 | 2916.00 | 3007.42 | 3008.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 3041.75 | 2995.01 | 3001.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 2933.00 | 2997.44 | 3002.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 2824.25 | 2682.80 | 2774.92 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 15:15:00 | 3023.00 | 2815.55 | 2814.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 3036.10 | 2817.74 | 2815.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 3647.80 | 3664.86 | 3458.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 09:15:00 | 3713.50 | 3634.56 | 3513.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 3591.50 | 3730.77 | 3610.81 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 3245.30 | 3535.22 | 3535.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 3235.10 | 3529.41 | 3532.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 3293.50 | 3273.12 | 3363.86 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 3452.30 | 3389.34 | 3389.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 3466.00 | 3390.73 | 3389.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 3872.20 | 3902.40 | 3742.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-03 09:15:00 | 3962.70 | 3890.91 | 3784.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 3765.00 | 3890.86 | 3798.19 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 3710.00 | 3791.08 | 3791.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 3681.00 | 3787.24 | 3789.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 3787.60 | 3764.52 | 3776.80 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 3878.10 | 3788.50 | 3788.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 3920.40 | 3792.05 | 3789.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 3851.30 | 3862.14 | 3829.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 09:15:00 | 3888.10 | 3862.48 | 3830.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-19 14:15:00 | 3820.40 | 3865.45 | 3834.68 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3460.00 | 3808.39 | 3810.01 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-17 11:15:00 | 2195.00 | 2023-10-17 13:15:00 | 2181.62 | TARGET | 13.38 |
| SELL | 2023-10-17 13:15:00 | 2182.00 | 2023-10-17 15:15:00 | 2226.00 | EXIT_EMA400 | -44.00 |
| BUY | 2024-07-24 09:15:00 | 2646.75 | 2024-08-05 09:15:00 | 2599.20 | EXIT_EMA400 | -47.55 |
| BUY | 2024-07-29 10:15:00 | 2686.00 | 2024-08-05 09:15:00 | 2599.20 | EXIT_EMA400 | -86.80 |
| BUY | 2024-12-26 09:15:00 | 3076.00 | 2025-01-14 14:15:00 | 3030.60 | EXIT_EMA400 | -45.40 |
| SELL | 2025-02-06 09:15:00 | 2933.00 | 2025-02-11 15:15:00 | 2724.86 | TARGET | 208.14 |
| BUY | 2025-07-03 09:15:00 | 3713.50 | 2025-07-21 09:15:00 | 3591.50 | EXIT_EMA400 | -122.00 |
| BUY | 2025-12-03 09:15:00 | 3962.70 | 2025-12-09 09:15:00 | 3765.00 | EXIT_EMA400 | -197.70 |
| BUY | 2026-02-17 09:15:00 | 3888.10 | 2026-02-19 14:15:00 | 3820.40 | EXIT_EMA400 | -67.70 |
