# Balkrishna Industries Ltd. (BALKRISIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 2160.80
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
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -571.13
- **Avg P&L per closed trade:** -51.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 2448.60 | 2553.56 | 2553.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 2436.10 | 2548.50 | 2551.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 12:15:00 | 2330.30 | 2330.11 | 2399.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-13 12:15:00 | 2304.25 | 2330.03 | 2397.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 2345.20 | 2309.42 | 2356.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-05 09:15:00 | 2327.80 | 2310.24 | 2356.63 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-05 11:15:00 | 2379.00 | 2311.21 | 2356.66 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 10:15:00 | 2450.25 | 2376.90 | 2376.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 2463.65 | 2383.88 | 2380.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 13:15:00 | 3111.15 | 3120.99 | 2965.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-15 09:15:00 | 3147.05 | 3121.24 | 2968.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-09 14:15:00 | 3064.65 | 3182.34 | 3080.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 2851.65 | 3007.56 | 3008.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 2842.80 | 3005.92 | 3007.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 2959.80 | 2955.35 | 2977.92 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 3065.85 | 2993.16 | 2993.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 3070.15 | 2993.92 | 2993.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 2985.55 | 3026.06 | 3011.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-27 09:15:00 | 3048.70 | 3024.64 | 3011.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3048.70 | 3024.64 | 3011.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-27 10:15:00 | 3074.90 | 3025.14 | 3011.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-30 09:15:00 | 3004.80 | 3026.56 | 3012.83 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 2873.85 | 3001.70 | 3002.18 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 3038.50 | 3002.60 | 3002.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 14:15:00 | 3062.50 | 3005.75 | 3004.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 2983.70 | 3006.38 | 3004.52 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 2985.95 | 3003.17 | 3003.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 2962.00 | 3001.27 | 3002.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 2819.90 | 2818.86 | 2878.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 15:15:00 | 2806.70 | 2825.56 | 2874.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 10:15:00 | 2878.15 | 2826.02 | 2873.95 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 2748.70 | 2610.01 | 2609.41 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 2476.30 | 2617.04 | 2617.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 2448.70 | 2595.16 | 2605.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 2551.90 | 2550.72 | 2578.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-11 12:15:00 | 2500.00 | 2549.75 | 2577.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-02 09:15:00 | 2568.50 | 2480.67 | 2523.15 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 2672.00 | 2551.50 | 2551.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2724.20 | 2560.14 | 2555.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2625.20 | 2648.83 | 2610.45 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 2413.60 | 2582.05 | 2582.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 2402.20 | 2578.66 | 2580.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 2409.00 | 2407.91 | 2466.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 14:15:00 | 2389.70 | 2408.51 | 2465.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2460.90 | 2409.07 | 2463.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 14:15:00 | 2469.30 | 2411.39 | 2462.87 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 2395.80 | 2350.93 | 2350.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 2429.60 | 2357.14 | 2353.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 2376.50 | 2384.32 | 2370.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 2424.90 | 2372.09 | 2365.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-26 09:15:00 | 2448.10 | 2510.33 | 2461.78 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 15:15:00 | 2228.00 | 2426.00 | 2426.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 2225.00 | 2410.74 | 2418.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2221.40 | 2210.04 | 2287.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 2194.30 | 2210.35 | 2285.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2277.20 | 2212.28 | 2283.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-13 09:15:00 | 2228.30 | 2215.31 | 2282.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 14:15:00 | 2315.20 | 2219.78 | 2281.29 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-13 12:15:00 | 2304.25 | 2024-04-05 11:15:00 | 2379.00 | EXIT_EMA400 | -74.75 |
| SELL | 2024-04-05 09:15:00 | 2327.80 | 2024-04-05 11:15:00 | 2379.00 | EXIT_EMA400 | -51.20 |
| BUY | 2024-07-15 09:15:00 | 3147.05 | 2024-08-09 14:15:00 | 3064.65 | EXIT_EMA400 | -82.40 |
| BUY | 2024-09-27 09:15:00 | 3048.70 | 2024-09-30 09:15:00 | 3004.80 | EXIT_EMA400 | -43.90 |
| BUY | 2024-09-27 10:15:00 | 3074.90 | 2024-09-30 09:15:00 | 3004.80 | EXIT_EMA400 | -70.10 |
| SELL | 2024-12-06 15:15:00 | 2806.70 | 2024-12-09 10:15:00 | 2878.15 | EXIT_EMA400 | -71.45 |
| SELL | 2025-06-11 12:15:00 | 2500.00 | 2025-07-02 09:15:00 | 2568.50 | EXIT_EMA400 | -68.50 |
| SELL | 2025-09-12 14:15:00 | 2389.70 | 2025-09-16 14:15:00 | 2469.30 | EXIT_EMA400 | -79.60 |
| BUY | 2026-02-03 09:15:00 | 2424.90 | 2026-02-03 12:15:00 | 2603.47 | TARGET | 178.57 |
| SELL | 2026-04-09 09:15:00 | 2194.30 | 2026-04-15 14:15:00 | 2315.20 | EXIT_EMA400 | -120.90 |
| SELL | 2026-04-13 09:15:00 | 2228.30 | 2026-04-15 14:15:00 | 2315.20 | EXIT_EMA400 | -86.90 |
