# Balkrishna Industries Ltd. (BALKRISIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2161.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -17.15
- **Avg P&L per closed trade:** -2.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 2847.10 | 3017.20 | 3017.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 2836.55 | 3004.17 | 3010.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 2955.80 | 2955.38 | 2981.59 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 3075.00 | 2999.17 | 2998.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 3087.95 | 3005.42 | 3002.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 2985.55 | 3025.69 | 3013.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-27 09:15:00 | 3048.70 | 3024.29 | 3013.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3048.70 | 3024.29 | 3013.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-27 10:15:00 | 3074.90 | 3024.79 | 3013.50 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-30 09:15:00 | 3004.80 | 3026.28 | 3014.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 2806.10 | 3004.64 | 3005.25 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 3062.50 | 3005.52 | 3005.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 15:15:00 | 3068.00 | 3006.14 | 3005.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 2983.70 | 3006.23 | 3005.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-11 15:15:00 | 3039.00 | 3006.90 | 3006.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 3013.50 | 3007.33 | 3006.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-14 12:15:00 | 2996.45 | 3007.22 | 3006.38 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 2968.25 | 3005.60 | 3005.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2946.30 | 3003.57 | 3004.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 3010.65 | 3002.63 | 3004.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 14:15:00 | 2969.05 | 3001.68 | 3003.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 2878.30 | 2820.36 | 2876.84 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 2748.70 | 2609.96 | 2609.27 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 2476.30 | 2617.13 | 2617.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 2448.70 | 2595.16 | 2605.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 2551.90 | 2550.72 | 2578.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-11 12:15:00 | 2500.00 | 2549.55 | 2577.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-02 09:15:00 | 2569.60 | 2480.62 | 2523.08 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 2670.20 | 2551.44 | 2551.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2724.20 | 2559.99 | 2555.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2625.20 | 2648.61 | 2610.29 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 2413.60 | 2582.03 | 2582.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 2402.20 | 2578.59 | 2580.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 2409.30 | 2407.80 | 2466.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 14:15:00 | 2388.50 | 2408.35 | 2465.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2460.90 | 2408.95 | 2462.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 14:15:00 | 2468.30 | 2411.27 | 2462.76 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 2413.30 | 2350.56 | 2350.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 2429.60 | 2357.25 | 2353.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 2424.90 | 2367.23 | 2363.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-26 09:15:00 | 2448.10 | 2508.51 | 2460.38 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 2239.00 | 2423.36 | 2424.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 2233.50 | 2411.85 | 2418.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2221.40 | 2209.58 | 2287.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-28 14:15:00 | 2189.90 | 2242.24 | 2278.36 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-27 09:15:00 | 3048.70 | 2024-09-30 09:15:00 | 3004.80 | EXIT_EMA400 | -43.90 |
| BUY | 2024-09-27 10:15:00 | 3074.90 | 2024-09-30 09:15:00 | 3004.80 | EXIT_EMA400 | -70.10 |
| BUY | 2024-10-11 15:15:00 | 3039.00 | 2024-10-14 12:15:00 | 2996.45 | EXIT_EMA400 | -42.55 |
| SELL | 2024-10-21 14:15:00 | 2969.05 | 2024-10-28 09:15:00 | 2865.51 | TARGET | 103.54 |
| SELL | 2025-06-11 12:15:00 | 2500.00 | 2025-07-02 09:15:00 | 2569.60 | EXIT_EMA400 | -69.60 |
| SELL | 2025-09-12 14:15:00 | 2388.50 | 2025-09-16 14:15:00 | 2468.30 | EXIT_EMA400 | -79.80 |
| BUY | 2026-02-03 09:15:00 | 2424.90 | 2026-02-04 11:15:00 | 2610.16 | TARGET | 185.26 |
