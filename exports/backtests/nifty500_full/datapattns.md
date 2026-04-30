# Data Patterns (India) Ltd. (DATAPATTNS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 4081.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 637.91
- **Avg P&L per closed trade:** 53.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 09:15:00 | 1907.35 | 2087.54 | 2087.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 1894.60 | 2083.81 | 2085.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1937.85 | 1922.16 | 1977.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-01 09:15:00 | 1835.00 | 1958.84 | 1977.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1960.00 | 1935.93 | 1961.75 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-08 12:15:00 | 1981.00 | 1936.38 | 1961.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 2179.00 | 1948.09 | 1947.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 2234.25 | 1950.94 | 1949.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 2259.95 | 2316.12 | 2169.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-27 13:15:00 | 2471.55 | 2319.03 | 2212.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2824.95 | 2902.77 | 2758.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 2670.00 | 2900.46 | 2757.96 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 2879.60 | 2992.24 | 2992.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 2867.85 | 2989.88 | 2991.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 15:15:00 | 2533.00 | 2531.37 | 2662.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 2460.00 | 2530.66 | 2661.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 2531.65 | 2339.00 | 2443.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 2540.00 | 2503.59 | 2503.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 11:15:00 | 2561.75 | 2504.79 | 2504.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 2495.15 | 2505.07 | 2504.22 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 09:15:00 | 2487.90 | 2503.44 | 2503.48 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 2515.20 | 2503.54 | 2503.52 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 2477.95 | 2503.33 | 2503.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2462.70 | 2502.93 | 2503.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 13:15:00 | 2500.45 | 2497.31 | 2500.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 2444.60 | 2497.00 | 2499.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2444.60 | 2497.00 | 2499.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-06 10:15:00 | 2388.85 | 2495.93 | 2499.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 10:15:00 | 1857.30 | 1685.39 | 1794.45 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 2407.40 | 1870.31 | 1868.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 12:15:00 | 2474.40 | 1876.32 | 1871.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 2807.20 | 2849.85 | 2607.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-02 11:15:00 | 2945.10 | 2859.04 | 2653.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 2755.00 | 2888.05 | 2746.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 2674.60 | 2885.93 | 2746.00 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 2512.00 | 2685.28 | 2685.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 2505.70 | 2683.50 | 2684.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2576.70 | 2568.54 | 2610.32 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2836.00 | 2638.84 | 2638.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 2843.90 | 2640.88 | 2639.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 2661.00 | 2678.82 | 2660.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 2736.70 | 2660.43 | 2653.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2736.70 | 2660.43 | 2653.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 13:15:00 | 2827.00 | 2665.01 | 2655.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2702.50 | 2713.63 | 2686.83 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-16 11:15:00 | 2713.20 | 2713.63 | 2686.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-30 12:15:00 | 2711.40 | 2744.95 | 2712.04 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2570.30 | 2779.14 | 2779.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2531.20 | 2770.76 | 2775.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 2720.60 | 2711.64 | 2741.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 13:15:00 | 2685.00 | 2711.17 | 2741.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2716.10 | 2710.65 | 2740.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-26 12:15:00 | 2710.30 | 2710.64 | 2739.99 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2689.10 | 2682.80 | 2719.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 2722.00 | 2683.19 | 2719.46 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 2821.60 | 2657.57 | 2657.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 2841.00 | 2659.40 | 2658.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 13:15:00 | 3075.50 | 3076.41 | 2920.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 3098.90 | 3076.63 | 2921.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 2959.90 | 3127.71 | 3000.56 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-01 09:15:00 | 1835.00 | 2024-01-08 12:15:00 | 1981.00 | EXIT_EMA400 | -146.00 |
| BUY | 2024-03-27 13:15:00 | 2471.55 | 2024-05-21 09:15:00 | 3249.57 | TARGET | 778.02 |
| SELL | 2024-10-17 09:15:00 | 2460.00 | 2024-11-28 09:15:00 | 2531.65 | EXIT_EMA400 | -71.65 |
| SELL | 2025-01-06 09:15:00 | 2444.60 | 2025-01-10 09:15:00 | 2278.82 | TARGET | 165.78 |
| SELL | 2025-01-06 10:15:00 | 2388.85 | 2025-01-13 14:15:00 | 2057.48 | TARGET | 331.37 |
| BUY | 2025-07-02 11:15:00 | 2945.10 | 2025-07-21 09:15:00 | 2674.60 | EXIT_EMA400 | -270.50 |
| BUY | 2025-10-16 11:15:00 | 2713.20 | 2025-10-17 09:15:00 | 2791.91 | TARGET | 78.71 |
| BUY | 2025-10-03 09:15:00 | 2736.70 | 2025-10-30 12:15:00 | 2711.40 | EXIT_EMA400 | -25.30 |
| BUY | 2025-10-03 13:15:00 | 2827.00 | 2025-10-30 12:15:00 | 2711.40 | EXIT_EMA400 | -115.60 |
| SELL | 2025-12-26 12:15:00 | 2710.30 | 2025-12-30 09:15:00 | 2621.23 | TARGET | 89.07 |
| SELL | 2025-12-24 13:15:00 | 2685.00 | 2026-01-05 11:15:00 | 2722.00 | EXIT_EMA400 | -37.00 |
| BUY | 2026-03-16 14:15:00 | 3098.90 | 2026-04-02 09:15:00 | 2959.90 | EXIT_EMA400 | -139.00 |
