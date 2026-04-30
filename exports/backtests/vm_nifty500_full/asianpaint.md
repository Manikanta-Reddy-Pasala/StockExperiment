# Asian Paints Ltd. (ASIANPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2444.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -281.37
- **Avg P&L per closed trade:** -28.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 3233.00 | 3158.34 | 3157.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 3242.95 | 3163.42 | 3160.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 3291.85 | 3297.58 | 3247.02 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-01-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 15:15:00 | 2949.10 | 3222.40 | 3223.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 09:15:00 | 2944.45 | 3171.65 | 3196.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 13:15:00 | 2881.75 | 2880.13 | 2951.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-03 14:15:00 | 2865.30 | 2879.98 | 2950.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 2906.00 | 2864.69 | 2910.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-30 13:15:00 | 2889.95 | 2865.57 | 2910.26 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2909.95 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2942.55 | 2899.08 | 2898.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-10 13:15:00 | 2995.00 | 2905.39 | 2902.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2943.00 | 2927.26 | 2914.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-18 12:15:00 | 2908.55 | 2927.01 | 2914.57 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 2987.40 | 3120.09 | 3120.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2959.95 | 3114.78 | 3117.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2301.20 | 2298.95 | 2414.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-04 09:15:00 | 2274.20 | 2299.05 | 2409.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2395.70 | 2300.28 | 2407.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-05 09:15:00 | 2262.65 | 2300.99 | 2406.71 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 2286.20 | 2247.85 | 2299.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 10:15:00 | 2300.00 | 2250.98 | 2298.94 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 2442.00 | 2322.15 | 2321.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 2453.80 | 2323.46 | 2322.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2378.70 | 2391.07 | 2364.48 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 2319.50 | 2348.46 | 2348.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 2309.10 | 2348.07 | 2348.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-27 13:15:00 | 2329.60 | 2346.62 | 2347.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2329.60 | 2346.62 | 2347.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-27 14:15:00 | 2326.10 | 2346.41 | 2347.49 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2295.00 | 2278.70 | 2302.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-24 12:15:00 | 2282.00 | 2278.96 | 2301.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 2317.70 | 2279.89 | 2300.38 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.85 | 2317.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.84 | 2350.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 15:15:00 | 2378.80 | 2371.90 | 2350.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.65 | 2351.68 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.58 | 2447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.12 | 2441.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.31 | 2422.54 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2518.80 | 2438.26 | 2437.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.70 | 2440.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.80 | 2683.55 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.50 | 2720.59 | 2721.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 2407.70 | 2706.31 | 2713.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2264.25 | 2369.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 10:15:00 | 2269.10 | 2264.29 | 2369.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-03 14:15:00 | 2865.30 | 2024-05-02 09:15:00 | 2919.70 | EXIT_EMA400 | -54.40 |
| SELL | 2024-04-30 13:15:00 | 2889.95 | 2024-05-02 09:15:00 | 2919.70 | EXIT_EMA400 | -29.75 |
| BUY | 2024-07-10 13:15:00 | 2995.00 | 2024-07-18 12:15:00 | 2908.55 | EXIT_EMA400 | -86.45 |
| SELL | 2025-02-04 09:15:00 | 2274.20 | 2025-03-21 10:15:00 | 2300.00 | EXIT_EMA400 | -25.80 |
| SELL | 2025-02-05 09:15:00 | 2262.65 | 2025-03-21 10:15:00 | 2300.00 | EXIT_EMA400 | -37.35 |
| SELL | 2025-05-27 13:15:00 | 2329.60 | 2025-05-30 10:15:00 | 2275.60 | TARGET | 54.00 |
| SELL | 2025-05-27 14:15:00 | 2326.10 | 2025-05-30 14:15:00 | 2261.92 | TARGET | 64.18 |
| SELL | 2025-06-24 12:15:00 | 2282.00 | 2025-06-27 09:15:00 | 2317.70 | EXIT_EMA400 | -35.70 |
| BUY | 2025-07-21 15:15:00 | 2378.80 | 2025-07-24 11:15:00 | 2344.00 | EXIT_EMA400 | -34.80 |
| SELL | 2026-04-08 10:15:00 | 2269.10 | 2026-04-10 09:15:00 | 2364.40 | EXIT_EMA400 | -95.30 |
