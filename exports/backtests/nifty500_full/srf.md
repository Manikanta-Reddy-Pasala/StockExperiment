# SRF Ltd. (SRF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 2518.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 7 |
| ENTRY1 | 13 |
| ENTRY2 | 3 |
| EXIT | 13 |

## P&L

- **Trades closed:** 16
- **Trades open at end:** 0
- **Winners / losers:** 0 / 16
- **Target hits / EMA400 exits:** 0 / 16
- **Total realized P&L (per unit):** -985.35
- **Avg P&L per closed trade:** -61.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 09:15:00 | 2454.90 | 2329.05 | 2328.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 11:15:00 | 2466.70 | 2331.68 | 2330.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 2348.20 | 2360.38 | 2346.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-18 11:15:00 | 2354.65 | 2360.21 | 2346.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 2354.65 | 2360.21 | 2346.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-09-18 12:15:00 | 2364.20 | 2360.25 | 2346.96 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 2352.80 | 2360.35 | 2347.27 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-20 10:15:00 | 2339.75 | 2360.15 | 2347.23 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 2234.85 | 2335.99 | 2336.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 2227.65 | 2326.76 | 2331.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 12:15:00 | 2285.65 | 2281.93 | 2302.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-18 10:15:00 | 2234.55 | 2281.15 | 2301.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 2262.75 | 2233.18 | 2266.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-03 14:15:00 | 2252.90 | 2233.37 | 2266.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-06 09:15:00 | 2304.55 | 2234.28 | 2266.36 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 13:15:00 | 2328.95 | 2289.40 | 2289.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 2339.30 | 2290.67 | 2289.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 2369.75 | 2386.88 | 2352.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 2413.95 | 2387.46 | 2354.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 2393.00 | 2434.19 | 2392.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-09 09:15:00 | 2357.20 | 2433.03 | 2392.22 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 2278.20 | 2365.39 | 2365.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 13:15:00 | 2273.10 | 2364.47 | 2365.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 2324.10 | 2320.97 | 2338.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-14 09:15:00 | 2310.00 | 2320.88 | 2338.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 2337.55 | 2321.02 | 2338.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-14 14:15:00 | 2358.00 | 2321.70 | 2338.09 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 2381.00 | 2349.67 | 2349.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 2391.40 | 2350.44 | 2349.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 2520.00 | 2527.12 | 2467.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-18 09:15:00 | 2551.05 | 2527.27 | 2468.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-07 09:15:00 | 2471.20 | 2573.10 | 2515.85 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 2290.15 | 2470.34 | 2471.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 2283.00 | 2466.68 | 2469.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 10:15:00 | 2343.15 | 2333.64 | 2383.47 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 2510.65 | 2396.77 | 2396.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 11:15:00 | 2534.40 | 2399.40 | 2397.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 2431.75 | 2442.83 | 2421.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 2540.00 | 2444.73 | 2423.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 2463.25 | 2480.30 | 2447.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-14 12:15:00 | 2503.00 | 2480.53 | 2447.45 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-09 09:15:00 | 2484.80 | 2525.59 | 2488.33 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 2377.05 | 2471.04 | 2471.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 2348.40 | 2469.82 | 2470.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 2324.80 | 2317.80 | 2370.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 2294.00 | 2323.24 | 2368.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 13:15:00 | 2318.00 | 2272.28 | 2316.93 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 2596.35 | 2317.37 | 2316.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 2617.10 | 2377.82 | 2348.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 2879.05 | 2888.62 | 2776.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 11:15:00 | 2938.70 | 2888.87 | 2781.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 2681.45 | 2886.11 | 2789.74 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2843.00 | 3058.79 | 3058.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 2831.00 | 2981.13 | 3013.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2961.50 | 2938.97 | 2979.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 14:15:00 | 2948.80 | 2946.92 | 2979.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 2982.00 | 2947.00 | 2977.63 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 3190.00 | 2969.78 | 2969.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3204.10 | 2972.11 | 2970.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3010.10 | 3013.51 | 2994.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 12:15:00 | 3027.50 | 3013.65 | 2994.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-30 09:15:00 | 2988.80 | 3015.17 | 2996.17 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 2902.50 | 2981.93 | 2981.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2885.80 | 2979.56 | 2980.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2919.80 | 2895.89 | 2930.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 13:15:00 | 2832.80 | 2894.04 | 2925.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-10 09:15:00 | 2936.70 | 2884.33 | 2915.59 | Close above EMA400 |

### Cycle 13 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 3083.00 | 2938.32 | 2937.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3124.50 | 2940.17 | 2938.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 2976.30 | 3009.68 | 2980.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 3084.80 | 3011.89 | 2982.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 2994.00 | 3023.74 | 2995.76 | Close below EMA400 |

### Cycle 14 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2706.60 | 2972.33 | 2972.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2693.80 | 2954.82 | 2963.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2928.00 | 2902.90 | 2933.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 13:15:00 | 2921.00 | 2904.06 | 2933.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2928.20 | 2904.48 | 2933.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 2939.50 | 2904.91 | 2933.31 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-18 11:15:00 | 2354.65 | 2023-09-20 10:15:00 | 2339.75 | EXIT_EMA400 | -14.90 |
| BUY | 2023-09-18 12:15:00 | 2364.20 | 2023-09-20 10:15:00 | 2339.75 | EXIT_EMA400 | -24.45 |
| SELL | 2023-10-18 10:15:00 | 2234.55 | 2023-11-06 09:15:00 | 2304.55 | EXIT_EMA400 | -70.00 |
| SELL | 2023-11-03 14:15:00 | 2252.90 | 2023-11-06 09:15:00 | 2304.55 | EXIT_EMA400 | -51.65 |
| BUY | 2023-12-22 09:15:00 | 2413.95 | 2024-01-09 09:15:00 | 2357.20 | EXIT_EMA400 | -56.75 |
| SELL | 2024-02-14 09:15:00 | 2310.00 | 2024-02-14 14:15:00 | 2358.00 | EXIT_EMA400 | -48.00 |
| BUY | 2024-04-18 09:15:00 | 2551.05 | 2024-05-07 09:15:00 | 2471.20 | EXIT_EMA400 | -79.85 |
| BUY | 2024-08-06 09:15:00 | 2540.00 | 2024-09-09 09:15:00 | 2484.80 | EXIT_EMA400 | -55.20 |
| BUY | 2024-08-14 12:15:00 | 2503.00 | 2024-09-09 09:15:00 | 2484.80 | EXIT_EMA400 | -18.20 |
| SELL | 2024-11-08 15:15:00 | 2294.00 | 2024-12-03 13:15:00 | 2318.00 | EXIT_EMA400 | -24.00 |
| BUY | 2025-04-02 11:15:00 | 2938.70 | 2025-04-07 09:15:00 | 2681.45 | EXIT_EMA400 | -257.25 |
| SELL | 2025-09-15 14:15:00 | 2948.80 | 2025-09-17 09:15:00 | 2982.00 | EXIT_EMA400 | -33.20 |
| BUY | 2025-10-28 12:15:00 | 3027.50 | 2025-10-30 09:15:00 | 2988.80 | EXIT_EMA400 | -38.70 |
| SELL | 2025-12-03 13:15:00 | 2832.80 | 2025-12-10 09:15:00 | 2936.70 | EXIT_EMA400 | -103.90 |
| BUY | 2026-01-07 09:15:00 | 3084.80 | 2026-01-19 09:15:00 | 2994.00 | EXIT_EMA400 | -90.80 |
| SELL | 2026-02-03 13:15:00 | 2921.00 | 2026-02-04 11:15:00 | 2939.50 | EXIT_EMA400 | -18.50 |
