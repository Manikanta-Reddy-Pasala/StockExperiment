# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4707.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 143 |
| ALERT2 | 142 |
| ALERT2_SKIP | 96 |
| ALERT3 | 336 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 127 |
| PARTIAL | 7 |
| TARGET_HIT | 9 |
| STOP_HIT | 122 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 138 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 107
- **Target hits / Stop hits / Partials:** 9 / 122 / 7
- **Avg / median % per leg:** -0.08% / -0.93%
- **Sum % (uncompounded):** -11.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 14 | 23.0% | 7 | 54 | 0 | -0.17% | -10.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 61 | 14 | 23.0% | 7 | 54 | 0 | -0.17% | -10.4% |
| SELL (all) | 77 | 17 | 22.1% | 2 | 68 | 7 | -0.01% | -0.7% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.57% | -6.3% |
| SELL @ 3rd Alert (retest2) | 73 | 17 | 23.3% | 2 | 64 | 7 | 0.08% | 5.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.57% | -6.3% |
| retest2 (combined) | 134 | 31 | 23.1% | 9 | 118 | 7 | -0.04% | -4.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 12:15:00 | 2355.30 | 2424.90 | 2429.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 13:15:00 | 2315.15 | 2402.95 | 2419.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 15:15:00 | 2252.00 | 2248.99 | 2284.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 2273.90 | 2253.97 | 2283.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 2273.90 | 2253.97 | 2283.95 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 11:15:00 | 2289.95 | 2263.12 | 2262.17 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 13:15:00 | 2254.00 | 2262.40 | 2263.39 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 2318.30 | 2272.14 | 2267.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 11:15:00 | 2340.55 | 2293.73 | 2278.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 10:15:00 | 2336.00 | 2336.31 | 2320.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 2355.00 | 2353.25 | 2335.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 2355.00 | 2353.25 | 2335.60 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 2304.45 | 2355.13 | 2360.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 10:15:00 | 2297.00 | 2322.25 | 2338.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 13:15:00 | 2324.80 | 2320.32 | 2333.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 14:15:00 | 2330.50 | 2322.36 | 2333.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 2330.50 | 2322.36 | 2333.24 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 14:15:00 | 2345.70 | 2335.88 | 2334.90 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 14:15:00 | 2329.00 | 2333.98 | 2334.65 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 15:15:00 | 2342.00 | 2333.94 | 2333.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 11:15:00 | 2361.20 | 2340.68 | 2336.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 11:15:00 | 2358.00 | 2358.63 | 2349.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 12:15:00 | 2346.00 | 2356.10 | 2349.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 2346.00 | 2356.10 | 2349.55 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 2354.10 | 2371.13 | 2372.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 09:15:00 | 2322.05 | 2352.94 | 2362.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 2238.70 | 2225.90 | 2251.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 12:15:00 | 2230.10 | 2226.74 | 2249.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 2230.10 | 2226.74 | 2249.15 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 2266.70 | 2239.25 | 2238.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 13:15:00 | 2284.90 | 2261.92 | 2250.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 14:15:00 | 2270.15 | 2279.16 | 2267.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 14:15:00 | 2270.15 | 2279.16 | 2267.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 2270.15 | 2279.16 | 2267.91 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 2250.00 | 2263.88 | 2264.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 13:15:00 | 2243.75 | 2254.38 | 2257.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 09:15:00 | 2270.00 | 2254.19 | 2256.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 2270.00 | 2254.19 | 2256.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 2270.00 | 2254.19 | 2256.09 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 13:15:00 | 2270.05 | 2256.70 | 2256.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 14:15:00 | 2281.70 | 2261.70 | 2258.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 11:15:00 | 2260.95 | 2276.77 | 2271.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 11:15:00 | 2260.95 | 2276.77 | 2271.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 2260.95 | 2276.77 | 2271.39 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 12:15:00 | 2477.70 | 2524.14 | 2528.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 13:15:00 | 2469.90 | 2513.30 | 2522.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 09:15:00 | 2518.85 | 2498.10 | 2512.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 2518.85 | 2498.10 | 2512.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 2518.85 | 2498.10 | 2512.09 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 2541.80 | 2515.38 | 2514.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 2590.00 | 2530.31 | 2521.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 2548.40 | 2568.72 | 2550.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 2548.40 | 2568.72 | 2550.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 2548.40 | 2568.72 | 2550.20 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 15:15:00 | 2510.00 | 2540.83 | 2542.23 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 09:15:00 | 2590.35 | 2550.73 | 2546.61 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 11:15:00 | 2526.75 | 2557.18 | 2559.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 12:15:00 | 2501.45 | 2530.38 | 2542.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 14:15:00 | 2510.75 | 2508.98 | 2521.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 2546.70 | 2516.69 | 2522.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 2546.70 | 2516.69 | 2522.92 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 11:15:00 | 2548.90 | 2527.50 | 2527.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 12:15:00 | 2555.00 | 2545.06 | 2538.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 14:15:00 | 2544.90 | 2555.73 | 2549.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 14:15:00 | 2544.90 | 2555.73 | 2549.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 2544.90 | 2555.73 | 2549.63 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 2535.00 | 2545.86 | 2547.16 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 14:15:00 | 2562.80 | 2547.13 | 2546.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 12:15:00 | 2571.75 | 2553.17 | 2550.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 2571.20 | 2576.17 | 2564.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 2571.20 | 2576.17 | 2564.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 2571.20 | 2576.17 | 2564.93 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 2697.35 | 2720.39 | 2722.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 2692.75 | 2714.86 | 2719.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 15:15:00 | 2720.00 | 2715.89 | 2719.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 15:15:00 | 2720.00 | 2715.89 | 2719.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 2720.00 | 2715.89 | 2719.79 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 2750.15 | 2722.74 | 2722.55 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 11:15:00 | 2712.10 | 2720.65 | 2721.64 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 2792.00 | 2734.15 | 2727.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 2802.80 | 2774.70 | 2752.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 2766.45 | 2782.25 | 2762.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 11:15:00 | 2773.15 | 2780.43 | 2763.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 2773.15 | 2780.43 | 2763.10 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 2816.75 | 2911.19 | 2914.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 10:15:00 | 2802.25 | 2870.21 | 2893.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 09:15:00 | 2814.95 | 2782.08 | 2809.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 2814.95 | 2782.08 | 2809.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 2814.95 | 2782.08 | 2809.35 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 2861.10 | 2797.22 | 2795.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 10:15:00 | 2872.45 | 2812.27 | 2802.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 11:15:00 | 2876.20 | 2877.25 | 2850.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 2856.10 | 2873.02 | 2850.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 2856.10 | 2873.02 | 2850.83 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 15:15:00 | 3108.00 | 3142.38 | 3145.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 3079.85 | 3129.88 | 3139.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 2950.10 | 2936.15 | 2971.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 11:15:00 | 2971.60 | 2946.63 | 2970.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 2971.60 | 2946.63 | 2970.61 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 3015.30 | 2974.25 | 2974.17 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 14:15:00 | 2954.35 | 2972.89 | 2974.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 09:15:00 | 2906.80 | 2942.09 | 2954.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 2925.35 | 2922.03 | 2936.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 11:15:00 | 2932.55 | 2923.72 | 2934.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 2932.55 | 2923.72 | 2934.32 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 13:15:00 | 3134.80 | 2978.86 | 2958.34 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 3057.00 | 3097.96 | 3100.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 3003.40 | 3079.05 | 3091.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 2887.90 | 2881.47 | 2923.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 13:15:00 | 2846.25 | 2814.55 | 2849.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 2846.25 | 2814.55 | 2849.47 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 11:15:00 | 2889.00 | 2848.24 | 2846.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 12:15:00 | 2924.40 | 2863.48 | 2853.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 13:15:00 | 2901.00 | 2910.53 | 2887.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 15:15:00 | 2894.90 | 2911.00 | 2891.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 2894.90 | 2911.00 | 2891.95 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 15:15:00 | 2893.00 | 2906.83 | 2908.70 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 10:15:00 | 2929.60 | 2912.71 | 2911.13 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 14:15:00 | 2878.10 | 2904.63 | 2908.14 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 13:15:00 | 2917.80 | 2908.39 | 2907.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 14:15:00 | 2943.90 | 2915.49 | 2910.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 2923.75 | 2926.04 | 2918.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 13:15:00 | 2907.00 | 2922.23 | 2917.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 2907.00 | 2922.23 | 2917.82 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 2899.35 | 2914.27 | 2915.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 15:15:00 | 2880.00 | 2902.41 | 2909.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 2918.00 | 2905.53 | 2909.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 2918.00 | 2905.53 | 2909.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 2918.00 | 2905.53 | 2909.90 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 13:15:00 | 2614.35 | 2593.66 | 2592.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 15:15:00 | 2625.00 | 2602.60 | 2597.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 2688.30 | 2688.49 | 2656.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 15:15:00 | 2667.85 | 2680.22 | 2665.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 2667.85 | 2680.22 | 2665.93 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 13:15:00 | 2680.20 | 2686.57 | 2687.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 09:15:00 | 2665.80 | 2680.50 | 2684.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 10:15:00 | 2685.10 | 2681.42 | 2684.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 10:15:00 | 2685.10 | 2681.42 | 2684.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 2685.10 | 2681.42 | 2684.13 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 2726.85 | 2690.51 | 2688.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 15:15:00 | 2734.00 | 2712.42 | 2700.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 09:15:00 | 2700.05 | 2709.95 | 2700.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 2700.05 | 2709.95 | 2700.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 2700.05 | 2709.95 | 2700.35 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 11:15:00 | 3006.15 | 3060.43 | 3063.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 13:15:00 | 3000.00 | 3039.78 | 3052.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-28 09:15:00 | 3119.00 | 3048.07 | 3052.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 3119.00 | 3048.07 | 3052.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 3119.00 | 3048.07 | 3052.42 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 10:15:00 | 3128.05 | 3064.07 | 3059.30 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 3080.00 | 3087.43 | 3087.71 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 13:15:00 | 3105.30 | 3091.32 | 3089.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 3168.40 | 3106.67 | 3096.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 3190.00 | 3196.17 | 3170.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 3192.05 | 3201.72 | 3185.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 3192.05 | 3201.72 | 3185.39 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 3189.00 | 3197.60 | 3197.74 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 11:15:00 | 3210.00 | 3200.08 | 3198.86 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 15:15:00 | 3185.00 | 3197.62 | 3198.31 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 3215.40 | 3201.69 | 3200.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 3255.05 | 3215.98 | 3207.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 11:15:00 | 3217.95 | 3218.83 | 3210.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 12:15:00 | 3205.50 | 3216.17 | 3210.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 12:15:00 | 3205.50 | 3216.17 | 3210.08 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 3146.25 | 3201.08 | 3205.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 10:15:00 | 3131.60 | 3171.14 | 3185.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 14:15:00 | 3160.00 | 3146.22 | 3166.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 15:15:00 | 3175.00 | 3151.98 | 3167.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 3175.00 | 3151.98 | 3167.35 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 3125.70 | 3058.69 | 3053.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 3155.75 | 3115.22 | 3098.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 13:15:00 | 3100.50 | 3120.96 | 3107.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 13:15:00 | 3100.50 | 3120.96 | 3107.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 3100.50 | 3120.96 | 3107.46 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 3127.25 | 3137.55 | 3138.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 11:15:00 | 3106.05 | 3131.72 | 3135.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 3106.45 | 3104.65 | 3118.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 3136.05 | 3110.93 | 3119.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 3136.05 | 3110.93 | 3119.73 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 3269.05 | 3141.77 | 3130.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 15:15:00 | 3300.00 | 3246.04 | 3201.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 12:15:00 | 3267.10 | 3270.56 | 3229.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 13:15:00 | 3306.90 | 3277.83 | 3236.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 3306.90 | 3277.83 | 3236.27 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 3643.95 | 3698.24 | 3701.48 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 3674.00 | 3635.43 | 3634.34 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 3609.05 | 3633.00 | 3633.72 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 3675.00 | 3634.69 | 3633.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 14:15:00 | 3710.00 | 3658.31 | 3645.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 3650.90 | 3657.37 | 3647.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 3650.90 | 3657.37 | 3647.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 3650.90 | 3657.37 | 3647.06 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 3565.20 | 3637.06 | 3645.02 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 13:15:00 | 3665.55 | 3630.33 | 3628.05 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 3484.95 | 3603.11 | 3616.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 3432.05 | 3568.90 | 3599.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 13:15:00 | 3606.65 | 3569.82 | 3592.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 13:15:00 | 3606.65 | 3569.82 | 3592.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 3606.65 | 3569.82 | 3592.06 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-13 12:15:00 | 3617.70 | 3593.42 | 3591.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-13 14:15:00 | 3638.15 | 3606.63 | 3598.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-14 14:15:00 | 3614.95 | 3618.93 | 3610.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 3614.95 | 3618.93 | 3610.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 3614.95 | 3618.93 | 3610.31 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 3531.45 | 3598.40 | 3602.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 3493.00 | 3577.32 | 3592.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 11:15:00 | 3504.70 | 3498.91 | 3538.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 12:15:00 | 3560.10 | 3511.15 | 3540.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 3560.10 | 3511.15 | 3540.01 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 3710.00 | 3584.42 | 3568.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 3776.00 | 3665.36 | 3626.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 13:15:00 | 4306.10 | 4306.36 | 4252.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 4549.50 | 4572.95 | 4547.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 4549.50 | 4572.95 | 4547.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 4540.15 | 4557.25 | 4552.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 4553.65 | 4556.53 | 4552.91 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 4515.05 | 4546.12 | 4548.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 4514.30 | 4539.75 | 4545.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 14:15:00 | 4536.80 | 4535.88 | 4542.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 14:15:00 | 4536.80 | 4535.88 | 4542.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 4536.80 | 4535.88 | 4542.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:45:00 | 4544.70 | 4535.88 | 4542.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 4554.95 | 4539.70 | 4543.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 4552.10 | 4539.70 | 4543.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 09:15:00 | 4583.00 | 4548.36 | 4547.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 10:15:00 | 4720.00 | 4582.69 | 4562.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 12:15:00 | 4698.90 | 4715.50 | 4666.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-16 13:00:00 | 4698.90 | 4715.50 | 4666.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 4680.05 | 4705.30 | 4670.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:45:00 | 4674.65 | 4705.30 | 4670.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 4652.20 | 4694.68 | 4668.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 4714.60 | 4694.68 | 4668.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 4766.85 | 4709.11 | 4677.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 09:30:00 | 4781.00 | 4740.20 | 4711.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:30:00 | 4792.20 | 4759.35 | 4728.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:45:00 | 4788.30 | 4789.47 | 4758.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 4649.10 | 4735.28 | 4739.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 4649.10 | 4735.28 | 4739.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 14:15:00 | 4601.95 | 4708.62 | 4726.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 09:15:00 | 4352.65 | 4350.77 | 4432.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 10:45:00 | 4319.00 | 4344.06 | 4421.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 4357.30 | 4353.71 | 4407.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 4443.20 | 4377.98 | 4405.71 | SL hit (close>ema400) qty=1.00 sl=4405.71 alert=retest1 |

### Cycle 66 — BUY (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 14:15:00 | 4460.95 | 4423.98 | 4420.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 09:15:00 | 4679.00 | 4482.47 | 4448.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 15:15:00 | 4731.50 | 4736.10 | 4683.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 09:15:00 | 4625.00 | 4736.10 | 4683.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 4535.55 | 4695.99 | 4670.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 4535.55 | 4695.99 | 4670.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 4528.60 | 4662.51 | 4657.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:45:00 | 4530.00 | 4662.51 | 4657.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 4629.85 | 4649.98 | 4652.03 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 13:15:00 | 4670.75 | 4654.13 | 4653.73 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 14:15:00 | 4646.35 | 4652.57 | 4653.06 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 15:15:00 | 4662.95 | 4654.65 | 4653.96 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 4588.70 | 4641.46 | 4648.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 4502.45 | 4613.66 | 4634.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 4667.05 | 4586.85 | 4606.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 4667.05 | 4586.85 | 4606.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 4667.05 | 4586.85 | 4606.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:30:00 | 4691.00 | 4586.85 | 4606.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 4680.25 | 4605.53 | 4613.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:45:00 | 4713.80 | 4605.53 | 4613.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 4696.90 | 4623.80 | 4620.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 4715.20 | 4642.08 | 4629.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 4610.15 | 4657.96 | 4643.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 4610.15 | 4657.96 | 4643.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 4610.15 | 4657.96 | 4643.37 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 4545.25 | 4621.24 | 4628.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 4504.95 | 4597.98 | 4617.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 11:15:00 | 4542.70 | 4540.78 | 4573.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 11:15:00 | 4542.70 | 4540.78 | 4573.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 4542.70 | 4540.78 | 4573.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:00:00 | 4542.70 | 4540.78 | 4573.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 4581.65 | 4548.98 | 4571.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 4608.20 | 4548.98 | 4571.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 4584.10 | 4556.00 | 4572.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 4584.10 | 4556.00 | 4572.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 4609.30 | 4570.82 | 4576.50 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 4595.15 | 4581.81 | 4580.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 4610.50 | 4597.45 | 4589.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 5015.40 | 5049.98 | 4932.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 5015.40 | 5049.98 | 4932.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 5015.40 | 5049.98 | 4932.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 4985.30 | 5049.98 | 4932.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 4938.55 | 5023.15 | 4940.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:30:00 | 4938.90 | 5023.15 | 4940.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 4940.45 | 5006.61 | 4940.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 4945.15 | 5006.61 | 4940.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 4978.00 | 5000.89 | 4944.07 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 4902.85 | 4944.61 | 4945.06 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 15:15:00 | 5010.00 | 4945.40 | 4943.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 5137.80 | 4983.88 | 4960.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 13:15:00 | 4959.10 | 5020.06 | 4990.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 13:15:00 | 4959.10 | 5020.06 | 4990.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 4959.10 | 5020.06 | 4990.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:45:00 | 4934.85 | 5020.06 | 4990.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 5052.35 | 5026.52 | 4996.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:30:00 | 5074.15 | 5039.41 | 5007.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:45:00 | 5058.60 | 5047.71 | 5014.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:45:00 | 5058.55 | 5056.11 | 5023.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-29 09:15:00 | 5564.46 | 5339.50 | 5289.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5030.15 | 5396.85 | 5436.62 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 5289.00 | 5255.56 | 5252.80 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 13:15:00 | 5200.00 | 5248.37 | 5250.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-10 14:15:00 | 5149.95 | 5209.60 | 5227.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 11:15:00 | 5238.80 | 5207.26 | 5219.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 11:15:00 | 5238.80 | 5207.26 | 5219.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 5238.80 | 5207.26 | 5219.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:30:00 | 5225.75 | 5207.26 | 5219.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 5195.40 | 5204.89 | 5217.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:30:00 | 5235.85 | 5204.89 | 5217.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 5247.75 | 5210.60 | 5215.80 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 5272.30 | 5222.94 | 5220.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 14:15:00 | 5415.35 | 5271.70 | 5244.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 12:15:00 | 5348.80 | 5354.40 | 5324.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 13:00:00 | 5348.80 | 5354.40 | 5324.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 5286.90 | 5340.90 | 5320.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 5286.90 | 5340.90 | 5320.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 5297.45 | 5332.21 | 5318.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 5297.45 | 5332.21 | 5318.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 5311.65 | 5329.34 | 5319.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 5311.65 | 5329.34 | 5319.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 5290.00 | 5321.47 | 5317.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:15:00 | 5282.90 | 5321.47 | 5317.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 5288.35 | 5314.85 | 5314.40 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 12:15:00 | 5300.00 | 5311.88 | 5313.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 13:15:00 | 5260.15 | 5301.53 | 5308.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 5149.40 | 5133.76 | 5172.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 14:15:00 | 5149.40 | 5133.76 | 5172.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 5149.40 | 5133.76 | 5172.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 5149.40 | 5133.76 | 5172.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 5090.00 | 5130.81 | 5165.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 5068.70 | 5130.81 | 5165.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 11:00:00 | 5073.90 | 5076.06 | 5110.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 5199.05 | 5119.44 | 5121.08 | SL hit (close>static) qty=1.00 sl=5179.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 5170.40 | 5129.63 | 5125.56 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 5075.25 | 5123.71 | 5129.33 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 5150.00 | 5130.03 | 5128.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 5340.10 | 5175.75 | 5150.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 5326.15 | 5327.80 | 5267.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 09:45:00 | 5314.25 | 5327.80 | 5267.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 5284.80 | 5312.12 | 5282.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:30:00 | 5273.35 | 5312.12 | 5282.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 5299.00 | 5309.50 | 5283.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 5250.05 | 5309.50 | 5283.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 5237.50 | 5295.10 | 5279.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 5243.80 | 5295.10 | 5279.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 5225.90 | 5281.26 | 5274.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 5222.00 | 5281.26 | 5274.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 5221.90 | 5269.39 | 5270.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 5185.60 | 5252.63 | 5262.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 13:15:00 | 5193.00 | 5185.70 | 5214.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:00:00 | 5193.00 | 5185.70 | 5214.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 5310.95 | 5210.75 | 5223.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 5310.95 | 5210.75 | 5223.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 5300.00 | 5228.60 | 5230.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 5283.45 | 5228.60 | 5230.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 5297.20 | 5242.32 | 5236.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 5394.00 | 5299.37 | 5270.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 11:15:00 | 5655.60 | 5666.04 | 5579.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 12:00:00 | 5655.60 | 5666.04 | 5579.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 5597.85 | 5652.40 | 5580.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 5597.85 | 5652.40 | 5580.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 5590.55 | 5640.03 | 5581.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:30:00 | 5583.55 | 5640.03 | 5581.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 5561.00 | 5624.23 | 5579.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 5561.00 | 5624.23 | 5579.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 5569.70 | 5613.32 | 5578.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 5445.00 | 5613.32 | 5578.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 5377.85 | 5536.11 | 5547.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 5341.00 | 5455.37 | 5504.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 15:15:00 | 5279.85 | 5255.62 | 5303.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 15:15:00 | 5279.85 | 5255.62 | 5303.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 5279.85 | 5255.62 | 5303.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 5220.00 | 5255.62 | 5303.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:15:00 | 4959.00 | 5083.47 | 5154.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 4978.25 | 4934.54 | 5006.27 | SL hit (close>ema200) qty=0.50 sl=4934.54 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 5024.65 | 4966.35 | 4961.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 5044.40 | 4990.15 | 4974.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 5049.80 | 5060.04 | 5024.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 5049.80 | 5060.04 | 5024.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 5044.65 | 5056.36 | 5045.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 5074.65 | 5056.36 | 5045.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 5049.55 | 5055.00 | 5045.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:30:00 | 5038.10 | 5055.00 | 5045.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 5043.70 | 5052.74 | 5045.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:00:00 | 5043.70 | 5052.74 | 5045.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 5057.45 | 5053.68 | 5046.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:30:00 | 5080.00 | 5057.26 | 5048.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 14:00:00 | 5071.60 | 5057.26 | 5048.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 5126.10 | 5057.76 | 5050.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 4985.00 | 5086.12 | 5087.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 4985.00 | 5086.12 | 5087.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 4796.00 | 5010.95 | 5052.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 12:15:00 | 4383.60 | 4373.72 | 4439.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 12:15:00 | 4383.60 | 4373.72 | 4439.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 4383.60 | 4373.72 | 4439.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 4383.60 | 4373.72 | 4439.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 4381.30 | 4288.88 | 4334.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 4381.30 | 4288.88 | 4334.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 4308.00 | 4292.70 | 4331.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 4260.75 | 4311.36 | 4327.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 4363.85 | 4322.83 | 4319.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 4363.85 | 4322.83 | 4319.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 12:15:00 | 4390.55 | 4353.80 | 4346.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 4507.95 | 4548.41 | 4507.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 4507.95 | 4548.41 | 4507.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 4507.95 | 4548.41 | 4507.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 4529.95 | 4548.41 | 4507.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 4516.35 | 4542.00 | 4508.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 4516.35 | 4542.00 | 4508.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 4498.95 | 4533.39 | 4507.59 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 4434.20 | 4485.01 | 4490.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 4407.65 | 4452.85 | 4473.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 4451.70 | 4444.64 | 4463.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 4451.70 | 4444.64 | 4463.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 4451.70 | 4444.64 | 4463.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 4418.00 | 4435.76 | 4450.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 4423.85 | 4429.43 | 4437.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 4394.50 | 4387.70 | 4403.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 4419.95 | 4392.95 | 4392.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 4419.95 | 4392.95 | 4392.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 4436.05 | 4406.27 | 4398.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 4368.65 | 4403.24 | 4399.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 4368.65 | 4403.24 | 4399.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 4368.65 | 4403.24 | 4399.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 4368.65 | 4403.24 | 4399.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 4372.55 | 4397.11 | 4396.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 4336.05 | 4397.11 | 4396.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 4358.80 | 4389.44 | 4393.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 4276.00 | 4309.14 | 4337.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 10:15:00 | 4323.40 | 4308.96 | 4329.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 10:15:00 | 4323.40 | 4308.96 | 4329.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 4323.40 | 4308.96 | 4329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:45:00 | 4336.00 | 4308.96 | 4329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 4342.10 | 4315.59 | 4330.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:00:00 | 4342.10 | 4315.59 | 4330.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 4372.00 | 4326.87 | 4334.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 4372.00 | 4326.87 | 4334.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 4361.60 | 4333.82 | 4337.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 4360.00 | 4333.82 | 4337.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 4335.00 | 4332.76 | 4335.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 4304.70 | 4332.76 | 4335.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 4353.20 | 4336.85 | 4337.47 | SL hit (close>static) qty=1.00 sl=4343.20 alert=retest2 |

### Cycle 94 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 4375.35 | 4335.55 | 4332.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 4389.90 | 4354.20 | 4341.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 5137.10 | 5172.99 | 5083.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 5137.10 | 5172.99 | 5083.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 5104.65 | 5159.32 | 5085.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 5107.50 | 5159.32 | 5085.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 5096.05 | 5146.67 | 5086.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 5070.35 | 5146.67 | 5086.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 5087.90 | 5131.45 | 5089.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 5087.90 | 5131.45 | 5089.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 5229.05 | 5150.97 | 5102.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 5380.95 | 5186.37 | 5144.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:00:00 | 5239.25 | 5253.67 | 5201.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:00:00 | 5238.00 | 5246.48 | 5207.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:15:00 | 5238.60 | 5219.68 | 5206.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 5212.45 | 5218.24 | 5207.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:45:00 | 5216.05 | 5218.24 | 5207.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 5316.25 | 5237.84 | 5217.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 5211.45 | 5237.84 | 5217.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 5227.55 | 5245.73 | 5224.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 5227.55 | 5245.73 | 5224.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 5290.15 | 5254.61 | 5230.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 5230.05 | 5254.61 | 5230.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 5220.20 | 5246.36 | 5231.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 5220.80 | 5246.36 | 5231.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 5193.15 | 5235.72 | 5227.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:15:00 | 5191.10 | 5235.72 | 5227.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 5203.75 | 5219.04 | 5221.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 5203.75 | 5219.04 | 5221.11 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 5240.00 | 5223.23 | 5222.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 11:15:00 | 5270.45 | 5232.67 | 5227.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 09:15:00 | 5251.20 | 5258.45 | 5244.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 5251.20 | 5258.45 | 5244.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 5251.20 | 5258.45 | 5244.36 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 5206.70 | 5235.35 | 5236.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 5200.15 | 5228.31 | 5233.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 5025.00 | 4969.93 | 5022.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 5025.00 | 4969.93 | 5022.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 5025.00 | 4969.93 | 5022.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 5025.00 | 4969.93 | 5022.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 5005.50 | 4977.04 | 5020.62 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 15:15:00 | 5180.00 | 5061.54 | 5051.33 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 5037.20 | 5052.26 | 5053.13 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 5207.00 | 5083.06 | 5066.42 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 5040.05 | 5085.05 | 5089.69 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 5101.85 | 5089.90 | 5089.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 5167.65 | 5105.45 | 5096.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 5093.55 | 5107.24 | 5099.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 5093.55 | 5107.24 | 5099.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 5093.55 | 5107.24 | 5099.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 5091.95 | 5107.24 | 5099.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 5091.85 | 5104.16 | 5098.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 5091.85 | 5104.16 | 5098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 5099.85 | 5103.30 | 5098.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 5135.15 | 5111.75 | 5103.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 5137.45 | 5109.40 | 5103.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 5128.70 | 5106.99 | 5103.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:15:00 | 5138.00 | 5107.21 | 5104.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 5138.00 | 5113.37 | 5107.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 5145.05 | 5113.37 | 5107.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 5100.00 | 5143.34 | 5137.77 | SL hit (close<static) qty=1.00 sl=5105.60 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 5100.05 | 5130.21 | 5133.02 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 5139.20 | 5133.99 | 5133.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 5199.25 | 5149.11 | 5140.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 5149.95 | 5160.44 | 5149.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 5149.95 | 5160.44 | 5149.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 5149.95 | 5160.44 | 5149.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 12:15:00 | 5191.50 | 5162.30 | 5151.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 5116.95 | 5155.52 | 5150.94 | SL hit (close<static) qty=1.00 sl=5135.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 5081.25 | 5136.41 | 5142.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 5050.00 | 5119.13 | 5134.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 5120.35 | 5097.61 | 5116.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 14:15:00 | 5120.35 | 5097.61 | 5116.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 5120.35 | 5097.61 | 5116.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 5120.35 | 5097.61 | 5116.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 5085.50 | 5095.19 | 5113.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 5030.05 | 5095.19 | 5113.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 5021.55 | 5067.91 | 5084.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 5188.75 | 5094.50 | 5088.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 14:15:00 | 5188.75 | 5094.50 | 5088.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 09:15:00 | 5310.00 | 5148.08 | 5114.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 13:15:00 | 5428.65 | 5449.75 | 5339.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 14:15:00 | 5307.45 | 5421.29 | 5336.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 5307.45 | 5421.29 | 5336.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 5307.45 | 5421.29 | 5336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 5327.95 | 5402.62 | 5335.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 5359.45 | 5402.62 | 5335.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 5145.80 | 5326.67 | 5311.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 5145.80 | 5326.67 | 5311.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 5144.60 | 5290.26 | 5296.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 5093.00 | 5250.81 | 5277.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 4991.60 | 4976.78 | 5047.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 4991.60 | 4976.78 | 5047.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 4991.00 | 4978.94 | 5036.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:30:00 | 5037.25 | 4978.94 | 5036.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 4986.70 | 4964.80 | 4991.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:30:00 | 4950.00 | 4964.80 | 4991.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 4992.50 | 4970.34 | 4991.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 4992.50 | 4970.34 | 4991.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 5006.95 | 4977.66 | 4993.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 5006.95 | 4977.66 | 4993.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 5020.00 | 4986.13 | 4995.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 5054.80 | 4986.13 | 4995.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 5061.10 | 5001.12 | 5001.57 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 5055.00 | 5011.90 | 5006.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 5095.00 | 5028.52 | 5014.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 14:15:00 | 5122.55 | 5133.67 | 5105.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 5122.55 | 5133.67 | 5105.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 5110.00 | 5128.94 | 5106.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 5215.00 | 5128.94 | 5106.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 5221.90 | 5147.53 | 5116.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:15:00 | 5264.80 | 5162.44 | 5126.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:45:00 | 5264.90 | 5199.06 | 5157.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 5244.90 | 5199.06 | 5157.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 15:15:00 | 5100.00 | 5173.00 | 5174.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 5100.00 | 5173.00 | 5174.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 5083.00 | 5155.00 | 5166.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 5007.85 | 5003.33 | 5068.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:15:00 | 5081.60 | 5003.33 | 5068.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 5080.05 | 5018.68 | 5069.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 5091.35 | 5018.68 | 5069.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 5091.75 | 5033.29 | 5071.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:15:00 | 5107.80 | 5033.29 | 5071.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 5107.70 | 5048.17 | 5075.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:30:00 | 5100.00 | 5048.17 | 5075.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 5018.70 | 5027.95 | 5055.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:15:00 | 4960.00 | 5020.43 | 5049.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:15:00 | 4712.00 | 4875.46 | 4956.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-21 09:15:00 | 4464.00 | 4645.16 | 4787.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 4607.50 | 4529.93 | 4526.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 4657.40 | 4616.94 | 4590.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 4607.55 | 4615.64 | 4596.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 4607.55 | 4615.64 | 4596.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 4563.90 | 4606.07 | 4595.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 4563.90 | 4606.07 | 4595.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 4593.95 | 4603.65 | 4595.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 4584.60 | 4603.65 | 4595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 4519.55 | 4586.83 | 4588.48 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 4633.00 | 4591.70 | 4587.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 4697.00 | 4616.88 | 4600.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 4738.05 | 4754.72 | 4706.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 4738.05 | 4754.72 | 4706.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 4738.05 | 4754.72 | 4706.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 4714.55 | 4754.72 | 4706.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 4704.80 | 4738.11 | 4716.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 4704.80 | 4738.11 | 4716.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 4658.00 | 4722.08 | 4711.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 4659.05 | 4722.08 | 4711.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 4686.90 | 4711.36 | 4708.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:45:00 | 4703.90 | 4710.19 | 4708.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 4713.40 | 4710.83 | 4708.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 4654.80 | 4698.85 | 4703.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 4654.80 | 4698.85 | 4703.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 4637.70 | 4656.66 | 4664.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 4705.55 | 4662.68 | 4664.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 4705.55 | 4662.68 | 4664.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 4705.55 | 4662.68 | 4664.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 4705.55 | 4662.68 | 4664.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 4715.05 | 4673.15 | 4669.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 4732.50 | 4685.02 | 4674.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 4725.00 | 4754.22 | 4732.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 4725.00 | 4754.22 | 4732.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 4725.00 | 4754.22 | 4732.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 4725.00 | 4754.22 | 4732.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 4699.10 | 4743.20 | 4729.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 4698.35 | 4743.20 | 4729.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 4720.25 | 4738.61 | 4728.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:45:00 | 4739.00 | 4736.90 | 4729.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 4725.00 | 4803.32 | 4804.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 4725.00 | 4803.32 | 4804.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 4646.05 | 4737.82 | 4766.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 14:15:00 | 4675.30 | 4609.71 | 4657.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 4675.30 | 4609.71 | 4657.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 4675.30 | 4609.71 | 4657.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 4675.30 | 4609.71 | 4657.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 4628.45 | 4613.46 | 4654.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 4599.10 | 4613.46 | 4654.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 4369.15 | 4457.04 | 4520.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 4458.00 | 4457.23 | 4515.06 | SL hit (close>ema200) qty=0.50 sl=4457.23 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 4151.50 | 4072.20 | 4066.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 09:15:00 | 4184.45 | 4154.22 | 4129.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 12:15:00 | 4157.50 | 4158.28 | 4138.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 4155.05 | 4158.28 | 4138.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 4147.05 | 4156.04 | 4138.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 4142.90 | 4156.04 | 4138.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 4084.55 | 4141.74 | 4133.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 4084.55 | 4141.74 | 4133.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 4121.00 | 4137.59 | 4132.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 4055.25 | 4137.59 | 4132.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 4052.60 | 4120.59 | 4125.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 4022.75 | 4101.02 | 4116.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 3800.10 | 3707.55 | 3773.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 3800.10 | 3707.55 | 3773.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 3800.10 | 3707.55 | 3773.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 3800.10 | 3707.55 | 3773.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 3744.95 | 3715.03 | 3770.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 3726.85 | 3715.03 | 3770.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 3728.20 | 3717.65 | 3758.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 3824.35 | 3746.54 | 3761.81 | SL hit (close>static) qty=1.00 sl=3805.20 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 4023.85 | 3810.97 | 3787.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 4069.90 | 3862.76 | 3813.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 09:15:00 | 3821.35 | 3854.47 | 3814.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 10:00:00 | 3821.35 | 3854.47 | 3814.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 3845.00 | 3852.58 | 3816.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:45:00 | 3831.20 | 3852.58 | 3816.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 3834.10 | 3846.00 | 3819.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 3836.65 | 3846.00 | 3819.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 3953.30 | 3867.46 | 3831.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 4016.95 | 3867.46 | 3831.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 3989.10 | 3925.53 | 3871.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:00:00 | 4001.95 | 3940.81 | 3882.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:30:00 | 3994.75 | 3950.37 | 3933.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 3951.90 | 3953.02 | 3937.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 3938.20 | 3953.02 | 3937.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3935.35 | 3949.48 | 3937.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 13:15:00 | 3901.90 | 3927.48 | 3929.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 3901.90 | 3927.48 | 3929.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 3831.60 | 3902.87 | 3917.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 3795.00 | 3785.91 | 3833.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 15:15:00 | 3800.00 | 3794.79 | 3820.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 3800.00 | 3794.79 | 3820.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 3790.00 | 3794.79 | 3820.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3747.25 | 3785.28 | 3814.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 3736.65 | 3764.81 | 3799.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 3549.82 | 3587.62 | 3661.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 3604.80 | 3582.95 | 3640.02 | SL hit (close>ema200) qty=0.50 sl=3582.95 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 3727.95 | 3662.72 | 3657.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 3758.10 | 3681.80 | 3666.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 3717.15 | 3718.38 | 3694.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 3717.15 | 3718.38 | 3694.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 3691.80 | 3715.27 | 3697.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 3691.90 | 3715.27 | 3697.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 3678.00 | 3707.81 | 3695.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:45:00 | 3679.85 | 3707.81 | 3695.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 3677.00 | 3701.65 | 3693.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 3711.30 | 3701.65 | 3693.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 3594.75 | 3788.61 | 3774.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 3594.75 | 3788.61 | 3774.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 3623.35 | 3755.56 | 3760.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 14:15:00 | 3517.85 | 3598.05 | 3648.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 3325.00 | 3319.50 | 3392.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:30:00 | 3319.00 | 3319.50 | 3392.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 3364.90 | 3328.58 | 3389.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 3383.20 | 3328.58 | 3389.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 3400.35 | 3356.74 | 3388.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 3403.15 | 3356.74 | 3388.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 3382.00 | 3361.80 | 3387.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:30:00 | 3407.85 | 3361.80 | 3387.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 3384.65 | 3366.37 | 3387.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 3349.60 | 3366.37 | 3387.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 3182.12 | 3272.72 | 3323.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 3202.90 | 3186.47 | 3243.02 | SL hit (close>ema200) qty=0.50 sl=3186.47 alert=retest2 |

### Cycle 122 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 3171.95 | 3106.23 | 3099.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 3351.20 | 3182.65 | 3138.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 3315.55 | 3328.02 | 3266.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 3315.55 | 3328.02 | 3266.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 3274.75 | 3318.20 | 3281.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 3274.75 | 3318.20 | 3281.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 3260.20 | 3306.60 | 3279.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 3260.20 | 3306.60 | 3279.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 3298.75 | 3305.03 | 3281.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:30:00 | 3315.40 | 3302.80 | 3282.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 3330.80 | 3306.06 | 3287.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 3240.15 | 3287.73 | 3281.91 | SL hit (close<static) qty=1.00 sl=3254.15 alert=retest2 |

### Cycle 123 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 3224.10 | 3275.93 | 3280.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 3203.85 | 3223.14 | 3246.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 3196.80 | 3193.42 | 3221.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 3196.80 | 3193.42 | 3221.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 3207.95 | 3192.88 | 3215.78 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 3380.00 | 3238.57 | 3226.54 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 11:15:00 | 3197.00 | 3220.59 | 3220.71 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 3273.95 | 3225.28 | 3222.06 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 15:15:00 | 3204.00 | 3221.17 | 3222.00 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 3230.85 | 3223.11 | 3222.80 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 10:15:00 | 3218.90 | 3222.26 | 3222.45 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 3339.30 | 3243.86 | 3231.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 3364.00 | 3267.88 | 3243.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 3370.05 | 3374.96 | 3340.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:45:00 | 3370.20 | 3374.96 | 3340.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 3339.10 | 3367.79 | 3340.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 3339.10 | 3367.79 | 3340.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 3327.70 | 3359.77 | 3338.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:15:00 | 3321.00 | 3359.77 | 3338.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 3321.00 | 3352.02 | 3337.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 3320.20 | 3352.02 | 3337.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 3329.95 | 3347.60 | 3336.67 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 3276.70 | 3322.68 | 3327.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 3260.50 | 3292.66 | 3309.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 3269.60 | 3253.61 | 3280.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 3269.60 | 3253.61 | 3280.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 3344.00 | 3275.11 | 3285.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 3344.00 | 3275.11 | 3285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 3394.20 | 3298.93 | 3295.44 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 3234.85 | 3306.28 | 3308.74 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 3350.05 | 3290.49 | 3286.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 3376.45 | 3307.68 | 3294.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 3505.35 | 3517.61 | 3465.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 14:15:00 | 3505.35 | 3517.61 | 3465.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 3505.35 | 3517.61 | 3465.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 3505.35 | 3517.61 | 3465.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 3475.20 | 3492.77 | 3474.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:45:00 | 3474.90 | 3492.77 | 3474.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 3461.80 | 3486.58 | 3473.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 3461.80 | 3486.58 | 3473.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 3479.00 | 3485.06 | 3474.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 3523.10 | 3485.06 | 3474.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 3480.05 | 3517.76 | 3511.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 3461.25 | 3499.77 | 3504.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 3461.25 | 3499.77 | 3504.18 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 3537.30 | 3504.75 | 3503.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 3767.00 | 3569.96 | 3534.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 15:15:00 | 3630.60 | 3636.14 | 3598.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:15:00 | 3601.00 | 3636.14 | 3598.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 3579.35 | 3624.79 | 3596.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 3575.30 | 3624.79 | 3596.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 3570.55 | 3613.94 | 3594.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 3570.55 | 3613.94 | 3594.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 3554.25 | 3581.34 | 3583.04 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 3629.00 | 3586.65 | 3584.90 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 3529.05 | 3575.13 | 3579.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 13:15:00 | 3514.25 | 3557.67 | 3570.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 10:15:00 | 3555.30 | 3549.13 | 3561.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 11:00:00 | 3555.30 | 3549.13 | 3561.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 3545.50 | 3548.41 | 3560.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:45:00 | 3552.95 | 3548.41 | 3560.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 3463.05 | 3528.79 | 3546.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 3455.10 | 3528.79 | 3546.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 3328.55 | 3516.29 | 3528.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 3109.59 | 3467.53 | 3505.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 3253.90 | 3177.52 | 3171.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 3301.90 | 3212.25 | 3189.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 3422.00 | 3446.51 | 3416.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 3422.00 | 3446.51 | 3416.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 3423.70 | 3437.19 | 3419.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 3421.60 | 3437.19 | 3419.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 3424.50 | 3434.66 | 3420.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 3421.10 | 3434.66 | 3420.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 3449.80 | 3437.68 | 3422.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 3423.40 | 3437.68 | 3422.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 3420.00 | 3434.15 | 3422.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 3611.80 | 3434.15 | 3422.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 3454.30 | 3505.61 | 3504.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 3454.30 | 3495.35 | 3499.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 3454.30 | 3495.35 | 3499.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 09:15:00 | 3367.10 | 3425.65 | 3456.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 3227.00 | 3211.56 | 3237.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 3229.00 | 3211.56 | 3237.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 3223.20 | 3213.89 | 3235.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 3225.50 | 3213.89 | 3235.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 3235.20 | 3218.15 | 3235.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 3235.20 | 3218.15 | 3235.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 3248.90 | 3224.30 | 3237.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 3248.90 | 3224.30 | 3237.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 3250.00 | 3229.44 | 3238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 3240.80 | 3229.44 | 3238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 3242.00 | 3231.95 | 3238.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 3204.80 | 3226.32 | 3234.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 3251.00 | 3209.18 | 3207.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3251.00 | 3209.18 | 3207.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 3416.50 | 3343.31 | 3303.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 3420.50 | 3428.47 | 3402.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:00:00 | 3420.50 | 3428.47 | 3402.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3556.80 | 3574.59 | 3553.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 3556.80 | 3574.59 | 3553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 3560.70 | 3571.81 | 3554.14 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 3514.80 | 3545.67 | 3547.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 3455.20 | 3491.71 | 3514.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 15:15:00 | 3493.00 | 3475.25 | 3494.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 3489.00 | 3476.91 | 3491.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 3489.00 | 3476.91 | 3491.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 3440.90 | 3477.37 | 3486.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 3449.30 | 3436.04 | 3435.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 3449.30 | 3436.04 | 3435.92 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3420.90 | 3433.01 | 3434.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 3394.50 | 3423.48 | 3429.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 3420.30 | 3412.06 | 3421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 3474.90 | 3424.62 | 3426.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 3474.90 | 3424.62 | 3426.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 3501.50 | 3440.00 | 3432.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 3515.50 | 3455.10 | 3440.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 3524.80 | 3526.87 | 3495.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 15:00:00 | 3524.80 | 3526.87 | 3495.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 3552.00 | 3558.16 | 3545.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 3552.00 | 3558.16 | 3545.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3549.80 | 3558.86 | 3548.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 3546.90 | 3558.86 | 3548.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3545.80 | 3556.25 | 3548.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 3545.80 | 3556.25 | 3548.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 3542.30 | 3553.46 | 3548.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 3544.00 | 3553.46 | 3548.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3540.10 | 3550.79 | 3547.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 3540.10 | 3550.79 | 3547.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 3563.40 | 3552.87 | 3549.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 3545.00 | 3552.87 | 3549.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 3555.00 | 3564.58 | 3558.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 3508.60 | 3564.58 | 3558.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3555.40 | 3562.75 | 3558.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 3573.90 | 3562.75 | 3558.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 3528.40 | 3554.83 | 3556.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 3528.40 | 3554.83 | 3556.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 3483.30 | 3538.87 | 3548.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 3534.10 | 3521.84 | 3534.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3531.50 | 3523.77 | 3534.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 3534.50 | 3523.77 | 3534.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3522.10 | 3523.43 | 3532.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 3491.50 | 3516.15 | 3527.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 3505.00 | 3498.18 | 3513.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 3505.20 | 3498.18 | 3513.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 3504.70 | 3499.34 | 3512.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 3503.40 | 3501.40 | 3510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 3501.00 | 3501.40 | 3510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 3513.70 | 3503.86 | 3510.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 3513.70 | 3503.86 | 3510.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 3517.00 | 3506.49 | 3511.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 3558.90 | 3506.49 | 3511.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 3475.20 | 3521.50 | 3522.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3456.70 | 3493.07 | 3508.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 3484.00 | 3462.26 | 3486.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 3476.80 | 3462.26 | 3486.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3490.00 | 3467.81 | 3486.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 3488.00 | 3467.81 | 3486.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 3503.40 | 3474.93 | 3488.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 3503.40 | 3474.93 | 3488.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 3465.00 | 3475.08 | 3484.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 3456.50 | 3475.08 | 3484.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 3457.30 | 3453.31 | 3465.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:00:00 | 3462.10 | 3442.13 | 3450.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 3464.90 | 3449.86 | 3453.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 3450.00 | 3454.85 | 3455.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 3443.20 | 3452.52 | 3454.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 3410.30 | 3407.27 | 3428.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:15:00 | 3429.40 | 3407.27 | 3428.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 3441.30 | 3414.08 | 3429.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 3433.10 | 3414.08 | 3429.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 3443.00 | 3419.86 | 3430.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 3447.60 | 3419.86 | 3430.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 3407.90 | 3417.47 | 3428.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:30:00 | 3403.70 | 3416.06 | 3427.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 3401.40 | 3412.37 | 3422.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 3404.30 | 3377.23 | 3391.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 3404.60 | 3384.23 | 3392.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 3434.80 | 3394.34 | 3396.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 3434.80 | 3394.34 | 3396.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 3489.30 | 3418.93 | 3407.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 3460.10 | 3474.41 | 3453.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 3460.10 | 3474.41 | 3453.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 3455.00 | 3470.53 | 3454.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 3451.80 | 3470.53 | 3454.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 3453.00 | 3467.03 | 3453.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:15:00 | 3442.40 | 3467.03 | 3453.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3459.50 | 3465.52 | 3454.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 3449.50 | 3465.52 | 3454.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 3450.20 | 3462.46 | 3454.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 3455.00 | 3462.46 | 3454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3490.80 | 3468.12 | 3457.37 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 3438.60 | 3452.17 | 3453.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 3429.70 | 3447.68 | 3451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 3452.90 | 3435.14 | 3441.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 3440.00 | 3436.11 | 3441.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 3435.80 | 3438.16 | 3441.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 3435.00 | 3438.57 | 3441.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 3434.00 | 3429.18 | 3435.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 3473.10 | 3443.20 | 3438.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 3470.70 | 3447.60 | 3444.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:30:00 | 3471.50 | 3451.40 | 3446.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 3469.80 | 3451.40 | 3446.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 3471.30 | 3461.18 | 3452.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 3817.77 | 3647.74 | 3569.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 3872.00 | 3876.37 | 3876.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 3852.60 | 3869.00 | 3872.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 3766.50 | 3741.29 | 3768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 3769.80 | 3746.99 | 3768.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 3763.00 | 3746.99 | 3768.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 3794.00 | 3769.61 | 3773.69 | SL hit (close>static) qty=1.00 sl=3792.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 3817.30 | 3779.15 | 3777.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 3847.80 | 3816.28 | 3798.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 3821.00 | 3818.06 | 3804.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 3836.00 | 3867.41 | 3847.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 3767.00 | 3823.23 | 3830.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 3767.00 | 3823.23 | 3830.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 3697.00 | 3797.99 | 3818.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 3566.00 | 3560.74 | 3635.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 15:00:00 | 3566.00 | 3560.74 | 3635.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3298.20 | 3283.92 | 3319.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 3300.00 | 3283.92 | 3319.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 3216.70 | 3278.03 | 3301.26 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 3298.10 | 3287.88 | 3286.74 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 3266.70 | 3286.80 | 3286.85 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 3303.00 | 3286.74 | 3285.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 3314.50 | 3292.29 | 3288.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 3266.00 | 3291.37 | 3288.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 3280.90 | 3289.28 | 3287.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 3278.70 | 3289.28 | 3287.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 3278.10 | 3287.04 | 3286.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 3278.90 | 3287.04 | 3286.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 3261.60 | 3281.95 | 3284.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 3234.20 | 3269.29 | 3278.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 3241.30 | 3230.51 | 3250.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:45:00 | 3238.70 | 3230.51 | 3250.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 3230.20 | 3232.13 | 3246.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 3225.10 | 3230.40 | 3242.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 3271.00 | 3243.96 | 3247.26 | SL hit (close>static) qty=1.00 sl=3268.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 3279.30 | 3254.24 | 3251.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 3301.00 | 3275.42 | 3265.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 3259.40 | 3283.31 | 3273.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 3299.00 | 3286.45 | 3275.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 3359.80 | 3286.45 | 3275.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 3232.40 | 3270.59 | 3271.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 3232.40 | 3270.59 | 3271.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 3177.30 | 3251.93 | 3263.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 13:15:00 | 3207.90 | 3207.80 | 3229.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:45:00 | 3211.40 | 3207.80 | 3229.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3183.60 | 3204.48 | 3222.70 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 3255.90 | 3225.93 | 3224.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 3268.90 | 3234.53 | 3228.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 3269.40 | 3264.17 | 3248.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 3295.90 | 3312.80 | 3293.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 3295.90 | 3312.80 | 3293.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 3267.80 | 3303.80 | 3290.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 3269.60 | 3303.80 | 3290.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3262.10 | 3295.46 | 3288.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 3266.60 | 3295.46 | 3288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 3262.30 | 3288.34 | 3286.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 3262.30 | 3288.34 | 3286.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 3315.10 | 3293.69 | 3288.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 3320.80 | 3305.05 | 3294.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 3437.00 | 3329.88 | 3323.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 3313.20 | 3341.61 | 3344.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 3313.20 | 3341.61 | 3344.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 3306.00 | 3318.80 | 3329.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 3329.00 | 3312.45 | 3320.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3322.80 | 3314.52 | 3321.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 3306.60 | 3314.56 | 3319.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3345.70 | 3325.27 | 3322.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 3345.70 | 3325.27 | 3322.99 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 3303.60 | 3320.91 | 3321.62 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 3329.10 | 3322.02 | 3321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 12:15:00 | 3351.30 | 3327.87 | 3324.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 3330.00 | 3330.02 | 3325.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 3324.00 | 3328.81 | 3325.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 3336.30 | 3328.81 | 3325.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 3359.20 | 3334.89 | 3328.73 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 3299.20 | 3325.17 | 3325.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 3291.20 | 3310.14 | 3316.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 3350.20 | 3306.53 | 3312.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 3329.80 | 3311.18 | 3313.96 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 3359.00 | 3324.43 | 3319.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 3367.10 | 3332.96 | 3324.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 3335.70 | 3345.05 | 3335.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 3323.70 | 3340.78 | 3334.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 3323.70 | 3340.78 | 3334.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 3313.60 | 3335.34 | 3332.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 3313.60 | 3335.34 | 3332.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 3282.50 | 3322.14 | 3326.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 3267.10 | 3311.13 | 3321.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 3172.10 | 3162.16 | 3195.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:30:00 | 3184.00 | 3162.16 | 3195.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 3178.10 | 3170.90 | 3186.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 3163.00 | 3168.65 | 3182.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 3159.90 | 3166.15 | 3177.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 3163.00 | 3165.52 | 3176.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 3151.90 | 3167.17 | 3174.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 3173.60 | 3168.46 | 3174.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 3173.60 | 3168.46 | 3174.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 3189.10 | 3171.37 | 3174.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 3189.10 | 3171.37 | 3174.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 3182.00 | 3173.50 | 3175.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 3170.00 | 3173.50 | 3175.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 3172.20 | 3166.86 | 3169.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 3162.50 | 3166.86 | 3169.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3134.20 | 3163.65 | 3166.66 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 3180.30 | 3166.05 | 3164.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 3198.70 | 3174.61 | 3168.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3184.60 | 3200.72 | 3187.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 3192.20 | 3199.01 | 3188.34 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 3155.00 | 3181.90 | 3184.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 3121.70 | 3165.57 | 3176.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3165.00 | 3127.08 | 3140.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 3165.30 | 3134.72 | 3143.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 3167.90 | 3134.72 | 3143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 3152.00 | 3138.18 | 3143.82 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3168.30 | 3149.29 | 3148.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3208.00 | 3166.08 | 3156.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 3157.00 | 3167.51 | 3159.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3132.60 | 3160.53 | 3156.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 3135.10 | 3160.53 | 3156.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 3121.00 | 3152.62 | 3153.43 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3191.70 | 3158.24 | 3155.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 3231.60 | 3205.05 | 3186.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 3184.70 | 3209.54 | 3195.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 3184.80 | 3204.59 | 3194.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 3212.00 | 3204.59 | 3194.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 3253.30 | 3258.75 | 3258.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 3253.30 | 3258.75 | 3258.78 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 3268.70 | 3260.74 | 3259.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 3277.90 | 3264.17 | 3261.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 3248.40 | 3260.97 | 3261.08 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 3264.20 | 3261.62 | 3261.37 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 3250.50 | 3259.39 | 3260.38 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 3271.30 | 3261.77 | 3261.37 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 3254.00 | 3260.22 | 3260.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 3241.10 | 3256.40 | 3258.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 3287.00 | 3225.81 | 3231.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 3262.80 | 3233.21 | 3234.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 3250.10 | 3233.21 | 3234.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 3258.20 | 3238.20 | 3236.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 3258.20 | 3238.20 | 3236.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 3267.10 | 3243.98 | 3239.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 3239.20 | 3252.90 | 3245.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3257.30 | 3253.78 | 3246.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 3245.00 | 3253.78 | 3246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 3251.00 | 3253.22 | 3247.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 3251.00 | 3253.22 | 3247.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 3235.50 | 3249.68 | 3245.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 3236.80 | 3249.68 | 3245.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 3237.30 | 3247.20 | 3245.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 3248.70 | 3248.90 | 3246.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3178.70 | 3233.20 | 3239.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 3178.70 | 3233.20 | 3239.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 3163.00 | 3219.16 | 3232.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 3176.70 | 3157.08 | 3177.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 3158.80 | 3157.42 | 3175.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 3055.70 | 3170.97 | 3177.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 2902.91 | 2954.41 | 2985.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 2975.00 | 2940.24 | 2958.67 | SL hit (close>ema200) qty=0.50 sl=2940.24 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 2921.00 | 2897.20 | 2894.32 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 2903.40 | 2916.48 | 2916.68 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 2919.20 | 2917.03 | 2916.91 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 2915.20 | 2916.66 | 2916.75 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 2920.00 | 2917.30 | 2917.03 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 2915.00 | 2916.89 | 2916.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 2902.20 | 2913.95 | 2915.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 2914.10 | 2911.10 | 2913.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2926.30 | 2914.14 | 2914.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 2926.30 | 2914.14 | 2914.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 2922.50 | 2915.81 | 2915.32 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 2907.60 | 2914.17 | 2914.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 2877.30 | 2894.14 | 2902.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2811.50 | 2799.48 | 2828.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 2811.50 | 2799.48 | 2828.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2824.10 | 2801.10 | 2813.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 2824.10 | 2801.10 | 2813.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 2828.60 | 2806.60 | 2814.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2815.00 | 2812.61 | 2816.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 2916.00 | 2812.35 | 2803.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 2916.00 | 2812.35 | 2803.63 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 2836.50 | 2864.93 | 2868.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 2831.90 | 2853.52 | 2862.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 2854.00 | 2853.33 | 2859.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 2865.90 | 2853.33 | 2859.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2852.80 | 2853.22 | 2859.14 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2870.40 | 2862.66 | 2862.09 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 15:15:00 | 2853.00 | 2860.73 | 2861.27 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2883.50 | 2865.28 | 2863.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 2894.20 | 2878.87 | 2872.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 3001.00 | 3004.39 | 2978.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 10:45:00 | 2995.90 | 3004.39 | 2978.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2971.90 | 2999.41 | 2987.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 3010.50 | 3000.43 | 2990.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 3006.60 | 3006.32 | 2999.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 3011.80 | 3010.26 | 3004.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 3083.80 | 3042.75 | 3039.70 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 3033.40 | 3051.57 | 3052.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 3007.00 | 3042.65 | 3048.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 2966.60 | 2964.90 | 2989.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 14:15:00 | 2946.40 | 2963.60 | 2983.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:15:00 | 2947.70 | 2962.92 | 2979.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2955.00 | 2961.34 | 2977.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 2975.70 | 2969.62 | 2974.76 | SL hit (close>ema400) qty=1.00 sl=2974.76 alert=retest1 |

### Cycle 204 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 2923.20 | 2928.34 | 2928.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2859.50 | 2911.90 | 2920.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 2857.60 | 2856.05 | 2881.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:45:00 | 2860.70 | 2856.05 | 2881.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2880.00 | 2863.07 | 2873.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 2851.10 | 2863.07 | 2873.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2893.00 | 2855.05 | 2850.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2893.00 | 2855.05 | 2850.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2906.40 | 2872.50 | 2860.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2868.00 | 2884.23 | 2869.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2877.70 | 2882.92 | 2870.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 2871.70 | 2882.92 | 2870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2897.50 | 2886.61 | 2875.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 2853.00 | 2886.61 | 2875.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2825.00 | 2874.29 | 2870.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 2825.00 | 2874.29 | 2870.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 2822.20 | 2863.87 | 2866.42 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2945.00 | 2862.41 | 2862.08 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2854.50 | 2912.55 | 2917.13 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 2939.10 | 2900.74 | 2899.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 2949.90 | 2918.07 | 2907.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 2939.40 | 2946.63 | 2930.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2937.80 | 2944.86 | 2931.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 2936.20 | 2944.86 | 2931.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 2935.80 | 2943.05 | 2931.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 2935.80 | 2943.05 | 2931.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 2931.70 | 2940.78 | 2931.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 2931.70 | 2940.78 | 2931.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 2925.00 | 2937.62 | 2931.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 2903.70 | 2937.62 | 2931.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2896.40 | 2929.38 | 2928.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 2896.40 | 2929.38 | 2928.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 2892.90 | 2922.08 | 2924.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 2824.00 | 2897.35 | 2912.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 2857.50 | 2830.18 | 2863.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 13:00:00 | 2857.50 | 2830.18 | 2863.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 2877.90 | 2839.72 | 2864.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:30:00 | 2882.00 | 2839.72 | 2864.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 2890.00 | 2849.78 | 2867.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2864.50 | 2855.82 | 2868.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 2867.70 | 2858.20 | 2868.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 2908.00 | 2872.56 | 2872.60 | SL hit (close>static) qty=1.00 sl=2895.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 2892.90 | 2876.62 | 2874.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 2925.90 | 2886.48 | 2879.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3004.90 | 3059.93 | 3035.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3023.50 | 3052.64 | 3034.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3027.80 | 3037.09 | 3029.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 3026.20 | 3042.27 | 3032.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 3100.60 | 3151.23 | 3156.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 3100.60 | 3151.23 | 3156.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 3054.50 | 3108.94 | 3130.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 3100.90 | 3080.54 | 3105.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3060.40 | 3033.38 | 3058.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 3067.90 | 3033.38 | 3058.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 3052.10 | 3037.12 | 3058.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 3057.30 | 3037.12 | 3058.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 3059.20 | 3041.54 | 3058.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 3059.20 | 3041.54 | 3058.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3045.90 | 3042.41 | 3057.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 3045.80 | 3042.53 | 3055.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 3070.80 | 3048.18 | 3057.33 | SL hit (close>static) qty=1.00 sl=3062.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 3137.40 | 3066.80 | 3064.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 3154.50 | 3084.34 | 3072.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 3070.10 | 3089.82 | 3091.32 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3199.00 | 3111.07 | 3099.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 3208.20 | 3161.05 | 3133.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 3185.00 | 3185.50 | 3160.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 3118.10 | 3185.50 | 3160.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3117.30 | 3171.86 | 3156.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 3117.10 | 3171.86 | 3156.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 3112.50 | 3159.99 | 3152.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:45:00 | 3108.80 | 3159.99 | 3152.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 3097.80 | 3147.55 | 3147.72 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 3165.60 | 3149.13 | 3147.73 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3137.20 | 3146.74 | 3146.77 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 3202.80 | 3157.34 | 3151.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3220.60 | 3183.97 | 3166.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3188.50 | 3247.99 | 3223.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 3237.90 | 3245.97 | 3224.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 3260.20 | 3246.96 | 3228.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:00:00 | 3263.10 | 3245.19 | 3232.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 15:15:00 | 3210.00 | 3267.62 | 3265.01 | SL hit (close<static) qty=1.00 sl=3212.50 alert=retest2 |

### Cycle 221 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 3236.00 | 3261.30 | 3262.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3069.10 | 3200.87 | 3229.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3152.50 | 3124.74 | 3160.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 3152.50 | 3124.74 | 3160.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3202.90 | 3140.37 | 3163.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 3202.90 | 3140.37 | 3163.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 3235.40 | 3159.37 | 3170.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 3232.80 | 3159.37 | 3170.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3275.80 | 3191.20 | 3183.06 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3158.10 | 3201.45 | 3205.23 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 3223.70 | 3208.41 | 3207.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 3250.80 | 3216.89 | 3211.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 3222.00 | 3313.92 | 3277.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3210.00 | 3293.14 | 3271.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 3210.00 | 3293.14 | 3271.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 3255.00 | 3285.51 | 3270.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 3273.30 | 3284.43 | 3271.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 3258.80 | 3279.88 | 3274.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 3259.00 | 3273.42 | 3274.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 3259.00 | 3273.42 | 3274.00 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 3290.00 | 3263.72 | 3260.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 3404.70 | 3294.68 | 3275.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 15:15:00 | 4080.10 | 4096.26 | 3988.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 4094.10 | 4095.83 | 3998.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4284.80 | 4150.22 | 4073.06 | EMA400 retest candle locked (from upside) |

### Cycle 227 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 4060.00 | 4111.07 | 4114.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 15:15:00 | 4049.80 | 4079.68 | 4097.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4083.20 | 4080.38 | 4095.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 10:00:00 | 4083.20 | 4080.38 | 4095.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 4126.10 | 4089.53 | 4098.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 4130.60 | 4089.53 | 4098.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 4083.10 | 4088.24 | 4097.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:15:00 | 4066.80 | 4088.24 | 4097.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:00:00 | 4081.90 | 4036.14 | 4054.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 4081.60 | 4047.05 | 4057.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 4029.30 | 4072.35 | 4077.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 4023.60 | 4062.60 | 4072.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 4004.90 | 4001.39 | 4028.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 3962.30 | 3993.57 | 4022.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 4020.20 | 3992.61 | 4016.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 4020.20 | 3992.61 | 4016.49 | SL hit (close>ema400) qty=1.00 sl=4016.49 alert=retest1 |

### Cycle 230 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 4100.00 | 4036.87 | 4032.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 4136.00 | 4070.83 | 4050.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 4142.40 | 4146.92 | 4110.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 4142.40 | 4146.92 | 4110.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4117.40 | 4141.01 | 4111.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 4121.50 | 4141.01 | 4111.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 4112.80 | 4135.37 | 4111.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 4109.80 | 4135.37 | 4111.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4125.00 | 4133.30 | 4112.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4106.00 | 4133.30 | 4112.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4108.20 | 4128.28 | 4112.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 4108.20 | 4128.28 | 4112.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 4121.40 | 4126.90 | 4113.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 4096.80 | 4126.90 | 4113.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 4089.90 | 4119.50 | 4110.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 4098.40 | 4119.50 | 4110.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 4089.80 | 4113.56 | 4109.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 4088.10 | 4113.56 | 4109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 4068.00 | 4104.45 | 4105.28 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 4277.70 | 4129.67 | 4115.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 4530.00 | 4262.62 | 4195.58 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 09:30:00 | 4781.00 | 2024-04-22 13:15:00 | 4649.10 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-04-19 12:30:00 | 4792.20 | 2024-04-22 13:15:00 | 4649.10 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-04-22 10:45:00 | 4788.30 | 2024-04-22 13:15:00 | 4649.10 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest1 | 2024-04-26 10:45:00 | 4319.00 | 2024-04-29 09:15:00 | 4443.20 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-05-23 09:30:00 | 5074.15 | 2024-05-29 09:15:00 | 5564.46 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2024-05-23 10:45:00 | 5058.60 | 2024-05-29 09:15:00 | 5564.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-23 12:45:00 | 5058.55 | 2024-06-03 09:15:00 | 5581.56 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2024-06-04 11:15:00 | 5243.75 | 2024-06-04 11:15:00 | 5030.15 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2024-06-21 10:15:00 | 5068.70 | 2024-06-24 14:15:00 | 5199.05 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-06-24 11:00:00 | 5073.90 | 2024-06-24 14:15:00 | 5199.05 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-07-15 09:15:00 | 5220.00 | 2024-07-18 10:15:00 | 4959.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:15:00 | 5220.00 | 2024-07-19 14:15:00 | 4978.25 | STOP_HIT | 0.50 | 4.63% |
| BUY | retest2 | 2024-07-31 13:30:00 | 5080.00 | 2024-08-02 14:15:00 | 4985.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-07-31 14:00:00 | 5071.60 | 2024-08-02 14:15:00 | 4985.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-08-01 09:15:00 | 5126.10 | 2024-08-02 14:15:00 | 4985.00 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-08-14 09:15:00 | 4260.75 | 2024-08-16 11:15:00 | 4363.85 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-08-28 09:30:00 | 4418.00 | 2024-09-03 10:15:00 | 4419.95 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-08-29 10:15:00 | 4423.85 | 2024-09-03 10:15:00 | 4419.95 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-08-30 11:45:00 | 4394.50 | 2024-09-03 10:15:00 | 4419.95 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-09-09 09:15:00 | 4304.70 | 2024-09-09 09:15:00 | 4353.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-09 11:30:00 | 4316.50 | 2024-09-10 12:15:00 | 4367.90 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-09 13:30:00 | 4315.95 | 2024-09-10 12:15:00 | 4367.90 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-09 15:00:00 | 4320.60 | 2024-09-10 12:15:00 | 4367.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-20 15:00:00 | 5380.95 | 2024-09-26 09:15:00 | 5203.75 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-09-23 13:00:00 | 5239.25 | 2024-09-26 09:15:00 | 5203.75 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-09-23 15:00:00 | 5238.00 | 2024-09-26 09:15:00 | 5203.75 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-09-24 13:15:00 | 5238.60 | 2024-09-26 09:15:00 | 5203.75 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-10-14 15:00:00 | 5135.15 | 2024-10-17 11:15:00 | 5100.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-10-15 09:15:00 | 5137.45 | 2024-10-17 15:15:00 | 5100.05 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-10-15 10:30:00 | 5128.70 | 2024-10-17 15:15:00 | 5100.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-10-15 15:15:00 | 5138.00 | 2024-10-17 15:15:00 | 5100.05 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-16 09:15:00 | 5145.05 | 2024-10-17 15:15:00 | 5100.05 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-10-21 12:15:00 | 5191.50 | 2024-10-21 14:15:00 | 5116.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-21 14:30:00 | 5183.00 | 2024-10-21 15:15:00 | 5128.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-10-23 09:15:00 | 5030.05 | 2024-10-24 14:15:00 | 5188.75 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-10-24 09:15:00 | 5021.55 | 2024-10-24 14:15:00 | 5188.75 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-11-11 11:15:00 | 5264.80 | 2024-11-12 15:15:00 | 5100.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-11-11 14:45:00 | 5264.90 | 2024-11-12 15:15:00 | 5100.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-11-11 15:15:00 | 5244.90 | 2024-11-12 15:15:00 | 5100.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-11-18 11:15:00 | 4960.00 | 2024-11-19 09:15:00 | 4712.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-18 11:15:00 | 4960.00 | 2024-11-21 09:15:00 | 4464.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-05 13:45:00 | 4703.90 | 2024-12-06 09:15:00 | 4654.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-05 15:00:00 | 4713.40 | 2024-12-06 09:15:00 | 4654.80 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-12-13 13:45:00 | 4739.00 | 2024-12-18 09:15:00 | 4725.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-23 09:15:00 | 4599.10 | 2024-12-26 09:15:00 | 4369.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 09:15:00 | 4599.10 | 2024-12-26 10:15:00 | 4458.00 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2025-01-14 12:15:00 | 3726.85 | 2025-01-15 10:15:00 | 3824.35 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-01-14 15:00:00 | 3728.20 | 2025-01-15 10:15:00 | 3824.35 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-01-16 14:15:00 | 4016.95 | 2025-01-21 13:15:00 | 3901.90 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-01-17 09:45:00 | 3989.10 | 2025-01-21 13:15:00 | 3901.90 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-01-17 11:00:00 | 4001.95 | 2025-01-21 13:15:00 | 3901.90 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-01-20 14:30:00 | 3994.75 | 2025-01-21 13:15:00 | 3901.90 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-01-24 11:30:00 | 3736.65 | 2025-01-28 09:15:00 | 3549.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 11:30:00 | 3736.65 | 2025-01-28 12:15:00 | 3604.80 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-02-10 09:15:00 | 3349.60 | 2025-02-11 09:15:00 | 3182.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 3349.60 | 2025-02-12 10:15:00 | 3202.90 | STOP_HIT | 0.50 | 4.38% |
| BUY | retest2 | 2025-02-21 12:30:00 | 3315.40 | 2025-02-24 09:15:00 | 3240.15 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-02-21 15:00:00 | 3330.80 | 2025-02-24 09:15:00 | 3240.15 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-02-24 11:30:00 | 3309.45 | 2025-02-25 09:15:00 | 3224.10 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-02-24 12:15:00 | 3310.60 | 2025-02-25 09:15:00 | 3224.10 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-03-25 09:15:00 | 3523.10 | 2025-03-26 13:15:00 | 3461.25 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-03-26 12:15:00 | 3480.05 | 2025-03-26 13:15:00 | 3461.25 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-04-04 10:15:00 | 3455.10 | 2025-04-07 09:15:00 | 3109.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 3328.55 | 2025-04-07 09:15:00 | 3162.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 3328.55 | 2025-04-15 09:15:00 | 3157.00 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2025-04-24 09:15:00 | 3611.80 | 2025-04-25 15:15:00 | 3454.30 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-04-25 15:15:00 | 3454.30 | 2025-04-25 15:15:00 | 3454.30 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 3204.80 | 2025-05-12 09:15:00 | 3251.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-29 09:15:00 | 3440.90 | 2025-06-03 09:15:00 | 3449.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-06-13 10:15:00 | 3573.90 | 2025-06-13 13:15:00 | 3528.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-17 12:30:00 | 3491.50 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-06-18 09:45:00 | 3505.00 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-18 10:15:00 | 3505.20 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-18 10:45:00 | 3504.70 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-06-24 09:15:00 | 3456.50 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-06-25 09:30:00 | 3457.30 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-26 10:00:00 | 3462.10 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2025-06-26 12:00:00 | 3464.90 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-06-30 13:30:00 | 3403.70 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-01 10:00:00 | 3401.40 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-02 12:15:00 | 3404.30 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-07-02 13:15:00 | 3404.60 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-09 11:00:00 | 3435.80 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-09 12:15:00 | 3435.00 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-10 09:30:00 | 3434.00 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-15 10:00:00 | 3470.70 | 2025-07-17 09:15:00 | 3817.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 10:30:00 | 3471.50 | 2025-07-17 09:15:00 | 3818.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 11:15:00 | 3469.80 | 2025-07-17 09:15:00 | 3816.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 14:15:00 | 3471.30 | 2025-07-17 09:15:00 | 3818.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-24 09:15:00 | 3889.00 | 2025-07-24 10:15:00 | 3872.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-29 12:15:00 | 3763.00 | 2025-07-29 15:15:00 | 3794.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-31 11:30:00 | 3821.00 | 2025-08-01 15:15:00 | 3767.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-01 12:45:00 | 3836.00 | 2025-08-01 15:15:00 | 3767.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-08-22 12:00:00 | 3225.10 | 2025-08-22 13:15:00 | 3271.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-28 09:15:00 | 3359.80 | 2025-08-28 12:15:00 | 3232.40 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-09-08 09:30:00 | 3320.80 | 2025-09-12 09:15:00 | 3313.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-10 09:15:00 | 3437.00 | 2025-09-12 09:15:00 | 3313.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-09-16 13:00:00 | 3306.60 | 2025-09-17 10:15:00 | 3345.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-01 10:30:00 | 3163.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 13:45:00 | 3159.90 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-01 15:00:00 | 3163.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-03 10:45:00 | 3151.90 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-06 09:15:00 | 3170.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-06 14:30:00 | 3172.20 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-06 15:15:00 | 3162.50 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-24 09:15:00 | 3212.00 | 2025-10-29 11:15:00 | 3253.30 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-11-04 11:15:00 | 3250.10 | 2025-11-04 11:15:00 | 3258.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-06 14:30:00 | 3248.70 | 2025-11-07 09:15:00 | 3178.70 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-11-12 09:15:00 | 3055.70 | 2025-11-19 09:15:00 | 2902.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 09:15:00 | 3055.70 | 2025-11-20 10:15:00 | 2975.00 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2815.00 | 2025-12-12 09:15:00 | 2916.00 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-12-30 11:45:00 | 3010.50 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-12-31 12:00:00 | 3006.60 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2026-01-01 09:30:00 | 3011.80 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2026-01-13 14:15:00 | 2946.40 | 2026-01-14 15:15:00 | 2975.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest1 | 2026-01-14 09:15:00 | 2947.70 | 2026-01-14 15:15:00 | 2975.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2926.30 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2935.10 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-19 10:15:00 | 2931.80 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-19 11:15:00 | 2940.00 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-01-21 09:15:00 | 2860.40 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-01-22 10:15:00 | 2916.50 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-22 11:00:00 | 2913.00 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-22 15:00:00 | 2912.10 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2851.10 | 2026-01-30 13:15:00 | 2893.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-16 09:15:00 | 2864.50 | 2026-02-16 12:15:00 | 2908.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-16 10:00:00 | 2867.70 | 2026-02-16 12:15:00 | 2908.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3027.80 | 2026-02-27 10:15:00 | 3100.60 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2026-02-20 11:30:00 | 3026.20 | 2026-02-27 10:15:00 | 3100.60 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2026-03-05 13:45:00 | 3045.80 | 2026-03-05 14:15:00 | 3070.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-03-17 13:45:00 | 3260.20 | 2026-03-19 15:15:00 | 3210.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-18 10:00:00 | 3263.10 | 2026-03-19 15:15:00 | 3210.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 12:30:00 | 3273.30 | 2026-04-07 09:15:00 | 3259.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-04-06 10:30:00 | 3258.80 | 2026-04-07 09:15:00 | 3259.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-04-23 12:15:00 | 4066.80 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-27 10:00:00 | 4081.90 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-04-27 11:15:00 | 4081.60 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3962.30 | 2026-04-30 11:15:00 | 4020.20 | STOP_HIT | 1.00 | -1.46% |
