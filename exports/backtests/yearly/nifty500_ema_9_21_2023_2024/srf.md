# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2778.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 243 |
| ALERT1 | 158 |
| ALERT2 | 155 |
| ALERT2_SKIP | 103 |
| ALERT3 | 326 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 129 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 125 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 106
- **Target hits / Stop hits / Partials:** 6 / 125 / 2
- **Avg / median % per leg:** -0.30% / -1.06%
- **Sum % (uncompounded):** -40.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 21 | 36.2% | 6 | 52 | 0 | 0.67% | 38.9% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.58% | 9.2% |
| BUY @ 3rd Alert (retest2) | 56 | 20 | 35.7% | 5 | 51 | 0 | 0.53% | 29.8% |
| SELL (all) | 75 | 6 | 8.0% | 0 | 73 | 2 | -1.06% | -79.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 75 | 6 | 8.0% | 0 | 73 | 2 | -1.06% | -79.2% |
| retest1 (combined) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.58% | 9.2% |
| retest2 (combined) | 131 | 26 | 19.8% | 5 | 124 | 2 | -0.38% | -49.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 2478.00 | 2452.87 | 2451.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 2491.40 | 2460.58 | 2454.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 2470.00 | 2470.75 | 2461.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 15:15:00 | 2477.00 | 2472.00 | 2463.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 2477.00 | 2472.00 | 2463.01 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 2451.80 | 2463.39 | 2463.39 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 2474.90 | 2463.63 | 2462.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 2491.80 | 2470.44 | 2466.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 2514.90 | 2520.04 | 2506.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 2520.95 | 2520.77 | 2513.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 2520.95 | 2520.77 | 2513.00 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 10:15:00 | 2486.90 | 2507.77 | 2510.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 14:15:00 | 2478.05 | 2494.00 | 2502.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 09:15:00 | 2503.00 | 2493.56 | 2500.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 2503.00 | 2493.56 | 2500.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 2503.00 | 2493.56 | 2500.53 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 13:15:00 | 2513.55 | 2504.48 | 2504.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 2539.00 | 2513.29 | 2508.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 10:15:00 | 2517.00 | 2526.12 | 2519.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 10:15:00 | 2517.00 | 2526.12 | 2519.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 2517.00 | 2526.12 | 2519.79 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 2514.00 | 2536.72 | 2537.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 2507.45 | 2524.30 | 2530.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 2364.35 | 2351.67 | 2381.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 2378.15 | 2357.51 | 2369.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 2378.15 | 2357.51 | 2369.75 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 2390.60 | 2374.54 | 2373.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 11:15:00 | 2395.00 | 2378.63 | 2375.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 2391.65 | 2399.84 | 2392.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 2391.65 | 2399.84 | 2392.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 2391.65 | 2399.84 | 2392.95 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 2383.10 | 2392.73 | 2393.79 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 09:15:00 | 2404.00 | 2394.98 | 2394.72 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 2364.35 | 2389.19 | 2392.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 15:15:00 | 2357.00 | 2374.23 | 2383.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 2337.45 | 2327.65 | 2341.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 2337.45 | 2327.65 | 2341.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 2337.45 | 2327.65 | 2341.71 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 2284.90 | 2267.22 | 2266.54 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 2254.50 | 2264.68 | 2265.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 11:15:00 | 2228.05 | 2257.35 | 2262.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 13:15:00 | 2185.00 | 2174.26 | 2188.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 14:15:00 | 2210.00 | 2181.40 | 2190.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 2210.00 | 2181.40 | 2190.75 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 2216.10 | 2192.61 | 2190.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 14:15:00 | 2239.00 | 2213.40 | 2202.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 10:15:00 | 2222.30 | 2224.96 | 2217.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 2240.00 | 2243.65 | 2236.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 2240.00 | 2243.65 | 2236.34 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 2202.80 | 2230.89 | 2231.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 11:15:00 | 2185.00 | 2207.85 | 2218.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 13:15:00 | 2208.20 | 2206.45 | 2216.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 14:15:00 | 2149.75 | 2195.11 | 2210.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 2149.75 | 2195.11 | 2210.13 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 10:15:00 | 2168.00 | 2163.30 | 2162.98 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 2160.00 | 2162.64 | 2162.71 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 12:15:00 | 2164.00 | 2162.91 | 2162.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 2170.15 | 2164.77 | 2163.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 12:15:00 | 2167.00 | 2168.13 | 2166.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 12:15:00 | 2167.00 | 2168.13 | 2166.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 2167.00 | 2168.13 | 2166.12 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 2281.90 | 2291.40 | 2291.82 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 2296.25 | 2289.47 | 2288.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 2305.00 | 2292.58 | 2290.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 14:15:00 | 2295.65 | 2297.67 | 2294.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 15:15:00 | 2292.00 | 2296.54 | 2293.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 2292.00 | 2296.54 | 2293.96 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 2274.85 | 2292.20 | 2292.22 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 2297.10 | 2289.59 | 2288.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 2304.25 | 2292.52 | 2290.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 2302.05 | 2308.78 | 2304.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 15:15:00 | 2302.05 | 2308.78 | 2304.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 2302.05 | 2308.78 | 2304.28 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 09:15:00 | 2306.70 | 2310.43 | 2310.82 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 2349.30 | 2318.07 | 2313.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 10:15:00 | 2373.10 | 2329.08 | 2319.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 2370.30 | 2372.15 | 2356.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 14:15:00 | 2360.05 | 2367.60 | 2361.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 2360.05 | 2367.60 | 2361.46 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 13:15:00 | 2426.00 | 2444.58 | 2445.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 2406.05 | 2434.78 | 2440.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 2387.20 | 2385.87 | 2405.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 2395.00 | 2389.32 | 2402.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 2395.00 | 2389.32 | 2402.56 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 2259.00 | 2247.64 | 2246.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 2264.15 | 2253.13 | 2249.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 09:15:00 | 2253.00 | 2254.22 | 2250.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 10:15:00 | 2246.10 | 2252.60 | 2250.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 2246.10 | 2252.60 | 2250.41 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 14:15:00 | 2245.95 | 2248.86 | 2249.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 2217.60 | 2241.98 | 2245.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 2217.05 | 2215.27 | 2226.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 12:15:00 | 2222.90 | 2218.05 | 2226.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 2222.90 | 2218.05 | 2226.16 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2259.75 | 2230.55 | 2229.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 12:15:00 | 2285.65 | 2261.28 | 2256.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 2234.55 | 2262.38 | 2259.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 2234.55 | 2262.38 | 2259.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 2234.55 | 2262.38 | 2259.73 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 2232.00 | 2256.30 | 2257.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 2225.05 | 2243.94 | 2250.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 2255.05 | 2245.69 | 2249.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 2255.05 | 2245.69 | 2249.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 2255.05 | 2245.69 | 2249.85 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 2168.20 | 2148.66 | 2147.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 2178.85 | 2154.70 | 2150.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 15:15:00 | 2188.00 | 2188.99 | 2177.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 2176.95 | 2186.58 | 2177.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 2176.95 | 2186.58 | 2177.12 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 2334.65 | 2339.33 | 2339.40 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 2353.95 | 2341.38 | 2340.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 11:15:00 | 2360.00 | 2345.05 | 2342.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 12:15:00 | 2352.70 | 2355.08 | 2350.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 13:15:00 | 2352.90 | 2354.64 | 2350.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 2352.90 | 2354.64 | 2350.56 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 10:15:00 | 2332.45 | 2351.99 | 2353.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 13:15:00 | 2328.95 | 2341.79 | 2347.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 2339.30 | 2337.75 | 2344.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 2339.30 | 2337.75 | 2344.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 2339.30 | 2337.75 | 2344.27 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 2353.00 | 2346.79 | 2346.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 2359.00 | 2349.23 | 2347.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 10:15:00 | 2347.75 | 2348.94 | 2347.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 10:15:00 | 2347.75 | 2348.94 | 2347.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 2347.75 | 2348.94 | 2347.67 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 2336.75 | 2344.87 | 2345.94 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 2350.50 | 2344.50 | 2344.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 14:15:00 | 2359.50 | 2347.50 | 2345.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 2360.35 | 2362.46 | 2355.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 14:15:00 | 2360.35 | 2362.46 | 2355.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 2360.35 | 2362.46 | 2355.92 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 12:15:00 | 2346.10 | 2351.92 | 2352.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 2340.00 | 2349.54 | 2351.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 2350.10 | 2349.65 | 2351.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 2350.10 | 2349.65 | 2351.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 2350.10 | 2349.65 | 2351.27 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 2359.00 | 2353.02 | 2352.53 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 2348.60 | 2352.24 | 2352.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 2356.25 | 2353.04 | 2352.89 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 12:15:00 | 2346.60 | 2352.36 | 2352.64 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 2371.65 | 2356.14 | 2354.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 2373.45 | 2359.60 | 2356.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 13:15:00 | 2456.00 | 2457.49 | 2441.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 2446.50 | 2454.49 | 2443.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 2446.50 | 2454.49 | 2443.74 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 2414.00 | 2437.04 | 2440.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 14:15:00 | 2378.45 | 2416.50 | 2425.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 2405.45 | 2399.40 | 2411.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 14:15:00 | 2408.15 | 2401.15 | 2410.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 2408.15 | 2401.15 | 2410.86 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 2424.05 | 2413.65 | 2412.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 2471.05 | 2425.13 | 2418.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 2441.50 | 2444.12 | 2436.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 14:15:00 | 2441.50 | 2444.12 | 2436.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 2441.50 | 2444.12 | 2436.94 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 11:15:00 | 2421.05 | 2432.48 | 2433.10 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 2445.05 | 2433.65 | 2433.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 10:15:00 | 2463.20 | 2439.56 | 2435.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 2406.15 | 2443.83 | 2439.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 2406.15 | 2443.83 | 2439.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 2406.15 | 2443.83 | 2439.78 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 2370.00 | 2429.06 | 2433.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 2360.30 | 2415.31 | 2426.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 2413.95 | 2402.18 | 2411.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 2413.95 | 2402.18 | 2411.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 2413.95 | 2402.18 | 2411.57 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 2450.50 | 2417.41 | 2417.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 2456.60 | 2434.54 | 2425.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 2472.00 | 2478.69 | 2462.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 2470.20 | 2476.99 | 2463.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 2470.20 | 2476.99 | 2463.58 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 2462.20 | 2476.29 | 2476.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 2445.20 | 2468.27 | 2472.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 2483.80 | 2465.89 | 2469.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 13:15:00 | 2483.80 | 2465.89 | 2469.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 2483.80 | 2465.89 | 2469.87 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 15:15:00 | 2490.00 | 2474.40 | 2473.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 11:15:00 | 2515.00 | 2484.03 | 2478.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 10:15:00 | 2499.05 | 2500.99 | 2490.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 11:15:00 | 2501.70 | 2501.13 | 2491.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 2501.70 | 2501.13 | 2491.89 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 2478.60 | 2492.69 | 2493.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 2433.05 | 2477.65 | 2486.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 2315.85 | 2306.61 | 2342.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 2336.00 | 2321.18 | 2333.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 2336.00 | 2321.18 | 2333.15 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 13:15:00 | 2365.30 | 2338.86 | 2338.37 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 2325.50 | 2337.24 | 2337.86 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 14:15:00 | 2350.65 | 2339.94 | 2338.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 2356.20 | 2344.40 | 2341.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 2342.50 | 2352.21 | 2347.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 2342.50 | 2352.21 | 2347.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 2342.50 | 2352.21 | 2347.88 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 2322.30 | 2342.66 | 2344.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 2312.15 | 2334.48 | 2339.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 2336.35 | 2308.00 | 2317.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 2336.35 | 2308.00 | 2317.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2336.35 | 2308.00 | 2317.66 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 2294.20 | 2285.66 | 2285.20 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 13:15:00 | 2273.10 | 2282.90 | 2284.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 15:15:00 | 2268.95 | 2278.30 | 2281.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 2248.05 | 2240.67 | 2251.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 2248.05 | 2240.67 | 2251.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 2248.05 | 2240.67 | 2251.20 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 2317.90 | 2261.88 | 2259.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 13:15:00 | 2323.85 | 2281.90 | 2269.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 10:15:00 | 2296.20 | 2298.45 | 2282.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 2298.00 | 2298.36 | 2284.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 2298.00 | 2298.36 | 2284.14 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 12:15:00 | 2272.90 | 2296.78 | 2297.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 2253.00 | 2284.03 | 2291.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 2289.75 | 2278.88 | 2286.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 10:15:00 | 2289.75 | 2278.88 | 2286.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 2289.75 | 2278.88 | 2286.63 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 13:15:00 | 2298.25 | 2287.52 | 2286.31 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 2282.00 | 2289.07 | 2289.91 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 14:15:00 | 2298.80 | 2291.18 | 2290.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 15:15:00 | 2304.90 | 2293.92 | 2291.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 2293.30 | 2293.80 | 2292.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 2293.30 | 2293.80 | 2292.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 2293.30 | 2293.80 | 2292.09 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 2279.75 | 2289.52 | 2290.48 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 10:15:00 | 2315.35 | 2294.63 | 2292.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 14:15:00 | 2324.10 | 2306.33 | 2298.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 09:15:00 | 2310.00 | 2310.05 | 2302.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 2310.00 | 2310.05 | 2302.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 2310.00 | 2310.05 | 2302.08 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 09:15:00 | 2344.95 | 2354.34 | 2354.66 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 14:15:00 | 2385.00 | 2358.50 | 2355.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 15:15:00 | 2389.00 | 2364.60 | 2358.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 09:15:00 | 2378.85 | 2391.18 | 2380.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 2378.85 | 2391.18 | 2380.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 2378.85 | 2391.18 | 2380.18 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 2366.00 | 2389.77 | 2392.54 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 2400.90 | 2388.04 | 2386.46 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 2375.10 | 2389.19 | 2389.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 12:15:00 | 2370.15 | 2381.30 | 2385.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 2383.00 | 2381.64 | 2385.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 2375.20 | 2380.35 | 2384.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 2375.20 | 2380.35 | 2384.14 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 10:15:00 | 2391.95 | 2386.60 | 2386.24 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 2372.00 | 2384.58 | 2386.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 2369.55 | 2381.57 | 2384.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 12:15:00 | 2382.60 | 2381.78 | 2384.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 2404.55 | 2386.33 | 2386.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 2404.55 | 2386.33 | 2386.35 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 14:15:00 | 2411.00 | 2391.27 | 2388.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 09:15:00 | 2457.00 | 2407.41 | 2396.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 2437.85 | 2438.16 | 2421.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 10:15:00 | 2433.00 | 2441.25 | 2432.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 2433.00 | 2441.25 | 2432.03 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 2398.30 | 2422.19 | 2425.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 13:15:00 | 2392.45 | 2413.63 | 2420.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 2402.55 | 2401.90 | 2412.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 2417.75 | 2405.07 | 2413.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 2417.75 | 2405.07 | 2413.18 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 11:15:00 | 2418.95 | 2415.85 | 2415.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 12:15:00 | 2446.25 | 2421.93 | 2418.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 2422.00 | 2423.88 | 2420.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 2422.00 | 2423.88 | 2420.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 2422.00 | 2423.88 | 2420.51 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 2430.55 | 2435.20 | 2435.36 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 12:15:00 | 2445.55 | 2436.50 | 2435.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 13:15:00 | 2452.55 | 2439.71 | 2437.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 2568.55 | 2574.67 | 2550.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 2524.30 | 2562.84 | 2554.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 2524.30 | 2562.84 | 2554.07 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 2587.90 | 2609.61 | 2610.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 2575.00 | 2593.69 | 2600.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 14:15:00 | 2606.00 | 2585.90 | 2593.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 14:15:00 | 2606.00 | 2585.90 | 2593.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 2606.00 | 2585.90 | 2593.17 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 10:15:00 | 2614.30 | 2597.10 | 2597.06 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 2585.65 | 2595.34 | 2596.37 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 2666.80 | 2610.52 | 2602.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 2670.10 | 2622.44 | 2608.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 2637.75 | 2642.08 | 2627.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 12:30:00 | 2634.15 | 2642.08 | 2627.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 2631.80 | 2639.19 | 2629.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 2631.80 | 2639.19 | 2629.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 2629.95 | 2637.34 | 2629.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 2579.20 | 2637.34 | 2629.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 2598.05 | 2629.48 | 2626.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 2575.00 | 2629.48 | 2626.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 2594.80 | 2622.54 | 2623.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 12:15:00 | 2583.40 | 2611.96 | 2618.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 2551.05 | 2544.74 | 2568.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 10:00:00 | 2551.05 | 2544.74 | 2568.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2564.15 | 2548.63 | 2567.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:45:00 | 2561.00 | 2548.63 | 2567.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 2564.30 | 2551.76 | 2567.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 2562.40 | 2551.76 | 2567.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 2563.00 | 2554.01 | 2567.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 2559.80 | 2554.01 | 2567.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 2537.15 | 2548.90 | 2562.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 12:15:00 | 2567.05 | 2547.09 | 2546.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 2567.05 | 2547.09 | 2546.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 2574.20 | 2559.37 | 2553.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 2620.95 | 2626.79 | 2605.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 15:00:00 | 2620.95 | 2626.79 | 2605.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 2613.60 | 2622.27 | 2607.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:30:00 | 2611.00 | 2622.27 | 2607.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 2619.45 | 2621.70 | 2608.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 2616.70 | 2621.70 | 2608.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 2630.40 | 2635.59 | 2627.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 2663.10 | 2635.59 | 2627.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:00:00 | 2645.45 | 2637.56 | 2628.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:30:00 | 2646.00 | 2634.69 | 2631.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 11:15:00 | 2647.60 | 2635.15 | 2631.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 2622.00 | 2636.92 | 2634.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 2622.00 | 2636.92 | 2634.21 | SL hit (close<static) qty=1.00 sl=2626.95 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 2583.10 | 2637.36 | 2640.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 12:15:00 | 2571.65 | 2624.22 | 2634.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 11:15:00 | 2597.00 | 2594.52 | 2611.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 11:45:00 | 2589.00 | 2594.52 | 2611.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 2263.55 | 2247.43 | 2268.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 2263.55 | 2247.43 | 2268.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 2275.65 | 2253.07 | 2269.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 2275.65 | 2253.07 | 2269.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 2286.80 | 2259.82 | 2270.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 2297.00 | 2259.82 | 2270.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 2315.00 | 2279.05 | 2278.00 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 2268.95 | 2279.88 | 2280.89 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 12:15:00 | 2284.90 | 2276.97 | 2276.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 2290.20 | 2280.84 | 2278.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 2286.05 | 2290.24 | 2286.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 2286.05 | 2290.24 | 2286.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 2286.05 | 2290.24 | 2286.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 2286.05 | 2290.24 | 2286.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 2287.50 | 2289.69 | 2286.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 2278.00 | 2289.69 | 2286.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 2294.80 | 2290.71 | 2287.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 13:45:00 | 2301.15 | 2292.97 | 2289.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 2304.85 | 2300.28 | 2295.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 2298.20 | 2296.70 | 2294.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 2279.15 | 2295.60 | 2297.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 2279.15 | 2295.60 | 2297.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 2269.05 | 2284.02 | 2290.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 2225.35 | 2219.79 | 2232.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2225.35 | 2219.79 | 2232.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2225.35 | 2219.79 | 2232.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 2221.20 | 2223.06 | 2233.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 14:15:00 | 2266.20 | 2241.06 | 2239.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 2266.20 | 2241.06 | 2239.24 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2173.00 | 2229.24 | 2234.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2143.90 | 2212.17 | 2226.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 2276.25 | 2215.75 | 2220.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 2276.25 | 2215.75 | 2220.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2276.25 | 2215.75 | 2220.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 2276.25 | 2215.75 | 2220.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 2275.00 | 2227.60 | 2225.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 2287.70 | 2239.62 | 2231.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 12:15:00 | 2285.10 | 2286.51 | 2265.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 13:00:00 | 2285.10 | 2286.51 | 2265.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 2290.10 | 2291.15 | 2275.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:15:00 | 2302.00 | 2291.15 | 2275.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:45:00 | 2298.10 | 2292.72 | 2277.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 13:30:00 | 2299.50 | 2292.59 | 2280.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 2420.10 | 2448.51 | 2451.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 2420.10 | 2448.51 | 2451.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 2417.00 | 2442.21 | 2447.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 2442.45 | 2408.36 | 2413.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 2442.45 | 2408.36 | 2413.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 2442.45 | 2408.36 | 2413.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 2442.45 | 2408.36 | 2413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 2435.90 | 2413.87 | 2415.62 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 2441.20 | 2419.34 | 2417.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 2447.35 | 2424.94 | 2420.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 10:15:00 | 2433.45 | 2440.02 | 2431.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 2433.45 | 2440.02 | 2431.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 2433.45 | 2440.02 | 2431.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 2433.45 | 2440.02 | 2431.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 2416.30 | 2435.27 | 2429.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 2416.30 | 2435.27 | 2429.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 2433.50 | 2434.92 | 2430.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 13:15:00 | 2436.30 | 2434.92 | 2430.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:00:00 | 2439.00 | 2435.74 | 2430.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:45:00 | 2438.00 | 2433.38 | 2430.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:45:00 | 2437.00 | 2434.87 | 2431.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 2428.00 | 2435.59 | 2432.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 2428.00 | 2435.59 | 2432.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 2440.00 | 2436.47 | 2433.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 2441.45 | 2436.47 | 2433.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 09:45:00 | 2448.90 | 2446.47 | 2439.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 11:15:00 | 2382.50 | 2430.81 | 2433.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 2382.50 | 2430.81 | 2433.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 2380.70 | 2420.79 | 2428.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 11:15:00 | 2406.30 | 2391.06 | 2400.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 11:15:00 | 2406.30 | 2391.06 | 2400.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 2406.30 | 2391.06 | 2400.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 2406.30 | 2391.06 | 2400.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 2399.80 | 2392.81 | 2400.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 2389.30 | 2392.31 | 2399.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:30:00 | 2395.50 | 2395.88 | 2398.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 2393.00 | 2396.92 | 2398.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 2395.00 | 2397.74 | 2398.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 2395.00 | 2397.19 | 2398.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 2404.95 | 2397.19 | 2398.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2391.15 | 2395.98 | 2397.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 2393.45 | 2395.98 | 2397.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 2397.85 | 2396.36 | 2397.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:30:00 | 2413.50 | 2396.36 | 2397.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 2391.35 | 2395.36 | 2397.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 13:30:00 | 2373.60 | 2389.52 | 2394.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 10:45:00 | 2380.15 | 2381.35 | 2388.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 2386.45 | 2382.92 | 2388.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 2386.80 | 2387.27 | 2388.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 2374.40 | 2384.70 | 2387.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-10 15:15:00 | 2398.00 | 2388.65 | 2387.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 2398.00 | 2388.65 | 2387.96 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 2377.85 | 2386.00 | 2386.85 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 2393.00 | 2387.69 | 2387.43 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 10:15:00 | 2382.95 | 2387.46 | 2387.60 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 2390.00 | 2387.97 | 2387.82 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 10:15:00 | 2383.80 | 2387.62 | 2387.91 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 2394.40 | 2388.98 | 2388.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 13:15:00 | 2404.95 | 2395.62 | 2392.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 2374.50 | 2394.21 | 2393.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 2374.50 | 2394.21 | 2393.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 2374.50 | 2394.21 | 2393.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 2376.20 | 2394.21 | 2393.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 2359.15 | 2387.20 | 2390.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 2342.90 | 2369.09 | 2380.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 2360.00 | 2355.55 | 2367.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 2360.00 | 2355.55 | 2367.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 2360.00 | 2355.55 | 2367.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 2357.45 | 2355.55 | 2367.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2360.55 | 2356.55 | 2366.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 2367.50 | 2356.55 | 2366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 2367.35 | 2358.71 | 2367.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 2368.95 | 2358.71 | 2367.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 2361.10 | 2359.19 | 2366.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 2365.75 | 2359.19 | 2366.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2370.00 | 2361.35 | 2366.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:15:00 | 2381.60 | 2361.35 | 2366.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2387.70 | 2366.62 | 2368.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 2380.00 | 2366.62 | 2368.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 2345.05 | 2355.49 | 2362.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 2362.15 | 2355.49 | 2362.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2338.35 | 2351.66 | 2359.17 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 2384.55 | 2363.77 | 2361.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 10:15:00 | 2394.90 | 2370.00 | 2364.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 2613.20 | 2619.40 | 2577.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 11:00:00 | 2613.20 | 2619.40 | 2577.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 2586.35 | 2612.12 | 2592.07 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 2552.35 | 2579.03 | 2581.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 2524.05 | 2568.04 | 2575.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 2500.50 | 2489.43 | 2520.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:15:00 | 2499.05 | 2489.43 | 2520.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2540.00 | 2499.54 | 2522.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 2540.00 | 2499.54 | 2522.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 2525.50 | 2504.74 | 2522.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:00:00 | 2505.75 | 2507.88 | 2521.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 2501.25 | 2506.55 | 2519.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 2569.85 | 2526.27 | 2522.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 2569.85 | 2526.27 | 2522.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 13:15:00 | 2592.00 | 2539.42 | 2529.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 2566.20 | 2566.58 | 2550.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 2566.20 | 2566.58 | 2550.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 2536.00 | 2558.96 | 2549.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 2536.00 | 2558.96 | 2549.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 2547.00 | 2556.57 | 2549.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 2560.65 | 2556.57 | 2549.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:30:00 | 2581.95 | 2565.51 | 2560.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:15:00 | 2570.00 | 2559.55 | 2558.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 2551.80 | 2561.83 | 2560.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 2559.70 | 2561.40 | 2560.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 2532.25 | 2555.57 | 2557.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 2532.25 | 2555.57 | 2557.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 2520.80 | 2543.81 | 2551.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 2503.00 | 2502.68 | 2525.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 2503.00 | 2502.68 | 2525.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2493.10 | 2496.20 | 2514.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 2479.95 | 2489.64 | 2506.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:30:00 | 2479.95 | 2488.46 | 2504.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 2480.00 | 2487.14 | 2502.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 2476.00 | 2487.14 | 2502.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 2485.10 | 2476.94 | 2490.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 2485.10 | 2476.94 | 2490.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 2465.00 | 2473.89 | 2484.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 2557.90 | 2490.14 | 2482.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 2557.90 | 2490.14 | 2482.37 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 2490.45 | 2498.01 | 2498.07 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 2515.00 | 2500.93 | 2499.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 2522.85 | 2509.84 | 2504.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 15:15:00 | 2547.00 | 2552.00 | 2535.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:15:00 | 2538.55 | 2552.00 | 2535.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2523.50 | 2546.30 | 2534.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 2521.40 | 2546.30 | 2534.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 2544.40 | 2545.92 | 2535.58 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 2515.05 | 2532.89 | 2533.92 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 2549.00 | 2533.94 | 2533.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 2564.65 | 2540.08 | 2536.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 13:15:00 | 2548.40 | 2548.92 | 2542.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 14:00:00 | 2548.40 | 2548.92 | 2542.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 2572.40 | 2553.62 | 2545.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 2572.40 | 2553.62 | 2545.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2594.65 | 2593.93 | 2584.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 2606.35 | 2592.29 | 2585.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:45:00 | 2603.65 | 2594.86 | 2587.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 2536.15 | 2589.16 | 2594.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 2536.15 | 2589.16 | 2594.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 2519.50 | 2568.43 | 2583.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 2531.00 | 2517.11 | 2539.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 2531.00 | 2517.11 | 2539.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2552.55 | 2526.73 | 2540.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 2550.65 | 2526.73 | 2540.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 2551.20 | 2531.62 | 2541.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 2555.60 | 2531.62 | 2541.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 2546.15 | 2534.98 | 2541.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 2546.15 | 2534.98 | 2541.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 2556.20 | 2539.22 | 2542.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 2556.20 | 2539.22 | 2542.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 2516.00 | 2534.77 | 2539.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:00:00 | 2493.00 | 2521.70 | 2531.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 2368.35 | 2392.19 | 2414.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 2403.10 | 2389.16 | 2405.19 | SL hit (close>ema200) qty=0.50 sl=2389.16 alert=retest2 |

### Cycle 111 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 2428.60 | 2413.92 | 2411.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 2440.00 | 2425.13 | 2419.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 2431.05 | 2437.06 | 2429.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 2431.05 | 2437.06 | 2429.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 2431.05 | 2437.06 | 2429.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 2431.50 | 2437.06 | 2429.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 2430.00 | 2435.65 | 2429.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 2437.65 | 2435.65 | 2429.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2425.45 | 2433.61 | 2429.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 2425.45 | 2433.61 | 2429.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 2434.80 | 2433.85 | 2430.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:30:00 | 2438.95 | 2434.63 | 2431.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 2447.65 | 2434.63 | 2431.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:15:00 | 2444.55 | 2446.26 | 2440.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:00:00 | 2439.45 | 2475.46 | 2472.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 2475.05 | 2475.22 | 2472.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 2481.10 | 2475.22 | 2472.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2480.15 | 2476.20 | 2473.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 2454.65 | 2471.44 | 2471.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 2454.65 | 2471.44 | 2471.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 2447.75 | 2466.71 | 2469.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2332.80 | 2326.91 | 2363.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2332.80 | 2326.91 | 2363.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2342.30 | 2331.54 | 2348.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 2337.55 | 2331.54 | 2348.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2345.45 | 2334.32 | 2347.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 2352.00 | 2334.32 | 2347.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 2344.05 | 2336.27 | 2347.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 2330.55 | 2337.59 | 2345.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:15:00 | 2335.30 | 2338.30 | 2343.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 13:15:00 | 2335.30 | 2338.12 | 2342.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 2355.40 | 2345.94 | 2344.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 2355.40 | 2345.94 | 2344.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 2363.45 | 2349.44 | 2346.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 13:15:00 | 2345.25 | 2348.61 | 2346.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 13:15:00 | 2345.25 | 2348.61 | 2346.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 2345.25 | 2348.61 | 2346.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 2345.25 | 2348.61 | 2346.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 2348.45 | 2348.57 | 2346.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:15:00 | 2351.45 | 2348.57 | 2346.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 2327.15 | 2344.75 | 2345.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 2327.15 | 2344.75 | 2345.32 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 2352.15 | 2343.57 | 2342.73 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 2308.15 | 2337.03 | 2340.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 2272.05 | 2310.22 | 2324.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 15:15:00 | 2282.00 | 2276.58 | 2298.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:30:00 | 2282.20 | 2275.31 | 2295.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 2297.50 | 2279.75 | 2295.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 2297.50 | 2279.75 | 2295.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 2313.20 | 2286.44 | 2297.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 2313.20 | 2286.44 | 2297.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 2313.95 | 2291.94 | 2298.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 2320.20 | 2291.94 | 2298.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 2325.25 | 2304.15 | 2303.51 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 2281.95 | 2301.87 | 2302.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 2258.00 | 2293.09 | 2298.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 13:15:00 | 2292.00 | 2291.03 | 2296.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 13:15:00 | 2292.00 | 2291.03 | 2296.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 2292.00 | 2291.03 | 2296.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:45:00 | 2299.00 | 2291.03 | 2296.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2275.65 | 2287.95 | 2294.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 2255.30 | 2280.15 | 2290.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 2296.75 | 2254.52 | 2262.48 | SL hit (close>static) qty=1.00 sl=2296.70 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 13:15:00 | 2312.75 | 2275.51 | 2271.21 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 2237.80 | 2266.17 | 2267.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2222.25 | 2250.34 | 2259.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 2256.05 | 2247.75 | 2255.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 14:15:00 | 2256.05 | 2247.75 | 2255.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 2256.05 | 2247.75 | 2255.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 2264.65 | 2247.75 | 2255.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 2258.00 | 2249.80 | 2255.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:00:00 | 2232.00 | 2246.24 | 2253.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 10:30:00 | 2247.25 | 2233.69 | 2239.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 2247.00 | 2236.95 | 2240.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 2248.30 | 2239.63 | 2241.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 2259.30 | 2243.56 | 2243.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 2259.30 | 2243.56 | 2243.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 2268.70 | 2259.07 | 2253.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 2247.40 | 2257.94 | 2254.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 2247.40 | 2257.94 | 2254.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2247.40 | 2257.94 | 2254.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 2247.65 | 2257.94 | 2254.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2239.00 | 2254.15 | 2253.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 2239.00 | 2254.15 | 2253.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 2244.20 | 2252.16 | 2252.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2231.35 | 2248.00 | 2250.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 2245.55 | 2244.41 | 2248.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 2245.55 | 2244.41 | 2248.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 2240.40 | 2243.61 | 2247.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 2250.40 | 2244.97 | 2247.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 2253.50 | 2246.68 | 2248.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 2226.65 | 2246.68 | 2248.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 2242.40 | 2243.93 | 2246.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:00:00 | 2244.05 | 2243.93 | 2246.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 2244.55 | 2244.52 | 2246.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 2249.40 | 2245.50 | 2246.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 2249.40 | 2245.50 | 2246.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 2248.25 | 2246.05 | 2246.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 2256.40 | 2246.05 | 2246.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2251.00 | 2247.04 | 2247.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 2254.60 | 2248.55 | 2247.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 2254.60 | 2248.55 | 2247.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 2277.25 | 2254.15 | 2250.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 2358.10 | 2362.47 | 2338.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 11:00:00 | 2358.10 | 2362.47 | 2338.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 2345.60 | 2359.10 | 2339.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 2345.60 | 2359.10 | 2339.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 2337.05 | 2354.69 | 2339.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 2337.05 | 2354.69 | 2339.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 2320.30 | 2347.81 | 2337.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 2320.30 | 2347.81 | 2337.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 2315.25 | 2341.30 | 2335.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 2315.25 | 2341.30 | 2335.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 2263.85 | 2318.24 | 2325.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 2250.40 | 2275.06 | 2291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2230.15 | 2222.22 | 2247.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 2230.15 | 2222.22 | 2247.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 2234.60 | 2227.33 | 2241.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 2227.85 | 2230.76 | 2240.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 2221.75 | 2181.99 | 2181.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 2221.75 | 2181.99 | 2181.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 2234.55 | 2215.02 | 2200.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 2272.00 | 2275.60 | 2256.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 2272.00 | 2275.60 | 2256.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 2261.85 | 2271.15 | 2257.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:30:00 | 2257.80 | 2271.15 | 2257.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 2264.90 | 2269.90 | 2258.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 2262.25 | 2269.90 | 2258.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2270.10 | 2269.94 | 2259.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 2283.90 | 2265.44 | 2261.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 12:15:00 | 2298.35 | 2311.41 | 2313.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 2298.35 | 2311.41 | 2313.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 2294.10 | 2306.05 | 2310.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 13:15:00 | 2294.40 | 2289.13 | 2298.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 13:15:00 | 2294.40 | 2289.13 | 2298.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 2294.40 | 2289.13 | 2298.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 2295.00 | 2289.13 | 2298.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 2300.00 | 2291.30 | 2298.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:30:00 | 2295.80 | 2291.30 | 2298.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 2318.10 | 2296.66 | 2300.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 2343.80 | 2296.66 | 2300.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 2348.60 | 2307.05 | 2304.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 2368.60 | 2344.05 | 2328.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 2336.80 | 2345.14 | 2334.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 15:00:00 | 2336.80 | 2345.14 | 2334.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 2340.90 | 2344.30 | 2334.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 2330.35 | 2344.30 | 2334.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2318.20 | 2339.08 | 2333.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 2318.20 | 2339.08 | 2333.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2302.65 | 2331.79 | 2330.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 2302.95 | 2331.79 | 2330.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 2307.40 | 2326.91 | 2328.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 2298.50 | 2316.31 | 2322.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 2292.40 | 2290.09 | 2303.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 2292.40 | 2290.09 | 2303.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 2298.35 | 2291.74 | 2302.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 2297.00 | 2291.74 | 2302.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2301.20 | 2293.63 | 2302.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 2288.90 | 2293.63 | 2302.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 2295.05 | 2294.97 | 2302.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 2295.30 | 2302.35 | 2303.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 2293.25 | 2288.36 | 2294.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2299.85 | 2290.65 | 2295.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:45:00 | 2299.90 | 2290.65 | 2295.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2291.50 | 2290.82 | 2294.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:30:00 | 2283.00 | 2289.88 | 2294.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 13:30:00 | 2283.20 | 2288.70 | 2293.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 2284.00 | 2288.70 | 2293.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 2326.15 | 2291.21 | 2287.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 2326.15 | 2291.21 | 2287.47 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 2282.80 | 2290.75 | 2291.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 2264.50 | 2278.78 | 2283.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 15:15:00 | 2266.60 | 2265.66 | 2273.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 09:15:00 | 2268.95 | 2265.66 | 2273.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2276.05 | 2267.74 | 2273.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 2276.05 | 2267.74 | 2273.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 2270.75 | 2268.34 | 2273.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 2275.05 | 2268.34 | 2273.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 2264.65 | 2264.55 | 2269.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 2264.00 | 2264.55 | 2269.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 2261.15 | 2262.98 | 2268.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 2227.15 | 2255.33 | 2263.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2235.10 | 2248.31 | 2257.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 2226.50 | 2234.02 | 2244.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 11:15:00 | 2272.70 | 2236.36 | 2231.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 2272.70 | 2236.36 | 2231.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 12:15:00 | 2286.80 | 2246.45 | 2236.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 12:15:00 | 2271.95 | 2272.95 | 2258.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:15:00 | 2317.00 | 2272.66 | 2261.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-09 09:15:00 | 2548.70 | 2400.08 | 2345.75 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 132 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 2506.40 | 2530.25 | 2530.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 2494.20 | 2511.20 | 2519.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 2541.95 | 2510.61 | 2515.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 2541.95 | 2510.61 | 2515.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 2541.95 | 2510.61 | 2515.44 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 2554.70 | 2519.43 | 2519.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 2563.35 | 2528.21 | 2523.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 2587.00 | 2595.21 | 2575.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:45:00 | 2587.85 | 2595.21 | 2575.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2593.20 | 2603.18 | 2591.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 2593.20 | 2603.18 | 2591.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2589.85 | 2600.51 | 2591.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:00:00 | 2589.85 | 2600.51 | 2591.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 2596.65 | 2599.74 | 2591.55 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2532.50 | 2579.17 | 2584.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 2522.00 | 2567.74 | 2578.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2594.65 | 2543.17 | 2556.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 2594.65 | 2543.17 | 2556.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2594.65 | 2543.17 | 2556.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2594.65 | 2543.17 | 2556.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2602.85 | 2555.10 | 2560.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 2608.75 | 2555.10 | 2560.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 2590.45 | 2565.07 | 2564.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 2598.65 | 2575.62 | 2569.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2569.10 | 2576.35 | 2571.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2569.10 | 2576.35 | 2571.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2569.10 | 2576.35 | 2571.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2569.10 | 2576.35 | 2571.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2576.90 | 2576.46 | 2571.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 14:45:00 | 2583.95 | 2575.63 | 2572.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 2585.50 | 2576.91 | 2573.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 2606.10 | 2585.15 | 2581.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 2591.65 | 2588.09 | 2587.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 2607.60 | 2591.99 | 2589.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 2642.45 | 2601.13 | 2593.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-30 14:15:00 | 2842.35 | 2731.97 | 2674.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 2878.30 | 2913.87 | 2917.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 2875.00 | 2900.27 | 2910.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 2834.75 | 2811.18 | 2834.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 2834.75 | 2811.18 | 2834.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 2834.75 | 2811.18 | 2834.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 2834.75 | 2811.18 | 2834.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 2852.15 | 2819.37 | 2835.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 2849.25 | 2819.37 | 2835.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 2861.00 | 2827.70 | 2838.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 2861.00 | 2827.70 | 2838.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2847.00 | 2832.50 | 2836.96 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 2869.30 | 2843.34 | 2841.33 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 2783.80 | 2831.20 | 2836.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 2764.45 | 2817.85 | 2830.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 2750.00 | 2744.98 | 2770.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 2750.00 | 2744.98 | 2770.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2731.50 | 2743.09 | 2764.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 2722.40 | 2739.43 | 2749.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 2721.15 | 2738.31 | 2748.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 2722.90 | 2734.92 | 2744.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 2769.20 | 2745.94 | 2747.75 | SL hit (close>static) qty=1.00 sl=2765.45 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 2785.15 | 2755.68 | 2751.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 2798.20 | 2764.18 | 2756.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2757.75 | 2769.07 | 2760.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2757.75 | 2769.07 | 2760.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2757.75 | 2769.07 | 2760.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2757.75 | 2769.07 | 2760.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2768.60 | 2768.97 | 2760.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:45:00 | 2784.90 | 2769.72 | 2762.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:15:00 | 2783.55 | 2768.98 | 2762.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 2783.20 | 2771.55 | 2765.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 2750.95 | 2768.42 | 2765.63 | SL hit (close<static) qty=1.00 sl=2754.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 2751.70 | 2762.13 | 2763.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 2729.05 | 2754.13 | 2759.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 2770.55 | 2749.65 | 2754.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 13:15:00 | 2770.55 | 2749.65 | 2754.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 2770.55 | 2749.65 | 2754.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 2770.55 | 2749.65 | 2754.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 2767.95 | 2753.31 | 2755.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:30:00 | 2775.10 | 2753.31 | 2755.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 2850.00 | 2774.29 | 2764.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 10:15:00 | 2858.95 | 2791.23 | 2773.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 2785.90 | 2819.13 | 2799.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 2785.90 | 2819.13 | 2799.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 2785.90 | 2819.13 | 2799.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:30:00 | 2780.00 | 2819.13 | 2799.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 2800.00 | 2815.31 | 2799.22 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 2759.15 | 2790.43 | 2792.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 2737.65 | 2779.87 | 2787.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 2807.10 | 2780.67 | 2785.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 2807.10 | 2780.67 | 2785.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 2807.10 | 2780.67 | 2785.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 2807.10 | 2780.67 | 2785.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 2808.00 | 2786.14 | 2787.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 2810.95 | 2786.14 | 2787.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2826.25 | 2794.16 | 2791.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 2836.70 | 2807.89 | 2798.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 2848.80 | 2851.50 | 2835.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 12:30:00 | 2846.60 | 2851.50 | 2835.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 2835.15 | 2846.51 | 2838.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 2850.60 | 2846.51 | 2838.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 2868.00 | 2850.81 | 2840.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 12:15:00 | 2869.00 | 2854.18 | 2843.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:30:00 | 2896.55 | 2870.72 | 2855.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 13:15:00 | 2888.00 | 2893.41 | 2893.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 2888.00 | 2893.41 | 2893.92 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 2900.80 | 2894.89 | 2894.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 2944.65 | 2904.84 | 2899.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 11:15:00 | 2929.45 | 2933.77 | 2922.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 12:00:00 | 2929.45 | 2933.77 | 2922.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 2928.70 | 2932.76 | 2922.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:45:00 | 2930.00 | 2932.76 | 2922.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 2976.55 | 2990.88 | 2972.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 2976.55 | 2990.88 | 2972.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 2999.75 | 3002.75 | 2991.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 3018.15 | 3002.75 | 2991.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:00:00 | 3010.25 | 3003.03 | 2993.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:30:00 | 3006.00 | 3002.43 | 2994.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:15:00 | 3004.00 | 3001.10 | 2995.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 2994.00 | 2999.68 | 2995.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 2994.00 | 2999.68 | 2995.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 2990.00 | 2997.75 | 2994.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:15:00 | 2982.50 | 2997.75 | 2994.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2992.30 | 2996.66 | 2994.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 3007.85 | 2996.66 | 2994.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 3034.60 | 3004.25 | 2998.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-24 15:15:00 | 2991.80 | 3004.20 | 3005.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 15:15:00 | 2991.80 | 3004.20 | 3005.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 2945.00 | 2992.36 | 3000.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 2940.85 | 2916.39 | 2934.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 2940.85 | 2916.39 | 2934.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 2940.85 | 2916.39 | 2934.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 2940.85 | 2916.39 | 2934.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 2947.95 | 2922.70 | 2936.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 2947.00 | 2922.70 | 2936.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2957.00 | 2929.56 | 2938.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2978.85 | 2929.56 | 2938.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 2943.65 | 2942.81 | 2942.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 2916.35 | 2941.28 | 2942.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 2933.90 | 2904.06 | 2912.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 2961.70 | 2921.96 | 2919.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 2961.70 | 2921.96 | 2919.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 2981.80 | 2933.92 | 2925.13 | Break + close above crossover candle high |

### Cycle 148 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 2823.35 | 2918.17 | 2919.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 2680.15 | 2829.17 | 2863.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2747.00 | 2738.03 | 2788.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2747.00 | 2738.03 | 2788.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2747.00 | 2738.03 | 2788.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 2715.00 | 2750.95 | 2772.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2850.00 | 2763.67 | 2765.94 | SL hit (close>static) qty=1.00 sl=2806.25 alert=retest2 |

### Cycle 149 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2888.75 | 2788.69 | 2777.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 2913.50 | 2813.65 | 2789.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 2994.80 | 2995.04 | 2953.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:45:00 | 2990.00 | 2995.04 | 2953.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 2980.00 | 2988.25 | 2978.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:15:00 | 2980.50 | 2988.25 | 2978.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 3001.00 | 2990.80 | 2980.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:15:00 | 3007.90 | 2990.80 | 2980.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 2986.50 | 3004.38 | 3004.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 2986.50 | 3004.38 | 3004.71 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 3022.90 | 3007.49 | 3005.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 12:15:00 | 3048.40 | 3015.68 | 3009.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 2996.80 | 3020.75 | 3014.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 2996.80 | 3020.75 | 3014.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2996.80 | 3020.75 | 3014.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2996.80 | 3020.75 | 3014.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 2970.00 | 3010.60 | 3010.89 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 3041.00 | 3003.81 | 3003.39 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 3013.90 | 3017.23 | 3017.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 3000.00 | 3013.79 | 3016.06 | Break + close below crossover candle low |

### Cycle 155 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 3041.80 | 3019.39 | 3018.40 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 2991.50 | 3014.55 | 3016.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 2975.30 | 3006.70 | 3012.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 3025.00 | 2999.43 | 3006.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 3025.00 | 2999.43 | 3006.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 3025.00 | 2999.43 | 3006.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 3023.30 | 2999.43 | 3006.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 3024.90 | 3004.53 | 3007.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 3021.30 | 3004.53 | 3007.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 3003.30 | 3007.30 | 3008.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 3003.30 | 3007.30 | 3008.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2978.40 | 2954.66 | 2972.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 2990.00 | 2954.66 | 2972.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 3000.00 | 2963.73 | 2975.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 3000.00 | 2963.73 | 2975.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 3039.40 | 2978.86 | 2980.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 3039.40 | 2978.86 | 2980.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 3073.80 | 2997.85 | 2989.42 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 2934.20 | 2996.66 | 3003.35 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3022.20 | 3005.94 | 3004.60 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 2955.00 | 2999.95 | 3003.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 2912.20 | 2964.56 | 2985.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 2922.50 | 2889.23 | 2904.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2908.00 | 2892.98 | 2904.65 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 2975.10 | 2921.56 | 2914.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 3019.20 | 2941.09 | 2924.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 2982.80 | 2983.80 | 2961.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:45:00 | 2976.00 | 2983.80 | 2961.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2952.10 | 2975.27 | 2961.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 2945.90 | 2975.27 | 2961.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2950.30 | 2970.28 | 2960.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2957.00 | 2970.28 | 2960.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2959.80 | 2972.63 | 2964.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 2959.80 | 2972.63 | 2964.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2967.20 | 2971.54 | 2964.63 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 2912.30 | 2953.82 | 2958.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 15:15:00 | 2901.00 | 2918.19 | 2935.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 2904.90 | 2918.43 | 2933.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 2904.90 | 2915.12 | 2929.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 2898.50 | 2911.07 | 2925.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 2897.00 | 2904.67 | 2918.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 2914.40 | 2908.43 | 2917.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:30:00 | 2906.30 | 2907.84 | 2916.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 2909.00 | 2894.85 | 2899.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 2904.50 | 2896.78 | 2900.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:30:00 | 2908.80 | 2898.43 | 2900.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2897.50 | 2900.41 | 2901.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 2890.00 | 2900.41 | 2901.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:00:00 | 2893.50 | 2890.94 | 2893.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 2944.30 | 2891.64 | 2885.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 2944.30 | 2891.64 | 2885.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2987.90 | 2949.87 | 2923.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 3122.00 | 3126.16 | 3103.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:15:00 | 3136.30 | 3126.16 | 3103.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 3109.70 | 3130.02 | 3111.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 11:15:00 | 3109.70 | 3130.02 | 3111.64 | SL hit (close<ema400) qty=1.00 sl=3111.64 alert=retest1 |

### Cycle 164 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 3068.60 | 3100.74 | 3103.01 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 3120.00 | 3100.55 | 3098.31 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 11:15:00 | 3089.00 | 3098.67 | 3099.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 12:15:00 | 3081.00 | 3095.14 | 3097.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 3101.40 | 3096.39 | 3098.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3096.90 | 3096.49 | 3098.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:15:00 | 3104.70 | 3096.49 | 3098.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3104.70 | 3098.13 | 3098.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 3127.00 | 3098.13 | 3098.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3097.70 | 3098.05 | 3098.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 3114.60 | 3098.05 | 3098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3102.70 | 3098.98 | 3098.96 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 3088.90 | 3098.93 | 3099.46 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 3114.60 | 3102.06 | 3100.84 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 3088.70 | 3099.39 | 3099.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 3075.10 | 3089.57 | 3094.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 3077.70 | 3045.72 | 3043.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 3091.40 | 3060.12 | 3052.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 3069.60 | 3073.77 | 3064.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3190.50 | 3197.55 | 3183.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 3255.60 | 3208.84 | 3189.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 3222.50 | 3232.67 | 3231.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 3252.80 | 3262.15 | 3262.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 3252.80 | 3262.15 | 3262.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 3244.00 | 3258.52 | 3260.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 3227.30 | 3206.14 | 3205.07 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 3196.00 | 3209.94 | 3210.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3185.80 | 3201.87 | 3205.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 3135.30 | 3130.19 | 3155.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 3135.30 | 3130.19 | 3155.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3117.20 | 3130.51 | 3147.92 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 3206.40 | 3158.47 | 3153.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 3229.70 | 3172.71 | 3160.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 3178.20 | 3179.38 | 3166.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 3161.70 | 3176.58 | 3167.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 3161.70 | 3176.58 | 3167.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 3144.30 | 3170.12 | 3165.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 3144.30 | 3170.12 | 3165.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 3141.00 | 3164.30 | 3162.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 3062.40 | 3164.30 | 3162.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 3082.20 | 3147.88 | 3155.56 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 3109.00 | 3095.05 | 3094.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 3137.50 | 3103.54 | 3098.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 3100.60 | 3107.40 | 3101.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 3094.20 | 3104.76 | 3101.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 3086.20 | 3104.76 | 3101.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 3084.80 | 3100.77 | 3099.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 3084.80 | 3100.77 | 3099.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 3086.20 | 3097.86 | 3098.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3020.10 | 3082.31 | 3091.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 3069.60 | 3066.96 | 3080.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:45:00 | 3073.40 | 3066.96 | 3080.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3049.50 | 3018.46 | 3038.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 3049.50 | 3018.46 | 3038.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3058.40 | 3026.44 | 3040.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 3058.40 | 3026.44 | 3040.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 3077.30 | 3052.63 | 3049.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 3087.50 | 3059.60 | 3052.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 3049.90 | 3058.49 | 3053.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 3050.30 | 3056.86 | 3053.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 3050.30 | 3056.86 | 3053.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 2974.50 | 3036.49 | 3044.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 2919.90 | 2974.18 | 3003.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 2872.00 | 2868.98 | 2903.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:45:00 | 2873.70 | 2868.98 | 2903.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2871.80 | 2859.14 | 2873.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 2842.60 | 2854.60 | 2864.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 2843.90 | 2850.41 | 2860.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 2845.00 | 2847.80 | 2857.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2911.50 | 2864.02 | 2862.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2911.50 | 2864.02 | 2862.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 2931.00 | 2892.41 | 2876.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 12:15:00 | 2924.10 | 2926.93 | 2904.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 2924.10 | 2926.93 | 2904.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2916.50 | 2925.49 | 2910.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 2916.50 | 2925.49 | 2910.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2936.70 | 2927.73 | 2913.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 2941.40 | 2927.73 | 2913.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:00:00 | 2940.00 | 2928.95 | 2916.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 2953.70 | 2924.63 | 2921.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 2944.00 | 2924.63 | 2921.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 2915.00 | 2930.05 | 2926.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 2915.00 | 2930.05 | 2926.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 2910.00 | 2926.04 | 2925.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 2867.00 | 2926.04 | 2925.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 2901.30 | 2921.09 | 2922.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2901.30 | 2921.09 | 2922.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 2853.30 | 2896.84 | 2909.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 2857.50 | 2852.25 | 2870.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 2857.50 | 2852.25 | 2870.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2863.20 | 2854.44 | 2869.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 2864.70 | 2854.44 | 2869.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2864.50 | 2852.04 | 2862.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 2863.20 | 2852.04 | 2862.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2877.80 | 2857.19 | 2863.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 2872.30 | 2857.19 | 2863.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 2872.30 | 2860.21 | 2864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:15:00 | 2878.70 | 2860.21 | 2864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 2888.00 | 2869.45 | 2868.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 2899.60 | 2875.48 | 2870.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 2925.00 | 2931.98 | 2917.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:15:00 | 2918.30 | 2931.98 | 2917.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2914.60 | 2928.50 | 2917.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 2914.60 | 2928.50 | 2917.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 2899.60 | 2922.72 | 2915.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 2899.60 | 2922.72 | 2915.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 2885.70 | 2915.32 | 2913.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 2885.70 | 2915.32 | 2913.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 2887.80 | 2909.81 | 2910.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2867.30 | 2901.31 | 2906.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2876.30 | 2867.13 | 2882.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 2887.50 | 2872.64 | 2882.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 2887.50 | 2872.64 | 2882.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 2887.50 | 2872.64 | 2882.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 2891.40 | 2876.39 | 2883.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 2891.40 | 2876.39 | 2883.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2896.60 | 2880.43 | 2884.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 2896.60 | 2880.43 | 2884.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 2885.00 | 2882.06 | 2884.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 2890.50 | 2882.06 | 2884.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2897.60 | 2885.17 | 2885.66 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 2893.00 | 2887.11 | 2886.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 2916.30 | 2892.95 | 2889.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 2964.00 | 2965.38 | 2944.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 2964.00 | 2965.38 | 2944.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2970.50 | 2976.50 | 2964.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 2965.00 | 2976.50 | 2964.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2973.50 | 2976.44 | 2966.81 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 2938.70 | 2960.12 | 2963.00 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 2980.00 | 2960.56 | 2959.15 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 2949.70 | 2959.33 | 2959.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 2932.90 | 2951.11 | 2955.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 2939.50 | 2935.57 | 2944.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:45:00 | 2936.10 | 2935.57 | 2944.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2953.00 | 2938.97 | 2944.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:15:00 | 2955.00 | 2938.97 | 2944.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 2950.50 | 2941.27 | 2945.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 2954.00 | 2941.27 | 2945.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2943.10 | 2940.68 | 2943.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 2943.10 | 2940.68 | 2943.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 2939.50 | 2940.45 | 2943.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:15:00 | 2947.30 | 2940.45 | 2943.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 2947.30 | 2941.82 | 2943.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 2946.60 | 2941.82 | 2943.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 2938.90 | 2941.23 | 2943.41 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 2951.20 | 2944.92 | 2944.81 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 2928.90 | 2943.36 | 2944.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 2916.00 | 2937.89 | 2941.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 2907.80 | 2907.28 | 2921.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 2907.80 | 2907.28 | 2921.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2911.10 | 2906.36 | 2915.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 2904.50 | 2906.49 | 2914.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 2903.30 | 2908.03 | 2914.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 2896.30 | 2908.82 | 2913.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 2903.80 | 2907.94 | 2913.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2917.00 | 2909.75 | 2913.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 2917.00 | 2909.75 | 2913.43 | SL hit (close>static) qty=1.00 sl=2916.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 2864.40 | 2832.32 | 2830.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 2879.80 | 2841.82 | 2835.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 2970.00 | 2972.61 | 2953.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 2970.00 | 2972.61 | 2953.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3022.00 | 3011.18 | 2995.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3003.90 | 3011.18 | 2995.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 3170.00 | 3182.74 | 3168.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3173.30 | 3182.74 | 3168.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3144.00 | 3174.99 | 3166.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 3144.00 | 3174.99 | 3166.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3146.10 | 3169.21 | 3164.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 3146.10 | 3169.21 | 3164.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 3123.10 | 3159.99 | 3160.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3116.10 | 3151.21 | 3156.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3078.20 | 3074.45 | 3101.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 3078.20 | 3074.45 | 3101.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3024.20 | 3065.61 | 3092.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 3002.50 | 3019.93 | 3033.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 2934.70 | 2906.42 | 2904.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 2934.70 | 2906.42 | 2904.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2941.80 | 2921.66 | 2913.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 2925.60 | 2930.38 | 2921.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 2925.60 | 2930.38 | 2921.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 2905.70 | 2924.57 | 2920.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 2905.70 | 2924.57 | 2920.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 2903.70 | 2920.39 | 2918.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 2903.70 | 2920.39 | 2918.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 2889.00 | 2914.11 | 2916.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2857.20 | 2902.73 | 2910.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 2831.90 | 2831.79 | 2845.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 2831.90 | 2831.79 | 2845.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2812.00 | 2800.02 | 2816.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 2806.40 | 2800.02 | 2816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 2836.70 | 2807.35 | 2817.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 2838.50 | 2807.35 | 2817.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 2841.00 | 2814.08 | 2820.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 2846.00 | 2814.08 | 2820.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2870.70 | 2832.86 | 2828.01 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 2823.60 | 2832.68 | 2833.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2805.20 | 2827.18 | 2830.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2812.80 | 2803.49 | 2811.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2811.00 | 2804.99 | 2811.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 2806.70 | 2806.64 | 2811.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 2854.30 | 2817.18 | 2814.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2854.30 | 2817.18 | 2814.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 2919.80 | 2850.13 | 2833.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 2897.70 | 2904.41 | 2876.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 2898.20 | 2904.41 | 2876.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2887.00 | 2905.99 | 2891.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 2890.20 | 2905.99 | 2891.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2887.50 | 2902.29 | 2890.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 2884.30 | 2902.29 | 2890.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2877.80 | 2897.39 | 2889.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 2877.80 | 2897.39 | 2889.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 2855.80 | 2880.36 | 2883.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 2840.90 | 2867.20 | 2875.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 2846.80 | 2838.65 | 2853.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 2846.80 | 2838.65 | 2853.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2840.20 | 2839.61 | 2848.12 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2878.70 | 2856.71 | 2854.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 2887.90 | 2867.62 | 2859.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 2842.90 | 2862.68 | 2858.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2846.00 | 2859.34 | 2857.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 2846.00 | 2859.34 | 2857.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 2839.90 | 2855.45 | 2855.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 2822.80 | 2848.92 | 2852.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 2838.80 | 2838.04 | 2845.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:15:00 | 2863.00 | 2838.04 | 2845.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2888.30 | 2848.09 | 2849.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 2888.30 | 2848.09 | 2849.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2886.40 | 2855.75 | 2852.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 2895.70 | 2868.90 | 2859.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 2992.60 | 2997.05 | 2975.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 2992.60 | 2997.05 | 2975.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3011.30 | 3011.01 | 2998.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:45:00 | 3016.90 | 3012.36 | 3001.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:00:00 | 3020.50 | 3013.99 | 3002.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 3016.50 | 3014.49 | 3004.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 3020.00 | 3018.57 | 3007.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3104.90 | 3092.74 | 3076.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 3087.00 | 3103.36 | 3103.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3087.00 | 3103.36 | 3103.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 3080.30 | 3094.58 | 3099.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 3083.10 | 3047.67 | 3066.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3071.20 | 3052.38 | 3066.94 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 3077.10 | 3064.46 | 3064.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 3085.50 | 3068.67 | 3066.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 3052.60 | 3074.41 | 3071.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 2976.30 | 3054.78 | 3062.44 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3080.50 | 3060.30 | 3059.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 3090.40 | 3071.43 | 3065.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 3058.80 | 3075.66 | 3069.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 3046.80 | 3069.89 | 3067.41 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3048.50 | 3065.61 | 3065.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3041.30 | 3056.25 | 3060.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 3039.90 | 3031.34 | 3041.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3011.30 | 3027.33 | 3039.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 3010.80 | 3027.33 | 3039.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 3051.30 | 3037.79 | 3039.01 | SL hit (close>static) qty=1.00 sl=3045.10 alert=retest2 |

### Cycle 207 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 3055.90 | 3039.78 | 3039.40 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 3033.90 | 3038.95 | 3039.49 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 3059.50 | 3041.08 | 3040.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 3065.00 | 3048.68 | 3044.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 3023.20 | 3043.75 | 3042.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 3027.50 | 3040.50 | 3041.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2994.00 | 3029.10 | 3035.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 2733.20 | 2730.75 | 2792.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:00:00 | 2733.20 | 2730.75 | 2792.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2737.30 | 2697.27 | 2717.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 2738.00 | 2697.27 | 2717.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 2772.90 | 2712.39 | 2722.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 2772.90 | 2712.39 | 2722.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 2773.50 | 2733.70 | 2731.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 2802.10 | 2753.57 | 2741.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2801.00 | 2801.34 | 2779.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2801.00 | 2801.34 | 2779.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2800.20 | 2805.20 | 2789.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 2799.60 | 2805.20 | 2789.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2815.00 | 2808.03 | 2794.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 2847.40 | 2821.63 | 2801.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 2783.00 | 2818.39 | 2804.15 | SL hit (close<static) qty=1.00 sl=2792.30 alert=retest2 |

### Cycle 212 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2717.00 | 2790.02 | 2793.22 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2929.60 | 2808.92 | 2794.61 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2928.80 | 2940.16 | 2941.66 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2948.70 | 2942.80 | 2942.66 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 2933.70 | 2940.98 | 2941.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2905.90 | 2933.96 | 2938.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2838.10 | 2831.95 | 2853.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2838.10 | 2831.95 | 2853.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2847.50 | 2838.14 | 2851.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 2847.50 | 2838.14 | 2851.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2845.00 | 2839.51 | 2850.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 2837.40 | 2839.51 | 2850.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 2867.50 | 2845.11 | 2852.35 | SL hit (close>static) qty=1.00 sl=2851.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 2876.00 | 2856.86 | 2856.34 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 2768.50 | 2839.19 | 2848.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 2751.50 | 2821.65 | 2839.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 2701.60 | 2696.79 | 2731.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 2671.50 | 2689.08 | 2710.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2671.50 | 2689.08 | 2710.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 2667.30 | 2684.26 | 2706.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2533.93 | 2570.99 | 2590.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 12:15:00 | 2533.10 | 2530.62 | 2551.66 | SL hit (close>ema200) qty=0.50 sl=2530.62 alert=retest2 |

### Cycle 219 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 2564.90 | 2552.83 | 2552.78 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 14:15:00 | 2550.80 | 2552.42 | 2552.60 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2571.00 | 2556.14 | 2554.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2618.20 | 2568.55 | 2560.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 2562.50 | 2600.59 | 2586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2550.20 | 2590.51 | 2583.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 2544.10 | 2590.51 | 2583.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2546.80 | 2574.02 | 2576.35 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 2582.10 | 2574.41 | 2574.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 2598.10 | 2581.07 | 2577.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 2567.10 | 2583.23 | 2579.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 2543.60 | 2575.30 | 2576.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 2526.50 | 2560.77 | 2569.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 2573.90 | 2534.85 | 2549.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 2584.40 | 2544.76 | 2552.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 2584.40 | 2544.76 | 2552.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 2610.30 | 2557.87 | 2557.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 2625.60 | 2571.42 | 2563.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 2543.00 | 2580.50 | 2571.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2524.30 | 2569.26 | 2567.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2523.10 | 2569.26 | 2567.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 2524.20 | 2560.25 | 2563.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2512.30 | 2550.66 | 2558.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 2479.70 | 2472.82 | 2499.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:00:00 | 2479.70 | 2472.82 | 2499.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 2489.20 | 2478.84 | 2497.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 2497.70 | 2478.84 | 2497.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 2504.30 | 2483.94 | 2498.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 2504.30 | 2483.94 | 2498.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 2499.10 | 2486.97 | 2498.42 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2564.90 | 2504.64 | 2504.59 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 2474.50 | 2518.85 | 2520.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 2466.40 | 2491.47 | 2505.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 2393.00 | 2404.73 | 2434.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 2479.80 | 2442.69 | 2442.90 | SL hit (close>static) qty=1.00 sl=2470.00 alert=retest2 |

### Cycle 229 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2538.40 | 2461.83 | 2451.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 2568.50 | 2510.56 | 2478.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2524.80 | 2538.24 | 2504.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 2537.00 | 2538.24 | 2504.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2492.30 | 2523.59 | 2505.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 2492.30 | 2523.59 | 2505.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2512.00 | 2521.27 | 2506.41 | EMA400 retest candle locked (from upside) |

### Cycle 230 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2443.30 | 2498.21 | 2498.94 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2530.70 | 2485.75 | 2485.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 2572.00 | 2503.00 | 2493.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 2412.00 | 2511.80 | 2505.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 2404.00 | 2490.24 | 2496.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 2374.90 | 2429.81 | 2459.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 2439.10 | 2408.89 | 2434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 2435.40 | 2414.20 | 2434.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 2408.20 | 2410.76 | 2431.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2477.90 | 2409.55 | 2416.60 | SL hit (close>static) qty=1.00 sl=2442.90 alert=retest2 |

### Cycle 233 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 2474.70 | 2422.58 | 2421.89 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 2399.40 | 2421.53 | 2424.51 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 11:15:00 | 2447.40 | 2427.34 | 2426.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 12:15:00 | 2461.70 | 2434.21 | 2429.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2444.70 | 2446.69 | 2438.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 2475.20 | 2485.94 | 2486.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 2475.20 | 2485.94 | 2486.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 2460.10 | 2480.77 | 2484.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 2498.10 | 2478.39 | 2481.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 2488.00 | 2480.31 | 2482.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 11:15:00 | 2482.70 | 2480.31 | 2482.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 2496.90 | 2479.67 | 2480.45 | SL hit (close>static) qty=1.00 sl=2495.70 alert=retest2 |

### Cycle 237 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 2500.80 | 2483.90 | 2482.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 2511.40 | 2494.46 | 2488.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 2500.80 | 2523.07 | 2509.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2483.80 | 2515.22 | 2506.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 2483.80 | 2515.22 | 2506.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 2484.20 | 2509.01 | 2504.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 2476.00 | 2509.01 | 2504.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 238 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 2489.60 | 2499.93 | 2501.09 | EMA200 below EMA400 |

### Cycle 239 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 2522.50 | 2501.88 | 2501.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 2595.50 | 2533.64 | 2519.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 2543.30 | 2551.59 | 2536.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:45:00 | 2543.10 | 2551.59 | 2536.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 2496.60 | 2540.01 | 2533.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 2496.60 | 2540.01 | 2533.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 2497.00 | 2531.41 | 2530.12 | EMA400 retest candle locked (from upside) |

### Cycle 240 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2505.20 | 2526.17 | 2527.86 | EMA200 below EMA400 |

### Cycle 241 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 2555.50 | 2530.19 | 2527.99 | EMA200 above EMA400 |

### Cycle 242 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 2518.30 | 2531.48 | 2532.69 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 2638.00 | 2553.45 | 2542.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 2694.00 | 2581.56 | 2556.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 13:15:00 | 2559.80 | 2024-04-22 12:15:00 | 2567.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-04-18 14:30:00 | 2537.15 | 2024-04-22 12:15:00 | 2567.05 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-04-29 09:15:00 | 2663.10 | 2024-04-30 14:15:00 | 2622.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-04-29 10:00:00 | 2645.45 | 2024-04-30 14:15:00 | 2622.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-30 09:30:00 | 2646.00 | 2024-04-30 14:15:00 | 2622.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-04-30 11:15:00 | 2647.60 | 2024-04-30 14:15:00 | 2622.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-05-02 09:15:00 | 2644.25 | 2024-05-03 11:15:00 | 2583.10 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-05-23 13:45:00 | 2301.15 | 2024-05-28 12:15:00 | 2279.15 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-05-24 12:30:00 | 2304.85 | 2024-05-28 12:15:00 | 2279.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-05-27 09:15:00 | 2298.20 | 2024-05-28 12:15:00 | 2279.15 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-06-03 10:30:00 | 2221.20 | 2024-06-03 14:15:00 | 2266.20 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-06-07 10:15:00 | 2302.00 | 2024-06-24 12:15:00 | 2420.10 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2024-06-07 10:45:00 | 2298.10 | 2024-06-24 12:15:00 | 2420.10 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2024-06-07 13:30:00 | 2299.50 | 2024-06-24 12:15:00 | 2420.10 | STOP_HIT | 1.00 | 5.24% |
| BUY | retest2 | 2024-06-28 13:15:00 | 2436.30 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-06-28 14:00:00 | 2439.00 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-07-01 09:45:00 | 2438.00 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-07-01 10:45:00 | 2437.00 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-07-01 14:15:00 | 2441.45 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-02 09:45:00 | 2448.90 | 2024-07-02 11:15:00 | 2382.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-07-04 13:30:00 | 2389.30 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-07-05 11:30:00 | 2395.50 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-07-05 14:00:00 | 2393.00 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-07-05 15:15:00 | 2395.00 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-07-08 13:30:00 | 2373.60 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-07-09 10:45:00 | 2380.15 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-07-09 12:15:00 | 2386.45 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-07-10 09:15:00 | 2386.80 | 2024-07-10 15:15:00 | 2398.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-08-06 13:00:00 | 2505.75 | 2024-08-07 12:15:00 | 2569.85 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-08-06 14:00:00 | 2501.25 | 2024-08-07 12:15:00 | 2569.85 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-08-09 09:15:00 | 2560.65 | 2024-08-13 11:15:00 | 2532.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-08-12 09:30:00 | 2581.95 | 2024-08-13 11:15:00 | 2532.25 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-08-12 12:15:00 | 2570.00 | 2024-08-13 11:15:00 | 2532.25 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-08-13 09:30:00 | 2551.80 | 2024-08-13 11:15:00 | 2532.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-08-16 13:00:00 | 2479.95 | 2024-08-22 09:15:00 | 2557.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-08-16 13:30:00 | 2479.95 | 2024-08-22 09:15:00 | 2557.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-08-16 14:30:00 | 2480.00 | 2024-08-22 09:15:00 | 2557.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-08-16 15:15:00 | 2476.00 | 2024-08-22 09:15:00 | 2557.90 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-09-04 12:30:00 | 2606.35 | 2024-09-06 10:15:00 | 2536.15 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-09-04 13:45:00 | 2603.65 | 2024-09-06 10:15:00 | 2536.15 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-09-11 14:00:00 | 2493.00 | 2024-09-19 10:15:00 | 2368.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 14:00:00 | 2493.00 | 2024-09-19 14:15:00 | 2403.10 | STOP_HIT | 0.50 | 3.61% |
| BUY | retest2 | 2024-09-25 12:30:00 | 2438.95 | 2024-10-03 11:15:00 | 2454.65 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-09-25 13:15:00 | 2447.65 | 2024-10-03 11:15:00 | 2454.65 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-09-26 13:15:00 | 2444.55 | 2024-10-03 11:15:00 | 2454.65 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-10-01 11:00:00 | 2439.45 | 2024-10-03 11:15:00 | 2454.65 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-10-09 15:00:00 | 2330.55 | 2024-10-11 11:15:00 | 2355.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-10-10 11:15:00 | 2335.30 | 2024-10-11 11:15:00 | 2355.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-10-10 13:15:00 | 2335.30 | 2024-10-11 11:15:00 | 2355.40 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-10-11 15:15:00 | 2351.45 | 2024-10-14 09:15:00 | 2327.15 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-10-22 10:00:00 | 2255.30 | 2024-10-23 11:15:00 | 2296.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-10-25 10:00:00 | 2232.00 | 2024-10-28 13:15:00 | 2259.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-10-28 10:30:00 | 2247.25 | 2024-10-28 13:15:00 | 2259.30 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-28 11:30:00 | 2247.00 | 2024-10-28 13:15:00 | 2259.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-28 12:30:00 | 2248.30 | 2024-10-28 13:15:00 | 2259.30 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-11-04 09:15:00 | 2226.65 | 2024-11-05 10:15:00 | 2254.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-11-04 11:30:00 | 2242.40 | 2024-11-05 10:15:00 | 2254.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-11-04 12:00:00 | 2244.05 | 2024-11-05 10:15:00 | 2254.60 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-11-04 14:00:00 | 2244.55 | 2024-11-05 10:15:00 | 2254.60 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-11-18 09:15:00 | 2227.85 | 2024-11-25 10:15:00 | 2221.75 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-12-02 09:15:00 | 2283.90 | 2024-12-06 12:15:00 | 2298.35 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2024-12-16 10:15:00 | 2288.90 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-12-16 11:15:00 | 2295.05 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-17 12:00:00 | 2295.30 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-12-18 10:00:00 | 2293.25 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-12-18 12:30:00 | 2283.00 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-12-18 13:30:00 | 2283.20 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-18 14:00:00 | 2284.00 | 2024-12-20 10:15:00 | 2326.15 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-12-30 13:15:00 | 2227.15 | 2025-01-03 11:15:00 | 2272.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-12-31 09:15:00 | 2235.10 | 2025-01-03 11:15:00 | 2272.70 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-12-31 15:15:00 | 2226.50 | 2025-01-03 11:15:00 | 2272.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest1 | 2025-01-07 09:15:00 | 2317.00 | 2025-01-09 09:15:00 | 2548.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-24 14:45:00 | 2583.95 | 2025-01-30 14:15:00 | 2842.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-27 09:15:00 | 2585.50 | 2025-01-30 14:15:00 | 2844.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-28 09:15:00 | 2606.10 | 2025-01-31 09:15:00 | 2866.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-29 11:15:00 | 2591.65 | 2025-01-31 09:15:00 | 2850.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-29 12:45:00 | 2642.45 | 2025-02-03 09:15:00 | 2906.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-19 13:00:00 | 2722.40 | 2025-02-20 11:15:00 | 2769.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-02-19 14:15:00 | 2721.15 | 2025-02-20 11:15:00 | 2769.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-02-20 09:15:00 | 2722.90 | 2025-02-20 11:15:00 | 2769.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-02-21 12:45:00 | 2784.90 | 2025-02-24 11:15:00 | 2750.95 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-02-21 14:15:00 | 2783.55 | 2025-02-24 11:15:00 | 2750.95 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-02-24 09:15:00 | 2783.20 | 2025-02-24 11:15:00 | 2750.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-03-06 12:15:00 | 2869.00 | 2025-03-11 13:15:00 | 2888.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-03-07 09:30:00 | 2896.55 | 2025-03-11 13:15:00 | 2888.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-03-20 09:15:00 | 3018.15 | 2025-03-24 15:15:00 | 2991.80 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-20 11:00:00 | 3010.25 | 2025-03-24 15:15:00 | 2991.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-03-20 11:30:00 | 3006.00 | 2025-03-24 15:15:00 | 2991.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-03-20 14:15:00 | 3004.00 | 2025-03-24 15:15:00 | 2991.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-04-01 09:15:00 | 2916.35 | 2025-04-02 13:15:00 | 2961.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-04-02 12:15:00 | 2933.90 | 2025-04-02 13:15:00 | 2961.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-04-09 10:15:00 | 2715.00 | 2025-04-11 09:15:00 | 2850.00 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2025-04-21 14:15:00 | 3007.90 | 2025-04-23 12:15:00 | 2986.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-05-23 10:15:00 | 2904.90 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-23 12:15:00 | 2904.90 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-23 14:15:00 | 2898.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-26 09:30:00 | 2897.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-05-26 12:30:00 | 2906.30 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-28 11:15:00 | 2909.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-28 12:00:00 | 2904.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-28 12:30:00 | 2908.80 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-28 15:15:00 | 2890.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-05-30 10:00:00 | 2893.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest1 | 2025-06-10 09:15:00 | 3136.30 | 2025-06-10 11:15:00 | 3109.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-02 10:45:00 | 3255.60 | 2025-07-10 15:15:00 | 3252.80 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-07-07 09:45:00 | 3222.50 | 2025-07-10 15:15:00 | 3252.80 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-08-14 09:45:00 | 2842.60 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-14 12:00:00 | 2843.90 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-14 13:30:00 | 2845.00 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-08-20 11:15:00 | 2941.40 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-20 13:00:00 | 2940.00 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-25 09:30:00 | 2953.70 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-25 10:15:00 | 2944.00 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-24 13:15:00 | 2904.50 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-24 14:30:00 | 2903.30 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-09-25 09:15:00 | 2896.30 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-25 09:45:00 | 2903.80 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2855.90 | 2025-10-01 12:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-30 09:30:00 | 3002.50 | 2025-11-11 15:15:00 | 2934.70 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest2 | 2025-11-26 13:15:00 | 2806.70 | 2025-11-27 09:15:00 | 2854.30 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-17 11:45:00 | 3016.90 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-12-17 13:00:00 | 3020.50 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-17 14:00:00 | 3016.50 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-12-17 14:30:00 | 3020.00 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2026-01-12 11:15:00 | 3010.80 | 2026-01-13 11:15:00 | 3051.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-01 10:45:00 | 2847.40 | 2026-02-01 12:15:00 | 2783.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-02-17 09:15:00 | 2837.40 | 2026-02-17 09:15:00 | 2867.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-23 10:45:00 | 2667.30 | 2026-03-02 09:15:00 | 2533.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:45:00 | 2667.30 | 2026-03-04 12:15:00 | 2533.10 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2026-03-24 10:45:00 | 2393.00 | 2026-03-24 15:15:00 | 2479.80 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-04-07 09:30:00 | 2408.20 | 2026-04-08 09:15:00 | 2477.90 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2444.70 | 2026-04-20 12:15:00 | 2475.20 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2026-04-21 11:15:00 | 2482.70 | 2026-04-22 10:15:00 | 2496.90 | STOP_HIT | 1.00 | -0.57% |
