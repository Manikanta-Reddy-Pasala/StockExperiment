# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-02-27 15:25:00 (33571 bars)
- **Last close:** 1578.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 44 |
| TARGET_HIT | 21 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 75
- **Target hits / Stop hits / Partials:** 21 / 75 / 44
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 28.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 34 | 46.6% | 12 | 39 | 22 | 0.19% | 13.8% |
| BUY @ 2nd Alert (retest1) | 73 | 34 | 46.6% | 12 | 39 | 22 | 0.19% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 67 | 31 | 46.3% | 9 | 36 | 22 | 0.22% | 14.7% |
| SELL @ 2nd Alert (retest1) | 67 | 31 | 46.3% | 9 | 36 | 22 | 0.22% | 14.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 140 | 65 | 46.4% | 21 | 75 | 44 | 0.20% | 28.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 2458.15 | 2462.96 | 0.00 | ORB-short ORB[2464.25,2495.00] vol=4.3x ATR=7.29 |
| Stop hit — per-position SL triggered | 2024-05-14 10:45:00 | 2465.44 | 2461.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 2448.00 | 2464.54 | 0.00 | ORB-short ORB[2452.20,2485.00] vol=1.7x ATR=8.09 |
| Stop hit — per-position SL triggered | 2024-05-15 10:15:00 | 2456.09 | 2463.41 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 2429.55 | 2435.66 | 0.00 | ORB-short ORB[2431.05,2451.75] vol=1.9x ATR=7.30 |
| Stop hit — per-position SL triggered | 2024-05-16 09:50:00 | 2436.85 | 2435.30 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:10:00 | 2482.90 | 2467.38 | 0.00 | ORB-long ORB[2454.05,2478.80] vol=2.1x ATR=7.21 |
| Stop hit — per-position SL triggered | 2024-05-17 11:25:00 | 2475.69 | 2468.32 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:50:00 | 2359.95 | 2345.31 | 0.00 | ORB-long ORB[2320.45,2352.00] vol=1.7x ATR=6.70 |
| Stop hit — per-position SL triggered | 2024-05-24 11:40:00 | 2353.25 | 2348.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 2327.20 | 2342.98 | 0.00 | ORB-short ORB[2330.55,2361.40] vol=2.5x ATR=7.00 |
| Stop hit — per-position SL triggered | 2024-05-28 09:35:00 | 2334.20 | 2340.47 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:10:00 | 2170.05 | 2177.69 | 0.00 | ORB-short ORB[2175.65,2205.25] vol=3.4x ATR=5.98 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 2176.03 | 2177.23 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:30:00 | 2255.00 | 2242.86 | 0.00 | ORB-long ORB[2225.00,2248.00] vol=2.7x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 2266.36 | 2250.26 | 0.00 | T1 1.5R @ 2266.36 |
| Target hit | 2024-06-06 15:20:00 | 2280.00 | 2268.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 11:00:00 | 2323.70 | 2314.93 | 0.00 | ORB-long ORB[2300.20,2318.15] vol=3.2x ATR=5.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:15:00 | 2332.15 | 2318.21 | 0.00 | T1 1.5R @ 2332.15 |
| Stop hit — per-position SL triggered | 2024-06-11 12:00:00 | 2323.70 | 2321.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:25:00 | 2313.75 | 2319.91 | 0.00 | ORB-short ORB[2316.00,2334.20] vol=1.6x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-06-12 10:50:00 | 2319.34 | 2318.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:55:00 | 2376.85 | 2364.93 | 0.00 | ORB-long ORB[2352.60,2374.75] vol=1.8x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:00:00 | 2386.53 | 2369.74 | 0.00 | T1 1.5R @ 2386.53 |
| Target hit | 2024-06-13 11:20:00 | 2380.05 | 2380.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:35:00 | 2508.20 | 2495.67 | 0.00 | ORB-long ORB[2469.00,2504.50] vol=1.8x ATR=10.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:40:00 | 2523.73 | 2499.68 | 0.00 | T1 1.5R @ 2523.73 |
| Target hit | 2024-06-20 11:55:00 | 2531.65 | 2532.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 2500.00 | 2509.05 | 0.00 | ORB-short ORB[2506.00,2527.85] vol=1.7x ATR=7.53 |
| Stop hit — per-position SL triggered | 2024-06-25 09:40:00 | 2507.53 | 2508.73 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 2539.10 | 2525.21 | 0.00 | ORB-long ORB[2510.00,2530.00] vol=3.3x ATR=9.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:50:00 | 2552.88 | 2536.68 | 0.00 | T1 1.5R @ 2552.88 |
| Target hit | 2024-07-01 15:20:00 | 2617.65 | 2580.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:15:00 | 2653.00 | 2686.66 | 0.00 | ORB-short ORB[2690.00,2720.00] vol=1.6x ATR=9.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:30:00 | 2638.86 | 2683.17 | 0.00 | T1 1.5R @ 2638.86 |
| Stop hit — per-position SL triggered | 2024-07-03 11:40:00 | 2653.00 | 2682.27 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 2700.40 | 2690.05 | 0.00 | ORB-long ORB[2675.00,2692.70] vol=3.1x ATR=8.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:40:00 | 2713.03 | 2699.11 | 0.00 | T1 1.5R @ 2713.03 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 2700.40 | 2702.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:25:00 | 2713.35 | 2697.10 | 0.00 | ORB-long ORB[2684.20,2709.55] vol=3.2x ATR=8.97 |
| Stop hit — per-position SL triggered | 2024-07-08 10:30:00 | 2704.38 | 2697.67 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 2729.85 | 2716.90 | 0.00 | ORB-long ORB[2694.95,2720.00] vol=4.7x ATR=9.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:55:00 | 2743.41 | 2728.82 | 0.00 | T1 1.5R @ 2743.41 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 2729.85 | 2733.61 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 2755.00 | 2729.29 | 0.00 | ORB-long ORB[2715.50,2747.00] vol=4.3x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-07-11 11:10:00 | 2747.27 | 2732.15 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:25:00 | 2795.45 | 2772.63 | 0.00 | ORB-long ORB[2757.35,2783.85] vol=1.5x ATR=9.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:40:00 | 2810.21 | 2786.09 | 0.00 | T1 1.5R @ 2810.21 |
| Target hit | 2024-07-15 15:10:00 | 2797.20 | 2805.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-07-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 10:20:00 | 2844.05 | 2823.09 | 0.00 | ORB-long ORB[2801.30,2827.75] vol=2.3x ATR=8.56 |
| Stop hit — per-position SL triggered | 2024-07-19 10:30:00 | 2835.49 | 2826.31 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 2915.35 | 2903.88 | 0.00 | ORB-long ORB[2871.05,2902.95] vol=3.3x ATR=9.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 12:45:00 | 2929.45 | 2915.13 | 0.00 | T1 1.5R @ 2929.45 |
| Target hit | 2024-07-25 15:20:00 | 2957.50 | 2923.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 3143.90 | 3116.60 | 0.00 | ORB-long ORB[3083.05,3126.00] vol=2.9x ATR=11.92 |
| Stop hit — per-position SL triggered | 2024-08-01 09:45:00 | 3131.98 | 3125.68 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:10:00 | 3100.00 | 3069.80 | 0.00 | ORB-long ORB[3043.35,3083.80] vol=2.4x ATR=11.35 |
| Stop hit — per-position SL triggered | 2024-08-02 11:50:00 | 3088.65 | 3075.12 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:40:00 | 3138.30 | 3113.53 | 0.00 | ORB-long ORB[3080.00,3115.45] vol=3.5x ATR=13.50 |
| Stop hit — per-position SL triggered | 2024-08-09 09:55:00 | 3124.80 | 3120.60 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:50:00 | 3024.90 | 3052.90 | 0.00 | ORB-short ORB[3041.65,3069.55] vol=1.5x ATR=12.46 |
| Stop hit — per-position SL triggered | 2024-08-12 10:05:00 | 3037.36 | 3044.17 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 2847.10 | 2860.99 | 0.00 | ORB-short ORB[2859.95,2884.75] vol=3.7x ATR=8.76 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 2855.86 | 2858.95 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:40:00 | 2875.95 | 2876.12 | 0.00 | ORB-short ORB[2878.75,2902.95] vol=2.6x ATR=7.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:50:00 | 2865.26 | 2875.78 | 0.00 | T1 1.5R @ 2865.26 |
| Target hit | 2024-08-20 12:30:00 | 2871.95 | 2870.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 2841.10 | 2856.02 | 0.00 | ORB-short ORB[2853.30,2886.95] vol=2.1x ATR=9.44 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 2850.54 | 2855.05 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:50:00 | 2916.10 | 2898.41 | 0.00 | ORB-long ORB[2867.50,2901.95] vol=3.1x ATR=8.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:00:00 | 2929.26 | 2906.83 | 0.00 | T1 1.5R @ 2929.26 |
| Stop hit — per-position SL triggered | 2024-08-29 10:05:00 | 2916.10 | 2907.31 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 2943.70 | 2925.38 | 0.00 | ORB-long ORB[2905.00,2930.00] vol=4.4x ATR=8.04 |
| Stop hit — per-position SL triggered | 2024-09-03 09:35:00 | 2935.66 | 2927.60 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 2965.40 | 2948.45 | 0.00 | ORB-long ORB[2917.35,2961.00] vol=1.5x ATR=7.89 |
| Stop hit — per-position SL triggered | 2024-09-04 09:45:00 | 2957.51 | 2952.90 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 3001.95 | 2988.94 | 0.00 | ORB-long ORB[2971.75,2999.90] vol=2.0x ATR=9.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:40:00 | 3016.65 | 2996.30 | 0.00 | T1 1.5R @ 3016.65 |
| Stop hit — per-position SL triggered | 2024-09-05 09:45:00 | 3001.95 | 2997.61 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:40:00 | 2982.00 | 3008.77 | 0.00 | ORB-short ORB[2999.10,3023.80] vol=2.3x ATR=8.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:50:00 | 2969.34 | 3002.15 | 0.00 | T1 1.5R @ 2969.34 |
| Stop hit — per-position SL triggered | 2024-09-06 11:00:00 | 2982.00 | 2983.94 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:35:00 | 2975.65 | 2964.82 | 0.00 | ORB-long ORB[2954.05,2972.60] vol=2.3x ATR=7.63 |
| Stop hit — per-position SL triggered | 2024-09-10 10:45:00 | 2968.02 | 2966.04 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 2945.50 | 2964.67 | 0.00 | ORB-short ORB[2955.05,2988.00] vol=1.8x ATR=8.32 |
| Stop hit — per-position SL triggered | 2024-09-11 09:35:00 | 2953.82 | 2962.49 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 2951.70 | 2938.40 | 0.00 | ORB-long ORB[2924.45,2945.90] vol=1.9x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:55:00 | 2962.49 | 2949.41 | 0.00 | T1 1.5R @ 2962.49 |
| Stop hit — per-position SL triggered | 2024-09-13 10:10:00 | 2951.70 | 2950.94 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:00:00 | 2912.75 | 2917.91 | 0.00 | ORB-short ORB[2919.45,2935.00] vol=3.0x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-09-16 11:20:00 | 2918.12 | 2917.59 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 2914.45 | 2927.45 | 0.00 | ORB-short ORB[2925.00,2945.90] vol=2.5x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 2904.30 | 2922.23 | 0.00 | T1 1.5R @ 2904.30 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 2914.45 | 2921.85 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:10:00 | 2899.75 | 2908.28 | 0.00 | ORB-short ORB[2900.80,2914.60] vol=4.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-09-18 11:30:00 | 2905.03 | 2907.86 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:15:00 | 2847.15 | 2879.95 | 0.00 | ORB-short ORB[2894.70,2914.55] vol=1.7x ATR=7.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:25:00 | 2835.30 | 2875.53 | 0.00 | T1 1.5R @ 2835.30 |
| Stop hit — per-position SL triggered | 2024-09-19 12:35:00 | 2847.15 | 2852.99 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 2858.35 | 2853.48 | 0.00 | ORB-long ORB[2835.00,2850.15] vol=1.8x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 12:25:00 | 2867.62 | 2854.96 | 0.00 | T1 1.5R @ 2867.62 |
| Target hit | 2024-09-23 15:20:00 | 2868.20 | 2861.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:35:00 | 2841.40 | 2854.05 | 0.00 | ORB-short ORB[2850.00,2881.35] vol=1.6x ATR=7.69 |
| Stop hit — per-position SL triggered | 2024-09-25 09:40:00 | 2849.09 | 2852.55 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:40:00 | 2811.40 | 2817.09 | 0.00 | ORB-short ORB[2813.00,2840.95] vol=1.5x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:55:00 | 2802.05 | 2813.11 | 0.00 | T1 1.5R @ 2802.05 |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 2811.40 | 2812.09 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:50:00 | 2943.60 | 2926.49 | 0.00 | ORB-long ORB[2901.05,2928.65] vol=4.9x ATR=10.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:10:00 | 2959.07 | 2939.29 | 0.00 | T1 1.5R @ 2959.07 |
| Stop hit — per-position SL triggered | 2024-10-01 10:30:00 | 2943.60 | 2942.76 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 2818.50 | 2829.41 | 0.00 | ORB-short ORB[2825.10,2848.50] vol=1.7x ATR=11.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:05:00 | 2801.80 | 2821.22 | 0.00 | T1 1.5R @ 2801.80 |
| Target hit | 2024-10-07 11:15:00 | 2809.35 | 2800.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — BUY (started 2024-10-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:45:00 | 2790.70 | 2769.96 | 0.00 | ORB-long ORB[2749.75,2775.00] vol=1.9x ATR=13.23 |
| Stop hit — per-position SL triggered | 2024-10-08 10:00:00 | 2777.47 | 2773.35 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 2845.00 | 2833.79 | 0.00 | ORB-long ORB[2822.00,2843.75] vol=2.3x ATR=8.30 |
| Stop hit — per-position SL triggered | 2024-10-14 09:40:00 | 2836.70 | 2837.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 2957.05 | 2945.21 | 0.00 | ORB-long ORB[2925.00,2943.40] vol=4.4x ATR=8.13 |
| Stop hit — per-position SL triggered | 2024-10-16 09:35:00 | 2948.92 | 2947.46 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:10:00 | 2886.45 | 2934.30 | 0.00 | ORB-short ORB[2961.45,2999.00] vol=2.0x ATR=12.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:20:00 | 2867.87 | 2925.24 | 0.00 | T1 1.5R @ 2867.87 |
| Target hit | 2024-10-17 15:20:00 | 2872.50 | 2881.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 2806.85 | 2821.71 | 0.00 | ORB-short ORB[2816.25,2858.00] vol=1.6x ATR=9.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:45:00 | 2792.87 | 2815.12 | 0.00 | T1 1.5R @ 2792.87 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 2806.85 | 2809.88 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:20:00 | 2744.05 | 2770.21 | 0.00 | ORB-short ORB[2765.00,2793.45] vol=1.7x ATR=10.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:25:00 | 2728.68 | 2753.45 | 0.00 | T1 1.5R @ 2728.68 |
| Target hit | 2024-10-22 15:20:00 | 2656.50 | 2704.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-10-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:00:00 | 2680.60 | 2699.42 | 0.00 | ORB-short ORB[2691.00,2726.15] vol=1.9x ATR=10.35 |
| Stop hit — per-position SL triggered | 2024-10-25 10:05:00 | 2690.95 | 2697.80 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 11:00:00 | 2713.45 | 2685.41 | 0.00 | ORB-long ORB[2647.85,2683.90] vol=1.7x ATR=9.73 |
| Stop hit — per-position SL triggered | 2024-10-28 12:20:00 | 2703.72 | 2693.51 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-10-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:00:00 | 2673.90 | 2695.20 | 0.00 | ORB-short ORB[2691.25,2714.10] vol=1.5x ATR=8.57 |
| Stop hit — per-position SL triggered | 2024-10-29 10:15:00 | 2682.47 | 2693.07 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:15:00 | 2721.50 | 2708.96 | 0.00 | ORB-long ORB[2695.35,2718.00] vol=1.7x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:35:00 | 2735.07 | 2713.00 | 0.00 | T1 1.5R @ 2735.07 |
| Stop hit — per-position SL triggered | 2024-10-30 11:50:00 | 2721.50 | 2721.03 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:35:00 | 2742.90 | 2731.29 | 0.00 | ORB-long ORB[2701.90,2730.00] vol=6.2x ATR=10.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:05:00 | 2759.23 | 2736.69 | 0.00 | T1 1.5R @ 2759.23 |
| Stop hit — per-position SL triggered | 2024-11-06 10:25:00 | 2742.90 | 2738.63 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 11:15:00 | 2607.90 | 2618.41 | 0.00 | ORB-short ORB[2621.00,2654.30] vol=1.7x ATR=8.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:30:00 | 2595.36 | 2610.51 | 0.00 | T1 1.5R @ 2595.36 |
| Target hit | 2024-11-12 15:20:00 | 2555.75 | 2593.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 2738.40 | 2719.61 | 0.00 | ORB-long ORB[2700.15,2730.00] vol=3.0x ATR=8.28 |
| Stop hit — per-position SL triggered | 2024-11-27 10:05:00 | 2730.12 | 2726.17 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 2739.85 | 2724.62 | 0.00 | ORB-long ORB[2702.00,2729.25] vol=1.8x ATR=6.64 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 2733.21 | 2728.33 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 2713.80 | 2719.24 | 0.00 | ORB-short ORB[2715.05,2737.00] vol=1.9x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:40:00 | 2704.52 | 2714.93 | 0.00 | T1 1.5R @ 2704.52 |
| Target hit | 2024-12-12 14:05:00 | 2688.00 | 2686.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — BUY (started 2024-12-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:10:00 | 2700.00 | 2682.17 | 0.00 | ORB-long ORB[2670.70,2692.65] vol=1.9x ATR=6.37 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 2693.63 | 2683.64 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:05:00 | 2626.45 | 2615.09 | 0.00 | ORB-long ORB[2595.05,2625.90] vol=1.7x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:25:00 | 2637.09 | 2618.47 | 0.00 | T1 1.5R @ 2637.09 |
| Target hit | 2024-12-19 15:20:00 | 2636.40 | 2627.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-12-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:55:00 | 2636.00 | 2646.32 | 0.00 | ORB-short ORB[2636.95,2661.55] vol=1.7x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:00:00 | 2625.40 | 2644.03 | 0.00 | T1 1.5R @ 2625.40 |
| Stop hit — per-position SL triggered | 2024-12-20 10:30:00 | 2636.00 | 2639.74 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:15:00 | 2625.05 | 2604.18 | 0.00 | ORB-long ORB[2585.10,2603.85] vol=1.6x ATR=7.70 |
| Stop hit — per-position SL triggered | 2024-12-24 11:40:00 | 2617.35 | 2614.27 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 2581.25 | 2591.63 | 0.00 | ORB-short ORB[2589.00,2618.00] vol=1.8x ATR=7.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 2570.05 | 2583.98 | 0.00 | T1 1.5R @ 2570.05 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 2581.25 | 2578.41 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 2602.40 | 2585.80 | 0.00 | ORB-long ORB[2571.35,2588.25] vol=2.1x ATR=7.52 |
| Stop hit — per-position SL triggered | 2024-12-27 10:10:00 | 2594.88 | 2590.92 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 2541.40 | 2554.76 | 0.00 | ORB-short ORB[2550.05,2580.00] vol=2.2x ATR=6.34 |
| Stop hit — per-position SL triggered | 2024-12-30 09:45:00 | 2547.74 | 2549.75 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 2488.25 | 2494.25 | 0.00 | ORB-short ORB[2491.70,2516.75] vol=2.1x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:25:00 | 2480.91 | 2489.85 | 0.00 | T1 1.5R @ 2480.91 |
| Stop hit — per-position SL triggered | 2025-01-02 10:45:00 | 2488.25 | 2488.98 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 2500.00 | 2510.25 | 0.00 | ORB-short ORB[2505.25,2527.35] vol=4.0x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:55:00 | 2490.80 | 2502.54 | 0.00 | T1 1.5R @ 2490.80 |
| Stop hit — per-position SL triggered | 2025-01-03 10:35:00 | 2500.00 | 2500.57 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 2369.00 | 2379.52 | 0.00 | ORB-short ORB[2375.30,2400.40] vol=2.4x ATR=8.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:55:00 | 2355.74 | 2374.37 | 0.00 | T1 1.5R @ 2355.74 |
| Target hit | 2025-01-13 15:20:00 | 2328.75 | 2357.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2025-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:50:00 | 2361.65 | 2340.01 | 0.00 | ORB-long ORB[2329.80,2358.00] vol=2.0x ATR=7.27 |
| Stop hit — per-position SL triggered | 2025-01-15 10:55:00 | 2354.38 | 2342.78 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:50:00 | 2347.10 | 2367.22 | 0.00 | ORB-short ORB[2354.05,2379.90] vol=1.7x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-01-16 11:25:00 | 2352.97 | 2362.29 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 2331.30 | 2343.77 | 0.00 | ORB-short ORB[2339.00,2373.90] vol=1.7x ATR=6.62 |
| Stop hit — per-position SL triggered | 2025-01-20 10:25:00 | 2337.92 | 2339.50 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 2355.05 | 2364.75 | 0.00 | ORB-short ORB[2360.00,2376.75] vol=1.8x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:40:00 | 2344.98 | 2358.32 | 0.00 | T1 1.5R @ 2344.98 |
| Target hit | 2025-01-24 11:00:00 | 2335.25 | 2331.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2025-01-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:35:00 | 2207.00 | 2222.33 | 0.00 | ORB-short ORB[2212.25,2244.00] vol=1.7x ATR=8.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:55:00 | 2193.87 | 2212.88 | 0.00 | T1 1.5R @ 2193.87 |
| Stop hit — per-position SL triggered | 2025-01-28 10:00:00 | 2207.00 | 2212.76 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 2214.50 | 2203.70 | 0.00 | ORB-long ORB[2190.10,2207.00] vol=2.8x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:35:00 | 2226.27 | 2206.89 | 0.00 | T1 1.5R @ 2226.27 |
| Target hit | 2025-01-29 15:20:00 | 2272.90 | 2258.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2025-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:40:00 | 2300.85 | 2290.36 | 0.00 | ORB-long ORB[2273.00,2295.70] vol=1.6x ATR=6.69 |
| Stop hit — per-position SL triggered | 2025-01-30 09:50:00 | 2294.16 | 2291.35 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 2357.05 | 2344.58 | 0.00 | ORB-long ORB[2331.80,2353.50] vol=2.3x ATR=7.04 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 2350.01 | 2346.41 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:45:00 | 2368.85 | 2358.27 | 0.00 | ORB-long ORB[2347.80,2368.05] vol=3.6x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:25:00 | 2380.22 | 2365.98 | 0.00 | T1 1.5R @ 2380.22 |
| Target hit | 2025-02-06 10:40:00 | 2371.20 | 2371.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:15:00 | 1899.90 | 1914.00 | 0.00 | ORB-short ORB[1906.00,1930.45] vol=2.0x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-02-27 11:25:00 | 1905.25 | 1913.69 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 09:40:00 | 1857.30 | 1841.23 | 0.00 | ORB-long ORB[1818.70,1845.65] vol=2.1x ATR=9.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 10:15:00 | 1870.93 | 1851.00 | 0.00 | T1 1.5R @ 1870.93 |
| Stop hit — per-position SL triggered | 2025-03-04 12:20:00 | 1857.30 | 1859.22 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:00:00 | 1988.25 | 1973.86 | 0.00 | ORB-long ORB[1947.05,1972.00] vol=1.5x ATR=6.03 |
| Stop hit — per-position SL triggered | 2025-03-07 11:40:00 | 1982.22 | 1975.98 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:05:00 | 1928.00 | 1955.35 | 0.00 | ORB-short ORB[1959.50,1985.00] vol=3.5x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:35:00 | 1918.60 | 1949.59 | 0.00 | T1 1.5R @ 1918.60 |
| Stop hit — per-position SL triggered | 2025-03-12 13:15:00 | 1928.00 | 1937.24 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 2005.75 | 1996.84 | 0.00 | ORB-long ORB[1981.55,2001.35] vol=1.6x ATR=5.66 |
| Stop hit — per-position SL triggered | 2025-03-18 10:05:00 | 2000.09 | 2000.86 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:30:00 | 2017.65 | 2003.71 | 0.00 | ORB-long ORB[1988.55,2007.50] vol=2.7x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 09:35:00 | 2027.25 | 2013.32 | 0.00 | T1 1.5R @ 2027.25 |
| Target hit | 2025-03-19 13:40:00 | 2025.85 | 2027.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 87 — SELL (started 2025-03-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:00:00 | 2007.55 | 2022.08 | 0.00 | ORB-short ORB[2022.00,2044.00] vol=7.2x ATR=6.06 |
| Stop hit — per-position SL triggered | 2025-03-26 11:05:00 | 2013.61 | 2021.43 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 1951.05 | 1974.65 | 0.00 | ORB-short ORB[1970.00,1989.25] vol=1.9x ATR=6.12 |
| Stop hit — per-position SL triggered | 2025-04-01 11:20:00 | 1957.17 | 1965.47 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-04-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:40:00 | 2001.05 | 1986.00 | 0.00 | ORB-long ORB[1966.35,1994.60] vol=3.1x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 11:00:00 | 2010.44 | 1990.00 | 0.00 | T1 1.5R @ 2010.44 |
| Target hit | 2025-04-02 15:20:00 | 2033.50 | 2018.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — SELL (started 2025-04-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:10:00 | 1800.30 | 1810.01 | 0.00 | ORB-short ORB[1810.10,1827.75] vol=1.6x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:35:00 | 1790.30 | 1806.01 | 0.00 | T1 1.5R @ 1790.30 |
| Stop hit — per-position SL triggered | 2025-04-09 11:30:00 | 1800.30 | 1802.46 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:05:00 | 1962.20 | 1957.31 | 0.00 | ORB-long ORB[1932.80,1962.00] vol=2.4x ATR=7.35 |
| Stop hit — per-position SL triggered | 2025-04-15 10:20:00 | 1954.85 | 1957.35 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 2047.70 | 2033.14 | 0.00 | ORB-long ORB[2008.90,2039.50] vol=2.1x ATR=8.23 |
| Stop hit — per-position SL triggered | 2025-04-22 09:55:00 | 2039.47 | 2035.79 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 2003.10 | 2021.13 | 0.00 | ORB-short ORB[2021.00,2043.20] vol=1.7x ATR=7.37 |
| Stop hit — per-position SL triggered | 2025-04-23 10:25:00 | 2010.47 | 2016.34 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 2029.30 | 2020.26 | 0.00 | ORB-long ORB[2008.80,2027.50] vol=1.9x ATR=6.46 |
| Stop hit — per-position SL triggered | 2025-04-24 10:00:00 | 2022.84 | 2022.52 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 1999.90 | 1984.89 | 0.00 | ORB-long ORB[1970.10,1993.50] vol=2.3x ATR=8.10 |
| Stop hit — per-position SL triggered | 2025-04-30 09:40:00 | 1991.80 | 1986.51 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1990.00 | 1998.49 | 0.00 | ORB-short ORB[1994.30,2021.90] vol=1.5x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:15:00 | 1980.18 | 1995.44 | 0.00 | T1 1.5R @ 1980.18 |
| Target hit | 2025-05-06 15:20:00 | 1956.30 | 1978.37 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:25:00 | 2458.15 | 2024-05-14 10:45:00 | 2465.44 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-15 10:05:00 | 2448.00 | 2024-05-15 10:15:00 | 2456.09 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-16 09:40:00 | 2429.55 | 2024-05-16 09:50:00 | 2436.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-17 11:10:00 | 2482.90 | 2024-05-17 11:25:00 | 2475.69 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-24 10:50:00 | 2359.95 | 2024-05-24 11:40:00 | 2353.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-28 09:30:00 | 2327.20 | 2024-05-28 09:35:00 | 2334.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-31 11:10:00 | 2170.05 | 2024-05-31 11:15:00 | 2176.03 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-06 09:30:00 | 2255.00 | 2024-06-06 09:45:00 | 2266.36 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-06 09:30:00 | 2255.00 | 2024-06-06 15:20:00 | 2280.00 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2024-06-11 11:00:00 | 2323.70 | 2024-06-11 11:15:00 | 2332.15 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-06-11 11:00:00 | 2323.70 | 2024-06-11 12:00:00 | 2323.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:25:00 | 2313.75 | 2024-06-12 10:50:00 | 2319.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-13 09:55:00 | 2376.85 | 2024-06-13 10:00:00 | 2386.53 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-06-13 09:55:00 | 2376.85 | 2024-06-13 11:20:00 | 2380.05 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-06-20 09:35:00 | 2508.20 | 2024-06-20 09:40:00 | 2523.73 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-06-20 09:35:00 | 2508.20 | 2024-06-20 11:55:00 | 2531.65 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-06-25 09:35:00 | 2500.00 | 2024-06-25 09:40:00 | 2507.53 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-01 09:35:00 | 2539.10 | 2024-07-01 09:50:00 | 2552.88 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-01 09:35:00 | 2539.10 | 2024-07-01 15:20:00 | 2617.65 | TARGET_HIT | 0.50 | 3.09% |
| SELL | retest1 | 2024-07-03 11:15:00 | 2653.00 | 2024-07-03 11:30:00 | 2638.86 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-03 11:15:00 | 2653.00 | 2024-07-03 11:40:00 | 2653.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 09:35:00 | 2700.40 | 2024-07-05 09:40:00 | 2713.03 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-05 09:35:00 | 2700.40 | 2024-07-05 09:50:00 | 2700.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 10:25:00 | 2713.35 | 2024-07-08 10:30:00 | 2704.38 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-09 09:40:00 | 2729.85 | 2024-07-09 09:55:00 | 2743.41 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-09 09:40:00 | 2729.85 | 2024-07-09 10:15:00 | 2729.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 11:05:00 | 2755.00 | 2024-07-11 11:10:00 | 2747.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-15 10:25:00 | 2795.45 | 2024-07-15 11:40:00 | 2810.21 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-15 10:25:00 | 2795.45 | 2024-07-15 15:10:00 | 2797.20 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-07-19 10:20:00 | 2844.05 | 2024-07-19 10:30:00 | 2835.49 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-25 09:40:00 | 2915.35 | 2024-07-25 12:45:00 | 2929.45 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-25 09:40:00 | 2915.35 | 2024-07-25 15:20:00 | 2957.50 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2024-08-01 09:35:00 | 3143.90 | 2024-08-01 09:45:00 | 3131.98 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-02 11:10:00 | 3100.00 | 2024-08-02 11:50:00 | 3088.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-09 09:40:00 | 3138.30 | 2024-08-09 09:55:00 | 3124.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-12 09:50:00 | 3024.90 | 2024-08-12 10:05:00 | 3037.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-16 09:30:00 | 2847.10 | 2024-08-16 09:40:00 | 2855.86 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-20 10:40:00 | 2875.95 | 2024-08-20 10:50:00 | 2865.26 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-20 10:40:00 | 2875.95 | 2024-08-20 12:30:00 | 2871.95 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-08-28 09:30:00 | 2841.10 | 2024-08-28 09:35:00 | 2850.54 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-29 09:50:00 | 2916.10 | 2024-08-29 10:00:00 | 2929.26 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-29 09:50:00 | 2916.10 | 2024-08-29 10:05:00 | 2916.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:30:00 | 2943.70 | 2024-09-03 09:35:00 | 2935.66 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-04 09:30:00 | 2965.40 | 2024-09-04 09:45:00 | 2957.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-05 09:30:00 | 3001.95 | 2024-09-05 09:40:00 | 3016.65 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-05 09:30:00 | 3001.95 | 2024-09-05 09:45:00 | 3001.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:40:00 | 2982.00 | 2024-09-06 09:50:00 | 2969.34 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-06 09:40:00 | 2982.00 | 2024-09-06 11:00:00 | 2982.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 10:35:00 | 2975.65 | 2024-09-10 10:45:00 | 2968.02 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-11 09:30:00 | 2945.50 | 2024-09-11 09:35:00 | 2953.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-13 09:30:00 | 2951.70 | 2024-09-13 09:55:00 | 2962.49 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-13 09:30:00 | 2951.70 | 2024-09-13 10:10:00 | 2951.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 11:00:00 | 2912.75 | 2024-09-16 11:20:00 | 2918.12 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-17 09:40:00 | 2914.45 | 2024-09-17 09:55:00 | 2904.30 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-17 09:40:00 | 2914.45 | 2024-09-17 10:00:00 | 2914.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 11:10:00 | 2899.75 | 2024-09-18 11:30:00 | 2905.03 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-19 11:15:00 | 2847.15 | 2024-09-19 11:25:00 | 2835.30 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-19 11:15:00 | 2847.15 | 2024-09-19 12:35:00 | 2847.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 11:15:00 | 2858.35 | 2024-09-23 12:25:00 | 2867.62 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-09-23 11:15:00 | 2858.35 | 2024-09-23 15:20:00 | 2868.20 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-25 09:35:00 | 2841.40 | 2024-09-25 09:40:00 | 2849.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-26 09:40:00 | 2811.40 | 2024-09-26 09:55:00 | 2802.05 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-26 09:40:00 | 2811.40 | 2024-09-26 10:15:00 | 2811.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 09:50:00 | 2943.60 | 2024-10-01 10:10:00 | 2959.07 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-01 09:50:00 | 2943.60 | 2024-10-01 10:30:00 | 2943.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 09:45:00 | 2818.50 | 2024-10-07 10:05:00 | 2801.80 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-10-07 09:45:00 | 2818.50 | 2024-10-07 11:15:00 | 2809.35 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-10-08 09:45:00 | 2790.70 | 2024-10-08 10:00:00 | 2777.47 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-14 09:30:00 | 2845.00 | 2024-10-14 09:40:00 | 2836.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-16 09:30:00 | 2957.05 | 2024-10-16 09:35:00 | 2948.92 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-17 11:10:00 | 2886.45 | 2024-10-17 11:20:00 | 2867.87 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-10-17 11:10:00 | 2886.45 | 2024-10-17 15:20:00 | 2872.50 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-21 09:35:00 | 2806.85 | 2024-10-21 09:45:00 | 2792.87 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-21 09:35:00 | 2806.85 | 2024-10-21 10:00:00 | 2806.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:20:00 | 2744.05 | 2024-10-22 11:25:00 | 2728.68 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-10-22 10:20:00 | 2744.05 | 2024-10-22 15:20:00 | 2656.50 | TARGET_HIT | 0.50 | 3.19% |
| SELL | retest1 | 2024-10-25 10:00:00 | 2680.60 | 2024-10-25 10:05:00 | 2690.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-28 11:00:00 | 2713.45 | 2024-10-28 12:20:00 | 2703.72 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-29 10:00:00 | 2673.90 | 2024-10-29 10:15:00 | 2682.47 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-30 10:15:00 | 2721.50 | 2024-10-30 10:35:00 | 2735.07 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-30 10:15:00 | 2721.50 | 2024-10-30 11:50:00 | 2721.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 09:35:00 | 2742.90 | 2024-11-06 10:05:00 | 2759.23 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-06 09:35:00 | 2742.90 | 2024-11-06 10:25:00 | 2742.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 11:15:00 | 2607.90 | 2024-11-12 12:30:00 | 2595.36 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-12 11:15:00 | 2607.90 | 2024-11-12 15:20:00 | 2555.75 | TARGET_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2024-11-27 09:50:00 | 2738.40 | 2024-11-27 10:05:00 | 2730.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-28 09:50:00 | 2739.85 | 2024-11-28 10:05:00 | 2733.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-12 09:30:00 | 2713.80 | 2024-12-12 09:40:00 | 2704.52 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-12 09:30:00 | 2713.80 | 2024-12-12 14:05:00 | 2688.00 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2024-12-17 10:10:00 | 2700.00 | 2024-12-17 10:20:00 | 2693.63 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-19 11:05:00 | 2626.45 | 2024-12-19 11:25:00 | 2637.09 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-19 11:05:00 | 2626.45 | 2024-12-19 15:20:00 | 2636.40 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-20 09:55:00 | 2636.00 | 2024-12-20 10:00:00 | 2625.40 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-20 09:55:00 | 2636.00 | 2024-12-20 10:30:00 | 2636.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:15:00 | 2625.05 | 2024-12-24 11:40:00 | 2617.35 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-26 09:35:00 | 2581.25 | 2024-12-26 10:15:00 | 2570.05 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-26 09:35:00 | 2581.25 | 2024-12-26 11:00:00 | 2581.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:50:00 | 2602.40 | 2024-12-27 10:10:00 | 2594.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-30 09:30:00 | 2541.40 | 2024-12-30 09:45:00 | 2547.74 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-02 09:50:00 | 2488.25 | 2025-01-02 10:25:00 | 2480.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-01-02 09:50:00 | 2488.25 | 2025-01-02 10:45:00 | 2488.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 09:30:00 | 2500.00 | 2025-01-03 09:55:00 | 2490.80 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-01-03 09:30:00 | 2500.00 | 2025-01-03 10:35:00 | 2500.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-13 11:10:00 | 2369.00 | 2025-01-13 12:55:00 | 2355.74 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-13 11:10:00 | 2369.00 | 2025-01-13 15:20:00 | 2328.75 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-01-15 10:50:00 | 2361.65 | 2025-01-15 10:55:00 | 2354.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-16 10:50:00 | 2347.10 | 2025-01-16 11:25:00 | 2352.97 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-20 09:45:00 | 2331.30 | 2025-01-20 10:25:00 | 2337.92 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-24 09:30:00 | 2355.05 | 2025-01-24 09:40:00 | 2344.98 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-24 09:30:00 | 2355.05 | 2025-01-24 11:00:00 | 2335.25 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-01-28 09:35:00 | 2207.00 | 2025-01-28 09:55:00 | 2193.87 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-28 09:35:00 | 2207.00 | 2025-01-28 10:00:00 | 2207.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 09:30:00 | 2214.50 | 2025-01-29 09:35:00 | 2226.27 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-29 09:30:00 | 2214.50 | 2025-01-29 15:20:00 | 2272.90 | TARGET_HIT | 0.50 | 2.64% |
| BUY | retest1 | 2025-01-30 09:40:00 | 2300.85 | 2025-01-30 09:50:00 | 2294.16 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-05 09:35:00 | 2357.05 | 2025-02-05 09:45:00 | 2350.01 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-06 09:45:00 | 2368.85 | 2025-02-06 10:25:00 | 2380.22 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-02-06 09:45:00 | 2368.85 | 2025-02-06 10:40:00 | 2371.20 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-02-27 11:15:00 | 1899.90 | 2025-02-27 11:25:00 | 1905.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-04 09:40:00 | 1857.30 | 2025-03-04 10:15:00 | 1870.93 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-03-04 09:40:00 | 1857.30 | 2025-03-04 12:20:00 | 1857.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 11:00:00 | 1988.25 | 2025-03-07 11:40:00 | 1982.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-12 11:05:00 | 1928.00 | 2025-03-12 11:35:00 | 1918.60 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-03-12 11:05:00 | 1928.00 | 2025-03-12 13:15:00 | 1928.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 09:40:00 | 2005.75 | 2025-03-18 10:05:00 | 2000.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-19 09:30:00 | 2017.65 | 2025-03-19 09:35:00 | 2027.25 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-19 09:30:00 | 2017.65 | 2025-03-19 13:40:00 | 2025.85 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2025-03-26 11:00:00 | 2007.55 | 2025-03-26 11:05:00 | 2013.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-04-01 11:00:00 | 1951.05 | 2025-04-01 11:20:00 | 1957.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-02 10:40:00 | 2001.05 | 2025-04-02 11:00:00 | 2010.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-02 10:40:00 | 2001.05 | 2025-04-02 15:20:00 | 2033.50 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2025-04-09 10:10:00 | 1800.30 | 2025-04-09 10:35:00 | 1790.30 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-09 10:10:00 | 1800.30 | 2025-04-09 11:30:00 | 1800.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 10:05:00 | 1962.20 | 2025-04-15 10:20:00 | 1954.85 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-22 09:40:00 | 2047.70 | 2025-04-22 09:55:00 | 2039.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-23 10:05:00 | 2003.10 | 2025-04-23 10:25:00 | 2010.47 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-24 09:30:00 | 2029.30 | 2025-04-24 10:00:00 | 2022.84 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-30 09:30:00 | 1999.90 | 2025-04-30 09:40:00 | 1991.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-05-06 10:15:00 | 1990.00 | 2025-05-06 11:15:00 | 1980.18 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-06 10:15:00 | 1990.00 | 2025-05-06 15:20:00 | 1956.30 | TARGET_HIT | 0.50 | 1.69% |
