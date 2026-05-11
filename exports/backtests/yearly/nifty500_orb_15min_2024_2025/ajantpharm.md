# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 3033.00
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
| ENTRY1 | 38 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 11 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 27
- **Target hits / Stop hits / Partials:** 11 / 27 / 18
- **Avg / median % per leg:** 0.21% / 0.13%
- **Sum % (uncompounded):** 12.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 11 | 47.8% | 4 | 12 | 7 | 0.15% | 3.5% |
| BUY @ 2nd Alert (retest1) | 23 | 11 | 47.8% | 4 | 12 | 7 | 0.15% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 18 | 54.5% | 7 | 15 | 11 | 0.26% | 8.5% |
| SELL @ 2nd Alert (retest1) | 33 | 18 | 54.5% | 7 | 15 | 11 | 0.26% | 8.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 56 | 29 | 51.8% | 11 | 27 | 18 | 0.21% | 12.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:35:00 | 2387.20 | 2391.84 | 0.00 | ORB-short ORB[2390.40,2401.85] vol=1.6x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-05-17 10:40:00 | 2392.18 | 2391.58 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 2411.00 | 2403.52 | 0.00 | ORB-long ORB[2392.65,2409.00] vol=2.8x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 12:20:00 | 2419.18 | 2408.73 | 0.00 | T1 1.5R @ 2419.18 |
| Stop hit — per-position SL triggered | 2024-05-24 14:00:00 | 2411.00 | 2414.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 2442.25 | 2426.55 | 0.00 | ORB-long ORB[2404.00,2436.45] vol=1.9x ATR=8.16 |
| Stop hit — per-position SL triggered | 2024-05-27 10:00:00 | 2434.09 | 2431.92 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-06 11:10:00 | 2298.05 | 2309.01 | 0.00 | ORB-short ORB[2307.30,2329.05] vol=1.9x ATR=6.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 11:50:00 | 2287.96 | 2304.87 | 0.00 | T1 1.5R @ 2287.96 |
| Stop hit — per-position SL triggered | 2024-06-06 11:55:00 | 2298.05 | 2304.65 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:40:00 | 2339.05 | 2327.51 | 0.00 | ORB-long ORB[2302.20,2330.00] vol=2.0x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:50:00 | 2350.99 | 2336.65 | 0.00 | T1 1.5R @ 2350.99 |
| Target hit | 2024-06-07 10:30:00 | 2356.35 | 2357.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 2396.40 | 2405.60 | 0.00 | ORB-short ORB[2406.40,2439.95] vol=4.1x ATR=9.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 09:55:00 | 2381.76 | 2398.96 | 0.00 | T1 1.5R @ 2381.76 |
| Target hit | 2024-06-11 10:10:00 | 2394.70 | 2394.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 2343.65 | 2337.55 | 0.00 | ORB-long ORB[2321.30,2341.40] vol=1.6x ATR=6.57 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 2337.08 | 2338.02 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 2244.30 | 2259.80 | 0.00 | ORB-short ORB[2253.00,2280.00] vol=2.4x ATR=5.09 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 2249.39 | 2259.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:10:00 | 2259.15 | 2251.32 | 0.00 | ORB-long ORB[2229.40,2256.95] vol=1.6x ATR=8.60 |
| Stop hit — per-position SL triggered | 2024-07-03 10:30:00 | 2250.55 | 2251.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:45:00 | 2266.00 | 2237.69 | 0.00 | ORB-long ORB[2217.65,2248.05] vol=2.0x ATR=6.73 |
| Stop hit — per-position SL triggered | 2024-07-04 10:50:00 | 2259.27 | 2243.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:55:00 | 2267.80 | 2279.72 | 0.00 | ORB-short ORB[2286.70,2304.90] vol=1.9x ATR=7.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:05:00 | 2256.22 | 2277.10 | 0.00 | T1 1.5R @ 2256.22 |
| Stop hit — per-position SL triggered | 2024-07-09 10:10:00 | 2267.80 | 2276.01 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 2252.35 | 2255.02 | 0.00 | ORB-short ORB[2252.40,2285.80] vol=12.3x ATR=12.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 12:25:00 | 2233.50 | 2254.08 | 0.00 | T1 1.5R @ 2233.50 |
| Target hit | 2024-07-10 14:15:00 | 2249.50 | 2247.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2024-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:00:00 | 2244.40 | 2255.48 | 0.00 | ORB-short ORB[2256.15,2284.90] vol=2.2x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:25:00 | 2234.34 | 2251.84 | 0.00 | T1 1.5R @ 2234.34 |
| Target hit | 2024-07-11 15:20:00 | 2212.70 | 2221.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:40:00 | 2237.60 | 2231.89 | 0.00 | ORB-long ORB[2213.85,2234.15] vol=7.2x ATR=5.00 |
| Stop hit — per-position SL triggered | 2024-07-12 10:10:00 | 2232.60 | 2232.36 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 2239.80 | 2230.41 | 0.00 | ORB-long ORB[2203.00,2222.85] vol=1.6x ATR=8.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 15:00:00 | 2252.62 | 2242.56 | 0.00 | T1 1.5R @ 2252.62 |
| Target hit | 2024-07-15 15:20:00 | 2248.50 | 2248.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-07-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:50:00 | 2292.40 | 2274.94 | 0.00 | ORB-long ORB[2251.00,2273.15] vol=2.9x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:35:00 | 2303.43 | 2283.97 | 0.00 | T1 1.5R @ 2303.43 |
| Stop hit — per-position SL triggered | 2024-07-22 12:40:00 | 2292.40 | 2294.60 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 2970.00 | 2987.51 | 0.00 | ORB-short ORB[2985.00,3013.25] vol=1.8x ATR=11.25 |
| Stop hit — per-position SL triggered | 2024-08-27 09:35:00 | 2981.25 | 2985.81 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 3235.90 | 3209.71 | 0.00 | ORB-long ORB[3190.00,3213.00] vol=2.8x ATR=11.76 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 3224.14 | 3212.11 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 3107.60 | 3137.28 | 0.00 | ORB-short ORB[3134.05,3172.70] vol=2.9x ATR=12.29 |
| Stop hit — per-position SL triggered | 2024-09-18 09:50:00 | 3119.89 | 3128.48 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 3029.15 | 3091.38 | 0.00 | ORB-short ORB[3124.35,3160.00] vol=2.3x ATR=15.47 |
| Stop hit — per-position SL triggered | 2024-10-22 10:45:00 | 3044.62 | 3080.64 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-11-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:10:00 | 3085.75 | 3101.15 | 0.00 | ORB-short ORB[3086.40,3125.00] vol=2.6x ATR=10.98 |
| Stop hit — per-position SL triggered | 2024-11-06 14:45:00 | 3096.73 | 3093.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-11-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:45:00 | 2956.25 | 2950.49 | 0.00 | ORB-long ORB[2930.35,2956.10] vol=5.8x ATR=9.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:00:00 | 2969.76 | 2951.94 | 0.00 | T1 1.5R @ 2969.76 |
| Target hit | 2024-11-29 13:55:00 | 3000.00 | 3001.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — SELL (started 2024-12-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:45:00 | 3026.45 | 3039.36 | 0.00 | ORB-short ORB[3036.05,3059.35] vol=2.7x ATR=9.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:55:00 | 3012.89 | 3035.87 | 0.00 | T1 1.5R @ 3012.89 |
| Target hit | 2024-12-03 12:50:00 | 3004.50 | 3003.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:05:00 | 2892.50 | 2887.80 | 0.00 | ORB-long ORB[2860.00,2892.25] vol=1.5x ATR=8.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:05:00 | 2904.65 | 2890.68 | 0.00 | T1 1.5R @ 2904.65 |
| Stop hit — per-position SL triggered | 2024-12-18 12:35:00 | 2892.50 | 2891.54 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 2793.40 | 2798.98 | 0.00 | ORB-short ORB[2795.10,2818.75] vol=2.3x ATR=9.18 |
| Stop hit — per-position SL triggered | 2024-12-24 13:10:00 | 2802.58 | 2797.44 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:05:00 | 2770.00 | 2789.18 | 0.00 | ORB-short ORB[2787.45,2807.50] vol=5.7x ATR=10.84 |
| Stop hit — per-position SL triggered | 2024-12-26 10:10:00 | 2780.84 | 2787.78 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 2935.00 | 2949.90 | 0.00 | ORB-short ORB[2961.00,2990.45] vol=1.5x ATR=8.95 |
| Stop hit — per-position SL triggered | 2025-01-02 11:25:00 | 2943.95 | 2944.81 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:30:00 | 2890.00 | 2909.02 | 0.00 | ORB-short ORB[2902.40,2934.00] vol=1.6x ATR=14.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:50:00 | 2868.11 | 2891.71 | 0.00 | T1 1.5R @ 2868.11 |
| Target hit | 2025-01-10 11:15:00 | 2881.45 | 2878.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2025-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:55:00 | 2724.75 | 2775.59 | 0.00 | ORB-short ORB[2807.75,2842.25] vol=2.4x ATR=15.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:25:00 | 2701.89 | 2759.16 | 0.00 | T1 1.5R @ 2701.89 |
| Stop hit — per-position SL triggered | 2025-01-13 11:40:00 | 2724.75 | 2754.96 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 2857.05 | 2869.54 | 0.00 | ORB-short ORB[2868.00,2883.15] vol=1.7x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:50:00 | 2844.19 | 2863.83 | 0.00 | T1 1.5R @ 2844.19 |
| Target hit | 2025-01-21 11:45:00 | 2829.05 | 2827.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2025-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:00:00 | 2869.05 | 2851.02 | 0.00 | ORB-long ORB[2826.00,2855.90] vol=1.5x ATR=12.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:20:00 | 2887.76 | 2858.20 | 0.00 | T1 1.5R @ 2887.76 |
| Stop hit — per-position SL triggered | 2025-01-23 12:05:00 | 2869.05 | 2886.87 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-02-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 11:00:00 | 2800.95 | 2796.71 | 0.00 | ORB-long ORB[2761.15,2798.80] vol=2.3x ATR=12.91 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 2788.04 | 2796.68 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:15:00 | 2742.00 | 2767.49 | 0.00 | ORB-short ORB[2756.15,2797.30] vol=1.6x ATR=9.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:30:00 | 2727.18 | 2758.86 | 0.00 | T1 1.5R @ 2727.18 |
| Target hit | 2025-02-06 15:20:00 | 2689.45 | 2723.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-02-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:55:00 | 2672.40 | 2692.23 | 0.00 | ORB-short ORB[2684.35,2713.95] vol=1.7x ATR=11.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 10:40:00 | 2654.98 | 2686.43 | 0.00 | T1 1.5R @ 2654.98 |
| Stop hit — per-position SL triggered | 2025-02-07 11:00:00 | 2672.40 | 2682.92 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:40:00 | 2600.35 | 2585.08 | 0.00 | ORB-long ORB[2567.90,2599.00] vol=2.2x ATR=11.28 |
| Stop hit — per-position SL triggered | 2025-03-18 10:45:00 | 2589.07 | 2585.35 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:55:00 | 2762.30 | 2735.28 | 0.00 | ORB-long ORB[2725.00,2759.00] vol=2.4x ATR=15.36 |
| Target hit | 2025-03-24 15:20:00 | 2769.00 | 2753.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 2722.70 | 2738.36 | 0.00 | ORB-short ORB[2745.00,2765.95] vol=2.0x ATR=14.64 |
| Stop hit — per-position SL triggered | 2025-03-26 15:20:00 | 2726.20 | 2724.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:55:00 | 2610.90 | 2627.76 | 0.00 | ORB-short ORB[2611.00,2648.00] vol=1.9x ATR=8.33 |
| Stop hit — per-position SL triggered | 2025-03-28 11:00:00 | 2619.23 | 2627.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 10:35:00 | 2387.20 | 2024-05-17 10:40:00 | 2392.18 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-24 10:40:00 | 2411.00 | 2024-05-24 12:20:00 | 2419.18 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-05-24 10:40:00 | 2411.00 | 2024-05-24 14:00:00 | 2411.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-27 09:30:00 | 2442.25 | 2024-05-27 10:00:00 | 2434.09 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-06 11:10:00 | 2298.05 | 2024-06-06 11:50:00 | 2287.96 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-06-06 11:10:00 | 2298.05 | 2024-06-06 11:55:00 | 2298.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:40:00 | 2339.05 | 2024-06-07 09:50:00 | 2350.99 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-06-07 09:40:00 | 2339.05 | 2024-06-07 10:30:00 | 2356.35 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2024-06-11 09:30:00 | 2396.40 | 2024-06-11 09:55:00 | 2381.76 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-11 09:30:00 | 2396.40 | 2024-06-11 10:10:00 | 2394.70 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-06-27 09:45:00 | 2343.65 | 2024-06-27 10:05:00 | 2337.08 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-02 10:35:00 | 2244.30 | 2024-07-02 10:40:00 | 2249.39 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-03 10:10:00 | 2259.15 | 2024-07-03 10:30:00 | 2250.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-04 10:45:00 | 2266.00 | 2024-07-04 10:50:00 | 2259.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-09 09:55:00 | 2267.80 | 2024-07-09 10:05:00 | 2256.22 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-09 09:55:00 | 2267.80 | 2024-07-09 10:10:00 | 2267.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:05:00 | 2252.35 | 2024-07-10 12:25:00 | 2233.50 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-07-10 10:05:00 | 2252.35 | 2024-07-10 14:15:00 | 2249.50 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-07-11 11:00:00 | 2244.40 | 2024-07-11 11:25:00 | 2234.34 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-11 11:00:00 | 2244.40 | 2024-07-11 15:20:00 | 2212.70 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-07-12 09:40:00 | 2237.60 | 2024-07-12 10:10:00 | 2232.60 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-15 10:50:00 | 2239.80 | 2024-07-15 15:00:00 | 2252.62 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-15 10:50:00 | 2239.80 | 2024-07-15 15:20:00 | 2248.50 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-22 10:50:00 | 2292.40 | 2024-07-22 11:35:00 | 2303.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-22 10:50:00 | 2292.40 | 2024-07-22 12:40:00 | 2292.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 09:30:00 | 2970.00 | 2024-08-27 09:35:00 | 2981.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-03 09:45:00 | 3235.90 | 2024-09-03 09:50:00 | 3224.14 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-18 09:45:00 | 3107.60 | 2024-09-18 09:50:00 | 3119.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-22 10:30:00 | 3029.15 | 2024-10-22 10:45:00 | 3044.62 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-11-06 11:10:00 | 3085.75 | 2024-11-06 14:45:00 | 3096.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-29 10:45:00 | 2956.25 | 2024-11-29 11:00:00 | 2969.76 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-11-29 10:45:00 | 2956.25 | 2024-11-29 13:55:00 | 3000.00 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2024-12-03 10:45:00 | 3026.45 | 2024-12-03 10:55:00 | 3012.89 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-03 10:45:00 | 3026.45 | 2024-12-03 12:50:00 | 3004.50 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2024-12-18 11:05:00 | 2892.50 | 2024-12-18 12:05:00 | 2904.65 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-18 11:05:00 | 2892.50 | 2024-12-18 12:35:00 | 2892.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 10:05:00 | 2793.40 | 2024-12-24 13:10:00 | 2802.58 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-26 10:05:00 | 2770.00 | 2024-12-26 10:10:00 | 2780.84 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-02 10:55:00 | 2935.00 | 2025-01-02 11:25:00 | 2943.95 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-10 09:30:00 | 2890.00 | 2025-01-10 09:50:00 | 2868.11 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-01-10 09:30:00 | 2890.00 | 2025-01-10 11:15:00 | 2881.45 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-01-13 10:55:00 | 2724.75 | 2025-01-13 11:25:00 | 2701.89 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2025-01-13 10:55:00 | 2724.75 | 2025-01-13 11:40:00 | 2724.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 09:30:00 | 2857.05 | 2025-01-21 09:50:00 | 2844.19 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-21 09:30:00 | 2857.05 | 2025-01-21 11:45:00 | 2829.05 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-01-23 10:00:00 | 2869.05 | 2025-01-23 10:20:00 | 2887.76 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-23 10:00:00 | 2869.05 | 2025-01-23 12:05:00 | 2869.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 11:00:00 | 2800.95 | 2025-02-05 11:20:00 | 2788.04 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-02-06 10:15:00 | 2742.00 | 2025-02-06 10:30:00 | 2727.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-06 10:15:00 | 2742.00 | 2025-02-06 15:20:00 | 2689.45 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2025-02-07 09:55:00 | 2672.40 | 2025-02-07 10:40:00 | 2654.98 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-02-07 09:55:00 | 2672.40 | 2025-02-07 11:00:00 | 2672.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:40:00 | 2600.35 | 2025-03-18 10:45:00 | 2589.07 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-24 10:55:00 | 2762.30 | 2025-03-24 15:20:00 | 2769.00 | TARGET_HIT | 1.00 | 0.24% |
| SELL | retest1 | 2025-03-26 09:35:00 | 2722.70 | 2025-03-26 15:20:00 | 2726.20 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-03-28 10:55:00 | 2610.90 | 2025-03-28 11:00:00 | 2619.23 | STOP_HIT | 1.00 | -0.32% |
