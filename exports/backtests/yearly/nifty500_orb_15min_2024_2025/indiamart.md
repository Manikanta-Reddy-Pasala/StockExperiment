# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (32275 bars)
- **Last close:** 2091.00
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
| ENTRY1 | 61 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 13 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 48
- **Target hits / Stop hits / Partials:** 13 / 48 / 28
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 14.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 19 | 44.2% | 5 | 24 | 14 | 0.17% | 7.4% |
| BUY @ 2nd Alert (retest1) | 43 | 19 | 44.2% | 5 | 24 | 14 | 0.17% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 22 | 47.8% | 8 | 24 | 14 | 0.16% | 7.2% |
| SELL @ 2nd Alert (retest1) | 46 | 22 | 47.8% | 8 | 24 | 14 | 0.16% | 7.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 89 | 41 | 46.1% | 13 | 48 | 28 | 0.16% | 14.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:50:00 | 2728.00 | 2745.04 | 0.00 | ORB-short ORB[2744.50,2780.05] vol=3.5x ATR=14.21 |
| Stop hit — per-position SL triggered | 2024-08-12 10:35:00 | 2742.21 | 2738.63 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 2696.95 | 2707.16 | 0.00 | ORB-short ORB[2701.05,2721.55] vol=3.1x ATR=11.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:35:00 | 2680.16 | 2702.68 | 0.00 | T1 1.5R @ 2680.16 |
| Target hit | 2024-08-14 12:20:00 | 2683.95 | 2683.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 2796.00 | 2772.34 | 0.00 | ORB-long ORB[2761.00,2790.00] vol=3.0x ATR=10.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:55:00 | 2811.40 | 2778.18 | 0.00 | T1 1.5R @ 2811.40 |
| Target hit | 2024-08-19 15:20:00 | 2855.75 | 2818.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 2953.95 | 2937.90 | 0.00 | ORB-long ORB[2913.70,2941.90] vol=1.6x ATR=11.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:40:00 | 2971.28 | 2951.30 | 0.00 | T1 1.5R @ 2971.28 |
| Stop hit — per-position SL triggered | 2024-08-22 09:50:00 | 2953.95 | 2954.74 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-08-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:25:00 | 2999.95 | 2986.91 | 0.00 | ORB-long ORB[2967.05,2996.05] vol=1.6x ATR=10.23 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 2989.72 | 2991.59 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 2991.95 | 3018.94 | 0.00 | ORB-short ORB[3025.15,3050.00] vol=1.5x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:45:00 | 2978.86 | 3011.98 | 0.00 | T1 1.5R @ 2978.86 |
| Stop hit — per-position SL triggered | 2024-08-29 12:30:00 | 2991.95 | 3004.94 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 2980.05 | 2997.05 | 0.00 | ORB-short ORB[2997.15,3019.65] vol=3.5x ATR=9.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 2966.30 | 2989.23 | 0.00 | T1 1.5R @ 2966.30 |
| Stop hit — per-position SL triggered | 2024-09-06 10:10:00 | 2980.05 | 2988.80 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 3157.35 | 3144.79 | 0.00 | ORB-long ORB[3128.00,3149.00] vol=4.5x ATR=9.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:40:00 | 3171.29 | 3155.00 | 0.00 | T1 1.5R @ 3171.29 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 3157.35 | 3155.58 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:45:00 | 3005.30 | 3023.78 | 0.00 | ORB-short ORB[3033.00,3059.90] vol=5.2x ATR=6.94 |
| Stop hit — per-position SL triggered | 2024-09-18 10:55:00 | 3012.24 | 3022.30 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-09-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:00:00 | 3044.00 | 3013.02 | 0.00 | ORB-long ORB[2996.05,3024.75] vol=2.7x ATR=9.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:15:00 | 3057.76 | 3022.03 | 0.00 | T1 1.5R @ 3057.76 |
| Target hit | 2024-09-20 13:05:00 | 3048.45 | 3054.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-09-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:25:00 | 2953.20 | 2981.02 | 0.00 | ORB-short ORB[2968.25,2993.90] vol=1.6x ATR=11.44 |
| Stop hit — per-position SL triggered | 2024-09-25 10:45:00 | 2964.64 | 2978.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:45:00 | 2947.45 | 2924.97 | 0.00 | ORB-long ORB[2910.05,2938.20] vol=2.5x ATR=9.47 |
| Stop hit — per-position SL triggered | 2024-09-30 10:55:00 | 2937.98 | 2926.19 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 3003.00 | 2991.69 | 0.00 | ORB-long ORB[2960.05,3000.00] vol=4.1x ATR=11.69 |
| Stop hit — per-position SL triggered | 2024-10-01 09:45:00 | 2991.31 | 2994.87 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:55:00 | 2946.05 | 2962.22 | 0.00 | ORB-short ORB[2954.35,2979.85] vol=2.3x ATR=8.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:15:00 | 2933.75 | 2960.18 | 0.00 | T1 1.5R @ 2933.75 |
| Stop hit — per-position SL triggered | 2024-10-03 12:45:00 | 2946.05 | 2942.34 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-10-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:45:00 | 3006.75 | 3000.40 | 0.00 | ORB-long ORB[2985.05,3001.50] vol=4.2x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:50:00 | 3016.62 | 3002.15 | 0.00 | T1 1.5R @ 3016.62 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 3006.75 | 3002.46 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:55:00 | 3088.20 | 3041.88 | 0.00 | ORB-long ORB[3007.05,3051.00] vol=1.9x ATR=15.67 |
| Stop hit — per-position SL triggered | 2024-10-16 10:05:00 | 3072.53 | 3043.82 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-10-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:05:00 | 3050.00 | 3017.90 | 0.00 | ORB-long ORB[2987.80,3020.00] vol=1.6x ATR=14.12 |
| Stop hit — per-position SL triggered | 2024-10-18 11:20:00 | 3035.88 | 3032.20 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:45:00 | 2439.00 | 2467.06 | 0.00 | ORB-short ORB[2470.10,2500.00] vol=2.9x ATR=11.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:00:00 | 2421.62 | 2452.93 | 0.00 | T1 1.5R @ 2421.62 |
| Stop hit — per-position SL triggered | 2024-10-25 10:05:00 | 2439.00 | 2452.11 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 11:15:00 | 2443.50 | 2454.84 | 0.00 | ORB-short ORB[2449.15,2472.15] vol=1.6x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:25:00 | 2435.10 | 2454.03 | 0.00 | T1 1.5R @ 2435.10 |
| Target hit | 2024-11-05 13:45:00 | 2436.40 | 2435.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:15:00 | 2378.35 | 2396.00 | 0.00 | ORB-short ORB[2391.05,2425.45] vol=3.0x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-11-08 11:25:00 | 2384.19 | 2395.25 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:00:00 | 2386.70 | 2373.43 | 0.00 | ORB-long ORB[2351.80,2373.45] vol=1.7x ATR=7.05 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 2379.65 | 2376.32 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:45:00 | 2351.75 | 2366.79 | 0.00 | ORB-short ORB[2361.60,2387.65] vol=2.1x ATR=7.26 |
| Stop hit — per-position SL triggered | 2024-11-12 11:05:00 | 2359.01 | 2364.74 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-11-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:55:00 | 2290.65 | 2271.25 | 0.00 | ORB-long ORB[2245.05,2271.00] vol=2.6x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:45:00 | 2301.08 | 2279.91 | 0.00 | T1 1.5R @ 2301.08 |
| Stop hit — per-position SL triggered | 2024-11-19 12:10:00 | 2290.65 | 2285.83 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:20:00 | 2268.30 | 2258.25 | 0.00 | ORB-long ORB[2244.10,2262.95] vol=1.8x ATR=5.36 |
| Stop hit — per-position SL triggered | 2024-11-22 10:25:00 | 2262.94 | 2259.63 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:35:00 | 2336.90 | 2328.52 | 0.00 | ORB-long ORB[2309.70,2335.95] vol=1.7x ATR=7.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 09:45:00 | 2348.06 | 2339.98 | 0.00 | T1 1.5R @ 2348.06 |
| Stop hit — per-position SL triggered | 2024-11-26 10:35:00 | 2336.90 | 2341.45 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 2354.75 | 2347.94 | 0.00 | ORB-long ORB[2324.80,2344.10] vol=2.9x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:05:00 | 2364.62 | 2352.93 | 0.00 | T1 1.5R @ 2364.62 |
| Stop hit — per-position SL triggered | 2024-11-27 10:30:00 | 2354.75 | 2354.88 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:05:00 | 2360.70 | 2374.14 | 0.00 | ORB-short ORB[2360.75,2389.00] vol=1.8x ATR=7.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:20:00 | 2349.72 | 2371.85 | 0.00 | T1 1.5R @ 2349.72 |
| Stop hit — per-position SL triggered | 2024-11-28 11:40:00 | 2360.70 | 2370.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:15:00 | 2328.75 | 2340.58 | 0.00 | ORB-short ORB[2336.05,2355.90] vol=1.7x ATR=7.58 |
| Stop hit — per-position SL triggered | 2024-11-29 10:30:00 | 2336.33 | 2339.56 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 2373.75 | 2364.53 | 0.00 | ORB-long ORB[2352.00,2368.80] vol=2.2x ATR=6.80 |
| Stop hit — per-position SL triggered | 2024-12-04 11:20:00 | 2366.95 | 2364.84 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:35:00 | 2363.00 | 2373.34 | 0.00 | ORB-short ORB[2365.50,2387.95] vol=1.5x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:50:00 | 2354.45 | 2368.33 | 0.00 | T1 1.5R @ 2354.45 |
| Target hit | 2024-12-05 10:40:00 | 2360.00 | 2359.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2024-12-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:55:00 | 2352.10 | 2363.38 | 0.00 | ORB-short ORB[2353.15,2379.95] vol=1.8x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 2358.94 | 2362.99 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 2374.00 | 2369.68 | 0.00 | ORB-long ORB[2357.00,2373.60] vol=2.0x ATR=5.17 |
| Stop hit — per-position SL triggered | 2024-12-09 09:50:00 | 2368.83 | 2371.53 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:40:00 | 2383.00 | 2372.18 | 0.00 | ORB-long ORB[2361.20,2372.90] vol=1.5x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:45:00 | 2390.17 | 2379.03 | 0.00 | T1 1.5R @ 2390.17 |
| Stop hit — per-position SL triggered | 2024-12-10 10:05:00 | 2383.00 | 2386.78 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:15:00 | 2402.00 | 2387.41 | 0.00 | ORB-long ORB[2380.00,2393.85] vol=4.2x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:20:00 | 2409.95 | 2389.46 | 0.00 | T1 1.5R @ 2409.95 |
| Stop hit — per-position SL triggered | 2024-12-11 11:25:00 | 2402.00 | 2389.78 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 2389.00 | 2395.22 | 0.00 | ORB-short ORB[2392.00,2409.00] vol=3.8x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:25:00 | 2380.84 | 2394.00 | 0.00 | T1 1.5R @ 2380.84 |
| Target hit | 2024-12-12 15:20:00 | 2343.40 | 2374.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 11:15:00 | 2354.15 | 2330.13 | 0.00 | ORB-long ORB[2329.00,2350.70] vol=3.0x ATR=8.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 12:15:00 | 2366.20 | 2335.54 | 0.00 | T1 1.5R @ 2366.20 |
| Target hit | 2024-12-13 15:20:00 | 2383.05 | 2350.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-12-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:45:00 | 2313.45 | 2345.98 | 0.00 | ORB-short ORB[2349.00,2373.00] vol=3.0x ATR=8.17 |
| Stop hit — per-position SL triggered | 2024-12-20 11:25:00 | 2321.62 | 2338.99 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 2242.45 | 2248.28 | 0.00 | ORB-short ORB[2244.75,2264.55] vol=1.7x ATR=7.05 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 2249.50 | 2249.04 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:40:00 | 2215.65 | 2228.51 | 0.00 | ORB-short ORB[2220.20,2251.85] vol=2.2x ATR=7.17 |
| Stop hit — per-position SL triggered | 2024-12-26 09:45:00 | 2222.82 | 2227.46 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 2232.00 | 2244.15 | 0.00 | ORB-short ORB[2232.30,2258.50] vol=1.7x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-12-27 11:05:00 | 2237.21 | 2243.39 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:50:00 | 2269.70 | 2260.41 | 0.00 | ORB-long ORB[2252.15,2266.95] vol=2.2x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 2263.73 | 2261.04 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 2279.95 | 2267.89 | 0.00 | ORB-long ORB[2245.15,2263.80] vol=3.8x ATR=6.21 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 2273.74 | 2269.18 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:10:00 | 2276.30 | 2282.54 | 0.00 | ORB-short ORB[2280.00,2292.10] vol=1.8x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 2281.41 | 2282.36 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:05:00 | 2278.00 | 2299.42 | 0.00 | ORB-short ORB[2293.00,2314.95] vol=1.5x ATR=8.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:35:00 | 2265.51 | 2294.21 | 0.00 | T1 1.5R @ 2265.51 |
| Target hit | 2025-01-13 15:20:00 | 2242.70 | 2270.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-01-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 11:00:00 | 2257.15 | 2240.71 | 0.00 | ORB-long ORB[2227.00,2250.85] vol=3.5x ATR=7.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:10:00 | 2269.07 | 2246.19 | 0.00 | T1 1.5R @ 2269.07 |
| Stop hit — per-position SL triggered | 2025-01-17 11:50:00 | 2257.15 | 2257.00 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:05:00 | 2054.10 | 2061.81 | 0.00 | ORB-short ORB[2055.00,2077.50] vol=3.0x ATR=7.27 |
| Stop hit — per-position SL triggered | 2025-01-27 10:40:00 | 2061.37 | 2059.79 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-29 09:40:00 | 2065.70 | 2081.21 | 0.00 | ORB-short ORB[2070.65,2100.45] vol=2.8x ATR=10.10 |
| Stop hit — per-position SL triggered | 2025-01-29 09:45:00 | 2075.80 | 2080.74 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:30:00 | 2064.50 | 2076.35 | 0.00 | ORB-short ORB[2075.00,2097.00] vol=1.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2025-01-31 11:20:00 | 2070.02 | 2072.36 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:05:00 | 2149.40 | 2170.30 | 0.00 | ORB-short ORB[2167.65,2198.45] vol=2.2x ATR=6.75 |
| Stop hit — per-position SL triggered | 2025-02-04 11:25:00 | 2156.15 | 2168.17 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 10:20:00 | 2196.55 | 2184.88 | 0.00 | ORB-long ORB[2171.95,2191.95] vol=1.7x ATR=7.41 |
| Stop hit — per-position SL triggered | 2025-02-10 11:05:00 | 2189.14 | 2191.77 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:55:00 | 2152.85 | 2167.10 | 0.00 | ORB-short ORB[2160.00,2189.35] vol=1.7x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:10:00 | 2141.57 | 2163.57 | 0.00 | T1 1.5R @ 2141.57 |
| Target hit | 2025-02-11 14:50:00 | 2149.95 | 2144.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2025-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:35:00 | 2200.00 | 2188.80 | 0.00 | ORB-long ORB[2173.00,2197.00] vol=1.7x ATR=8.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:45:00 | 2213.04 | 2195.50 | 0.00 | T1 1.5R @ 2213.04 |
| Target hit | 2025-02-13 12:40:00 | 2212.65 | 2214.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-02-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:05:00 | 2154.05 | 2173.42 | 0.00 | ORB-short ORB[2177.55,2207.90] vol=1.6x ATR=9.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 2140.01 | 2170.60 | 0.00 | T1 1.5R @ 2140.01 |
| Target hit | 2025-02-14 15:20:00 | 2123.50 | 2137.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 2078.90 | 2069.85 | 0.00 | ORB-long ORB[2055.00,2072.80] vol=1.6x ATR=7.87 |
| Stop hit — per-position SL triggered | 2025-02-20 09:55:00 | 2071.03 | 2072.48 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 09:55:00 | 1956.00 | 1950.39 | 0.00 | ORB-long ORB[1925.65,1954.05] vol=5.5x ATR=7.20 |
| Stop hit — per-position SL triggered | 2025-03-04 10:05:00 | 1948.80 | 1950.63 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 2007.30 | 2011.44 | 0.00 | ORB-short ORB[2009.45,2027.85] vol=5.7x ATR=8.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 09:50:00 | 1995.02 | 2008.75 | 0.00 | T1 1.5R @ 1995.02 |
| Stop hit — per-position SL triggered | 2025-03-07 10:25:00 | 2007.30 | 2007.14 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:55:00 | 1939.95 | 1941.49 | 0.00 | ORB-short ORB[1945.15,1964.00] vol=2.0x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:35:00 | 1930.92 | 1939.82 | 0.00 | T1 1.5R @ 1930.92 |
| Target hit | 2025-03-12 14:45:00 | 1933.75 | 1933.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 1961.20 | 1949.26 | 0.00 | ORB-long ORB[1930.00,1957.80] vol=2.1x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-03-17 09:50:00 | 1954.73 | 1952.23 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:40:00 | 2088.45 | 2103.36 | 0.00 | ORB-short ORB[2099.00,2118.00] vol=1.9x ATR=8.80 |
| Stop hit — per-position SL triggered | 2025-03-20 10:50:00 | 2097.25 | 2102.97 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:25:00 | 2258.90 | 2265.58 | 0.00 | ORB-short ORB[2265.90,2298.60] vol=1.5x ATR=9.56 |
| Stop hit — per-position SL triggered | 2025-04-23 10:30:00 | 2268.46 | 2265.54 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:45:00 | 2283.00 | 2262.93 | 0.00 | ORB-long ORB[2243.30,2271.70] vol=1.6x ATR=8.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:50:00 | 2296.18 | 2269.69 | 0.00 | T1 1.5R @ 2296.18 |
| Target hit | 2025-04-29 15:20:00 | 2322.40 | 2318.54 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-12 09:50:00 | 2728.00 | 2024-08-12 10:35:00 | 2742.21 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-08-14 09:30:00 | 2696.95 | 2024-08-14 09:35:00 | 2680.16 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-08-14 09:30:00 | 2696.95 | 2024-08-14 12:20:00 | 2683.95 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-19 10:45:00 | 2796.00 | 2024-08-19 10:55:00 | 2811.40 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-19 10:45:00 | 2796.00 | 2024-08-19 15:20:00 | 2855.75 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2024-08-22 09:30:00 | 2953.95 | 2024-08-22 09:40:00 | 2971.28 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-22 09:30:00 | 2953.95 | 2024-08-22 09:50:00 | 2953.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 10:25:00 | 2999.95 | 2024-08-27 11:30:00 | 2989.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-29 11:10:00 | 2991.95 | 2024-08-29 11:45:00 | 2978.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-29 11:10:00 | 2991.95 | 2024-08-29 12:30:00 | 2991.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2980.05 | 2024-09-06 10:05:00 | 2966.30 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2980.05 | 2024-09-06 10:10:00 | 2980.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 09:30:00 | 3157.35 | 2024-09-13 09:40:00 | 3171.29 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-13 09:30:00 | 3157.35 | 2024-09-13 09:50:00 | 3157.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:45:00 | 3005.30 | 2024-09-18 10:55:00 | 3012.24 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-20 11:00:00 | 3044.00 | 2024-09-20 11:15:00 | 3057.76 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-20 11:00:00 | 3044.00 | 2024-09-20 13:05:00 | 3048.45 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2024-09-25 10:25:00 | 2953.20 | 2024-09-25 10:45:00 | 2964.64 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-30 10:45:00 | 2947.45 | 2024-09-30 10:55:00 | 2937.98 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-01 09:30:00 | 3003.00 | 2024-10-01 09:45:00 | 2991.31 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-03 10:55:00 | 2946.05 | 2024-10-03 11:15:00 | 2933.75 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-03 10:55:00 | 2946.05 | 2024-10-03 12:45:00 | 2946.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 10:45:00 | 3006.75 | 2024-10-11 10:50:00 | 3016.62 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-10-11 10:45:00 | 3006.75 | 2024-10-11 10:55:00 | 3006.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 09:55:00 | 3088.20 | 2024-10-16 10:05:00 | 3072.53 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-10-18 10:05:00 | 3050.00 | 2024-10-18 11:20:00 | 3035.88 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-25 09:45:00 | 2439.00 | 2024-10-25 10:00:00 | 2421.62 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-10-25 09:45:00 | 2439.00 | 2024-10-25 10:05:00 | 2439.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 11:15:00 | 2443.50 | 2024-11-05 11:25:00 | 2435.10 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-11-05 11:15:00 | 2443.50 | 2024-11-05 13:45:00 | 2436.40 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-11-08 11:15:00 | 2378.35 | 2024-11-08 11:25:00 | 2384.19 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-11 11:00:00 | 2386.70 | 2024-11-11 12:00:00 | 2379.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-12 10:45:00 | 2351.75 | 2024-11-12 11:05:00 | 2359.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-19 09:55:00 | 2290.65 | 2024-11-19 10:45:00 | 2301.08 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-11-19 09:55:00 | 2290.65 | 2024-11-19 12:10:00 | 2290.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:20:00 | 2268.30 | 2024-11-22 10:25:00 | 2262.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-26 09:35:00 | 2336.90 | 2024-11-26 09:45:00 | 2348.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-26 09:35:00 | 2336.90 | 2024-11-26 10:35:00 | 2336.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:50:00 | 2354.75 | 2024-11-27 10:05:00 | 2364.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-27 09:50:00 | 2354.75 | 2024-11-27 10:30:00 | 2354.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 11:05:00 | 2360.70 | 2024-11-28 11:20:00 | 2349.72 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-11-28 11:05:00 | 2360.70 | 2024-11-28 11:40:00 | 2360.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 10:15:00 | 2328.75 | 2024-11-29 10:30:00 | 2336.33 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-04 11:00:00 | 2373.75 | 2024-12-04 11:20:00 | 2366.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-05 09:35:00 | 2363.00 | 2024-12-05 09:50:00 | 2354.45 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-05 09:35:00 | 2363.00 | 2024-12-05 10:40:00 | 2360.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-12-06 09:55:00 | 2352.10 | 2024-12-06 10:20:00 | 2358.94 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-09 09:30:00 | 2374.00 | 2024-12-09 09:50:00 | 2368.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-10 09:40:00 | 2383.00 | 2024-12-10 09:45:00 | 2390.17 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-12-10 09:40:00 | 2383.00 | 2024-12-10 10:05:00 | 2383.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:15:00 | 2402.00 | 2024-12-11 11:20:00 | 2409.95 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-12-11 11:15:00 | 2402.00 | 2024-12-11 11:25:00 | 2402.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:15:00 | 2389.00 | 2024-12-12 10:25:00 | 2380.84 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-12 10:15:00 | 2389.00 | 2024-12-12 15:20:00 | 2343.40 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2024-12-13 11:15:00 | 2354.15 | 2024-12-13 12:15:00 | 2366.20 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-13 11:15:00 | 2354.15 | 2024-12-13 15:20:00 | 2383.05 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2024-12-20 10:45:00 | 2313.45 | 2024-12-20 11:25:00 | 2321.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-24 09:30:00 | 2242.45 | 2024-12-24 09:40:00 | 2249.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-26 09:40:00 | 2215.65 | 2024-12-26 09:45:00 | 2222.82 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-27 10:55:00 | 2232.00 | 2024-12-27 11:05:00 | 2237.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-30 09:50:00 | 2269.70 | 2024-12-30 10:05:00 | 2263.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-01 10:50:00 | 2279.95 | 2025-01-01 11:10:00 | 2273.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-02 10:10:00 | 2276.30 | 2025-01-02 10:15:00 | 2281.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-13 11:05:00 | 2278.00 | 2025-01-13 11:35:00 | 2265.51 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-13 11:05:00 | 2278.00 | 2025-01-13 15:20:00 | 2242.70 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2025-01-17 11:00:00 | 2257.15 | 2025-01-17 11:10:00 | 2269.07 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-17 11:00:00 | 2257.15 | 2025-01-17 11:50:00 | 2257.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:05:00 | 2054.10 | 2025-01-27 10:40:00 | 2061.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-29 09:40:00 | 2065.70 | 2025-01-29 09:45:00 | 2075.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-31 10:30:00 | 2064.50 | 2025-01-31 11:20:00 | 2070.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-04 11:05:00 | 2149.40 | 2025-02-04 11:25:00 | 2156.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-10 10:20:00 | 2196.55 | 2025-02-10 11:05:00 | 2189.14 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-11 09:55:00 | 2152.85 | 2025-02-11 10:10:00 | 2141.57 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-11 09:55:00 | 2152.85 | 2025-02-11 14:50:00 | 2149.95 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-02-13 09:35:00 | 2200.00 | 2025-02-13 09:45:00 | 2213.04 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-02-13 09:35:00 | 2200.00 | 2025-02-13 12:40:00 | 2212.65 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-02-14 10:05:00 | 2154.05 | 2025-02-14 10:15:00 | 2140.01 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-02-14 10:05:00 | 2154.05 | 2025-02-14 15:20:00 | 2123.50 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2025-02-20 09:35:00 | 2078.90 | 2025-02-20 09:55:00 | 2071.03 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-04 09:55:00 | 1956.00 | 2025-03-04 10:05:00 | 1948.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-07 09:30:00 | 2007.30 | 2025-03-07 09:50:00 | 1995.02 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-03-07 09:30:00 | 2007.30 | 2025-03-07 10:25:00 | 2007.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 10:55:00 | 1939.95 | 2025-03-12 11:35:00 | 1930.92 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-03-12 10:55:00 | 1939.95 | 2025-03-12 14:45:00 | 1933.75 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-03-17 09:35:00 | 1961.20 | 2025-03-17 09:50:00 | 1954.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-20 10:40:00 | 2088.45 | 2025-03-20 10:50:00 | 2097.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-23 10:25:00 | 2258.90 | 2025-04-23 10:30:00 | 2268.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-29 10:45:00 | 2283.00 | 2025-04-29 10:50:00 | 2296.18 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-04-29 10:45:00 | 2283.00 | 2025-04-29 15:20:00 | 2322.40 | TARGET_HIT | 0.50 | 1.73% |
