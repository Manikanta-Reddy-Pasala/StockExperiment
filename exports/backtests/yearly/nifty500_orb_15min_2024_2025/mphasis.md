# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 2214.50
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 55
- **Target hits / Stop hits / Partials:** 7 / 55 / 19
- **Avg / median % per leg:** 0.02% / -0.23%
- **Sum % (uncompounded):** 1.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 20 | 39.2% | 6 | 31 | 14 | 0.04% | 2.3% |
| BUY @ 2nd Alert (retest1) | 51 | 20 | 39.2% | 6 | 31 | 14 | 0.04% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 6 | 20.0% | 1 | 24 | 5 | -0.03% | -0.9% |
| SELL @ 2nd Alert (retest1) | 30 | 6 | 20.0% | 1 | 24 | 5 | -0.03% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 26 | 32.1% | 7 | 55 | 19 | 0.02% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 2333.00 | 2327.16 | 0.00 | ORB-long ORB[2312.30,2332.20] vol=2.0x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 12:50:00 | 2343.36 | 2331.33 | 0.00 | T1 1.5R @ 2343.36 |
| Target hit | 2024-05-21 15:20:00 | 2372.85 | 2353.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:45:00 | 2390.00 | 2380.56 | 0.00 | ORB-long ORB[2365.00,2380.35] vol=2.0x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:55:00 | 2400.42 | 2385.20 | 0.00 | T1 1.5R @ 2400.42 |
| Target hit | 2024-05-23 13:10:00 | 2396.80 | 2400.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 2419.00 | 2407.68 | 0.00 | ORB-long ORB[2383.40,2415.00] vol=2.4x ATR=9.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:05:00 | 2432.80 | 2417.63 | 0.00 | T1 1.5R @ 2432.80 |
| Target hit | 2024-05-27 14:50:00 | 2441.85 | 2450.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2024-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:00:00 | 2330.20 | 2344.46 | 0.00 | ORB-short ORB[2343.50,2362.90] vol=1.9x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-05-30 11:10:00 | 2336.70 | 2343.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 2436.60 | 2425.45 | 0.00 | ORB-long ORB[2409.05,2430.00] vol=3.5x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:35:00 | 2450.53 | 2433.81 | 0.00 | T1 1.5R @ 2450.53 |
| Stop hit — per-position SL triggered | 2024-06-13 11:20:00 | 2436.60 | 2436.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:45:00 | 2410.10 | 2402.46 | 0.00 | ORB-long ORB[2386.05,2409.95] vol=1.5x ATR=8.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:15:00 | 2423.22 | 2404.30 | 0.00 | T1 1.5R @ 2423.22 |
| Stop hit — per-position SL triggered | 2024-06-24 11:40:00 | 2410.10 | 2405.27 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 2380.10 | 2388.08 | 0.00 | ORB-short ORB[2385.00,2405.65] vol=2.2x ATR=4.97 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 2385.07 | 2387.89 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:45:00 | 2430.60 | 2422.19 | 0.00 | ORB-long ORB[2411.20,2421.85] vol=3.1x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:25:00 | 2438.09 | 2425.92 | 0.00 | T1 1.5R @ 2438.09 |
| Stop hit — per-position SL triggered | 2024-06-26 11:40:00 | 2430.60 | 2426.65 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 11:05:00 | 2427.00 | 2413.87 | 0.00 | ORB-long ORB[2407.20,2418.45] vol=1.6x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:20:00 | 2436.30 | 2417.80 | 0.00 | T1 1.5R @ 2436.30 |
| Target hit | 2024-06-27 13:35:00 | 2432.70 | 2436.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 2485.90 | 2477.30 | 0.00 | ORB-long ORB[2457.80,2484.30] vol=1.9x ATR=7.42 |
| Stop hit — per-position SL triggered | 2024-07-01 09:50:00 | 2478.48 | 2479.19 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:50:00 | 2608.75 | 2593.02 | 0.00 | ORB-long ORB[2564.00,2597.70] vol=2.0x ATR=9.60 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 2599.15 | 2603.10 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 2623.05 | 2629.29 | 0.00 | ORB-short ORB[2624.55,2652.00] vol=4.5x ATR=8.65 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 2631.70 | 2629.21 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 2520.90 | 2535.07 | 0.00 | ORB-short ORB[2536.25,2560.75] vol=2.3x ATR=9.20 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 2530.10 | 2527.70 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:10:00 | 2566.90 | 2559.43 | 0.00 | ORB-long ORB[2538.60,2565.00] vol=1.6x ATR=6.08 |
| Stop hit — per-position SL triggered | 2024-07-11 11:45:00 | 2560.82 | 2559.56 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 2762.65 | 2735.32 | 0.00 | ORB-long ORB[2704.00,2744.95] vol=1.6x ATR=11.75 |
| Stop hit — per-position SL triggered | 2024-07-15 12:00:00 | 2750.90 | 2743.53 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:55:00 | 2939.90 | 2915.46 | 0.00 | ORB-long ORB[2902.00,2919.90] vol=2.0x ATR=7.58 |
| Stop hit — per-position SL triggered | 2024-07-31 10:30:00 | 2932.32 | 2925.72 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:50:00 | 2668.00 | 2682.01 | 0.00 | ORB-short ORB[2675.00,2713.65] vol=1.5x ATR=9.14 |
| Stop hit — per-position SL triggered | 2024-08-12 11:10:00 | 2677.14 | 2680.18 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 2988.55 | 2959.62 | 0.00 | ORB-long ORB[2922.30,2953.95] vol=2.9x ATR=11.90 |
| Stop hit — per-position SL triggered | 2024-08-19 09:40:00 | 2976.65 | 2964.90 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 3022.25 | 3010.22 | 0.00 | ORB-long ORB[2985.00,3014.55] vol=1.6x ATR=9.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:55:00 | 3036.85 | 3019.24 | 0.00 | T1 1.5R @ 3036.85 |
| Stop hit — per-position SL triggered | 2024-08-20 11:35:00 | 3022.25 | 3020.22 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:20:00 | 3075.25 | 3042.09 | 0.00 | ORB-long ORB[3025.05,3061.05] vol=1.6x ATR=11.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:35:00 | 3092.55 | 3073.79 | 0.00 | T1 1.5R @ 3092.55 |
| Target hit | 2024-08-28 14:50:00 | 3090.00 | 3093.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 3098.95 | 3080.74 | 0.00 | ORB-long ORB[3052.00,3080.70] vol=1.7x ATR=10.12 |
| Stop hit — per-position SL triggered | 2024-08-29 11:20:00 | 3088.83 | 3081.04 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:30:00 | 3016.60 | 3028.73 | 0.00 | ORB-short ORB[3025.10,3059.00] vol=4.2x ATR=10.26 |
| Stop hit — per-position SL triggered | 2024-09-10 10:40:00 | 3026.86 | 3028.65 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:00:00 | 3101.65 | 3085.27 | 0.00 | ORB-long ORB[3065.00,3094.90] vol=1.9x ATR=8.40 |
| Stop hit — per-position SL triggered | 2024-09-11 11:20:00 | 3093.25 | 3088.24 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:25:00 | 3140.00 | 3122.57 | 0.00 | ORB-long ORB[3103.85,3124.90] vol=1.8x ATR=8.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:30:00 | 3152.24 | 3128.11 | 0.00 | T1 1.5R @ 3152.24 |
| Target hit | 2024-09-13 12:05:00 | 3146.30 | 3147.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:55:00 | 3018.50 | 3037.96 | 0.00 | ORB-short ORB[3031.75,3072.00] vol=1.9x ATR=9.72 |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 3028.22 | 3035.51 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 3064.40 | 3045.53 | 0.00 | ORB-long ORB[3031.00,3056.40] vol=2.6x ATR=8.52 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 3055.88 | 3050.00 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:35:00 | 3053.00 | 3065.06 | 0.00 | ORB-short ORB[3054.25,3080.80] vol=1.6x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-09-25 11:00:00 | 3062.30 | 3063.90 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:00:00 | 3007.50 | 3020.38 | 0.00 | ORB-short ORB[3020.00,3050.00] vol=11.5x ATR=9.49 |
| Stop hit — per-position SL triggered | 2024-10-01 11:25:00 | 3016.99 | 3018.93 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:55:00 | 2956.20 | 2923.47 | 0.00 | ORB-long ORB[2889.05,2930.00] vol=4.2x ATR=13.72 |
| Stop hit — per-position SL triggered | 2024-10-04 10:10:00 | 2942.48 | 2928.19 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:55:00 | 2911.25 | 2932.44 | 0.00 | ORB-short ORB[2923.65,2957.00] vol=3.4x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:40:00 | 2897.31 | 2924.99 | 0.00 | T1 1.5R @ 2897.31 |
| Stop hit — per-position SL triggered | 2024-10-09 12:10:00 | 2911.25 | 2913.15 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 10:55:00 | 3020.35 | 3014.11 | 0.00 | ORB-long ORB[2984.95,3012.30] vol=1.5x ATR=13.67 |
| Stop hit — per-position SL triggered | 2024-10-22 14:05:00 | 3006.68 | 3017.88 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 3069.50 | 3081.46 | 0.00 | ORB-short ORB[3076.85,3119.55] vol=1.6x ATR=11.62 |
| Stop hit — per-position SL triggered | 2024-10-25 11:00:00 | 3081.12 | 3081.30 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 11:10:00 | 2996.15 | 3038.86 | 0.00 | ORB-short ORB[3049.20,3084.85] vol=1.8x ATR=13.14 |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 3009.29 | 3038.05 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:30:00 | 3051.65 | 3034.68 | 0.00 | ORB-long ORB[3007.10,3030.00] vol=3.8x ATR=9.71 |
| Stop hit — per-position SL triggered | 2024-10-30 09:35:00 | 3041.94 | 3036.48 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:50:00 | 2842.00 | 2860.98 | 0.00 | ORB-short ORB[2866.00,2884.75] vol=2.1x ATR=7.61 |
| Stop hit — per-position SL triggered | 2024-11-05 10:55:00 | 2849.61 | 2860.64 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:30:00 | 2865.00 | 2858.14 | 0.00 | ORB-long ORB[2839.15,2862.60] vol=2.8x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:40:00 | 2876.02 | 2865.05 | 0.00 | T1 1.5R @ 2876.02 |
| Stop hit — per-position SL triggered | 2024-11-08 09:55:00 | 2865.00 | 2868.41 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:55:00 | 2896.05 | 2877.15 | 0.00 | ORB-long ORB[2861.00,2892.00] vol=1.5x ATR=10.12 |
| Stop hit — per-position SL triggered | 2024-11-12 10:05:00 | 2885.93 | 2878.88 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:00:00 | 3023.25 | 3018.83 | 0.00 | ORB-long ORB[2996.00,3022.35] vol=8.2x ATR=8.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:15:00 | 3035.38 | 3019.62 | 0.00 | T1 1.5R @ 3035.38 |
| Stop hit — per-position SL triggered | 2024-11-27 10:45:00 | 3023.25 | 3020.77 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:10:00 | 2962.45 | 2988.16 | 0.00 | ORB-short ORB[2985.40,3026.85] vol=3.9x ATR=9.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 12:20:00 | 2947.92 | 2981.19 | 0.00 | T1 1.5R @ 2947.92 |
| Stop hit — per-position SL triggered | 2024-11-28 12:45:00 | 2962.45 | 2978.44 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 3011.55 | 2993.52 | 0.00 | ORB-long ORB[2953.55,2991.70] vol=2.5x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-12-02 11:10:00 | 3004.71 | 2994.04 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 11:15:00 | 3174.95 | 3158.12 | 0.00 | ORB-long ORB[3142.50,3170.95] vol=3.1x ATR=10.32 |
| Stop hit — per-position SL triggered | 2024-12-13 11:30:00 | 3164.63 | 3159.51 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 2888.20 | 2903.33 | 0.00 | ORB-short ORB[2913.00,2951.40] vol=1.7x ATR=7.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:35:00 | 2876.95 | 2898.50 | 0.00 | T1 1.5R @ 2876.95 |
| Stop hit — per-position SL triggered | 2024-12-26 11:40:00 | 2888.20 | 2897.72 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 2893.25 | 2909.81 | 0.00 | ORB-short ORB[2903.05,2945.00] vol=1.8x ATR=6.66 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 2899.91 | 2909.52 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 2861.90 | 2875.08 | 0.00 | ORB-short ORB[2867.75,2887.40] vol=3.1x ATR=7.90 |
| Stop hit — per-position SL triggered | 2024-12-30 09:50:00 | 2869.80 | 2868.04 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 2888.15 | 2877.52 | 0.00 | ORB-long ORB[2847.30,2872.95] vol=1.7x ATR=8.79 |
| Stop hit — per-position SL triggered | 2025-01-02 12:10:00 | 2879.36 | 2884.50 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 2867.10 | 2879.28 | 0.00 | ORB-short ORB[2876.80,2909.00] vol=2.7x ATR=9.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:20:00 | 2853.37 | 2876.20 | 0.00 | T1 1.5R @ 2853.37 |
| Stop hit — per-position SL triggered | 2025-01-03 11:40:00 | 2867.10 | 2865.35 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:05:00 | 2896.35 | 2908.07 | 0.00 | ORB-short ORB[2897.65,2930.00] vol=2.0x ATR=10.48 |
| Stop hit — per-position SL triggered | 2025-01-07 10:10:00 | 2906.83 | 2908.00 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 2890.00 | 2891.79 | 0.00 | ORB-short ORB[2892.50,2920.45] vol=1.8x ATR=8.70 |
| Stop hit — per-position SL triggered | 2025-01-08 11:25:00 | 2898.70 | 2892.50 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:35:00 | 2939.90 | 2920.55 | 0.00 | ORB-long ORB[2900.00,2924.10] vol=2.5x ATR=9.99 |
| Stop hit — per-position SL triggered | 2025-01-09 10:40:00 | 2929.91 | 2922.11 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 2948.50 | 2943.41 | 0.00 | ORB-long ORB[2900.10,2937.00] vol=1.9x ATR=9.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:20:00 | 2962.76 | 2944.78 | 0.00 | T1 1.5R @ 2962.76 |
| Stop hit — per-position SL triggered | 2025-01-29 12:40:00 | 2948.50 | 2955.81 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:20:00 | 2970.55 | 2979.98 | 0.00 | ORB-short ORB[2982.50,3011.20] vol=6.7x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:30:00 | 2953.91 | 2977.23 | 0.00 | T1 1.5R @ 2953.91 |
| Target hit | 2025-01-30 15:20:00 | 2875.30 | 2895.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 2864.25 | 2880.24 | 0.00 | ORB-short ORB[2865.40,2900.00] vol=2.1x ATR=10.08 |
| Stop hit — per-position SL triggered | 2025-01-31 11:45:00 | 2874.33 | 2875.89 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:35:00 | 2870.90 | 2837.52 | 0.00 | ORB-long ORB[2817.80,2837.95] vol=1.5x ATR=9.30 |
| Stop hit — per-position SL triggered | 2025-02-07 11:05:00 | 2861.60 | 2845.55 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-02-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 10:10:00 | 2598.90 | 2591.93 | 0.00 | ORB-long ORB[2552.30,2584.95] vol=1.6x ATR=10.89 |
| Stop hit — per-position SL triggered | 2025-02-19 10:30:00 | 2588.01 | 2593.37 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:45:00 | 2622.65 | 2608.34 | 0.00 | ORB-long ORB[2589.50,2610.85] vol=1.5x ATR=8.67 |
| Stop hit — per-position SL triggered | 2025-02-20 10:55:00 | 2613.98 | 2609.72 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:30:00 | 2588.70 | 2608.94 | 0.00 | ORB-short ORB[2618.00,2650.95] vol=1.8x ATR=9.31 |
| Stop hit — per-position SL triggered | 2025-02-21 11:25:00 | 2598.01 | 2603.02 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:45:00 | 2452.05 | 2474.72 | 0.00 | ORB-short ORB[2464.60,2489.90] vol=2.1x ATR=8.36 |
| Stop hit — per-position SL triggered | 2025-02-25 12:20:00 | 2460.41 | 2464.49 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 2356.75 | 2340.15 | 0.00 | ORB-long ORB[2321.00,2342.30] vol=3.0x ATR=9.15 |
| Stop hit — per-position SL triggered | 2025-03-06 09:45:00 | 2347.60 | 2344.91 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 11:00:00 | 2191.35 | 2211.76 | 0.00 | ORB-short ORB[2198.10,2226.60] vol=1.8x ATR=8.30 |
| Stop hit — per-position SL triggered | 2025-03-17 12:00:00 | 2199.65 | 2202.47 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:45:00 | 2548.30 | 2539.37 | 0.00 | ORB-long ORB[2518.65,2542.15] vol=4.5x ATR=9.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:55:00 | 2562.81 | 2541.86 | 0.00 | T1 1.5R @ 2562.81 |
| Stop hit — per-position SL triggered | 2025-03-26 10:20:00 | 2548.30 | 2545.54 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:55:00 | 2497.25 | 2463.33 | 0.00 | ORB-long ORB[2437.35,2471.65] vol=1.6x ATR=11.28 |
| Stop hit — per-position SL triggered | 2025-04-02 11:10:00 | 2485.97 | 2466.15 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:35:00 | 2229.10 | 2206.97 | 0.00 | ORB-long ORB[2185.00,2215.00] vol=2.5x ATR=12.66 |
| Stop hit — per-position SL triggered | 2025-04-15 10:00:00 | 2216.44 | 2211.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 09:30:00 | 2333.00 | 2024-05-21 12:50:00 | 2343.36 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-21 09:30:00 | 2333.00 | 2024-05-21 15:20:00 | 2372.85 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2024-05-23 10:45:00 | 2390.00 | 2024-05-23 10:55:00 | 2400.42 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-23 10:45:00 | 2390.00 | 2024-05-23 13:10:00 | 2396.80 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-05-27 09:35:00 | 2419.00 | 2024-05-27 10:05:00 | 2432.80 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-05-27 09:35:00 | 2419.00 | 2024-05-27 14:50:00 | 2441.85 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-05-30 11:00:00 | 2330.20 | 2024-05-30 11:10:00 | 2336.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-13 09:35:00 | 2436.60 | 2024-06-13 10:35:00 | 2450.53 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-13 09:35:00 | 2436.60 | 2024-06-13 11:20:00 | 2436.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 10:45:00 | 2410.10 | 2024-06-24 11:15:00 | 2423.22 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-24 10:45:00 | 2410.10 | 2024-06-24 11:40:00 | 2410.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:15:00 | 2380.10 | 2024-06-25 11:20:00 | 2385.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-26 10:45:00 | 2430.60 | 2024-06-26 11:25:00 | 2438.09 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-06-26 10:45:00 | 2430.60 | 2024-06-26 11:40:00 | 2430.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 11:05:00 | 2427.00 | 2024-06-27 11:20:00 | 2436.30 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-27 11:05:00 | 2427.00 | 2024-06-27 13:35:00 | 2432.70 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-07-01 09:30:00 | 2485.90 | 2024-07-01 09:50:00 | 2478.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-04 09:50:00 | 2608.75 | 2024-07-04 10:35:00 | 2599.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-08 09:40:00 | 2623.05 | 2024-07-08 09:45:00 | 2631.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-10 10:10:00 | 2520.90 | 2024-07-10 10:40:00 | 2530.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-11 11:10:00 | 2566.90 | 2024-07-11 11:45:00 | 2560.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-15 10:50:00 | 2762.65 | 2024-07-15 12:00:00 | 2750.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-31 09:55:00 | 2939.90 | 2024-07-31 10:30:00 | 2932.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-12 10:50:00 | 2668.00 | 2024-08-12 11:10:00 | 2677.14 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-19 09:35:00 | 2988.55 | 2024-08-19 09:40:00 | 2976.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-20 09:30:00 | 3022.25 | 2024-08-20 10:55:00 | 3036.85 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-20 09:30:00 | 3022.25 | 2024-08-20 11:35:00 | 3022.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 10:20:00 | 3075.25 | 2024-08-28 10:35:00 | 3092.55 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-28 10:20:00 | 3075.25 | 2024-08-28 14:50:00 | 3090.00 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-29 11:10:00 | 3098.95 | 2024-08-29 11:20:00 | 3088.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-10 10:30:00 | 3016.60 | 2024-09-10 10:40:00 | 3026.86 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-11 11:00:00 | 3101.65 | 2024-09-11 11:20:00 | 3093.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-13 10:25:00 | 3140.00 | 2024-09-13 10:30:00 | 3152.24 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-13 10:25:00 | 3140.00 | 2024-09-13 12:05:00 | 3146.30 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-09-23 10:55:00 | 3018.50 | 2024-09-23 11:15:00 | 3028.22 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-24 10:55:00 | 3064.40 | 2024-09-24 11:15:00 | 3055.88 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-25 10:35:00 | 3053.00 | 2024-09-25 11:00:00 | 3062.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-01 11:00:00 | 3007.50 | 2024-10-01 11:25:00 | 3016.99 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-04 09:55:00 | 2956.20 | 2024-10-04 10:10:00 | 2942.48 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-09 10:55:00 | 2911.25 | 2024-10-09 11:40:00 | 2897.31 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-09 10:55:00 | 2911.25 | 2024-10-09 12:10:00 | 2911.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-22 10:55:00 | 3020.35 | 2024-10-22 14:05:00 | 3006.68 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-25 10:55:00 | 3069.50 | 2024-10-25 11:00:00 | 3081.12 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-28 11:10:00 | 2996.15 | 2024-10-28 11:15:00 | 3009.29 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-30 09:30:00 | 3051.65 | 2024-10-30 09:35:00 | 3041.94 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-05 10:50:00 | 2842.00 | 2024-11-05 10:55:00 | 2849.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-08 09:30:00 | 2865.00 | 2024-11-08 09:40:00 | 2876.02 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-11-08 09:30:00 | 2865.00 | 2024-11-08 09:55:00 | 2865.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 09:55:00 | 2896.05 | 2024-11-12 10:05:00 | 2885.93 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-27 10:00:00 | 3023.25 | 2024-11-27 10:15:00 | 3035.38 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-27 10:00:00 | 3023.25 | 2024-11-27 10:45:00 | 3023.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 11:10:00 | 2962.45 | 2024-11-28 12:20:00 | 2947.92 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-11-28 11:10:00 | 2962.45 | 2024-11-28 12:45:00 | 2962.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 11:00:00 | 3011.55 | 2024-12-02 11:10:00 | 3004.71 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-13 11:15:00 | 3174.95 | 2024-12-13 11:30:00 | 3164.63 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-26 10:50:00 | 2888.20 | 2024-12-26 11:35:00 | 2876.95 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-26 10:50:00 | 2888.20 | 2024-12-26 11:40:00 | 2888.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 11:05:00 | 2893.25 | 2024-12-27 11:10:00 | 2899.91 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-30 09:30:00 | 2861.90 | 2024-12-30 09:50:00 | 2869.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-02 10:55:00 | 2888.15 | 2025-01-02 12:10:00 | 2879.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-03 10:00:00 | 2867.10 | 2025-01-03 10:20:00 | 2853.37 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-03 10:00:00 | 2867.10 | 2025-01-03 11:40:00 | 2867.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 10:05:00 | 2896.35 | 2025-01-07 10:10:00 | 2906.83 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-08 11:15:00 | 2890.00 | 2025-01-08 11:25:00 | 2898.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-09 10:35:00 | 2939.90 | 2025-01-09 10:40:00 | 2929.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-29 11:10:00 | 2948.50 | 2025-01-29 11:20:00 | 2962.76 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-29 11:10:00 | 2948.50 | 2025-01-29 12:40:00 | 2948.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 10:20:00 | 2970.55 | 2025-01-30 10:30:00 | 2953.91 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-30 10:20:00 | 2970.55 | 2025-01-30 15:20:00 | 2875.30 | TARGET_HIT | 0.50 | 3.21% |
| SELL | retest1 | 2025-01-31 11:00:00 | 2864.25 | 2025-01-31 11:45:00 | 2874.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-07 10:35:00 | 2870.90 | 2025-02-07 11:05:00 | 2861.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-19 10:10:00 | 2598.90 | 2025-02-19 10:30:00 | 2588.01 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-20 10:45:00 | 2622.65 | 2025-02-20 10:55:00 | 2613.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-21 10:30:00 | 2588.70 | 2025-02-21 11:25:00 | 2598.01 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-25 10:45:00 | 2452.05 | 2025-02-25 12:20:00 | 2460.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-06 09:35:00 | 2356.75 | 2025-03-06 09:45:00 | 2347.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-17 11:00:00 | 2191.35 | 2025-03-17 12:00:00 | 2199.65 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-26 09:45:00 | 2548.30 | 2025-03-26 09:55:00 | 2562.81 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-03-26 09:45:00 | 2548.30 | 2025-03-26 10:20:00 | 2548.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 10:55:00 | 2497.25 | 2025-04-02 11:10:00 | 2485.97 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-15 09:35:00 | 2229.10 | 2025-04-15 10:00:00 | 2216.44 | STOP_HIT | 1.00 | -0.57% |
