# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-11-06 15:25:00 (9238 bars)
- **Last close:** 3050.00
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 6 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 33
- **Target hits / Stop hits / Partials:** 6 / 33 / 16
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 6.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 16 | 50.0% | 5 | 16 | 11 | 0.21% | 6.6% |
| BUY @ 2nd Alert (retest1) | 32 | 16 | 50.0% | 5 | 16 | 11 | 0.21% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 6 | 26.1% | 1 | 17 | 5 | -0.01% | -0.3% |
| SELL @ 2nd Alert (retest1) | 23 | 6 | 26.1% | 1 | 17 | 5 | -0.01% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 55 | 22 | 40.0% | 6 | 33 | 16 | 0.11% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:00:00 | 2977.00 | 2987.19 | 0.00 | ORB-short ORB[2981.00,3012.00] vol=1.8x ATR=10.57 |
| Stop hit — per-position SL triggered | 2025-05-22 10:20:00 | 2987.57 | 2986.58 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 3182.20 | 3161.83 | 0.00 | ORB-long ORB[3133.50,3170.90] vol=1.8x ATR=15.43 |
| Stop hit — per-position SL triggered | 2025-05-29 09:45:00 | 3166.77 | 3165.37 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 3231.10 | 3177.87 | 0.00 | ORB-long ORB[3101.70,3149.90] vol=2.3x ATR=12.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:10:00 | 3250.57 | 3195.24 | 0.00 | T1 1.5R @ 3250.57 |
| Stop hit — per-position SL triggered | 2025-06-02 11:45:00 | 3231.10 | 3213.71 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:00:00 | 3212.70 | 3208.82 | 0.00 | ORB-long ORB[3180.10,3206.80] vol=11.3x ATR=12.94 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 3199.76 | 3208.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 11:15:00 | 3239.30 | 3222.54 | 0.00 | ORB-long ORB[3170.70,3217.20] vol=2.5x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 11:35:00 | 3251.97 | 3228.36 | 0.00 | T1 1.5R @ 3251.97 |
| Stop hit — per-position SL triggered | 2025-06-18 12:50:00 | 3239.30 | 3231.36 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 3204.30 | 3219.80 | 0.00 | ORB-short ORB[3220.00,3248.00] vol=1.8x ATR=8.35 |
| Stop hit — per-position SL triggered | 2025-06-19 10:45:00 | 3212.65 | 3219.55 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:00:00 | 3332.70 | 3316.67 | 0.00 | ORB-long ORB[3289.50,3324.40] vol=2.4x ATR=12.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:10:00 | 3350.91 | 3324.08 | 0.00 | T1 1.5R @ 3350.91 |
| Target hit | 2025-06-24 11:35:00 | 3416.40 | 3439.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:55:00 | 3359.50 | 3348.87 | 0.00 | ORB-long ORB[3323.90,3350.60] vol=1.7x ATR=8.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:00:00 | 3372.68 | 3366.56 | 0.00 | T1 1.5R @ 3372.68 |
| Target hit | 2025-06-27 10:25:00 | 3360.00 | 3367.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 3392.00 | 3379.96 | 0.00 | ORB-long ORB[3365.80,3391.30] vol=1.6x ATR=10.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:35:00 | 3408.26 | 3386.63 | 0.00 | T1 1.5R @ 3408.26 |
| Stop hit — per-position SL triggered | 2025-07-03 10:40:00 | 3392.00 | 3393.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:50:00 | 3366.30 | 3334.99 | 0.00 | ORB-long ORB[3299.40,3346.40] vol=2.8x ATR=10.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 12:45:00 | 3382.37 | 3352.17 | 0.00 | T1 1.5R @ 3382.37 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 3366.30 | 3360.15 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 10:05:00 | 3360.00 | 3349.61 | 0.00 | ORB-long ORB[3328.00,3354.10] vol=2.2x ATR=8.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 3372.46 | 3356.93 | 0.00 | T1 1.5R @ 3372.46 |
| Stop hit — per-position SL triggered | 2025-07-11 10:55:00 | 3360.00 | 3367.84 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 11:05:00 | 3363.60 | 3388.91 | 0.00 | ORB-short ORB[3375.80,3420.00] vol=1.6x ATR=10.28 |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 3373.88 | 3383.39 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:50:00 | 3394.50 | 3390.08 | 0.00 | ORB-long ORB[3360.00,3390.40] vol=13.4x ATR=9.31 |
| Stop hit — per-position SL triggered | 2025-07-16 09:55:00 | 3385.19 | 3390.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:45:00 | 3453.00 | 3429.49 | 0.00 | ORB-long ORB[3396.00,3430.90] vol=1.5x ATR=12.17 |
| Stop hit — per-position SL triggered | 2025-07-21 09:50:00 | 3440.83 | 3430.68 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:30:00 | 3403.00 | 3434.37 | 0.00 | ORB-short ORB[3454.40,3497.90] vol=2.1x ATR=11.71 |
| Stop hit — per-position SL triggered | 2025-07-25 11:50:00 | 3414.71 | 3425.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:20:00 | 3401.80 | 3391.56 | 0.00 | ORB-long ORB[3359.10,3398.80] vol=2.4x ATR=9.12 |
| Stop hit — per-position SL triggered | 2025-07-30 11:00:00 | 3392.68 | 3393.48 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:30:00 | 3030.20 | 3033.76 | 0.00 | ORB-short ORB[3062.30,3100.00] vol=14.7x ATR=9.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 12:00:00 | 3016.45 | 3033.10 | 0.00 | T1 1.5R @ 3016.45 |
| Target hit | 2025-08-05 15:20:00 | 2993.10 | 3024.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:15:00 | 2856.40 | 2885.60 | 0.00 | ORB-short ORB[2856.70,2899.10] vol=3.9x ATR=9.31 |
| Stop hit — per-position SL triggered | 2025-08-07 12:00:00 | 2865.71 | 2879.69 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:15:00 | 2891.60 | 2872.64 | 0.00 | ORB-long ORB[2838.60,2876.50] vol=1.8x ATR=10.84 |
| Stop hit — per-position SL triggered | 2025-08-14 10:20:00 | 2880.76 | 2874.00 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 11:00:00 | 2865.80 | 2848.11 | 0.00 | ORB-long ORB[2831.20,2855.60] vol=2.4x ATR=5.90 |
| Stop hit — per-position SL triggered | 2025-08-18 11:35:00 | 2859.90 | 2850.93 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 2888.10 | 2867.90 | 0.00 | ORB-long ORB[2836.10,2874.60] vol=2.0x ATR=8.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 09:55:00 | 2901.33 | 2889.38 | 0.00 | T1 1.5R @ 2901.33 |
| Target hit | 2025-08-19 15:20:00 | 2938.20 | 2923.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-08-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:25:00 | 2886.80 | 2873.76 | 0.00 | ORB-long ORB[2843.10,2882.60] vol=1.7x ATR=8.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:35:00 | 2900.07 | 2876.29 | 0.00 | T1 1.5R @ 2900.07 |
| Stop hit — per-position SL triggered | 2025-08-29 15:00:00 | 2886.80 | 2888.25 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:00:00 | 2931.90 | 2906.15 | 0.00 | ORB-long ORB[2853.10,2890.60] vol=4.0x ATR=10.52 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 2921.38 | 2915.33 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:45:00 | 3002.50 | 3023.46 | 0.00 | ORB-short ORB[3015.00,3045.60] vol=2.1x ATR=7.15 |
| Stop hit — per-position SL triggered | 2025-09-11 10:50:00 | 3009.65 | 3022.82 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:55:00 | 3057.00 | 3048.35 | 0.00 | ORB-long ORB[3029.70,3050.00] vol=2.4x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-09-16 10:25:00 | 3050.36 | 3051.39 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3030.30 | 3045.05 | 0.00 | ORB-short ORB[3039.00,3056.70] vol=2.5x ATR=8.46 |
| Stop hit — per-position SL triggered | 2025-09-17 10:00:00 | 3038.76 | 3043.07 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:25:00 | 3023.00 | 3037.48 | 0.00 | ORB-short ORB[3036.30,3066.80] vol=3.6x ATR=7.59 |
| Stop hit — per-position SL triggered | 2025-09-18 10:40:00 | 3030.59 | 3034.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:15:00 | 3028.00 | 3031.15 | 0.00 | ORB-short ORB[3030.90,3054.00] vol=6.9x ATR=5.08 |
| Stop hit — per-position SL triggered | 2025-09-23 13:05:00 | 3033.08 | 3029.86 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:05:00 | 2950.20 | 2966.12 | 0.00 | ORB-short ORB[2959.10,2984.60] vol=1.6x ATR=8.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:30:00 | 2937.94 | 2962.27 | 0.00 | T1 1.5R @ 2937.94 |
| Stop hit — per-position SL triggered | 2025-09-29 11:35:00 | 2950.20 | 2960.56 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 2988.50 | 2995.11 | 0.00 | ORB-short ORB[2989.00,3009.90] vol=2.1x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:05:00 | 2981.92 | 2993.18 | 0.00 | T1 1.5R @ 2981.92 |
| Stop hit — per-position SL triggered | 2025-10-01 11:35:00 | 2988.50 | 2991.75 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 10:55:00 | 2971.50 | 2986.40 | 0.00 | ORB-short ORB[2972.60,2998.30] vol=2.2x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 11:20:00 | 2962.73 | 2984.29 | 0.00 | T1 1.5R @ 2962.73 |
| Stop hit — per-position SL triggered | 2025-10-10 13:35:00 | 2971.50 | 2976.64 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:05:00 | 2929.70 | 2945.58 | 0.00 | ORB-short ORB[2945.40,2987.90] vol=5.1x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:35:00 | 2915.76 | 2941.46 | 0.00 | T1 1.5R @ 2915.76 |
| Stop hit — per-position SL triggered | 2025-10-13 12:25:00 | 2929.70 | 2930.89 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:10:00 | 2925.60 | 2936.11 | 0.00 | ORB-short ORB[2938.60,2958.30] vol=4.6x ATR=5.73 |
| Stop hit — per-position SL triggered | 2025-10-17 10:55:00 | 2931.33 | 2933.68 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:50:00 | 2920.40 | 2923.33 | 0.00 | ORB-short ORB[2924.60,2957.70] vol=6.0x ATR=7.08 |
| Stop hit — per-position SL triggered | 2025-10-20 11:25:00 | 2927.48 | 2923.52 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:25:00 | 2935.20 | 2955.91 | 0.00 | ORB-short ORB[2944.00,2978.60] vol=4.8x ATR=8.69 |
| Stop hit — per-position SL triggered | 2025-10-23 10:40:00 | 2943.89 | 2954.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:50:00 | 3005.50 | 2988.89 | 0.00 | ORB-long ORB[2972.60,2992.60] vol=4.4x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:05:00 | 3016.88 | 2994.83 | 0.00 | T1 1.5R @ 3016.88 |
| Target hit | 2025-10-27 15:20:00 | 3016.10 | 3011.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-10-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:20:00 | 3043.90 | 3036.93 | 0.00 | ORB-long ORB[3022.00,3039.20] vol=8.1x ATR=5.46 |
| Stop hit — per-position SL triggered | 2025-10-28 10:30:00 | 3038.44 | 3037.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:15:00 | 3015.30 | 3025.74 | 0.00 | ORB-short ORB[3024.60,3050.00] vol=1.9x ATR=4.41 |
| Stop hit — per-position SL triggered | 2025-10-29 11:20:00 | 3019.71 | 3025.55 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:05:00 | 3091.00 | 3073.71 | 0.00 | ORB-long ORB[3036.80,3075.20] vol=1.6x ATR=9.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:25:00 | 3104.55 | 3090.59 | 0.00 | T1 1.5R @ 3104.55 |
| Target hit | 2025-10-31 11:05:00 | 3095.70 | 3096.78 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-22 10:00:00 | 2977.00 | 2025-05-22 10:20:00 | 2987.57 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-29 09:30:00 | 3182.20 | 2025-05-29 09:45:00 | 3166.77 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-06-02 11:00:00 | 3231.10 | 2025-06-02 11:10:00 | 3250.57 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-06-02 11:00:00 | 3231.10 | 2025-06-02 11:45:00 | 3231.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 10:00:00 | 3212.70 | 2025-06-03 10:05:00 | 3199.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-18 11:15:00 | 3239.30 | 2025-06-18 11:35:00 | 3251.97 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-18 11:15:00 | 3239.30 | 2025-06-18 12:50:00 | 3239.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 10:35:00 | 3204.30 | 2025-06-19 10:45:00 | 3212.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-24 10:00:00 | 3332.70 | 2025-06-24 10:10:00 | 3350.91 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-24 10:00:00 | 3332.70 | 2025-06-24 11:35:00 | 3416.40 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2025-06-27 09:55:00 | 3359.50 | 2025-06-27 10:00:00 | 3372.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-27 09:55:00 | 3359.50 | 2025-06-27 10:25:00 | 3360.00 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-07-03 10:00:00 | 3392.00 | 2025-07-03 10:35:00 | 3408.26 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-03 10:00:00 | 3392.00 | 2025-07-03 10:40:00 | 3392.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 10:50:00 | 3366.30 | 2025-07-09 12:45:00 | 3382.37 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-09 10:50:00 | 3366.30 | 2025-07-09 14:15:00 | 3366.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-11 10:05:00 | 3360.00 | 2025-07-11 10:15:00 | 3372.46 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-11 10:05:00 | 3360.00 | 2025-07-11 10:55:00 | 3360.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-14 11:05:00 | 3363.60 | 2025-07-14 13:15:00 | 3373.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-16 09:50:00 | 3394.50 | 2025-07-16 09:55:00 | 3385.19 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-21 09:45:00 | 3453.00 | 2025-07-21 09:50:00 | 3440.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-25 10:30:00 | 3403.00 | 2025-07-25 11:50:00 | 3414.71 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-30 10:20:00 | 3401.80 | 2025-07-30 11:00:00 | 3392.68 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-05 10:30:00 | 3030.20 | 2025-08-05 12:00:00 | 3016.45 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-05 10:30:00 | 3030.20 | 2025-08-05 15:20:00 | 2993.10 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2025-08-07 11:15:00 | 2856.40 | 2025-08-07 12:00:00 | 2865.71 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-14 10:15:00 | 2891.60 | 2025-08-14 10:20:00 | 2880.76 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-08-18 11:00:00 | 2865.80 | 2025-08-18 11:35:00 | 2859.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-19 09:50:00 | 2888.10 | 2025-08-19 09:55:00 | 2901.33 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-19 09:50:00 | 2888.10 | 2025-08-19 15:20:00 | 2938.20 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2025-08-29 10:25:00 | 2886.80 | 2025-08-29 10:35:00 | 2900.07 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-29 10:25:00 | 2886.80 | 2025-08-29 15:00:00 | 2886.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 10:00:00 | 2931.90 | 2025-09-08 10:30:00 | 2921.38 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-11 10:45:00 | 3002.50 | 2025-09-11 10:50:00 | 3009.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-16 09:55:00 | 3057.00 | 2025-09-16 10:25:00 | 3050.36 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-17 09:45:00 | 3030.30 | 2025-09-17 10:00:00 | 3038.76 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-18 10:25:00 | 3023.00 | 2025-09-18 10:40:00 | 3030.59 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-23 11:15:00 | 3028.00 | 2025-09-23 13:05:00 | 3033.08 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-29 11:05:00 | 2950.20 | 2025-09-29 11:30:00 | 2937.94 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-09-29 11:05:00 | 2950.20 | 2025-09-29 11:35:00 | 2950.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-01 10:45:00 | 2988.50 | 2025-10-01 11:05:00 | 2981.92 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-10-01 10:45:00 | 2988.50 | 2025-10-01 11:35:00 | 2988.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-10 10:55:00 | 2971.50 | 2025-10-10 11:20:00 | 2962.73 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-10 10:55:00 | 2971.50 | 2025-10-10 13:35:00 | 2971.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 10:05:00 | 2929.70 | 2025-10-13 10:35:00 | 2915.76 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-13 10:05:00 | 2929.70 | 2025-10-13 12:25:00 | 2929.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 10:10:00 | 2925.60 | 2025-10-17 10:55:00 | 2931.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-20 10:50:00 | 2920.40 | 2025-10-20 11:25:00 | 2927.48 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-23 10:25:00 | 2935.20 | 2025-10-23 10:40:00 | 2943.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-27 10:50:00 | 3005.50 | 2025-10-27 11:05:00 | 3016.88 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-27 10:50:00 | 3005.50 | 2025-10-27 15:20:00 | 3016.10 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-28 10:20:00 | 3043.90 | 2025-10-28 10:30:00 | 3038.44 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-29 11:15:00 | 3015.30 | 2025-10-29 11:20:00 | 3019.71 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-31 10:05:00 | 3091.00 | 2025-10-31 10:25:00 | 3104.55 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-31 10:05:00 | 3091.00 | 2025-10-31 11:05:00 | 3095.70 | TARGET_HIT | 0.50 | 0.15% |
