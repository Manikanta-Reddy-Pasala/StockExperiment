# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 2600.00
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
| ENTRY1 | 94 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 15 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 79
- **Target hits / Stop hits / Partials:** 15 / 79 / 38
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 10.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 20 | 34.5% | 5 | 38 | 15 | 0.02% | 1.2% |
| BUY @ 2nd Alert (retest1) | 58 | 20 | 34.5% | 5 | 38 | 15 | 0.02% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 74 | 33 | 44.6% | 10 | 41 | 23 | 0.12% | 8.9% |
| SELL @ 2nd Alert (retest1) | 74 | 33 | 44.6% | 10 | 41 | 23 | 0.12% | 8.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 53 | 40.2% | 15 | 79 | 38 | 0.08% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:55:00 | 3160.00 | 3152.82 | 0.00 | ORB-long ORB[3130.00,3159.15] vol=1.9x ATR=6.85 |
| Stop hit — per-position SL triggered | 2023-05-16 10:10:00 | 3153.15 | 3153.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 09:35:00 | 3128.40 | 3135.13 | 0.00 | ORB-short ORB[3132.85,3147.95] vol=1.6x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 09:40:00 | 3119.75 | 3131.86 | 0.00 | T1 1.5R @ 3119.75 |
| Target hit | 2023-05-17 15:20:00 | 3096.15 | 3100.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2023-05-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 11:00:00 | 3113.95 | 3105.37 | 0.00 | ORB-long ORB[3100.00,3112.70] vol=2.5x ATR=4.96 |
| Stop hit — per-position SL triggered | 2023-05-18 11:20:00 | 3108.99 | 3106.81 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:50:00 | 3105.90 | 3114.86 | 0.00 | ORB-short ORB[3111.40,3127.95] vol=1.6x ATR=7.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:00:00 | 3095.06 | 3111.95 | 0.00 | T1 1.5R @ 3095.06 |
| Target hit | 2023-05-19 15:20:00 | 3081.00 | 3090.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2023-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 11:10:00 | 3119.80 | 3109.28 | 0.00 | ORB-long ORB[3085.00,3114.70] vol=2.2x ATR=5.11 |
| Stop hit — per-position SL triggered | 2023-05-23 11:25:00 | 3114.69 | 3109.73 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:50:00 | 3124.15 | 3114.78 | 0.00 | ORB-long ORB[3101.65,3115.00] vol=1.7x ATR=4.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 11:00:00 | 3131.48 | 3116.63 | 0.00 | T1 1.5R @ 3131.48 |
| Stop hit — per-position SL triggered | 2023-05-24 11:15:00 | 3124.15 | 3117.77 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 10:55:00 | 3108.45 | 3098.92 | 0.00 | ORB-long ORB[3091.05,3104.90] vol=1.7x ATR=5.71 |
| Stop hit — per-position SL triggered | 2023-05-25 11:10:00 | 3102.74 | 3099.31 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 11:05:00 | 3111.95 | 3124.01 | 0.00 | ORB-short ORB[3113.00,3136.95] vol=1.5x ATR=5.62 |
| Stop hit — per-position SL triggered | 2023-05-26 11:20:00 | 3117.57 | 3123.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 3234.65 | 3223.41 | 0.00 | ORB-long ORB[3201.15,3230.00] vol=2.0x ATR=5.94 |
| Stop hit — per-position SL triggered | 2023-06-06 10:15:00 | 3228.71 | 3228.07 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 10:05:00 | 3162.75 | 3175.71 | 0.00 | ORB-short ORB[3180.55,3198.00] vol=5.0x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:15:00 | 3151.98 | 3174.30 | 0.00 | T1 1.5R @ 3151.98 |
| Stop hit — per-position SL triggered | 2023-06-12 10:25:00 | 3162.75 | 3173.76 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:45:00 | 3292.90 | 3285.81 | 0.00 | ORB-long ORB[3273.35,3292.00] vol=1.5x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:25:00 | 3301.44 | 3292.48 | 0.00 | T1 1.5R @ 3301.44 |
| Stop hit — per-position SL triggered | 2023-06-15 10:50:00 | 3292.90 | 3293.05 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:15:00 | 3315.30 | 3326.65 | 0.00 | ORB-short ORB[3317.05,3344.95] vol=1.7x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 12:30:00 | 3307.20 | 3324.19 | 0.00 | T1 1.5R @ 3307.20 |
| Stop hit — per-position SL triggered | 2023-06-19 13:00:00 | 3315.30 | 3323.09 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 3283.80 | 3299.19 | 0.00 | ORB-short ORB[3287.90,3320.00] vol=2.3x ATR=6.61 |
| Stop hit — per-position SL triggered | 2023-06-20 10:15:00 | 3290.41 | 3290.85 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 3394.90 | 3378.24 | 0.00 | ORB-long ORB[3340.75,3390.00] vol=3.0x ATR=8.02 |
| Stop hit — per-position SL triggered | 2023-07-05 09:35:00 | 3386.88 | 3379.65 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:50:00 | 3382.45 | 3400.71 | 0.00 | ORB-short ORB[3385.00,3410.00] vol=1.6x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 11:40:00 | 3373.44 | 3396.92 | 0.00 | T1 1.5R @ 3373.44 |
| Target hit | 2023-07-07 15:20:00 | 3341.90 | 3371.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 11:00:00 | 3337.40 | 3352.23 | 0.00 | ORB-short ORB[3340.60,3372.05] vol=1.8x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:10:00 | 3326.65 | 3349.76 | 0.00 | T1 1.5R @ 3326.65 |
| Target hit | 2023-07-10 13:35:00 | 3331.90 | 3324.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2023-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:35:00 | 3384.05 | 3369.41 | 0.00 | ORB-long ORB[3345.00,3379.95] vol=1.6x ATR=9.98 |
| Stop hit — per-position SL triggered | 2023-07-11 10:10:00 | 3374.07 | 3375.87 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:45:00 | 3468.00 | 3448.16 | 0.00 | ORB-long ORB[3425.00,3454.50] vol=1.5x ATR=8.70 |
| Stop hit — per-position SL triggered | 2023-07-17 10:05:00 | 3459.30 | 3454.32 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 09:50:00 | 3486.10 | 3515.19 | 0.00 | ORB-short ORB[3508.05,3547.90] vol=1.7x ATR=11.60 |
| Stop hit — per-position SL triggered | 2023-07-25 09:55:00 | 3497.70 | 3513.96 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:45:00 | 3339.05 | 3362.42 | 0.00 | ORB-short ORB[3362.05,3385.00] vol=1.6x ATR=6.58 |
| Stop hit — per-position SL triggered | 2023-08-01 10:55:00 | 3345.63 | 3360.33 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 10:20:00 | 3356.55 | 3345.96 | 0.00 | ORB-long ORB[3330.65,3354.40] vol=1.5x ATR=7.35 |
| Stop hit — per-position SL triggered | 2023-08-02 10:35:00 | 3349.20 | 3346.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 09:30:00 | 3288.35 | 3299.60 | 0.00 | ORB-short ORB[3291.00,3320.00] vol=1.7x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 09:40:00 | 3277.60 | 3294.35 | 0.00 | T1 1.5R @ 3277.60 |
| Target hit | 2023-08-10 15:20:00 | 3237.10 | 3253.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2023-08-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 11:05:00 | 3210.75 | 3220.71 | 0.00 | ORB-short ORB[3216.25,3248.95] vol=2.2x ATR=5.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 11:30:00 | 3202.06 | 3218.24 | 0.00 | T1 1.5R @ 3202.06 |
| Target hit | 2023-08-11 15:20:00 | 3183.60 | 3200.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:15:00 | 3178.00 | 3193.04 | 0.00 | ORB-short ORB[3191.05,3224.00] vol=2.2x ATR=7.03 |
| Stop hit — per-position SL triggered | 2023-08-17 10:30:00 | 3185.03 | 3191.87 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:45:00 | 3169.05 | 3161.65 | 0.00 | ORB-long ORB[3152.15,3163.55] vol=1.7x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 11:10:00 | 3176.49 | 3163.62 | 0.00 | T1 1.5R @ 3176.49 |
| Target hit | 2023-08-21 15:20:00 | 3182.50 | 3172.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-08-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:45:00 | 3179.95 | 3183.56 | 0.00 | ORB-short ORB[3180.35,3196.00] vol=1.5x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 11:10:00 | 3172.77 | 3182.66 | 0.00 | T1 1.5R @ 3172.77 |
| Stop hit — per-position SL triggered | 2023-08-23 11:40:00 | 3179.95 | 3182.06 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:35:00 | 3193.00 | 3185.65 | 0.00 | ORB-long ORB[3170.90,3189.80] vol=2.1x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 10:10:00 | 3199.87 | 3191.90 | 0.00 | T1 1.5R @ 3199.87 |
| Target hit | 2023-08-24 15:20:00 | 3226.60 | 3214.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2023-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:55:00 | 3278.85 | 3284.68 | 0.00 | ORB-short ORB[3280.00,3302.85] vol=1.6x ATR=4.60 |
| Stop hit — per-position SL triggered | 2023-08-30 11:00:00 | 3283.45 | 3284.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:00:00 | 3226.00 | 3232.23 | 0.00 | ORB-short ORB[3233.35,3255.00] vol=1.6x ATR=5.93 |
| Stop hit — per-position SL triggered | 2023-09-01 11:30:00 | 3231.93 | 3231.00 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:30:00 | 3241.35 | 3267.90 | 0.00 | ORB-short ORB[3273.15,3294.20] vol=1.7x ATR=8.26 |
| Stop hit — per-position SL triggered | 2023-09-14 11:00:00 | 3249.61 | 3263.23 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 09:30:00 | 3190.25 | 3205.32 | 0.00 | ORB-short ORB[3192.60,3240.00] vol=1.9x ATR=8.63 |
| Stop hit — per-position SL triggered | 2023-09-15 09:40:00 | 3198.88 | 3202.74 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 10:50:00 | 3263.55 | 3239.26 | 0.00 | ORB-long ORB[3221.15,3244.00] vol=1.9x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 11:00:00 | 3273.96 | 3244.45 | 0.00 | T1 1.5R @ 3273.96 |
| Target hit | 2023-09-22 15:20:00 | 3279.60 | 3261.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2023-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 09:40:00 | 3307.30 | 3292.01 | 0.00 | ORB-long ORB[3271.15,3297.25] vol=2.1x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 09:45:00 | 3317.01 | 3298.85 | 0.00 | T1 1.5R @ 3317.01 |
| Stop hit — per-position SL triggered | 2023-09-25 10:25:00 | 3307.30 | 3306.70 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:40:00 | 3262.55 | 3287.70 | 0.00 | ORB-short ORB[3287.55,3324.90] vol=1.6x ATR=9.50 |
| Stop hit — per-position SL triggered | 2023-09-26 09:50:00 | 3272.05 | 3284.36 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 11:15:00 | 3181.20 | 3172.23 | 0.00 | ORB-long ORB[3151.35,3180.00] vol=1.6x ATR=5.63 |
| Stop hit — per-position SL triggered | 2023-10-04 11:55:00 | 3175.57 | 3175.32 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 10:45:00 | 3142.20 | 3148.48 | 0.00 | ORB-short ORB[3143.05,3164.90] vol=1.6x ATR=3.75 |
| Stop hit — per-position SL triggered | 2023-10-10 11:10:00 | 3145.95 | 3148.14 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 11:00:00 | 3135.90 | 3151.72 | 0.00 | ORB-short ORB[3140.00,3172.00] vol=4.1x ATR=5.66 |
| Stop hit — per-position SL triggered | 2023-10-13 11:35:00 | 3141.56 | 3147.45 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:55:00 | 3102.65 | 3108.97 | 0.00 | ORB-short ORB[3103.20,3113.40] vol=2.6x ATR=3.47 |
| Stop hit — per-position SL triggered | 2023-10-18 11:05:00 | 3106.12 | 3108.81 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:55:00 | 2995.50 | 3016.96 | 0.00 | ORB-short ORB[3020.25,3055.00] vol=1.8x ATR=7.15 |
| Stop hit — per-position SL triggered | 2023-10-26 11:15:00 | 3002.65 | 3015.19 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 11:05:00 | 3005.00 | 2984.72 | 0.00 | ORB-long ORB[2966.00,2990.00] vol=2.0x ATR=6.08 |
| Stop hit — per-position SL triggered | 2023-10-31 11:15:00 | 2998.92 | 2986.34 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:45:00 | 3006.50 | 2999.73 | 0.00 | ORB-long ORB[2983.85,3005.00] vol=3.9x ATR=4.00 |
| Stop hit — per-position SL triggered | 2023-11-06 11:05:00 | 3002.50 | 3001.90 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-11-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:35:00 | 3089.00 | 3099.76 | 0.00 | ORB-short ORB[3090.00,3104.95] vol=1.7x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:15:00 | 3080.52 | 3097.45 | 0.00 | T1 1.5R @ 3080.52 |
| Stop hit — per-position SL triggered | 2023-11-09 11:35:00 | 3089.00 | 3096.58 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 3067.65 | 3076.84 | 0.00 | ORB-short ORB[3072.45,3098.00] vol=1.5x ATR=4.86 |
| Stop hit — per-position SL triggered | 2023-11-13 11:30:00 | 3072.51 | 3074.08 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:40:00 | 3125.95 | 3111.94 | 0.00 | ORB-long ORB[3097.60,3120.00] vol=1.8x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:00:00 | 3133.43 | 3116.03 | 0.00 | T1 1.5R @ 3133.43 |
| Stop hit — per-position SL triggered | 2023-11-16 11:30:00 | 3125.95 | 3118.56 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 11:15:00 | 3158.20 | 3149.29 | 0.00 | ORB-long ORB[3136.15,3156.60] vol=2.0x ATR=5.02 |
| Stop hit — per-position SL triggered | 2023-11-28 11:25:00 | 3153.18 | 3149.48 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:50:00 | 3169.00 | 3159.69 | 0.00 | ORB-long ORB[3145.50,3163.85] vol=1.9x ATR=5.47 |
| Stop hit — per-position SL triggered | 2023-11-29 10:00:00 | 3163.53 | 3160.46 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 11:00:00 | 3206.40 | 3194.51 | 0.00 | ORB-long ORB[3180.00,3206.35] vol=3.6x ATR=5.59 |
| Stop hit — per-position SL triggered | 2023-12-05 11:45:00 | 3200.81 | 3196.57 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-12-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:30:00 | 3291.05 | 3280.49 | 0.00 | ORB-long ORB[3266.20,3290.00] vol=4.2x ATR=9.25 |
| Stop hit — per-position SL triggered | 2023-12-07 09:55:00 | 3281.80 | 3282.65 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 3249.35 | 3255.57 | 0.00 | ORB-short ORB[3255.00,3269.80] vol=2.1x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:30:00 | 3240.84 | 3254.76 | 0.00 | T1 1.5R @ 3240.84 |
| Stop hit — per-position SL triggered | 2023-12-08 12:10:00 | 3249.35 | 3253.36 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:45:00 | 3215.65 | 3228.62 | 0.00 | ORB-short ORB[3226.35,3254.95] vol=1.7x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 09:55:00 | 3203.30 | 3225.69 | 0.00 | T1 1.5R @ 3203.30 |
| Target hit | 2023-12-13 12:35:00 | 3212.80 | 3212.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — BUY (started 2023-12-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 11:00:00 | 3272.95 | 3259.85 | 0.00 | ORB-long ORB[3242.00,3264.00] vol=3.3x ATR=6.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 11:10:00 | 3282.30 | 3263.09 | 0.00 | T1 1.5R @ 3282.30 |
| Stop hit — per-position SL triggered | 2023-12-15 12:10:00 | 3272.95 | 3265.47 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:40:00 | 3354.00 | 3342.93 | 0.00 | ORB-long ORB[3335.00,3352.00] vol=2.2x ATR=6.52 |
| Stop hit — per-position SL triggered | 2023-12-20 11:20:00 | 3347.48 | 3346.26 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:05:00 | 3293.70 | 3289.57 | 0.00 | ORB-long ORB[3271.85,3292.90] vol=9.2x ATR=8.29 |
| Stop hit — per-position SL triggered | 2023-12-21 10:15:00 | 3285.41 | 3289.56 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:35:00 | 3362.95 | 3352.15 | 0.00 | ORB-long ORB[3345.20,3357.50] vol=2.5x ATR=5.90 |
| Stop hit — per-position SL triggered | 2023-12-26 10:40:00 | 3357.05 | 3352.35 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:30:00 | 3364.90 | 3367.41 | 0.00 | ORB-short ORB[3371.05,3390.00] vol=1.6x ATR=5.80 |
| Stop hit — per-position SL triggered | 2023-12-27 10:35:00 | 3370.70 | 3367.60 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 11:05:00 | 3392.05 | 3399.12 | 0.00 | ORB-short ORB[3399.10,3411.80] vol=2.3x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-01-01 11:10:00 | 3396.18 | 3398.87 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 3368.90 | 3377.23 | 0.00 | ORB-short ORB[3373.65,3396.00] vol=3.4x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:55:00 | 3359.21 | 3372.31 | 0.00 | T1 1.5R @ 3359.21 |
| Target hit | 2024-01-02 10:55:00 | 3368.70 | 3365.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2024-01-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:40:00 | 3382.40 | 3373.83 | 0.00 | ORB-long ORB[3365.00,3382.00] vol=2.1x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 09:55:00 | 3392.42 | 3379.13 | 0.00 | T1 1.5R @ 3392.42 |
| Stop hit — per-position SL triggered | 2024-01-04 10:20:00 | 3382.40 | 3381.85 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:45:00 | 3327.65 | 3352.46 | 0.00 | ORB-short ORB[3353.15,3376.50] vol=2.0x ATR=7.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 11:05:00 | 3316.02 | 3346.02 | 0.00 | T1 1.5R @ 3316.02 |
| Target hit | 2024-01-08 15:20:00 | 3296.80 | 3322.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2024-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:35:00 | 3296.25 | 3306.23 | 0.00 | ORB-short ORB[3301.00,3315.00] vol=1.5x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:50:00 | 3287.09 | 3302.36 | 0.00 | T1 1.5R @ 3287.09 |
| Stop hit — per-position SL triggered | 2024-01-09 10:20:00 | 3296.25 | 3298.82 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:50:00 | 3276.00 | 3290.40 | 0.00 | ORB-short ORB[3286.85,3309.20] vol=1.6x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 3281.76 | 3287.08 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:35:00 | 3254.25 | 3265.55 | 0.00 | ORB-short ORB[3260.50,3294.95] vol=1.5x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 10:10:00 | 3244.46 | 3258.08 | 0.00 | T1 1.5R @ 3244.46 |
| Stop hit — per-position SL triggered | 2024-01-12 10:45:00 | 3254.25 | 3255.44 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 11:00:00 | 3269.45 | 3275.72 | 0.00 | ORB-short ORB[3270.00,3287.20] vol=2.1x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 12:10:00 | 3260.23 | 3273.08 | 0.00 | T1 1.5R @ 3260.23 |
| Stop hit — per-position SL triggered | 2024-01-15 14:10:00 | 3269.45 | 3268.52 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 09:30:00 | 3300.00 | 3291.89 | 0.00 | ORB-long ORB[3273.90,3298.60] vol=1.5x ATR=6.75 |
| Stop hit — per-position SL triggered | 2024-01-16 09:35:00 | 3293.25 | 3292.83 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:00:00 | 3160.25 | 3182.02 | 0.00 | ORB-short ORB[3169.95,3201.55] vol=1.9x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 11:15:00 | 3150.43 | 3177.40 | 0.00 | T1 1.5R @ 3150.43 |
| Stop hit — per-position SL triggered | 2024-01-19 11:20:00 | 3160.25 | 3176.99 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 2974.45 | 2994.38 | 0.00 | ORB-short ORB[2995.00,3016.00] vol=1.7x ATR=6.10 |
| Stop hit — per-position SL triggered | 2024-01-25 11:15:00 | 2980.55 | 2992.67 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:35:00 | 2945.90 | 2949.23 | 0.00 | ORB-short ORB[2952.15,2972.85] vol=1.6x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-02-01 10:40:00 | 2951.37 | 2949.22 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 2948.00 | 2963.88 | 0.00 | ORB-short ORB[2971.30,2999.75] vol=1.7x ATR=6.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:05:00 | 2937.54 | 2960.82 | 0.00 | T1 1.5R @ 2937.54 |
| Stop hit — per-position SL triggered | 2024-02-08 12:25:00 | 2948.00 | 2953.87 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:55:00 | 2970.00 | 2960.63 | 0.00 | ORB-long ORB[2948.85,2967.70] vol=2.9x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 13:00:00 | 2979.32 | 2966.19 | 0.00 | T1 1.5R @ 2979.32 |
| Stop hit — per-position SL triggered | 2024-02-13 13:05:00 | 2970.00 | 2966.37 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 09:50:00 | 2992.90 | 3004.44 | 0.00 | ORB-short ORB[3003.00,3018.70] vol=1.9x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-02-19 10:00:00 | 2999.34 | 3003.72 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-02-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:50:00 | 2990.95 | 3002.29 | 0.00 | ORB-short ORB[2993.15,3012.90] vol=1.6x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 09:55:00 | 2982.91 | 3000.63 | 0.00 | T1 1.5R @ 2982.91 |
| Stop hit — per-position SL triggered | 2024-02-20 10:00:00 | 2990.95 | 2999.67 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:35:00 | 3018.95 | 3012.61 | 0.00 | ORB-long ORB[3005.00,3015.10] vol=1.5x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-02-21 10:10:00 | 3013.43 | 3015.94 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 10:40:00 | 2969.75 | 2985.28 | 0.00 | ORB-short ORB[2975.05,3005.00] vol=1.6x ATR=8.08 |
| Stop hit — per-position SL triggered | 2024-02-23 10:50:00 | 2977.83 | 2984.13 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-02-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 09:30:00 | 2823.30 | 2837.42 | 0.00 | ORB-short ORB[2827.05,2859.40] vol=2.1x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 09:45:00 | 2813.91 | 2828.33 | 0.00 | T1 1.5R @ 2813.91 |
| Target hit | 2024-02-28 15:20:00 | 2790.85 | 2807.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2024-02-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:00:00 | 2806.25 | 2790.12 | 0.00 | ORB-long ORB[2776.75,2797.60] vol=1.6x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 12:00:00 | 2818.03 | 2795.57 | 0.00 | T1 1.5R @ 2818.03 |
| Stop hit — per-position SL triggered | 2024-02-29 14:30:00 | 2806.25 | 2802.59 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:30:00 | 2865.60 | 2854.04 | 0.00 | ORB-long ORB[2825.00,2861.90] vol=2.6x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-03-07 09:40:00 | 2857.74 | 2855.70 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:05:00 | 2863.30 | 2878.25 | 0.00 | ORB-short ORB[2874.10,2896.55] vol=1.8x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 12:10:00 | 2856.14 | 2874.31 | 0.00 | T1 1.5R @ 2856.14 |
| Stop hit — per-position SL triggered | 2024-03-11 12:25:00 | 2863.30 | 2873.21 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 11:00:00 | 2889.50 | 2877.90 | 0.00 | ORB-long ORB[2866.80,2885.00] vol=1.5x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 12:10:00 | 2897.33 | 2883.53 | 0.00 | T1 1.5R @ 2897.33 |
| Stop hit — per-position SL triggered | 2024-03-12 12:50:00 | 2889.50 | 2885.21 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 11:15:00 | 2862.00 | 2871.31 | 0.00 | ORB-short ORB[2870.15,2887.25] vol=4.1x ATR=5.36 |
| Stop hit — per-position SL triggered | 2024-03-13 11:20:00 | 2867.36 | 2871.03 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 10:50:00 | 2878.75 | 2871.09 | 0.00 | ORB-long ORB[2841.00,2860.00] vol=1.9x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 11:25:00 | 2888.03 | 2873.39 | 0.00 | T1 1.5R @ 2888.03 |
| Stop hit — per-position SL triggered | 2024-03-14 11:35:00 | 2878.75 | 2875.59 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:55:00 | 2811.05 | 2817.07 | 0.00 | ORB-short ORB[2821.10,2840.00] vol=3.0x ATR=5.96 |
| Stop hit — per-position SL triggered | 2024-03-19 12:35:00 | 2817.01 | 2813.45 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 11:10:00 | 2829.30 | 2827.48 | 0.00 | ORB-long ORB[2817.70,2827.10] vol=1.6x ATR=3.97 |
| Stop hit — per-position SL triggered | 2024-03-27 11:50:00 | 2825.33 | 2827.74 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-04-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:30:00 | 2890.25 | 2872.86 | 0.00 | ORB-long ORB[2856.00,2874.95] vol=2.9x ATR=5.58 |
| Stop hit — per-position SL triggered | 2024-04-02 10:35:00 | 2884.67 | 2873.80 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 2860.95 | 2871.53 | 0.00 | ORB-short ORB[2870.00,2886.00] vol=1.7x ATR=6.57 |
| Stop hit — per-position SL triggered | 2024-04-04 11:10:00 | 2867.52 | 2864.59 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:55:00 | 2915.10 | 2904.63 | 0.00 | ORB-long ORB[2890.00,2904.75] vol=1.6x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-04-09 10:00:00 | 2909.99 | 2905.20 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 2852.85 | 2868.91 | 0.00 | ORB-short ORB[2862.05,2893.00] vol=3.1x ATR=7.92 |
| Stop hit — per-position SL triggered | 2024-04-12 10:55:00 | 2860.77 | 2859.14 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 11:15:00 | 2835.05 | 2828.22 | 0.00 | ORB-long ORB[2812.35,2835.00] vol=2.5x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 11:30:00 | 2843.84 | 2828.60 | 0.00 | T1 1.5R @ 2843.84 |
| Target hit | 2024-04-15 15:20:00 | 2842.00 | 2838.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 2836.25 | 2833.03 | 0.00 | ORB-long ORB[2821.50,2834.70] vol=1.6x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-04-16 12:00:00 | 2831.24 | 2833.26 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 2826.50 | 2835.10 | 0.00 | ORB-short ORB[2830.40,2857.80] vol=1.6x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 09:45:00 | 2817.16 | 2832.11 | 0.00 | T1 1.5R @ 2817.16 |
| Stop hit — per-position SL triggered | 2024-04-18 09:50:00 | 2826.50 | 2831.88 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:50:00 | 2833.80 | 2843.68 | 0.00 | ORB-short ORB[2838.15,2867.40] vol=1.7x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-04-25 09:55:00 | 2839.77 | 2842.93 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:30:00 | 2880.00 | 2873.64 | 0.00 | ORB-long ORB[2855.00,2877.75] vol=1.6x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 09:40:00 | 2887.75 | 2879.53 | 0.00 | T1 1.5R @ 2887.75 |
| Target hit | 2024-04-30 13:05:00 | 2895.00 | 2895.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 92 — BUY (started 2024-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 10:40:00 | 2952.80 | 2931.69 | 0.00 | ORB-long ORB[2913.40,2951.95] vol=1.5x ATR=10.00 |
| Stop hit — per-position SL triggered | 2024-05-06 10:55:00 | 2942.80 | 2935.11 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:40:00 | 2967.50 | 2948.30 | 0.00 | ORB-long ORB[2918.25,2956.85] vol=2.5x ATR=9.09 |
| Stop hit — per-position SL triggered | 2024-05-07 09:45:00 | 2958.41 | 2949.70 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:05:00 | 2817.55 | 2831.49 | 0.00 | ORB-short ORB[2823.05,2846.85] vol=1.7x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-05-09 12:00:00 | 2824.67 | 2828.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 09:55:00 | 3160.00 | 2023-05-16 10:10:00 | 3153.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-05-17 09:35:00 | 3128.40 | 2023-05-17 09:40:00 | 3119.75 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-17 09:35:00 | 3128.40 | 2023-05-17 15:20:00 | 3096.15 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2023-05-18 11:00:00 | 3113.95 | 2023-05-18 11:20:00 | 3108.99 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-05-19 09:50:00 | 3105.90 | 2023-05-19 10:00:00 | 3095.06 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-05-19 09:50:00 | 3105.90 | 2023-05-19 15:20:00 | 3081.00 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2023-05-23 11:10:00 | 3119.80 | 2023-05-23 11:25:00 | 3114.69 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-05-24 10:50:00 | 3124.15 | 2023-05-24 11:00:00 | 3131.48 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-05-24 10:50:00 | 3124.15 | 2023-05-24 11:15:00 | 3124.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-25 10:55:00 | 3108.45 | 2023-05-25 11:10:00 | 3102.74 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-05-26 11:05:00 | 3111.95 | 2023-05-26 11:20:00 | 3117.57 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-06 09:30:00 | 3234.65 | 2023-06-06 10:15:00 | 3228.71 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-12 10:05:00 | 3162.75 | 2023-06-12 10:15:00 | 3151.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-12 10:05:00 | 3162.75 | 2023-06-12 10:25:00 | 3162.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-15 09:45:00 | 3292.90 | 2023-06-15 10:25:00 | 3301.44 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-06-15 09:45:00 | 3292.90 | 2023-06-15 10:50:00 | 3292.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-19 11:15:00 | 3315.30 | 2023-06-19 12:30:00 | 3307.20 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-06-19 11:15:00 | 3315.30 | 2023-06-19 13:00:00 | 3315.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-20 09:30:00 | 3283.80 | 2023-06-20 10:15:00 | 3290.41 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-05 09:30:00 | 3394.90 | 2023-07-05 09:35:00 | 3386.88 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-07 10:50:00 | 3382.45 | 2023-07-07 11:40:00 | 3373.44 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-07 10:50:00 | 3382.45 | 2023-07-07 15:20:00 | 3341.90 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2023-07-10 11:00:00 | 3337.40 | 2023-07-10 11:10:00 | 3326.65 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-07-10 11:00:00 | 3337.40 | 2023-07-10 13:35:00 | 3331.90 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-07-11 09:35:00 | 3384.05 | 2023-07-11 10:10:00 | 3374.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-17 09:45:00 | 3468.00 | 2023-07-17 10:05:00 | 3459.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-25 09:50:00 | 3486.10 | 2023-07-25 09:55:00 | 3497.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-01 10:45:00 | 3339.05 | 2023-08-01 10:55:00 | 3345.63 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-02 10:20:00 | 3356.55 | 2023-08-02 10:35:00 | 3349.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-10 09:30:00 | 3288.35 | 2023-08-10 09:40:00 | 3277.60 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-08-10 09:30:00 | 3288.35 | 2023-08-10 15:20:00 | 3237.10 | TARGET_HIT | 0.50 | 1.56% |
| SELL | retest1 | 2023-08-11 11:05:00 | 3210.75 | 2023-08-11 11:30:00 | 3202.06 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-11 11:05:00 | 3210.75 | 2023-08-11 15:20:00 | 3183.60 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2023-08-17 10:15:00 | 3178.00 | 2023-08-17 10:30:00 | 3185.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-21 10:45:00 | 3169.05 | 2023-08-21 11:10:00 | 3176.49 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-21 10:45:00 | 3169.05 | 2023-08-21 15:20:00 | 3182.50 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2023-08-23 10:45:00 | 3179.95 | 2023-08-23 11:10:00 | 3172.77 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-08-23 10:45:00 | 3179.95 | 2023-08-23 11:40:00 | 3179.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-24 09:35:00 | 3193.00 | 2023-08-24 10:10:00 | 3199.87 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-08-24 09:35:00 | 3193.00 | 2023-08-24 15:20:00 | 3226.60 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2023-08-30 10:55:00 | 3278.85 | 2023-08-30 11:00:00 | 3283.45 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-09-01 11:00:00 | 3226.00 | 2023-09-01 11:30:00 | 3231.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-09-14 10:30:00 | 3241.35 | 2023-09-14 11:00:00 | 3249.61 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-15 09:30:00 | 3190.25 | 2023-09-15 09:40:00 | 3198.88 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-22 10:50:00 | 3263.55 | 2023-09-22 11:00:00 | 3273.96 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-22 10:50:00 | 3263.55 | 2023-09-22 15:20:00 | 3279.60 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2023-09-25 09:40:00 | 3307.30 | 2023-09-25 09:45:00 | 3317.01 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-25 09:40:00 | 3307.30 | 2023-09-25 10:25:00 | 3307.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-26 09:40:00 | 3262.55 | 2023-09-26 09:50:00 | 3272.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-04 11:15:00 | 3181.20 | 2023-10-04 11:55:00 | 3175.57 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-10-10 10:45:00 | 3142.20 | 2023-10-10 11:10:00 | 3145.95 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-10-13 11:00:00 | 3135.90 | 2023-10-13 11:35:00 | 3141.56 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-10-18 10:55:00 | 3102.65 | 2023-10-18 11:05:00 | 3106.12 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2023-10-26 10:55:00 | 2995.50 | 2023-10-26 11:15:00 | 3002.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-31 11:05:00 | 3005.00 | 2023-10-31 11:15:00 | 2998.92 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-06 10:45:00 | 3006.50 | 2023-11-06 11:05:00 | 3002.50 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-11-09 10:35:00 | 3089.00 | 2023-11-09 11:15:00 | 3080.52 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-11-09 10:35:00 | 3089.00 | 2023-11-09 11:35:00 | 3089.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-13 10:20:00 | 3067.65 | 2023-11-13 11:30:00 | 3072.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-16 10:40:00 | 3125.95 | 2023-11-16 11:00:00 | 3133.43 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-11-16 10:40:00 | 3125.95 | 2023-11-16 11:30:00 | 3125.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-28 11:15:00 | 3158.20 | 2023-11-28 11:25:00 | 3153.18 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-29 09:50:00 | 3169.00 | 2023-11-29 10:00:00 | 3163.53 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-12-05 11:00:00 | 3206.40 | 2023-12-05 11:45:00 | 3200.81 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-12-07 09:30:00 | 3291.05 | 2023-12-07 09:55:00 | 3281.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-08 11:05:00 | 3249.35 | 2023-12-08 11:30:00 | 3240.84 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-12-08 11:05:00 | 3249.35 | 2023-12-08 12:10:00 | 3249.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 09:45:00 | 3215.65 | 2023-12-13 09:55:00 | 3203.30 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-13 09:45:00 | 3215.65 | 2023-12-13 12:35:00 | 3212.80 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-12-15 11:00:00 | 3272.95 | 2023-12-15 11:10:00 | 3282.30 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-15 11:00:00 | 3272.95 | 2023-12-15 12:10:00 | 3272.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 10:40:00 | 3354.00 | 2023-12-20 11:20:00 | 3347.48 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-12-21 10:05:00 | 3293.70 | 2023-12-21 10:15:00 | 3285.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-12-26 10:35:00 | 3362.95 | 2023-12-26 10:40:00 | 3357.05 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-12-27 10:30:00 | 3364.90 | 2023-12-27 10:35:00 | 3370.70 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-01 11:05:00 | 3392.05 | 2024-01-01 11:10:00 | 3396.18 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2024-01-02 09:35:00 | 3368.90 | 2024-01-02 09:55:00 | 3359.21 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-01-02 09:35:00 | 3368.90 | 2024-01-02 10:55:00 | 3368.70 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2024-01-04 09:40:00 | 3382.40 | 2024-01-04 09:55:00 | 3392.42 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-01-04 09:40:00 | 3382.40 | 2024-01-04 10:20:00 | 3382.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 10:45:00 | 3327.65 | 2024-01-08 11:05:00 | 3316.02 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-08 10:45:00 | 3327.65 | 2024-01-08 15:20:00 | 3296.80 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-01-09 09:35:00 | 3296.25 | 2024-01-09 09:50:00 | 3287.09 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-01-09 09:35:00 | 3296.25 | 2024-01-09 10:20:00 | 3296.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-11 10:50:00 | 3276.00 | 2024-01-11 11:15:00 | 3281.76 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-12 09:35:00 | 3254.25 | 2024-01-12 10:10:00 | 3244.46 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-01-12 09:35:00 | 3254.25 | 2024-01-12 10:45:00 | 3254.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-15 11:00:00 | 3269.45 | 2024-01-15 12:10:00 | 3260.23 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-01-15 11:00:00 | 3269.45 | 2024-01-15 14:10:00 | 3269.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-16 09:30:00 | 3300.00 | 2024-01-16 09:35:00 | 3293.25 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-19 11:00:00 | 3160.25 | 2024-01-19 11:15:00 | 3150.43 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-01-19 11:00:00 | 3160.25 | 2024-01-19 11:20:00 | 3160.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-25 10:55:00 | 2974.45 | 2024-01-25 11:15:00 | 2980.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-02-01 10:35:00 | 2945.90 | 2024-02-01 10:40:00 | 2951.37 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-08 11:00:00 | 2948.00 | 2024-02-08 11:05:00 | 2937.54 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-08 11:00:00 | 2948.00 | 2024-02-08 12:25:00 | 2948.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-13 10:55:00 | 2970.00 | 2024-02-13 13:00:00 | 2979.32 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-02-13 10:55:00 | 2970.00 | 2024-02-13 13:05:00 | 2970.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-19 09:50:00 | 2992.90 | 2024-02-19 10:00:00 | 2999.34 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-20 09:50:00 | 2990.95 | 2024-02-20 09:55:00 | 2982.91 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-02-20 09:50:00 | 2990.95 | 2024-02-20 10:00:00 | 2990.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 09:35:00 | 3018.95 | 2024-02-21 10:10:00 | 3013.43 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-02-23 10:40:00 | 2969.75 | 2024-02-23 10:50:00 | 2977.83 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-28 09:30:00 | 2823.30 | 2024-02-28 09:45:00 | 2813.91 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-02-28 09:30:00 | 2823.30 | 2024-02-28 15:20:00 | 2790.85 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-02-29 11:00:00 | 2806.25 | 2024-02-29 12:00:00 | 2818.03 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-29 11:00:00 | 2806.25 | 2024-02-29 14:30:00 | 2806.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-07 09:30:00 | 2865.60 | 2024-03-07 09:40:00 | 2857.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-03-11 11:05:00 | 2863.30 | 2024-03-11 12:10:00 | 2856.14 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-03-11 11:05:00 | 2863.30 | 2024-03-11 12:25:00 | 2863.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 11:00:00 | 2889.50 | 2024-03-12 12:10:00 | 2897.33 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-03-12 11:00:00 | 2889.50 | 2024-03-12 12:50:00 | 2889.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-13 11:15:00 | 2862.00 | 2024-03-13 11:20:00 | 2867.36 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-03-14 10:50:00 | 2878.75 | 2024-03-14 11:25:00 | 2888.03 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-14 10:50:00 | 2878.75 | 2024-03-14 11:35:00 | 2878.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 10:55:00 | 2811.05 | 2024-03-19 12:35:00 | 2817.01 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-03-27 11:10:00 | 2829.30 | 2024-03-27 11:50:00 | 2825.33 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-04-02 10:30:00 | 2890.25 | 2024-04-02 10:35:00 | 2884.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-04-04 09:50:00 | 2860.95 | 2024-04-04 11:10:00 | 2867.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-09 09:55:00 | 2915.10 | 2024-04-09 10:00:00 | 2909.99 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-04-12 09:40:00 | 2852.85 | 2024-04-12 10:55:00 | 2860.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-15 11:15:00 | 2835.05 | 2024-04-15 11:30:00 | 2843.84 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-04-15 11:15:00 | 2835.05 | 2024-04-15 15:20:00 | 2842.00 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2024-04-16 11:05:00 | 2836.25 | 2024-04-16 12:00:00 | 2831.24 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-04-18 09:35:00 | 2826.50 | 2024-04-18 09:45:00 | 2817.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-04-18 09:35:00 | 2826.50 | 2024-04-18 09:50:00 | 2826.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-25 09:50:00 | 2833.80 | 2024-04-25 09:55:00 | 2839.77 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-30 09:30:00 | 2880.00 | 2024-04-30 09:40:00 | 2887.75 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-04-30 09:30:00 | 2880.00 | 2024-04-30 13:05:00 | 2895.00 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-06 10:40:00 | 2952.80 | 2024-05-06 10:55:00 | 2942.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-07 09:40:00 | 2967.50 | 2024-05-07 09:45:00 | 2958.41 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-09 11:05:00 | 2817.55 | 2024-05-09 12:00:00 | 2824.67 | STOP_HIT | 1.00 | -0.25% |
