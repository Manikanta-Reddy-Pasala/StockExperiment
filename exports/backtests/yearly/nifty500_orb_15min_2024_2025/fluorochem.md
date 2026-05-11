# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-01-30 15:25:00 (31996 bars)
- **Last close:** 3054.00
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 10 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 55
- **Target hits / Stop hits / Partials:** 10 / 55 / 21
- **Avg / median % per leg:** 0.15% / -0.22%
- **Sum % (uncompounded):** 13.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 11 | 28.9% | 3 | 27 | 8 | 0.21% | 7.8% |
| BUY @ 2nd Alert (retest1) | 38 | 11 | 28.9% | 3 | 27 | 8 | 0.21% | 7.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.11% | 5.5% |
| SELL @ 2nd Alert (retest1) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.11% | 5.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 31 | 36.0% | 10 | 55 | 21 | 0.15% | 13.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:15:00 | 3244.10 | 3258.60 | 0.00 | ORB-short ORB[3249.50,3270.45] vol=1.5x ATR=12.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:35:00 | 3225.34 | 3253.77 | 0.00 | T1 1.5R @ 3225.34 |
| Target hit | 2024-05-15 11:35:00 | 3221.70 | 3217.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 3210.00 | 3217.43 | 0.00 | ORB-short ORB[3215.00,3232.50] vol=2.3x ATR=10.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:40:00 | 3194.30 | 3211.05 | 0.00 | T1 1.5R @ 3194.30 |
| Target hit | 2024-05-16 10:40:00 | 3209.00 | 3208.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 3209.70 | 3224.97 | 0.00 | ORB-short ORB[3219.70,3247.45] vol=3.4x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-05-17 11:05:00 | 3218.32 | 3223.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:35:00 | 3198.60 | 3209.33 | 0.00 | ORB-short ORB[3205.00,3239.10] vol=6.6x ATR=11.47 |
| Stop hit — per-position SL triggered | 2024-05-21 10:05:00 | 3210.07 | 3205.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:50:00 | 3174.30 | 3197.64 | 0.00 | ORB-short ORB[3210.35,3240.95] vol=1.9x ATR=12.75 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 3187.05 | 3192.80 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 10:50:00 | 3100.00 | 3086.87 | 0.00 | ORB-long ORB[3073.00,3098.00] vol=2.0x ATR=10.40 |
| Stop hit — per-position SL triggered | 2024-05-30 11:40:00 | 3089.60 | 3087.66 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:55:00 | 3040.60 | 3063.36 | 0.00 | ORB-short ORB[3043.35,3085.60] vol=1.5x ATR=12.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:10:00 | 3021.72 | 3052.59 | 0.00 | T1 1.5R @ 3021.72 |
| Target hit | 2024-05-31 15:00:00 | 3032.10 | 3030.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 3086.70 | 3090.77 | 0.00 | ORB-short ORB[3087.75,3113.55] vol=2.5x ATR=12.32 |
| Stop hit — per-position SL triggered | 2024-06-11 09:45:00 | 3099.02 | 3090.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:05:00 | 3207.05 | 3161.01 | 0.00 | ORB-long ORB[3144.50,3185.00] vol=6.6x ATR=14.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:10:00 | 3229.27 | 3211.02 | 0.00 | T1 1.5R @ 3229.27 |
| Target hit | 2024-06-12 12:00:00 | 3265.30 | 3266.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 11:05:00 | 3261.00 | 3243.09 | 0.00 | ORB-long ORB[3235.00,3259.95] vol=3.1x ATR=11.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:40:00 | 3277.94 | 3251.33 | 0.00 | T1 1.5R @ 3277.94 |
| Stop hit — per-position SL triggered | 2024-06-13 12:45:00 | 3261.00 | 3261.63 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 3211.25 | 3226.71 | 0.00 | ORB-short ORB[3227.80,3266.50] vol=4.9x ATR=13.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:35:00 | 3190.81 | 3218.95 | 0.00 | T1 1.5R @ 3190.81 |
| Stop hit — per-position SL triggered | 2024-06-18 09:45:00 | 3211.25 | 3214.10 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:40:00 | 3311.00 | 3273.29 | 0.00 | ORB-long ORB[3237.00,3273.90] vol=1.8x ATR=19.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:45:00 | 3339.97 | 3314.17 | 0.00 | T1 1.5R @ 3339.97 |
| Target hit | 2024-06-20 12:50:00 | 3425.85 | 3428.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-06-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:05:00 | 3199.70 | 3214.24 | 0.00 | ORB-short ORB[3224.00,3247.95] vol=4.3x ATR=14.52 |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 3214.22 | 3219.18 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 3307.00 | 3282.04 | 0.00 | ORB-long ORB[3257.45,3298.80] vol=1.7x ATR=12.49 |
| Stop hit — per-position SL triggered | 2024-07-02 11:20:00 | 3294.51 | 3292.09 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:40:00 | 3254.75 | 3243.39 | 0.00 | ORB-long ORB[3222.00,3252.80] vol=2.4x ATR=9.81 |
| Stop hit — per-position SL triggered | 2024-07-03 11:00:00 | 3244.94 | 3244.04 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 3224.40 | 3238.42 | 0.00 | ORB-short ORB[3225.00,3270.00] vol=2.9x ATR=11.91 |
| Stop hit — per-position SL triggered | 2024-07-05 09:40:00 | 3236.31 | 3237.43 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:35:00 | 3273.00 | 3254.79 | 0.00 | ORB-long ORB[3225.00,3254.45] vol=2.0x ATR=12.18 |
| Stop hit — per-position SL triggered | 2024-07-12 09:45:00 | 3260.82 | 3256.18 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:45:00 | 3245.00 | 3257.73 | 0.00 | ORB-short ORB[3254.75,3284.70] vol=2.1x ATR=12.59 |
| Stop hit — per-position SL triggered | 2024-07-15 10:00:00 | 3257.59 | 3257.30 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:50:00 | 3272.05 | 3301.99 | 0.00 | ORB-short ORB[3309.35,3350.00] vol=2.1x ATR=10.83 |
| Stop hit — per-position SL triggered | 2024-07-18 12:10:00 | 3282.88 | 3282.56 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 10:40:00 | 3166.00 | 3192.47 | 0.00 | ORB-short ORB[3183.00,3220.35] vol=2.1x ATR=12.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 12:15:00 | 3146.67 | 3178.05 | 0.00 | T1 1.5R @ 3146.67 |
| Stop hit — per-position SL triggered | 2024-07-24 14:05:00 | 3166.00 | 3162.46 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:15:00 | 3335.00 | 3301.70 | 0.00 | ORB-long ORB[3268.55,3310.00] vol=3.8x ATR=13.15 |
| Stop hit — per-position SL triggered | 2024-07-31 10:20:00 | 3321.85 | 3305.46 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:20:00 | 3406.50 | 3379.31 | 0.00 | ORB-long ORB[3363.05,3405.35] vol=1.8x ATR=17.18 |
| Stop hit — per-position SL triggered | 2024-08-12 10:45:00 | 3389.32 | 3382.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 3437.40 | 3402.93 | 0.00 | ORB-long ORB[3370.00,3409.90] vol=3.5x ATR=24.36 |
| Stop hit — per-position SL triggered | 2024-08-13 09:40:00 | 3413.04 | 3404.21 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:35:00 | 3318.30 | 3341.77 | 0.00 | ORB-short ORB[3351.75,3382.00] vol=1.6x ATR=11.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 13:05:00 | 3300.83 | 3325.25 | 0.00 | T1 1.5R @ 3300.83 |
| Stop hit — per-position SL triggered | 2024-08-20 13:25:00 | 3318.30 | 3322.84 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:00:00 | 3234.50 | 3243.87 | 0.00 | ORB-short ORB[3258.15,3304.80] vol=5.9x ATR=11.34 |
| Stop hit — per-position SL triggered | 2024-08-21 10:05:00 | 3245.84 | 3244.20 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 11:05:00 | 3198.10 | 3194.13 | 0.00 | ORB-long ORB[3165.00,3192.05] vol=13.7x ATR=8.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:40:00 | 3210.51 | 3194.66 | 0.00 | T1 1.5R @ 3210.51 |
| Stop hit — per-position SL triggered | 2024-08-23 13:05:00 | 3198.10 | 3196.36 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:10:00 | 3278.00 | 3241.76 | 0.00 | ORB-long ORB[3205.00,3248.50] vol=1.7x ATR=12.36 |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 3265.64 | 3242.48 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 3260.00 | 3251.06 | 0.00 | ORB-long ORB[3239.05,3254.95] vol=1.6x ATR=6.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:35:00 | 3270.34 | 3259.54 | 0.00 | T1 1.5R @ 3270.34 |
| Stop hit — per-position SL triggered | 2024-08-27 09:40:00 | 3260.00 | 3259.92 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 3254.75 | 3275.42 | 0.00 | ORB-short ORB[3273.40,3312.45] vol=2.9x ATR=13.65 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 3268.40 | 3272.35 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 3248.00 | 3270.84 | 0.00 | ORB-short ORB[3257.05,3305.85] vol=2.3x ATR=9.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:55:00 | 3233.29 | 3263.02 | 0.00 | T1 1.5R @ 3233.29 |
| Target hit | 2024-08-29 15:20:00 | 3205.00 | 3218.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 3304.50 | 3215.90 | 0.00 | ORB-long ORB[3172.65,3194.55] vol=1.9x ATR=17.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:15:00 | 3330.28 | 3228.18 | 0.00 | T1 1.5R @ 3330.28 |
| Target hit | 2024-09-05 15:20:00 | 3536.00 | 3457.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:50:00 | 4434.85 | 4414.47 | 0.00 | ORB-long ORB[4363.00,4414.00] vol=1.5x ATR=21.04 |
| Stop hit — per-position SL triggered | 2024-09-18 10:00:00 | 4413.81 | 4414.87 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 4260.30 | 4313.82 | 0.00 | ORB-short ORB[4335.70,4381.15] vol=2.8x ATR=25.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:00:00 | 4221.45 | 4282.15 | 0.00 | T1 1.5R @ 4221.45 |
| Target hit | 2024-09-19 15:10:00 | 4179.25 | 4175.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-10-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:40:00 | 4334.45 | 4299.86 | 0.00 | ORB-long ORB[4244.15,4292.15] vol=3.8x ATR=25.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:45:00 | 4372.00 | 4328.43 | 0.00 | T1 1.5R @ 4372.00 |
| Stop hit — per-position SL triggered | 2024-10-01 09:50:00 | 4334.45 | 4333.73 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:15:00 | 4044.10 | 3995.82 | 0.00 | ORB-long ORB[3950.00,3997.45] vol=3.4x ATR=23.40 |
| Stop hit — per-position SL triggered | 2024-10-08 11:20:00 | 4020.70 | 3996.45 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:45:00 | 4705.00 | 4647.80 | 0.00 | ORB-long ORB[4579.00,4628.00] vol=4.4x ATR=29.13 |
| Stop hit — per-position SL triggered | 2024-10-16 09:55:00 | 4675.87 | 4660.48 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:00:00 | 4227.60 | 4263.99 | 0.00 | ORB-short ORB[4241.85,4278.90] vol=1.8x ATR=21.23 |
| Stop hit — per-position SL triggered | 2024-10-31 12:30:00 | 4248.83 | 4254.89 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 11:00:00 | 4310.00 | 4279.34 | 0.00 | ORB-long ORB[4243.85,4296.85] vol=1.5x ATR=17.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:25:00 | 4335.55 | 4291.86 | 0.00 | T1 1.5R @ 4335.55 |
| Stop hit — per-position SL triggered | 2024-11-06 12:20:00 | 4310.00 | 4304.36 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 4380.50 | 4356.85 | 0.00 | ORB-long ORB[4323.50,4373.00] vol=1.8x ATR=20.05 |
| Stop hit — per-position SL triggered | 2024-11-07 09:40:00 | 4360.45 | 4356.11 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:50:00 | 4162.60 | 4189.18 | 0.00 | ORB-short ORB[4187.55,4238.60] vol=2.4x ATR=13.99 |
| Stop hit — per-position SL triggered | 2024-11-12 10:55:00 | 4176.59 | 4187.83 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:00:00 | 3920.05 | 3950.60 | 0.00 | ORB-short ORB[3940.05,3987.85] vol=3.2x ATR=11.13 |
| Stop hit — per-position SL triggered | 2024-11-29 11:25:00 | 3931.18 | 3948.63 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:55:00 | 4023.00 | 4007.70 | 0.00 | ORB-long ORB[3960.05,4013.95] vol=1.5x ATR=14.46 |
| Stop hit — per-position SL triggered | 2024-12-02 11:45:00 | 4008.54 | 4010.95 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:55:00 | 4379.15 | 4341.28 | 0.00 | ORB-long ORB[4300.45,4349.70] vol=2.2x ATR=26.48 |
| Stop hit — per-position SL triggered | 2024-12-06 10:10:00 | 4352.67 | 4343.93 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:40:00 | 4399.00 | 4355.12 | 0.00 | ORB-long ORB[4315.00,4361.65] vol=3.0x ATR=21.98 |
| Stop hit — per-position SL triggered | 2024-12-13 09:45:00 | 4377.02 | 4357.68 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:50:00 | 4390.00 | 4374.51 | 0.00 | ORB-long ORB[4340.70,4375.00] vol=2.4x ATR=17.81 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 4372.19 | 4374.50 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:10:00 | 4319.70 | 4350.59 | 0.00 | ORB-short ORB[4338.45,4389.65] vol=1.6x ATR=16.34 |
| Stop hit — per-position SL triggered | 2024-12-18 10:25:00 | 4336.04 | 4346.98 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:45:00 | 4305.15 | 4324.02 | 0.00 | ORB-short ORB[4322.30,4359.90] vol=4.2x ATR=20.33 |
| Stop hit — per-position SL triggered | 2024-12-20 11:00:00 | 4325.48 | 4315.84 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:50:00 | 4381.90 | 4331.67 | 0.00 | ORB-long ORB[4304.15,4350.00] vol=3.9x ATR=21.65 |
| Stop hit — per-position SL triggered | 2024-12-23 11:35:00 | 4360.25 | 4337.18 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:10:00 | 4317.95 | 4331.83 | 0.00 | ORB-short ORB[4326.05,4389.00] vol=1.9x ATR=12.57 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 4330.52 | 4330.38 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 4410.20 | 4390.28 | 0.00 | ORB-long ORB[4364.05,4405.00] vol=3.6x ATR=12.01 |
| Stop hit — per-position SL triggered | 2024-12-27 09:50:00 | 4398.19 | 4404.27 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 4200.60 | 4224.93 | 0.00 | ORB-short ORB[4212.60,4254.90] vol=2.2x ATR=10.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:15:00 | 4185.51 | 4221.07 | 0.00 | T1 1.5R @ 4185.51 |
| Stop hit — per-position SL triggered | 2025-01-02 11:30:00 | 4200.60 | 4218.64 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 4172.10 | 4196.64 | 0.00 | ORB-short ORB[4181.40,4234.95] vol=2.1x ATR=18.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 13:15:00 | 4144.37 | 4176.82 | 0.00 | T1 1.5R @ 4144.37 |
| Target hit | 2025-01-03 15:20:00 | 4104.35 | 4153.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:30:00 | 3723.70 | 3683.36 | 0.00 | ORB-long ORB[3620.00,3660.50] vol=2.3x ATR=17.48 |
| Stop hit — per-position SL triggered | 2025-01-23 10:40:00 | 3706.22 | 3689.67 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 3638.80 | 3675.42 | 0.00 | ORB-short ORB[3686.80,3725.00] vol=1.5x ATR=18.19 |
| Stop hit — per-position SL triggered | 2025-01-24 13:10:00 | 3656.99 | 3650.01 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:00:00 | 3633.85 | 3676.48 | 0.00 | ORB-short ORB[3685.10,3725.80] vol=2.1x ATR=12.72 |
| Stop hit — per-position SL triggered | 2025-02-21 10:30:00 | 3646.57 | 3667.69 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 11:05:00 | 3630.00 | 3648.99 | 0.00 | ORB-short ORB[3632.00,3680.85] vol=2.0x ATR=10.05 |
| Stop hit — per-position SL triggered | 2025-02-24 12:35:00 | 3640.05 | 3637.82 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:35:00 | 3648.05 | 3678.43 | 0.00 | ORB-short ORB[3674.25,3727.55] vol=1.6x ATR=12.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 09:55:00 | 3628.75 | 3665.23 | 0.00 | T1 1.5R @ 3628.75 |
| Stop hit — per-position SL triggered | 2025-02-27 10:10:00 | 3648.05 | 3662.61 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 09:55:00 | 3694.60 | 3720.81 | 0.00 | ORB-short ORB[3713.05,3762.20] vol=4.3x ATR=13.32 |
| Stop hit — per-position SL triggered | 2025-03-07 10:05:00 | 3707.92 | 3718.46 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:45:00 | 3900.30 | 3916.64 | 0.00 | ORB-short ORB[3905.20,3963.50] vol=6.8x ATR=11.86 |
| Stop hit — per-position SL triggered | 2025-04-17 11:30:00 | 3912.16 | 3911.56 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-21 10:50:00 | 3943.90 | 3947.92 | 0.00 | ORB-short ORB[3948.10,3985.00] vol=6.8x ATR=11.56 |
| Stop hit — per-position SL triggered | 2025-04-21 11:30:00 | 3955.46 | 3947.97 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:00:00 | 4004.40 | 3988.87 | 0.00 | ORB-long ORB[3972.50,4000.00] vol=8.3x ATR=8.76 |
| Stop hit — per-position SL triggered | 2025-04-22 11:05:00 | 3995.64 | 3989.32 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:15:00 | 3941.80 | 3954.42 | 0.00 | ORB-short ORB[3950.00,3995.00] vol=1.5x ATR=14.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 3919.59 | 3948.65 | 0.00 | T1 1.5R @ 3919.59 |
| Target hit | 2025-04-23 12:20:00 | 3939.20 | 3934.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2025-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:35:00 | 3991.10 | 3971.93 | 0.00 | ORB-long ORB[3954.20,3984.80] vol=3.1x ATR=9.16 |
| Stop hit — per-position SL triggered | 2025-04-24 10:45:00 | 3981.94 | 3972.60 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:55:00 | 3864.40 | 3916.46 | 0.00 | ORB-short ORB[3930.10,3987.00] vol=3.1x ATR=17.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:00:00 | 3837.71 | 3897.71 | 0.00 | T1 1.5R @ 3837.71 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 3864.40 | 3894.14 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:50:00 | 3890.10 | 3860.96 | 0.00 | ORB-long ORB[3830.00,3868.70] vol=2.1x ATR=17.07 |
| Stop hit — per-position SL triggered | 2025-05-05 10:10:00 | 3873.03 | 3865.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:15:00 | 3244.10 | 2024-05-15 10:35:00 | 3225.34 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-15 10:15:00 | 3244.10 | 2024-05-15 11:35:00 | 3221.70 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-05-16 09:30:00 | 3210.00 | 2024-05-16 09:40:00 | 3194.30 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-16 09:30:00 | 3210.00 | 2024-05-16 10:40:00 | 3209.00 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2024-05-17 10:55:00 | 3209.70 | 2024-05-17 11:05:00 | 3218.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-21 09:35:00 | 3198.60 | 2024-05-21 10:05:00 | 3210.07 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-27 09:50:00 | 3174.30 | 2024-05-27 10:05:00 | 3187.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-30 10:50:00 | 3100.00 | 2024-05-30 11:40:00 | 3089.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 09:55:00 | 3040.60 | 2024-05-31 10:10:00 | 3021.72 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-05-31 09:55:00 | 3040.60 | 2024-05-31 15:00:00 | 3032.10 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2024-06-11 09:30:00 | 3086.70 | 2024-06-11 09:45:00 | 3099.02 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-12 11:05:00 | 3207.05 | 2024-06-12 11:10:00 | 3229.27 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-12 11:05:00 | 3207.05 | 2024-06-12 12:00:00 | 3265.30 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2024-06-13 11:05:00 | 3261.00 | 2024-06-13 11:40:00 | 3277.94 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-13 11:05:00 | 3261.00 | 2024-06-13 12:45:00 | 3261.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 09:30:00 | 3211.25 | 2024-06-18 09:35:00 | 3190.81 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-06-18 09:30:00 | 3211.25 | 2024-06-18 09:45:00 | 3211.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:40:00 | 3311.00 | 2024-06-20 09:45:00 | 3339.97 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-06-20 09:40:00 | 3311.00 | 2024-06-20 12:50:00 | 3425.85 | TARGET_HIT | 0.50 | 3.47% |
| SELL | retest1 | 2024-06-28 10:05:00 | 3199.70 | 2024-06-28 11:15:00 | 3214.22 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-02 10:20:00 | 3307.00 | 2024-07-02 11:20:00 | 3294.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-03 10:40:00 | 3254.75 | 2024-07-03 11:00:00 | 3244.94 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-05 09:35:00 | 3224.40 | 2024-07-05 09:40:00 | 3236.31 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-12 09:35:00 | 3273.00 | 2024-07-12 09:45:00 | 3260.82 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-15 09:45:00 | 3245.00 | 2024-07-15 10:00:00 | 3257.59 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-18 10:50:00 | 3272.05 | 2024-07-18 12:10:00 | 3282.88 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-24 10:40:00 | 3166.00 | 2024-07-24 12:15:00 | 3146.67 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-07-24 10:40:00 | 3166.00 | 2024-07-24 14:05:00 | 3166.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:15:00 | 3335.00 | 2024-07-31 10:20:00 | 3321.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-12 10:20:00 | 3406.50 | 2024-08-12 10:45:00 | 3389.32 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-08-13 09:35:00 | 3437.40 | 2024-08-13 09:40:00 | 3413.04 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2024-08-20 10:35:00 | 3318.30 | 2024-08-20 13:05:00 | 3300.83 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-20 10:35:00 | 3318.30 | 2024-08-20 13:25:00 | 3318.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 10:00:00 | 3234.50 | 2024-08-21 10:05:00 | 3245.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-23 11:05:00 | 3198.10 | 2024-08-23 11:40:00 | 3210.51 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-23 11:05:00 | 3198.10 | 2024-08-23 13:05:00 | 3198.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 10:10:00 | 3278.00 | 2024-08-26 10:15:00 | 3265.64 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-27 09:30:00 | 3260.00 | 2024-08-27 09:35:00 | 3270.34 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-27 09:30:00 | 3260.00 | 2024-08-27 09:40:00 | 3260.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 3254.75 | 2024-08-28 09:35:00 | 3268.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-29 10:45:00 | 3248.00 | 2024-08-29 10:55:00 | 3233.29 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-08-29 10:45:00 | 3248.00 | 2024-08-29 15:20:00 | 3205.00 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2024-09-05 11:00:00 | 3304.50 | 2024-09-05 11:15:00 | 3330.28 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-09-05 11:00:00 | 3304.50 | 2024-09-05 15:20:00 | 3536.00 | TARGET_HIT | 0.50 | 7.01% |
| BUY | retest1 | 2024-09-18 09:50:00 | 4434.85 | 2024-09-18 10:00:00 | 4413.81 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-09-19 09:40:00 | 4260.30 | 2024-09-19 10:00:00 | 4221.45 | PARTIAL | 0.50 | 0.91% |
| SELL | retest1 | 2024-09-19 09:40:00 | 4260.30 | 2024-09-19 15:10:00 | 4179.25 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2024-10-01 09:40:00 | 4334.45 | 2024-10-01 09:45:00 | 4372.00 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-10-01 09:40:00 | 4334.45 | 2024-10-01 09:50:00 | 4334.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 11:15:00 | 4044.10 | 2024-10-08 11:20:00 | 4020.70 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-10-16 09:45:00 | 4705.00 | 2024-10-16 09:55:00 | 4675.87 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-10-31 11:00:00 | 4227.60 | 2024-10-31 12:30:00 | 4248.83 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-06 11:00:00 | 4310.00 | 2024-11-06 11:25:00 | 4335.55 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-11-06 11:00:00 | 4310.00 | 2024-11-06 12:20:00 | 4310.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-07 09:35:00 | 4380.50 | 2024-11-07 09:40:00 | 4360.45 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-11-12 10:50:00 | 4162.60 | 2024-11-12 10:55:00 | 4176.59 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-29 11:00:00 | 3920.05 | 2024-11-29 11:25:00 | 3931.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-02 10:55:00 | 4023.00 | 2024-12-02 11:45:00 | 4008.54 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-06 09:55:00 | 4379.15 | 2024-12-06 10:10:00 | 4352.67 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-12-13 09:40:00 | 4399.00 | 2024-12-13 09:45:00 | 4377.02 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-17 09:50:00 | 4390.00 | 2024-12-17 10:20:00 | 4372.19 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-18 10:10:00 | 4319.70 | 2024-12-18 10:25:00 | 4336.04 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-20 09:45:00 | 4305.15 | 2024-12-20 11:00:00 | 4325.48 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-12-23 10:50:00 | 4381.90 | 2024-12-23 11:35:00 | 4360.25 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-12-26 10:10:00 | 4317.95 | 2024-12-26 11:10:00 | 4330.52 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-27 09:30:00 | 4410.20 | 2024-12-27 09:50:00 | 4398.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-02 10:55:00 | 4200.60 | 2025-01-02 11:15:00 | 4185.51 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-02 10:55:00 | 4200.60 | 2025-01-02 11:30:00 | 4200.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 09:30:00 | 4172.10 | 2025-01-03 13:15:00 | 4144.37 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-01-03 09:30:00 | 4172.10 | 2025-01-03 15:20:00 | 4104.35 | TARGET_HIT | 0.50 | 1.62% |
| BUY | retest1 | 2025-01-23 10:30:00 | 3723.70 | 2025-01-23 10:40:00 | 3706.22 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-01-24 09:45:00 | 3638.80 | 2025-01-24 13:10:00 | 3656.99 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-02-21 10:00:00 | 3633.85 | 2025-02-21 10:30:00 | 3646.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-24 11:05:00 | 3630.00 | 2025-02-24 12:35:00 | 3640.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-27 09:35:00 | 3648.05 | 2025-02-27 09:55:00 | 3628.75 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-02-27 09:35:00 | 3648.05 | 2025-02-27 10:10:00 | 3648.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-07 09:55:00 | 3694.60 | 2025-03-07 10:05:00 | 3707.92 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-17 10:45:00 | 3900.30 | 2025-04-17 11:30:00 | 3912.16 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-04-21 10:50:00 | 3943.90 | 2025-04-21 11:30:00 | 3955.46 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-22 11:00:00 | 4004.40 | 2025-04-22 11:05:00 | 3995.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-04-23 10:15:00 | 3941.80 | 2025-04-23 10:55:00 | 3919.59 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-23 10:15:00 | 3941.80 | 2025-04-23 12:20:00 | 3939.20 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-04-24 10:35:00 | 3991.10 | 2025-04-24 10:45:00 | 3981.94 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-25 09:55:00 | 3864.40 | 2025-04-25 10:00:00 | 3837.71 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-04-25 09:55:00 | 3864.40 | 2025-04-25 10:10:00 | 3864.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:50:00 | 3890.10 | 2025-05-05 10:10:00 | 3873.03 | STOP_HIT | 1.00 | -0.44% |
