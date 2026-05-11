# TCS (TCS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-08-02 15:25:00 (22779 bars)
- **Last close:** 4283.85
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
| PARTIAL | 41 |
| TARGET_HIT | 18 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 76
- **Target hits / Stop hits / Partials:** 18 / 76 / 41
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 19.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 31 | 41.3% | 10 | 44 | 21 | 0.16% | 12.1% |
| BUY @ 2nd Alert (retest1) | 75 | 31 | 41.3% | 10 | 44 | 21 | 0.16% | 12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 28 | 46.7% | 8 | 32 | 20 | 0.13% | 7.6% |
| SELL @ 2nd Alert (retest1) | 60 | 28 | 46.7% | 8 | 32 | 20 | 0.13% | 7.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 135 | 59 | 43.7% | 18 | 76 | 41 | 0.15% | 19.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:25:00 | 3235.80 | 3241.47 | 0.00 | ORB-short ORB[3237.00,3280.00] vol=2.2x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 10:35:00 | 3227.32 | 3240.22 | 0.00 | T1 1.5R @ 3227.32 |
| Target hit | 2023-05-17 15:20:00 | 3207.85 | 3217.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-05-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:55:00 | 3210.75 | 3213.88 | 0.00 | ORB-short ORB[3216.80,3228.95] vol=2.1x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 10:35:00 | 3200.88 | 3211.25 | 0.00 | T1 1.5R @ 3200.88 |
| Target hit | 2023-05-18 15:20:00 | 3200.00 | 3203.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 11:15:00 | 3227.95 | 3219.07 | 0.00 | ORB-long ORB[3203.05,3218.50] vol=3.8x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-05-19 11:35:00 | 3223.17 | 3220.85 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 09:30:00 | 3245.45 | 3233.11 | 0.00 | ORB-long ORB[3217.05,3239.65] vol=1.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2023-05-22 10:00:00 | 3239.93 | 3237.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:30:00 | 3321.40 | 3308.80 | 0.00 | ORB-long ORB[3292.50,3317.00] vol=2.2x ATR=6.98 |
| Stop hit — per-position SL triggered | 2023-05-23 11:20:00 | 3314.42 | 3316.40 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:25:00 | 3304.70 | 3299.14 | 0.00 | ORB-long ORB[3276.00,3295.35] vol=4.2x ATR=4.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 11:20:00 | 3312.04 | 3301.29 | 0.00 | T1 1.5R @ 3312.04 |
| Stop hit — per-position SL triggered | 2023-05-24 12:35:00 | 3304.70 | 3303.35 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 10:45:00 | 3281.60 | 3288.83 | 0.00 | ORB-short ORB[3293.25,3306.00] vol=1.7x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 11:10:00 | 3273.24 | 3283.60 | 0.00 | T1 1.5R @ 3273.24 |
| Stop hit — per-position SL triggered | 2023-05-25 11:45:00 | 3281.60 | 3282.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 11:10:00 | 3337.70 | 3348.53 | 0.00 | ORB-short ORB[3350.10,3372.00] vol=2.1x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 11:25:00 | 3329.79 | 3346.60 | 0.00 | T1 1.5R @ 3329.79 |
| Target hit | 2023-05-29 15:20:00 | 3320.00 | 3333.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2023-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:45:00 | 3217.00 | 3232.21 | 0.00 | ORB-short ORB[3237.10,3250.00] vol=1.9x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 10:55:00 | 3209.52 | 3228.21 | 0.00 | T1 1.5R @ 3209.52 |
| Stop hit — per-position SL triggered | 2023-06-09 11:15:00 | 3217.00 | 3225.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 11:00:00 | 3232.70 | 3228.43 | 0.00 | ORB-long ORB[3211.00,3232.50] vol=1.6x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 11:15:00 | 3239.81 | 3229.54 | 0.00 | T1 1.5R @ 3239.81 |
| Target hit | 2023-06-12 15:20:00 | 3247.95 | 3238.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2023-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 11:05:00 | 3219.85 | 3225.82 | 0.00 | ORB-short ORB[3226.00,3249.40] vol=1.8x ATR=4.17 |
| Stop hit — per-position SL triggered | 2023-06-23 12:55:00 | 3224.02 | 3223.57 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-26 09:45:00 | 3184.90 | 3192.98 | 0.00 | ORB-short ORB[3187.05,3213.90] vol=1.5x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:55:00 | 3176.16 | 3190.24 | 0.00 | T1 1.5R @ 3176.16 |
| Stop hit — per-position SL triggered | 2023-06-26 11:00:00 | 3184.90 | 3187.37 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:10:00 | 3210.35 | 3201.56 | 0.00 | ORB-long ORB[3196.05,3208.00] vol=3.0x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 11:30:00 | 3215.27 | 3203.24 | 0.00 | T1 1.5R @ 3215.27 |
| Stop hit — per-position SL triggered | 2023-06-28 12:15:00 | 3210.35 | 3206.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 3250.70 | 3235.56 | 0.00 | ORB-long ORB[3214.10,3243.90] vol=1.6x ATR=6.78 |
| Stop hit — per-position SL triggered | 2023-06-30 09:50:00 | 3243.92 | 3241.20 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:15:00 | 3323.10 | 3312.32 | 0.00 | ORB-long ORB[3302.00,3317.95] vol=1.9x ATR=4.76 |
| Stop hit — per-position SL triggered | 2023-07-06 10:30:00 | 3318.34 | 3313.44 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:50:00 | 3345.65 | 3323.94 | 0.00 | ORB-long ORB[3302.00,3319.90] vol=3.3x ATR=7.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 09:55:00 | 3356.85 | 3332.13 | 0.00 | T1 1.5R @ 3356.85 |
| Stop hit — per-position SL triggered | 2023-07-07 10:00:00 | 3345.65 | 3334.28 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 10:55:00 | 3264.05 | 3268.96 | 0.00 | ORB-short ORB[3275.00,3289.15] vol=2.3x ATR=4.09 |
| Stop hit — per-position SL triggered | 2023-07-12 11:00:00 | 3268.14 | 3268.85 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:40:00 | 3322.10 | 3302.41 | 0.00 | ORB-long ORB[3272.75,3311.90] vol=2.2x ATR=9.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 10:00:00 | 3336.62 | 3310.59 | 0.00 | T1 1.5R @ 3336.62 |
| Target hit | 2023-07-13 14:35:00 | 3335.00 | 3339.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2023-07-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:40:00 | 3378.75 | 3366.46 | 0.00 | ORB-long ORB[3352.65,3375.00] vol=1.6x ATR=6.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:55:00 | 3388.50 | 3369.93 | 0.00 | T1 1.5R @ 3388.50 |
| Target hit | 2023-07-14 15:20:00 | 3517.00 | 3442.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 3473.90 | 3486.60 | 0.00 | ORB-short ORB[3475.50,3510.00] vol=1.5x ATR=9.56 |
| Stop hit — per-position SL triggered | 2023-07-18 09:45:00 | 3483.46 | 3483.21 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 3477.95 | 3494.84 | 0.00 | ORB-short ORB[3496.00,3516.85] vol=1.5x ATR=7.25 |
| Stop hit — per-position SL triggered | 2023-07-19 10:50:00 | 3485.20 | 3494.36 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:30:00 | 3399.60 | 3389.88 | 0.00 | ORB-long ORB[3372.10,3399.30] vol=1.5x ATR=6.73 |
| Stop hit — per-position SL triggered | 2023-07-24 11:20:00 | 3392.87 | 3391.76 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 11:15:00 | 3418.30 | 3407.10 | 0.00 | ORB-long ORB[3390.00,3405.00] vol=1.9x ATR=4.40 |
| Stop hit — per-position SL triggered | 2023-07-26 11:25:00 | 3413.90 | 3407.64 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 09:35:00 | 3362.15 | 3369.36 | 0.00 | ORB-short ORB[3366.00,3393.70] vol=1.8x ATR=6.49 |
| Stop hit — per-position SL triggered | 2023-07-28 09:50:00 | 3368.64 | 3368.74 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:35:00 | 3412.95 | 3387.50 | 0.00 | ORB-long ORB[3360.90,3384.00] vol=1.6x ATR=6.98 |
| Stop hit — per-position SL triggered | 2023-07-31 11:50:00 | 3405.97 | 3395.40 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:55:00 | 3399.25 | 3412.31 | 0.00 | ORB-short ORB[3414.75,3432.00] vol=2.2x ATR=7.24 |
| Stop hit — per-position SL triggered | 2023-08-03 11:00:00 | 3406.49 | 3411.95 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:30:00 | 3436.90 | 3423.90 | 0.00 | ORB-long ORB[3401.25,3433.00] vol=1.6x ATR=6.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 09:50:00 | 3447.39 | 3431.24 | 0.00 | T1 1.5R @ 3447.39 |
| Target hit | 2023-08-04 10:45:00 | 3443.35 | 3444.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2023-08-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:00:00 | 3446.20 | 3452.11 | 0.00 | ORB-short ORB[3450.05,3466.00] vol=2.2x ATR=6.61 |
| Stop hit — per-position SL triggered | 2023-08-07 12:20:00 | 3452.81 | 3449.78 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:00:00 | 3436.35 | 3445.02 | 0.00 | ORB-short ORB[3450.15,3464.90] vol=3.2x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 10:15:00 | 3428.58 | 3443.52 | 0.00 | T1 1.5R @ 3428.58 |
| Stop hit — per-position SL triggered | 2023-08-09 10:35:00 | 3436.35 | 3440.66 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:35:00 | 3428.25 | 3442.62 | 0.00 | ORB-short ORB[3437.90,3450.45] vol=1.7x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 09:55:00 | 3418.84 | 3437.40 | 0.00 | T1 1.5R @ 3418.84 |
| Stop hit — per-position SL triggered | 2023-08-11 10:15:00 | 3428.25 | 3435.41 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-14 10:55:00 | 3414.65 | 3420.73 | 0.00 | ORB-short ORB[3418.00,3440.70] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2023-08-14 11:15:00 | 3419.98 | 3420.30 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:55:00 | 3425.20 | 3439.57 | 0.00 | ORB-short ORB[3446.40,3460.80] vol=1.6x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:15:00 | 3416.06 | 3436.53 | 0.00 | T1 1.5R @ 3416.06 |
| Stop hit — per-position SL triggered | 2023-08-17 11:45:00 | 3425.20 | 3433.94 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 11:00:00 | 3372.75 | 3379.64 | 0.00 | ORB-short ORB[3376.50,3389.55] vol=1.6x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 11:55:00 | 3366.59 | 3377.73 | 0.00 | T1 1.5R @ 3366.59 |
| Stop hit — per-position SL triggered | 2023-08-29 12:00:00 | 3372.75 | 3377.66 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:50:00 | 3398.10 | 3392.65 | 0.00 | ORB-long ORB[3383.05,3398.00] vol=2.1x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 11:05:00 | 3404.61 | 3396.14 | 0.00 | T1 1.5R @ 3404.61 |
| Stop hit — per-position SL triggered | 2023-08-30 11:45:00 | 3398.10 | 3397.69 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:50:00 | 3416.85 | 3403.04 | 0.00 | ORB-long ORB[3393.45,3405.90] vol=2.0x ATR=5.43 |
| Stop hit — per-position SL triggered | 2023-08-31 10:00:00 | 3411.42 | 3404.96 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:00:00 | 3384.30 | 3378.35 | 0.00 | ORB-long ORB[3356.80,3377.95] vol=1.9x ATR=5.30 |
| Stop hit — per-position SL triggered | 2023-09-01 12:20:00 | 3379.00 | 3381.58 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 3427.60 | 3421.91 | 0.00 | ORB-long ORB[3405.15,3426.00] vol=1.6x ATR=6.89 |
| Stop hit — per-position SL triggered | 2023-09-05 10:35:00 | 3420.71 | 3422.53 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 10:45:00 | 3425.00 | 3431.35 | 0.00 | ORB-short ORB[3427.30,3443.90] vol=1.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2023-09-06 10:50:00 | 3429.50 | 3431.22 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:05:00 | 3481.10 | 3470.36 | 0.00 | ORB-long ORB[3445.00,3474.55] vol=2.3x ATR=7.13 |
| Stop hit — per-position SL triggered | 2023-09-11 10:25:00 | 3473.97 | 3471.14 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 3502.00 | 3496.48 | 0.00 | ORB-long ORB[3483.00,3500.00] vol=1.9x ATR=5.34 |
| Stop hit — per-position SL triggered | 2023-09-12 09:35:00 | 3496.66 | 3496.93 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 11:10:00 | 3622.30 | 3612.07 | 0.00 | ORB-long ORB[3580.00,3619.25] vol=1.9x ATR=5.09 |
| Stop hit — per-position SL triggered | 2023-09-20 11:50:00 | 3617.21 | 3613.84 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 11:05:00 | 3501.80 | 3502.53 | 0.00 | ORB-short ORB[3508.00,3534.20] vol=1.8x ATR=7.95 |
| Stop hit — per-position SL triggered | 2023-10-03 11:30:00 | 3509.75 | 3502.76 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:50:00 | 3603.00 | 3584.68 | 0.00 | ORB-long ORB[3551.20,3582.65] vol=1.7x ATR=9.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 11:00:00 | 3616.85 | 3589.50 | 0.00 | T1 1.5R @ 3616.85 |
| Stop hit — per-position SL triggered | 2023-10-05 11:20:00 | 3603.00 | 3592.05 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:40:00 | 3669.00 | 3645.97 | 0.00 | ORB-long ORB[3631.05,3659.00] vol=1.6x ATR=10.90 |
| Stop hit — per-position SL triggered | 2023-10-09 09:55:00 | 3658.10 | 3651.59 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 09:30:00 | 3605.00 | 3619.93 | 0.00 | ORB-short ORB[3605.50,3648.00] vol=1.5x ATR=8.48 |
| Stop hit — per-position SL triggered | 2023-10-10 09:40:00 | 3613.48 | 3618.06 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 09:30:00 | 3545.60 | 3555.73 | 0.00 | ORB-short ORB[3551.00,3574.65] vol=1.8x ATR=6.95 |
| Stop hit — per-position SL triggered | 2023-10-16 09:35:00 | 3552.55 | 3554.98 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 11:05:00 | 3491.10 | 3466.66 | 0.00 | ORB-long ORB[3444.05,3466.75] vol=1.7x ATR=5.49 |
| Stop hit — per-position SL triggered | 2023-10-20 11:10:00 | 3485.61 | 3468.66 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:55:00 | 3373.15 | 3360.95 | 0.00 | ORB-long ORB[3333.75,3370.00] vol=2.4x ATR=6.53 |
| Stop hit — per-position SL triggered | 2023-10-30 12:15:00 | 3366.62 | 3364.91 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 11:05:00 | 3364.50 | 3375.36 | 0.00 | ORB-short ORB[3372.65,3387.90] vol=2.2x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 13:55:00 | 3357.53 | 3367.51 | 0.00 | T1 1.5R @ 3357.53 |
| Stop hit — per-position SL triggered | 2023-11-03 14:35:00 | 3364.50 | 3366.65 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:00:00 | 3399.70 | 3384.85 | 0.00 | ORB-long ORB[3369.55,3384.55] vol=2.0x ATR=4.93 |
| Stop hit — per-position SL triggered | 2023-11-08 10:25:00 | 3394.77 | 3388.53 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 11:00:00 | 3327.30 | 3337.20 | 0.00 | ORB-short ORB[3330.30,3347.20] vol=2.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2023-11-10 12:50:00 | 3331.78 | 3332.06 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:55:00 | 3334.50 | 3340.96 | 0.00 | ORB-short ORB[3337.30,3356.00] vol=2.0x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 11:35:00 | 3328.66 | 3337.54 | 0.00 | T1 1.5R @ 3328.66 |
| Stop hit — per-position SL triggered | 2023-11-13 14:10:00 | 3334.50 | 3334.29 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:55:00 | 3453.75 | 3433.87 | 0.00 | ORB-long ORB[3415.00,3441.40] vol=1.6x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:00:00 | 3462.24 | 3435.78 | 0.00 | T1 1.5R @ 3462.24 |
| Target hit | 2023-11-16 15:20:00 | 3490.50 | 3484.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2023-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:35:00 | 3507.45 | 3495.69 | 0.00 | ORB-long ORB[3481.10,3500.00] vol=2.3x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 13:40:00 | 3518.06 | 3504.51 | 0.00 | T1 1.5R @ 3518.06 |
| Target hit | 2023-11-22 15:20:00 | 3529.90 | 3510.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2023-11-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:35:00 | 3539.00 | 3532.83 | 0.00 | ORB-long ORB[3522.10,3535.50] vol=1.8x ATR=4.90 |
| Stop hit — per-position SL triggered | 2023-11-23 09:45:00 | 3534.10 | 3533.78 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:50:00 | 3483.15 | 3492.37 | 0.00 | ORB-short ORB[3491.00,3504.50] vol=2.7x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 11:35:00 | 3475.29 | 3489.66 | 0.00 | T1 1.5R @ 3475.29 |
| Target hit | 2023-11-24 15:20:00 | 3457.60 | 3475.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2023-11-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 11:05:00 | 3497.90 | 3503.36 | 0.00 | ORB-short ORB[3498.00,3516.65] vol=2.9x ATR=6.65 |
| Stop hit — per-position SL triggered | 2023-11-30 13:40:00 | 3504.55 | 3500.93 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 3553.95 | 3540.52 | 0.00 | ORB-long ORB[3525.15,3547.80] vol=2.3x ATR=6.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:45:00 | 3564.41 | 3552.66 | 0.00 | T1 1.5R @ 3564.41 |
| Target hit | 2023-12-06 10:20:00 | 3555.30 | 3557.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2023-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:45:00 | 3668.00 | 3654.76 | 0.00 | ORB-long ORB[3631.00,3657.15] vol=1.6x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 09:55:00 | 3677.68 | 3660.75 | 0.00 | T1 1.5R @ 3677.68 |
| Target hit | 2023-12-12 14:20:00 | 3673.00 | 3675.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2023-12-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:30:00 | 3672.45 | 3656.16 | 0.00 | ORB-long ORB[3627.60,3666.00] vol=1.8x ATR=10.44 |
| Stop hit — per-position SL triggered | 2023-12-14 09:40:00 | 3662.01 | 3658.30 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:45:00 | 3700.05 | 3688.03 | 0.00 | ORB-long ORB[3666.70,3692.50] vol=1.6x ATR=7.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:00:00 | 3710.74 | 3693.93 | 0.00 | T1 1.5R @ 3710.74 |
| Target hit | 2023-12-15 15:20:00 | 3885.60 | 3808.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 3809.30 | 3825.56 | 0.00 | ORB-short ORB[3815.05,3855.10] vol=1.8x ATR=9.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 11:15:00 | 3794.44 | 3815.67 | 0.00 | T1 1.5R @ 3794.44 |
| Stop hit — per-position SL triggered | 2023-12-19 11:55:00 | 3809.30 | 3813.16 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 3877.40 | 3861.28 | 0.00 | ORB-long ORB[3827.25,3875.00] vol=2.2x ATR=12.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 09:35:00 | 3895.44 | 3868.41 | 0.00 | T1 1.5R @ 3895.44 |
| Stop hit — per-position SL triggered | 2023-12-20 09:50:00 | 3877.40 | 3872.57 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:35:00 | 3833.65 | 3810.13 | 0.00 | ORB-long ORB[3791.50,3819.85] vol=1.9x ATR=10.10 |
| Stop hit — per-position SL triggered | 2023-12-26 10:40:00 | 3823.55 | 3811.14 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 11:10:00 | 3788.50 | 3800.19 | 0.00 | ORB-short ORB[3790.00,3815.00] vol=3.2x ATR=5.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 11:20:00 | 3780.05 | 3797.87 | 0.00 | T1 1.5R @ 3780.05 |
| Stop hit — per-position SL triggered | 2023-12-27 14:00:00 | 3788.50 | 3787.03 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 3835.90 | 3822.07 | 0.00 | ORB-long ORB[3811.20,3824.50] vol=1.6x ATR=7.14 |
| Stop hit — per-position SL triggered | 2023-12-28 09:50:00 | 3828.76 | 3825.97 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 11:15:00 | 3795.55 | 3782.78 | 0.00 | ORB-long ORB[3775.90,3793.00] vol=4.0x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 11:25:00 | 3803.57 | 3785.97 | 0.00 | T1 1.5R @ 3803.57 |
| Stop hit — per-position SL triggered | 2024-01-01 11:40:00 | 3795.55 | 3787.41 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 3735.10 | 3751.89 | 0.00 | ORB-short ORB[3745.40,3771.85] vol=1.5x ATR=8.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 09:40:00 | 3722.17 | 3743.14 | 0.00 | T1 1.5R @ 3722.17 |
| Target hit | 2024-01-03 15:20:00 | 3692.35 | 3712.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2024-01-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:30:00 | 3686.65 | 3703.78 | 0.00 | ORB-short ORB[3696.15,3719.00] vol=2.2x ATR=7.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 09:40:00 | 3675.34 | 3691.73 | 0.00 | T1 1.5R @ 3675.34 |
| Target hit | 2024-01-04 13:50:00 | 3668.50 | 3668.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — BUY (started 2024-01-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 11:00:00 | 3730.00 | 3704.26 | 0.00 | ORB-long ORB[3674.85,3694.70] vol=2.7x ATR=8.73 |
| Stop hit — per-position SL triggered | 2024-01-05 11:15:00 | 3721.27 | 3707.20 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 3697.75 | 3712.21 | 0.00 | ORB-short ORB[3699.00,3739.75] vol=2.2x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-01-08 11:45:00 | 3705.61 | 3709.94 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:10:00 | 3741.80 | 3728.18 | 0.00 | ORB-long ORB[3706.00,3726.50] vol=1.8x ATR=8.58 |
| Stop hit — per-position SL triggered | 2024-01-09 10:30:00 | 3733.22 | 3729.44 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 11:15:00 | 3717.50 | 3709.83 | 0.00 | ORB-long ORB[3690.00,3714.45] vol=1.9x ATR=7.87 |
| Stop hit — per-position SL triggered | 2024-01-10 12:05:00 | 3709.63 | 3711.32 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:30:00 | 3750.90 | 3722.95 | 0.00 | ORB-long ORB[3707.00,3730.10] vol=3.3x ATR=10.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:35:00 | 3766.45 | 3733.04 | 0.00 | T1 1.5R @ 3766.45 |
| Stop hit — per-position SL triggered | 2024-01-11 10:40:00 | 3750.90 | 3734.81 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:35:00 | 3874.65 | 3849.61 | 0.00 | ORB-long ORB[3821.60,3860.25] vol=2.9x ATR=16.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 12:40:00 | 3899.81 | 3867.39 | 0.00 | T1 1.5R @ 3899.81 |
| Stop hit — per-position SL triggered | 2024-01-12 14:00:00 | 3874.65 | 3870.50 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:50:00 | 3913.70 | 3934.90 | 0.00 | ORB-short ORB[3930.00,3955.65] vol=1.7x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 11:10:00 | 3903.66 | 3930.45 | 0.00 | T1 1.5R @ 3903.66 |
| Target hit | 2024-01-20 15:20:00 | 3861.75 | 3897.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:40:00 | 3929.00 | 3904.79 | 0.00 | ORB-long ORB[3888.95,3916.50] vol=2.0x ATR=9.89 |
| Stop hit — per-position SL triggered | 2024-01-23 09:50:00 | 3919.11 | 3906.90 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 11:05:00 | 3781.00 | 3806.12 | 0.00 | ORB-short ORB[3813.30,3839.90] vol=1.8x ATR=8.27 |
| Stop hit — per-position SL triggered | 2024-01-25 11:20:00 | 3789.27 | 3804.63 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 11:05:00 | 3787.55 | 3802.50 | 0.00 | ORB-short ORB[3792.40,3819.95] vol=1.7x ATR=9.01 |
| Stop hit — per-position SL triggered | 2024-01-29 11:30:00 | 3796.56 | 3800.05 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 09:35:00 | 3849.90 | 3830.03 | 0.00 | ORB-long ORB[3805.05,3835.35] vol=1.5x ATR=9.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 09:40:00 | 3864.02 | 3835.56 | 0.00 | T1 1.5R @ 3864.02 |
| Target hit | 2024-02-01 15:00:00 | 3855.00 | 3864.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 3949.55 | 3922.17 | 0.00 | ORB-long ORB[3872.00,3913.90] vol=3.4x ATR=9.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:35:00 | 3963.79 | 3929.03 | 0.00 | T1 1.5R @ 3963.79 |
| Stop hit — per-position SL triggered | 2024-02-02 12:20:00 | 3949.55 | 3936.70 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:45:00 | 4131.85 | 4119.06 | 0.00 | ORB-long ORB[4086.05,4123.40] vol=1.7x ATR=10.15 |
| Stop hit — per-position SL triggered | 2024-02-08 10:15:00 | 4121.70 | 4125.54 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:55:00 | 4056.25 | 4031.11 | 0.00 | ORB-long ORB[3982.60,4026.85] vol=3.4x ATR=11.89 |
| Stop hit — per-position SL triggered | 2024-02-21 10:20:00 | 4044.36 | 4036.53 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 10:00:00 | 4144.05 | 4110.97 | 0.00 | ORB-long ORB[4089.00,4108.85] vol=1.5x ATR=10.42 |
| Stop hit — per-position SL triggered | 2024-02-28 10:50:00 | 4133.63 | 4126.36 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-02-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:10:00 | 4118.40 | 4107.74 | 0.00 | ORB-long ORB[4080.00,4114.90] vol=1.5x ATR=12.22 |
| Stop hit — per-position SL triggered | 2024-02-29 11:15:00 | 4106.18 | 4108.01 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 10:55:00 | 4089.10 | 4100.43 | 0.00 | ORB-short ORB[4092.40,4117.90] vol=1.8x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 11:55:00 | 4077.97 | 4096.72 | 0.00 | T1 1.5R @ 4077.97 |
| Stop hit — per-position SL triggered | 2024-03-04 12:40:00 | 4089.10 | 4095.16 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-03-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:45:00 | 4097.95 | 4070.56 | 0.00 | ORB-long ORB[4037.70,4091.45] vol=2.0x ATR=11.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:55:00 | 4114.66 | 4077.97 | 0.00 | T1 1.5R @ 4114.66 |
| Stop hit — per-position SL triggered | 2024-03-07 12:05:00 | 4097.95 | 4089.11 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 11:00:00 | 3980.00 | 3985.89 | 0.00 | ORB-short ORB[3982.00,4008.40] vol=1.6x ATR=7.07 |
| Stop hit — per-position SL triggered | 2024-03-21 11:20:00 | 3987.07 | 3985.50 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-04-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 11:10:00 | 3963.00 | 3910.34 | 0.00 | ORB-long ORB[3851.00,3889.00] vol=3.2x ATR=10.24 |
| Stop hit — per-position SL triggered | 2024-04-03 11:25:00 | 3952.76 | 3916.77 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-04-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:35:00 | 3976.80 | 3985.69 | 0.00 | ORB-short ORB[3978.00,3995.00] vol=2.5x ATR=7.28 |
| Stop hit — per-position SL triggered | 2024-04-08 11:10:00 | 3984.08 | 3983.71 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:30:00 | 4013.30 | 3994.73 | 0.00 | ORB-long ORB[3968.85,4006.10] vol=1.7x ATR=10.38 |
| Stop hit — per-position SL triggered | 2024-04-09 09:35:00 | 4002.92 | 3996.99 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 11:05:00 | 3929.55 | 3897.65 | 0.00 | ORB-long ORB[3875.15,3899.65] vol=2.2x ATR=9.81 |
| Stop hit — per-position SL triggered | 2024-04-18 11:40:00 | 3919.74 | 3902.59 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 11:10:00 | 3867.50 | 3882.81 | 0.00 | ORB-short ORB[3875.00,3898.00] vol=1.6x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 11:40:00 | 3860.39 | 3880.00 | 0.00 | T1 1.5R @ 3860.39 |
| Target hit | 2024-04-24 15:20:00 | 3834.60 | 3854.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — SELL (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 11:15:00 | 3844.50 | 3865.76 | 0.00 | ORB-short ORB[3857.00,3875.70] vol=1.8x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-04-30 11:20:00 | 3850.26 | 3865.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 10:25:00 | 3235.80 | 2023-05-17 10:35:00 | 3227.32 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-05-17 10:25:00 | 3235.80 | 2023-05-17 15:20:00 | 3207.85 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2023-05-18 09:55:00 | 3210.75 | 2023-05-18 10:35:00 | 3200.88 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-05-18 09:55:00 | 3210.75 | 2023-05-18 15:20:00 | 3200.00 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2023-05-19 11:15:00 | 3227.95 | 2023-05-19 11:35:00 | 3223.17 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-05-22 09:30:00 | 3245.45 | 2023-05-22 10:00:00 | 3239.93 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-05-23 09:30:00 | 3321.40 | 2023-05-23 11:20:00 | 3314.42 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-05-24 10:25:00 | 3304.70 | 2023-05-24 11:20:00 | 3312.04 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-05-24 10:25:00 | 3304.70 | 2023-05-24 12:35:00 | 3304.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-25 10:45:00 | 3281.60 | 2023-05-25 11:10:00 | 3273.24 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-05-25 10:45:00 | 3281.60 | 2023-05-25 11:45:00 | 3281.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-29 11:10:00 | 3337.70 | 2023-05-29 11:25:00 | 3329.79 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-05-29 11:10:00 | 3337.70 | 2023-05-29 15:20:00 | 3320.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2023-06-09 10:45:00 | 3217.00 | 2023-06-09 10:55:00 | 3209.52 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-06-09 10:45:00 | 3217.00 | 2023-06-09 11:15:00 | 3217.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-12 11:00:00 | 3232.70 | 2023-06-12 11:15:00 | 3239.81 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-06-12 11:00:00 | 3232.70 | 2023-06-12 15:20:00 | 3247.95 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2023-06-23 11:05:00 | 3219.85 | 2023-06-23 12:55:00 | 3224.02 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-06-26 09:45:00 | 3184.90 | 2023-06-26 09:55:00 | 3176.16 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-06-26 09:45:00 | 3184.90 | 2023-06-26 11:00:00 | 3184.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-28 11:10:00 | 3210.35 | 2023-06-28 11:30:00 | 3215.27 | PARTIAL | 0.50 | 0.15% |
| BUY | retest1 | 2023-06-28 11:10:00 | 3210.35 | 2023-06-28 12:15:00 | 3210.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-30 09:30:00 | 3250.70 | 2023-06-30 09:50:00 | 3243.92 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-06 10:15:00 | 3323.10 | 2023-07-06 10:30:00 | 3318.34 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-07-07 09:50:00 | 3345.65 | 2023-07-07 09:55:00 | 3356.85 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-07-07 09:50:00 | 3345.65 | 2023-07-07 10:00:00 | 3345.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-12 10:55:00 | 3264.05 | 2023-07-12 11:00:00 | 3268.14 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-07-13 09:40:00 | 3322.10 | 2023-07-13 10:00:00 | 3336.62 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-13 09:40:00 | 3322.10 | 2023-07-13 14:35:00 | 3335.00 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-07-14 10:40:00 | 3378.75 | 2023-07-14 10:55:00 | 3388.50 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-14 10:40:00 | 3378.75 | 2023-07-14 15:20:00 | 3517.00 | TARGET_HIT | 0.50 | 4.09% |
| SELL | retest1 | 2023-07-18 09:30:00 | 3473.90 | 2023-07-18 09:45:00 | 3483.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-19 10:40:00 | 3477.95 | 2023-07-19 10:50:00 | 3485.20 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-24 10:30:00 | 3399.60 | 2023-07-24 11:20:00 | 3392.87 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-26 11:15:00 | 3418.30 | 2023-07-26 11:25:00 | 3413.90 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-07-28 09:35:00 | 3362.15 | 2023-07-28 09:50:00 | 3368.64 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-31 10:35:00 | 3412.95 | 2023-07-31 11:50:00 | 3405.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-03 10:55:00 | 3399.25 | 2023-08-03 11:00:00 | 3406.49 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-04 09:30:00 | 3436.90 | 2023-08-04 09:50:00 | 3447.39 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-08-04 09:30:00 | 3436.90 | 2023-08-04 10:45:00 | 3443.35 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-08-07 10:00:00 | 3446.20 | 2023-08-07 12:20:00 | 3452.81 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-08-09 10:00:00 | 3436.35 | 2023-08-09 10:15:00 | 3428.58 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-08-09 10:00:00 | 3436.35 | 2023-08-09 10:35:00 | 3436.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-11 09:35:00 | 3428.25 | 2023-08-11 09:55:00 | 3418.84 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-11 09:35:00 | 3428.25 | 2023-08-11 10:15:00 | 3428.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-14 10:55:00 | 3414.65 | 2023-08-14 11:15:00 | 3419.98 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-17 10:55:00 | 3425.20 | 2023-08-17 11:15:00 | 3416.06 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-17 10:55:00 | 3425.20 | 2023-08-17 11:45:00 | 3425.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-29 11:00:00 | 3372.75 | 2023-08-29 11:55:00 | 3366.59 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2023-08-29 11:00:00 | 3372.75 | 2023-08-29 12:00:00 | 3372.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 10:50:00 | 3398.10 | 2023-08-30 11:05:00 | 3404.61 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2023-08-30 10:50:00 | 3398.10 | 2023-08-30 11:45:00 | 3398.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-31 09:50:00 | 3416.85 | 2023-08-31 10:00:00 | 3411.42 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-01 11:00:00 | 3384.30 | 2023-09-01 12:20:00 | 3379.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-05 10:15:00 | 3427.60 | 2023-09-05 10:35:00 | 3420.71 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-06 10:45:00 | 3425.00 | 2023-09-06 10:50:00 | 3429.50 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-09-11 10:05:00 | 3481.10 | 2023-09-11 10:25:00 | 3473.97 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-12 09:30:00 | 3502.00 | 2023-09-12 09:35:00 | 3496.66 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-20 11:10:00 | 3622.30 | 2023-09-20 11:50:00 | 3617.21 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-10-03 11:05:00 | 3501.80 | 2023-10-03 11:30:00 | 3509.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-05 10:50:00 | 3603.00 | 2023-10-05 11:00:00 | 3616.85 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-10-05 10:50:00 | 3603.00 | 2023-10-05 11:20:00 | 3603.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 09:40:00 | 3669.00 | 2023-10-09 09:55:00 | 3658.10 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-10-10 09:30:00 | 3605.00 | 2023-10-10 09:40:00 | 3613.48 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-10-16 09:30:00 | 3545.60 | 2023-10-16 09:35:00 | 3552.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-20 11:05:00 | 3491.10 | 2023-10-20 11:10:00 | 3485.61 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-10-30 10:55:00 | 3373.15 | 2023-10-30 12:15:00 | 3366.62 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-03 11:05:00 | 3364.50 | 2023-11-03 13:55:00 | 3357.53 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-11-03 11:05:00 | 3364.50 | 2023-11-03 14:35:00 | 3364.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-08 10:00:00 | 3399.70 | 2023-11-08 10:25:00 | 3394.77 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-11-10 11:00:00 | 3327.30 | 2023-11-10 12:50:00 | 3331.78 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-11-13 10:55:00 | 3334.50 | 2023-11-13 11:35:00 | 3328.66 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2023-11-13 10:55:00 | 3334.50 | 2023-11-13 14:10:00 | 3334.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 10:55:00 | 3453.75 | 2023-11-16 11:00:00 | 3462.24 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-11-16 10:55:00 | 3453.75 | 2023-11-16 15:20:00 | 3490.50 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2023-11-22 09:35:00 | 3507.45 | 2023-11-22 13:40:00 | 3518.06 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-22 09:35:00 | 3507.45 | 2023-11-22 15:20:00 | 3529.90 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2023-11-23 09:35:00 | 3539.00 | 2023-11-23 09:45:00 | 3534.10 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-11-24 10:50:00 | 3483.15 | 2023-11-24 11:35:00 | 3475.29 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-24 10:50:00 | 3483.15 | 2023-11-24 15:20:00 | 3457.60 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2023-11-30 11:05:00 | 3497.90 | 2023-11-30 13:40:00 | 3504.55 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-12-06 09:35:00 | 3553.95 | 2023-12-06 09:45:00 | 3564.41 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-06 09:35:00 | 3553.95 | 2023-12-06 10:20:00 | 3555.30 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2023-12-12 09:45:00 | 3668.00 | 2023-12-12 09:55:00 | 3677.68 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-12-12 09:45:00 | 3668.00 | 2023-12-12 14:20:00 | 3673.00 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-12-14 09:30:00 | 3672.45 | 2023-12-14 09:40:00 | 3662.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-15 09:45:00 | 3700.05 | 2023-12-15 10:00:00 | 3710.74 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-15 09:45:00 | 3700.05 | 2023-12-15 15:20:00 | 3885.60 | TARGET_HIT | 0.50 | 5.01% |
| SELL | retest1 | 2023-12-19 09:40:00 | 3809.30 | 2023-12-19 11:15:00 | 3794.44 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-19 09:40:00 | 3809.30 | 2023-12-19 11:55:00 | 3809.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 09:30:00 | 3877.40 | 2023-12-20 09:35:00 | 3895.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-12-20 09:30:00 | 3877.40 | 2023-12-20 09:50:00 | 3877.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 10:35:00 | 3833.65 | 2023-12-26 10:40:00 | 3823.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-12-27 11:10:00 | 3788.50 | 2023-12-27 11:20:00 | 3780.05 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-12-27 11:10:00 | 3788.50 | 2023-12-27 14:00:00 | 3788.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 09:30:00 | 3835.90 | 2023-12-28 09:50:00 | 3828.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-01-01 11:15:00 | 3795.55 | 2024-01-01 11:25:00 | 3803.57 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2024-01-01 11:15:00 | 3795.55 | 2024-01-01 11:40:00 | 3795.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-03 09:30:00 | 3735.10 | 2024-01-03 09:40:00 | 3722.17 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-03 09:30:00 | 3735.10 | 2024-01-03 15:20:00 | 3692.35 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-01-04 09:30:00 | 3686.65 | 2024-01-04 09:40:00 | 3675.34 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-01-04 09:30:00 | 3686.65 | 2024-01-04 13:50:00 | 3668.50 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-01-05 11:00:00 | 3730.00 | 2024-01-05 11:15:00 | 3721.27 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-08 11:05:00 | 3697.75 | 2024-01-08 11:45:00 | 3705.61 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-09 10:10:00 | 3741.80 | 2024-01-09 10:30:00 | 3733.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-01-10 11:15:00 | 3717.50 | 2024-01-10 12:05:00 | 3709.63 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-11 10:30:00 | 3750.90 | 2024-01-11 10:35:00 | 3766.45 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-01-11 10:30:00 | 3750.90 | 2024-01-11 10:40:00 | 3750.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-12 09:35:00 | 3874.65 | 2024-01-12 12:40:00 | 3899.81 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-01-12 09:35:00 | 3874.65 | 2024-01-12 14:00:00 | 3874.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:50:00 | 3913.70 | 2024-01-20 11:10:00 | 3903.66 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-01-20 10:50:00 | 3913.70 | 2024-01-20 15:20:00 | 3861.75 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2024-01-23 09:40:00 | 3929.00 | 2024-01-23 09:50:00 | 3919.11 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-25 11:05:00 | 3781.00 | 2024-01-25 11:20:00 | 3789.27 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-29 11:05:00 | 3787.55 | 2024-01-29 11:30:00 | 3796.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-01 09:35:00 | 3849.90 | 2024-02-01 09:40:00 | 3864.02 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-02-01 09:35:00 | 3849.90 | 2024-02-01 15:00:00 | 3855.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-02-02 11:10:00 | 3949.55 | 2024-02-02 11:35:00 | 3963.79 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-02-02 11:10:00 | 3949.55 | 2024-02-02 12:20:00 | 3949.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-08 09:45:00 | 4131.85 | 2024-02-08 10:15:00 | 4121.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-21 09:55:00 | 4056.25 | 2024-02-21 10:20:00 | 4044.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-28 10:00:00 | 4144.05 | 2024-02-28 10:50:00 | 4133.63 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-29 11:10:00 | 4118.40 | 2024-02-29 11:15:00 | 4106.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-04 10:55:00 | 4089.10 | 2024-03-04 11:55:00 | 4077.97 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-03-04 10:55:00 | 4089.10 | 2024-03-04 12:40:00 | 4089.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-07 10:45:00 | 4097.95 | 2024-03-07 10:55:00 | 4114.66 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-03-07 10:45:00 | 4097.95 | 2024-03-07 12:05:00 | 4097.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-21 11:00:00 | 3980.00 | 2024-03-21 11:20:00 | 3987.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-03 11:10:00 | 3963.00 | 2024-04-03 11:25:00 | 3952.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-08 10:35:00 | 3976.80 | 2024-04-08 11:10:00 | 3984.08 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-09 09:30:00 | 4013.30 | 2024-04-09 09:35:00 | 4002.92 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-18 11:05:00 | 3929.55 | 2024-04-18 11:40:00 | 3919.74 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-24 11:10:00 | 3867.50 | 2024-04-24 11:40:00 | 3860.39 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2024-04-24 11:10:00 | 3867.50 | 2024-04-24 15:20:00 | 3834.60 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2024-04-30 11:15:00 | 3844.50 | 2024-04-30 11:20:00 | 3850.26 | STOP_HIT | 1.00 | -0.15% |
