# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2025-07-10 09:15:00 → 2026-05-08 15:25:00 (15238 bars)
- **Last close:** 3326.00
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
| PARTIAL | 28 |
| TARGET_HIT | 12 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 50
- **Target hits / Stop hits / Partials:** 12 / 50 / 28
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 15.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 25 | 46.3% | 8 | 29 | 17 | 0.20% | 10.6% |
| BUY @ 2nd Alert (retest1) | 54 | 25 | 46.3% | 8 | 29 | 17 | 0.20% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 15 | 41.7% | 4 | 21 | 11 | 0.13% | 4.8% |
| SELL @ 2nd Alert (retest1) | 36 | 15 | 41.7% | 4 | 21 | 11 | 0.13% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 40 | 44.4% | 12 | 50 | 28 | 0.17% | 15.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:50:00 | 3722.70 | 3774.19 | 0.00 | ORB-short ORB[3785.00,3829.50] vol=3.6x ATR=13.67 |
| Stop hit — per-position SL triggered | 2025-07-11 12:30:00 | 3736.37 | 3754.31 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 3457.20 | 3479.82 | 0.00 | ORB-short ORB[3473.70,3504.80] vol=1.6x ATR=9.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:45:00 | 3443.54 | 3466.23 | 0.00 | T1 1.5R @ 3443.54 |
| Target hit | 2025-07-24 15:20:00 | 3360.00 | 3381.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 3370.20 | 3349.46 | 0.00 | ORB-long ORB[3318.00,3357.50] vol=1.6x ATR=15.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:55:00 | 3393.72 | 3362.09 | 0.00 | T1 1.5R @ 3393.72 |
| Stop hit — per-position SL triggered | 2025-07-25 10:20:00 | 3370.20 | 3372.35 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 09:30:00 | 3290.20 | 3313.79 | 0.00 | ORB-short ORB[3296.90,3343.90] vol=1.6x ATR=18.75 |
| Stop hit — per-position SL triggered | 2025-07-29 11:30:00 | 3308.95 | 3303.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:55:00 | 3312.00 | 3323.19 | 0.00 | ORB-short ORB[3330.00,3375.00] vol=9.6x ATR=11.05 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 3323.05 | 3321.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-08-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 09:50:00 | 3258.10 | 3243.80 | 0.00 | ORB-long ORB[3221.00,3258.00] vol=4.0x ATR=12.16 |
| Stop hit — per-position SL triggered | 2025-08-05 09:55:00 | 3245.94 | 3244.06 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-08-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:55:00 | 3176.20 | 3199.55 | 0.00 | ORB-short ORB[3182.80,3230.30] vol=2.0x ATR=9.82 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 3186.02 | 3197.82 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:40:00 | 3121.00 | 3137.51 | 0.00 | ORB-short ORB[3131.50,3160.00] vol=1.7x ATR=10.85 |
| Stop hit — per-position SL triggered | 2025-08-11 10:20:00 | 3131.85 | 3131.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:30:00 | 3101.50 | 3125.31 | 0.00 | ORB-short ORB[3117.10,3155.00] vol=2.0x ATR=9.12 |
| Stop hit — per-position SL triggered | 2025-08-12 14:20:00 | 3110.62 | 3111.80 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:55:00 | 3117.00 | 3095.81 | 0.00 | ORB-long ORB[3075.70,3111.50] vol=2.2x ATR=9.63 |
| Stop hit — per-position SL triggered | 2025-08-29 11:20:00 | 3107.37 | 3097.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:05:00 | 3178.90 | 3155.81 | 0.00 | ORB-long ORB[3128.00,3147.60] vol=3.5x ATR=9.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 12:40:00 | 3193.74 | 3168.04 | 0.00 | T1 1.5R @ 3193.74 |
| Target hit | 2025-09-01 15:20:00 | 3203.70 | 3182.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 3495.00 | 3475.64 | 0.00 | ORB-long ORB[3445.00,3489.60] vol=2.1x ATR=12.16 |
| Stop hit — per-position SL triggered | 2025-09-17 09:35:00 | 3482.84 | 3478.11 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 3395.80 | 3414.03 | 0.00 | ORB-short ORB[3412.10,3452.60] vol=4.3x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:00:00 | 3382.55 | 3409.00 | 0.00 | T1 1.5R @ 3382.55 |
| Stop hit — per-position SL triggered | 2025-09-19 10:10:00 | 3395.80 | 3408.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:55:00 | 3431.10 | 3418.01 | 0.00 | ORB-long ORB[3386.20,3426.80] vol=3.2x ATR=10.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:00:00 | 3446.46 | 3421.25 | 0.00 | T1 1.5R @ 3446.46 |
| Stop hit — per-position SL triggered | 2025-09-22 11:05:00 | 3431.10 | 3421.58 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 3403.80 | 3416.67 | 0.00 | ORB-short ORB[3410.10,3449.00] vol=2.2x ATR=12.93 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 3416.73 | 3417.03 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 09:30:00 | 3399.90 | 3386.43 | 0.00 | ORB-long ORB[3353.60,3398.00] vol=1.5x ATR=15.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:00:00 | 3423.42 | 3412.11 | 0.00 | T1 1.5R @ 3423.42 |
| Target hit | 2025-09-30 10:45:00 | 3419.70 | 3419.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 3506.20 | 3488.89 | 0.00 | ORB-long ORB[3467.30,3488.20] vol=2.0x ATR=10.64 |
| Target hit | 2025-10-03 15:20:00 | 3508.60 | 3504.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:45:00 | 3462.80 | 3471.99 | 0.00 | ORB-short ORB[3464.20,3511.10] vol=2.0x ATR=7.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:25:00 | 3451.06 | 3467.63 | 0.00 | T1 1.5R @ 3451.06 |
| Stop hit — per-position SL triggered | 2025-10-06 13:30:00 | 3462.80 | 3461.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:35:00 | 3494.50 | 3463.62 | 0.00 | ORB-long ORB[3425.80,3465.40] vol=5.6x ATR=11.11 |
| Stop hit — per-position SL triggered | 2025-10-08 10:45:00 | 3483.39 | 3470.63 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-10-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:50:00 | 3548.00 | 3578.53 | 0.00 | ORB-short ORB[3578.00,3604.00] vol=1.6x ATR=7.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:00:00 | 3536.01 | 3576.82 | 0.00 | T1 1.5R @ 3536.01 |
| Stop hit — per-position SL triggered | 2025-10-14 11:05:00 | 3548.00 | 3575.67 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:10:00 | 3484.90 | 3474.68 | 0.00 | ORB-long ORB[3445.50,3482.80] vol=2.8x ATR=10.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:40:00 | 3500.45 | 3479.01 | 0.00 | T1 1.5R @ 3500.45 |
| Target hit | 2025-10-15 15:20:00 | 3566.90 | 3516.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 3604.00 | 3585.91 | 0.00 | ORB-long ORB[3555.50,3592.50] vol=2.8x ATR=12.95 |
| Stop hit — per-position SL triggered | 2025-10-16 09:35:00 | 3591.05 | 3588.63 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:55:00 | 3751.70 | 3725.91 | 0.00 | ORB-long ORB[3700.00,3750.00] vol=1.7x ATR=14.95 |
| Stop hit — per-position SL triggered | 2025-10-17 11:00:00 | 3736.75 | 3726.07 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:15:00 | 4200.40 | 4262.40 | 0.00 | ORB-short ORB[4239.00,4285.40] vol=2.8x ATR=20.56 |
| Stop hit — per-position SL triggered | 2025-10-24 11:30:00 | 4220.96 | 4256.65 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:45:00 | 4070.90 | 4041.48 | 0.00 | ORB-long ORB[4004.80,4058.60] vol=1.7x ATR=15.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:50:00 | 4094.55 | 4050.60 | 0.00 | T1 1.5R @ 4094.55 |
| Target hit | 2025-10-30 12:45:00 | 4094.00 | 4094.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 4121.30 | 4095.21 | 0.00 | ORB-long ORB[4065.70,4109.00] vol=1.9x ATR=14.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:40:00 | 4143.48 | 4119.04 | 0.00 | T1 1.5R @ 4143.48 |
| Stop hit — per-position SL triggered | 2025-10-31 09:55:00 | 4121.30 | 4122.80 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:50:00 | 4083.30 | 4062.95 | 0.00 | ORB-long ORB[4034.30,4079.90] vol=2.4x ATR=12.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:55:00 | 4102.01 | 4116.80 | 0.00 | T1 1.5R @ 4102.01 |
| Target hit | 2025-11-04 10:45:00 | 4124.30 | 4133.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-11-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 4093.00 | 4048.41 | 0.00 | ORB-long ORB[4012.80,4050.00] vol=3.4x ATR=19.25 |
| Stop hit — per-position SL triggered | 2025-11-10 10:00:00 | 4073.75 | 4061.32 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 3934.20 | 3910.24 | 0.00 | ORB-long ORB[3871.40,3919.00] vol=2.8x ATR=11.69 |
| Stop hit — per-position SL triggered | 2025-11-21 09:40:00 | 3922.51 | 3912.97 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:20:00 | 3910.40 | 3874.13 | 0.00 | ORB-long ORB[3836.70,3890.70] vol=1.6x ATR=12.56 |
| Stop hit — per-position SL triggered | 2025-11-24 12:05:00 | 3897.84 | 3885.46 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:35:00 | 3762.70 | 3787.44 | 0.00 | ORB-short ORB[3782.30,3816.90] vol=1.6x ATR=12.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:40:00 | 3743.35 | 3782.68 | 0.00 | T1 1.5R @ 3743.35 |
| Stop hit — per-position SL triggered | 2025-12-11 09:50:00 | 3762.70 | 3779.17 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:35:00 | 3763.00 | 3790.35 | 0.00 | ORB-short ORB[3788.10,3811.10] vol=1.9x ATR=11.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 12:30:00 | 3746.24 | 3776.74 | 0.00 | T1 1.5R @ 3746.24 |
| Target hit | 2025-12-12 15:20:00 | 3720.00 | 3751.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 3840.10 | 3822.22 | 0.00 | ORB-long ORB[3801.20,3838.90] vol=2.4x ATR=15.69 |
| Stop hit — per-position SL triggered | 2025-12-16 09:55:00 | 3824.41 | 3824.84 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:45:00 | 3741.00 | 3767.94 | 0.00 | ORB-short ORB[3765.00,3789.30] vol=1.6x ATR=10.22 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 3751.22 | 3763.92 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:05:00 | 3667.00 | 3682.16 | 0.00 | ORB-short ORB[3688.00,3720.00] vol=2.6x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 3676.45 | 3681.95 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:40:00 | 3808.80 | 3768.61 | 0.00 | ORB-long ORB[3731.00,3780.80] vol=1.8x ATR=14.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 09:45:00 | 3829.98 | 3779.97 | 0.00 | T1 1.5R @ 3829.98 |
| Stop hit — per-position SL triggered | 2025-12-19 10:00:00 | 3808.80 | 3791.70 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:20:00 | 3950.50 | 3917.35 | 0.00 | ORB-long ORB[3896.40,3940.00] vol=2.9x ATR=15.04 |
| Stop hit — per-position SL triggered | 2025-12-23 10:25:00 | 3935.46 | 3920.02 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 3843.80 | 3858.08 | 0.00 | ORB-short ORB[3850.00,3882.90] vol=5.8x ATR=9.25 |
| Stop hit — per-position SL triggered | 2025-12-26 11:25:00 | 3853.05 | 3856.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 3824.10 | 3850.62 | 0.00 | ORB-short ORB[3853.00,3879.90] vol=4.0x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:30:00 | 3812.74 | 3847.59 | 0.00 | T1 1.5R @ 3812.74 |
| Stop hit — per-position SL triggered | 2025-12-29 11:50:00 | 3824.10 | 3846.19 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:55:00 | 3776.60 | 3805.62 | 0.00 | ORB-short ORB[3801.10,3833.60] vol=2.4x ATR=13.75 |
| Stop hit — per-position SL triggered | 2025-12-30 10:00:00 | 3790.35 | 3804.28 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 3791.00 | 3755.99 | 0.00 | ORB-long ORB[3712.50,3755.60] vol=1.5x ATR=11.80 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 3779.20 | 3758.59 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:50:00 | 3845.00 | 3823.90 | 0.00 | ORB-long ORB[3774.00,3823.30] vol=2.7x ATR=10.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:55:00 | 3860.28 | 3827.36 | 0.00 | T1 1.5R @ 3860.28 |
| Stop hit — per-position SL triggered | 2026-01-02 11:00:00 | 3845.00 | 3827.77 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-01-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:30:00 | 3792.00 | 3763.20 | 0.00 | ORB-long ORB[3735.00,3766.10] vol=4.2x ATR=12.10 |
| Stop hit — per-position SL triggered | 2026-01-07 10:45:00 | 3779.90 | 3765.81 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:40:00 | 3846.00 | 3824.80 | 0.00 | ORB-long ORB[3781.70,3830.00] vol=2.8x ATR=13.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:00:00 | 3866.94 | 3837.68 | 0.00 | T1 1.5R @ 3866.94 |
| Target hit | 2026-01-08 12:00:00 | 3882.00 | 3899.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2026-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:00:00 | 3792.60 | 3815.13 | 0.00 | ORB-short ORB[3812.70,3850.00] vol=2.1x ATR=12.49 |
| Stop hit — per-position SL triggered | 2026-01-16 10:05:00 | 3805.09 | 3814.49 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:30:00 | 3691.70 | 3709.73 | 0.00 | ORB-short ORB[3704.10,3734.90] vol=2.0x ATR=11.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:10:00 | 3675.06 | 3696.82 | 0.00 | T1 1.5R @ 3675.06 |
| Target hit | 2026-01-28 13:15:00 | 3684.00 | 3681.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — BUY (started 2026-02-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:55:00 | 3790.00 | 3757.83 | 0.00 | ORB-long ORB[3701.10,3750.00] vol=2.3x ATR=11.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:25:00 | 3807.60 | 3767.53 | 0.00 | T1 1.5R @ 3807.60 |
| Stop hit — per-position SL triggered | 2026-02-01 11:40:00 | 3790.00 | 3770.43 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:45:00 | 3895.10 | 3883.44 | 0.00 | ORB-long ORB[3851.50,3890.00] vol=2.7x ATR=16.27 |
| Stop hit — per-position SL triggered | 2026-02-06 09:55:00 | 3878.83 | 3883.47 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-02-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:30:00 | 3994.40 | 3972.01 | 0.00 | ORB-long ORB[3928.00,3985.00] vol=5.5x ATR=17.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:05:00 | 4020.53 | 3990.29 | 0.00 | T1 1.5R @ 4020.53 |
| Stop hit — per-position SL triggered | 2026-02-09 10:10:00 | 3994.40 | 3991.82 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 4091.90 | 4063.12 | 0.00 | ORB-long ORB[4011.00,4071.00] vol=5.2x ATR=17.81 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 4074.09 | 4064.40 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 4026.40 | 4038.68 | 0.00 | ORB-short ORB[4028.80,4066.10] vol=4.4x ATR=9.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 4012.15 | 4036.82 | 0.00 | T1 1.5R @ 4012.15 |
| Target hit | 2026-02-12 15:20:00 | 3986.20 | 4012.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 3874.30 | 3922.81 | 0.00 | ORB-short ORB[3930.00,3964.00] vol=2.6x ATR=12.89 |
| Stop hit — per-position SL triggered | 2026-02-16 10:45:00 | 3887.19 | 3911.70 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 3825.90 | 3811.72 | 0.00 | ORB-long ORB[3771.80,3822.20] vol=1.6x ATR=12.09 |
| Stop hit — per-position SL triggered | 2026-02-23 09:55:00 | 3813.81 | 3812.93 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 3727.70 | 3750.46 | 0.00 | ORB-short ORB[3742.00,3781.40] vol=1.6x ATR=12.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:55:00 | 3708.24 | 3731.28 | 0.00 | T1 1.5R @ 3708.24 |
| Stop hit — per-position SL triggered | 2026-02-24 15:05:00 | 3727.70 | 3716.11 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 3807.90 | 3781.44 | 0.00 | ORB-long ORB[3752.40,3795.00] vol=1.7x ATR=14.43 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 3793.47 | 3782.82 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 3487.60 | 3453.88 | 0.00 | ORB-long ORB[3422.00,3465.00] vol=3.8x ATR=16.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:25:00 | 3512.02 | 3471.31 | 0.00 | T1 1.5R @ 3512.02 |
| Target hit | 2026-03-18 15:20:00 | 3550.80 | 3519.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 3548.20 | 3516.51 | 0.00 | ORB-long ORB[3501.00,3542.90] vol=3.1x ATR=14.70 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 3533.50 | 3525.07 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-04-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 09:45:00 | 3367.90 | 3337.10 | 0.00 | ORB-long ORB[3302.50,3345.50] vol=2.0x ATR=16.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:50:00 | 3392.34 | 3356.43 | 0.00 | T1 1.5R @ 3392.34 |
| Stop hit — per-position SL triggered | 2026-04-07 10:55:00 | 3367.90 | 3358.56 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 3638.90 | 3611.80 | 0.00 | ORB-long ORB[3550.00,3599.40] vol=2.1x ATR=12.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:05:00 | 3657.77 | 3622.67 | 0.00 | T1 1.5R @ 3657.77 |
| Stop hit — per-position SL triggered | 2026-04-10 12:25:00 | 3638.90 | 3624.67 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-04-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:10:00 | 3602.00 | 3557.81 | 0.00 | ORB-long ORB[3514.30,3561.70] vol=1.7x ATR=12.71 |
| Stop hit — per-position SL triggered | 2026-04-13 12:55:00 | 3589.29 | 3563.19 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 3634.80 | 3659.25 | 0.00 | ORB-short ORB[3647.10,3684.40] vol=1.9x ATR=11.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:50:00 | 3617.41 | 3656.15 | 0.00 | T1 1.5R @ 3617.41 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 3634.80 | 3641.72 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 3550.10 | 3521.05 | 0.00 | ORB-long ORB[3480.20,3526.60] vol=2.0x ATR=11.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:25:00 | 3567.89 | 3530.92 | 0.00 | T1 1.5R @ 3567.89 |
| Stop hit — per-position SL triggered | 2026-04-27 14:55:00 | 3550.10 | 3541.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-11 10:50:00 | 3722.70 | 2025-07-11 12:30:00 | 3736.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-24 09:35:00 | 3457.20 | 2025-07-24 09:45:00 | 3443.54 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-24 09:35:00 | 3457.20 | 2025-07-24 15:20:00 | 3360.00 | TARGET_HIT | 0.50 | 2.81% |
| BUY | retest1 | 2025-07-25 09:30:00 | 3370.20 | 2025-07-25 09:55:00 | 3393.72 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-07-25 09:30:00 | 3370.20 | 2025-07-25 10:20:00 | 3370.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-29 09:30:00 | 3290.20 | 2025-07-29 11:30:00 | 3308.95 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-07-30 09:55:00 | 3312.00 | 2025-07-30 10:15:00 | 3323.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-05 09:50:00 | 3258.10 | 2025-08-05 09:55:00 | 3245.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-06 10:55:00 | 3176.20 | 2025-08-06 11:15:00 | 3186.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-11 09:40:00 | 3121.00 | 2025-08-11 10:20:00 | 3131.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-12 10:30:00 | 3101.50 | 2025-08-12 14:20:00 | 3110.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-29 10:55:00 | 3117.00 | 2025-08-29 11:20:00 | 3107.37 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-01 11:05:00 | 3178.90 | 2025-09-01 12:40:00 | 3193.74 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-09-01 11:05:00 | 3178.90 | 2025-09-01 15:20:00 | 3203.70 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-09-17 09:30:00 | 3495.00 | 2025-09-17 09:35:00 | 3482.84 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-19 09:55:00 | 3395.80 | 2025-09-19 10:00:00 | 3382.55 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-19 09:55:00 | 3395.80 | 2025-09-19 10:10:00 | 3395.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 10:55:00 | 3431.10 | 2025-09-22 11:00:00 | 3446.46 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-22 10:55:00 | 3431.10 | 2025-09-22 11:05:00 | 3431.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-26 09:30:00 | 3403.80 | 2025-09-26 09:40:00 | 3416.73 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-30 09:30:00 | 3399.90 | 2025-09-30 10:00:00 | 3423.42 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-09-30 09:30:00 | 3399.90 | 2025-09-30 10:45:00 | 3419.70 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-10-03 10:55:00 | 3506.20 | 2025-10-03 15:20:00 | 3508.60 | TARGET_HIT | 1.00 | 0.07% |
| SELL | retest1 | 2025-10-06 10:45:00 | 3462.80 | 2025-10-06 11:25:00 | 3451.06 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-06 10:45:00 | 3462.80 | 2025-10-06 13:30:00 | 3462.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-08 10:35:00 | 3494.50 | 2025-10-08 10:45:00 | 3483.39 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-14 10:50:00 | 3548.00 | 2025-10-14 11:00:00 | 3536.01 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-14 10:50:00 | 3548.00 | 2025-10-14 11:05:00 | 3548.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 11:10:00 | 3484.90 | 2025-10-15 11:40:00 | 3500.45 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-10-15 11:10:00 | 3484.90 | 2025-10-15 15:20:00 | 3566.90 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2025-10-16 09:30:00 | 3604.00 | 2025-10-16 09:35:00 | 3591.05 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-17 10:55:00 | 3751.70 | 2025-10-17 11:00:00 | 3736.75 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-24 11:15:00 | 4200.40 | 2025-10-24 11:30:00 | 4220.96 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-30 10:45:00 | 4070.90 | 2025-10-30 10:50:00 | 4094.55 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-10-30 10:45:00 | 4070.90 | 2025-10-30 12:45:00 | 4094.00 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-31 09:30:00 | 4121.30 | 2025-10-31 09:40:00 | 4143.48 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-31 09:30:00 | 4121.30 | 2025-10-31 09:55:00 | 4121.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 09:50:00 | 4083.30 | 2025-11-04 09:55:00 | 4102.01 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-04 09:50:00 | 4083.30 | 2025-11-04 10:45:00 | 4124.30 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2025-11-10 09:45:00 | 4093.00 | 2025-11-10 10:00:00 | 4073.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-11-21 09:35:00 | 3934.20 | 2025-11-21 09:40:00 | 3922.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-24 10:20:00 | 3910.40 | 2025-11-24 12:05:00 | 3897.84 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-11 09:35:00 | 3762.70 | 2025-12-11 09:40:00 | 3743.35 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-11 09:35:00 | 3762.70 | 2025-12-11 09:50:00 | 3762.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-12 10:35:00 | 3763.00 | 2025-12-12 12:30:00 | 3746.24 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-12 10:35:00 | 3763.00 | 2025-12-12 15:20:00 | 3720.00 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-12-16 09:40:00 | 3840.10 | 2025-12-16 09:55:00 | 3824.41 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-17 10:45:00 | 3741.00 | 2025-12-17 11:15:00 | 3751.22 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-18 11:05:00 | 3667.00 | 2025-12-18 11:15:00 | 3676.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-19 09:40:00 | 3808.80 | 2025-12-19 09:45:00 | 3829.98 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-19 09:40:00 | 3808.80 | 2025-12-19 10:00:00 | 3808.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 10:20:00 | 3950.50 | 2025-12-23 10:25:00 | 3935.46 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-26 11:05:00 | 3843.80 | 2025-12-26 11:25:00 | 3853.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-29 11:10:00 | 3824.10 | 2025-12-29 11:30:00 | 3812.74 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-29 11:10:00 | 3824.10 | 2025-12-29 11:50:00 | 3824.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 09:55:00 | 3776.60 | 2025-12-30 10:00:00 | 3790.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-31 11:00:00 | 3791.00 | 2025-12-31 11:20:00 | 3779.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-02 10:50:00 | 3845.00 | 2026-01-02 10:55:00 | 3860.28 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-02 10:50:00 | 3845.00 | 2026-01-02 11:00:00 | 3845.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 10:30:00 | 3792.00 | 2026-01-07 10:45:00 | 3779.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-08 09:40:00 | 3846.00 | 2026-01-08 10:00:00 | 3866.94 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-01-08 09:40:00 | 3846.00 | 2026-01-08 12:00:00 | 3882.00 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2026-01-16 10:00:00 | 3792.60 | 2026-01-16 10:05:00 | 3805.09 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-28 09:30:00 | 3691.70 | 2026-01-28 10:10:00 | 3675.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-28 09:30:00 | 3691.70 | 2026-01-28 13:15:00 | 3684.00 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-02-01 10:55:00 | 3790.00 | 2026-02-01 11:25:00 | 3807.60 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-01 10:55:00 | 3790.00 | 2026-02-01 11:40:00 | 3790.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-06 09:45:00 | 3895.10 | 2026-02-06 09:55:00 | 3878.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-09 09:30:00 | 3994.40 | 2026-02-09 10:05:00 | 4020.53 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-02-09 09:30:00 | 3994.40 | 2026-02-09 10:10:00 | 3994.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:40:00 | 4091.90 | 2026-02-10 09:50:00 | 4074.09 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-12 11:15:00 | 4026.40 | 2026-02-12 11:30:00 | 4012.15 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-12 11:15:00 | 4026.40 | 2026-02-12 15:20:00 | 3986.20 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-02-16 10:40:00 | 3874.30 | 2026-02-16 10:45:00 | 3887.19 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-23 09:40:00 | 3825.90 | 2026-02-23 09:55:00 | 3813.81 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-24 09:30:00 | 3727.70 | 2026-02-24 11:55:00 | 3708.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-24 09:30:00 | 3727.70 | 2026-02-24 15:05:00 | 3727.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:50:00 | 3807.90 | 2026-02-25 10:55:00 | 3793.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 09:35:00 | 3487.60 | 2026-03-18 10:25:00 | 3512.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-18 09:35:00 | 3487.60 | 2026-03-18 15:20:00 | 3550.80 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2026-03-20 11:00:00 | 3548.20 | 2026-03-20 12:15:00 | 3533.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-07 09:45:00 | 3367.90 | 2026-04-07 10:50:00 | 3392.34 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-07 09:45:00 | 3367.90 | 2026-04-07 10:55:00 | 3367.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:50:00 | 3638.90 | 2026-04-10 12:05:00 | 3657.77 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-10 10:50:00 | 3638.90 | 2026-04-10 12:25:00 | 3638.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 11:10:00 | 3602.00 | 2026-04-13 12:55:00 | 3589.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-16 09:45:00 | 3634.80 | 2026-04-16 09:50:00 | 3617.41 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-16 09:45:00 | 3634.80 | 2026-04-16 10:45:00 | 3634.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 11:05:00 | 3550.10 | 2026-04-27 12:25:00 | 3567.89 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-27 11:05:00 | 3550.10 | 2026-04-27 14:55:00 | 3550.10 | STOP_HIT | 0.50 | 0.00% |
