# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-07-04 15:25:00 (21280 bars)
- **Last close:** 4685.00
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
| ENTRY1 | 97 |
| ENTRY2 | 0 |
| PARTIAL | 45 |
| TARGET_HIT | 18 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 79
- **Target hits / Stop hits / Partials:** 18 / 79 / 45
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 21.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 41 | 47.7% | 11 | 45 | 30 | 0.17% | 14.9% |
| BUY @ 2nd Alert (retest1) | 86 | 41 | 47.7% | 11 | 45 | 30 | 0.17% | 14.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 22 | 39.3% | 7 | 34 | 15 | 0.12% | 6.6% |
| SELL @ 2nd Alert (retest1) | 56 | 22 | 39.3% | 7 | 34 | 15 | 0.12% | 6.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 142 | 63 | 44.4% | 18 | 79 | 45 | 0.15% | 21.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:40:00 | 3614.60 | 3618.01 | 0.00 | ORB-short ORB[3616.25,3649.95] vol=9.3x ATR=9.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 10:00:00 | 3600.42 | 3612.39 | 0.00 | T1 1.5R @ 3600.42 |
| Target hit | 2023-05-18 15:20:00 | 3578.45 | 3601.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 11:10:00 | 3571.95 | 3548.48 | 0.00 | ORB-long ORB[3530.05,3554.95] vol=2.6x ATR=6.25 |
| Stop hit — per-position SL triggered | 2023-05-23 11:25:00 | 3565.70 | 3550.58 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:30:00 | 3607.05 | 3594.65 | 0.00 | ORB-long ORB[3578.90,3600.05] vol=1.5x ATR=8.29 |
| Stop hit — per-position SL triggered | 2023-05-24 09:40:00 | 3598.76 | 3596.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:40:00 | 3643.00 | 3630.87 | 0.00 | ORB-long ORB[3606.90,3639.90] vol=1.8x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:45:00 | 3655.68 | 3634.65 | 0.00 | T1 1.5R @ 3655.68 |
| Target hit | 2023-05-25 10:20:00 | 3649.10 | 3653.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2023-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 09:35:00 | 3686.50 | 3700.26 | 0.00 | ORB-short ORB[3691.35,3718.20] vol=2.1x ATR=9.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:55:00 | 3672.28 | 3695.26 | 0.00 | T1 1.5R @ 3672.28 |
| Stop hit — per-position SL triggered | 2023-05-29 10:20:00 | 3686.50 | 3692.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 11:05:00 | 3712.40 | 3700.88 | 0.00 | ORB-long ORB[3680.00,3709.00] vol=2.0x ATR=7.52 |
| Stop hit — per-position SL triggered | 2023-06-06 12:20:00 | 3704.88 | 3704.08 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 11:15:00 | 3670.70 | 3686.97 | 0.00 | ORB-short ORB[3678.00,3727.10] vol=2.4x ATR=7.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:30:00 | 3659.73 | 3684.53 | 0.00 | T1 1.5R @ 3659.73 |
| Stop hit — per-position SL triggered | 2023-06-08 11:35:00 | 3670.70 | 3684.08 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 11:10:00 | 3627.10 | 3643.31 | 0.00 | ORB-short ORB[3634.25,3667.20] vol=2.8x ATR=8.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 13:10:00 | 3614.68 | 3637.59 | 0.00 | T1 1.5R @ 3614.68 |
| Target hit | 2023-06-09 15:20:00 | 3586.35 | 3609.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:15:00 | 3623.15 | 3618.82 | 0.00 | ORB-long ORB[3593.15,3622.50] vol=2.8x ATR=8.96 |
| Stop hit — per-position SL triggered | 2023-06-13 12:30:00 | 3614.19 | 3620.56 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:10:00 | 3527.80 | 3546.94 | 0.00 | ORB-short ORB[3536.75,3569.15] vol=1.7x ATR=8.66 |
| Stop hit — per-position SL triggered | 2023-06-19 11:20:00 | 3536.46 | 3546.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:05:00 | 3542.65 | 3555.75 | 0.00 | ORB-short ORB[3545.00,3585.55] vol=1.9x ATR=5.89 |
| Stop hit — per-position SL triggered | 2023-06-21 11:25:00 | 3548.54 | 3554.60 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 3598.50 | 3586.23 | 0.00 | ORB-long ORB[3568.00,3595.90] vol=1.7x ATR=8.47 |
| Stop hit — per-position SL triggered | 2023-06-22 09:50:00 | 3590.03 | 3587.55 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:35:00 | 3238.00 | 3227.81 | 0.00 | ORB-long ORB[3208.00,3234.35] vol=2.1x ATR=8.65 |
| Stop hit — per-position SL triggered | 2023-07-07 10:25:00 | 3229.35 | 3230.28 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 09:55:00 | 3214.70 | 3196.04 | 0.00 | ORB-long ORB[3175.15,3210.00] vol=1.7x ATR=8.63 |
| Stop hit — per-position SL triggered | 2023-07-10 10:20:00 | 3206.07 | 3202.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:55:00 | 3244.00 | 3218.87 | 0.00 | ORB-long ORB[3184.05,3210.00] vol=3.0x ATR=6.90 |
| Stop hit — per-position SL triggered | 2023-07-11 11:00:00 | 3237.10 | 3220.09 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:40:00 | 3273.35 | 3257.87 | 0.00 | ORB-long ORB[3235.00,3272.00] vol=1.6x ATR=9.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:25:00 | 3287.71 | 3269.53 | 0.00 | T1 1.5R @ 3287.71 |
| Target hit | 2023-07-12 12:55:00 | 3278.60 | 3284.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2023-07-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 11:00:00 | 3267.00 | 3283.13 | 0.00 | ORB-short ORB[3270.20,3305.00] vol=1.6x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 11:05:00 | 3256.94 | 3279.72 | 0.00 | T1 1.5R @ 3256.94 |
| Stop hit — per-position SL triggered | 2023-07-13 12:45:00 | 3267.00 | 3272.14 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:40:00 | 3287.45 | 3276.31 | 0.00 | ORB-long ORB[3256.00,3284.00] vol=1.7x ATR=7.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:25:00 | 3298.28 | 3281.47 | 0.00 | T1 1.5R @ 3298.28 |
| Target hit | 2023-07-14 15:20:00 | 3341.85 | 3320.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 3300.75 | 3315.58 | 0.00 | ORB-short ORB[3311.20,3343.00] vol=2.8x ATR=6.96 |
| Stop hit — per-position SL triggered | 2023-07-20 09:50:00 | 3307.71 | 3310.70 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:25:00 | 3351.60 | 3342.32 | 0.00 | ORB-long ORB[3318.15,3343.00] vol=5.0x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 10:30:00 | 3363.80 | 3344.13 | 0.00 | T1 1.5R @ 3363.80 |
| Stop hit — per-position SL triggered | 2023-07-25 10:35:00 | 3351.60 | 3344.36 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:20:00 | 3352.25 | 3343.11 | 0.00 | ORB-long ORB[3326.40,3347.45] vol=1.6x ATR=7.19 |
| Stop hit — per-position SL triggered | 2023-07-26 10:55:00 | 3345.06 | 3344.71 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:30:00 | 3396.80 | 3384.44 | 0.00 | ORB-long ORB[3367.05,3393.95] vol=2.0x ATR=8.46 |
| Stop hit — per-position SL triggered | 2023-08-01 09:40:00 | 3388.34 | 3386.09 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:50:00 | 3364.30 | 3383.44 | 0.00 | ORB-short ORB[3384.45,3430.00] vol=6.1x ATR=8.51 |
| Stop hit — per-position SL triggered | 2023-08-02 10:55:00 | 3372.81 | 3383.34 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 11:10:00 | 3363.90 | 3348.15 | 0.00 | ORB-long ORB[3301.35,3348.00] vol=1.8x ATR=10.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 11:25:00 | 3379.56 | 3350.87 | 0.00 | T1 1.5R @ 3379.56 |
| Stop hit — per-position SL triggered | 2023-08-03 13:50:00 | 3363.90 | 3361.68 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 10:30:00 | 3308.00 | 3301.98 | 0.00 | ORB-long ORB[3274.90,3299.00] vol=1.7x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:50:00 | 3316.50 | 3305.20 | 0.00 | T1 1.5R @ 3316.50 |
| Target hit | 2023-08-18 15:20:00 | 3351.45 | 3334.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2023-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 11:00:00 | 3372.05 | 3359.28 | 0.00 | ORB-long ORB[3330.30,3362.95] vol=2.2x ATR=6.78 |
| Stop hit — per-position SL triggered | 2023-08-23 11:30:00 | 3365.27 | 3361.32 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:30:00 | 3387.00 | 3377.04 | 0.00 | ORB-long ORB[3356.50,3384.35] vol=1.5x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:35:00 | 3395.76 | 3387.68 | 0.00 | T1 1.5R @ 3395.76 |
| Stop hit — per-position SL triggered | 2023-08-30 10:00:00 | 3387.00 | 3390.14 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:15:00 | 3363.95 | 3373.35 | 0.00 | ORB-short ORB[3375.00,3410.00] vol=1.6x ATR=6.67 |
| Stop hit — per-position SL triggered | 2023-08-31 11:35:00 | 3370.62 | 3372.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 3397.00 | 3405.25 | 0.00 | ORB-short ORB[3401.50,3414.95] vol=1.8x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 3388.75 | 3403.18 | 0.00 | T1 1.5R @ 3388.75 |
| Target hit | 2023-09-12 15:20:00 | 3340.30 | 3359.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 09:30:00 | 3407.00 | 3392.29 | 0.00 | ORB-long ORB[3371.55,3403.80] vol=1.5x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 09:35:00 | 3417.80 | 3406.41 | 0.00 | T1 1.5R @ 3417.80 |
| Target hit | 2023-09-15 10:20:00 | 3418.65 | 3420.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2023-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:55:00 | 3404.35 | 3407.96 | 0.00 | ORB-short ORB[3416.00,3431.80] vol=1.9x ATR=6.81 |
| Stop hit — per-position SL triggered | 2023-09-22 12:00:00 | 3411.16 | 3407.56 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:40:00 | 3456.00 | 3436.69 | 0.00 | ORB-long ORB[3401.00,3444.45] vol=2.2x ATR=9.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:55:00 | 3470.40 | 3445.46 | 0.00 | T1 1.5R @ 3470.40 |
| Stop hit — per-position SL triggered | 2023-10-09 10:00:00 | 3456.00 | 3446.33 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:35:00 | 3498.00 | 3488.16 | 0.00 | ORB-long ORB[3476.45,3493.95] vol=1.6x ATR=6.43 |
| Stop hit — per-position SL triggered | 2023-10-12 09:40:00 | 3491.57 | 3490.31 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-10-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:35:00 | 3467.95 | 3485.59 | 0.00 | ORB-short ORB[3472.30,3500.00] vol=1.6x ATR=7.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:00:00 | 3457.24 | 3481.72 | 0.00 | T1 1.5R @ 3457.24 |
| Stop hit — per-position SL triggered | 2023-10-13 14:15:00 | 3467.95 | 3467.57 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:10:00 | 3496.00 | 3487.82 | 0.00 | ORB-long ORB[3462.10,3494.85] vol=2.8x ATR=7.30 |
| Stop hit — per-position SL triggered | 2023-10-16 10:25:00 | 3488.70 | 3489.19 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:45:00 | 3510.00 | 3498.97 | 0.00 | ORB-long ORB[3487.00,3505.60] vol=2.8x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:00:00 | 3520.48 | 3504.03 | 0.00 | T1 1.5R @ 3520.48 |
| Stop hit — per-position SL triggered | 2023-10-17 10:05:00 | 3510.00 | 3504.52 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:55:00 | 3522.05 | 3513.31 | 0.00 | ORB-long ORB[3492.90,3513.00] vol=2.7x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:25:00 | 3531.64 | 3522.63 | 0.00 | T1 1.5R @ 3531.64 |
| Stop hit — per-position SL triggered | 2023-10-18 10:40:00 | 3522.05 | 3525.63 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:30:00 | 3327.15 | 3344.94 | 0.00 | ORB-short ORB[3333.00,3375.00] vol=2.2x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:55:00 | 3311.90 | 3335.43 | 0.00 | T1 1.5R @ 3311.90 |
| Stop hit — per-position SL triggered | 2023-10-26 10:00:00 | 3327.15 | 3334.98 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 09:55:00 | 3361.35 | 3356.09 | 0.00 | ORB-long ORB[3326.80,3353.55] vol=1.7x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 10:40:00 | 3374.45 | 3359.44 | 0.00 | T1 1.5R @ 3374.45 |
| Target hit | 2023-10-27 12:35:00 | 3369.75 | 3372.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2023-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 09:40:00 | 3349.60 | 3374.82 | 0.00 | ORB-short ORB[3378.00,3426.00] vol=2.2x ATR=11.05 |
| Stop hit — per-position SL triggered | 2023-10-30 10:15:00 | 3360.65 | 3365.20 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 11:05:00 | 3339.05 | 3348.51 | 0.00 | ORB-short ORB[3340.00,3367.00] vol=2.8x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:20:00 | 3327.71 | 3347.08 | 0.00 | T1 1.5R @ 3327.71 |
| Target hit | 2023-10-31 15:20:00 | 3292.10 | 3320.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-11-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 09:45:00 | 3280.00 | 3297.30 | 0.00 | ORB-short ORB[3295.55,3315.10] vol=1.6x ATR=8.36 |
| Stop hit — per-position SL triggered | 2023-11-01 10:05:00 | 3288.36 | 3292.95 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 11:10:00 | 3314.00 | 3299.52 | 0.00 | ORB-long ORB[3285.00,3310.95] vol=2.1x ATR=7.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 12:10:00 | 3325.59 | 3307.29 | 0.00 | T1 1.5R @ 3325.59 |
| Stop hit — per-position SL triggered | 2023-11-02 12:35:00 | 3314.00 | 3307.78 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:00:00 | 3376.75 | 3369.39 | 0.00 | ORB-long ORB[3353.00,3369.95] vol=1.7x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 10:45:00 | 3389.82 | 3372.94 | 0.00 | T1 1.5R @ 3389.82 |
| Target hit | 2023-11-03 15:20:00 | 3427.45 | 3417.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2023-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:30:00 | 3492.95 | 3477.59 | 0.00 | ORB-long ORB[3437.45,3488.85] vol=1.7x ATR=11.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 09:40:00 | 3510.52 | 3491.19 | 0.00 | T1 1.5R @ 3510.52 |
| Target hit | 2023-11-06 14:50:00 | 3511.15 | 3514.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2023-11-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:10:00 | 3521.25 | 3501.15 | 0.00 | ORB-long ORB[3480.10,3512.45] vol=1.8x ATR=9.32 |
| Stop hit — per-position SL triggered | 2023-11-07 11:00:00 | 3511.93 | 3505.36 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 10:15:00 | 3562.30 | 3550.68 | 0.00 | ORB-long ORB[3528.45,3550.00] vol=1.5x ATR=7.64 |
| Stop hit — per-position SL triggered | 2023-11-09 10:25:00 | 3554.66 | 3552.39 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:35:00 | 3712.40 | 3698.65 | 0.00 | ORB-long ORB[3672.10,3698.00] vol=5.9x ATR=9.71 |
| Stop hit — per-position SL triggered | 2023-11-15 09:45:00 | 3702.69 | 3699.94 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 11:00:00 | 3823.55 | 3826.20 | 0.00 | ORB-short ORB[3830.05,3854.40] vol=1.5x ATR=7.91 |
| Stop hit — per-position SL triggered | 2023-11-22 11:15:00 | 3831.46 | 3826.32 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-11-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 11:10:00 | 3857.00 | 3882.94 | 0.00 | ORB-short ORB[3883.00,3920.00] vol=4.1x ATR=7.34 |
| Stop hit — per-position SL triggered | 2023-11-24 11:20:00 | 3864.34 | 3882.11 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 11:05:00 | 3811.40 | 3849.19 | 0.00 | ORB-short ORB[3855.95,3892.15] vol=1.7x ATR=8.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:35:00 | 3798.84 | 3843.13 | 0.00 | T1 1.5R @ 3798.84 |
| Stop hit — per-position SL triggered | 2023-11-28 13:40:00 | 3811.40 | 3822.23 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:45:00 | 3884.90 | 3850.83 | 0.00 | ORB-long ORB[3828.05,3870.00] vol=2.1x ATR=11.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 11:00:00 | 3902.62 | 3862.48 | 0.00 | T1 1.5R @ 3902.62 |
| Stop hit — per-position SL triggered | 2023-11-30 12:05:00 | 3884.90 | 3878.04 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 11:10:00 | 3934.00 | 3915.29 | 0.00 | ORB-long ORB[3896.90,3924.90] vol=4.3x ATR=9.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 11:50:00 | 3948.77 | 3921.38 | 0.00 | T1 1.5R @ 3948.77 |
| Stop hit — per-position SL triggered | 2023-12-01 11:55:00 | 3934.00 | 3922.61 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:45:00 | 3958.00 | 3941.71 | 0.00 | ORB-long ORB[3919.35,3950.00] vol=1.5x ATR=12.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 10:15:00 | 3976.95 | 3954.37 | 0.00 | T1 1.5R @ 3976.95 |
| Stop hit — per-position SL triggered | 2023-12-04 10:30:00 | 3958.00 | 3955.94 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:50:00 | 4100.35 | 4073.41 | 0.00 | ORB-long ORB[4048.05,4088.20] vol=1.7x ATR=12.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:00:00 | 4119.44 | 4087.81 | 0.00 | T1 1.5R @ 4119.44 |
| Stop hit — per-position SL triggered | 2023-12-07 10:05:00 | 4100.35 | 4091.35 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:50:00 | 4074.05 | 4093.04 | 0.00 | ORB-short ORB[4095.20,4112.40] vol=1.6x ATR=8.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:05:00 | 4061.40 | 4089.59 | 0.00 | T1 1.5R @ 4061.40 |
| Target hit | 2023-12-08 14:50:00 | 4064.70 | 4061.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 4023.30 | 4003.05 | 0.00 | ORB-long ORB[3961.60,4019.00] vol=2.1x ATR=13.30 |
| Stop hit — per-position SL triggered | 2023-12-13 09:50:00 | 4010.00 | 4010.14 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:45:00 | 4071.15 | 4058.49 | 0.00 | ORB-long ORB[4044.60,4069.50] vol=2.9x ATR=8.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:55:00 | 4083.78 | 4061.72 | 0.00 | T1 1.5R @ 4083.78 |
| Stop hit — per-position SL triggered | 2023-12-27 11:00:00 | 4071.15 | 4061.87 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 09:30:00 | 4162.50 | 4127.92 | 0.00 | ORB-long ORB[4094.00,4124.40] vol=2.7x ATR=14.47 |
| Stop hit — per-position SL triggered | 2023-12-29 09:35:00 | 4148.03 | 4133.61 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 3956.85 | 3982.27 | 0.00 | ORB-short ORB[3963.70,4010.75] vol=1.7x ATR=12.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:25:00 | 3937.51 | 3972.00 | 0.00 | T1 1.5R @ 3937.51 |
| Target hit | 2024-01-02 15:20:00 | 3893.35 | 3922.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2024-01-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 10:05:00 | 3855.00 | 3869.71 | 0.00 | ORB-short ORB[3861.95,3915.00] vol=2.3x ATR=8.88 |
| Stop hit — per-position SL triggered | 2024-01-04 10:10:00 | 3863.88 | 3869.24 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:25:00 | 3915.00 | 3896.09 | 0.00 | ORB-long ORB[3864.05,3895.90] vol=2.2x ATR=7.76 |
| Stop hit — per-position SL triggered | 2024-01-05 10:40:00 | 3907.24 | 3899.07 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 3875.55 | 3893.44 | 0.00 | ORB-short ORB[3885.95,3923.95] vol=2.2x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:35:00 | 3858.91 | 3887.07 | 0.00 | T1 1.5R @ 3858.91 |
| Stop hit — per-position SL triggered | 2024-01-09 10:10:00 | 3875.55 | 3875.29 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-01-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 11:10:00 | 3854.30 | 3867.14 | 0.00 | ORB-short ORB[3859.20,3899.00] vol=2.4x ATR=6.61 |
| Stop hit — per-position SL triggered | 2024-01-12 11:20:00 | 3860.91 | 3866.93 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 11:15:00 | 3682.00 | 3720.01 | 0.00 | ORB-short ORB[3702.05,3744.00] vol=1.5x ATR=11.96 |
| Stop hit — per-position SL triggered | 2024-01-23 11:20:00 | 3693.96 | 3719.31 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-24 09:40:00 | 3581.80 | 3595.88 | 0.00 | ORB-short ORB[3601.35,3642.95] vol=6.8x ATR=16.80 |
| Stop hit — per-position SL triggered | 2024-01-24 10:15:00 | 3598.60 | 3594.62 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:45:00 | 3918.55 | 3947.58 | 0.00 | ORB-short ORB[3946.35,3990.00] vol=1.8x ATR=9.54 |
| Stop hit — per-position SL triggered | 2024-02-07 11:00:00 | 3928.09 | 3938.04 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 3853.40 | 3895.05 | 0.00 | ORB-short ORB[3905.10,3957.30] vol=2.2x ATR=10.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:15:00 | 3837.84 | 3884.64 | 0.00 | T1 1.5R @ 3837.84 |
| Target hit | 2024-02-08 15:20:00 | 3811.05 | 3839.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2024-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:55:00 | 3766.60 | 3795.73 | 0.00 | ORB-short ORB[3780.00,3820.00] vol=1.9x ATR=12.69 |
| Stop hit — per-position SL triggered | 2024-02-09 11:10:00 | 3779.29 | 3792.66 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-02-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:45:00 | 3960.00 | 3953.19 | 0.00 | ORB-long ORB[3922.00,3958.50] vol=1.7x ATR=9.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 09:55:00 | 3974.71 | 3957.98 | 0.00 | T1 1.5R @ 3974.71 |
| Stop hit — per-position SL triggered | 2024-02-16 10:20:00 | 3960.00 | 3968.54 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-02-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:45:00 | 3877.50 | 3868.13 | 0.00 | ORB-long ORB[3828.00,3876.05] vol=1.9x ATR=11.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 10:35:00 | 3894.74 | 3871.96 | 0.00 | T1 1.5R @ 3894.74 |
| Stop hit — per-position SL triggered | 2024-02-21 10:45:00 | 3877.50 | 3872.68 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 11:05:00 | 3935.40 | 3944.33 | 0.00 | ORB-short ORB[3939.65,3972.00] vol=1.7x ATR=10.26 |
| Stop hit — per-position SL triggered | 2024-02-23 11:15:00 | 3945.66 | 3944.21 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 10:50:00 | 3938.00 | 3915.75 | 0.00 | ORB-long ORB[3895.30,3927.85] vol=2.4x ATR=10.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 12:10:00 | 3954.38 | 3929.66 | 0.00 | T1 1.5R @ 3954.38 |
| Target hit | 2024-02-26 15:15:00 | 3945.70 | 3947.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2024-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:30:00 | 3999.00 | 3985.88 | 0.00 | ORB-long ORB[3940.00,3995.00] vol=3.4x ATR=11.68 |
| Stop hit — per-position SL triggered | 2024-02-27 09:35:00 | 3987.32 | 3985.93 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 3957.05 | 3973.99 | 0.00 | ORB-short ORB[3981.45,4018.95] vol=1.8x ATR=10.36 |
| Stop hit — per-position SL triggered | 2024-02-28 11:05:00 | 3967.41 | 3972.41 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:50:00 | 3807.20 | 3847.54 | 0.00 | ORB-short ORB[3841.35,3890.00] vol=2.0x ATR=13.89 |
| Stop hit — per-position SL triggered | 2024-02-29 11:10:00 | 3821.09 | 3839.25 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-03-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:25:00 | 3831.30 | 3809.32 | 0.00 | ORB-long ORB[3787.70,3829.55] vol=1.6x ATR=14.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 10:50:00 | 3852.91 | 3819.35 | 0.00 | T1 1.5R @ 3852.91 |
| Stop hit — per-position SL triggered | 2024-03-01 12:10:00 | 3831.30 | 3829.35 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-03-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:20:00 | 3748.65 | 3768.79 | 0.00 | ORB-short ORB[3751.45,3793.00] vol=1.6x ATR=10.47 |
| Stop hit — per-position SL triggered | 2024-03-05 10:30:00 | 3759.12 | 3767.63 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 10:40:00 | 3763.60 | 3786.72 | 0.00 | ORB-short ORB[3783.05,3810.00] vol=1.6x ATR=9.08 |
| Stop hit — per-position SL triggered | 2024-03-07 11:00:00 | 3772.68 | 3781.81 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 10:20:00 | 3825.00 | 3809.75 | 0.00 | ORB-long ORB[3771.00,3808.55] vol=2.5x ATR=9.04 |
| Stop hit — per-position SL triggered | 2024-03-12 10:30:00 | 3815.96 | 3814.11 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 10:25:00 | 3687.35 | 3708.95 | 0.00 | ORB-short ORB[3690.70,3745.00] vol=1.6x ATR=10.65 |
| Stop hit — per-position SL triggered | 2024-03-18 10:35:00 | 3698.00 | 3707.12 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 10:00:00 | 3890.35 | 3902.14 | 0.00 | ORB-short ORB[3899.10,3949.00] vol=1.8x ATR=11.43 |
| Stop hit — per-position SL triggered | 2024-03-22 10:05:00 | 3901.78 | 3895.54 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:45:00 | 3968.70 | 3957.13 | 0.00 | ORB-long ORB[3920.40,3943.20] vol=1.9x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 11:35:00 | 3981.42 | 3965.57 | 0.00 | T1 1.5R @ 3981.42 |
| Stop hit — per-position SL triggered | 2024-03-27 12:00:00 | 3968.70 | 3966.51 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 3965.95 | 3927.90 | 0.00 | ORB-long ORB[3891.75,3944.80] vol=2.6x ATR=14.89 |
| Stop hit — per-position SL triggered | 2024-03-28 09:50:00 | 3951.06 | 3931.97 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 10:55:00 | 3979.70 | 4032.14 | 0.00 | ORB-short ORB[4024.55,4070.00] vol=2.5x ATR=13.98 |
| Stop hit — per-position SL triggered | 2024-04-01 12:05:00 | 3993.68 | 4014.79 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 3895.65 | 3937.60 | 0.00 | ORB-short ORB[3943.15,3964.35] vol=1.8x ATR=12.29 |
| Stop hit — per-position SL triggered | 2024-04-04 09:55:00 | 3907.94 | 3934.43 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:15:00 | 4074.80 | 4047.28 | 0.00 | ORB-long ORB[4011.60,4055.00] vol=4.3x ATR=11.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 10:20:00 | 4091.41 | 4062.13 | 0.00 | T1 1.5R @ 4091.41 |
| Target hit | 2024-04-08 15:10:00 | 4206.40 | 4206.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — BUY (started 2024-04-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:25:00 | 4355.90 | 4325.98 | 0.00 | ORB-long ORB[4267.80,4332.30] vol=4.4x ATR=15.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 11:50:00 | 4379.35 | 4340.06 | 0.00 | T1 1.5R @ 4379.35 |
| Stop hit — per-position SL triggered | 2024-04-12 12:15:00 | 4355.90 | 4342.22 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-04-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 10:40:00 | 4329.35 | 4276.46 | 0.00 | ORB-long ORB[4206.30,4265.60] vol=1.6x ATR=15.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 10:55:00 | 4352.91 | 4292.98 | 0.00 | T1 1.5R @ 4352.91 |
| Target hit | 2024-04-16 12:50:00 | 4341.80 | 4342.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 90 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 10:15:00 | 4416.50 | 4395.56 | 0.00 | ORB-long ORB[4356.00,4410.80] vol=1.8x ATR=18.53 |
| Stop hit — per-position SL triggered | 2024-04-18 10:25:00 | 4397.97 | 4396.86 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 10:30:00 | 4435.00 | 4404.79 | 0.00 | ORB-long ORB[4370.40,4407.75] vol=1.5x ATR=15.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:55:00 | 4458.96 | 4420.29 | 0.00 | T1 1.5R @ 4458.96 |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 4435.00 | 4430.34 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:45:00 | 4560.05 | 4547.64 | 0.00 | ORB-long ORB[4514.80,4545.00] vol=2.2x ATR=12.41 |
| Stop hit — per-position SL triggered | 2024-04-24 11:20:00 | 4547.64 | 4548.97 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:00:00 | 4534.85 | 4553.76 | 0.00 | ORB-short ORB[4541.25,4584.90] vol=1.7x ATR=15.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:20:00 | 4512.00 | 4541.69 | 0.00 | T1 1.5R @ 4512.00 |
| Stop hit — per-position SL triggered | 2024-04-25 12:25:00 | 4534.85 | 4533.52 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-04-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 10:35:00 | 4613.45 | 4599.82 | 0.00 | ORB-long ORB[4572.00,4599.00] vol=3.1x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 10:50:00 | 4629.72 | 4601.92 | 0.00 | T1 1.5R @ 4629.72 |
| Stop hit — per-position SL triggered | 2024-04-26 10:55:00 | 4613.45 | 4601.99 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-05-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 10:00:00 | 4566.65 | 4593.47 | 0.00 | ORB-short ORB[4577.00,4634.10] vol=2.9x ATR=18.55 |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 4585.20 | 4591.07 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-05-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 10:30:00 | 4629.50 | 4627.41 | 0.00 | ORB-long ORB[4580.00,4623.00] vol=3.9x ATR=13.35 |
| Stop hit — per-position SL triggered | 2024-05-03 10:40:00 | 4616.15 | 4626.88 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:40:00 | 4642.70 | 4628.56 | 0.00 | ORB-long ORB[4605.00,4639.45] vol=2.5x ATR=14.95 |
| Stop hit — per-position SL triggered | 2024-05-07 10:10:00 | 4627.75 | 4631.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-18 09:40:00 | 3614.60 | 2023-05-18 10:00:00 | 3600.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-05-18 09:40:00 | 3614.60 | 2023-05-18 15:20:00 | 3578.45 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2023-05-23 11:10:00 | 3571.95 | 2023-05-23 11:25:00 | 3565.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-05-24 09:30:00 | 3607.05 | 2023-05-24 09:40:00 | 3598.76 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-25 09:40:00 | 3643.00 | 2023-05-25 09:45:00 | 3655.68 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-25 09:40:00 | 3643.00 | 2023-05-25 10:20:00 | 3649.10 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2023-05-29 09:35:00 | 3686.50 | 2023-05-29 09:55:00 | 3672.28 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-05-29 09:35:00 | 3686.50 | 2023-05-29 10:20:00 | 3686.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-06 11:05:00 | 3712.40 | 2023-06-06 12:20:00 | 3704.88 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-08 11:15:00 | 3670.70 | 2023-06-08 11:30:00 | 3659.73 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-06-08 11:15:00 | 3670.70 | 2023-06-08 11:35:00 | 3670.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 11:10:00 | 3627.10 | 2023-06-09 13:10:00 | 3614.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-09 11:10:00 | 3627.10 | 2023-06-09 15:20:00 | 3586.35 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2023-06-13 10:15:00 | 3623.15 | 2023-06-13 12:30:00 | 3614.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-19 11:10:00 | 3527.80 | 2023-06-19 11:20:00 | 3536.46 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-21 11:05:00 | 3542.65 | 2023-06-21 11:25:00 | 3548.54 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-22 09:40:00 | 3598.50 | 2023-06-22 09:50:00 | 3590.03 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-07 09:35:00 | 3238.00 | 2023-07-07 10:25:00 | 3229.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-10 09:55:00 | 3214.70 | 2023-07-10 10:20:00 | 3206.07 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-11 10:55:00 | 3244.00 | 2023-07-11 11:00:00 | 3237.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-12 09:40:00 | 3273.35 | 2023-07-12 10:25:00 | 3287.71 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-12 09:40:00 | 3273.35 | 2023-07-12 12:55:00 | 3278.60 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2023-07-13 11:00:00 | 3267.00 | 2023-07-13 11:05:00 | 3256.94 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-13 11:00:00 | 3267.00 | 2023-07-13 12:45:00 | 3267.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 09:40:00 | 3287.45 | 2023-07-14 10:25:00 | 3298.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-07-14 09:40:00 | 3287.45 | 2023-07-14 15:20:00 | 3341.85 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2023-07-20 09:40:00 | 3300.75 | 2023-07-20 09:50:00 | 3307.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-25 10:25:00 | 3351.60 | 2023-07-25 10:30:00 | 3363.80 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-07-25 10:25:00 | 3351.60 | 2023-07-25 10:35:00 | 3351.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-26 10:20:00 | 3352.25 | 2023-07-26 10:55:00 | 3345.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-01 09:30:00 | 3396.80 | 2023-08-01 09:40:00 | 3388.34 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-02 10:50:00 | 3364.30 | 2023-08-02 10:55:00 | 3372.81 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-03 11:10:00 | 3363.90 | 2023-08-03 11:25:00 | 3379.56 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-08-03 11:10:00 | 3363.90 | 2023-08-03 13:50:00 | 3363.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-18 10:30:00 | 3308.00 | 2023-08-18 10:50:00 | 3316.50 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-08-18 10:30:00 | 3308.00 | 2023-08-18 15:20:00 | 3351.45 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2023-08-23 11:00:00 | 3372.05 | 2023-08-23 11:30:00 | 3365.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-30 09:30:00 | 3387.00 | 2023-08-30 09:35:00 | 3395.76 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-08-30 09:30:00 | 3387.00 | 2023-08-30 10:00:00 | 3387.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-31 11:15:00 | 3363.95 | 2023-08-31 11:35:00 | 3370.62 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-12 09:30:00 | 3397.00 | 2023-09-12 09:35:00 | 3388.75 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-09-12 09:30:00 | 3397.00 | 2023-09-12 15:20:00 | 3340.30 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2023-09-15 09:30:00 | 3407.00 | 2023-09-15 09:35:00 | 3417.80 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-15 09:30:00 | 3407.00 | 2023-09-15 10:20:00 | 3418.65 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2023-09-22 10:55:00 | 3404.35 | 2023-09-22 12:00:00 | 3411.16 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-09 09:40:00 | 3456.00 | 2023-10-09 09:55:00 | 3470.40 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-10-09 09:40:00 | 3456.00 | 2023-10-09 10:00:00 | 3456.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-12 09:35:00 | 3498.00 | 2023-10-12 09:40:00 | 3491.57 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-10-13 10:35:00 | 3467.95 | 2023-10-13 11:00:00 | 3457.24 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-10-13 10:35:00 | 3467.95 | 2023-10-13 14:15:00 | 3467.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 10:10:00 | 3496.00 | 2023-10-16 10:25:00 | 3488.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-10-17 09:45:00 | 3510.00 | 2023-10-17 10:00:00 | 3520.48 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-10-17 09:45:00 | 3510.00 | 2023-10-17 10:05:00 | 3510.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 09:55:00 | 3522.05 | 2023-10-18 10:25:00 | 3531.64 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-10-18 09:55:00 | 3522.05 | 2023-10-18 10:40:00 | 3522.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-26 09:30:00 | 3327.15 | 2023-10-26 09:55:00 | 3311.90 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-10-26 09:30:00 | 3327.15 | 2023-10-26 10:00:00 | 3327.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-27 09:55:00 | 3361.35 | 2023-10-27 10:40:00 | 3374.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-10-27 09:55:00 | 3361.35 | 2023-10-27 12:35:00 | 3369.75 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2023-10-30 09:40:00 | 3349.60 | 2023-10-30 10:15:00 | 3360.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-10-31 11:05:00 | 3339.05 | 2023-10-31 11:20:00 | 3327.71 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-31 11:05:00 | 3339.05 | 2023-10-31 15:20:00 | 3292.10 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2023-11-01 09:45:00 | 3280.00 | 2023-11-01 10:05:00 | 3288.36 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-02 11:10:00 | 3314.00 | 2023-11-02 12:10:00 | 3325.59 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-02 11:10:00 | 3314.00 | 2023-11-02 12:35:00 | 3314.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 10:00:00 | 3376.75 | 2023-11-03 10:45:00 | 3389.82 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-11-03 10:00:00 | 3376.75 | 2023-11-03 15:20:00 | 3427.45 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2023-11-06 09:30:00 | 3492.95 | 2023-11-06 09:40:00 | 3510.52 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-11-06 09:30:00 | 3492.95 | 2023-11-06 14:50:00 | 3511.15 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2023-11-07 10:10:00 | 3521.25 | 2023-11-07 11:00:00 | 3511.93 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-09 10:15:00 | 3562.30 | 2023-11-09 10:25:00 | 3554.66 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-11-15 09:35:00 | 3712.40 | 2023-11-15 09:45:00 | 3702.69 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-22 11:00:00 | 3823.55 | 2023-11-22 11:15:00 | 3831.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-24 11:10:00 | 3857.00 | 2023-11-24 11:20:00 | 3864.34 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-28 11:05:00 | 3811.40 | 2023-11-28 11:35:00 | 3798.84 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-11-28 11:05:00 | 3811.40 | 2023-11-28 13:40:00 | 3811.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 10:45:00 | 3884.90 | 2023-11-30 11:00:00 | 3902.62 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-11-30 10:45:00 | 3884.90 | 2023-11-30 12:05:00 | 3884.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-01 11:10:00 | 3934.00 | 2023-12-01 11:50:00 | 3948.77 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-12-01 11:10:00 | 3934.00 | 2023-12-01 11:55:00 | 3934.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-04 09:45:00 | 3958.00 | 2023-12-04 10:15:00 | 3976.95 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-12-04 09:45:00 | 3958.00 | 2023-12-04 10:30:00 | 3958.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-07 09:50:00 | 4100.35 | 2023-12-07 10:00:00 | 4119.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-12-07 09:50:00 | 4100.35 | 2023-12-07 10:05:00 | 4100.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 10:50:00 | 4074.05 | 2023-12-08 11:05:00 | 4061.40 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-12-08 10:50:00 | 4074.05 | 2023-12-08 14:50:00 | 4064.70 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2023-12-13 09:30:00 | 4023.30 | 2023-12-13 09:50:00 | 4010.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-27 10:45:00 | 4071.15 | 2023-12-27 10:55:00 | 4083.78 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-12-27 10:45:00 | 4071.15 | 2023-12-27 11:00:00 | 4071.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 09:30:00 | 4162.50 | 2023-12-29 09:35:00 | 4148.03 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-02 09:55:00 | 3956.85 | 2024-01-02 10:25:00 | 3937.51 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-01-02 09:55:00 | 3956.85 | 2024-01-02 15:20:00 | 3893.35 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2024-01-04 10:05:00 | 3855.00 | 2024-01-04 10:10:00 | 3863.88 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-01-05 10:25:00 | 3915.00 | 2024-01-05 10:40:00 | 3907.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-09 09:30:00 | 3875.55 | 2024-01-09 09:35:00 | 3858.91 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-09 09:30:00 | 3875.55 | 2024-01-09 10:10:00 | 3875.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-12 11:10:00 | 3854.30 | 2024-01-12 11:20:00 | 3860.91 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-23 11:15:00 | 3682.00 | 2024-01-23 11:20:00 | 3693.96 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-24 09:40:00 | 3581.80 | 2024-01-24 10:15:00 | 3598.60 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-02-07 09:45:00 | 3918.55 | 2024-02-07 11:00:00 | 3928.09 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-02-08 11:00:00 | 3853.40 | 2024-02-08 11:15:00 | 3837.84 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-02-08 11:00:00 | 3853.40 | 2024-02-08 15:20:00 | 3811.05 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2024-02-09 10:55:00 | 3766.60 | 2024-02-09 11:10:00 | 3779.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-16 09:45:00 | 3960.00 | 2024-02-16 09:55:00 | 3974.71 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-02-16 09:45:00 | 3960.00 | 2024-02-16 10:20:00 | 3960.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 09:45:00 | 3877.50 | 2024-02-21 10:35:00 | 3894.74 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-21 09:45:00 | 3877.50 | 2024-02-21 10:45:00 | 3877.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-23 11:05:00 | 3935.40 | 2024-02-23 11:15:00 | 3945.66 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-02-26 10:50:00 | 3938.00 | 2024-02-26 12:10:00 | 3954.38 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-26 10:50:00 | 3938.00 | 2024-02-26 15:15:00 | 3945.70 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-02-27 09:30:00 | 3999.00 | 2024-02-27 09:35:00 | 3987.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-28 10:50:00 | 3957.05 | 2024-02-28 11:05:00 | 3967.41 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-29 10:50:00 | 3807.20 | 2024-02-29 11:10:00 | 3821.09 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-01 10:25:00 | 3831.30 | 2024-03-01 10:50:00 | 3852.91 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-03-01 10:25:00 | 3831.30 | 2024-03-01 12:10:00 | 3831.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-05 10:20:00 | 3748.65 | 2024-03-05 10:30:00 | 3759.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-07 10:40:00 | 3763.60 | 2024-03-07 11:00:00 | 3772.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-03-12 10:20:00 | 3825.00 | 2024-03-12 10:30:00 | 3815.96 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-03-18 10:25:00 | 3687.35 | 2024-03-18 10:35:00 | 3698.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-22 10:00:00 | 3890.35 | 2024-03-22 10:05:00 | 3901.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-03-27 10:45:00 | 3968.70 | 2024-03-27 11:35:00 | 3981.42 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-27 10:45:00 | 3968.70 | 2024-03-27 12:00:00 | 3968.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-28 09:45:00 | 3965.95 | 2024-03-28 09:50:00 | 3951.06 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-01 10:55:00 | 3979.70 | 2024-04-01 12:05:00 | 3993.68 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-04-04 09:50:00 | 3895.65 | 2024-04-04 09:55:00 | 3907.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-08 10:15:00 | 4074.80 | 2024-04-08 10:20:00 | 4091.41 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-04-08 10:15:00 | 4074.80 | 2024-04-08 15:10:00 | 4206.40 | TARGET_HIT | 0.50 | 3.23% |
| BUY | retest1 | 2024-04-12 10:25:00 | 4355.90 | 2024-04-12 11:50:00 | 4379.35 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-12 10:25:00 | 4355.90 | 2024-04-12 12:15:00 | 4355.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 10:40:00 | 4329.35 | 2024-04-16 10:55:00 | 4352.91 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-16 10:40:00 | 4329.35 | 2024-04-16 12:50:00 | 4341.80 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-04-18 10:15:00 | 4416.50 | 2024-04-18 10:25:00 | 4397.97 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-04-22 10:30:00 | 4435.00 | 2024-04-22 10:55:00 | 4458.96 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-22 10:30:00 | 4435.00 | 2024-04-22 11:15:00 | 4435.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 10:45:00 | 4560.05 | 2024-04-24 11:20:00 | 4547.64 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-25 10:00:00 | 4534.85 | 2024-04-25 11:20:00 | 4512.00 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-04-25 10:00:00 | 4534.85 | 2024-04-25 12:25:00 | 4534.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 10:35:00 | 4613.45 | 2024-04-26 10:50:00 | 4629.72 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-04-26 10:35:00 | 4613.45 | 2024-04-26 10:55:00 | 4613.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-02 10:00:00 | 4566.65 | 2024-05-02 10:15:00 | 4585.20 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-03 10:30:00 | 4629.50 | 2024-05-03 10:40:00 | 4616.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-07 09:40:00 | 4642.70 | 2024-05-07 10:10:00 | 4627.75 | STOP_HIT | 1.00 | -0.32% |
