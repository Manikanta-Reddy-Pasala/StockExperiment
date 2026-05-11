# TITAN (TITAN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 4517.00
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
| ENTRY1 | 93 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 13 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 80
- **Target hits / Stop hits / Partials:** 13 / 80 / 40
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 17.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 81 | 35 | 43.2% | 9 | 46 | 26 | 0.17% | 13.8% |
| BUY @ 2nd Alert (retest1) | 81 | 35 | 43.2% | 9 | 46 | 26 | 0.17% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 18 | 34.6% | 4 | 34 | 14 | 0.06% | 3.2% |
| SELL @ 2nd Alert (retest1) | 52 | 18 | 34.6% | 4 | 34 | 14 | 0.06% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 133 | 53 | 39.8% | 13 | 80 | 40 | 0.13% | 17.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:00:00 | 3299.90 | 3275.18 | 0.00 | ORB-long ORB[3250.00,3266.95] vol=1.6x ATR=7.83 |
| Stop hit — per-position SL triggered | 2024-05-14 14:35:00 | 3292.07 | 3289.39 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:00:00 | 3278.45 | 3293.43 | 0.00 | ORB-short ORB[3285.50,3305.45] vol=2.2x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:30:00 | 3266.09 | 3287.65 | 0.00 | T1 1.5R @ 3266.09 |
| Stop hit — per-position SL triggered | 2024-05-15 11:00:00 | 3278.45 | 3283.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 3264.80 | 3274.98 | 0.00 | ORB-short ORB[3272.25,3290.95] vol=2.2x ATR=6.38 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 3271.18 | 3272.12 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 3406.00 | 3396.38 | 0.00 | ORB-long ORB[3377.75,3399.00] vol=1.7x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:40:00 | 3416.77 | 3399.67 | 0.00 | T1 1.5R @ 3416.77 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 3406.00 | 3400.97 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:55:00 | 3312.75 | 3339.27 | 0.00 | ORB-short ORB[3330.35,3365.00] vol=1.5x ATR=9.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:05:00 | 3297.88 | 3335.24 | 0.00 | T1 1.5R @ 3297.88 |
| Stop hit — per-position SL triggered | 2024-05-30 10:30:00 | 3312.75 | 3330.60 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:45:00 | 3339.50 | 3321.26 | 0.00 | ORB-long ORB[3290.75,3329.00] vol=1.5x ATR=10.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 11:00:00 | 3354.59 | 3324.37 | 0.00 | T1 1.5R @ 3354.59 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 3339.50 | 3330.51 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:00:00 | 3400.00 | 3413.33 | 0.00 | ORB-short ORB[3415.10,3452.00] vol=1.8x ATR=11.03 |
| Stop hit — per-position SL triggered | 2024-06-10 10:05:00 | 3411.03 | 3413.20 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 10:00:00 | 3445.55 | 3462.05 | 0.00 | ORB-short ORB[3459.75,3490.00] vol=2.4x ATR=9.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:30:00 | 3431.43 | 3457.09 | 0.00 | T1 1.5R @ 3431.43 |
| Stop hit — per-position SL triggered | 2024-06-20 13:20:00 | 3445.55 | 3443.75 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:05:00 | 3401.40 | 3390.11 | 0.00 | ORB-long ORB[3375.00,3394.75] vol=1.5x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:55:00 | 3411.78 | 3393.23 | 0.00 | T1 1.5R @ 3411.78 |
| Target hit | 2024-06-24 15:20:00 | 3414.70 | 3404.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 3393.80 | 3403.17 | 0.00 | ORB-short ORB[3393.85,3423.40] vol=2.3x ATR=5.02 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 3398.82 | 3401.53 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 3383.00 | 3386.07 | 0.00 | ORB-short ORB[3384.00,3401.50] vol=1.9x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-06-26 12:10:00 | 3387.72 | 3384.91 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:45:00 | 3418.00 | 3395.41 | 0.00 | ORB-long ORB[3366.35,3398.00] vol=1.5x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-06-28 10:50:00 | 3410.94 | 3396.29 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 3332.85 | 3353.81 | 0.00 | ORB-short ORB[3355.40,3369.90] vol=2.9x ATR=8.70 |
| Stop hit — per-position SL triggered | 2024-07-04 09:55:00 | 3341.55 | 3349.43 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:55:00 | 3275.55 | 3296.76 | 0.00 | ORB-short ORB[3285.00,3332.00] vol=2.7x ATR=10.11 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 3285.66 | 3290.38 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:40:00 | 3208.35 | 3228.30 | 0.00 | ORB-short ORB[3215.75,3248.00] vol=1.8x ATR=9.57 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 3217.92 | 3227.52 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:10:00 | 3242.10 | 3246.38 | 0.00 | ORB-short ORB[3245.00,3266.00] vol=2.1x ATR=6.86 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 3248.96 | 3246.40 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:00:00 | 3211.20 | 3232.18 | 0.00 | ORB-short ORB[3226.40,3252.40] vol=1.9x ATR=8.19 |
| Stop hit — per-position SL triggered | 2024-07-15 10:05:00 | 3219.39 | 3231.63 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 3245.60 | 3237.64 | 0.00 | ORB-long ORB[3224.10,3238.60] vol=1.8x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 3240.29 | 3238.51 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:45:00 | 3245.70 | 3223.66 | 0.00 | ORB-long ORB[3206.00,3239.80] vol=1.7x ATR=8.13 |
| Stop hit — per-position SL triggered | 2024-07-18 10:45:00 | 3237.57 | 3232.59 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:40:00 | 3261.10 | 3249.00 | 0.00 | ORB-long ORB[3223.20,3250.00] vol=2.9x ATR=8.40 |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 3252.70 | 3251.07 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:50:00 | 3446.00 | 3414.07 | 0.00 | ORB-long ORB[3393.00,3420.95] vol=2.2x ATR=8.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:10:00 | 3459.37 | 3421.15 | 0.00 | T1 1.5R @ 3459.37 |
| Target hit | 2024-07-26 15:20:00 | 3485.50 | 3467.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:05:00 | 3455.10 | 3438.17 | 0.00 | ORB-long ORB[3410.60,3434.00] vol=1.7x ATR=7.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:15:00 | 3466.51 | 3442.08 | 0.00 | T1 1.5R @ 3466.51 |
| Stop hit — per-position SL triggered | 2024-07-30 11:35:00 | 3455.10 | 3444.06 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:10:00 | 3296.90 | 3317.90 | 0.00 | ORB-short ORB[3312.05,3350.00] vol=8.1x ATR=8.94 |
| Stop hit — per-position SL triggered | 2024-08-09 12:40:00 | 3305.84 | 3308.26 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:35:00 | 3340.90 | 3321.58 | 0.00 | ORB-long ORB[3305.40,3334.95] vol=1.7x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 10:55:00 | 3351.46 | 3326.30 | 0.00 | T1 1.5R @ 3351.46 |
| Target hit | 2024-08-13 15:20:00 | 3387.25 | 3373.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 09:35:00 | 3399.85 | 3380.28 | 0.00 | ORB-long ORB[3356.85,3390.00] vol=1.9x ATR=10.86 |
| Stop hit — per-position SL triggered | 2024-08-14 10:30:00 | 3388.99 | 3388.17 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:05:00 | 3385.20 | 3394.63 | 0.00 | ORB-short ORB[3396.65,3424.20] vol=2.0x ATR=7.79 |
| Stop hit — per-position SL triggered | 2024-08-16 11:25:00 | 3392.99 | 3394.45 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:55:00 | 3506.85 | 3484.92 | 0.00 | ORB-long ORB[3454.55,3491.85] vol=4.3x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:30:00 | 3518.88 | 3496.40 | 0.00 | T1 1.5R @ 3518.88 |
| Target hit | 2024-08-21 15:20:00 | 3557.00 | 3543.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:30:00 | 3591.00 | 3576.95 | 0.00 | ORB-long ORB[3553.70,3589.25] vol=1.9x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:55:00 | 3603.68 | 3587.20 | 0.00 | T1 1.5R @ 3603.68 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 3591.00 | 3590.62 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 3567.70 | 3584.75 | 0.00 | ORB-short ORB[3577.70,3622.85] vol=2.0x ATR=10.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:40:00 | 3552.46 | 3573.71 | 0.00 | T1 1.5R @ 3552.46 |
| Stop hit — per-position SL triggered | 2024-08-23 09:45:00 | 3567.70 | 3570.45 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 3601.60 | 3588.67 | 0.00 | ORB-long ORB[3574.05,3598.75] vol=1.7x ATR=8.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:35:00 | 3614.74 | 3593.99 | 0.00 | T1 1.5R @ 3614.74 |
| Stop hit — per-position SL triggered | 2024-08-26 09:40:00 | 3601.60 | 3595.18 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 3649.45 | 3631.53 | 0.00 | ORB-long ORB[3600.40,3635.00] vol=4.4x ATR=8.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:35:00 | 3662.58 | 3646.45 | 0.00 | T1 1.5R @ 3662.58 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 3649.45 | 3647.84 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:15:00 | 3715.15 | 3691.12 | 0.00 | ORB-long ORB[3674.00,3709.90] vol=2.8x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 12:10:00 | 3728.73 | 3700.18 | 0.00 | T1 1.5R @ 3728.73 |
| Target hit | 2024-09-10 15:20:00 | 3725.00 | 3722.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-09-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:05:00 | 3758.05 | 3736.72 | 0.00 | ORB-long ORB[3711.10,3738.00] vol=2.3x ATR=11.36 |
| Stop hit — per-position SL triggered | 2024-09-12 11:50:00 | 3746.69 | 3745.91 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 3778.50 | 3771.68 | 0.00 | ORB-long ORB[3755.20,3777.50] vol=1.6x ATR=7.39 |
| Stop hit — per-position SL triggered | 2024-09-18 09:40:00 | 3771.11 | 3772.90 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 3775.10 | 3757.75 | 0.00 | ORB-long ORB[3741.10,3760.00] vol=3.6x ATR=10.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:45:00 | 3791.43 | 3768.28 | 0.00 | T1 1.5R @ 3791.43 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 3775.10 | 3769.05 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:40:00 | 3804.90 | 3777.37 | 0.00 | ORB-long ORB[3750.15,3800.00] vol=2.3x ATR=12.64 |
| Stop hit — per-position SL triggered | 2024-09-20 10:45:00 | 3792.26 | 3780.21 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:15:00 | 3828.75 | 3812.89 | 0.00 | ORB-long ORB[3800.00,3820.00] vol=1.9x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:25:00 | 3839.82 | 3819.45 | 0.00 | T1 1.5R @ 3839.82 |
| Stop hit — per-position SL triggered | 2024-09-24 10:30:00 | 3828.75 | 3819.82 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:55:00 | 3827.30 | 3808.56 | 0.00 | ORB-long ORB[3751.20,3797.80] vol=1.8x ATR=15.47 |
| Stop hit — per-position SL triggered | 2024-09-27 10:50:00 | 3811.83 | 3815.84 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 3776.00 | 3795.55 | 0.00 | ORB-short ORB[3798.10,3837.95] vol=1.7x ATR=10.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 13:00:00 | 3760.02 | 3782.77 | 0.00 | T1 1.5R @ 3760.02 |
| Stop hit — per-position SL triggered | 2024-10-01 13:30:00 | 3776.00 | 3780.32 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:10:00 | 3531.70 | 3511.44 | 0.00 | ORB-long ORB[3500.10,3525.00] vol=2.0x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:15:00 | 3544.57 | 3514.25 | 0.00 | T1 1.5R @ 3544.57 |
| Stop hit — per-position SL triggered | 2024-10-09 11:45:00 | 3531.70 | 3520.31 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:40:00 | 3491.85 | 3516.21 | 0.00 | ORB-short ORB[3502.50,3529.00] vol=2.2x ATR=8.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 10:50:00 | 3478.38 | 3510.49 | 0.00 | T1 1.5R @ 3478.38 |
| Target hit | 2024-10-10 15:20:00 | 3453.95 | 3462.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:40:00 | 3498.70 | 3471.64 | 0.00 | ORB-long ORB[3452.55,3484.55] vol=1.8x ATR=9.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:45:00 | 3513.00 | 3489.77 | 0.00 | T1 1.5R @ 3513.00 |
| Stop hit — per-position SL triggered | 2024-10-14 12:50:00 | 3498.70 | 3497.18 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:35:00 | 3466.30 | 3483.01 | 0.00 | ORB-short ORB[3478.20,3510.00] vol=1.5x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:15:00 | 3455.91 | 3475.77 | 0.00 | T1 1.5R @ 3455.91 |
| Stop hit — per-position SL triggered | 2024-10-16 14:25:00 | 3466.30 | 3464.29 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 3455.40 | 3465.22 | 0.00 | ORB-short ORB[3455.45,3487.00] vol=1.6x ATR=9.36 |
| Stop hit — per-position SL triggered | 2024-10-17 10:30:00 | 3464.76 | 3459.47 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:35:00 | 3283.20 | 3305.84 | 0.00 | ORB-short ORB[3314.80,3342.35] vol=1.6x ATR=9.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:40:00 | 3268.25 | 3301.88 | 0.00 | T1 1.5R @ 3268.25 |
| Target hit | 2024-10-25 15:15:00 | 3272.45 | 3270.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2024-10-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:45:00 | 3242.95 | 3264.24 | 0.00 | ORB-short ORB[3251.55,3298.00] vol=1.6x ATR=11.49 |
| Stop hit — per-position SL triggered | 2024-10-29 09:50:00 | 3254.44 | 3263.25 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:10:00 | 3285.15 | 3261.58 | 0.00 | ORB-long ORB[3235.10,3274.75] vol=1.6x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:35:00 | 3296.50 | 3266.16 | 0.00 | T1 1.5R @ 3296.50 |
| Target hit | 2024-10-30 15:20:00 | 3309.75 | 3294.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 3260.10 | 3271.36 | 0.00 | ORB-short ORB[3262.90,3299.90] vol=2.3x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-10-31 11:30:00 | 3266.97 | 3270.51 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 3210.05 | 3193.15 | 0.00 | ORB-long ORB[3158.25,3200.90] vol=1.8x ATR=9.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:15:00 | 3223.68 | 3199.83 | 0.00 | T1 1.5R @ 3223.68 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 3210.05 | 3202.07 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 3227.40 | 3220.74 | 0.00 | ORB-long ORB[3200.00,3226.00] vol=2.1x ATR=9.35 |
| Stop hit — per-position SL triggered | 2024-11-12 09:35:00 | 3218.05 | 3220.92 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:20:00 | 3249.05 | 3215.03 | 0.00 | ORB-long ORB[3172.30,3203.65] vol=2.4x ATR=8.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:35:00 | 3261.37 | 3227.91 | 0.00 | T1 1.5R @ 3261.37 |
| Stop hit — per-position SL triggered | 2024-11-19 14:30:00 | 3249.05 | 3249.34 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 3268.40 | 3290.21 | 0.00 | ORB-short ORB[3278.05,3300.70] vol=2.0x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:55:00 | 3255.23 | 3284.33 | 0.00 | T1 1.5R @ 3255.23 |
| Target hit | 2024-11-28 15:20:00 | 3221.55 | 3234.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2024-12-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:05:00 | 3267.90 | 3252.95 | 0.00 | ORB-long ORB[3222.05,3249.00] vol=1.7x ATR=9.41 |
| Stop hit — per-position SL triggered | 2024-12-02 10:25:00 | 3258.49 | 3254.35 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 11:00:00 | 3315.10 | 3303.50 | 0.00 | ORB-long ORB[3276.60,3304.40] vol=1.6x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 12:00:00 | 3324.80 | 3307.36 | 0.00 | T1 1.5R @ 3324.80 |
| Target hit | 2024-12-03 15:20:00 | 3328.00 | 3322.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2024-12-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:25:00 | 3369.85 | 3352.55 | 0.00 | ORB-long ORB[3340.50,3358.05] vol=2.1x ATR=7.97 |
| Stop hit — per-position SL triggered | 2024-12-04 10:45:00 | 3361.88 | 3355.14 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:05:00 | 3472.90 | 3447.85 | 0.00 | ORB-long ORB[3430.40,3453.95] vol=1.9x ATR=9.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:40:00 | 3487.45 | 3460.99 | 0.00 | T1 1.5R @ 3487.45 |
| Stop hit — per-position SL triggered | 2024-12-06 14:40:00 | 3472.90 | 3472.12 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:15:00 | 3502.45 | 3471.69 | 0.00 | ORB-long ORB[3436.05,3485.00] vol=1.5x ATR=8.70 |
| Stop hit — per-position SL triggered | 2024-12-10 10:25:00 | 3493.75 | 3473.19 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 3433.80 | 3445.66 | 0.00 | ORB-short ORB[3439.00,3473.10] vol=2.2x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:40:00 | 3422.73 | 3440.46 | 0.00 | T1 1.5R @ 3422.73 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 3433.80 | 3435.23 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:45:00 | 3395.35 | 3411.22 | 0.00 | ORB-short ORB[3418.05,3437.95] vol=1.5x ATR=8.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 11:05:00 | 3382.50 | 3409.35 | 0.00 | T1 1.5R @ 3382.50 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 3395.35 | 3408.25 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 3458.20 | 3475.51 | 0.00 | ORB-short ORB[3478.00,3511.00] vol=2.1x ATR=9.35 |
| Stop hit — per-position SL triggered | 2024-12-16 09:40:00 | 3467.55 | 3474.41 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 3449.30 | 3431.06 | 0.00 | ORB-long ORB[3417.00,3441.05] vol=1.5x ATR=8.55 |
| Stop hit — per-position SL triggered | 2024-12-17 09:45:00 | 3440.75 | 3431.89 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 3385.00 | 3360.07 | 0.00 | ORB-long ORB[3333.15,3368.00] vol=1.8x ATR=9.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:35:00 | 3399.35 | 3369.37 | 0.00 | T1 1.5R @ 3399.35 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 3385.00 | 3381.25 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:55:00 | 3400.20 | 3366.64 | 0.00 | ORB-long ORB[3357.05,3389.95] vol=3.4x ATR=12.32 |
| Stop hit — per-position SL triggered | 2024-12-23 12:10:00 | 3387.88 | 3379.47 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:45:00 | 3348.95 | 3355.48 | 0.00 | ORB-short ORB[3350.05,3371.85] vol=1.9x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 3356.81 | 3355.20 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 3274.05 | 3263.22 | 0.00 | ORB-long ORB[3251.00,3267.55] vol=2.6x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:20:00 | 3284.26 | 3267.53 | 0.00 | T1 1.5R @ 3284.26 |
| Target hit | 2025-01-02 15:20:00 | 3396.50 | 3339.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 3444.75 | 3419.11 | 0.00 | ORB-long ORB[3377.95,3412.25] vol=1.7x ATR=11.94 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 3432.81 | 3428.96 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 3356.30 | 3378.94 | 0.00 | ORB-short ORB[3393.15,3429.00] vol=1.8x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 11:20:00 | 3342.37 | 3373.17 | 0.00 | T1 1.5R @ 3342.37 |
| Target hit | 2025-01-14 15:20:00 | 3324.40 | 3339.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-01-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 11:05:00 | 3292.30 | 3301.88 | 0.00 | ORB-short ORB[3309.00,3340.50] vol=2.2x ATR=9.10 |
| Stop hit — per-position SL triggered | 2025-01-15 11:15:00 | 3301.40 | 3300.89 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:30:00 | 3291.55 | 3310.84 | 0.00 | ORB-short ORB[3305.65,3338.85] vol=1.7x ATR=7.84 |
| Stop hit — per-position SL triggered | 2025-01-16 10:40:00 | 3299.39 | 3308.84 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-01-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:05:00 | 3388.65 | 3365.58 | 0.00 | ORB-long ORB[3347.50,3375.00] vol=2.2x ATR=9.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:15:00 | 3402.29 | 3369.56 | 0.00 | T1 1.5R @ 3402.29 |
| Stop hit — per-position SL triggered | 2025-01-20 12:40:00 | 3388.65 | 3382.86 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:55:00 | 3350.90 | 3371.11 | 0.00 | ORB-short ORB[3385.00,3406.50] vol=1.9x ATR=9.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:25:00 | 3336.48 | 3364.60 | 0.00 | T1 1.5R @ 3336.48 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 3350.90 | 3367.72 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 11:05:00 | 3371.40 | 3366.89 | 0.00 | ORB-long ORB[3343.10,3362.35] vol=2.2x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-01-22 11:10:00 | 3361.95 | 3366.65 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:15:00 | 3389.45 | 3399.52 | 0.00 | ORB-short ORB[3394.05,3421.50] vol=2.6x ATR=9.00 |
| Stop hit — per-position SL triggered | 2025-01-24 10:20:00 | 3398.45 | 3398.73 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 11:00:00 | 3337.60 | 3318.27 | 0.00 | ORB-long ORB[3306.60,3333.00] vol=2.9x ATR=11.47 |
| Stop hit — per-position SL triggered | 2025-01-28 11:55:00 | 3326.13 | 3320.87 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 3349.35 | 3345.39 | 0.00 | ORB-long ORB[3309.05,3348.00] vol=4.6x ATR=8.94 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 3340.41 | 3344.88 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-01-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:40:00 | 3384.45 | 3368.66 | 0.00 | ORB-long ORB[3346.55,3379.00] vol=2.0x ATR=8.32 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 3376.13 | 3370.60 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:05:00 | 3196.55 | 3227.46 | 0.00 | ORB-short ORB[3221.85,3253.00] vol=2.2x ATR=8.69 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 3205.24 | 3226.42 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:55:00 | 3194.55 | 3171.03 | 0.00 | ORB-long ORB[3136.55,3166.95] vol=1.6x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 11:20:00 | 3204.93 | 3175.56 | 0.00 | T1 1.5R @ 3204.93 |
| Stop hit — per-position SL triggered | 2025-02-25 12:35:00 | 3194.55 | 3183.94 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:15:00 | 3190.20 | 3194.73 | 0.00 | ORB-short ORB[3194.90,3217.95] vol=1.8x ATR=6.06 |
| Stop hit — per-position SL triggered | 2025-02-27 11:30:00 | 3196.26 | 3194.69 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 09:35:00 | 3026.10 | 3040.46 | 0.00 | ORB-short ORB[3031.05,3061.95] vol=1.8x ATR=9.41 |
| Stop hit — per-position SL triggered | 2025-03-04 09:45:00 | 3035.51 | 3035.85 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:05:00 | 3087.80 | 3067.99 | 0.00 | ORB-long ORB[3035.70,3068.00] vol=1.6x ATR=7.19 |
| Stop hit — per-position SL triggered | 2025-03-05 11:25:00 | 3080.61 | 3069.84 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 3058.90 | 3078.58 | 0.00 | ORB-short ORB[3068.00,3100.00] vol=1.8x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 09:40:00 | 3047.80 | 3071.85 | 0.00 | T1 1.5R @ 3047.80 |
| Stop hit — per-position SL triggered | 2025-03-06 09:45:00 | 3058.90 | 3070.25 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 3055.00 | 3038.22 | 0.00 | ORB-long ORB[3010.00,3046.95] vol=2.1x ATR=8.75 |
| Stop hit — per-position SL triggered | 2025-03-11 11:10:00 | 3046.25 | 3038.79 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 3040.90 | 3023.83 | 0.00 | ORB-long ORB[3002.40,3034.50] vol=1.9x ATR=7.99 |
| Stop hit — per-position SL triggered | 2025-03-13 09:40:00 | 3032.91 | 3024.75 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 3124.40 | 3106.14 | 0.00 | ORB-long ORB[3080.00,3120.00] vol=2.1x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-03-20 09:35:00 | 3117.74 | 3107.60 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:05:00 | 3009.05 | 3041.95 | 0.00 | ORB-short ORB[3025.00,3061.25] vol=1.8x ATR=10.12 |
| Stop hit — per-position SL triggered | 2025-04-01 11:15:00 | 3019.17 | 3039.06 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-04-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:50:00 | 3013.00 | 2993.80 | 0.00 | ORB-long ORB[2981.10,3011.70] vol=1.9x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 11:10:00 | 3023.96 | 2998.13 | 0.00 | T1 1.5R @ 3023.96 |
| Target hit | 2025-04-02 15:20:00 | 3094.00 | 3056.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2025-04-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 09:35:00 | 3129.00 | 3104.50 | 0.00 | ORB-long ORB[3068.00,3113.60] vol=1.5x ATR=10.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:45:00 | 3145.42 | 3115.74 | 0.00 | T1 1.5R @ 3145.42 |
| Stop hit — per-position SL triggered | 2025-04-03 10:45:00 | 3129.00 | 3129.55 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:15:00 | 3344.30 | 3317.03 | 0.00 | ORB-long ORB[3290.00,3321.30] vol=1.9x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-04-21 11:30:00 | 3337.93 | 3318.91 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 3355.90 | 3335.79 | 0.00 | ORB-long ORB[3322.00,3344.00] vol=1.7x ATR=8.75 |
| Stop hit — per-position SL triggered | 2025-04-22 09:35:00 | 3347.15 | 3337.25 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 3375.20 | 3385.52 | 0.00 | ORB-short ORB[3375.90,3398.00] vol=1.7x ATR=7.74 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 3382.94 | 3384.90 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-05-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 10:00:00 | 3303.00 | 3278.98 | 0.00 | ORB-long ORB[3262.90,3284.60] vol=1.6x ATR=11.45 |
| Stop hit — per-position SL triggered | 2025-05-07 10:10:00 | 3291.55 | 3282.25 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:05:00 | 3397.60 | 3361.95 | 0.00 | ORB-long ORB[3325.00,3349.90] vol=3.8x ATR=10.85 |
| Stop hit — per-position SL triggered | 2025-05-08 11:10:00 | 3386.75 | 3363.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:00:00 | 3299.90 | 2024-05-14 14:35:00 | 3292.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-15 10:00:00 | 3278.45 | 2024-05-15 10:30:00 | 3266.09 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-05-15 10:00:00 | 3278.45 | 2024-05-15 11:00:00 | 3278.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 09:30:00 | 3264.80 | 2024-05-16 09:40:00 | 3271.18 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-23 09:35:00 | 3406.00 | 2024-05-23 09:40:00 | 3416.77 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-05-23 09:35:00 | 3406.00 | 2024-05-23 09:50:00 | 3406.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 09:55:00 | 3312.75 | 2024-05-30 10:05:00 | 3297.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-30 09:55:00 | 3312.75 | 2024-05-30 10:30:00 | 3312.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 10:45:00 | 3339.50 | 2024-06-06 11:00:00 | 3354.59 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-06 10:45:00 | 3339.50 | 2024-06-06 11:30:00 | 3339.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 10:00:00 | 3400.00 | 2024-06-10 10:05:00 | 3411.03 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-20 10:00:00 | 3445.55 | 2024-06-20 10:30:00 | 3431.43 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-20 10:00:00 | 3445.55 | 2024-06-20 13:20:00 | 3445.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 11:05:00 | 3401.40 | 2024-06-24 11:55:00 | 3411.78 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-06-24 11:05:00 | 3401.40 | 2024-06-24 15:20:00 | 3414.70 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-25 11:00:00 | 3393.80 | 2024-06-25 11:20:00 | 3398.82 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-06-26 11:15:00 | 3383.00 | 2024-06-26 12:10:00 | 3387.72 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-06-28 10:45:00 | 3418.00 | 2024-06-28 10:50:00 | 3410.94 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-04 09:45:00 | 3332.85 | 2024-07-04 09:55:00 | 3341.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-05 09:55:00 | 3275.55 | 2024-07-05 10:35:00 | 3285.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:40:00 | 3208.35 | 2024-07-10 10:55:00 | 3217.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-12 10:10:00 | 3242.10 | 2024-07-12 10:50:00 | 3248.96 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-15 10:00:00 | 3211.20 | 2024-07-15 10:05:00 | 3219.39 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-16 10:30:00 | 3245.60 | 2024-07-16 10:45:00 | 3240.29 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-18 09:45:00 | 3245.70 | 2024-07-18 10:45:00 | 3237.57 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-22 10:40:00 | 3261.10 | 2024-07-22 11:15:00 | 3252.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-26 10:50:00 | 3446.00 | 2024-07-26 11:10:00 | 3459.37 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-26 10:50:00 | 3446.00 | 2024-07-26 15:20:00 | 3485.50 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-07-30 11:05:00 | 3455.10 | 2024-07-30 11:15:00 | 3466.51 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-07-30 11:05:00 | 3455.10 | 2024-07-30 11:35:00 | 3455.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-09 11:10:00 | 3296.90 | 2024-08-09 12:40:00 | 3305.84 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-13 10:35:00 | 3340.90 | 2024-08-13 10:55:00 | 3351.46 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-13 10:35:00 | 3340.90 | 2024-08-13 15:20:00 | 3387.25 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2024-08-14 09:35:00 | 3399.85 | 2024-08-14 10:30:00 | 3388.99 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-16 11:05:00 | 3385.20 | 2024-08-16 11:25:00 | 3392.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-21 10:55:00 | 3506.85 | 2024-08-21 11:30:00 | 3518.88 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-21 10:55:00 | 3506.85 | 2024-08-21 15:20:00 | 3557.00 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-08-22 10:30:00 | 3591.00 | 2024-08-22 10:55:00 | 3603.68 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-22 10:30:00 | 3591.00 | 2024-08-22 11:15:00 | 3591.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 09:30:00 | 3567.70 | 2024-08-23 09:40:00 | 3552.46 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-23 09:30:00 | 3567.70 | 2024-08-23 09:45:00 | 3567.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:30:00 | 3601.60 | 2024-08-26 09:35:00 | 3614.74 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-26 09:30:00 | 3601.60 | 2024-08-26 09:40:00 | 3601.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 09:30:00 | 3649.45 | 2024-09-05 09:35:00 | 3662.58 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-09-05 09:30:00 | 3649.45 | 2024-09-05 09:50:00 | 3649.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 11:15:00 | 3715.15 | 2024-09-10 12:10:00 | 3728.73 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-10 11:15:00 | 3715.15 | 2024-09-10 15:20:00 | 3725.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2024-09-12 10:05:00 | 3758.05 | 2024-09-12 11:50:00 | 3746.69 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-18 09:30:00 | 3778.50 | 2024-09-18 09:40:00 | 3771.11 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-19 09:30:00 | 3775.10 | 2024-09-19 09:45:00 | 3791.43 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-19 09:30:00 | 3775.10 | 2024-09-19 09:50:00 | 3775.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 10:40:00 | 3804.90 | 2024-09-20 10:45:00 | 3792.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-24 10:15:00 | 3828.75 | 2024-09-24 10:25:00 | 3839.82 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-24 10:15:00 | 3828.75 | 2024-09-24 10:30:00 | 3828.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 09:55:00 | 3827.30 | 2024-09-27 10:50:00 | 3811.83 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-01 09:35:00 | 3776.00 | 2024-10-01 13:00:00 | 3760.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-01 09:35:00 | 3776.00 | 2024-10-01 13:30:00 | 3776.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 11:10:00 | 3531.70 | 2024-10-09 11:15:00 | 3544.57 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-10-09 11:10:00 | 3531.70 | 2024-10-09 11:45:00 | 3531.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 10:40:00 | 3491.85 | 2024-10-10 10:50:00 | 3478.38 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-10-10 10:40:00 | 3491.85 | 2024-10-10 15:20:00 | 3453.95 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2024-10-14 09:40:00 | 3498.70 | 2024-10-14 10:45:00 | 3513.00 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-14 09:40:00 | 3498.70 | 2024-10-14 12:50:00 | 3498.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 10:35:00 | 3466.30 | 2024-10-16 11:15:00 | 3455.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-10-16 10:35:00 | 3466.30 | 2024-10-16 14:25:00 | 3466.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:35:00 | 3455.40 | 2024-10-17 10:30:00 | 3464.76 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-25 10:35:00 | 3283.20 | 2024-10-25 10:40:00 | 3268.25 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-25 10:35:00 | 3283.20 | 2024-10-25 15:15:00 | 3272.45 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-10-29 09:45:00 | 3242.95 | 2024-10-29 09:50:00 | 3254.44 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-30 11:10:00 | 3285.15 | 2024-10-30 11:35:00 | 3296.50 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-10-30 11:10:00 | 3285.15 | 2024-10-30 15:20:00 | 3309.75 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-10-31 11:15:00 | 3260.10 | 2024-10-31 11:30:00 | 3266.97 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-11-11 10:55:00 | 3210.05 | 2024-11-11 11:15:00 | 3223.68 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-11 10:55:00 | 3210.05 | 2024-11-11 12:00:00 | 3210.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 09:30:00 | 3227.40 | 2024-11-12 09:35:00 | 3218.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-19 10:20:00 | 3249.05 | 2024-11-19 10:35:00 | 3261.37 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-11-19 10:20:00 | 3249.05 | 2024-11-19 14:30:00 | 3249.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 10:35:00 | 3268.40 | 2024-11-28 10:55:00 | 3255.23 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-28 10:35:00 | 3268.40 | 2024-11-28 15:20:00 | 3221.55 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-12-02 10:05:00 | 3267.90 | 2024-12-02 10:25:00 | 3258.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-03 11:00:00 | 3315.10 | 2024-12-03 12:00:00 | 3324.80 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-12-03 11:00:00 | 3315.10 | 2024-12-03 15:20:00 | 3328.00 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-04 10:25:00 | 3369.85 | 2024-12-04 10:45:00 | 3361.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-06 11:05:00 | 3472.90 | 2024-12-06 11:40:00 | 3487.45 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-06 11:05:00 | 3472.90 | 2024-12-06 14:40:00 | 3472.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 10:15:00 | 3502.45 | 2024-12-10 10:25:00 | 3493.75 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-12 09:30:00 | 3433.80 | 2024-12-12 09:40:00 | 3422.73 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-12 09:30:00 | 3433.80 | 2024-12-12 09:50:00 | 3433.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:45:00 | 3395.35 | 2024-12-13 11:05:00 | 3382.50 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-13 10:45:00 | 3395.35 | 2024-12-13 11:10:00 | 3395.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 09:35:00 | 3458.20 | 2024-12-16 09:40:00 | 3467.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-17 09:40:00 | 3449.30 | 2024-12-17 09:45:00 | 3440.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-20 09:30:00 | 3385.00 | 2024-12-20 09:35:00 | 3399.35 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-20 09:30:00 | 3385.00 | 2024-12-20 09:45:00 | 3385.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-23 10:55:00 | 3400.20 | 2024-12-23 12:10:00 | 3387.88 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-26 09:45:00 | 3348.95 | 2024-12-26 09:55:00 | 3356.81 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-02 10:45:00 | 3274.05 | 2025-01-02 11:20:00 | 3284.26 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-01-02 10:45:00 | 3274.05 | 2025-01-02 15:20:00 | 3396.50 | TARGET_HIT | 0.50 | 3.74% |
| BUY | retest1 | 2025-01-03 09:55:00 | 3444.75 | 2025-01-03 10:15:00 | 3432.81 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-14 11:00:00 | 3356.30 | 2025-01-14 11:20:00 | 3342.37 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-14 11:00:00 | 3356.30 | 2025-01-14 15:20:00 | 3324.40 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-01-15 11:05:00 | 3292.30 | 2025-01-15 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-16 10:30:00 | 3291.55 | 2025-01-16 10:40:00 | 3299.39 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-20 10:05:00 | 3388.65 | 2025-01-20 10:15:00 | 3402.29 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-20 10:05:00 | 3388.65 | 2025-01-20 12:40:00 | 3388.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 10:55:00 | 3350.90 | 2025-01-21 11:25:00 | 3336.48 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-21 10:55:00 | 3350.90 | 2025-01-21 11:45:00 | 3350.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-22 11:05:00 | 3371.40 | 2025-01-22 11:10:00 | 3361.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-24 10:15:00 | 3389.45 | 2025-01-24 10:20:00 | 3398.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-28 11:00:00 | 3337.60 | 2025-01-28 11:55:00 | 3326.13 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-29 11:10:00 | 3349.35 | 2025-01-29 11:20:00 | 3340.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-30 10:40:00 | 3384.45 | 2025-01-30 10:50:00 | 3376.13 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-14 11:05:00 | 3196.55 | 2025-02-14 11:10:00 | 3205.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-25 10:55:00 | 3194.55 | 2025-02-25 11:20:00 | 3204.93 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-02-25 10:55:00 | 3194.55 | 2025-02-25 12:35:00 | 3194.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-27 11:15:00 | 3190.20 | 2025-02-27 11:30:00 | 3196.26 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-03-04 09:35:00 | 3026.10 | 2025-03-04 09:45:00 | 3035.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-05 11:05:00 | 3087.80 | 2025-03-05 11:25:00 | 3080.61 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-06 09:30:00 | 3058.90 | 2025-03-06 09:40:00 | 3047.80 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-03-06 09:30:00 | 3058.90 | 2025-03-06 09:45:00 | 3058.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 11:00:00 | 3055.00 | 2025-03-11 11:10:00 | 3046.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-13 09:35:00 | 3040.90 | 2025-03-13 09:40:00 | 3032.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-20 09:30:00 | 3124.40 | 2025-03-20 09:35:00 | 3117.74 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-04-01 11:05:00 | 3009.05 | 2025-04-01 11:15:00 | 3019.17 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-02 10:50:00 | 3013.00 | 2025-04-02 11:10:00 | 3023.96 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-04-02 10:50:00 | 3013.00 | 2025-04-02 15:20:00 | 3094.00 | TARGET_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2025-04-03 09:35:00 | 3129.00 | 2025-04-03 09:45:00 | 3145.42 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-03 09:35:00 | 3129.00 | 2025-04-03 10:45:00 | 3129.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 11:15:00 | 3344.30 | 2025-04-21 11:30:00 | 3337.93 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-04-22 09:30:00 | 3355.90 | 2025-04-22 09:35:00 | 3347.15 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-29 09:40:00 | 3375.20 | 2025-04-29 09:45:00 | 3382.94 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-07 10:00:00 | 3303.00 | 2025-05-07 10:10:00 | 3291.55 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-08 11:05:00 | 3397.60 | 2025-05-08 11:10:00 | 3386.75 | STOP_HIT | 1.00 | -0.32% |
