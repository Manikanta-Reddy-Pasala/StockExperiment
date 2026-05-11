# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 4707.00
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 15 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 63
- **Target hits / Stop hits / Partials:** 15 / 63 / 30
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 14.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 23 | 39.7% | 8 | 35 | 15 | 0.13% | 7.4% |
| BUY @ 2nd Alert (retest1) | 58 | 23 | 39.7% | 8 | 35 | 15 | 0.13% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 50 | 22 | 44.0% | 7 | 28 | 15 | 0.15% | 7.5% |
| SELL @ 2nd Alert (retest1) | 50 | 22 | 44.0% | 7 | 28 | 15 | 0.15% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 45 | 41.7% | 15 | 63 | 30 | 0.14% | 14.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 3356.90 | 3325.87 | 0.00 | ORB-long ORB[3289.80,3321.30] vol=1.9x ATR=12.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 09:50:00 | 3375.50 | 3336.13 | 0.00 | T1 1.5R @ 3375.50 |
| Stop hit — per-position SL triggered | 2025-05-13 10:15:00 | 3356.90 | 3343.22 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:25:00 | 3473.80 | 3440.38 | 0.00 | ORB-long ORB[3405.10,3449.80] vol=2.0x ATR=11.51 |
| Stop hit — per-position SL triggered | 2025-05-15 11:00:00 | 3462.29 | 3448.10 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 11:00:00 | 3488.00 | 3473.83 | 0.00 | ORB-long ORB[3458.50,3483.30] vol=1.7x ATR=9.07 |
| Stop hit — per-position SL triggered | 2025-05-28 11:20:00 | 3478.93 | 3478.12 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:15:00 | 3417.50 | 3440.22 | 0.00 | ORB-short ORB[3438.30,3470.00] vol=3.1x ATR=10.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 12:55:00 | 3401.95 | 3434.52 | 0.00 | T1 1.5R @ 3401.95 |
| Target hit | 2025-06-03 15:20:00 | 3388.40 | 3413.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:25:00 | 3475.00 | 3442.89 | 0.00 | ORB-long ORB[3384.90,3429.90] vol=6.1x ATR=15.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:00:00 | 3498.09 | 3452.29 | 0.00 | T1 1.5R @ 3498.09 |
| Stop hit — per-position SL triggered | 2025-06-04 11:05:00 | 3475.00 | 3453.10 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 11:05:00 | 3554.50 | 3542.10 | 0.00 | ORB-long ORB[3506.40,3550.00] vol=4.3x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-06-05 13:20:00 | 3545.05 | 3545.06 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:15:00 | 3573.90 | 3560.71 | 0.00 | ORB-long ORB[3539.50,3569.90] vol=1.8x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:35:00 | 3584.34 | 3564.55 | 0.00 | T1 1.5R @ 3584.34 |
| Stop hit — per-position SL triggered | 2025-06-12 11:45:00 | 3573.90 | 3565.00 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 3589.00 | 3551.66 | 0.00 | ORB-long ORB[3509.50,3559.00] vol=4.8x ATR=16.05 |
| Stop hit — per-position SL triggered | 2025-06-19 09:35:00 | 3572.95 | 3555.34 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:50:00 | 3485.00 | 3508.29 | 0.00 | ORB-short ORB[3501.00,3547.70] vol=1.6x ATR=16.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:55:00 | 3460.85 | 3500.92 | 0.00 | T1 1.5R @ 3460.85 |
| Stop hit — per-position SL triggered | 2025-06-20 10:00:00 | 3485.00 | 3499.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 3495.60 | 3475.32 | 0.00 | ORB-long ORB[3445.00,3479.00] vol=3.8x ATR=12.71 |
| Stop hit — per-position SL triggered | 2025-06-26 09:40:00 | 3482.89 | 3477.58 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 3473.80 | 3450.63 | 0.00 | ORB-long ORB[3420.90,3458.00] vol=2.2x ATR=13.43 |
| Stop hit — per-position SL triggered | 2025-07-03 09:40:00 | 3460.37 | 3451.74 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:05:00 | 3495.40 | 3467.86 | 0.00 | ORB-long ORB[3422.40,3464.00] vol=2.0x ATR=13.16 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 3482.24 | 3469.59 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 3406.00 | 3432.63 | 0.00 | ORB-short ORB[3423.00,3470.00] vol=1.5x ATR=8.06 |
| Stop hit — per-position SL triggered | 2025-07-08 11:30:00 | 3414.06 | 3429.85 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:10:00 | 3906.30 | 3872.71 | 0.00 | ORB-long ORB[3860.00,3904.80] vol=2.4x ATR=18.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:15:00 | 3933.96 | 3880.68 | 0.00 | T1 1.5R @ 3933.96 |
| Stop hit — per-position SL triggered | 2025-07-21 10:20:00 | 3906.30 | 3881.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:00:00 | 3907.00 | 3920.88 | 0.00 | ORB-short ORB[3908.00,3957.70] vol=1.6x ATR=17.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 10:35:00 | 3881.18 | 3916.84 | 0.00 | T1 1.5R @ 3881.18 |
| Target hit | 2025-07-23 15:20:00 | 3871.90 | 3882.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:25:00 | 3747.60 | 3725.15 | 0.00 | ORB-long ORB[3691.30,3731.10] vol=1.7x ATR=12.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:30:00 | 3766.74 | 3736.17 | 0.00 | T1 1.5R @ 3766.74 |
| Target hit | 2025-07-29 15:20:00 | 3770.00 | 3766.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-08-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:00:00 | 3279.60 | 3270.04 | 0.00 | ORB-long ORB[3225.70,3268.60] vol=2.0x ATR=9.09 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 3270.51 | 3270.29 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:40:00 | 3284.00 | 3258.39 | 0.00 | ORB-long ORB[3241.00,3270.00] vol=2.0x ATR=12.15 |
| Stop hit — per-position SL triggered | 2025-08-25 14:45:00 | 3271.85 | 3269.65 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:30:00 | 3252.20 | 3229.31 | 0.00 | ORB-long ORB[3212.50,3243.60] vol=2.2x ATR=9.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:15:00 | 3266.81 | 3236.33 | 0.00 | T1 1.5R @ 3266.81 |
| Target hit | 2025-09-02 15:20:00 | 3285.00 | 3257.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:45:00 | 3240.90 | 3271.16 | 0.00 | ORB-short ORB[3253.30,3293.50] vol=1.9x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-09-03 10:50:00 | 3250.09 | 3270.59 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:35:00 | 3301.00 | 3284.45 | 0.00 | ORB-long ORB[3264.00,3299.20] vol=1.6x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 09:40:00 | 3313.12 | 3289.65 | 0.00 | T1 1.5R @ 3313.12 |
| Target hit | 2025-09-04 10:30:00 | 3321.10 | 3325.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-09-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:00:00 | 3348.60 | 3327.96 | 0.00 | ORB-long ORB[3288.40,3328.00] vol=1.6x ATR=13.35 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 3335.25 | 3333.94 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:40:00 | 3376.10 | 3357.19 | 0.00 | ORB-long ORB[3325.00,3358.00] vol=2.8x ATR=11.90 |
| Stop hit — per-position SL triggered | 2025-09-09 10:00:00 | 3364.20 | 3365.73 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:30:00 | 3300.10 | 3322.60 | 0.00 | ORB-short ORB[3316.30,3340.50] vol=1.6x ATR=8.51 |
| Stop hit — per-position SL triggered | 2025-09-15 09:35:00 | 3308.61 | 3320.17 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:10:00 | 3337.90 | 3324.43 | 0.00 | ORB-long ORB[3309.30,3334.90] vol=2.0x ATR=8.43 |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3329.47 | 3324.63 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:15:00 | 3336.30 | 3321.90 | 0.00 | ORB-long ORB[3306.00,3326.80] vol=1.9x ATR=8.08 |
| Stop hit — per-position SL triggered | 2025-09-18 10:55:00 | 3328.22 | 3327.81 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:10:00 | 3359.20 | 3337.92 | 0.00 | ORB-long ORB[3314.60,3349.90] vol=2.4x ATR=10.62 |
| Stop hit — per-position SL triggered | 2025-09-19 10:55:00 | 3348.58 | 3343.01 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:55:00 | 3350.00 | 3328.57 | 0.00 | ORB-long ORB[3305.50,3349.80] vol=2.5x ATR=15.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:05:00 | 3372.66 | 3345.99 | 0.00 | T1 1.5R @ 3372.66 |
| Stop hit — per-position SL triggered | 2025-09-23 10:10:00 | 3350.00 | 3347.25 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:20:00 | 3270.90 | 3289.29 | 0.00 | ORB-short ORB[3286.10,3326.50] vol=1.7x ATR=9.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:40:00 | 3257.02 | 3279.00 | 0.00 | T1 1.5R @ 3257.02 |
| Stop hit — per-position SL triggered | 2025-09-25 12:30:00 | 3270.90 | 3275.93 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:05:00 | 3184.00 | 3170.00 | 0.00 | ORB-long ORB[3139.10,3165.00] vol=2.0x ATR=7.48 |
| Stop hit — per-position SL triggered | 2025-09-30 11:20:00 | 3176.52 | 3170.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:00:00 | 3148.80 | 3167.01 | 0.00 | ORB-short ORB[3161.30,3187.90] vol=2.1x ATR=8.65 |
| Stop hit — per-position SL triggered | 2025-10-06 10:20:00 | 3157.45 | 3163.84 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 3146.00 | 3166.17 | 0.00 | ORB-short ORB[3148.10,3181.10] vol=1.7x ATR=6.45 |
| Stop hit — per-position SL triggered | 2025-10-07 11:00:00 | 3152.45 | 3165.29 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:35:00 | 3156.00 | 3171.31 | 0.00 | ORB-short ORB[3159.20,3191.50] vol=1.9x ATR=10.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:00:00 | 3140.35 | 3161.50 | 0.00 | T1 1.5R @ 3140.35 |
| Stop hit — per-position SL triggered | 2025-10-08 11:55:00 | 3156.00 | 3155.50 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:30:00 | 3142.00 | 3162.87 | 0.00 | ORB-short ORB[3153.10,3191.60] vol=4.1x ATR=5.91 |
| Stop hit — per-position SL triggered | 2025-10-14 10:40:00 | 3147.91 | 3161.70 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:45:00 | 3170.00 | 3151.74 | 0.00 | ORB-long ORB[3113.00,3154.10] vol=5.2x ATR=10.84 |
| Stop hit — per-position SL triggered | 2025-10-16 11:20:00 | 3159.16 | 3154.88 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:45:00 | 3224.00 | 3208.91 | 0.00 | ORB-long ORB[3188.00,3215.00] vol=1.9x ATR=7.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:50:00 | 3235.62 | 3212.68 | 0.00 | T1 1.5R @ 3235.62 |
| Target hit | 2025-10-24 15:20:00 | 3277.10 | 3253.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 3290.70 | 3323.80 | 0.00 | ORB-short ORB[3322.80,3353.00] vol=1.6x ATR=9.61 |
| Stop hit — per-position SL triggered | 2025-10-28 11:10:00 | 3300.31 | 3323.53 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:50:00 | 3256.70 | 3246.35 | 0.00 | ORB-long ORB[3240.00,3254.80] vol=1.6x ATR=7.65 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 3249.05 | 3247.39 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:25:00 | 3273.50 | 3263.48 | 0.00 | ORB-long ORB[3250.00,3269.80] vol=1.9x ATR=7.49 |
| Stop hit — per-position SL triggered | 2025-10-30 11:00:00 | 3266.01 | 3265.10 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:15:00 | 3242.60 | 3268.46 | 0.00 | ORB-short ORB[3248.60,3277.10] vol=3.3x ATR=7.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 12:40:00 | 3231.16 | 3260.11 | 0.00 | T1 1.5R @ 3231.16 |
| Target hit | 2025-10-31 15:20:00 | 3220.00 | 3242.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-11-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 10:00:00 | 3209.60 | 3220.34 | 0.00 | ORB-short ORB[3214.50,3235.40] vol=2.4x ATR=10.39 |
| Stop hit — per-position SL triggered | 2025-11-03 10:30:00 | 3219.99 | 3218.65 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 11:10:00 | 3159.10 | 3167.88 | 0.00 | ORB-short ORB[3160.80,3195.50] vol=1.6x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:45:00 | 3150.01 | 3164.27 | 0.00 | T1 1.5R @ 3150.01 |
| Target hit | 2025-11-10 15:20:00 | 3138.90 | 3147.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-11-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:40:00 | 3168.20 | 3153.23 | 0.00 | ORB-long ORB[3136.00,3158.50] vol=1.6x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:55:00 | 3178.21 | 3157.00 | 0.00 | T1 1.5R @ 3178.21 |
| Stop hit — per-position SL triggered | 2025-11-11 11:15:00 | 3168.20 | 3160.74 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 3009.00 | 3016.03 | 0.00 | ORB-short ORB[3010.00,3023.60] vol=1.6x ATR=7.08 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 3016.08 | 3014.75 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:20:00 | 3042.50 | 3032.74 | 0.00 | ORB-long ORB[3007.50,3038.00] vol=3.8x ATR=7.98 |
| Stop hit — per-position SL triggered | 2025-11-17 10:25:00 | 3034.52 | 3032.78 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 11:10:00 | 2976.40 | 2989.89 | 0.00 | ORB-short ORB[2996.00,3013.70] vol=2.5x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 11:25:00 | 2968.94 | 2987.02 | 0.00 | T1 1.5R @ 2968.94 |
| Target hit | 2025-11-18 15:20:00 | 2947.00 | 2964.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:50:00 | 2907.40 | 2925.33 | 0.00 | ORB-short ORB[2935.00,2967.80] vol=2.9x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:55:00 | 2896.64 | 2911.52 | 0.00 | T1 1.5R @ 2896.64 |
| Target hit | 2025-11-21 15:20:00 | 2883.00 | 2905.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:45:00 | 2888.00 | 2894.59 | 0.00 | ORB-short ORB[2890.50,2929.40] vol=2.1x ATR=6.29 |
| Stop hit — per-position SL triggered | 2025-12-03 09:55:00 | 2894.29 | 2895.09 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:45:00 | 2821.00 | 2837.19 | 0.00 | ORB-short ORB[2836.70,2868.90] vol=1.5x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:55:00 | 2811.11 | 2831.46 | 0.00 | T1 1.5R @ 2811.11 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 2821.00 | 2828.64 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 2817.00 | 2835.41 | 0.00 | ORB-short ORB[2817.10,2849.60] vol=1.9x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 12:00:00 | 2803.43 | 2829.70 | 0.00 | T1 1.5R @ 2803.43 |
| Target hit | 2025-12-10 15:20:00 | 2771.10 | 2792.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-12-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:05:00 | 2872.20 | 2878.83 | 0.00 | ORB-short ORB[2877.70,2889.40] vol=1.6x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:25:00 | 2863.71 | 2877.07 | 0.00 | T1 1.5R @ 2863.71 |
| Stop hit — per-position SL triggered | 2025-12-17 11:05:00 | 2872.20 | 2874.20 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:45:00 | 2835.90 | 2850.03 | 0.00 | ORB-short ORB[2850.00,2874.00] vol=1.6x ATR=7.30 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 2843.20 | 2847.74 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 2869.40 | 2860.48 | 0.00 | ORB-long ORB[2845.80,2868.70] vol=2.2x ATR=7.10 |
| Stop hit — per-position SL triggered | 2025-12-19 12:05:00 | 2862.30 | 2862.04 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:40:00 | 2895.00 | 2887.87 | 0.00 | ORB-long ORB[2866.80,2884.70] vol=1.9x ATR=6.34 |
| Stop hit — per-position SL triggered | 2025-12-23 10:25:00 | 2888.66 | 2889.31 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 3039.30 | 3017.84 | 0.00 | ORB-long ORB[3004.00,3030.20] vol=9.1x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:25:00 | 3051.08 | 3030.01 | 0.00 | T1 1.5R @ 3051.08 |
| Target hit | 2026-01-01 13:20:00 | 3044.70 | 3045.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2026-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:30:00 | 3018.20 | 3036.08 | 0.00 | ORB-short ORB[3028.90,3070.20] vol=2.4x ATR=7.82 |
| Stop hit — per-position SL triggered | 2026-01-02 14:20:00 | 3026.02 | 3024.75 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:10:00 | 2913.00 | 2926.99 | 0.00 | ORB-short ORB[2920.00,2947.50] vol=5.4x ATR=7.36 |
| Stop hit — per-position SL triggered | 2026-01-22 11:20:00 | 2920.36 | 2926.40 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:10:00 | 2804.10 | 2837.11 | 0.00 | ORB-short ORB[2851.10,2880.60] vol=2.0x ATR=9.30 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 2813.40 | 2832.37 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 2929.30 | 2900.58 | 0.00 | ORB-long ORB[2863.30,2879.00] vol=3.1x ATR=8.65 |
| Stop hit — per-position SL triggered | 2026-02-01 11:20:00 | 2920.65 | 2902.96 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 2918.70 | 2896.00 | 0.00 | ORB-long ORB[2862.00,2902.00] vol=3.5x ATR=13.53 |
| Stop hit — per-position SL triggered | 2026-02-02 09:50:00 | 2905.17 | 2900.91 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:15:00 | 2868.00 | 2872.91 | 0.00 | ORB-short ORB[2875.50,2915.00] vol=2.0x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:20:00 | 2854.76 | 2871.67 | 0.00 | T1 1.5R @ 2854.76 |
| Stop hit — per-position SL triggered | 2026-02-06 14:05:00 | 2868.00 | 2868.97 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2949.40 | 2937.07 | 0.00 | ORB-long ORB[2901.30,2943.90] vol=2.0x ATR=6.90 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 2942.50 | 2938.20 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 2956.60 | 2937.70 | 0.00 | ORB-long ORB[2915.60,2948.50] vol=2.7x ATR=12.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 2974.67 | 2948.24 | 0.00 | T1 1.5R @ 2974.67 |
| Target hit | 2026-02-17 15:20:00 | 3056.00 | 3012.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 3042.00 | 3065.37 | 0.00 | ORB-short ORB[3048.00,3073.90] vol=1.7x ATR=11.42 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 3053.42 | 3061.37 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 3066.10 | 3074.97 | 0.00 | ORB-short ORB[3068.00,3100.00] vol=1.6x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 3074.39 | 3074.87 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 3027.80 | 3001.54 | 0.00 | ORB-long ORB[2975.00,3008.70] vol=2.0x ATR=10.80 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 3017.00 | 3003.12 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 3172.40 | 3153.39 | 0.00 | ORB-long ORB[3133.30,3170.00] vol=3.4x ATR=10.53 |
| Stop hit — per-position SL triggered | 2026-02-25 09:35:00 | 3161.87 | 3158.99 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-02-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:25:00 | 3180.30 | 3210.28 | 0.00 | ORB-short ORB[3202.50,3246.60] vol=1.9x ATR=11.13 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 3191.43 | 3206.33 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 3084.50 | 3107.04 | 0.00 | ORB-short ORB[3106.70,3148.00] vol=1.8x ATR=8.97 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 3093.47 | 3104.18 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 3028.90 | 3052.94 | 0.00 | ORB-short ORB[3045.90,3079.30] vol=2.1x ATR=11.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:40:00 | 3011.14 | 3041.71 | 0.00 | T1 1.5R @ 3011.14 |
| Stop hit — per-position SL triggered | 2026-03-04 09:45:00 | 3028.90 | 3040.17 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:00:00 | 3128.00 | 3112.98 | 0.00 | ORB-long ORB[3040.00,3086.60] vol=2.3x ATR=15.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:20:00 | 3151.98 | 3127.42 | 0.00 | T1 1.5R @ 3151.98 |
| Target hit | 2026-03-06 14:45:00 | 3155.60 | 3157.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 3143.20 | 3115.81 | 0.00 | ORB-long ORB[3084.90,3125.70] vol=2.3x ATR=10.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:35:00 | 3159.29 | 3123.04 | 0.00 | T1 1.5R @ 3159.29 |
| Target hit | 2026-03-10 13:40:00 | 3160.00 | 3161.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 3619.90 | 3577.79 | 0.00 | ORB-long ORB[3530.00,3583.90] vol=1.6x ATR=20.56 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 3599.34 | 3588.78 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 4121.60 | 4102.58 | 0.00 | ORB-long ORB[4085.20,4119.70] vol=1.7x ATR=20.48 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 4101.12 | 4107.15 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 4096.90 | 4071.78 | 0.00 | ORB-long ORB[4030.20,4084.90] vol=2.1x ATR=15.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:30:00 | 4119.66 | 4086.30 | 0.00 | T1 1.5R @ 4119.66 |
| Stop hit — per-position SL triggered | 2026-04-23 12:10:00 | 4096.90 | 4098.54 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 3994.80 | 4015.82 | 0.00 | ORB-short ORB[4022.10,4058.40] vol=2.0x ATR=11.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:45:00 | 3977.54 | 4004.74 | 0.00 | T1 1.5R @ 3977.54 |
| Stop hit — per-position SL triggered | 2026-04-29 12:55:00 | 3994.80 | 4001.64 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-04-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:40:00 | 3936.90 | 3964.82 | 0.00 | ORB-short ORB[3959.20,4004.00] vol=2.8x ATR=15.23 |
| Stop hit — per-position SL triggered | 2026-04-30 11:00:00 | 3952.13 | 3960.40 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 4147.10 | 4185.09 | 0.00 | ORB-short ORB[4172.00,4207.60] vol=1.6x ATR=14.89 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 4161.99 | 4181.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:35:00 | 3356.90 | 2025-05-13 09:50:00 | 3375.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-05-13 09:35:00 | 3356.90 | 2025-05-13 10:15:00 | 3356.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:25:00 | 3473.80 | 2025-05-15 11:00:00 | 3462.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-28 11:00:00 | 3488.00 | 2025-05-28 11:20:00 | 3478.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-03 11:15:00 | 3417.50 | 2025-06-03 12:55:00 | 3401.95 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-03 11:15:00 | 3417.50 | 2025-06-03 15:20:00 | 3388.40 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2025-06-04 10:25:00 | 3475.00 | 2025-06-04 11:00:00 | 3498.09 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-06-04 10:25:00 | 3475.00 | 2025-06-04 11:05:00 | 3475.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 11:05:00 | 3554.50 | 2025-06-05 13:20:00 | 3545.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-12 11:15:00 | 3573.90 | 2025-06-12 11:35:00 | 3584.34 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-06-12 11:15:00 | 3573.90 | 2025-06-12 11:45:00 | 3573.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-19 09:30:00 | 3589.00 | 2025-06-19 09:35:00 | 3572.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-06-20 09:50:00 | 3485.00 | 2025-06-20 09:55:00 | 3460.85 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-06-20 09:50:00 | 3485.00 | 2025-06-20 10:00:00 | 3485.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-26 09:30:00 | 3495.60 | 2025-06-26 09:40:00 | 3482.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-03 09:35:00 | 3473.80 | 2025-07-03 09:40:00 | 3460.37 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-07 10:05:00 | 3495.40 | 2025-07-07 10:15:00 | 3482.24 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-08 11:05:00 | 3406.00 | 2025-07-08 11:30:00 | 3414.06 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-21 10:10:00 | 3906.30 | 2025-07-21 10:15:00 | 3933.96 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-21 10:10:00 | 3906.30 | 2025-07-21 10:20:00 | 3906.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 10:00:00 | 3907.00 | 2025-07-23 10:35:00 | 3881.18 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-07-23 10:00:00 | 3907.00 | 2025-07-23 15:20:00 | 3871.90 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2025-07-29 10:25:00 | 3747.60 | 2025-07-29 10:30:00 | 3766.74 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-07-29 10:25:00 | 3747.60 | 2025-07-29 15:20:00 | 3770.00 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-08-11 11:00:00 | 3279.60 | 2025-08-11 11:10:00 | 3270.51 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-25 10:40:00 | 3284.00 | 2025-08-25 14:45:00 | 3271.85 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-02 10:30:00 | 3252.20 | 2025-09-02 11:15:00 | 3266.81 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-02 10:30:00 | 3252.20 | 2025-09-02 15:20:00 | 3285.00 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2025-09-03 10:45:00 | 3240.90 | 2025-09-03 10:50:00 | 3250.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-04 09:35:00 | 3301.00 | 2025-09-04 09:40:00 | 3313.12 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-04 09:35:00 | 3301.00 | 2025-09-04 10:30:00 | 3321.10 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2025-09-08 10:00:00 | 3348.60 | 2025-09-08 10:30:00 | 3335.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-09-09 09:40:00 | 3376.10 | 2025-09-09 10:00:00 | 3364.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-15 09:30:00 | 3300.10 | 2025-09-15 09:35:00 | 3308.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-17 10:10:00 | 3337.90 | 2025-09-17 10:15:00 | 3329.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-18 10:15:00 | 3336.30 | 2025-09-18 10:55:00 | 3328.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-19 10:10:00 | 3359.20 | 2025-09-19 10:55:00 | 3348.58 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-23 09:55:00 | 3350.00 | 2025-09-23 10:05:00 | 3372.66 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-09-23 09:55:00 | 3350.00 | 2025-09-23 10:10:00 | 3350.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 10:20:00 | 3270.90 | 2025-09-25 11:40:00 | 3257.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-09-25 10:20:00 | 3270.90 | 2025-09-25 12:30:00 | 3270.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 11:05:00 | 3184.00 | 2025-09-30 11:20:00 | 3176.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-06 10:00:00 | 3148.80 | 2025-10-06 10:20:00 | 3157.45 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-07 10:55:00 | 3146.00 | 2025-10-07 11:00:00 | 3152.45 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-08 09:35:00 | 3156.00 | 2025-10-08 11:00:00 | 3140.35 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-10-08 09:35:00 | 3156.00 | 2025-10-08 11:55:00 | 3156.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 10:30:00 | 3142.00 | 2025-10-14 10:40:00 | 3147.91 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-16 10:45:00 | 3170.00 | 2025-10-16 11:20:00 | 3159.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-24 10:45:00 | 3224.00 | 2025-10-24 10:50:00 | 3235.62 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-24 10:45:00 | 3224.00 | 2025-10-24 15:20:00 | 3277.10 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2025-10-28 11:00:00 | 3290.70 | 2025-10-28 11:10:00 | 3300.31 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-29 10:50:00 | 3256.70 | 2025-10-29 11:10:00 | 3249.05 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-30 10:25:00 | 3273.50 | 2025-10-30 11:00:00 | 3266.01 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-31 11:15:00 | 3242.60 | 2025-10-31 12:40:00 | 3231.16 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-31 11:15:00 | 3242.60 | 2025-10-31 15:20:00 | 3220.00 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2025-11-03 10:00:00 | 3209.60 | 2025-11-03 10:30:00 | 3219.99 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-10 11:10:00 | 3159.10 | 2025-11-10 11:45:00 | 3150.01 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-10 11:10:00 | 3159.10 | 2025-11-10 15:20:00 | 3138.90 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-11-11 10:40:00 | 3168.20 | 2025-11-11 10:55:00 | 3178.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-11-11 10:40:00 | 3168.20 | 2025-11-11 11:15:00 | 3168.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-14 10:50:00 | 3009.00 | 2025-11-14 11:15:00 | 3016.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-17 10:20:00 | 3042.50 | 2025-11-17 10:25:00 | 3034.52 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-18 11:10:00 | 2976.40 | 2025-11-18 11:25:00 | 2968.94 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-18 11:10:00 | 2976.40 | 2025-11-18 15:20:00 | 2947.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2025-11-21 10:50:00 | 2907.40 | 2025-11-21 14:55:00 | 2896.64 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-21 10:50:00 | 2907.40 | 2025-11-21 15:20:00 | 2883.00 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-12-03 09:45:00 | 2888.00 | 2025-12-03 09:55:00 | 2894.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-05 09:45:00 | 2821.00 | 2025-12-05 09:55:00 | 2811.11 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-05 09:45:00 | 2821.00 | 2025-12-05 10:00:00 | 2821.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-10 10:50:00 | 2817.00 | 2025-12-10 12:00:00 | 2803.43 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-10 10:50:00 | 2817.00 | 2025-12-10 15:20:00 | 2771.10 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2025-12-17 10:05:00 | 2872.20 | 2025-12-17 10:25:00 | 2863.71 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-17 10:05:00 | 2872.20 | 2025-12-17 11:05:00 | 2872.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:45:00 | 2835.90 | 2025-12-18 10:05:00 | 2843.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-19 10:45:00 | 2869.40 | 2025-12-19 12:05:00 | 2862.30 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-23 09:40:00 | 2895.00 | 2025-12-23 10:25:00 | 2888.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-01 11:10:00 | 3039.30 | 2026-01-01 11:25:00 | 3051.08 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-01 11:10:00 | 3039.30 | 2026-01-01 13:20:00 | 3044.70 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-01-02 10:30:00 | 3018.20 | 2026-01-02 14:20:00 | 3026.02 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-22 11:10:00 | 2913.00 | 2026-01-22 11:20:00 | 2920.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-29 10:10:00 | 2804.10 | 2026-01-29 10:15:00 | 2813.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-01 11:10:00 | 2929.30 | 2026-02-01 11:20:00 | 2920.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-02 09:30:00 | 2918.70 | 2026-02-02 09:50:00 | 2905.17 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-06 11:15:00 | 2868.00 | 2026-02-06 12:20:00 | 2854.76 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-06 11:15:00 | 2868.00 | 2026-02-06 14:05:00 | 2868.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 11:00:00 | 2949.40 | 2026-02-10 11:25:00 | 2942.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-17 10:05:00 | 2956.60 | 2026-02-17 10:10:00 | 2974.67 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-17 10:05:00 | 2956.60 | 2026-02-17 15:20:00 | 3056.00 | TARGET_HIT | 0.50 | 3.36% |
| SELL | retest1 | 2026-02-18 10:00:00 | 3042.00 | 2026-02-18 10:15:00 | 3053.42 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-19 09:30:00 | 3066.10 | 2026-02-19 09:35:00 | 3074.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:40:00 | 3027.80 | 2026-02-20 10:55:00 | 3017.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-25 09:30:00 | 3172.40 | 2026-02-25 09:35:00 | 3161.87 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-26 10:25:00 | 3180.30 | 2026-02-26 10:50:00 | 3191.43 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 10:50:00 | 3084.50 | 2026-02-27 11:05:00 | 3093.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-04 09:30:00 | 3028.90 | 2026-03-04 09:40:00 | 3011.14 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-04 09:30:00 | 3028.90 | 2026-03-04 09:45:00 | 3028.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:00:00 | 3128.00 | 2026-03-06 10:20:00 | 3151.98 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-06 10:00:00 | 3128.00 | 2026-03-06 14:45:00 | 3155.60 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-03-10 10:30:00 | 3143.20 | 2026-03-10 10:35:00 | 3159.29 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-10 10:30:00 | 3143.20 | 2026-03-10 13:40:00 | 3160.00 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 09:45:00 | 3619.90 | 2026-04-10 10:50:00 | 3599.34 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-17 09:40:00 | 4121.60 | 2026-04-17 09:45:00 | 4101.12 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-23 10:15:00 | 4096.90 | 2026-04-23 10:30:00 | 4119.66 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-23 10:15:00 | 4096.90 | 2026-04-23 12:10:00 | 4096.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:45:00 | 3994.80 | 2026-04-29 11:45:00 | 3977.54 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-29 10:45:00 | 3994.80 | 2026-04-29 12:55:00 | 3994.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:40:00 | 3936.90 | 2026-04-30 11:00:00 | 3952.13 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-05 11:05:00 | 4147.10 | 2026-05-05 11:30:00 | 4161.99 | STOP_HIT | 1.00 | -0.36% |
