# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 5560.00
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
| ENTRY1 | 100 |
| ENTRY2 | 0 |
| PARTIAL | 45 |
| TARGET_HIT | 18 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 82
- **Target hits / Stop hits / Partials:** 18 / 82 / 45
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 15.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 34 | 44.7% | 10 | 42 | 24 | 0.14% | 10.5% |
| BUY @ 2nd Alert (retest1) | 76 | 34 | 44.7% | 10 | 42 | 24 | 0.14% | 10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 29 | 42.0% | 8 | 40 | 21 | 0.08% | 5.3% |
| SELL @ 2nd Alert (retest1) | 69 | 29 | 42.0% | 8 | 40 | 21 | 0.08% | 5.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 145 | 63 | 43.4% | 18 | 82 | 45 | 0.11% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:40:00 | 3453.20 | 3445.34 | 0.00 | ORB-long ORB[3421.00,3449.95] vol=2.8x ATR=9.64 |
| Stop hit — per-position SL triggered | 2023-05-16 09:45:00 | 3443.56 | 3445.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 09:30:00 | 3395.00 | 3410.41 | 0.00 | ORB-short ORB[3413.65,3429.80] vol=1.9x ATR=9.26 |
| Stop hit — per-position SL triggered | 2023-05-17 09:35:00 | 3404.26 | 3407.62 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 11:05:00 | 3342.55 | 3366.18 | 0.00 | ORB-short ORB[3350.00,3392.05] vol=3.6x ATR=10.03 |
| Stop hit — per-position SL triggered | 2023-05-18 11:10:00 | 3352.58 | 3359.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:45:00 | 3365.40 | 3349.15 | 0.00 | ORB-long ORB[3327.65,3349.90] vol=3.1x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 10:55:00 | 3374.83 | 3352.73 | 0.00 | T1 1.5R @ 3374.83 |
| Stop hit — per-position SL triggered | 2023-05-29 12:30:00 | 3365.40 | 3360.91 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 11:10:00 | 3384.50 | 3382.54 | 0.00 | ORB-long ORB[3354.85,3368.00] vol=11.5x ATR=6.31 |
| Stop hit — per-position SL triggered | 2023-05-31 11:20:00 | 3378.19 | 3380.63 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 3371.05 | 3379.65 | 0.00 | ORB-short ORB[3373.05,3397.10] vol=1.7x ATR=6.11 |
| Stop hit — per-position SL triggered | 2023-06-02 11:10:00 | 3377.16 | 3379.31 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:55:00 | 3381.40 | 3384.81 | 0.00 | ORB-short ORB[3382.05,3409.90] vol=3.8x ATR=11.01 |
| Stop hit — per-position SL triggered | 2023-06-06 10:00:00 | 3392.41 | 3386.75 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:00:00 | 3405.65 | 3400.74 | 0.00 | ORB-long ORB[3385.35,3401.85] vol=2.2x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:10:00 | 3415.53 | 3404.17 | 0.00 | T1 1.5R @ 3415.53 |
| Stop hit — per-position SL triggered | 2023-06-07 10:40:00 | 3405.65 | 3404.42 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:00:00 | 3422.75 | 3434.33 | 0.00 | ORB-short ORB[3434.05,3460.00] vol=1.8x ATR=8.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 10:10:00 | 3409.92 | 3430.19 | 0.00 | T1 1.5R @ 3409.92 |
| Target hit | 2023-06-09 12:05:00 | 3413.00 | 3409.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2023-06-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:00:00 | 3419.95 | 3392.75 | 0.00 | ORB-long ORB[3350.00,3377.60] vol=2.1x ATR=9.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:50:00 | 3434.43 | 3409.01 | 0.00 | T1 1.5R @ 3434.43 |
| Target hit | 2023-06-13 12:15:00 | 3423.35 | 3423.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2023-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:35:00 | 3420.45 | 3416.59 | 0.00 | ORB-long ORB[3395.60,3414.35] vol=13.3x ATR=6.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 09:40:00 | 3430.72 | 3417.61 | 0.00 | T1 1.5R @ 3430.72 |
| Stop hit — per-position SL triggered | 2023-06-19 11:50:00 | 3420.45 | 3425.54 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 3415.00 | 3440.33 | 0.00 | ORB-short ORB[3425.00,3455.90] vol=2.3x ATR=13.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:25:00 | 3395.27 | 3428.87 | 0.00 | T1 1.5R @ 3395.27 |
| Target hit | 2023-06-20 15:20:00 | 3394.50 | 3393.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-06-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 10:35:00 | 3401.95 | 3368.90 | 0.00 | ORB-long ORB[3335.80,3359.15] vol=2.0x ATR=9.70 |
| Stop hit — per-position SL triggered | 2023-06-26 10:55:00 | 3392.25 | 3377.57 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:40:00 | 3417.95 | 3405.91 | 0.00 | ORB-long ORB[3398.15,3413.30] vol=2.9x ATR=8.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 09:45:00 | 3431.40 | 3406.30 | 0.00 | T1 1.5R @ 3431.40 |
| Stop hit — per-position SL triggered | 2023-06-28 09:50:00 | 3417.95 | 3406.41 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:45:00 | 3487.40 | 3471.80 | 0.00 | ORB-long ORB[3449.05,3476.80] vol=1.9x ATR=10.20 |
| Stop hit — per-position SL triggered | 2023-06-30 09:55:00 | 3477.20 | 3472.39 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:35:00 | 3519.90 | 3510.29 | 0.00 | ORB-long ORB[3482.55,3512.30] vol=2.4x ATR=7.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 09:55:00 | 3531.21 | 3517.35 | 0.00 | T1 1.5R @ 3531.21 |
| Stop hit — per-position SL triggered | 2023-07-04 10:10:00 | 3519.90 | 3518.09 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:45:00 | 3498.05 | 3512.56 | 0.00 | ORB-short ORB[3510.55,3536.00] vol=2.1x ATR=6.89 |
| Stop hit — per-position SL triggered | 2023-07-12 09:50:00 | 3504.94 | 3510.78 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:15:00 | 3548.75 | 3534.16 | 0.00 | ORB-long ORB[3517.15,3535.00] vol=4.5x ATR=7.44 |
| Stop hit — per-position SL triggered | 2023-07-13 10:25:00 | 3541.31 | 3536.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:30:00 | 3549.00 | 3535.77 | 0.00 | ORB-long ORB[3508.30,3538.40] vol=4.8x ATR=8.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 09:35:00 | 3561.33 | 3563.04 | 0.00 | T1 1.5R @ 3561.33 |
| Target hit | 2023-07-17 09:45:00 | 3561.40 | 3564.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2023-07-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:35:00 | 3691.55 | 3678.58 | 0.00 | ORB-long ORB[3635.00,3672.00] vol=4.8x ATR=9.98 |
| Stop hit — per-position SL triggered | 2023-07-20 09:40:00 | 3681.57 | 3680.28 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 10:50:00 | 3690.05 | 3685.73 | 0.00 | ORB-long ORB[3651.10,3686.35] vol=4.6x ATR=8.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 11:25:00 | 3702.94 | 3687.41 | 0.00 | T1 1.5R @ 3702.94 |
| Stop hit — per-position SL triggered | 2023-07-21 11:45:00 | 3690.05 | 3688.51 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 11:10:00 | 3737.00 | 3717.04 | 0.00 | ORB-long ORB[3693.65,3721.10] vol=1.8x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:45:00 | 3746.54 | 3730.89 | 0.00 | T1 1.5R @ 3746.54 |
| Target hit | 2023-07-24 15:20:00 | 3780.00 | 3759.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2023-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:05:00 | 3987.50 | 3977.94 | 0.00 | ORB-long ORB[3914.70,3966.00] vol=2.4x ATR=13.28 |
| Stop hit — per-position SL triggered | 2023-07-26 10:40:00 | 3974.22 | 3982.05 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 11:00:00 | 3974.05 | 3990.19 | 0.00 | ORB-short ORB[3978.80,4017.60] vol=1.6x ATR=8.47 |
| Stop hit — per-position SL triggered | 2023-07-31 11:40:00 | 3982.52 | 3987.77 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 09:30:00 | 4191.05 | 4180.03 | 0.00 | ORB-long ORB[4130.60,4183.00] vol=4.0x ATR=13.18 |
| Stop hit — per-position SL triggered | 2023-08-09 09:35:00 | 4177.87 | 4179.05 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 10:50:00 | 3799.40 | 3779.38 | 0.00 | ORB-long ORB[3746.20,3785.05] vol=1.7x ATR=9.60 |
| Stop hit — per-position SL triggered | 2023-08-16 11:00:00 | 3789.80 | 3779.87 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:05:00 | 3771.00 | 3795.01 | 0.00 | ORB-short ORB[3783.00,3809.95] vol=1.5x ATR=8.75 |
| Stop hit — per-position SL triggered | 2023-08-17 11:10:00 | 3779.75 | 3793.21 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:30:00 | 3731.55 | 3758.08 | 0.00 | ORB-short ORB[3757.40,3796.95] vol=6.5x ATR=11.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 09:40:00 | 3714.38 | 3749.13 | 0.00 | T1 1.5R @ 3714.38 |
| Stop hit — per-position SL triggered | 2023-08-18 10:50:00 | 3731.55 | 3735.32 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:15:00 | 3777.35 | 3803.50 | 0.00 | ORB-short ORB[3820.00,3844.65] vol=1.5x ATR=5.23 |
| Stop hit — per-position SL triggered | 2023-08-22 11:25:00 | 3782.58 | 3802.32 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 10:35:00 | 3758.30 | 3774.74 | 0.00 | ORB-short ORB[3760.20,3791.95] vol=2.2x ATR=6.63 |
| Stop hit — per-position SL triggered | 2023-08-24 10:40:00 | 3764.93 | 3773.23 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 11:15:00 | 3702.05 | 3715.18 | 0.00 | ORB-short ORB[3707.00,3729.00] vol=1.5x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 12:05:00 | 3692.51 | 3711.29 | 0.00 | T1 1.5R @ 3692.51 |
| Stop hit — per-position SL triggered | 2023-08-25 12:10:00 | 3702.05 | 3710.67 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 10:45:00 | 3719.35 | 3697.11 | 0.00 | ORB-long ORB[3666.00,3699.30] vol=1.8x ATR=8.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 11:20:00 | 3731.80 | 3717.65 | 0.00 | T1 1.5R @ 3731.80 |
| Target hit | 2023-08-28 12:35:00 | 3730.30 | 3730.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2023-08-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:05:00 | 3668.80 | 3680.72 | 0.00 | ORB-short ORB[3680.00,3699.90] vol=2.0x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:30:00 | 3661.76 | 3676.50 | 0.00 | T1 1.5R @ 3661.76 |
| Stop hit — per-position SL triggered | 2023-08-31 11:50:00 | 3668.80 | 3675.89 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 3642.80 | 3633.00 | 0.00 | ORB-long ORB[3614.35,3639.70] vol=2.9x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 11:45:00 | 3650.85 | 3636.83 | 0.00 | T1 1.5R @ 3650.85 |
| Target hit | 2023-09-04 13:40:00 | 3650.30 | 3650.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — SELL (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 3640.60 | 3645.89 | 0.00 | ORB-short ORB[3642.20,3669.90] vol=2.4x ATR=7.64 |
| Stop hit — per-position SL triggered | 2023-09-05 09:50:00 | 3648.24 | 3646.78 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:45:00 | 3628.00 | 3639.03 | 0.00 | ORB-short ORB[3637.10,3656.95] vol=2.3x ATR=7.48 |
| Stop hit — per-position SL triggered | 2023-09-08 13:25:00 | 3635.48 | 3633.98 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 10:00:00 | 3686.60 | 3663.94 | 0.00 | ORB-long ORB[3635.90,3664.80] vol=1.7x ATR=9.73 |
| Stop hit — per-position SL triggered | 2023-09-13 10:10:00 | 3676.87 | 3668.05 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 10:40:00 | 3648.90 | 3677.29 | 0.00 | ORB-short ORB[3681.85,3706.00] vol=2.2x ATR=9.80 |
| Stop hit — per-position SL triggered | 2023-09-18 11:40:00 | 3658.70 | 3666.48 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 3670.75 | 3678.38 | 0.00 | ORB-short ORB[3674.90,3690.00] vol=1.9x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 09:35:00 | 3661.38 | 3675.65 | 0.00 | T1 1.5R @ 3661.38 |
| Target hit | 2023-09-21 10:35:00 | 3670.00 | 3662.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — SELL (started 2023-09-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:50:00 | 3591.15 | 3618.37 | 0.00 | ORB-short ORB[3635.00,3662.15] vol=3.2x ATR=10.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 10:00:00 | 3574.79 | 3602.18 | 0.00 | T1 1.5R @ 3574.79 |
| Target hit | 2023-09-22 12:30:00 | 3565.85 | 3562.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2023-09-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:25:00 | 3545.65 | 3541.50 | 0.00 | ORB-long ORB[3521.20,3544.50] vol=2.3x ATR=6.48 |
| Stop hit — per-position SL triggered | 2023-09-27 10:35:00 | 3539.17 | 3541.79 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:35:00 | 3575.10 | 3557.84 | 0.00 | ORB-long ORB[3522.00,3544.20] vol=1.7x ATR=8.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:40:00 | 3588.14 | 3575.24 | 0.00 | T1 1.5R @ 3588.14 |
| Target hit | 2023-09-29 15:20:00 | 3615.85 | 3601.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-10-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:50:00 | 3512.05 | 3497.78 | 0.00 | ORB-long ORB[3479.05,3510.00] vol=2.1x ATR=7.85 |
| Stop hit — per-position SL triggered | 2023-10-05 11:35:00 | 3504.20 | 3500.80 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 10:55:00 | 3518.85 | 3510.85 | 0.00 | ORB-long ORB[3490.05,3509.55] vol=1.8x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 12:20:00 | 3527.14 | 3515.79 | 0.00 | T1 1.5R @ 3527.14 |
| Target hit | 2023-10-09 15:20:00 | 3547.95 | 3536.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2023-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 09:45:00 | 3618.00 | 3600.67 | 0.00 | ORB-long ORB[3570.00,3600.85] vol=2.4x ATR=8.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 09:55:00 | 3630.92 | 3617.25 | 0.00 | T1 1.5R @ 3630.92 |
| Stop hit — per-position SL triggered | 2023-10-13 10:15:00 | 3618.00 | 3624.24 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 09:35:00 | 3594.10 | 3601.24 | 0.00 | ORB-short ORB[3595.30,3619.90] vol=3.5x ATR=7.61 |
| Stop hit — per-position SL triggered | 2023-10-16 09:40:00 | 3601.71 | 3600.23 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 10:20:00 | 3565.50 | 3550.85 | 0.00 | ORB-long ORB[3531.80,3552.10] vol=1.9x ATR=8.42 |
| Stop hit — per-position SL triggered | 2023-10-25 11:35:00 | 3557.08 | 3557.10 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 11:10:00 | 3533.35 | 3536.54 | 0.00 | ORB-short ORB[3552.05,3586.00] vol=1.9x ATR=8.85 |
| Stop hit — per-position SL triggered | 2023-10-26 11:40:00 | 3542.20 | 3535.87 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 09:30:00 | 3611.05 | 3598.31 | 0.00 | ORB-long ORB[3558.00,3605.95] vol=3.0x ATR=11.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 09:50:00 | 3628.03 | 3605.34 | 0.00 | T1 1.5R @ 3628.03 |
| Target hit | 2023-10-27 15:20:00 | 3679.00 | 3649.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2023-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:40:00 | 3712.10 | 3701.13 | 0.00 | ORB-long ORB[3684.15,3700.00] vol=4.3x ATR=9.49 |
| Stop hit — per-position SL triggered | 2023-10-31 09:55:00 | 3702.61 | 3702.80 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:45:00 | 3700.95 | 3702.43 | 0.00 | ORB-short ORB[3701.00,3739.95] vol=9.7x ATR=10.74 |
| Stop hit — per-position SL triggered | 2023-11-01 10:55:00 | 3711.69 | 3702.92 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:55:00 | 3761.80 | 3745.92 | 0.00 | ORB-long ORB[3723.00,3750.00] vol=3.3x ATR=10.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 10:05:00 | 3776.93 | 3751.48 | 0.00 | T1 1.5R @ 3776.93 |
| Stop hit — per-position SL triggered | 2023-11-02 10:30:00 | 3761.80 | 3765.12 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:35:00 | 3823.95 | 3823.45 | 0.00 | ORB-long ORB[3785.15,3813.95] vol=4.6x ATR=10.94 |
| Stop hit — per-position SL triggered | 2023-11-06 13:40:00 | 3813.01 | 3825.04 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:35:00 | 3869.85 | 3847.93 | 0.00 | ORB-long ORB[3814.80,3854.85] vol=2.1x ATR=11.28 |
| Stop hit — per-position SL triggered | 2023-11-07 09:40:00 | 3858.57 | 3850.04 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 09:35:00 | 4217.00 | 4250.99 | 0.00 | ORB-short ORB[4260.45,4295.15] vol=5.2x ATR=15.41 |
| Stop hit — per-position SL triggered | 2023-11-10 09:40:00 | 4232.41 | 4246.95 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:35:00 | 4375.00 | 4357.87 | 0.00 | ORB-long ORB[4322.25,4366.95] vol=2.3x ATR=12.09 |
| Stop hit — per-position SL triggered | 2023-11-16 10:00:00 | 4362.91 | 4360.98 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 10:05:00 | 4394.15 | 4414.03 | 0.00 | ORB-short ORB[4425.55,4450.00] vol=2.8x ATR=10.02 |
| Stop hit — per-position SL triggered | 2023-11-20 10:10:00 | 4404.17 | 4413.45 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:45:00 | 4446.95 | 4435.17 | 0.00 | ORB-long ORB[4401.05,4442.20] vol=2.1x ATR=7.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 09:55:00 | 4457.64 | 4446.57 | 0.00 | T1 1.5R @ 4457.64 |
| Stop hit — per-position SL triggered | 2023-11-21 10:05:00 | 4446.95 | 4447.71 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:30:00 | 4538.70 | 4513.40 | 0.00 | ORB-long ORB[4476.05,4500.00] vol=5.2x ATR=10.45 |
| Stop hit — per-position SL triggered | 2023-11-22 09:35:00 | 4528.25 | 4520.46 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:45:00 | 4585.15 | 4566.29 | 0.00 | ORB-long ORB[4526.30,4579.00] vol=1.7x ATR=10.87 |
| Stop hit — per-position SL triggered | 2023-11-24 09:50:00 | 4574.28 | 4568.28 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:45:00 | 4603.00 | 4593.34 | 0.00 | ORB-long ORB[4565.15,4600.00] vol=1.5x ATR=10.10 |
| Stop hit — per-position SL triggered | 2023-11-28 10:00:00 | 4592.90 | 4594.69 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 09:50:00 | 4572.55 | 4582.78 | 0.00 | ORB-short ORB[4575.90,4606.70] vol=2.1x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 09:55:00 | 4560.36 | 4581.26 | 0.00 | T1 1.5R @ 4560.36 |
| Stop hit — per-position SL triggered | 2023-11-29 15:20:00 | 4596.75 | 4563.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:15:00 | 4597.55 | 4581.28 | 0.00 | ORB-long ORB[4565.35,4597.00] vol=2.1x ATR=10.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 10:40:00 | 4613.60 | 4592.21 | 0.00 | T1 1.5R @ 4613.60 |
| Target hit | 2023-11-30 15:20:00 | 4743.90 | 4672.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2023-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 11:05:00 | 4620.15 | 4634.15 | 0.00 | ORB-short ORB[4622.80,4667.15] vol=1.6x ATR=11.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 11:15:00 | 4602.35 | 4631.62 | 0.00 | T1 1.5R @ 4602.35 |
| Stop hit — per-position SL triggered | 2023-12-04 11:25:00 | 4620.15 | 4630.42 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:20:00 | 4776.50 | 4729.36 | 0.00 | ORB-long ORB[4660.00,4723.00] vol=2.1x ATR=15.96 |
| Stop hit — per-position SL triggered | 2023-12-06 11:10:00 | 4760.54 | 4753.52 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:35:00 | 4769.40 | 4757.07 | 0.00 | ORB-long ORB[4701.90,4765.90] vol=2.0x ATR=14.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 09:45:00 | 4790.52 | 4763.53 | 0.00 | T1 1.5R @ 4790.52 |
| Target hit | 2023-12-07 10:25:00 | 4784.95 | 4785.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 67 — BUY (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 10:15:00 | 4848.00 | 4832.41 | 0.00 | ORB-long ORB[4781.35,4834.90] vol=1.5x ATR=15.33 |
| Stop hit — per-position SL triggered | 2023-12-13 10:20:00 | 4832.67 | 4832.95 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 11:15:00 | 4941.55 | 4901.48 | 0.00 | ORB-long ORB[4855.55,4917.80] vol=3.2x ATR=11.81 |
| Stop hit — per-position SL triggered | 2023-12-15 11:50:00 | 4929.74 | 4911.53 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 4983.55 | 4966.12 | 0.00 | ORB-long ORB[4921.10,4960.00] vol=5.4x ATR=11.23 |
| Stop hit — per-position SL triggered | 2023-12-20 09:35:00 | 4972.32 | 4974.17 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2023-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:50:00 | 4890.25 | 4915.68 | 0.00 | ORB-short ORB[4915.15,4955.00] vol=6.4x ATR=15.64 |
| Stop hit — per-position SL triggered | 2023-12-22 11:00:00 | 4905.89 | 4909.96 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 11:05:00 | 5151.00 | 5188.07 | 0.00 | ORB-short ORB[5178.05,5225.00] vol=1.7x ATR=11.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 11:20:00 | 5133.22 | 5183.97 | 0.00 | T1 1.5R @ 5133.22 |
| Stop hit — per-position SL triggered | 2024-01-01 11:40:00 | 5151.00 | 5177.09 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:35:00 | 5300.00 | 5273.70 | 0.00 | ORB-long ORB[5228.60,5252.00] vol=2.7x ATR=15.61 |
| Stop hit — per-position SL triggered | 2024-01-04 10:45:00 | 5284.39 | 5274.72 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 5176.80 | 5196.16 | 0.00 | ORB-short ORB[5187.00,5235.35] vol=2.2x ATR=13.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 11:10:00 | 5157.26 | 5193.89 | 0.00 | T1 1.5R @ 5157.26 |
| Stop hit — per-position SL triggered | 2024-01-08 11:25:00 | 5176.80 | 5192.91 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 5260.95 | 5245.71 | 0.00 | ORB-long ORB[5225.00,5257.45] vol=1.6x ATR=13.96 |
| Stop hit — per-position SL triggered | 2024-01-09 09:35:00 | 5246.99 | 5247.18 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:50:00 | 5116.45 | 5165.06 | 0.00 | ORB-short ORB[5170.05,5225.00] vol=1.8x ATR=15.62 |
| Stop hit — per-position SL triggered | 2024-01-10 11:10:00 | 5132.07 | 5157.27 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 09:30:00 | 5081.60 | 5047.63 | 0.00 | ORB-long ORB[5014.00,5060.60] vol=1.5x ATR=13.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:45:00 | 5101.99 | 5061.64 | 0.00 | T1 1.5R @ 5101.99 |
| Stop hit — per-position SL triggered | 2024-01-17 09:50:00 | 5081.60 | 5062.88 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 4961.10 | 5011.80 | 0.00 | ORB-short ORB[5020.70,5072.60] vol=1.8x ATR=18.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 4933.87 | 4996.04 | 0.00 | T1 1.5R @ 4933.87 |
| Stop hit — per-position SL triggered | 2024-01-18 10:30:00 | 4961.10 | 4960.25 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:55:00 | 4920.00 | 4938.25 | 0.00 | ORB-short ORB[4934.00,4979.00] vol=2.7x ATR=8.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 11:05:00 | 4907.52 | 4933.88 | 0.00 | T1 1.5R @ 4907.52 |
| Target hit | 2024-01-20 15:20:00 | 4879.65 | 4902.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2024-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:30:00 | 4921.00 | 4903.20 | 0.00 | ORB-long ORB[4872.00,4916.95] vol=2.3x ATR=14.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:40:00 | 4942.41 | 4916.11 | 0.00 | T1 1.5R @ 4942.41 |
| Stop hit — per-position SL triggered | 2024-01-23 09:55:00 | 4921.00 | 4921.64 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-01-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 11:05:00 | 4973.35 | 4922.66 | 0.00 | ORB-long ORB[4816.50,4887.50] vol=1.6x ATR=19.41 |
| Stop hit — per-position SL triggered | 2024-01-24 11:15:00 | 4953.94 | 4927.24 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-01-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 11:10:00 | 4923.00 | 4952.80 | 0.00 | ORB-short ORB[4944.15,4989.55] vol=5.8x ATR=13.65 |
| Stop hit — per-position SL triggered | 2024-01-30 11:15:00 | 4936.65 | 4949.23 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 11:10:00 | 5075.90 | 5029.55 | 0.00 | ORB-long ORB[4925.35,4996.90] vol=3.5x ATR=14.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:20:00 | 5098.34 | 5042.46 | 0.00 | T1 1.5R @ 5098.34 |
| Stop hit — per-position SL triggered | 2024-02-05 11:55:00 | 5075.90 | 5072.26 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:45:00 | 5276.95 | 5219.89 | 0.00 | ORB-long ORB[5160.00,5233.25] vol=1.6x ATR=20.72 |
| Stop hit — per-position SL triggered | 2024-02-13 10:50:00 | 5256.23 | 5222.69 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:50:00 | 5483.00 | 5450.89 | 0.00 | ORB-long ORB[5390.10,5470.00] vol=1.9x ATR=20.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 10:20:00 | 5513.62 | 5477.53 | 0.00 | T1 1.5R @ 5513.62 |
| Stop hit — per-position SL triggered | 2024-02-21 10:50:00 | 5483.00 | 5484.73 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 10:40:00 | 5372.75 | 5414.21 | 0.00 | ORB-short ORB[5403.85,5447.15] vol=2.5x ATR=14.84 |
| Stop hit — per-position SL triggered | 2024-02-23 10:50:00 | 5387.59 | 5410.95 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 11:00:00 | 5032.60 | 5055.97 | 0.00 | ORB-short ORB[5046.05,5079.80] vol=2.0x ATR=12.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:35:00 | 5013.14 | 5052.24 | 0.00 | T1 1.5R @ 5013.14 |
| Target hit | 2024-02-28 15:20:00 | 5001.40 | 5011.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2024-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:35:00 | 5118.45 | 5127.69 | 0.00 | ORB-short ORB[5120.00,5156.55] vol=6.7x ATR=13.26 |
| Stop hit — per-position SL triggered | 2024-03-04 09:40:00 | 5131.71 | 5127.30 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 11:00:00 | 5070.45 | 5091.87 | 0.00 | ORB-short ORB[5080.20,5139.65] vol=1.5x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 12:15:00 | 5055.04 | 5085.23 | 0.00 | T1 1.5R @ 5055.04 |
| Stop hit — per-position SL triggered | 2024-03-05 14:55:00 | 5070.45 | 5076.57 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 11:05:00 | 5016.90 | 5028.27 | 0.00 | ORB-short ORB[5017.85,5080.90] vol=1.6x ATR=17.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:55:00 | 4990.68 | 5019.65 | 0.00 | T1 1.5R @ 4990.68 |
| Stop hit — per-position SL triggered | 2024-03-13 12:55:00 | 5016.90 | 5012.45 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:30:00 | 4901.15 | 4927.93 | 0.00 | ORB-short ORB[4939.90,5000.30] vol=4.0x ATR=12.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:45:00 | 4881.80 | 4915.26 | 0.00 | T1 1.5R @ 4881.80 |
| Stop hit — per-position SL triggered | 2024-03-19 12:40:00 | 4901.15 | 4910.00 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:00:00 | 4833.85 | 4863.34 | 0.00 | ORB-short ORB[4863.30,4903.10] vol=1.9x ATR=14.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:20:00 | 4812.46 | 4853.96 | 0.00 | T1 1.5R @ 4812.46 |
| Target hit | 2024-03-20 15:20:00 | 4781.20 | 4805.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — SELL (started 2024-04-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 09:50:00 | 4941.10 | 4980.08 | 0.00 | ORB-short ORB[4972.30,5009.00] vol=1.7x ATR=18.89 |
| Stop hit — per-position SL triggered | 2024-04-02 09:55:00 | 4959.99 | 4964.28 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:55:00 | 4897.50 | 4916.08 | 0.00 | ORB-short ORB[4919.05,4939.95] vol=1.9x ATR=8.84 |
| Stop hit — per-position SL triggered | 2024-04-03 11:25:00 | 4906.34 | 4913.32 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 4855.95 | 4862.37 | 0.00 | ORB-short ORB[4875.15,4915.00] vol=5.5x ATR=9.51 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 4865.46 | 4862.49 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:30:00 | 4876.60 | 4890.25 | 0.00 | ORB-short ORB[4881.95,4950.00] vol=1.7x ATR=17.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 13:30:00 | 4850.74 | 4869.57 | 0.00 | T1 1.5R @ 4850.74 |
| Stop hit — per-position SL triggered | 2024-04-08 14:55:00 | 4876.60 | 4868.72 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 10:15:00 | 4677.00 | 4683.11 | 0.00 | ORB-short ORB[4710.40,4749.15] vol=14.4x ATR=14.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 10:20:00 | 4655.80 | 4682.75 | 0.00 | T1 1.5R @ 4655.80 |
| Stop hit — per-position SL triggered | 2024-04-18 10:25:00 | 4677.00 | 4682.62 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-04-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-19 10:45:00 | 4581.45 | 4606.06 | 0.00 | ORB-short ORB[4592.15,4644.95] vol=1.8x ATR=14.37 |
| Stop hit — per-position SL triggered | 2024-04-19 11:20:00 | 4595.82 | 4601.48 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-04-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 10:50:00 | 4914.05 | 4898.60 | 0.00 | ORB-long ORB[4866.30,4897.10] vol=1.6x ATR=13.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 11:00:00 | 4934.19 | 4901.10 | 0.00 | T1 1.5R @ 4934.19 |
| Stop hit — per-position SL triggered | 2024-04-26 12:00:00 | 4914.05 | 4909.12 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-05-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 09:35:00 | 4803.30 | 4823.00 | 0.00 | ORB-short ORB[4817.80,4845.70] vol=1.7x ATR=13.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 09:40:00 | 4783.17 | 4799.25 | 0.00 | T1 1.5R @ 4783.17 |
| Target hit | 2024-05-02 11:25:00 | 4783.45 | 4780.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 100 — SELL (started 2024-05-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:55:00 | 5124.25 | 5138.87 | 0.00 | ORB-short ORB[5136.15,5188.40] vol=6.1x ATR=18.01 |
| Stop hit — per-position SL triggered | 2024-05-09 11:40:00 | 5142.26 | 5130.26 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 09:40:00 | 3453.20 | 2023-05-16 09:45:00 | 3443.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-17 09:30:00 | 3395.00 | 2023-05-17 09:35:00 | 3404.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-18 11:05:00 | 3342.55 | 2023-05-18 11:10:00 | 3352.58 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-29 10:45:00 | 3365.40 | 2023-05-29 10:55:00 | 3374.83 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-05-29 10:45:00 | 3365.40 | 2023-05-29 12:30:00 | 3365.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-31 11:10:00 | 3384.50 | 2023-05-31 11:20:00 | 3378.19 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-02 10:55:00 | 3371.05 | 2023-06-02 11:10:00 | 3377.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-06 09:55:00 | 3381.40 | 2023-06-06 10:00:00 | 3392.41 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-07 10:00:00 | 3405.65 | 2023-06-07 10:10:00 | 3415.53 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-06-07 10:00:00 | 3405.65 | 2023-06-07 10:40:00 | 3405.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 10:00:00 | 3422.75 | 2023-06-09 10:10:00 | 3409.92 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-06-09 10:00:00 | 3422.75 | 2023-06-09 12:05:00 | 3413.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2023-06-13 10:00:00 | 3419.95 | 2023-06-13 10:50:00 | 3434.43 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-06-13 10:00:00 | 3419.95 | 2023-06-13 12:15:00 | 3423.35 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-06-19 09:35:00 | 3420.45 | 2023-06-19 09:40:00 | 3430.72 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-06-19 09:35:00 | 3420.45 | 2023-06-19 11:50:00 | 3420.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-20 09:30:00 | 3415.00 | 2023-06-20 10:25:00 | 3395.27 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-06-20 09:30:00 | 3415.00 | 2023-06-20 15:20:00 | 3394.50 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2023-06-26 10:35:00 | 3401.95 | 2023-06-26 10:55:00 | 3392.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-28 09:40:00 | 3417.95 | 2023-06-28 09:45:00 | 3431.40 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-06-28 09:40:00 | 3417.95 | 2023-06-28 09:50:00 | 3417.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-30 09:45:00 | 3487.40 | 2023-06-30 09:55:00 | 3477.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-04 09:35:00 | 3519.90 | 2023-07-04 09:55:00 | 3531.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-04 09:35:00 | 3519.90 | 2023-07-04 10:10:00 | 3519.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-12 09:45:00 | 3498.05 | 2023-07-12 09:50:00 | 3504.94 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-13 10:15:00 | 3548.75 | 2023-07-13 10:25:00 | 3541.31 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-17 09:30:00 | 3549.00 | 2023-07-17 09:35:00 | 3561.33 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-17 09:30:00 | 3549.00 | 2023-07-17 09:45:00 | 3561.40 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-20 09:35:00 | 3691.55 | 2023-07-20 09:40:00 | 3681.57 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-21 10:50:00 | 3690.05 | 2023-07-21 11:25:00 | 3702.94 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-21 10:50:00 | 3690.05 | 2023-07-21 11:45:00 | 3690.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-24 11:10:00 | 3737.00 | 2023-07-24 11:45:00 | 3746.54 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-07-24 11:10:00 | 3737.00 | 2023-07-24 15:20:00 | 3780.00 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2023-07-26 10:05:00 | 3987.50 | 2023-07-26 10:40:00 | 3974.22 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-31 11:00:00 | 3974.05 | 2023-07-31 11:40:00 | 3982.52 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-09 09:30:00 | 4191.05 | 2023-08-09 09:35:00 | 4177.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-16 10:50:00 | 3799.40 | 2023-08-16 11:00:00 | 3789.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-17 11:05:00 | 3771.00 | 2023-08-17 11:10:00 | 3779.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-08-18 09:30:00 | 3731.55 | 2023-08-18 09:40:00 | 3714.38 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-08-18 09:30:00 | 3731.55 | 2023-08-18 10:50:00 | 3731.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-22 11:15:00 | 3777.35 | 2023-08-22 11:25:00 | 3782.58 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-24 10:35:00 | 3758.30 | 2023-08-24 10:40:00 | 3764.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-25 11:15:00 | 3702.05 | 2023-08-25 12:05:00 | 3692.51 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-25 11:15:00 | 3702.05 | 2023-08-25 12:10:00 | 3702.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-28 10:45:00 | 3719.35 | 2023-08-28 11:20:00 | 3731.80 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-08-28 10:45:00 | 3719.35 | 2023-08-28 12:35:00 | 3730.30 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2023-08-31 11:05:00 | 3668.80 | 2023-08-31 11:30:00 | 3661.76 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2023-08-31 11:05:00 | 3668.80 | 2023-08-31 11:50:00 | 3668.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-04 10:50:00 | 3642.80 | 2023-09-04 11:45:00 | 3650.85 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-09-04 10:50:00 | 3642.80 | 2023-09-04 13:40:00 | 3650.30 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2023-09-05 09:35:00 | 3640.60 | 2023-09-05 09:50:00 | 3648.24 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-09-08 10:45:00 | 3628.00 | 2023-09-08 13:25:00 | 3635.48 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-13 10:00:00 | 3686.60 | 2023-09-13 10:10:00 | 3676.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-18 10:40:00 | 3648.90 | 2023-09-18 11:40:00 | 3658.70 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-21 09:30:00 | 3670.75 | 2023-09-21 09:35:00 | 3661.38 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-09-21 09:30:00 | 3670.75 | 2023-09-21 10:35:00 | 3670.00 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2023-09-22 09:50:00 | 3591.15 | 2023-09-22 10:00:00 | 3574.79 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-22 09:50:00 | 3591.15 | 2023-09-22 12:30:00 | 3565.85 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2023-09-27 10:25:00 | 3545.65 | 2023-09-27 10:35:00 | 3539.17 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-09-29 10:35:00 | 3575.10 | 2023-09-29 10:40:00 | 3588.14 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-09-29 10:35:00 | 3575.10 | 2023-09-29 15:20:00 | 3615.85 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2023-10-05 10:50:00 | 3512.05 | 2023-10-05 11:35:00 | 3504.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-10-09 10:55:00 | 3518.85 | 2023-10-09 12:20:00 | 3527.14 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-10-09 10:55:00 | 3518.85 | 2023-10-09 15:20:00 | 3547.95 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2023-10-13 09:45:00 | 3618.00 | 2023-10-13 09:55:00 | 3630.92 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-10-13 09:45:00 | 3618.00 | 2023-10-13 10:15:00 | 3618.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-16 09:35:00 | 3594.10 | 2023-10-16 09:40:00 | 3601.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-10-25 10:20:00 | 3565.50 | 2023-10-25 11:35:00 | 3557.08 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-10-26 11:10:00 | 3533.35 | 2023-10-26 11:40:00 | 3542.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-27 09:30:00 | 3611.05 | 2023-10-27 09:50:00 | 3628.03 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-10-27 09:30:00 | 3611.05 | 2023-10-27 15:20:00 | 3679.00 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2023-10-31 09:40:00 | 3712.10 | 2023-10-31 09:55:00 | 3702.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-01 10:45:00 | 3700.95 | 2023-11-01 10:55:00 | 3711.69 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-11-02 09:55:00 | 3761.80 | 2023-11-02 10:05:00 | 3776.93 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-11-02 09:55:00 | 3761.80 | 2023-11-02 10:30:00 | 3761.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 10:35:00 | 3823.95 | 2023-11-06 13:40:00 | 3813.01 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-11-07 09:35:00 | 3869.85 | 2023-11-07 09:40:00 | 3858.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-10 09:35:00 | 4217.00 | 2023-11-10 09:40:00 | 4232.41 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-16 09:35:00 | 4375.00 | 2023-11-16 10:00:00 | 4362.91 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-20 10:05:00 | 4394.15 | 2023-11-20 10:10:00 | 4404.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-21 09:45:00 | 4446.95 | 2023-11-21 09:55:00 | 4457.64 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-11-21 09:45:00 | 4446.95 | 2023-11-21 10:05:00 | 4446.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 09:30:00 | 4538.70 | 2023-11-22 09:35:00 | 4528.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-24 09:45:00 | 4585.15 | 2023-11-24 09:50:00 | 4574.28 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-28 09:45:00 | 4603.00 | 2023-11-28 10:00:00 | 4592.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-11-29 09:50:00 | 4572.55 | 2023-11-29 09:55:00 | 4560.36 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-11-29 09:50:00 | 4572.55 | 2023-11-29 15:20:00 | 4596.75 | STOP_HIT | 0.50 | -0.53% |
| BUY | retest1 | 2023-11-30 10:15:00 | 4597.55 | 2023-11-30 10:40:00 | 4613.60 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-30 10:15:00 | 4597.55 | 2023-11-30 15:20:00 | 4743.90 | TARGET_HIT | 0.50 | 3.18% |
| SELL | retest1 | 2023-12-04 11:05:00 | 4620.15 | 2023-12-04 11:15:00 | 4602.35 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-04 11:05:00 | 4620.15 | 2023-12-04 11:25:00 | 4620.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-06 10:20:00 | 4776.50 | 2023-12-06 11:10:00 | 4760.54 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-07 09:35:00 | 4769.40 | 2023-12-07 09:45:00 | 4790.52 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-12-07 09:35:00 | 4769.40 | 2023-12-07 10:25:00 | 4784.95 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-13 10:15:00 | 4848.00 | 2023-12-13 10:20:00 | 4832.67 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-15 11:15:00 | 4941.55 | 2023-12-15 11:50:00 | 4929.74 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-20 09:30:00 | 4983.55 | 2023-12-20 09:35:00 | 4972.32 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-22 10:50:00 | 4890.25 | 2023-12-22 11:00:00 | 4905.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-01 11:05:00 | 5151.00 | 2024-01-01 11:20:00 | 5133.22 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-01 11:05:00 | 5151.00 | 2024-01-01 11:40:00 | 5151.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 10:35:00 | 5300.00 | 2024-01-04 10:45:00 | 5284.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-08 11:05:00 | 5176.80 | 2024-01-08 11:10:00 | 5157.26 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-01-08 11:05:00 | 5176.80 | 2024-01-08 11:25:00 | 5176.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-09 09:30:00 | 5260.95 | 2024-01-09 09:35:00 | 5246.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-10 10:50:00 | 5116.45 | 2024-01-10 11:10:00 | 5132.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-01-17 09:30:00 | 5081.60 | 2024-01-17 09:45:00 | 5101.99 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-01-17 09:30:00 | 5081.60 | 2024-01-17 09:50:00 | 5081.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:35:00 | 4961.10 | 2024-01-18 09:45:00 | 4933.87 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-01-18 09:35:00 | 4961.10 | 2024-01-18 10:30:00 | 4961.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:55:00 | 4920.00 | 2024-01-20 11:05:00 | 4907.52 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-01-20 10:55:00 | 4920.00 | 2024-01-20 15:20:00 | 4879.65 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-01-23 09:30:00 | 4921.00 | 2024-01-23 09:40:00 | 4942.41 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-01-23 09:30:00 | 4921.00 | 2024-01-23 09:55:00 | 4921.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-24 11:05:00 | 4973.35 | 2024-01-24 11:15:00 | 4953.94 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-01-30 11:10:00 | 4923.00 | 2024-01-30 11:15:00 | 4936.65 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-02-05 11:10:00 | 5075.90 | 2024-02-05 11:20:00 | 5098.34 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-05 11:10:00 | 5075.90 | 2024-02-05 11:55:00 | 5075.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-13 10:45:00 | 5276.95 | 2024-02-13 10:50:00 | 5256.23 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-02-21 09:50:00 | 5483.00 | 2024-02-21 10:20:00 | 5513.62 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-02-21 09:50:00 | 5483.00 | 2024-02-21 10:50:00 | 5483.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-23 10:40:00 | 5372.75 | 2024-02-23 10:50:00 | 5387.59 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-28 11:00:00 | 5032.60 | 2024-02-28 11:35:00 | 5013.14 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-28 11:00:00 | 5032.60 | 2024-02-28 15:20:00 | 5001.40 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-03-04 09:35:00 | 5118.45 | 2024-03-04 09:40:00 | 5131.71 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-03-05 11:00:00 | 5070.45 | 2024-03-05 12:15:00 | 5055.04 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-03-05 11:00:00 | 5070.45 | 2024-03-05 14:55:00 | 5070.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-13 11:05:00 | 5016.90 | 2024-03-13 11:55:00 | 4990.68 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-13 11:05:00 | 5016.90 | 2024-03-13 12:55:00 | 5016.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 10:30:00 | 4901.15 | 2024-03-19 11:45:00 | 4881.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-19 10:30:00 | 4901.15 | 2024-03-19 12:40:00 | 4901.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 10:00:00 | 4833.85 | 2024-03-20 10:20:00 | 4812.46 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-03-20 10:00:00 | 4833.85 | 2024-03-20 15:20:00 | 4781.20 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2024-04-02 09:50:00 | 4941.10 | 2024-04-02 09:55:00 | 4959.99 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-03 10:55:00 | 4897.50 | 2024-04-03 11:25:00 | 4906.34 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-04-04 10:50:00 | 4855.95 | 2024-04-04 10:55:00 | 4865.46 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-08 09:30:00 | 4876.60 | 2024-04-08 13:30:00 | 4850.74 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-04-08 09:30:00 | 4876.60 | 2024-04-08 14:55:00 | 4876.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-18 10:15:00 | 4677.00 | 2024-04-18 10:20:00 | 4655.80 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-04-18 10:15:00 | 4677.00 | 2024-04-18 10:25:00 | 4677.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-19 10:45:00 | 4581.45 | 2024-04-19 11:20:00 | 4595.82 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-26 10:50:00 | 4914.05 | 2024-04-26 11:00:00 | 4934.19 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-04-26 10:50:00 | 4914.05 | 2024-04-26 12:00:00 | 4914.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-02 09:35:00 | 4803.30 | 2024-05-02 09:40:00 | 4783.17 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-02 09:35:00 | 4803.30 | 2024-05-02 11:25:00 | 4783.45 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-09 09:55:00 | 5124.25 | 2024-05-09 11:40:00 | 5142.26 | STOP_HIT | 1.00 | -0.35% |
