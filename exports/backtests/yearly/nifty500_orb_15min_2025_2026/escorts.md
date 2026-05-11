# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2025-07-10 09:15:00 → 2026-05-08 15:25:00 (12088 bars)
- **Last close:** 3148.00
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 9 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 44
- **Target hits / Stop hits / Partials:** 9 / 44 / 20
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 10.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 14 | 35.9% | 4 | 25 | 10 | 0.05% | 2.1% |
| BUY @ 2nd Alert (retest1) | 39 | 14 | 35.9% | 4 | 25 | 10 | 0.05% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 15 | 44.1% | 5 | 19 | 10 | 0.24% | 8.1% |
| SELL @ 2nd Alert (retest1) | 34 | 15 | 44.1% | 5 | 19 | 10 | 0.24% | 8.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 73 | 29 | 39.7% | 9 | 44 | 20 | 0.14% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:25:00 | 3337.80 | 3363.26 | 0.00 | ORB-short ORB[3366.90,3394.90] vol=1.6x ATR=14.99 |
| Stop hit — per-position SL triggered | 2025-07-10 11:15:00 | 3352.79 | 3355.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:05:00 | 3304.90 | 3330.77 | 0.00 | ORB-short ORB[3320.00,3358.80] vol=2.4x ATR=7.82 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 3312.72 | 3330.61 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-07-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:00:00 | 3272.90 | 3289.97 | 0.00 | ORB-short ORB[3280.00,3321.20] vol=3.0x ATR=9.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:05:00 | 3259.34 | 3277.98 | 0.00 | T1 1.5R @ 3259.34 |
| Stop hit — per-position SL triggered | 2025-07-14 10:10:00 | 3272.90 | 3277.80 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 3457.00 | 3443.43 | 0.00 | ORB-long ORB[3406.00,3456.90] vol=3.8x ATR=10.16 |
| Stop hit — per-position SL triggered | 2025-07-18 09:35:00 | 3446.84 | 3445.06 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 3424.00 | 3436.98 | 0.00 | ORB-short ORB[3435.00,3456.70] vol=2.5x ATR=7.67 |
| Stop hit — per-position SL triggered | 2025-07-23 09:35:00 | 3431.67 | 3436.97 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:00:00 | 3446.80 | 3458.27 | 0.00 | ORB-short ORB[3454.30,3493.50] vol=2.0x ATR=7.97 |
| Stop hit — per-position SL triggered | 2025-07-24 10:05:00 | 3454.77 | 3457.81 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 3458.50 | 3441.55 | 0.00 | ORB-long ORB[3410.90,3449.30] vol=1.6x ATR=10.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:45:00 | 3474.85 | 3456.88 | 0.00 | T1 1.5R @ 3474.85 |
| Target hit | 2025-07-29 11:45:00 | 3472.90 | 3478.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-08-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 10:25:00 | 3323.20 | 3285.25 | 0.00 | ORB-long ORB[3250.70,3300.00] vol=3.1x ATR=13.90 |
| Stop hit — per-position SL triggered | 2025-08-04 12:40:00 | 3309.30 | 3302.44 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:25:00 | 3411.50 | 3393.25 | 0.00 | ORB-long ORB[3363.20,3410.00] vol=2.2x ATR=10.68 |
| Stop hit — per-position SL triggered | 2025-08-14 11:05:00 | 3400.82 | 3402.02 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 3625.50 | 3619.02 | 0.00 | ORB-long ORB[3578.00,3615.00] vol=5.4x ATR=10.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:25:00 | 3640.94 | 3623.56 | 0.00 | T1 1.5R @ 3640.94 |
| Stop hit — per-position SL triggered | 2025-08-20 10:50:00 | 3625.50 | 3627.68 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:55:00 | 3544.00 | 3560.28 | 0.00 | ORB-short ORB[3557.00,3599.00] vol=6.9x ATR=9.63 |
| Stop hit — per-position SL triggered | 2025-08-22 12:20:00 | 3553.63 | 3556.64 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-08-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 11:05:00 | 3591.60 | 3566.18 | 0.00 | ORB-long ORB[3550.00,3590.60] vol=1.9x ATR=9.44 |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 3582.16 | 3567.75 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:35:00 | 3507.10 | 3467.28 | 0.00 | ORB-long ORB[3449.70,3499.00] vol=2.0x ATR=12.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:40:00 | 3525.68 | 3479.61 | 0.00 | T1 1.5R @ 3525.68 |
| Target hit | 2025-08-29 13:20:00 | 3555.50 | 3556.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2025-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:45:00 | 3755.30 | 3731.69 | 0.00 | ORB-long ORB[3688.20,3738.70] vol=2.2x ATR=11.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:55:00 | 3772.65 | 3741.38 | 0.00 | T1 1.5R @ 3772.65 |
| Stop hit — per-position SL triggered | 2025-10-16 10:00:00 | 3755.30 | 3744.72 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 09:45:00 | 3680.00 | 3693.70 | 0.00 | ORB-short ORB[3684.20,3722.20] vol=2.6x ATR=9.55 |
| Stop hit — per-position SL triggered | 2025-10-23 10:00:00 | 3689.55 | 3691.67 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:15:00 | 3662.20 | 3674.26 | 0.00 | ORB-short ORB[3669.90,3686.50] vol=1.7x ATR=6.50 |
| Stop hit — per-position SL triggered | 2025-10-27 11:30:00 | 3668.70 | 3673.40 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 3695.00 | 3681.86 | 0.00 | ORB-long ORB[3651.00,3676.50] vol=6.0x ATR=9.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:35:00 | 3709.41 | 3690.69 | 0.00 | T1 1.5R @ 3709.41 |
| Stop hit — per-position SL triggered | 2025-10-28 10:10:00 | 3695.00 | 3698.29 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 3738.60 | 3721.93 | 0.00 | ORB-long ORB[3697.30,3731.00] vol=2.3x ATR=9.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:25:00 | 3752.60 | 3729.77 | 0.00 | T1 1.5R @ 3752.60 |
| Stop hit — per-position SL triggered | 2025-10-31 11:40:00 | 3738.60 | 3732.50 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:40:00 | 3727.20 | 3749.64 | 0.00 | ORB-short ORB[3761.70,3810.00] vol=1.6x ATR=14.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:55:00 | 3705.17 | 3740.84 | 0.00 | T1 1.5R @ 3705.17 |
| Target hit | 2025-11-06 15:20:00 | 3679.90 | 3718.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:35:00 | 3586.50 | 3566.82 | 0.00 | ORB-long ORB[3545.00,3578.00] vol=2.4x ATR=10.57 |
| Stop hit — per-position SL triggered | 2025-11-11 09:40:00 | 3575.93 | 3568.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-11-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 11:00:00 | 3578.60 | 3593.11 | 0.00 | ORB-short ORB[3587.10,3630.00] vol=2.3x ATR=8.17 |
| Stop hit — per-position SL triggered | 2025-11-12 11:10:00 | 3586.77 | 3592.00 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-11-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:05:00 | 3638.00 | 3607.59 | 0.00 | ORB-long ORB[3584.10,3618.00] vol=3.2x ATR=11.57 |
| Stop hit — per-position SL triggered | 2025-11-13 10:45:00 | 3626.43 | 3617.28 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-11-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:40:00 | 3600.50 | 3589.20 | 0.00 | ORB-long ORB[3552.00,3589.70] vol=1.9x ATR=10.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:50:00 | 3615.79 | 3597.15 | 0.00 | T1 1.5R @ 3615.79 |
| Stop hit — per-position SL triggered | 2025-11-17 10:00:00 | 3600.50 | 3598.84 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:10:00 | 3580.00 | 3592.46 | 0.00 | ORB-short ORB[3583.10,3615.40] vol=2.3x ATR=7.17 |
| Stop hit — per-position SL triggered | 2025-11-20 12:45:00 | 3587.17 | 3588.58 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 3578.00 | 3552.88 | 0.00 | ORB-long ORB[3542.50,3569.20] vol=1.6x ATR=10.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:00:00 | 3593.59 | 3560.29 | 0.00 | T1 1.5R @ 3593.59 |
| Stop hit — per-position SL triggered | 2025-11-21 10:05:00 | 3578.00 | 3561.27 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 3662.90 | 3643.88 | 0.00 | ORB-long ORB[3596.40,3645.00] vol=2.4x ATR=15.29 |
| Target hit | 2025-11-24 15:20:00 | 3682.20 | 3664.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 3660.90 | 3642.50 | 0.00 | ORB-long ORB[3620.60,3655.90] vol=2.3x ATR=10.79 |
| Stop hit — per-position SL triggered | 2025-11-26 09:40:00 | 3650.11 | 3643.55 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 3753.00 | 3750.69 | 0.00 | ORB-long ORB[3725.40,3742.40] vol=2.8x ATR=11.96 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 3741.04 | 3749.73 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 09:35:00 | 3734.20 | 3720.38 | 0.00 | ORB-long ORB[3710.00,3726.40] vol=1.8x ATR=10.11 |
| Stop hit — per-position SL triggered | 2025-12-08 09:50:00 | 3724.09 | 3724.73 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 3726.60 | 3704.70 | 0.00 | ORB-long ORB[3676.10,3703.00] vol=5.2x ATR=10.90 |
| Stop hit — per-position SL triggered | 2025-12-12 09:40:00 | 3715.70 | 3707.25 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:00:00 | 3685.90 | 3668.11 | 0.00 | ORB-long ORB[3645.00,3680.40] vol=2.2x ATR=8.45 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 3677.45 | 3668.33 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:40:00 | 3616.80 | 3628.27 | 0.00 | ORB-short ORB[3618.60,3666.90] vol=2.5x ATR=12.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 12:35:00 | 3598.77 | 3620.05 | 0.00 | T1 1.5R @ 3598.77 |
| Stop hit — per-position SL triggered | 2025-12-18 13:20:00 | 3616.80 | 3608.30 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:00:00 | 3637.50 | 3660.91 | 0.00 | ORB-short ORB[3639.10,3674.90] vol=2.3x ATR=14.05 |
| Stop hit — per-position SL triggered | 2025-12-19 11:20:00 | 3651.55 | 3647.14 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 11:05:00 | 3752.90 | 3731.46 | 0.00 | ORB-long ORB[3713.50,3750.20] vol=2.0x ATR=8.71 |
| Stop hit — per-position SL triggered | 2025-12-24 11:30:00 | 3744.19 | 3733.91 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 3860.30 | 3846.81 | 0.00 | ORB-long ORB[3822.20,3857.70] vol=1.6x ATR=8.61 |
| Stop hit — per-position SL triggered | 2026-01-02 10:20:00 | 3851.69 | 3849.24 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 3878.80 | 3863.99 | 0.00 | ORB-long ORB[3847.00,3877.00] vol=1.9x ATR=10.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:40:00 | 3894.91 | 3873.82 | 0.00 | T1 1.5R @ 3894.91 |
| Target hit | 2026-01-05 14:05:00 | 3923.10 | 3933.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2026-02-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:30:00 | 3830.00 | 3804.81 | 0.00 | ORB-long ORB[3762.20,3819.00] vol=1.6x ATR=15.88 |
| Stop hit — per-position SL triggered | 2026-02-09 09:35:00 | 3814.12 | 3807.64 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:05:00 | 3631.00 | 3670.89 | 0.00 | ORB-short ORB[3644.10,3697.00] vol=2.4x ATR=12.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 3612.58 | 3660.81 | 0.00 | T1 1.5R @ 3612.58 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 3631.00 | 3643.79 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 3551.50 | 3538.39 | 0.00 | ORB-long ORB[3514.90,3545.90] vol=1.9x ATR=13.13 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 3538.37 | 3539.47 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:25:00 | 3482.10 | 3522.08 | 0.00 | ORB-short ORB[3525.30,3568.00] vol=2.6x ATR=11.60 |
| Stop hit — per-position SL triggered | 2026-02-19 10:30:00 | 3493.70 | 3518.19 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 3598.30 | 3615.35 | 0.00 | ORB-short ORB[3599.80,3626.30] vol=1.7x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 3581.82 | 3610.39 | 0.00 | T1 1.5R @ 3581.82 |
| Stop hit — per-position SL triggered | 2026-02-26 14:35:00 | 3598.30 | 3597.56 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 3256.60 | 3267.52 | 0.00 | ORB-short ORB[3258.90,3288.30] vol=1.8x ATR=9.36 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 3265.96 | 3264.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 3370.80 | 3369.04 | 0.00 | ORB-long ORB[3335.10,3370.00] vol=2.1x ATR=15.45 |
| Stop hit — per-position SL triggered | 2026-03-11 10:20:00 | 3355.35 | 3369.52 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:10:00 | 3171.00 | 3199.35 | 0.00 | ORB-short ORB[3205.00,3229.80] vol=2.3x ATR=9.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:20:00 | 3156.58 | 3189.84 | 0.00 | T1 1.5R @ 3156.58 |
| Stop hit — per-position SL triggered | 2026-03-13 11:40:00 | 3171.00 | 3186.34 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 3006.30 | 3030.55 | 0.00 | ORB-short ORB[3016.00,3060.00] vol=1.9x ATR=11.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:55:00 | 2989.61 | 3021.96 | 0.00 | T1 1.5R @ 2989.61 |
| Target hit | 2026-03-19 15:20:00 | 2912.00 | 2943.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 3293.00 | 3269.60 | 0.00 | ORB-long ORB[3242.10,3287.00] vol=1.6x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:35:00 | 3303.72 | 3276.07 | 0.00 | T1 1.5R @ 3303.72 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 3293.00 | 3276.36 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 3302.90 | 3281.65 | 0.00 | ORB-long ORB[3273.00,3296.10] vol=1.7x ATR=10.13 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 3292.77 | 3287.24 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 3325.60 | 3354.96 | 0.00 | ORB-short ORB[3333.10,3377.90] vol=1.8x ATR=7.43 |
| Stop hit — per-position SL triggered | 2026-04-22 11:10:00 | 3333.03 | 3353.59 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3318.00 | 3351.13 | 0.00 | ORB-short ORB[3333.00,3382.00] vol=1.7x ATR=10.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 3302.40 | 3344.41 | 0.00 | T1 1.5R @ 3302.40 |
| Target hit | 2026-04-29 15:20:00 | 3301.70 | 3306.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 3267.10 | 3291.21 | 0.00 | ORB-short ORB[3277.00,3314.40] vol=1.5x ATR=12.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:00:00 | 3248.80 | 3273.54 | 0.00 | T1 1.5R @ 3248.80 |
| Target hit | 2026-04-30 13:55:00 | 3262.20 | 3242.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — SELL (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 3261.00 | 3292.19 | 0.00 | ORB-short ORB[3280.00,3325.60] vol=3.1x ATR=11.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:50:00 | 3243.14 | 3282.37 | 0.00 | T1 1.5R @ 3243.14 |
| Target hit | 2026-05-04 15:20:00 | 3203.90 | 3234.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 3200.00 | 3227.78 | 0.00 | ORB-short ORB[3202.90,3242.10] vol=1.9x ATR=9.05 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 3209.05 | 3226.11 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 3279.90 | 3262.05 | 0.00 | ORB-long ORB[3220.30,3263.90] vol=6.5x ATR=10.72 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 3269.18 | 3263.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-10 10:25:00 | 3337.80 | 2025-07-10 11:15:00 | 3352.79 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-11 11:05:00 | 3304.90 | 2025-07-11 11:10:00 | 3312.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-14 10:00:00 | 3272.90 | 2025-07-14 10:05:00 | 3259.34 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-14 10:00:00 | 3272.90 | 2025-07-14 10:10:00 | 3272.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-18 09:30:00 | 3457.00 | 2025-07-18 09:35:00 | 3446.84 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-23 09:30:00 | 3424.00 | 2025-07-23 09:35:00 | 3431.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-24 10:00:00 | 3446.80 | 2025-07-24 10:05:00 | 3454.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-29 09:40:00 | 3458.50 | 2025-07-29 09:45:00 | 3474.85 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-29 09:40:00 | 3458.50 | 2025-07-29 11:45:00 | 3472.90 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-04 10:25:00 | 3323.20 | 2025-08-04 12:40:00 | 3309.30 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-08-14 10:25:00 | 3411.50 | 2025-08-14 11:05:00 | 3400.82 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-20 10:10:00 | 3625.50 | 2025-08-20 10:25:00 | 3640.94 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-20 10:10:00 | 3625.50 | 2025-08-20 10:50:00 | 3625.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:55:00 | 3544.00 | 2025-08-22 12:20:00 | 3553.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-25 11:05:00 | 3591.60 | 2025-08-25 11:15:00 | 3582.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-29 10:35:00 | 3507.10 | 2025-08-29 11:40:00 | 3525.68 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-29 10:35:00 | 3507.10 | 2025-08-29 13:20:00 | 3555.50 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2025-10-16 09:45:00 | 3755.30 | 2025-10-16 09:55:00 | 3772.65 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-16 09:45:00 | 3755.30 | 2025-10-16 10:00:00 | 3755.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-23 09:45:00 | 3680.00 | 2025-10-23 10:00:00 | 3689.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-27 11:15:00 | 3662.20 | 2025-10-27 11:30:00 | 3668.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-28 09:30:00 | 3695.00 | 2025-10-28 09:35:00 | 3709.41 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-10-28 09:30:00 | 3695.00 | 2025-10-28 10:10:00 | 3695.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 11:10:00 | 3738.60 | 2025-10-31 11:25:00 | 3752.60 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-31 11:10:00 | 3738.60 | 2025-10-31 11:40:00 | 3738.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 10:40:00 | 3727.20 | 2025-11-06 10:55:00 | 3705.17 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-11-06 10:40:00 | 3727.20 | 2025-11-06 15:20:00 | 3679.90 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-11-11 09:35:00 | 3586.50 | 2025-11-11 09:40:00 | 3575.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-12 11:00:00 | 3578.60 | 2025-11-12 11:10:00 | 3586.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-13 10:05:00 | 3638.00 | 2025-11-13 10:45:00 | 3626.43 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-17 09:40:00 | 3600.50 | 2025-11-17 09:50:00 | 3615.79 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-11-17 09:40:00 | 3600.50 | 2025-11-17 10:00:00 | 3600.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-20 11:10:00 | 3580.00 | 2025-11-20 12:45:00 | 3587.17 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-21 09:35:00 | 3578.00 | 2025-11-21 10:00:00 | 3593.59 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-21 09:35:00 | 3578.00 | 2025-11-21 10:05:00 | 3578.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 09:30:00 | 3662.90 | 2025-11-24 15:20:00 | 3682.20 | TARGET_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2025-11-26 09:35:00 | 3660.90 | 2025-11-26 09:40:00 | 3650.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-05 10:05:00 | 3753.00 | 2025-12-05 10:25:00 | 3741.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-08 09:35:00 | 3734.20 | 2025-12-08 09:50:00 | 3724.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-12 09:35:00 | 3726.60 | 2025-12-12 09:40:00 | 3715.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-15 11:00:00 | 3685.90 | 2025-12-15 11:05:00 | 3677.45 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-18 10:40:00 | 3616.80 | 2025-12-18 12:35:00 | 3598.77 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-18 10:40:00 | 3616.80 | 2025-12-18 13:20:00 | 3616.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 10:00:00 | 3637.50 | 2025-12-19 11:20:00 | 3651.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-24 11:05:00 | 3752.90 | 2025-12-24 11:30:00 | 3744.19 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-02 10:00:00 | 3860.30 | 2026-01-02 10:20:00 | 3851.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-05 09:30:00 | 3878.80 | 2026-01-05 09:40:00 | 3894.91 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-05 09:30:00 | 3878.80 | 2026-01-05 14:05:00 | 3923.10 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2026-02-09 09:30:00 | 3830.00 | 2026-02-09 09:35:00 | 3814.12 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-12 11:05:00 | 3631.00 | 2026-02-12 11:15:00 | 3612.58 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-12 11:05:00 | 3631.00 | 2026-02-12 11:30:00 | 3631.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:35:00 | 3551.50 | 2026-02-18 09:40:00 | 3538.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-19 10:25:00 | 3482.10 | 2026-02-19 10:30:00 | 3493.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-26 11:05:00 | 3598.30 | 2026-02-26 11:30:00 | 3581.82 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-26 11:05:00 | 3598.30 | 2026-02-26 14:35:00 | 3598.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 09:50:00 | 3256.60 | 2026-03-05 11:20:00 | 3265.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-11 09:35:00 | 3370.80 | 2026-03-11 10:20:00 | 3355.35 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-13 11:10:00 | 3171.00 | 2026-03-13 11:20:00 | 3156.58 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-13 11:10:00 | 3171.00 | 2026-03-13 11:40:00 | 3171.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:50:00 | 3006.30 | 2026-03-19 09:55:00 | 2989.61 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-19 09:50:00 | 3006.30 | 2026-03-19 15:20:00 | 2912.00 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2026-04-16 11:15:00 | 3293.00 | 2026-04-16 11:35:00 | 3303.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-16 11:15:00 | 3293.00 | 2026-04-16 11:40:00 | 3293.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:05:00 | 3302.90 | 2026-04-17 10:30:00 | 3292.77 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-22 11:05:00 | 3325.60 | 2026-04-22 11:10:00 | 3333.03 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-29 10:30:00 | 3318.00 | 2026-04-29 10:45:00 | 3302.40 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-29 10:30:00 | 3318.00 | 2026-04-29 15:20:00 | 3301.70 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3267.10 | 2026-04-30 11:00:00 | 3248.80 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3267.10 | 2026-04-30 13:55:00 | 3262.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-05-04 11:10:00 | 3261.00 | 2026-05-04 11:50:00 | 3243.14 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-04 11:10:00 | 3261.00 | 2026-05-04 15:20:00 | 3203.90 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2026-05-05 11:00:00 | 3200.00 | 2026-05-05 11:35:00 | 3209.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-06 09:30:00 | 3279.90 | 2026-05-06 09:35:00 | 3269.18 | STOP_HIT | 1.00 | -0.33% |
