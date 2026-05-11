# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 3955.00
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
| PARTIAL | 40 |
| TARGET_HIT | 19 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 78
- **Target hits / Stop hits / Partials:** 19 / 78 / 40
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 24.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 36 | 47.4% | 13 | 40 | 23 | 0.25% | 19.1% |
| BUY @ 2nd Alert (retest1) | 76 | 36 | 47.4% | 13 | 40 | 23 | 0.25% | 19.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 23 | 37.7% | 6 | 38 | 17 | 0.08% | 5.0% |
| SELL @ 2nd Alert (retest1) | 61 | 23 | 37.7% | 6 | 38 | 17 | 0.08% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 137 | 59 | 43.1% | 19 | 78 | 40 | 0.18% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:45:00 | 2765.50 | 2779.62 | 0.00 | ORB-short ORB[2769.05,2793.85] vol=1.9x ATR=6.65 |
| Stop hit — per-position SL triggered | 2023-05-17 10:50:00 | 2772.15 | 2779.32 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 10:05:00 | 2838.60 | 2831.04 | 0.00 | ORB-long ORB[2801.10,2837.95] vol=2.0x ATR=7.73 |
| Stop hit — per-position SL triggered | 2023-05-18 10:25:00 | 2830.87 | 2830.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 11:05:00 | 2839.60 | 2854.04 | 0.00 | ORB-short ORB[2856.00,2875.00] vol=1.8x ATR=8.49 |
| Stop hit — per-position SL triggered | 2023-05-19 11:15:00 | 2848.09 | 2853.76 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 11:10:00 | 2867.40 | 2859.51 | 0.00 | ORB-long ORB[2825.00,2865.00] vol=2.0x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 12:00:00 | 2877.25 | 2863.73 | 0.00 | T1 1.5R @ 2877.25 |
| Target hit | 2023-05-22 14:20:00 | 2872.70 | 2872.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2023-05-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:55:00 | 2913.55 | 2888.36 | 0.00 | ORB-long ORB[2876.40,2893.40] vol=2.9x ATR=10.20 |
| Stop hit — per-position SL triggered | 2023-05-23 10:00:00 | 2903.35 | 2891.31 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:05:00 | 2858.00 | 2833.92 | 0.00 | ORB-long ORB[2791.35,2825.95] vol=3.2x ATR=12.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 10:25:00 | 2876.75 | 2847.80 | 0.00 | T1 1.5R @ 2876.75 |
| Target hit | 2023-05-30 15:20:00 | 2982.00 | 2958.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 10:15:00 | 3139.95 | 3114.25 | 0.00 | ORB-long ORB[3091.90,3131.25] vol=2.4x ATR=11.17 |
| Stop hit — per-position SL triggered | 2023-06-06 10:20:00 | 3128.78 | 3115.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:35:00 | 3138.30 | 3124.69 | 0.00 | ORB-long ORB[3095.45,3128.60] vol=2.1x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:50:00 | 3153.56 | 3133.52 | 0.00 | T1 1.5R @ 3153.56 |
| Target hit | 2023-06-07 12:40:00 | 3142.75 | 3144.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 3166.75 | 3151.86 | 0.00 | ORB-long ORB[3125.35,3164.00] vol=1.9x ATR=7.71 |
| Stop hit — per-position SL triggered | 2023-06-09 09:35:00 | 3159.04 | 3153.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:30:00 | 3333.05 | 3324.25 | 0.00 | ORB-long ORB[3289.65,3328.45] vol=3.2x ATR=9.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 09:40:00 | 3346.79 | 3330.41 | 0.00 | T1 1.5R @ 3346.79 |
| Target hit | 2023-06-15 10:40:00 | 3345.65 | 3350.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2023-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 09:30:00 | 3416.05 | 3431.77 | 0.00 | ORB-short ORB[3422.05,3460.00] vol=2.2x ATR=14.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 10:25:00 | 3394.76 | 3420.19 | 0.00 | T1 1.5R @ 3394.76 |
| Target hit | 2023-06-19 15:20:00 | 3404.80 | 3410.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-06-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:35:00 | 3338.30 | 3370.21 | 0.00 | ORB-short ORB[3358.65,3398.00] vol=4.0x ATR=10.21 |
| Stop hit — per-position SL triggered | 2023-06-21 10:40:00 | 3348.51 | 3369.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 10:20:00 | 3339.00 | 3354.88 | 0.00 | ORB-short ORB[3345.70,3383.15] vol=1.6x ATR=11.43 |
| Stop hit — per-position SL triggered | 2023-06-22 11:00:00 | 3350.43 | 3354.59 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:30:00 | 3351.85 | 3341.69 | 0.00 | ORB-long ORB[3316.10,3348.45] vol=1.6x ATR=13.97 |
| Stop hit — per-position SL triggered | 2023-06-26 09:50:00 | 3337.88 | 3343.02 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:35:00 | 3284.95 | 3310.94 | 0.00 | ORB-short ORB[3304.25,3327.40] vol=3.5x ATR=7.92 |
| Stop hit — per-position SL triggered | 2023-06-27 10:40:00 | 3292.87 | 3309.80 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:50:00 | 3169.55 | 3183.94 | 0.00 | ORB-short ORB[3184.95,3216.55] vol=1.9x ATR=11.66 |
| Stop hit — per-position SL triggered | 2023-07-03 10:05:00 | 3181.21 | 3179.61 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:00:00 | 3189.20 | 3168.80 | 0.00 | ORB-long ORB[3162.05,3187.00] vol=1.8x ATR=14.79 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 3174.41 | 3168.41 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:30:00 | 3222.95 | 3200.76 | 0.00 | ORB-long ORB[3175.00,3221.45] vol=2.1x ATR=11.72 |
| Stop hit — per-position SL triggered | 2023-07-06 12:55:00 | 3211.23 | 3212.13 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 10:45:00 | 3194.10 | 3205.04 | 0.00 | ORB-short ORB[3200.15,3235.25] vol=2.1x ATR=7.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:25:00 | 3182.96 | 3200.75 | 0.00 | T1 1.5R @ 3182.96 |
| Stop hit — per-position SL triggered | 2023-07-10 11:40:00 | 3194.10 | 3199.77 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 3385.90 | 3379.16 | 0.00 | ORB-long ORB[3361.20,3385.05] vol=1.7x ATR=11.05 |
| Stop hit — per-position SL triggered | 2023-07-14 10:20:00 | 3374.85 | 3381.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 11:10:00 | 3395.40 | 3386.32 | 0.00 | ORB-long ORB[3359.25,3394.70] vol=1.6x ATR=7.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:15:00 | 3407.28 | 3388.76 | 0.00 | T1 1.5R @ 3407.28 |
| Stop hit — per-position SL triggered | 2023-07-18 11:35:00 | 3395.40 | 3397.05 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:05:00 | 3527.10 | 3520.75 | 0.00 | ORB-long ORB[3501.00,3526.85] vol=2.3x ATR=10.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 11:10:00 | 3542.48 | 3524.61 | 0.00 | T1 1.5R @ 3542.48 |
| Stop hit — per-position SL triggered | 2023-07-20 11:30:00 | 3527.10 | 3526.27 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:25:00 | 3483.25 | 3516.05 | 0.00 | ORB-short ORB[3522.80,3569.90] vol=4.7x ATR=11.54 |
| Stop hit — per-position SL triggered | 2023-07-25 10:30:00 | 3494.79 | 3503.96 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 10:45:00 | 3511.30 | 3500.05 | 0.00 | ORB-long ORB[3461.60,3498.75] vol=1.6x ATR=10.87 |
| Stop hit — per-position SL triggered | 2023-08-01 12:05:00 | 3500.43 | 3501.97 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:55:00 | 3419.50 | 3432.41 | 0.00 | ORB-short ORB[3420.00,3464.80] vol=1.9x ATR=8.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 11:15:00 | 3406.58 | 3426.61 | 0.00 | T1 1.5R @ 3406.58 |
| Stop hit — per-position SL triggered | 2023-08-04 11:25:00 | 3419.50 | 3410.78 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:55:00 | 3575.35 | 3576.36 | 0.00 | ORB-short ORB[3577.75,3625.95] vol=2.0x ATR=12.29 |
| Stop hit — per-position SL triggered | 2023-08-10 11:40:00 | 3587.64 | 3575.49 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:40:00 | 3631.95 | 3617.96 | 0.00 | ORB-long ORB[3580.00,3626.05] vol=2.1x ATR=16.92 |
| Stop hit — per-position SL triggered | 2023-08-11 09:50:00 | 3615.03 | 3618.21 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 3586.25 | 3598.51 | 0.00 | ORB-short ORB[3588.00,3621.70] vol=1.9x ATR=6.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 12:10:00 | 3577.01 | 3594.74 | 0.00 | T1 1.5R @ 3577.01 |
| Target hit | 2023-08-17 15:20:00 | 3526.90 | 3563.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2023-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:50:00 | 3520.85 | 3539.63 | 0.00 | ORB-short ORB[3531.00,3561.40] vol=2.1x ATR=12.06 |
| Stop hit — per-position SL triggered | 2023-08-18 10:20:00 | 3532.91 | 3537.03 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:45:00 | 3697.80 | 3682.10 | 0.00 | ORB-long ORB[3656.00,3686.25] vol=2.4x ATR=9.60 |
| Stop hit — per-position SL triggered | 2023-08-29 12:40:00 | 3688.20 | 3696.04 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 3659.00 | 3682.54 | 0.00 | ORB-short ORB[3666.05,3720.00] vol=2.8x ATR=9.06 |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 3668.06 | 3677.89 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:50:00 | 3611.70 | 3621.37 | 0.00 | ORB-short ORB[3613.25,3649.80] vol=1.9x ATR=8.90 |
| Stop hit — per-position SL triggered | 2023-09-07 11:55:00 | 3620.60 | 3618.39 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:30:00 | 3596.10 | 3607.72 | 0.00 | ORB-short ORB[3600.00,3629.00] vol=2.2x ATR=10.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:35:00 | 3579.65 | 3596.84 | 0.00 | T1 1.5R @ 3579.65 |
| Target hit | 2023-09-20 14:15:00 | 3568.65 | 3566.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2023-09-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:00:00 | 3503.00 | 3528.60 | 0.00 | ORB-short ORB[3531.15,3566.30] vol=2.0x ATR=10.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 10:15:00 | 3487.03 | 3513.06 | 0.00 | T1 1.5R @ 3487.03 |
| Stop hit — per-position SL triggered | 2023-09-22 12:05:00 | 3503.00 | 3501.67 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:45:00 | 3421.05 | 3452.91 | 0.00 | ORB-short ORB[3454.20,3492.80] vol=1.6x ATR=10.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 11:15:00 | 3405.27 | 3445.73 | 0.00 | T1 1.5R @ 3405.27 |
| Stop hit — per-position SL triggered | 2023-09-25 12:30:00 | 3421.05 | 3438.28 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:45:00 | 3493.15 | 3470.51 | 0.00 | ORB-long ORB[3440.60,3481.80] vol=1.7x ATR=11.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 09:55:00 | 3509.86 | 3481.10 | 0.00 | T1 1.5R @ 3509.86 |
| Stop hit — per-position SL triggered | 2023-09-28 10:05:00 | 3493.15 | 3484.26 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 11:05:00 | 3533.50 | 3506.25 | 0.00 | ORB-long ORB[3477.10,3519.65] vol=1.7x ATR=8.49 |
| Stop hit — per-position SL triggered | 2023-09-29 11:20:00 | 3525.01 | 3507.23 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 09:35:00 | 3545.40 | 3535.53 | 0.00 | ORB-long ORB[3502.30,3539.95] vol=1.7x ATR=12.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 09:40:00 | 3564.32 | 3549.48 | 0.00 | T1 1.5R @ 3564.32 |
| Target hit | 2023-10-03 10:40:00 | 3555.80 | 3570.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2023-10-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:35:00 | 3499.70 | 3536.62 | 0.00 | ORB-short ORB[3509.95,3544.20] vol=3.5x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 11:05:00 | 3483.43 | 3523.26 | 0.00 | T1 1.5R @ 3483.43 |
| Stop hit — per-position SL triggered | 2023-10-05 11:10:00 | 3499.70 | 3517.27 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 09:45:00 | 3418.20 | 3425.39 | 0.00 | ORB-short ORB[3419.15,3450.30] vol=1.8x ATR=8.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 09:50:00 | 3405.75 | 3421.47 | 0.00 | T1 1.5R @ 3405.75 |
| Target hit | 2023-10-12 15:20:00 | 3372.00 | 3386.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2023-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:30:00 | 3394.00 | 3388.74 | 0.00 | ORB-long ORB[3380.15,3392.60] vol=2.3x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 10:35:00 | 3404.04 | 3389.59 | 0.00 | T1 1.5R @ 3404.04 |
| Target hit | 2023-10-16 15:20:00 | 3466.60 | 3423.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2023-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:30:00 | 3539.00 | 3532.17 | 0.00 | ORB-long ORB[3505.05,3538.80] vol=1.7x ATR=14.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:55:00 | 3560.47 | 3551.75 | 0.00 | T1 1.5R @ 3560.47 |
| Target hit | 2023-10-18 11:45:00 | 3554.50 | 3557.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:15:00 | 3558.10 | 3530.04 | 0.00 | ORB-long ORB[3500.00,3538.10] vol=1.8x ATR=12.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 10:35:00 | 3577.21 | 3540.10 | 0.00 | T1 1.5R @ 3577.21 |
| Stop hit — per-position SL triggered | 2023-10-19 10:40:00 | 3558.10 | 3547.41 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 09:45:00 | 3490.00 | 3483.67 | 0.00 | ORB-long ORB[3454.05,3489.00] vol=2.2x ATR=20.61 |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 3469.39 | 3484.16 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 10:40:00 | 3550.70 | 3527.52 | 0.00 | ORB-long ORB[3502.90,3543.00] vol=5.1x ATR=15.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 11:10:00 | 3573.35 | 3565.96 | 0.00 | T1 1.5R @ 3573.35 |
| Target hit | 2023-11-01 15:20:00 | 3673.10 | 3646.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2023-11-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:50:00 | 3708.90 | 3684.74 | 0.00 | ORB-long ORB[3661.30,3699.90] vol=2.4x ATR=12.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:05:00 | 3727.96 | 3705.51 | 0.00 | T1 1.5R @ 3727.96 |
| Target hit | 2023-11-02 12:05:00 | 3727.25 | 3731.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2023-11-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:20:00 | 3750.00 | 3729.73 | 0.00 | ORB-long ORB[3710.00,3739.50] vol=1.7x ATR=12.77 |
| Stop hit — per-position SL triggered | 2023-11-03 10:30:00 | 3737.23 | 3730.45 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 09:50:00 | 3582.00 | 3606.89 | 0.00 | ORB-short ORB[3601.10,3647.05] vol=1.7x ATR=15.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:25:00 | 3558.93 | 3597.38 | 0.00 | T1 1.5R @ 3558.93 |
| Stop hit — per-position SL triggered | 2023-11-07 13:30:00 | 3582.00 | 3573.49 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:35:00 | 3630.95 | 3608.24 | 0.00 | ORB-long ORB[3594.00,3619.00] vol=1.7x ATR=10.47 |
| Stop hit — per-position SL triggered | 2023-11-10 10:55:00 | 3620.48 | 3610.56 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-11-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 09:30:00 | 3566.60 | 3579.18 | 0.00 | ORB-short ORB[3569.40,3603.80] vol=1.5x ATR=11.44 |
| Stop hit — per-position SL triggered | 2023-11-16 10:35:00 | 3578.04 | 3567.09 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 11:05:00 | 3596.45 | 3576.12 | 0.00 | ORB-long ORB[3548.35,3584.95] vol=5.6x ATR=9.47 |
| Stop hit — per-position SL triggered | 2023-11-29 11:50:00 | 3586.98 | 3582.28 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 3586.40 | 3606.71 | 0.00 | ORB-short ORB[3603.00,3631.00] vol=3.5x ATR=11.57 |
| Stop hit — per-position SL triggered | 2023-11-30 10:20:00 | 3597.97 | 3598.65 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:55:00 | 3684.45 | 3665.81 | 0.00 | ORB-long ORB[3653.55,3673.05] vol=3.3x ATR=8.57 |
| Stop hit — per-position SL triggered | 2023-12-05 12:35:00 | 3675.88 | 3675.65 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:10:00 | 3712.00 | 3709.26 | 0.00 | ORB-long ORB[3695.05,3709.00] vol=1.7x ATR=9.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:45:00 | 3725.81 | 3712.35 | 0.00 | T1 1.5R @ 3725.81 |
| Target hit | 2023-12-07 15:20:00 | 3764.00 | 3746.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2023-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:25:00 | 3741.35 | 3755.88 | 0.00 | ORB-short ORB[3747.70,3769.00] vol=1.5x ATR=10.01 |
| Stop hit — per-position SL triggered | 2023-12-08 10:30:00 | 3751.36 | 3755.73 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:20:00 | 3710.20 | 3729.84 | 0.00 | ORB-short ORB[3723.25,3770.40] vol=1.8x ATR=13.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 10:30:00 | 3689.31 | 3723.78 | 0.00 | T1 1.5R @ 3689.31 |
| Stop hit — per-position SL triggered | 2023-12-11 10:40:00 | 3710.20 | 3719.40 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:55:00 | 3668.00 | 3703.39 | 0.00 | ORB-short ORB[3684.55,3725.90] vol=2.8x ATR=9.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 11:30:00 | 3654.19 | 3697.93 | 0.00 | T1 1.5R @ 3654.19 |
| Target hit | 2023-12-12 15:20:00 | 3610.00 | 3642.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2023-12-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 11:00:00 | 3643.20 | 3655.47 | 0.00 | ORB-short ORB[3646.20,3678.90] vol=2.6x ATR=10.88 |
| Stop hit — per-position SL triggered | 2023-12-14 11:25:00 | 3654.08 | 3654.98 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:05:00 | 3564.90 | 3554.51 | 0.00 | ORB-long ORB[3540.00,3561.35] vol=5.3x ATR=9.65 |
| Stop hit — per-position SL triggered | 2023-12-20 10:25:00 | 3555.25 | 3555.88 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:00:00 | 3531.30 | 3518.76 | 0.00 | ORB-long ORB[3497.10,3529.95] vol=1.9x ATR=9.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:35:00 | 3545.40 | 3525.49 | 0.00 | T1 1.5R @ 3545.40 |
| Target hit | 2023-12-22 12:10:00 | 3547.90 | 3548.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2023-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:45:00 | 3603.15 | 3590.20 | 0.00 | ORB-long ORB[3576.00,3595.85] vol=1.9x ATR=8.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 12:55:00 | 3615.79 | 3598.53 | 0.00 | T1 1.5R @ 3615.79 |
| Stop hit — per-position SL triggered | 2023-12-27 13:10:00 | 3603.15 | 3598.85 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:35:00 | 3631.55 | 3617.77 | 0.00 | ORB-long ORB[3596.20,3628.60] vol=2.6x ATR=8.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:50:00 | 3644.45 | 3624.56 | 0.00 | T1 1.5R @ 3644.45 |
| Target hit | 2023-12-28 15:20:00 | 3679.95 | 3660.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2023-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:40:00 | 3689.25 | 3672.56 | 0.00 | ORB-long ORB[3640.00,3679.00] vol=3.4x ATR=9.83 |
| Stop hit — per-position SL triggered | 2023-12-29 15:20:00 | 3687.25 | 3688.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2024-01-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 10:50:00 | 3627.45 | 3604.69 | 0.00 | ORB-long ORB[3594.30,3620.50] vol=1.6x ATR=8.54 |
| Stop hit — per-position SL triggered | 2024-01-03 11:05:00 | 3618.91 | 3607.32 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:55:00 | 3601.50 | 3583.89 | 0.00 | ORB-long ORB[3525.75,3579.00] vol=12.3x ATR=10.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 11:15:00 | 3617.02 | 3587.63 | 0.00 | T1 1.5R @ 3617.02 |
| Stop hit — per-position SL triggered | 2024-01-04 12:35:00 | 3601.50 | 3599.07 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:40:00 | 3588.90 | 3604.20 | 0.00 | ORB-short ORB[3589.30,3614.90] vol=3.1x ATR=8.11 |
| Stop hit — per-position SL triggered | 2024-01-05 11:30:00 | 3597.01 | 3601.16 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:55:00 | 3596.45 | 3615.62 | 0.00 | ORB-short ORB[3614.50,3650.00] vol=2.9x ATR=10.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:50:00 | 3580.23 | 3604.53 | 0.00 | T1 1.5R @ 3580.23 |
| Stop hit — per-position SL triggered | 2024-01-08 11:00:00 | 3596.45 | 3604.20 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:40:00 | 3646.95 | 3625.96 | 0.00 | ORB-long ORB[3610.10,3633.40] vol=2.6x ATR=11.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:45:00 | 3663.48 | 3635.23 | 0.00 | T1 1.5R @ 3663.48 |
| Stop hit — per-position SL triggered | 2024-01-09 10:50:00 | 3646.95 | 3635.76 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:30:00 | 3652.40 | 3635.91 | 0.00 | ORB-long ORB[3601.10,3648.80] vol=1.8x ATR=13.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 09:45:00 | 3673.01 | 3655.77 | 0.00 | T1 1.5R @ 3673.01 |
| Target hit | 2024-01-10 10:45:00 | 3767.00 | 3767.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — BUY (started 2024-01-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 11:05:00 | 3778.05 | 3747.35 | 0.00 | ORB-long ORB[3721.70,3762.35] vol=3.2x ATR=11.09 |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 3766.96 | 3748.94 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:15:00 | 3703.80 | 3677.42 | 0.00 | ORB-long ORB[3646.05,3677.70] vol=1.9x ATR=10.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:40:00 | 3719.76 | 3687.74 | 0.00 | T1 1.5R @ 3719.76 |
| Stop hit — per-position SL triggered | 2024-01-19 10:55:00 | 3703.80 | 3706.71 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:50:00 | 3710.05 | 3721.51 | 0.00 | ORB-short ORB[3719.65,3739.75] vol=1.9x ATR=8.21 |
| Stop hit — per-position SL triggered | 2024-01-20 11:20:00 | 3718.26 | 3720.18 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 10:05:00 | 3652.65 | 3678.56 | 0.00 | ORB-short ORB[3670.25,3708.95] vol=4.5x ATR=14.27 |
| Stop hit — per-position SL triggered | 2024-01-23 10:10:00 | 3666.92 | 3675.37 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 09:30:00 | 4012.25 | 3983.75 | 0.00 | ORB-long ORB[3931.55,3987.15] vol=2.9x ATR=20.68 |
| Stop hit — per-position SL triggered | 2024-01-29 09:45:00 | 3991.57 | 3991.59 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:30:00 | 4431.95 | 4413.68 | 0.00 | ORB-long ORB[4360.10,4420.45] vol=2.6x ATR=26.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 09:35:00 | 4471.93 | 4426.39 | 0.00 | T1 1.5R @ 4471.93 |
| Stop hit — per-position SL triggered | 2024-02-07 09:40:00 | 4431.95 | 4428.00 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 11:15:00 | 3952.00 | 3931.53 | 0.00 | ORB-long ORB[3897.05,3950.00] vol=1.8x ATR=13.74 |
| Stop hit — per-position SL triggered | 2024-02-14 11:20:00 | 3938.26 | 3931.81 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-02-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:30:00 | 3722.15 | 3768.16 | 0.00 | ORB-short ORB[3765.40,3817.70] vol=1.5x ATR=13.99 |
| Stop hit — per-position SL triggered | 2024-02-22 10:35:00 | 3736.14 | 3761.94 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:50:00 | 3694.80 | 3717.50 | 0.00 | ORB-short ORB[3712.50,3750.20] vol=2.1x ATR=10.05 |
| Stop hit — per-position SL triggered | 2024-02-26 10:05:00 | 3704.85 | 3714.64 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 09:50:00 | 3695.40 | 3705.71 | 0.00 | ORB-short ORB[3700.00,3734.70] vol=3.4x ATR=10.65 |
| Stop hit — per-position SL triggered | 2024-02-27 10:25:00 | 3706.05 | 3702.50 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 3731.00 | 3774.22 | 0.00 | ORB-short ORB[3780.05,3811.60] vol=2.3x ATR=12.32 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 3743.32 | 3771.68 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:15:00 | 3642.85 | 3664.89 | 0.00 | ORB-short ORB[3650.00,3695.00] vol=2.7x ATR=14.48 |
| Stop hit — per-position SL triggered | 2024-02-29 12:40:00 | 3657.33 | 3656.29 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-03-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:50:00 | 3702.65 | 3695.68 | 0.00 | ORB-long ORB[3635.25,3676.50] vol=1.9x ATR=18.03 |
| Stop hit — per-position SL triggered | 2024-03-01 15:10:00 | 3684.62 | 3699.68 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 09:35:00 | 3731.60 | 3715.07 | 0.00 | ORB-long ORB[3685.00,3714.95] vol=1.8x ATR=13.23 |
| Stop hit — per-position SL triggered | 2024-03-05 09:40:00 | 3718.37 | 3715.02 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 11:10:00 | 3633.70 | 3664.26 | 0.00 | ORB-short ORB[3639.05,3687.95] vol=7.9x ATR=12.56 |
| Stop hit — per-position SL triggered | 2024-03-12 11:15:00 | 3646.26 | 3662.93 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:55:00 | 3736.60 | 3758.28 | 0.00 | ORB-short ORB[3774.05,3810.00] vol=12.5x ATR=12.13 |
| Stop hit — per-position SL triggered | 2024-03-26 11:35:00 | 3748.73 | 3756.07 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 09:30:00 | 3874.65 | 3846.88 | 0.00 | ORB-long ORB[3800.00,3845.50] vol=4.6x ATR=9.29 |
| Stop hit — per-position SL triggered | 2024-03-27 09:35:00 | 3865.36 | 3849.25 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 3976.15 | 3997.36 | 0.00 | ORB-short ORB[3985.00,4030.60] vol=4.3x ATR=15.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 10:50:00 | 3953.08 | 3988.28 | 0.00 | T1 1.5R @ 3953.08 |
| Stop hit — per-position SL triggered | 2024-04-04 12:55:00 | 3976.15 | 3971.20 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:55:00 | 4010.00 | 3981.74 | 0.00 | ORB-long ORB[3952.80,3997.60] vol=4.9x ATR=15.95 |
| Stop hit — per-position SL triggered | 2024-04-05 10:10:00 | 3994.05 | 3989.86 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:40:00 | 4062.00 | 4083.46 | 0.00 | ORB-short ORB[4078.15,4130.55] vol=2.7x ATR=9.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 11:05:00 | 4048.24 | 4060.25 | 0.00 | T1 1.5R @ 4048.24 |
| Target hit | 2024-04-10 12:30:00 | 4060.00 | 4054.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — SELL (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 3996.00 | 4007.60 | 0.00 | ORB-short ORB[4001.00,4032.00] vol=1.6x ATR=13.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 10:20:00 | 3975.32 | 4000.92 | 0.00 | T1 1.5R @ 3975.32 |
| Stop hit — per-position SL triggered | 2024-04-12 10:30:00 | 3996.00 | 3999.73 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:30:00 | 3857.05 | 3848.35 | 0.00 | ORB-long ORB[3800.00,3843.30] vol=14.4x ATR=14.61 |
| Stop hit — per-position SL triggered | 2024-04-16 09:35:00 | 3842.44 | 3848.35 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 11:05:00 | 3859.90 | 3815.76 | 0.00 | ORB-long ORB[3773.40,3830.00] vol=3.2x ATR=11.93 |
| Stop hit — per-position SL triggered | 2024-04-19 11:15:00 | 3847.97 | 3822.28 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:00:00 | 3839.70 | 3866.39 | 0.00 | ORB-short ORB[3852.10,3888.00] vol=1.9x ATR=11.23 |
| Stop hit — per-position SL triggered | 2024-04-22 10:45:00 | 3850.93 | 3855.65 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 11:10:00 | 3833.10 | 3834.33 | 0.00 | ORB-short ORB[3841.00,3877.60] vol=5.6x ATR=8.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 11:25:00 | 3820.92 | 3834.08 | 0.00 | T1 1.5R @ 3820.92 |
| Stop hit — per-position SL triggered | 2024-04-23 11:35:00 | 3833.10 | 3834.06 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:50:00 | 3906.95 | 3877.94 | 0.00 | ORB-long ORB[3833.30,3890.80] vol=2.0x ATR=11.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:10:00 | 3924.18 | 3893.56 | 0.00 | T1 1.5R @ 3924.18 |
| Stop hit — per-position SL triggered | 2024-04-24 10:30:00 | 3906.95 | 3898.29 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-05-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 09:30:00 | 3722.30 | 3746.65 | 0.00 | ORB-short ORB[3735.15,3780.00] vol=1.9x ATR=14.42 |
| Stop hit — per-position SL triggered | 2024-05-03 09:45:00 | 3736.72 | 3729.14 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:10:00 | 3841.95 | 3800.23 | 0.00 | ORB-long ORB[3775.30,3825.00] vol=1.8x ATR=16.91 |
| Stop hit — per-position SL triggered | 2024-05-08 10:15:00 | 3825.04 | 3812.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 10:45:00 | 2765.50 | 2023-05-17 10:50:00 | 2772.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-18 10:05:00 | 2838.60 | 2023-05-18 10:25:00 | 2830.87 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-19 11:05:00 | 2839.60 | 2023-05-19 11:15:00 | 2848.09 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-22 11:10:00 | 2867.40 | 2023-05-22 12:00:00 | 2877.25 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-05-22 11:10:00 | 2867.40 | 2023-05-22 14:20:00 | 2872.70 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2023-05-23 09:55:00 | 2913.55 | 2023-05-23 10:00:00 | 2903.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-30 10:05:00 | 2858.00 | 2023-05-30 10:25:00 | 2876.75 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-05-30 10:05:00 | 2858.00 | 2023-05-30 15:20:00 | 2982.00 | TARGET_HIT | 0.50 | 4.34% |
| BUY | retest1 | 2023-06-06 10:15:00 | 3139.95 | 2023-06-06 10:20:00 | 3128.78 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-06-07 09:35:00 | 3138.30 | 2023-06-07 10:50:00 | 3153.56 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-06-07 09:35:00 | 3138.30 | 2023-06-07 12:40:00 | 3142.75 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-06-09 09:30:00 | 3166.75 | 2023-06-09 09:35:00 | 3159.04 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-15 09:30:00 | 3333.05 | 2023-06-15 09:40:00 | 3346.79 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-06-15 09:30:00 | 3333.05 | 2023-06-15 10:40:00 | 3345.65 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-19 09:30:00 | 3416.05 | 2023-06-19 10:25:00 | 3394.76 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-06-19 09:30:00 | 3416.05 | 2023-06-19 15:20:00 | 3404.80 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2023-06-21 10:35:00 | 3338.30 | 2023-06-21 10:40:00 | 3348.51 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-06-22 10:20:00 | 3339.00 | 2023-06-22 11:00:00 | 3350.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-06-26 09:30:00 | 3351.85 | 2023-06-26 09:50:00 | 3337.88 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-06-27 10:35:00 | 3284.95 | 2023-06-27 10:40:00 | 3292.87 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-03 09:50:00 | 3169.55 | 2023-07-03 10:05:00 | 3181.21 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-04 10:00:00 | 3189.20 | 2023-07-04 10:05:00 | 3174.41 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-07-06 10:30:00 | 3222.95 | 2023-07-06 12:55:00 | 3211.23 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-10 10:45:00 | 3194.10 | 2023-07-10 11:25:00 | 3182.96 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-07-10 10:45:00 | 3194.10 | 2023-07-10 11:40:00 | 3194.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 09:30:00 | 3385.90 | 2023-07-14 10:20:00 | 3374.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-18 11:10:00 | 3395.40 | 2023-07-18 11:15:00 | 3407.28 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-18 11:10:00 | 3395.40 | 2023-07-18 11:35:00 | 3395.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:05:00 | 3527.10 | 2023-07-20 11:10:00 | 3542.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-20 10:05:00 | 3527.10 | 2023-07-20 11:30:00 | 3527.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-25 10:25:00 | 3483.25 | 2023-07-25 10:30:00 | 3494.79 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-08-01 10:45:00 | 3511.30 | 2023-08-01 12:05:00 | 3500.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-04 10:55:00 | 3419.50 | 2023-08-04 11:15:00 | 3406.58 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-08-04 10:55:00 | 3419.50 | 2023-08-04 11:25:00 | 3419.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-10 10:55:00 | 3575.35 | 2023-08-10 11:40:00 | 3587.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-11 09:40:00 | 3631.95 | 2023-08-11 09:50:00 | 3615.03 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-08-17 11:15:00 | 3586.25 | 2023-08-17 12:10:00 | 3577.01 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-17 11:15:00 | 3586.25 | 2023-08-17 15:20:00 | 3526.90 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2023-08-18 09:50:00 | 3520.85 | 2023-08-18 10:20:00 | 3532.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-29 09:45:00 | 3697.80 | 2023-08-29 12:40:00 | 3688.20 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-04 10:50:00 | 3659.00 | 2023-09-04 11:15:00 | 3668.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-07 10:50:00 | 3611.70 | 2023-09-07 11:55:00 | 3620.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-20 09:30:00 | 3596.10 | 2023-09-20 09:35:00 | 3579.65 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-20 09:30:00 | 3596.10 | 2023-09-20 14:15:00 | 3568.65 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2023-09-22 10:00:00 | 3503.00 | 2023-09-22 10:15:00 | 3487.03 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-22 10:00:00 | 3503.00 | 2023-09-22 12:05:00 | 3503.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-25 10:45:00 | 3421.05 | 2023-09-25 11:15:00 | 3405.27 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-25 10:45:00 | 3421.05 | 2023-09-25 12:30:00 | 3421.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-28 09:45:00 | 3493.15 | 2023-09-28 09:55:00 | 3509.86 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-09-28 09:45:00 | 3493.15 | 2023-09-28 10:05:00 | 3493.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-29 11:05:00 | 3533.50 | 2023-09-29 11:20:00 | 3525.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-03 09:35:00 | 3545.40 | 2023-10-03 09:40:00 | 3564.32 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-10-03 09:35:00 | 3545.40 | 2023-10-03 10:40:00 | 3555.80 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2023-10-05 10:35:00 | 3499.70 | 2023-10-05 11:05:00 | 3483.43 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-10-05 10:35:00 | 3499.70 | 2023-10-05 11:10:00 | 3499.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 09:45:00 | 3418.20 | 2023-10-12 09:50:00 | 3405.75 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-10-12 09:45:00 | 3418.20 | 2023-10-12 15:20:00 | 3372.00 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2023-10-16 10:30:00 | 3394.00 | 2023-10-16 10:35:00 | 3404.04 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-10-16 10:30:00 | 3394.00 | 2023-10-16 15:20:00 | 3466.60 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2023-10-18 09:30:00 | 3539.00 | 2023-10-18 10:55:00 | 3560.47 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-10-18 09:30:00 | 3539.00 | 2023-10-18 11:45:00 | 3554.50 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2023-10-19 10:15:00 | 3558.10 | 2023-10-19 10:35:00 | 3577.21 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-10-19 10:15:00 | 3558.10 | 2023-10-19 10:40:00 | 3558.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-23 09:45:00 | 3490.00 | 2023-10-23 10:15:00 | 3469.39 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2023-11-01 10:40:00 | 3550.70 | 2023-11-01 11:10:00 | 3573.35 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-11-01 10:40:00 | 3550.70 | 2023-11-01 15:20:00 | 3673.10 | TARGET_HIT | 0.50 | 3.45% |
| BUY | retest1 | 2023-11-02 10:50:00 | 3708.90 | 2023-11-02 11:05:00 | 3727.96 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-11-02 10:50:00 | 3708.90 | 2023-11-02 12:05:00 | 3727.25 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2023-11-03 10:20:00 | 3750.00 | 2023-11-03 10:30:00 | 3737.23 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-11-07 09:50:00 | 3582.00 | 2023-11-07 10:25:00 | 3558.93 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2023-11-07 09:50:00 | 3582.00 | 2023-11-07 13:30:00 | 3582.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-10 10:35:00 | 3630.95 | 2023-11-10 10:55:00 | 3620.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-16 09:30:00 | 3566.60 | 2023-11-16 10:35:00 | 3578.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-11-29 11:05:00 | 3596.45 | 2023-11-29 11:50:00 | 3586.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-30 09:50:00 | 3586.40 | 2023-11-30 10:20:00 | 3597.97 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-05 10:55:00 | 3684.45 | 2023-12-05 12:35:00 | 3675.88 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-07 10:10:00 | 3712.00 | 2023-12-07 10:45:00 | 3725.81 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-07 10:10:00 | 3712.00 | 2023-12-07 15:20:00 | 3764.00 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2023-12-08 10:25:00 | 3741.35 | 2023-12-08 10:30:00 | 3751.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-12-11 10:20:00 | 3710.20 | 2023-12-11 10:30:00 | 3689.31 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-12-11 10:20:00 | 3710.20 | 2023-12-11 10:40:00 | 3710.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-12 10:55:00 | 3668.00 | 2023-12-12 11:30:00 | 3654.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-12 10:55:00 | 3668.00 | 2023-12-12 15:20:00 | 3610.00 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2023-12-14 11:00:00 | 3643.20 | 2023-12-14 11:25:00 | 3654.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-20 10:05:00 | 3564.90 | 2023-12-20 10:25:00 | 3555.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-22 10:00:00 | 3531.30 | 2023-12-22 10:35:00 | 3545.40 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-22 10:00:00 | 3531.30 | 2023-12-22 12:10:00 | 3547.90 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2023-12-27 10:45:00 | 3603.15 | 2023-12-27 12:55:00 | 3615.79 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-12-27 10:45:00 | 3603.15 | 2023-12-27 13:10:00 | 3603.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 10:35:00 | 3631.55 | 2023-12-28 10:50:00 | 3644.45 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-12-28 10:35:00 | 3631.55 | 2023-12-28 15:20:00 | 3679.95 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2023-12-29 10:40:00 | 3689.25 | 2023-12-29 15:20:00 | 3687.25 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest1 | 2024-01-03 10:50:00 | 3627.45 | 2024-01-03 11:05:00 | 3618.91 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-01-04 10:55:00 | 3601.50 | 2024-01-04 11:15:00 | 3617.02 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-01-04 10:55:00 | 3601.50 | 2024-01-04 12:35:00 | 3601.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 10:40:00 | 3588.90 | 2024-01-05 11:30:00 | 3597.01 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-08 09:55:00 | 3596.45 | 2024-01-08 10:50:00 | 3580.23 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-01-08 09:55:00 | 3596.45 | 2024-01-08 11:00:00 | 3596.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-09 10:40:00 | 3646.95 | 2024-01-09 10:45:00 | 3663.48 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-01-09 10:40:00 | 3646.95 | 2024-01-09 10:50:00 | 3646.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-10 09:30:00 | 3652.40 | 2024-01-10 09:45:00 | 3673.01 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-10 09:30:00 | 3652.40 | 2024-01-10 10:45:00 | 3767.00 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2024-01-11 11:05:00 | 3778.05 | 2024-01-11 11:15:00 | 3766.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-19 10:15:00 | 3703.80 | 2024-01-19 10:40:00 | 3719.76 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-01-19 10:15:00 | 3703.80 | 2024-01-19 10:55:00 | 3703.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:50:00 | 3710.05 | 2024-01-20 11:20:00 | 3718.26 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-23 10:05:00 | 3652.65 | 2024-01-23 10:10:00 | 3666.92 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-01-29 09:30:00 | 4012.25 | 2024-01-29 09:45:00 | 3991.57 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-02-07 09:30:00 | 4431.95 | 2024-02-07 09:35:00 | 4471.93 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2024-02-07 09:30:00 | 4431.95 | 2024-02-07 09:40:00 | 4431.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-14 11:15:00 | 3952.00 | 2024-02-14 11:20:00 | 3938.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-22 10:30:00 | 3722.15 | 2024-02-22 10:35:00 | 3736.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-26 09:50:00 | 3694.80 | 2024-02-26 10:05:00 | 3704.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-27 09:50:00 | 3695.40 | 2024-02-27 10:25:00 | 3706.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-28 10:50:00 | 3731.00 | 2024-02-28 11:00:00 | 3743.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-29 10:15:00 | 3642.85 | 2024-02-29 12:40:00 | 3657.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-03-01 10:50:00 | 3702.65 | 2024-03-01 15:10:00 | 3684.62 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-03-05 09:35:00 | 3731.60 | 2024-03-05 09:40:00 | 3718.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-12 11:10:00 | 3633.70 | 2024-03-12 11:15:00 | 3646.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-26 10:55:00 | 3736.60 | 2024-03-26 11:35:00 | 3748.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-27 09:30:00 | 3874.65 | 2024-03-27 09:35:00 | 3865.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-04 09:50:00 | 3976.15 | 2024-04-04 10:50:00 | 3953.08 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-04-04 09:50:00 | 3976.15 | 2024-04-04 12:55:00 | 3976.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-05 09:55:00 | 4010.00 | 2024-04-05 10:10:00 | 3994.05 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-04-10 10:40:00 | 4062.00 | 2024-04-10 11:05:00 | 4048.24 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-04-10 10:40:00 | 4062.00 | 2024-04-10 12:30:00 | 4060.00 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-04-12 09:35:00 | 3996.00 | 2024-04-12 10:20:00 | 3975.32 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-04-12 09:35:00 | 3996.00 | 2024-04-12 10:30:00 | 3996.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 09:30:00 | 3857.05 | 2024-04-16 09:35:00 | 3842.44 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-04-19 11:05:00 | 3859.90 | 2024-04-19 11:15:00 | 3847.97 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-22 10:00:00 | 3839.70 | 2024-04-22 10:45:00 | 3850.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-04-23 11:10:00 | 3833.10 | 2024-04-23 11:25:00 | 3820.92 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-04-23 11:10:00 | 3833.10 | 2024-04-23 11:35:00 | 3833.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 09:50:00 | 3906.95 | 2024-04-24 10:10:00 | 3924.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-04-24 09:50:00 | 3906.95 | 2024-04-24 10:30:00 | 3906.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 09:30:00 | 3722.30 | 2024-05-03 09:45:00 | 3736.72 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-08 10:10:00 | 3841.95 | 2024-05-08 10:15:00 | 3825.04 | STOP_HIT | 1.00 | -0.44% |
