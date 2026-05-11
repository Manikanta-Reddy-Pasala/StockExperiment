# TCS (TCS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2397.20
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 19 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 110 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 60
- **Target hits / Stop hits / Partials:** 19 / 60 / 31
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 13.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 21 | 45.7% | 8 | 25 | 13 | 0.12% | 5.4% |
| BUY @ 2nd Alert (retest1) | 46 | 21 | 45.7% | 8 | 25 | 13 | 0.12% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 29 | 45.3% | 11 | 35 | 18 | 0.13% | 8.2% |
| SELL @ 2nd Alert (retest1) | 64 | 29 | 45.3% | 11 | 35 | 18 | 0.13% | 8.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 110 | 50 | 45.5% | 19 | 60 | 31 | 0.12% | 13.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 10:55:00 | 3554.00 | 3566.11 | 0.00 | ORB-short ORB[3561.70,3593.90] vol=1.8x ATR=6.76 |
| Stop hit — per-position SL triggered | 2025-05-16 12:45:00 | 3560.76 | 3559.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 3529.70 | 3518.50 | 0.00 | ORB-long ORB[3498.30,3524.00] vol=1.6x ATR=6.72 |
| Stop hit — per-position SL triggered | 2025-05-21 11:30:00 | 3522.98 | 3520.63 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 3500.00 | 3513.88 | 0.00 | ORB-short ORB[3504.70,3532.50] vol=2.0x ATR=5.32 |
| Stop hit — per-position SL triggered | 2025-05-29 11:50:00 | 3505.32 | 3512.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:40:00 | 3433.80 | 3441.98 | 0.00 | ORB-short ORB[3439.50,3473.50] vol=1.7x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 12:05:00 | 3423.88 | 3437.85 | 0.00 | T1 1.5R @ 3423.88 |
| Target hit | 2025-06-03 15:20:00 | 3407.20 | 3418.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:50:00 | 3464.10 | 3452.41 | 0.00 | ORB-long ORB[3423.20,3452.80] vol=1.9x ATR=5.77 |
| Stop hit — per-position SL triggered | 2025-06-10 10:55:00 | 3458.33 | 3452.61 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:50:00 | 3450.30 | 3456.20 | 0.00 | ORB-short ORB[3459.00,3474.70] vol=1.8x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-06-12 11:05:00 | 3455.49 | 3456.06 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:45:00 | 3499.40 | 3474.78 | 0.00 | ORB-long ORB[3426.20,3476.60] vol=2.5x ATR=7.48 |
| Stop hit — per-position SL triggered | 2025-06-16 12:05:00 | 3491.92 | 3486.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:45:00 | 3478.20 | 3508.66 | 0.00 | ORB-short ORB[3514.70,3538.00] vol=2.9x ATR=7.05 |
| Stop hit — per-position SL triggered | 2025-06-18 10:55:00 | 3485.25 | 3505.83 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:40:00 | 3414.50 | 3429.83 | 0.00 | ORB-short ORB[3421.00,3451.00] vol=2.5x ATR=7.16 |
| Stop hit — per-position SL triggered | 2025-06-19 10:55:00 | 3421.66 | 3427.83 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 3415.00 | 3410.22 | 0.00 | ORB-long ORB[3400.00,3411.00] vol=1.7x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-06-25 12:05:00 | 3410.34 | 3410.87 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 3478.60 | 3467.11 | 0.00 | ORB-long ORB[3451.60,3473.40] vol=2.3x ATR=5.51 |
| Stop hit — per-position SL triggered | 2025-07-01 09:35:00 | 3473.09 | 3468.11 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:10:00 | 3438.00 | 3464.70 | 0.00 | ORB-short ORB[3441.10,3487.20] vol=1.7x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 13:30:00 | 3427.80 | 3451.45 | 0.00 | T1 1.5R @ 3427.80 |
| Target hit | 2025-07-02 15:20:00 | 3421.10 | 3445.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 11:15:00 | 3421.00 | 3405.87 | 0.00 | ORB-long ORB[3393.90,3414.00] vol=1.6x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-07-04 11:20:00 | 3416.61 | 3406.14 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 3396.00 | 3403.41 | 0.00 | ORB-short ORB[3401.00,3419.00] vol=4.4x ATR=4.36 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 3400.36 | 3402.46 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 11:00:00 | 3391.20 | 3402.25 | 0.00 | ORB-short ORB[3396.60,3414.00] vol=1.7x ATR=4.59 |
| Stop hit — per-position SL triggered | 2025-07-09 11:10:00 | 3395.79 | 3401.76 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:35:00 | 3395.10 | 3376.09 | 0.00 | ORB-long ORB[3356.00,3385.00] vol=1.5x ATR=6.85 |
| Stop hit — per-position SL triggered | 2025-07-10 09:45:00 | 3388.25 | 3378.26 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:15:00 | 3291.60 | 3313.97 | 0.00 | ORB-short ORB[3297.00,3335.00] vol=3.3x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 12:35:00 | 3280.26 | 3309.58 | 0.00 | T1 1.5R @ 3280.26 |
| Target hit | 2025-07-11 15:20:00 | 3264.40 | 3296.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 11:15:00 | 3230.50 | 3241.80 | 0.00 | ORB-short ORB[3235.90,3272.00] vol=1.6x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:35:00 | 3223.52 | 3239.81 | 0.00 | T1 1.5R @ 3223.52 |
| Target hit | 2025-07-14 15:20:00 | 3222.50 | 3224.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-07-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:25:00 | 3207.80 | 3212.57 | 0.00 | ORB-short ORB[3211.00,3228.80] vol=3.0x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:55:00 | 3199.81 | 3210.85 | 0.00 | T1 1.5R @ 3199.81 |
| Target hit | 2025-07-18 15:20:00 | 3188.40 | 3201.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:45:00 | 3171.20 | 3164.00 | 0.00 | ORB-long ORB[3154.60,3168.20] vol=1.9x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:00:00 | 3177.03 | 3164.70 | 0.00 | T1 1.5R @ 3177.03 |
| Stop hit — per-position SL triggered | 2025-07-22 11:25:00 | 3171.20 | 3166.20 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 3015.90 | 3022.11 | 0.00 | ORB-short ORB[3016.00,3032.00] vol=1.9x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-08-07 11:40:00 | 3020.57 | 3021.20 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:05:00 | 3044.00 | 3034.59 | 0.00 | ORB-long ORB[3023.10,3037.90] vol=2.5x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-08-11 11:45:00 | 3039.09 | 3036.74 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:15:00 | 3042.00 | 3030.71 | 0.00 | ORB-long ORB[3010.90,3028.90] vol=1.9x ATR=5.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:35:00 | 3049.63 | 3034.02 | 0.00 | T1 1.5R @ 3049.63 |
| Target hit | 2025-08-20 15:20:00 | 3093.50 | 3077.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 3116.30 | 3101.53 | 0.00 | ORB-long ORB[3077.10,3111.00] vol=2.4x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:40:00 | 3127.66 | 3108.05 | 0.00 | T1 1.5R @ 3127.66 |
| Target hit | 2025-08-25 15:20:00 | 3140.80 | 3133.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:10:00 | 3135.60 | 3123.24 | 0.00 | ORB-long ORB[3103.10,3122.00] vol=2.3x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:20:00 | 3141.34 | 3124.22 | 0.00 | T1 1.5R @ 3141.34 |
| Stop hit — per-position SL triggered | 2025-09-02 13:05:00 | 3135.60 | 3134.93 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:05:00 | 3082.80 | 3092.56 | 0.00 | ORB-short ORB[3090.10,3103.40] vol=2.5x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:10:00 | 3073.86 | 3080.58 | 0.00 | T1 1.5R @ 3073.86 |
| Target hit | 2025-09-05 15:20:00 | 3047.90 | 3052.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:05:00 | 3025.80 | 3034.89 | 0.00 | ORB-short ORB[3042.50,3064.90] vol=11.9x ATR=5.96 |
| Stop hit — per-position SL triggered | 2025-09-08 10:10:00 | 3031.76 | 3034.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 10:25:00 | 3100.00 | 3112.91 | 0.00 | ORB-short ORB[3114.00,3134.00] vol=2.4x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 3104.21 | 3106.54 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 2920.40 | 2933.10 | 0.00 | ORB-short ORB[2931.00,2954.80] vol=1.5x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:30:00 | 2912.47 | 2930.69 | 0.00 | T1 1.5R @ 2912.47 |
| Stop hit — per-position SL triggered | 2025-09-26 12:20:00 | 2920.40 | 2927.86 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 2963.80 | 2977.75 | 0.00 | ORB-short ORB[2966.10,3004.50] vol=1.8x ATR=8.68 |
| Stop hit — per-position SL triggered | 2025-10-07 09:50:00 | 2972.48 | 2974.17 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:55:00 | 3001.00 | 3016.78 | 0.00 | ORB-short ORB[3002.00,3035.50] vol=1.6x ATR=5.90 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 3006.90 | 3015.95 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:00:00 | 3000.40 | 3010.53 | 0.00 | ORB-short ORB[3010.00,3027.20] vol=1.5x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 2992.69 | 3008.62 | 0.00 | T1 1.5R @ 2992.69 |
| Target hit | 2025-10-14 15:20:00 | 2959.60 | 2976.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:05:00 | 2955.70 | 2962.66 | 0.00 | ORB-short ORB[2962.00,2979.70] vol=1.8x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-10-15 11:35:00 | 2961.77 | 2959.57 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 2982.00 | 2968.98 | 0.00 | ORB-long ORB[2956.00,2976.60] vol=2.6x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-10-17 10:30:00 | 2975.53 | 2969.48 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:00:00 | 2995.00 | 2993.73 | 0.00 | ORB-long ORB[2978.50,2993.50] vol=2.0x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:40:00 | 3002.12 | 2995.60 | 0.00 | T1 1.5R @ 3002.12 |
| Target hit | 2025-10-20 15:20:00 | 3014.00 | 3007.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:05:00 | 3060.70 | 3047.10 | 0.00 | ORB-long ORB[3036.10,3055.00] vol=2.6x ATR=7.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:20:00 | 3072.03 | 3050.66 | 0.00 | T1 1.5R @ 3072.03 |
| Target hit | 2025-10-23 15:00:00 | 3075.10 | 3076.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2025-10-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:10:00 | 3087.50 | 3081.23 | 0.00 | ORB-long ORB[3060.20,3074.90] vol=1.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 3083.37 | 3081.30 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 3054.10 | 3073.40 | 0.00 | ORB-short ORB[3065.00,3090.20] vol=1.5x ATR=4.35 |
| Stop hit — per-position SL triggered | 2025-10-28 11:05:00 | 3058.45 | 3072.86 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:35:00 | 3070.60 | 3059.44 | 0.00 | ORB-long ORB[3055.90,3067.50] vol=2.0x ATR=6.25 |
| Stop hit — per-position SL triggered | 2025-10-29 10:45:00 | 3064.35 | 3060.33 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 3039.00 | 3048.02 | 0.00 | ORB-short ORB[3047.60,3063.80] vol=1.6x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-10-30 11:25:00 | 3042.47 | 3045.30 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 3034.20 | 3040.60 | 0.00 | ORB-short ORB[3042.20,3058.00] vol=1.9x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-11-03 11:20:00 | 3039.01 | 3039.85 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 2993.30 | 3005.58 | 0.00 | ORB-short ORB[3011.00,3029.30] vol=2.3x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:05:00 | 2987.69 | 3003.82 | 0.00 | T1 1.5R @ 2987.69 |
| Target hit | 2025-11-04 15:20:00 | 2990.40 | 2991.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 3099.90 | 3086.28 | 0.00 | ORB-long ORB[3057.10,3094.40] vol=1.7x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:50:00 | 3111.69 | 3099.02 | 0.00 | T1 1.5R @ 3111.69 |
| Target hit | 2025-11-12 15:20:00 | 3128.00 | 3114.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-11-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:25:00 | 3088.80 | 3095.73 | 0.00 | ORB-short ORB[3091.90,3123.50] vol=1.7x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-11-17 11:10:00 | 3095.16 | 3094.83 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:20:00 | 3141.00 | 3119.28 | 0.00 | ORB-long ORB[3083.50,3114.50] vol=1.5x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 14:40:00 | 3151.61 | 3133.93 | 0.00 | T1 1.5R @ 3151.61 |
| Target hit | 2025-11-19 15:20:00 | 3147.20 | 3136.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:00:00 | 3131.80 | 3140.66 | 0.00 | ORB-short ORB[3137.60,3150.00] vol=2.1x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 3136.27 | 3139.81 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:20:00 | 3180.10 | 3166.60 | 0.00 | ORB-long ORB[3139.00,3177.70] vol=1.6x ATR=7.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:55:00 | 3191.98 | 3173.43 | 0.00 | T1 1.5R @ 3191.98 |
| Target hit | 2025-12-03 12:35:00 | 3183.70 | 3187.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:55:00 | 3235.00 | 3217.41 | 0.00 | ORB-long ORB[3187.40,3224.00] vol=1.7x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:35:00 | 3246.29 | 3230.36 | 0.00 | T1 1.5R @ 3246.29 |
| Stop hit — per-position SL triggered | 2025-12-04 11:40:00 | 3235.00 | 3230.54 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:35:00 | 3255.90 | 3246.37 | 0.00 | ORB-long ORB[3222.40,3245.00] vol=6.4x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:40:00 | 3265.77 | 3247.72 | 0.00 | T1 1.5R @ 3265.77 |
| Stop hit — per-position SL triggered | 2025-12-05 11:20:00 | 3255.90 | 3253.79 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:35:00 | 3195.00 | 3196.47 | 0.00 | ORB-short ORB[3198.00,3233.70] vol=2.2x ATR=8.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:45:00 | 3182.82 | 3195.49 | 0.00 | T1 1.5R @ 3182.82 |
| Stop hit — per-position SL triggered | 2025-12-09 11:20:00 | 3195.00 | 3193.63 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 3181.60 | 3201.00 | 0.00 | ORB-short ORB[3197.00,3225.00] vol=4.0x ATR=5.59 |
| Stop hit — per-position SL triggered | 2025-12-10 11:05:00 | 3187.19 | 3199.04 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:30:00 | 3189.80 | 3198.32 | 0.00 | ORB-short ORB[3194.80,3208.40] vol=2.0x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:35:00 | 3179.80 | 3190.99 | 0.00 | T1 1.5R @ 3179.80 |
| Target hit | 2025-12-11 10:25:00 | 3183.70 | 3183.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 11:15:00 | 3312.00 | 3309.24 | 0.00 | ORB-long ORB[3296.10,3307.90] vol=2.4x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-12-24 11:40:00 | 3308.53 | 3309.87 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:55:00 | 3289.20 | 3301.98 | 0.00 | ORB-short ORB[3306.00,3320.00] vol=2.0x ATR=3.83 |
| Stop hit — per-position SL triggered | 2025-12-26 11:00:00 | 3293.03 | 3298.02 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:55:00 | 3260.10 | 3267.92 | 0.00 | ORB-short ORB[3267.20,3288.00] vol=1.5x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:20:00 | 3252.03 | 3263.99 | 0.00 | T1 1.5R @ 3252.03 |
| Stop hit — per-position SL triggered | 2025-12-29 13:10:00 | 3260.10 | 3259.86 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 3253.30 | 3248.93 | 0.00 | ORB-long ORB[3239.80,3253.00] vol=2.2x ATR=4.48 |
| Stop hit — per-position SL triggered | 2025-12-30 11:10:00 | 3248.82 | 3249.49 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:45:00 | 3219.90 | 3226.42 | 0.00 | ORB-short ORB[3226.60,3246.00] vol=1.6x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:55:00 | 3212.69 | 3222.74 | 0.00 | T1 1.5R @ 3212.69 |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 3219.90 | 3222.58 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 3204.00 | 3222.89 | 0.00 | ORB-short ORB[3217.00,3247.00] vol=2.6x ATR=6.67 |
| Stop hit — per-position SL triggered | 2026-01-05 09:40:00 | 3210.67 | 3218.21 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:40:00 | 3205.30 | 3223.51 | 0.00 | ORB-short ORB[3238.10,3260.00] vol=2.4x ATR=7.05 |
| Stop hit — per-position SL triggered | 2026-01-14 11:20:00 | 3212.35 | 3218.57 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:55:00 | 3182.60 | 3191.82 | 0.00 | ORB-short ORB[3183.70,3220.00] vol=1.5x ATR=8.11 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3190.71 | 3189.71 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:00:00 | 3140.60 | 3146.64 | 0.00 | ORB-short ORB[3141.70,3160.00] vol=1.9x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:30:00 | 3132.67 | 3143.96 | 0.00 | T1 1.5R @ 3132.67 |
| Stop hit — per-position SL triggered | 2026-01-20 10:50:00 | 3140.60 | 3142.98 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-27 10:50:00 | 3133.00 | 3148.42 | 0.00 | ORB-short ORB[3134.50,3175.00] vol=1.9x ATR=6.87 |
| Stop hit — per-position SL triggered | 2026-01-27 11:20:00 | 3139.87 | 3144.40 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 2969.70 | 2960.27 | 0.00 | ORB-long ORB[2943.50,2965.50] vol=1.7x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 2978.08 | 2964.62 | 0.00 | T1 1.5R @ 2978.08 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 2969.70 | 2965.93 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 2682.50 | 2687.61 | 0.00 | ORB-short ORB[2692.60,2730.00] vol=3.2x ATR=9.65 |
| Stop hit — per-position SL triggered | 2026-02-18 12:35:00 | 2692.15 | 2685.93 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 2745.40 | 2726.81 | 0.00 | ORB-long ORB[2706.10,2733.60] vol=1.9x ATR=8.34 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 2737.06 | 2728.76 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 2480.90 | 2495.86 | 0.00 | ORB-short ORB[2495.10,2521.00] vol=1.5x ATR=5.57 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 2486.47 | 2493.90 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 2441.00 | 2427.08 | 0.00 | ORB-long ORB[2419.00,2438.00] vol=1.5x ATR=6.34 |
| Stop hit — per-position SL triggered | 2026-03-13 10:45:00 | 2434.66 | 2427.57 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 2373.90 | 2398.72 | 0.00 | ORB-short ORB[2397.10,2421.60] vol=1.7x ATR=7.17 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2381.07 | 2390.87 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:25:00 | 2415.00 | 2405.19 | 0.00 | ORB-long ORB[2390.70,2408.20] vol=1.6x ATR=6.45 |
| Stop hit — per-position SL triggered | 2026-03-25 12:55:00 | 2408.55 | 2411.73 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 2392.30 | 2378.58 | 0.00 | ORB-long ORB[2355.00,2389.80] vol=1.8x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-03-30 09:50:00 | 2384.86 | 2382.92 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 2445.50 | 2445.07 | 0.00 | ORB-long ORB[2408.30,2443.30] vol=1.6x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-04-01 12:25:00 | 2437.21 | 2445.52 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 09:40:00 | 2412.60 | 2399.47 | 0.00 | ORB-long ORB[2375.70,2411.00] vol=2.0x ATR=9.01 |
| Stop hit — per-position SL triggered | 2026-04-02 09:45:00 | 2403.59 | 2400.51 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 2516.80 | 2539.15 | 0.00 | ORB-short ORB[2530.10,2565.80] vol=2.5x ATR=12.21 |
| Stop hit — per-position SL triggered | 2026-04-10 13:40:00 | 2529.01 | 2522.44 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 2587.30 | 2580.41 | 0.00 | ORB-long ORB[2560.00,2578.40] vol=1.5x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 2594.96 | 2582.96 | 0.00 | T1 1.5R @ 2594.96 |
| Target hit | 2026-04-21 15:20:00 | 2611.60 | 2596.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 2554.60 | 2567.09 | 0.00 | ORB-short ORB[2558.10,2580.00] vol=1.6x ATR=5.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:40:00 | 2546.98 | 2564.01 | 0.00 | T1 1.5R @ 2546.98 |
| Target hit | 2026-04-22 13:55:00 | 2550.20 | 2543.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 2466.90 | 2481.45 | 0.00 | ORB-short ORB[2472.30,2505.00] vol=1.9x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:10:00 | 2455.51 | 2475.79 | 0.00 | T1 1.5R @ 2455.51 |
| Target hit | 2026-04-24 15:20:00 | 2398.90 | 2426.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 2480.00 | 2465.51 | 0.00 | ORB-long ORB[2447.60,2459.60] vol=1.6x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 2474.60 | 2468.35 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 2432.00 | 2448.80 | 0.00 | ORB-short ORB[2439.10,2469.00] vol=1.6x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:30:00 | 2423.74 | 2444.79 | 0.00 | T1 1.5R @ 2423.74 |
| Stop hit — per-position SL triggered | 2026-05-06 14:25:00 | 2432.00 | 2433.31 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:25:00 | 2384.80 | 2391.32 | 0.00 | ORB-short ORB[2388.70,2407.00] vol=1.5x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:15:00 | 2376.62 | 2389.10 | 0.00 | T1 1.5R @ 2376.62 |
| Stop hit — per-position SL triggered | 2026-05-08 13:05:00 | 2384.80 | 2386.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-16 10:55:00 | 3554.00 | 2025-05-16 12:45:00 | 3560.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-21 11:00:00 | 3529.70 | 2025-05-21 11:30:00 | 3522.98 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-05-29 11:10:00 | 3500.00 | 2025-05-29 11:50:00 | 3505.32 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-06-03 10:40:00 | 3433.80 | 2025-06-03 12:05:00 | 3423.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-03 10:40:00 | 3433.80 | 2025-06-03 15:20:00 | 3407.20 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-06-10 10:50:00 | 3464.10 | 2025-06-10 10:55:00 | 3458.33 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-12 10:50:00 | 3450.30 | 2025-06-12 11:05:00 | 3455.49 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-16 10:45:00 | 3499.40 | 2025-06-16 12:05:00 | 3491.92 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-18 10:45:00 | 3478.20 | 2025-06-18 10:55:00 | 3485.25 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-19 10:40:00 | 3414.50 | 2025-06-19 10:55:00 | 3421.66 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-25 11:05:00 | 3415.00 | 2025-06-25 12:05:00 | 3410.34 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-01 09:30:00 | 3478.60 | 2025-07-01 09:35:00 | 3473.09 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-02 11:10:00 | 3438.00 | 2025-07-02 13:30:00 | 3427.80 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-02 11:10:00 | 3438.00 | 2025-07-02 15:20:00 | 3421.10 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-04 11:15:00 | 3421.00 | 2025-07-04 11:20:00 | 3416.61 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-07-08 10:45:00 | 3396.00 | 2025-07-08 11:15:00 | 3400.36 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-07-09 11:00:00 | 3391.20 | 2025-07-09 11:10:00 | 3395.79 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-10 09:35:00 | 3395.10 | 2025-07-10 09:45:00 | 3388.25 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-11 11:15:00 | 3291.60 | 2025-07-11 12:35:00 | 3280.26 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-11 11:15:00 | 3291.60 | 2025-07-11 15:20:00 | 3264.40 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-07-14 11:15:00 | 3230.50 | 2025-07-14 11:35:00 | 3223.52 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-14 11:15:00 | 3230.50 | 2025-07-14 15:20:00 | 3222.50 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-18 10:25:00 | 3207.80 | 2025-07-18 10:55:00 | 3199.81 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-18 10:25:00 | 3207.80 | 2025-07-18 15:20:00 | 3188.40 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-07-22 10:45:00 | 3171.20 | 2025-07-22 11:00:00 | 3177.03 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2025-07-22 10:45:00 | 3171.20 | 2025-07-22 11:25:00 | 3171.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:05:00 | 3015.90 | 2025-08-07 11:40:00 | 3020.57 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-11 11:05:00 | 3044.00 | 2025-08-11 11:45:00 | 3039.09 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-20 10:15:00 | 3042.00 | 2025-08-20 10:35:00 | 3049.63 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-08-20 10:15:00 | 3042.00 | 2025-08-20 15:20:00 | 3093.50 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2025-08-25 09:30:00 | 3116.30 | 2025-08-25 09:40:00 | 3127.66 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-08-25 09:30:00 | 3116.30 | 2025-08-25 15:20:00 | 3140.80 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-09-02 11:10:00 | 3135.60 | 2025-09-02 11:20:00 | 3141.34 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2025-09-02 11:10:00 | 3135.60 | 2025-09-02 13:05:00 | 3135.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:05:00 | 3082.80 | 2025-09-05 10:10:00 | 3073.86 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-05 10:05:00 | 3082.80 | 2025-09-05 15:20:00 | 3047.90 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2025-09-08 10:05:00 | 3025.80 | 2025-09-08 10:10:00 | 3031.76 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-15 10:25:00 | 3100.00 | 2025-09-15 12:15:00 | 3104.21 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-09-26 11:00:00 | 2920.40 | 2025-09-26 11:30:00 | 2912.47 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-26 11:00:00 | 2920.40 | 2025-09-26 12:20:00 | 2920.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 09:30:00 | 2963.80 | 2025-10-07 09:50:00 | 2972.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-13 10:55:00 | 3001.00 | 2025-10-13 11:15:00 | 3006.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-14 10:00:00 | 3000.40 | 2025-10-14 10:15:00 | 2992.69 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-14 10:00:00 | 3000.40 | 2025-10-14 15:20:00 | 2959.60 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-10-15 10:05:00 | 2955.70 | 2025-10-15 11:35:00 | 2961.77 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-17 10:15:00 | 2982.00 | 2025-10-17 10:30:00 | 2975.53 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-20 11:00:00 | 2995.00 | 2025-10-20 11:40:00 | 3002.12 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-10-20 11:00:00 | 2995.00 | 2025-10-20 15:20:00 | 3014.00 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-10-23 10:05:00 | 3060.70 | 2025-10-23 10:20:00 | 3072.03 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-23 10:05:00 | 3060.70 | 2025-10-23 15:00:00 | 3075.10 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-27 11:10:00 | 3087.50 | 2025-10-27 11:15:00 | 3083.37 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-10-28 11:00:00 | 3054.10 | 2025-10-28 11:05:00 | 3058.45 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-10-29 10:35:00 | 3070.60 | 2025-10-29 10:45:00 | 3064.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-30 11:10:00 | 3039.00 | 2025-10-30 11:25:00 | 3042.47 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-11-03 11:00:00 | 3034.20 | 2025-11-03 11:20:00 | 3039.01 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-11-04 11:00:00 | 2993.30 | 2025-11-04 11:05:00 | 2987.69 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-11-04 11:00:00 | 2993.30 | 2025-11-04 15:20:00 | 2990.40 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-11-12 09:45:00 | 3099.90 | 2025-11-12 10:50:00 | 3111.69 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-12 09:45:00 | 3099.90 | 2025-11-12 15:20:00 | 3128.00 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2025-11-17 10:25:00 | 3088.80 | 2025-11-17 11:10:00 | 3095.16 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-19 10:20:00 | 3141.00 | 2025-11-19 14:40:00 | 3151.61 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-19 10:20:00 | 3141.00 | 2025-11-19 15:20:00 | 3147.20 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2025-12-01 11:00:00 | 3131.80 | 2025-12-01 11:15:00 | 3136.27 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-12-03 10:20:00 | 3180.10 | 2025-12-03 10:55:00 | 3191.98 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-03 10:20:00 | 3180.10 | 2025-12-03 12:35:00 | 3183.70 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-12-04 09:55:00 | 3235.00 | 2025-12-04 11:35:00 | 3246.29 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-04 09:55:00 | 3235.00 | 2025-12-04 11:40:00 | 3235.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:35:00 | 3255.90 | 2025-12-05 10:40:00 | 3265.77 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-05 10:35:00 | 3255.90 | 2025-12-05 11:20:00 | 3255.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-09 10:35:00 | 3195.00 | 2025-12-09 10:45:00 | 3182.82 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-09 10:35:00 | 3195.00 | 2025-12-09 11:20:00 | 3195.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-10 10:55:00 | 3181.60 | 2025-12-10 11:05:00 | 3187.19 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-11 09:30:00 | 3189.80 | 2025-12-11 09:35:00 | 3179.80 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-11 09:30:00 | 3189.80 | 2025-12-11 10:25:00 | 3183.70 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-12-24 11:15:00 | 3312.00 | 2025-12-24 11:40:00 | 3308.53 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2025-12-26 10:55:00 | 3289.20 | 2025-12-26 11:00:00 | 3293.03 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-12-29 10:55:00 | 3260.10 | 2025-12-29 11:20:00 | 3252.03 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-29 10:55:00 | 3260.10 | 2025-12-29 13:10:00 | 3260.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:55:00 | 3253.30 | 2025-12-30 11:10:00 | 3248.82 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-31 10:45:00 | 3219.90 | 2025-12-31 11:55:00 | 3212.69 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-31 10:45:00 | 3219.90 | 2025-12-31 12:15:00 | 3219.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-05 09:30:00 | 3204.00 | 2026-01-05 09:40:00 | 3210.67 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-14 10:40:00 | 3205.30 | 2026-01-14 11:20:00 | 3212.35 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-19 09:55:00 | 3182.60 | 2026-01-19 10:15:00 | 3190.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-20 10:00:00 | 3140.60 | 2026-01-20 10:30:00 | 3132.67 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-01-20 10:00:00 | 3140.60 | 2026-01-20 10:50:00 | 3140.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-27 10:50:00 | 3133.00 | 2026-01-27 11:20:00 | 3139.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-10 09:30:00 | 2969.70 | 2026-02-10 09:40:00 | 2978.08 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-10 09:30:00 | 2969.70 | 2026-02-10 09:55:00 | 2969.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:35:00 | 2682.50 | 2026-02-18 12:35:00 | 2692.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-19 10:05:00 | 2745.40 | 2026-02-19 10:15:00 | 2737.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-11 10:35:00 | 2480.90 | 2026-03-11 11:00:00 | 2486.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-03-13 10:40:00 | 2441.00 | 2026-03-13 10:45:00 | 2434.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-16 10:20:00 | 2373.90 | 2026-03-16 11:15:00 | 2381.07 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-25 10:25:00 | 2415.00 | 2026-03-25 12:55:00 | 2408.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-30 09:30:00 | 2392.30 | 2026-03-30 09:50:00 | 2384.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-01 11:00:00 | 2445.50 | 2026-04-01 12:25:00 | 2437.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-02 09:40:00 | 2412.60 | 2026-04-02 09:45:00 | 2403.59 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-10 09:35:00 | 2516.80 | 2026-04-10 13:40:00 | 2529.01 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-21 10:55:00 | 2587.30 | 2026-04-21 11:35:00 | 2594.96 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-21 10:55:00 | 2587.30 | 2026-04-21 15:20:00 | 2611.60 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2026-04-22 11:10:00 | 2554.60 | 2026-04-22 11:40:00 | 2546.98 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-22 11:10:00 | 2554.60 | 2026-04-22 13:55:00 | 2550.20 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-04-24 09:45:00 | 2466.90 | 2026-04-24 10:10:00 | 2455.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 09:45:00 | 2466.90 | 2026-04-24 15:20:00 | 2398.90 | TARGET_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2026-04-29 10:55:00 | 2480.00 | 2026-04-29 11:15:00 | 2474.60 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2432.00 | 2026-05-06 11:30:00 | 2423.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2432.00 | 2026-05-06 14:25:00 | 2432.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:25:00 | 2384.80 | 2026-05-08 11:15:00 | 2376.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-08 10:25:00 | 2384.80 | 2026-05-08 13:05:00 | 2384.80 | STOP_HIT | 0.50 | 0.00% |
