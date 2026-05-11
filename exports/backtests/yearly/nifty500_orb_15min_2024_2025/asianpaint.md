# Asian Paints Ltd. (ASIANPAINT)

## Backtest Summary

- **Window:** 2024-06-10 09:15:00 → 2026-05-08 15:25:00 (32500 bars)
- **Last close:** 2600.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 15 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 59
- **Target hits / Stop hits / Partials:** 15 / 59 / 26
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 8.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 16 | 30.2% | 4 | 37 | 12 | -0.01% | -0.3% |
| BUY @ 2nd Alert (retest1) | 53 | 16 | 30.2% | 4 | 37 | 12 | -0.01% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 25 | 53.2% | 11 | 22 | 14 | 0.18% | 8.4% |
| SELL @ 2nd Alert (retest1) | 47 | 25 | 53.2% | 11 | 22 | 14 | 0.18% | 8.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 100 | 41 | 41.0% | 15 | 59 | 26 | 0.08% | 8.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 2868.85 | 2880.07 | 0.00 | ORB-short ORB[2871.45,2909.00] vol=2.0x ATR=6.69 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 2875.54 | 2879.25 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:40:00 | 2905.60 | 2911.20 | 0.00 | ORB-short ORB[2910.05,2920.00] vol=1.5x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:55:00 | 2900.19 | 2905.55 | 0.00 | T1 1.5R @ 2900.19 |
| Target hit | 2024-06-19 12:10:00 | 2900.30 | 2897.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 2883.80 | 2900.51 | 0.00 | ORB-short ORB[2901.65,2928.00] vol=1.9x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-06-21 11:00:00 | 2890.05 | 2899.46 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 2922.00 | 2908.75 | 0.00 | ORB-long ORB[2888.00,2913.75] vol=2.0x ATR=6.88 |
| Stop hit — per-position SL triggered | 2024-07-01 09:55:00 | 2915.12 | 2911.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 2944.00 | 2935.24 | 0.00 | ORB-long ORB[2923.00,2940.00] vol=2.1x ATR=5.63 |
| Stop hit — per-position SL triggered | 2024-07-03 09:35:00 | 2938.37 | 2936.29 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 2949.50 | 2936.59 | 0.00 | ORB-long ORB[2923.00,2940.00] vol=1.5x ATR=5.77 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 2943.73 | 2938.63 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:15:00 | 2897.00 | 2900.86 | 0.00 | ORB-short ORB[2898.15,2909.95] vol=3.0x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-07-09 11:35:00 | 2901.29 | 2900.68 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 2915.45 | 2908.02 | 0.00 | ORB-long ORB[2900.00,2914.90] vol=1.5x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-07-10 10:10:00 | 2910.32 | 2908.97 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:50:00 | 2974.80 | 2960.25 | 0.00 | ORB-long ORB[2941.50,2962.00] vol=4.6x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:25:00 | 2983.89 | 2964.50 | 0.00 | T1 1.5R @ 2983.89 |
| Stop hit — per-position SL triggered | 2024-07-16 11:45:00 | 2974.80 | 2967.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:10:00 | 2920.00 | 2923.47 | 0.00 | ORB-short ORB[2922.65,2947.30] vol=2.0x ATR=6.99 |
| Stop hit — per-position SL triggered | 2024-07-23 11:00:00 | 2926.99 | 2922.71 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 11:10:00 | 2925.00 | 2901.84 | 0.00 | ORB-long ORB[2886.20,2908.95] vol=3.1x ATR=7.49 |
| Stop hit — per-position SL triggered | 2024-07-24 11:40:00 | 2917.51 | 2903.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:10:00 | 2943.80 | 2929.37 | 0.00 | ORB-long ORB[2900.70,2923.90] vol=1.7x ATR=5.44 |
| Stop hit — per-position SL triggered | 2024-07-26 11:40:00 | 2938.36 | 2931.09 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:10:00 | 3019.10 | 3009.45 | 0.00 | ORB-long ORB[2970.00,3015.00] vol=2.0x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-07-30 11:20:00 | 3012.23 | 3009.65 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 09:45:00 | 3117.00 | 3084.08 | 0.00 | ORB-long ORB[3055.30,3096.95] vol=1.7x ATR=12.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:00:00 | 3135.34 | 3094.31 | 0.00 | T1 1.5R @ 3135.34 |
| Stop hit — per-position SL triggered | 2024-08-05 10:45:00 | 3117.00 | 3103.45 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:55:00 | 3071.50 | 3083.64 | 0.00 | ORB-short ORB[3081.20,3118.50] vol=1.6x ATR=9.85 |
| Stop hit — per-position SL triggered | 2024-08-07 10:10:00 | 3081.35 | 3080.79 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 3056.00 | 3069.55 | 0.00 | ORB-short ORB[3072.00,3089.70] vol=1.8x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:05:00 | 3046.15 | 3067.46 | 0.00 | T1 1.5R @ 3046.15 |
| Target hit | 2024-08-08 15:20:00 | 2993.00 | 3028.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2024-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:00:00 | 3042.05 | 3057.15 | 0.00 | ORB-short ORB[3050.85,3068.90] vol=1.6x ATR=5.58 |
| Stop hit — per-position SL triggered | 2024-08-19 11:05:00 | 3047.63 | 3056.99 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:55:00 | 3100.75 | 3092.97 | 0.00 | ORB-long ORB[3078.00,3098.00] vol=1.7x ATR=6.66 |
| Stop hit — per-position SL triggered | 2024-08-20 11:00:00 | 3094.09 | 3096.17 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 3120.00 | 3106.31 | 0.00 | ORB-long ORB[3084.05,3103.20] vol=1.7x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:20:00 | 3129.90 | 3112.51 | 0.00 | T1 1.5R @ 3129.90 |
| Stop hit — per-position SL triggered | 2024-08-21 11:10:00 | 3120.00 | 3119.01 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:30:00 | 3187.30 | 3179.07 | 0.00 | ORB-long ORB[3156.05,3184.00] vol=1.8x ATR=8.11 |
| Stop hit — per-position SL triggered | 2024-08-22 15:20:00 | 3181.10 | 3185.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-08-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:30:00 | 3180.00 | 3166.61 | 0.00 | ORB-long ORB[3150.00,3164.00] vol=1.6x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:40:00 | 3189.51 | 3185.47 | 0.00 | T1 1.5R @ 3189.51 |
| Target hit | 2024-08-27 11:25:00 | 3186.05 | 3186.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2024-08-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:30:00 | 3133.85 | 3149.60 | 0.00 | ORB-short ORB[3150.60,3165.50] vol=1.6x ATR=6.04 |
| Stop hit — per-position SL triggered | 2024-08-28 10:50:00 | 3139.89 | 3144.96 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 3156.15 | 3143.21 | 0.00 | ORB-long ORB[3125.25,3140.40] vol=1.8x ATR=10.53 |
| Stop hit — per-position SL triggered | 2024-08-30 09:45:00 | 3145.62 | 3146.20 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 09:35:00 | 3302.95 | 3291.15 | 0.00 | ORB-long ORB[3273.00,3295.15] vol=1.7x ATR=9.48 |
| Stop hit — per-position SL triggered | 2024-09-09 09:50:00 | 3293.47 | 3294.77 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:55:00 | 3304.00 | 3293.11 | 0.00 | ORB-long ORB[3276.05,3296.00] vol=1.6x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:30:00 | 3315.38 | 3299.11 | 0.00 | T1 1.5R @ 3315.38 |
| Stop hit — per-position SL triggered | 2024-09-19 11:25:00 | 3304.00 | 3303.43 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 3313.05 | 3292.52 | 0.00 | ORB-long ORB[3277.50,3310.00] vol=2.2x ATR=9.24 |
| Stop hit — per-position SL triggered | 2024-09-20 10:45:00 | 3303.81 | 3296.46 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 3290.95 | 3286.29 | 0.00 | ORB-long ORB[3274.10,3289.95] vol=3.5x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-09-24 10:40:00 | 3285.19 | 3288.77 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 3214.85 | 3227.24 | 0.00 | ORB-short ORB[3221.05,3252.00] vol=1.6x ATR=6.67 |
| Stop hit — per-position SL triggered | 2024-09-25 09:40:00 | 3221.52 | 3224.12 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 3327.00 | 3304.91 | 0.00 | ORB-long ORB[3277.05,3299.00] vol=1.5x ATR=6.99 |
| Stop hit — per-position SL triggered | 2024-09-27 11:20:00 | 3320.01 | 3307.28 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:40:00 | 3148.35 | 3189.04 | 0.00 | ORB-short ORB[3186.05,3222.35] vol=1.6x ATR=11.76 |
| Stop hit — per-position SL triggered | 2024-10-03 11:40:00 | 3160.11 | 3175.68 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 3049.60 | 3066.17 | 0.00 | ORB-short ORB[3063.90,3098.55] vol=1.9x ATR=8.43 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 3058.03 | 3064.67 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 3057.35 | 3070.25 | 0.00 | ORB-short ORB[3062.00,3095.00] vol=2.3x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:30:00 | 3047.51 | 3065.96 | 0.00 | T1 1.5R @ 3047.51 |
| Stop hit — per-position SL triggered | 2024-10-10 13:00:00 | 3057.35 | 3059.54 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 3052.00 | 3063.35 | 0.00 | ORB-short ORB[3063.05,3085.00] vol=3.2x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:00:00 | 3043.11 | 3061.87 | 0.00 | T1 1.5R @ 3043.11 |
| Stop hit — per-position SL triggered | 2024-10-17 11:30:00 | 3052.00 | 3056.05 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 11:05:00 | 3029.00 | 3038.90 | 0.00 | ORB-short ORB[3029.15,3057.95] vol=1.9x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:25:00 | 3017.81 | 3035.28 | 0.00 | T1 1.5R @ 3017.81 |
| Target hit | 2024-10-22 15:20:00 | 3012.35 | 3017.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:45:00 | 2945.00 | 2979.36 | 0.00 | ORB-short ORB[2990.00,3028.00] vol=1.9x ATR=9.03 |
| Stop hit — per-position SL triggered | 2024-10-29 10:50:00 | 2954.03 | 2977.90 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:35:00 | 2855.35 | 2872.94 | 0.00 | ORB-short ORB[2870.40,2904.95] vol=1.8x ATR=7.81 |
| Stop hit — per-position SL triggered | 2024-11-07 10:50:00 | 2863.16 | 2871.02 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:05:00 | 2810.85 | 2824.01 | 0.00 | ORB-short ORB[2816.05,2847.00] vol=2.5x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:15:00 | 2802.34 | 2822.08 | 0.00 | T1 1.5R @ 2802.34 |
| Target hit | 2024-11-08 15:20:00 | 2772.35 | 2788.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-11-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:50:00 | 2446.75 | 2444.05 | 0.00 | ORB-long ORB[2422.95,2445.50] vol=4.4x ATR=6.20 |
| Stop hit — per-position SL triggered | 2024-11-22 11:20:00 | 2440.55 | 2444.20 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 11:05:00 | 2511.15 | 2491.21 | 0.00 | ORB-long ORB[2457.60,2484.30] vol=2.1x ATR=6.67 |
| Stop hit — per-position SL triggered | 2024-11-26 12:00:00 | 2504.48 | 2497.75 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 2475.85 | 2488.43 | 0.00 | ORB-short ORB[2485.05,2506.00] vol=2.9x ATR=5.12 |
| Stop hit — per-position SL triggered | 2024-11-28 10:50:00 | 2480.97 | 2486.27 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:05:00 | 2478.80 | 2471.90 | 0.00 | ORB-long ORB[2455.05,2474.00] vol=1.6x ATR=4.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:45:00 | 2486.27 | 2474.35 | 0.00 | T1 1.5R @ 2486.27 |
| Stop hit — per-position SL triggered | 2024-11-29 13:45:00 | 2478.80 | 2479.01 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:10:00 | 2470.00 | 2473.87 | 0.00 | ORB-short ORB[2470.25,2478.90] vol=1.8x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 2465.62 | 2471.88 | 0.00 | T1 1.5R @ 2465.62 |
| Target hit | 2024-12-04 15:20:00 | 2458.05 | 2466.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 2448.05 | 2454.34 | 0.00 | ORB-short ORB[2451.80,2465.55] vol=1.7x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:40:00 | 2442.09 | 2452.15 | 0.00 | T1 1.5R @ 2442.09 |
| Target hit | 2024-12-05 12:05:00 | 2435.95 | 2434.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2024-12-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:10:00 | 2440.80 | 2446.99 | 0.00 | ORB-short ORB[2446.95,2468.00] vol=1.9x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 12:25:00 | 2434.87 | 2444.47 | 0.00 | T1 1.5R @ 2434.87 |
| Target hit | 2024-12-06 15:20:00 | 2428.10 | 2436.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-01-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:40:00 | 2301.85 | 2311.15 | 0.00 | ORB-short ORB[2320.00,2340.00] vol=1.9x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 2291.30 | 2306.77 | 0.00 | T1 1.5R @ 2291.30 |
| Target hit | 2025-01-06 15:20:00 | 2268.20 | 2282.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:50:00 | 2307.55 | 2296.06 | 0.00 | ORB-long ORB[2272.00,2292.80] vol=2.7x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:00:00 | 2315.04 | 2298.00 | 0.00 | T1 1.5R @ 2315.04 |
| Stop hit — per-position SL triggered | 2025-01-07 11:10:00 | 2307.55 | 2299.00 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:30:00 | 2300.00 | 2287.67 | 0.00 | ORB-long ORB[2272.70,2298.85] vol=1.7x ATR=5.44 |
| Stop hit — per-position SL triggered | 2025-01-08 10:45:00 | 2294.56 | 2290.07 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 11:15:00 | 2356.00 | 2342.04 | 0.00 | ORB-long ORB[2332.30,2354.40] vol=1.9x ATR=6.33 |
| Stop hit — per-position SL triggered | 2025-01-09 11:25:00 | 2349.67 | 2343.09 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 2281.75 | 2289.61 | 0.00 | ORB-short ORB[2286.10,2301.90] vol=2.3x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:25:00 | 2273.45 | 2288.02 | 0.00 | T1 1.5R @ 2273.45 |
| Target hit | 2025-01-13 15:20:00 | 2250.85 | 2272.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 10:50:00 | 2229.65 | 2230.19 | 0.00 | ORB-short ORB[2241.25,2253.75] vol=2.2x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-01-15 11:05:00 | 2233.96 | 2230.15 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:40:00 | 2222.00 | 2227.35 | 0.00 | ORB-short ORB[2223.55,2239.85] vol=3.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 12:05:00 | 2215.96 | 2224.12 | 0.00 | T1 1.5R @ 2215.96 |
| Target hit | 2025-01-16 14:40:00 | 2219.40 | 2218.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 2276.30 | 2288.93 | 0.00 | ORB-short ORB[2280.25,2302.40] vol=2.4x ATR=6.03 |
| Stop hit — per-position SL triggered | 2025-01-21 10:40:00 | 2282.33 | 2286.65 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:25:00 | 2243.70 | 2251.57 | 0.00 | ORB-short ORB[2249.55,2263.85] vol=1.5x ATR=5.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:45:00 | 2235.26 | 2249.15 | 0.00 | T1 1.5R @ 2235.26 |
| Stop hit — per-position SL triggered | 2025-01-28 10:50:00 | 2243.70 | 2248.05 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-02-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:45:00 | 2319.95 | 2306.15 | 0.00 | ORB-long ORB[2285.30,2318.00] vol=1.5x ATR=7.67 |
| Stop hit — per-position SL triggered | 2025-02-01 09:50:00 | 2312.28 | 2307.30 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:50:00 | 2220.20 | 2229.06 | 0.00 | ORB-short ORB[2224.85,2243.75] vol=2.7x ATR=6.01 |
| Stop hit — per-position SL triggered | 2025-02-14 10:55:00 | 2226.21 | 2228.43 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:10:00 | 2255.00 | 2243.44 | 0.00 | ORB-long ORB[2236.00,2247.10] vol=2.3x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-02-20 11:45:00 | 2250.88 | 2245.96 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 11:15:00 | 2256.30 | 2251.14 | 0.00 | ORB-long ORB[2244.65,2253.95] vol=2.2x ATR=3.82 |
| Stop hit — per-position SL triggered | 2025-02-24 11:55:00 | 2252.48 | 2252.02 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 11:10:00 | 2135.85 | 2160.26 | 0.00 | ORB-short ORB[2173.05,2193.35] vol=1.6x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-03-03 11:20:00 | 2142.21 | 2157.85 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-03-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 10:05:00 | 2140.60 | 2147.15 | 0.00 | ORB-short ORB[2145.20,2159.85] vol=2.7x ATR=6.68 |
| Stop hit — per-position SL triggered | 2025-03-04 10:35:00 | 2147.28 | 2146.01 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:50:00 | 2147.70 | 2140.76 | 0.00 | ORB-long ORB[2124.75,2146.75] vol=2.0x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:05:00 | 2154.85 | 2142.26 | 0.00 | T1 1.5R @ 2154.85 |
| Target hit | 2025-03-05 15:20:00 | 2165.70 | 2158.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:00:00 | 2242.55 | 2215.12 | 0.00 | ORB-long ORB[2183.10,2215.50] vol=2.3x ATR=7.48 |
| Stop hit — per-position SL triggered | 2025-03-06 11:25:00 | 2235.07 | 2219.61 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 11:05:00 | 2286.85 | 2279.27 | 0.00 | ORB-long ORB[2261.85,2282.00] vol=1.9x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-03-10 11:55:00 | 2281.93 | 2281.33 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:45:00 | 2275.80 | 2280.90 | 0.00 | ORB-short ORB[2277.50,2295.95] vol=1.7x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:55:00 | 2269.21 | 2278.43 | 0.00 | T1 1.5R @ 2269.21 |
| Target hit | 2025-03-12 13:25:00 | 2266.60 | 2265.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2025-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:05:00 | 2269.95 | 2241.18 | 0.00 | ORB-long ORB[2222.75,2232.50] vol=2.1x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:15:00 | 2276.90 | 2244.14 | 0.00 | T1 1.5R @ 2276.90 |
| Stop hit — per-position SL triggered | 2025-03-18 12:30:00 | 2269.95 | 2255.79 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 2300.15 | 2291.91 | 0.00 | ORB-long ORB[2276.10,2298.00] vol=2.0x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-03-20 09:45:00 | 2294.65 | 2296.33 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:35:00 | 2304.85 | 2288.14 | 0.00 | ORB-long ORB[2277.85,2291.65] vol=2.4x ATR=5.02 |
| Stop hit — per-position SL triggered | 2025-03-21 11:05:00 | 2299.83 | 2294.42 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 11:15:00 | 2335.40 | 2324.00 | 0.00 | ORB-long ORB[2289.80,2314.40] vol=1.9x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:25:00 | 2342.12 | 2325.60 | 0.00 | T1 1.5R @ 2342.12 |
| Target hit | 2025-04-03 15:20:00 | 2347.15 | 2339.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 11:05:00 | 2398.90 | 2379.39 | 0.00 | ORB-long ORB[2356.00,2384.00] vol=1.8x ATR=8.22 |
| Stop hit — per-position SL triggered | 2025-04-08 11:25:00 | 2390.68 | 2383.79 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 11:10:00 | 2419.00 | 2409.74 | 0.00 | ORB-long ORB[2390.90,2412.00] vol=3.1x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 12:25:00 | 2428.20 | 2413.93 | 0.00 | T1 1.5R @ 2428.20 |
| Stop hit — per-position SL triggered | 2025-04-09 13:05:00 | 2419.00 | 2415.19 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:40:00 | 2435.00 | 2428.23 | 0.00 | ORB-long ORB[2403.70,2424.80] vol=3.4x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:25:00 | 2441.14 | 2430.33 | 0.00 | T1 1.5R @ 2441.14 |
| Target hit | 2025-04-16 15:20:00 | 2459.90 | 2450.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 2450.70 | 2441.45 | 0.00 | ORB-long ORB[2428.00,2447.80] vol=1.6x ATR=5.10 |
| Stop hit — per-position SL triggered | 2025-04-23 09:55:00 | 2445.60 | 2444.91 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:05:00 | 2489.30 | 2469.79 | 0.00 | ORB-long ORB[2435.90,2466.30] vol=1.7x ATR=6.73 |
| Stop hit — per-position SL triggered | 2025-04-24 10:15:00 | 2482.57 | 2471.66 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-04-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:25:00 | 2418.90 | 2441.88 | 0.00 | ORB-short ORB[2453.40,2468.70] vol=2.4x ATR=7.10 |
| Stop hit — per-position SL triggered | 2025-04-25 10:30:00 | 2426.00 | 2439.76 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:00:00 | 2469.00 | 2456.58 | 0.00 | ORB-long ORB[2441.60,2460.20] vol=1.5x ATR=5.74 |
| Stop hit — per-position SL triggered | 2025-04-29 10:10:00 | 2463.26 | 2457.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-12 10:35:00 | 2868.85 | 2024-06-12 10:45:00 | 2875.54 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-06-19 09:40:00 | 2905.60 | 2024-06-19 09:55:00 | 2900.19 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2024-06-19 09:40:00 | 2905.60 | 2024-06-19 12:10:00 | 2900.30 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-06-21 10:45:00 | 2883.80 | 2024-06-21 11:00:00 | 2890.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-01 09:35:00 | 2922.00 | 2024-07-01 09:55:00 | 2915.12 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-03 09:30:00 | 2944.00 | 2024-07-03 09:35:00 | 2938.37 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-04 09:55:00 | 2949.50 | 2024-07-04 10:10:00 | 2943.73 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-09 11:15:00 | 2897.00 | 2024-07-09 11:35:00 | 2901.29 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-07-10 10:05:00 | 2915.45 | 2024-07-10 10:10:00 | 2910.32 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-07-16 10:50:00 | 2974.80 | 2024-07-16 11:25:00 | 2983.89 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-07-16 10:50:00 | 2974.80 | 2024-07-16 11:45:00 | 2974.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 10:10:00 | 2920.00 | 2024-07-23 11:00:00 | 2926.99 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-24 11:10:00 | 2925.00 | 2024-07-24 11:40:00 | 2917.51 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-26 11:10:00 | 2943.80 | 2024-07-26 11:40:00 | 2938.36 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-07-30 11:10:00 | 3019.10 | 2024-07-30 11:20:00 | 3012.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-05 09:45:00 | 3117.00 | 2024-08-05 10:00:00 | 3135.34 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-05 09:45:00 | 3117.00 | 2024-08-05 10:45:00 | 3117.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-07 09:55:00 | 3071.50 | 2024-08-07 10:10:00 | 3081.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-08 10:55:00 | 3056.00 | 2024-08-08 11:05:00 | 3046.15 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-08-08 10:55:00 | 3056.00 | 2024-08-08 15:20:00 | 2993.00 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2024-08-19 11:00:00 | 3042.05 | 2024-08-19 11:05:00 | 3047.63 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-20 09:55:00 | 3100.75 | 2024-08-20 11:00:00 | 3094.09 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-21 10:10:00 | 3120.00 | 2024-08-21 10:20:00 | 3129.90 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-21 10:10:00 | 3120.00 | 2024-08-21 11:10:00 | 3120.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:30:00 | 3187.30 | 2024-08-22 15:20:00 | 3181.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-27 10:30:00 | 3180.00 | 2024-08-27 10:40:00 | 3189.51 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-08-27 10:30:00 | 3180.00 | 2024-08-27 11:25:00 | 3186.05 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-08-28 10:30:00 | 3133.85 | 2024-08-28 10:50:00 | 3139.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-30 09:35:00 | 3156.15 | 2024-08-30 09:45:00 | 3145.62 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-09 09:35:00 | 3302.95 | 2024-09-09 09:50:00 | 3293.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-19 09:55:00 | 3304.00 | 2024-09-19 10:30:00 | 3315.38 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-19 09:55:00 | 3304.00 | 2024-09-19 11:25:00 | 3304.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 10:35:00 | 3313.05 | 2024-09-20 10:45:00 | 3303.81 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-24 09:55:00 | 3290.95 | 2024-09-24 10:40:00 | 3285.19 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-25 09:30:00 | 3214.85 | 2024-09-25 09:40:00 | 3221.52 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-27 11:00:00 | 3327.00 | 2024-09-27 11:20:00 | 3320.01 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-03 10:40:00 | 3148.35 | 2024-10-03 11:40:00 | 3160.11 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-07 10:45:00 | 3049.60 | 2024-10-07 10:55:00 | 3058.03 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-10 11:00:00 | 3057.35 | 2024-10-10 11:30:00 | 3047.51 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-10-10 11:00:00 | 3057.35 | 2024-10-10 13:00:00 | 3057.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:55:00 | 3052.00 | 2024-10-17 11:00:00 | 3043.11 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-17 10:55:00 | 3052.00 | 2024-10-17 11:30:00 | 3052.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 11:05:00 | 3029.00 | 2024-10-22 11:25:00 | 3017.81 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-22 11:05:00 | 3029.00 | 2024-10-22 15:20:00 | 3012.35 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-29 10:45:00 | 2945.00 | 2024-10-29 10:50:00 | 2954.03 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-11-07 10:35:00 | 2855.35 | 2024-11-07 10:50:00 | 2863.16 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-08 11:05:00 | 2810.85 | 2024-11-08 11:15:00 | 2802.34 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-11-08 11:05:00 | 2810.85 | 2024-11-08 15:20:00 | 2772.35 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2024-11-22 10:50:00 | 2446.75 | 2024-11-22 11:20:00 | 2440.55 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-26 11:05:00 | 2511.15 | 2024-11-26 12:00:00 | 2504.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-28 10:35:00 | 2475.85 | 2024-11-28 10:50:00 | 2480.97 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-11-29 11:05:00 | 2478.80 | 2024-11-29 11:45:00 | 2486.27 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-11-29 11:05:00 | 2478.80 | 2024-11-29 13:45:00 | 2478.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 11:10:00 | 2470.00 | 2024-12-04 11:55:00 | 2465.62 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2024-12-04 11:10:00 | 2470.00 | 2024-12-04 15:20:00 | 2458.05 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-05 09:30:00 | 2448.05 | 2024-12-05 09:40:00 | 2442.09 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-12-05 09:30:00 | 2448.05 | 2024-12-05 12:05:00 | 2435.95 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-06 11:10:00 | 2440.80 | 2024-12-06 12:25:00 | 2434.87 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-12-06 11:10:00 | 2440.80 | 2024-12-06 15:20:00 | 2428.10 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-06 10:40:00 | 2301.85 | 2025-01-06 11:10:00 | 2291.30 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-06 10:40:00 | 2301.85 | 2025-01-06 15:20:00 | 2268.20 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-01-07 10:50:00 | 2307.55 | 2025-01-07 11:00:00 | 2315.04 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-01-07 10:50:00 | 2307.55 | 2025-01-07 11:10:00 | 2307.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-08 10:30:00 | 2300.00 | 2025-01-08 10:45:00 | 2294.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-09 11:15:00 | 2356.00 | 2025-01-09 11:25:00 | 2349.67 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-13 11:10:00 | 2281.75 | 2025-01-13 11:25:00 | 2273.45 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-13 11:10:00 | 2281.75 | 2025-01-13 15:20:00 | 2250.85 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2025-01-15 10:50:00 | 2229.65 | 2025-01-15 11:05:00 | 2233.96 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-01-16 10:40:00 | 2222.00 | 2025-01-16 12:05:00 | 2215.96 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-01-16 10:40:00 | 2222.00 | 2025-01-16 14:40:00 | 2219.40 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-01-21 10:20:00 | 2276.30 | 2025-01-21 10:40:00 | 2282.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-28 10:25:00 | 2243.70 | 2025-01-28 10:45:00 | 2235.26 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-28 10:25:00 | 2243.70 | 2025-01-28 10:50:00 | 2243.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 09:45:00 | 2319.95 | 2025-02-01 09:50:00 | 2312.28 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-14 10:50:00 | 2220.20 | 2025-02-14 10:55:00 | 2226.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-20 11:10:00 | 2255.00 | 2025-02-20 11:45:00 | 2250.88 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-02-24 11:15:00 | 2256.30 | 2025-02-24 11:55:00 | 2252.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-03-03 11:10:00 | 2135.85 | 2025-03-03 11:20:00 | 2142.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-04 10:05:00 | 2140.60 | 2025-03-04 10:35:00 | 2147.28 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-05 10:50:00 | 2147.70 | 2025-03-05 11:05:00 | 2154.85 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-03-05 10:50:00 | 2147.70 | 2025-03-05 15:20:00 | 2165.70 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2025-03-06 11:00:00 | 2242.55 | 2025-03-06 11:25:00 | 2235.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-10 11:05:00 | 2286.85 | 2025-03-10 11:55:00 | 2281.93 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-03-12 09:45:00 | 2275.80 | 2025-03-12 09:55:00 | 2269.21 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-03-12 09:45:00 | 2275.80 | 2025-03-12 13:25:00 | 2266.60 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-18 11:05:00 | 2269.95 | 2025-03-18 11:15:00 | 2276.90 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-03-18 11:05:00 | 2269.95 | 2025-03-18 12:30:00 | 2269.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 09:30:00 | 2300.15 | 2025-03-20 09:45:00 | 2294.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-21 10:35:00 | 2304.85 | 2025-03-21 11:05:00 | 2299.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-04-03 11:15:00 | 2335.40 | 2025-04-03 11:25:00 | 2342.12 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-04-03 11:15:00 | 2335.40 | 2025-04-03 15:20:00 | 2347.15 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-08 11:05:00 | 2398.90 | 2025-04-08 11:25:00 | 2390.68 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-09 11:10:00 | 2419.00 | 2025-04-09 12:25:00 | 2428.20 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-09 11:10:00 | 2419.00 | 2025-04-09 13:05:00 | 2419.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 10:40:00 | 2435.00 | 2025-04-16 11:25:00 | 2441.14 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-04-16 10:40:00 | 2435.00 | 2025-04-16 15:20:00 | 2459.90 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-04-23 09:30:00 | 2450.70 | 2025-04-23 09:55:00 | 2445.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-04-24 10:05:00 | 2489.30 | 2025-04-24 10:15:00 | 2482.57 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-25 10:25:00 | 2418.90 | 2025-04-25 10:30:00 | 2426.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-29 10:00:00 | 2469.00 | 2025-04-29 10:10:00 | 2463.26 | STOP_HIT | 1.00 | -0.23% |
