# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 2502.00
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
| ENTRY1 | 101 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 20 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 81
- **Target hits / Stop hits / Partials:** 20 / 81 / 41
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 35.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 33 | 42.9% | 11 | 44 | 22 | 0.34% | 26.0% |
| BUY @ 2nd Alert (retest1) | 77 | 33 | 42.9% | 11 | 44 | 22 | 0.34% | 26.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 65 | 28 | 43.1% | 9 | 37 | 19 | 0.15% | 9.9% |
| SELL @ 2nd Alert (retest1) | 65 | 28 | 43.1% | 9 | 37 | 19 | 0.15% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 142 | 61 | 43.0% | 20 | 81 | 41 | 0.25% | 35.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 10:50:00 | 2736.95 | 2723.35 | 0.00 | ORB-long ORB[2705.35,2730.12] vol=2.9x ATR=15.36 |
| Stop hit — per-position SL triggered | 2024-05-13 11:10:00 | 2721.59 | 2725.04 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:15:00 | 2826.68 | 2807.76 | 0.00 | ORB-long ORB[2798.90,2821.20] vol=2.7x ATR=10.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:30:00 | 2843.11 | 2819.62 | 0.00 | T1 1.5R @ 2843.11 |
| Target hit | 2024-05-14 15:20:00 | 2944.86 | 2904.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:50:00 | 2919.60 | 2936.80 | 0.00 | ORB-short ORB[2958.87,2983.06] vol=2.2x ATR=12.61 |
| Stop hit — per-position SL triggered | 2024-05-16 11:00:00 | 2932.21 | 2936.29 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:20:00 | 2984.51 | 2954.36 | 0.00 | ORB-long ORB[2923.58,2953.34] vol=2.0x ATR=12.69 |
| Stop hit — per-position SL triggered | 2024-05-17 10:25:00 | 2971.82 | 2955.60 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:10:00 | 2977.29 | 2960.17 | 0.00 | ORB-long ORB[2941.90,2976.32] vol=2.7x ATR=9.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 11:15:00 | 2991.09 | 2970.83 | 0.00 | T1 1.5R @ 2991.09 |
| Target hit | 2024-05-21 14:50:00 | 3027.70 | 3029.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:00:00 | 3071.38 | 3061.50 | 0.00 | ORB-long ORB[3039.87,3060.37] vol=2.0x ATR=9.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:05:00 | 3085.05 | 3064.57 | 0.00 | T1 1.5R @ 3085.05 |
| Target hit | 2024-05-23 15:20:00 | 3299.79 | 3201.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:15:00 | 3216.51 | 3195.28 | 0.00 | ORB-long ORB[3191.25,3209.00] vol=1.6x ATR=12.60 |
| Stop hit — per-position SL triggered | 2024-05-28 10:25:00 | 3203.91 | 3200.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 3151.26 | 3137.21 | 0.00 | ORB-long ORB[3113.99,3150.83] vol=1.5x ATR=12.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 09:55:00 | 3169.74 | 3146.59 | 0.00 | T1 1.5R @ 3169.74 |
| Stop hit — per-position SL triggered | 2024-05-29 10:20:00 | 3151.26 | 3152.36 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:55:00 | 3101.82 | 3087.02 | 0.00 | ORB-long ORB[3055.43,3100.41] vol=2.3x ATR=10.23 |
| Stop hit — per-position SL triggered | 2024-06-07 11:20:00 | 3091.59 | 3089.23 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:45:00 | 3127.66 | 3133.42 | 0.00 | ORB-short ORB[3128.58,3154.70] vol=1.5x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 3116.72 | 3129.92 | 0.00 | T1 1.5R @ 3116.72 |
| Target hit | 2024-06-13 13:40:00 | 3122.71 | 3122.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:15:00 | 3153.78 | 3143.89 | 0.00 | ORB-long ORB[3122.81,3152.28] vol=1.9x ATR=9.26 |
| Stop hit — per-position SL triggered | 2024-06-14 10:30:00 | 3144.52 | 3145.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 3203.18 | 3173.59 | 0.00 | ORB-long ORB[3141.13,3177.97] vol=2.2x ATR=11.56 |
| Stop hit — per-position SL triggered | 2024-06-20 10:40:00 | 3191.62 | 3182.84 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 3132.79 | 3145.81 | 0.00 | ORB-short ORB[3135.46,3177.10] vol=1.9x ATR=9.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:10:00 | 3118.74 | 3138.47 | 0.00 | T1 1.5R @ 3118.74 |
| Stop hit — per-position SL triggered | 2024-06-21 10:25:00 | 3132.79 | 3136.96 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:40:00 | 3069.83 | 3090.88 | 0.00 | ORB-short ORB[3088.83,3110.54] vol=2.0x ATR=6.96 |
| Stop hit — per-position SL triggered | 2024-06-25 10:45:00 | 3076.79 | 3090.14 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 3094.16 | 3084.17 | 0.00 | ORB-long ORB[3071.67,3087.42] vol=3.8x ATR=6.17 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 3087.99 | 3086.99 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 3104.24 | 3077.04 | 0.00 | ORB-long ORB[3063.57,3084.85] vol=4.5x ATR=8.73 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 3095.51 | 3082.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:50:00 | 3053.88 | 3069.05 | 0.00 | ORB-short ORB[3069.29,3096.97] vol=2.0x ATR=6.82 |
| Stop hit — per-position SL triggered | 2024-07-04 11:55:00 | 3060.70 | 3065.60 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:00:00 | 3027.70 | 3037.35 | 0.00 | ORB-short ORB[3036.43,3061.83] vol=2.3x ATR=6.45 |
| Stop hit — per-position SL triggered | 2024-07-08 10:05:00 | 3034.15 | 3036.76 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 2978.26 | 3009.49 | 0.00 | ORB-short ORB[3019.95,3031.82] vol=1.6x ATR=8.65 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 2986.91 | 3006.95 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 3026.30 | 3016.96 | 0.00 | ORB-long ORB[3005.40,3024.74] vol=1.7x ATR=8.23 |
| Stop hit — per-position SL triggered | 2024-07-11 09:40:00 | 3018.07 | 3018.19 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 2977.05 | 2996.34 | 0.00 | ORB-short ORB[2986.06,3016.07] vol=1.6x ATR=8.29 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 2985.34 | 2990.59 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:50:00 | 2952.95 | 2976.18 | 0.00 | ORB-short ORB[2977.58,2997.65] vol=2.3x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:55:00 | 2939.02 | 2972.90 | 0.00 | T1 1.5R @ 2939.02 |
| Stop hit — per-position SL triggered | 2024-07-19 11:05:00 | 2952.95 | 2969.57 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:00:00 | 2934.05 | 2920.80 | 0.00 | ORB-long ORB[2908.46,2932.50] vol=3.1x ATR=8.35 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 2925.70 | 2923.14 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:05:00 | 2886.88 | 2870.65 | 0.00 | ORB-long ORB[2855.23,2873.55] vol=1.8x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:10:00 | 2899.40 | 2876.78 | 0.00 | T1 1.5R @ 2899.40 |
| Target hit | 2024-07-25 12:00:00 | 2894.45 | 2895.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 2918.63 | 2908.95 | 0.00 | ORB-long ORB[2896.92,2914.22] vol=1.6x ATR=8.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:35:00 | 2931.69 | 2916.02 | 0.00 | T1 1.5R @ 2931.69 |
| Target hit | 2024-07-26 15:20:00 | 2981.17 | 2972.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 3072.69 | 3060.17 | 0.00 | ORB-long ORB[3039.34,3068.95] vol=1.8x ATR=9.28 |
| Stop hit — per-position SL triggered | 2024-07-31 10:10:00 | 3063.41 | 3060.49 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 3090.72 | 3081.40 | 0.00 | ORB-long ORB[3065.51,3088.25] vol=3.4x ATR=9.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:00:00 | 3104.72 | 3087.58 | 0.00 | T1 1.5R @ 3104.72 |
| Stop hit — per-position SL triggered | 2024-08-01 10:30:00 | 3090.72 | 3089.98 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:20:00 | 3054.94 | 3036.09 | 0.00 | ORB-long ORB[3017.04,3051.89] vol=2.0x ATR=12.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:55:00 | 3074.09 | 3046.64 | 0.00 | T1 1.5R @ 3074.09 |
| Stop hit — per-position SL triggered | 2024-08-07 12:40:00 | 3054.94 | 3049.72 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:35:00 | 3111.71 | 3090.36 | 0.00 | ORB-long ORB[3081.90,3106.42] vol=1.8x ATR=11.68 |
| Stop hit — per-position SL triggered | 2024-08-08 11:25:00 | 3100.03 | 3096.45 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:05:00 | 2948.98 | 2969.65 | 0.00 | ORB-short ORB[2963.62,2991.54] vol=1.5x ATR=9.67 |
| Stop hit — per-position SL triggered | 2024-08-14 12:25:00 | 2958.65 | 2964.72 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:50:00 | 2997.26 | 3012.15 | 0.00 | ORB-short ORB[3015.34,3030.61] vol=1.5x ATR=6.65 |
| Stop hit — per-position SL triggered | 2024-08-20 10:55:00 | 3003.91 | 3011.86 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:05:00 | 3019.95 | 2993.42 | 0.00 | ORB-long ORB[2974.77,2998.67] vol=2.2x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:10:00 | 3031.60 | 2999.11 | 0.00 | T1 1.5R @ 3031.60 |
| Stop hit — per-position SL triggered | 2024-08-21 11:30:00 | 3019.95 | 3016.19 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:30:00 | 3041.61 | 3033.22 | 0.00 | ORB-long ORB[3020.92,3035.22] vol=2.6x ATR=5.80 |
| Stop hit — per-position SL triggered | 2024-08-22 10:55:00 | 3035.81 | 3034.47 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:00:00 | 2999.25 | 3008.79 | 0.00 | ORB-short ORB[3006.37,3027.70] vol=1.6x ATR=5.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 12:20:00 | 2990.37 | 3004.06 | 0.00 | T1 1.5R @ 2990.37 |
| Target hit | 2024-08-23 14:50:00 | 2995.66 | 2995.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-08-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:05:00 | 2967.69 | 2977.46 | 0.00 | ORB-short ORB[2973.41,2995.71] vol=2.1x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-08-26 11:35:00 | 2973.90 | 2975.54 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 2987.81 | 2982.45 | 0.00 | ORB-long ORB[2970.84,2984.07] vol=2.7x ATR=5.58 |
| Stop hit — per-position SL triggered | 2024-08-27 09:50:00 | 2982.23 | 2983.49 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 2951.11 | 2962.62 | 0.00 | ORB-short ORB[2957.46,2984.07] vol=2.3x ATR=5.82 |
| Stop hit — per-position SL triggered | 2024-08-28 09:45:00 | 2956.93 | 2960.02 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 2908.89 | 2925.74 | 0.00 | ORB-short ORB[2928.81,2949.12] vol=1.5x ATR=7.07 |
| Stop hit — per-position SL triggered | 2024-08-29 11:20:00 | 2915.96 | 2924.05 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 2904.58 | 2916.69 | 0.00 | ORB-short ORB[2914.76,2932.69] vol=1.8x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:55:00 | 2894.88 | 2912.38 | 0.00 | T1 1.5R @ 2894.88 |
| Target hit | 2024-09-06 10:25:00 | 2904.43 | 2902.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — SELL (started 2024-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 11:00:00 | 2847.91 | 2851.32 | 0.00 | ORB-short ORB[2854.16,2867.30] vol=1.5x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:20:00 | 2837.04 | 2849.83 | 0.00 | T1 1.5R @ 2837.04 |
| Stop hit — per-position SL triggered | 2024-09-12 12:00:00 | 2847.91 | 2847.55 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 2925.91 | 2910.81 | 0.00 | ORB-long ORB[2881.31,2922.37] vol=2.4x ATR=6.77 |
| Stop hit — per-position SL triggered | 2024-09-16 09:55:00 | 2919.14 | 2917.08 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 2875.06 | 2884.85 | 0.00 | ORB-short ORB[2883.49,2904.48] vol=1.6x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:10:00 | 2866.68 | 2883.21 | 0.00 | T1 1.5R @ 2866.68 |
| Stop hit — per-position SL triggered | 2024-09-17 11:25:00 | 2875.06 | 2882.60 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-09-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:45:00 | 2978.55 | 2965.60 | 0.00 | ORB-long ORB[2947.48,2967.06] vol=2.6x ATR=7.67 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 2970.88 | 2972.67 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 11:15:00 | 3032.84 | 3014.15 | 0.00 | ORB-long ORB[2990.86,3019.95] vol=2.7x ATR=7.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:25:00 | 3043.42 | 3017.39 | 0.00 | T1 1.5R @ 3043.42 |
| Stop hit — per-position SL triggered | 2024-09-26 11:40:00 | 3032.84 | 3021.09 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:45:00 | 3091.45 | 3054.14 | 0.00 | ORB-long ORB[3025.18,3060.66] vol=1.7x ATR=10.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:05:00 | 3107.36 | 3068.02 | 0.00 | T1 1.5R @ 3107.36 |
| Stop hit — per-position SL triggered | 2024-09-30 11:10:00 | 3091.45 | 3069.74 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 3021.93 | 3030.17 | 0.00 | ORB-short ORB[3023.87,3048.69] vol=2.7x ATR=8.34 |
| Stop hit — per-position SL triggered | 2024-10-01 11:45:00 | 3030.27 | 3029.52 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 11:15:00 | 3037.98 | 3013.42 | 0.00 | ORB-long ORB[2986.01,3027.70] vol=1.7x ATR=9.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 12:25:00 | 3052.70 | 3021.07 | 0.00 | T1 1.5R @ 3052.70 |
| Stop hit — per-position SL triggered | 2024-10-04 12:35:00 | 3037.98 | 3022.27 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 2977.29 | 3008.05 | 0.00 | ORB-short ORB[3005.40,3033.03] vol=1.5x ATR=10.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:10:00 | 2961.23 | 2992.05 | 0.00 | T1 1.5R @ 2961.23 |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 2977.29 | 2991.27 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 2964.83 | 2946.88 | 0.00 | ORB-long ORB[2903.46,2927.51] vol=2.4x ATR=11.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:25:00 | 2981.79 | 2955.11 | 0.00 | T1 1.5R @ 2981.79 |
| Target hit | 2024-10-08 15:20:00 | 3069.78 | 3015.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-10-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:10:00 | 3039.34 | 3066.41 | 0.00 | ORB-short ORB[3055.19,3092.66] vol=2.3x ATR=11.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 14:15:00 | 3022.79 | 3056.84 | 0.00 | T1 1.5R @ 3022.79 |
| Stop hit — per-position SL triggered | 2024-10-11 14:35:00 | 3039.34 | 3056.15 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:45:00 | 3028.57 | 3042.71 | 0.00 | ORB-short ORB[3035.12,3053.73] vol=1.6x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-10-14 11:40:00 | 3035.11 | 3040.73 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 2993.09 | 3006.24 | 0.00 | ORB-short ORB[2995.71,3020.33] vol=2.9x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:45:00 | 2983.06 | 3003.20 | 0.00 | T1 1.5R @ 2983.06 |
| Stop hit — per-position SL triggered | 2024-10-16 13:20:00 | 2993.09 | 2996.66 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 2966.82 | 2981.92 | 0.00 | ORB-short ORB[2978.26,3005.40] vol=1.6x ATR=8.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:45:00 | 2954.73 | 2973.25 | 0.00 | T1 1.5R @ 2954.73 |
| Target hit | 2024-10-17 10:35:00 | 2960.03 | 2955.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 2886.93 | 2899.68 | 0.00 | ORB-short ORB[2892.60,2922.95] vol=2.4x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:45:00 | 2875.14 | 2890.80 | 0.00 | T1 1.5R @ 2875.14 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 2886.93 | 2887.11 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 11:00:00 | 2783.34 | 2825.21 | 0.00 | ORB-short ORB[2826.05,2851.01] vol=1.5x ATR=10.63 |
| Stop hit — per-position SL triggered | 2024-11-04 11:20:00 | 2793.97 | 2819.24 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 2873.55 | 2847.80 | 0.00 | ORB-long ORB[2828.96,2859.79] vol=1.5x ATR=9.68 |
| Stop hit — per-position SL triggered | 2024-11-06 09:55:00 | 2863.87 | 2856.19 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:15:00 | 2847.04 | 2824.90 | 0.00 | ORB-long ORB[2804.77,2838.65] vol=1.9x ATR=8.14 |
| Stop hit — per-position SL triggered | 2024-11-11 11:55:00 | 2838.90 | 2831.22 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:40:00 | 2762.11 | 2780.25 | 0.00 | ORB-short ORB[2763.03,2800.84] vol=2.4x ATR=11.11 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 2773.22 | 2775.96 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:40:00 | 2727.69 | 2741.62 | 0.00 | ORB-short ORB[2737.83,2757.99] vol=1.6x ATR=10.44 |
| Stop hit — per-position SL triggered | 2024-11-14 09:55:00 | 2738.13 | 2740.11 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 2773.70 | 2760.07 | 0.00 | ORB-long ORB[2737.29,2767.49] vol=1.6x ATR=9.40 |
| Stop hit — per-position SL triggered | 2024-11-19 09:50:00 | 2764.30 | 2764.81 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 11:00:00 | 2419.20 | 2405.33 | 0.00 | ORB-long ORB[2386.92,2416.93] vol=2.7x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:05:00 | 2432.29 | 2409.13 | 0.00 | T1 1.5R @ 2432.29 |
| Target hit | 2024-12-03 13:10:00 | 2429.19 | 2432.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — SELL (started 2024-12-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:00:00 | 2405.29 | 2414.68 | 0.00 | ORB-short ORB[2408.25,2424.68] vol=1.8x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-12-10 10:50:00 | 2410.88 | 2411.28 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 2379.31 | 2392.05 | 0.00 | ORB-short ORB[2385.42,2403.30] vol=2.0x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-12-11 11:10:00 | 2384.27 | 2391.83 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 2411.50 | 2384.30 | 0.00 | ORB-long ORB[2376.74,2389.25] vol=3.6x ATR=7.05 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 2404.45 | 2387.90 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:40:00 | 2444.31 | 2435.69 | 0.00 | ORB-long ORB[2417.94,2442.86] vol=1.6x ATR=7.50 |
| Stop hit — per-position SL triggered | 2024-12-13 09:45:00 | 2436.81 | 2436.14 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 2438.64 | 2446.12 | 0.00 | ORB-short ORB[2443.59,2460.07] vol=2.0x ATR=5.06 |
| Stop hit — per-position SL triggered | 2024-12-16 11:25:00 | 2443.70 | 2445.94 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 2443.01 | 2435.33 | 0.00 | ORB-long ORB[2422.79,2438.25] vol=1.6x ATR=5.87 |
| Stop hit — per-position SL triggered | 2024-12-17 10:50:00 | 2437.14 | 2435.51 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 2376.50 | 2365.47 | 0.00 | ORB-long ORB[2346.15,2368.55] vol=1.5x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 2370.93 | 2366.17 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-12-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:30:00 | 2324.78 | 2294.30 | 0.00 | ORB-long ORB[2272.23,2287.65] vol=1.9x ATR=7.87 |
| Stop hit — per-position SL triggered | 2024-12-24 10:55:00 | 2316.91 | 2302.58 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:40:00 | 2444.27 | 2466.38 | 0.00 | ORB-short ORB[2458.66,2490.46] vol=1.9x ATR=8.36 |
| Stop hit — per-position SL triggered | 2025-01-06 09:45:00 | 2452.63 | 2464.22 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:30:00 | 2423.13 | 2431.86 | 0.00 | ORB-short ORB[2427.40,2440.19] vol=1.5x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:45:00 | 2413.68 | 2428.77 | 0.00 | T1 1.5R @ 2413.68 |
| Target hit | 2025-01-09 15:20:00 | 2409.85 | 2411.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2025-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:55:00 | 2253.76 | 2264.48 | 0.00 | ORB-short ORB[2262.78,2292.83] vol=1.8x ATR=10.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:05:00 | 2237.83 | 2262.10 | 0.00 | T1 1.5R @ 2237.83 |
| Target hit | 2025-01-13 15:20:00 | 2157.83 | 2213.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:40:00 | 2335.78 | 2326.27 | 0.00 | ORB-long ORB[2314.40,2332.39] vol=2.0x ATR=9.32 |
| Stop hit — per-position SL triggered | 2025-01-15 09:55:00 | 2326.46 | 2329.13 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:55:00 | 2343.78 | 2330.76 | 0.00 | ORB-long ORB[2317.07,2341.79] vol=3.1x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 12:40:00 | 2353.49 | 2334.92 | 0.00 | T1 1.5R @ 2353.49 |
| Target hit | 2025-01-20 15:20:00 | 2361.28 | 2350.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2025-01-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:50:00 | 2300.25 | 2303.00 | 0.00 | ORB-short ORB[2305.63,2332.58] vol=1.6x ATR=8.50 |
| Stop hit — per-position SL triggered | 2025-01-24 11:25:00 | 2308.75 | 2302.05 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-01-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:00:00 | 2278.82 | 2258.72 | 0.00 | ORB-long ORB[2244.36,2260.84] vol=1.7x ATR=6.89 |
| Stop hit — per-position SL triggered | 2025-01-30 10:05:00 | 2271.93 | 2260.04 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-02-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:55:00 | 2270.97 | 2252.21 | 0.00 | ORB-long ORB[2238.59,2252.11] vol=2.1x ATR=7.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:15:00 | 2282.57 | 2256.97 | 0.00 | T1 1.5R @ 2282.57 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 2270.97 | 2271.58 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:40:00 | 2229.14 | 2242.13 | 0.00 | ORB-short ORB[2238.25,2266.56] vol=2.3x ATR=10.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:55:00 | 2213.86 | 2238.96 | 0.00 | T1 1.5R @ 2213.86 |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 2229.14 | 2234.30 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-02-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 09:55:00 | 2278.00 | 2287.59 | 0.00 | ORB-short ORB[2282.17,2300.35] vol=3.1x ATR=8.21 |
| Stop hit — per-position SL triggered | 2025-02-13 10:00:00 | 2286.21 | 2287.29 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 2104.27 | 2121.67 | 0.00 | ORB-short ORB[2113.62,2132.48] vol=2.4x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:50:00 | 2093.47 | 2115.52 | 0.00 | T1 1.5R @ 2093.47 |
| Target hit | 2025-02-21 12:30:00 | 2098.45 | 2098.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 81 — BUY (started 2025-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:40:00 | 2071.79 | 2063.63 | 0.00 | ORB-long ORB[2037.86,2065.00] vol=1.6x ATR=7.18 |
| Stop hit — per-position SL triggered | 2025-02-25 09:55:00 | 2064.61 | 2065.00 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:50:00 | 2061.95 | 2066.95 | 0.00 | ORB-short ORB[2063.26,2083.52] vol=2.2x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 10:00:00 | 2053.87 | 2065.28 | 0.00 | T1 1.5R @ 2053.87 |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 2061.95 | 2064.77 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-02-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:30:00 | 2011.10 | 2025.14 | 0.00 | ORB-short ORB[2016.63,2045.61] vol=1.8x ATR=8.83 |
| Stop hit — per-position SL triggered | 2025-02-28 09:45:00 | 2019.93 | 2021.47 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 2127.63 | 2108.75 | 0.00 | ORB-long ORB[2082.50,2108.63] vol=1.7x ATR=7.87 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 2119.76 | 2117.44 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 2213.72 | 2202.53 | 0.00 | ORB-long ORB[2190.79,2208.92] vol=1.7x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-03-18 09:40:00 | 2207.25 | 2203.05 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:30:00 | 2293.66 | 2281.32 | 0.00 | ORB-long ORB[2266.66,2284.11] vol=1.6x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:00:00 | 2302.84 | 2285.68 | 0.00 | T1 1.5R @ 2302.84 |
| Stop hit — per-position SL triggered | 2025-03-21 13:35:00 | 2293.66 | 2295.18 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-03-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:10:00 | 2282.60 | 2258.17 | 0.00 | ORB-long ORB[2237.62,2264.28] vol=1.8x ATR=7.61 |
| Stop hit — per-position SL triggered | 2025-03-26 10:20:00 | 2274.99 | 2262.98 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:10:00 | 2282.36 | 2292.20 | 0.00 | ORB-short ORB[2283.38,2314.11] vol=2.6x ATR=8.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 11:40:00 | 2270.23 | 2289.10 | 0.00 | T1 1.5R @ 2270.23 |
| Target hit | 2025-03-28 15:20:00 | 2235.63 | 2262.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2025-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:45:00 | 2283.33 | 2260.48 | 0.00 | ORB-long ORB[2243.92,2271.45] vol=1.6x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 12:10:00 | 2294.47 | 2270.80 | 0.00 | T1 1.5R @ 2294.47 |
| Target hit | 2025-04-02 15:20:00 | 2296.71 | 2280.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — SELL (started 2025-04-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:55:00 | 2274.27 | 2301.41 | 0.00 | ORB-short ORB[2302.53,2336.51] vol=2.5x ATR=9.18 |
| Stop hit — per-position SL triggered | 2025-04-04 10:35:00 | 2283.45 | 2290.74 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 2150.80 | 2166.85 | 0.00 | ORB-short ORB[2181.34,2209.99] vol=2.5x ATR=7.31 |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 2158.11 | 2165.16 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-04-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:30:00 | 2261.03 | 2237.01 | 0.00 | ORB-long ORB[2211.40,2244.26] vol=2.1x ATR=11.13 |
| Stop hit — per-position SL triggered | 2025-04-11 10:05:00 | 2249.90 | 2248.57 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:50:00 | 2327.73 | 2310.55 | 0.00 | ORB-long ORB[2287.02,2313.19] vol=1.9x ATR=6.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:55:00 | 2338.22 | 2317.84 | 0.00 | T1 1.5R @ 2338.22 |
| Target hit | 2025-04-15 13:45:00 | 2331.51 | 2332.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 94 — BUY (started 2025-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:30:00 | 2351.00 | 2333.44 | 0.00 | ORB-long ORB[2319.01,2345.77] vol=1.5x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:40:00 | 2363.12 | 2338.29 | 0.00 | T1 1.5R @ 2363.12 |
| Stop hit — per-position SL triggered | 2025-04-17 11:00:00 | 2351.00 | 2341.27 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:50:00 | 2386.87 | 2353.26 | 0.00 | ORB-long ORB[2323.86,2352.84] vol=1.9x ATR=7.92 |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 2378.95 | 2356.50 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:40:00 | 2401.41 | 2388.16 | 0.00 | ORB-long ORB[2369.91,2396.57] vol=1.5x ATR=6.93 |
| Stop hit — per-position SL triggered | 2025-04-22 11:20:00 | 2394.48 | 2391.46 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 2373.88 | 2383.23 | 0.00 | ORB-short ORB[2376.40,2398.31] vol=1.5x ATR=6.33 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 2380.21 | 2381.27 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 2338.40 | 2360.09 | 0.00 | ORB-short ORB[2358.27,2384.45] vol=2.1x ATR=8.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 2326.27 | 2349.02 | 0.00 | T1 1.5R @ 2326.27 |
| Target hit | 2025-04-25 13:55:00 | 2298.46 | 2291.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 99 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 10:15:00 | 2307.76 | 2288.63 | 0.00 | ORB-long ORB[2266.27,2296.61] vol=1.9x ATR=9.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 10:50:00 | 2321.82 | 2293.48 | 0.00 | T1 1.5R @ 2321.82 |
| Stop hit — per-position SL triggered | 2025-04-28 13:05:00 | 2307.76 | 2302.56 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2025-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:50:00 | 2288.57 | 2259.39 | 0.00 | ORB-long ORB[2227.68,2254.05] vol=2.3x ATR=9.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:05:00 | 2303.47 | 2272.27 | 0.00 | T1 1.5R @ 2303.47 |
| Target hit | 2025-05-05 15:20:00 | 2389.97 | 2362.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 101 — SELL (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 2263.75 | 2271.10 | 0.00 | ORB-short ORB[2264.62,2289.92] vol=1.7x ATR=6.74 |
| Stop hit — per-position SL triggered | 2025-05-08 09:55:00 | 2270.49 | 2270.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-13 10:50:00 | 2736.95 | 2024-05-13 11:10:00 | 2721.59 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-05-14 10:15:00 | 2826.68 | 2024-05-14 10:30:00 | 2843.11 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-05-14 10:15:00 | 2826.68 | 2024-05-14 15:20:00 | 2944.86 | TARGET_HIT | 0.50 | 4.18% |
| SELL | retest1 | 2024-05-16 10:50:00 | 2919.60 | 2024-05-16 11:00:00 | 2932.21 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-17 10:20:00 | 2984.51 | 2024-05-17 10:25:00 | 2971.82 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-21 10:10:00 | 2977.29 | 2024-05-21 11:15:00 | 2991.09 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-05-21 10:10:00 | 2977.29 | 2024-05-21 14:50:00 | 3027.70 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2024-05-23 11:00:00 | 3071.38 | 2024-05-23 11:05:00 | 3085.05 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-23 11:00:00 | 3071.38 | 2024-05-23 15:20:00 | 3299.79 | TARGET_HIT | 0.50 | 7.44% |
| BUY | retest1 | 2024-05-28 10:15:00 | 3216.51 | 2024-05-28 10:25:00 | 3203.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-29 09:30:00 | 3151.26 | 2024-05-29 09:55:00 | 3169.74 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-05-29 09:30:00 | 3151.26 | 2024-05-29 10:20:00 | 3151.26 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 10:55:00 | 3101.82 | 2024-06-07 11:20:00 | 3091.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-13 10:45:00 | 3127.66 | 2024-06-13 11:20:00 | 3116.72 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-06-13 10:45:00 | 3127.66 | 2024-06-13 13:40:00 | 3122.71 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2024-06-14 10:15:00 | 3153.78 | 2024-06-14 10:30:00 | 3144.52 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-20 10:10:00 | 3203.18 | 2024-06-20 10:40:00 | 3191.62 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-21 09:35:00 | 3132.79 | 2024-06-21 10:10:00 | 3118.74 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-21 09:35:00 | 3132.79 | 2024-06-21 10:25:00 | 3132.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 10:40:00 | 3069.83 | 2024-06-25 10:45:00 | 3076.79 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-27 09:40:00 | 3094.16 | 2024-06-27 09:45:00 | 3087.99 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-02 10:35:00 | 3104.24 | 2024-07-02 10:40:00 | 3095.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-04 10:50:00 | 3053.88 | 2024-07-04 11:55:00 | 3060.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-08 10:00:00 | 3027.70 | 2024-07-08 10:05:00 | 3034.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-10 10:35:00 | 2978.26 | 2024-07-10 10:40:00 | 2986.91 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-11 09:30:00 | 3026.30 | 2024-07-11 09:40:00 | 3018.07 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-18 09:30:00 | 2977.05 | 2024-07-18 09:40:00 | 2985.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-19 10:50:00 | 2952.95 | 2024-07-19 10:55:00 | 2939.02 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-19 10:50:00 | 2952.95 | 2024-07-19 11:05:00 | 2952.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 11:00:00 | 2934.05 | 2024-07-23 11:15:00 | 2925.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-25 10:05:00 | 2886.88 | 2024-07-25 10:10:00 | 2899.40 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-25 10:05:00 | 2886.88 | 2024-07-25 12:00:00 | 2894.45 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-07-26 09:50:00 | 2918.63 | 2024-07-26 10:35:00 | 2931.69 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-26 09:50:00 | 2918.63 | 2024-07-26 15:20:00 | 2981.17 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2024-07-31 10:00:00 | 3072.69 | 2024-07-31 10:10:00 | 3063.41 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-01 09:35:00 | 3090.72 | 2024-08-01 10:00:00 | 3104.72 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-01 09:35:00 | 3090.72 | 2024-08-01 10:30:00 | 3090.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-07 10:20:00 | 3054.94 | 2024-08-07 11:55:00 | 3074.09 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-07 10:20:00 | 3054.94 | 2024-08-07 12:40:00 | 3054.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:35:00 | 3111.71 | 2024-08-08 11:25:00 | 3100.03 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-14 11:05:00 | 2948.98 | 2024-08-14 12:25:00 | 2958.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-20 10:50:00 | 2997.26 | 2024-08-20 10:55:00 | 3003.91 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-21 10:05:00 | 3019.95 | 2024-08-21 10:10:00 | 3031.60 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-21 10:05:00 | 3019.95 | 2024-08-21 11:30:00 | 3019.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:30:00 | 3041.61 | 2024-08-22 10:55:00 | 3035.81 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-23 11:00:00 | 2999.25 | 2024-08-23 12:20:00 | 2990.37 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-08-23 11:00:00 | 2999.25 | 2024-08-23 14:50:00 | 2995.66 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2024-08-26 11:05:00 | 2967.69 | 2024-08-26 11:35:00 | 2973.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-27 09:40:00 | 2987.81 | 2024-08-27 09:50:00 | 2982.23 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-28 09:30:00 | 2951.11 | 2024-08-28 09:45:00 | 2956.93 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-29 10:55:00 | 2908.89 | 2024-08-29 11:20:00 | 2915.96 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2904.58 | 2024-09-06 09:55:00 | 2894.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2904.58 | 2024-09-06 10:25:00 | 2904.43 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-09-12 11:00:00 | 2847.91 | 2024-09-12 11:20:00 | 2837.04 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-12 11:00:00 | 2847.91 | 2024-09-12 12:00:00 | 2847.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 09:30:00 | 2925.91 | 2024-09-16 09:55:00 | 2919.14 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-17 10:55:00 | 2875.06 | 2024-09-17 11:10:00 | 2866.68 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-17 10:55:00 | 2875.06 | 2024-09-17 11:25:00 | 2875.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 09:45:00 | 2978.55 | 2024-09-24 10:10:00 | 2970.88 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-26 11:15:00 | 3032.84 | 2024-09-26 11:25:00 | 3043.42 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-26 11:15:00 | 3032.84 | 2024-09-26 11:40:00 | 3032.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-30 10:45:00 | 3091.45 | 2024-09-30 11:05:00 | 3107.36 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-30 10:45:00 | 3091.45 | 2024-09-30 11:10:00 | 3091.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 11:10:00 | 3021.93 | 2024-10-01 11:45:00 | 3030.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-04 11:15:00 | 3037.98 | 2024-10-04 12:25:00 | 3052.70 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-04 11:15:00 | 3037.98 | 2024-10-04 12:35:00 | 3037.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 09:45:00 | 2977.29 | 2024-10-07 10:10:00 | 2961.23 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-07 09:45:00 | 2977.29 | 2024-10-07 10:15:00 | 2977.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 11:05:00 | 2964.83 | 2024-10-08 11:25:00 | 2981.79 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-08 11:05:00 | 2964.83 | 2024-10-08 15:20:00 | 3069.78 | TARGET_HIT | 0.50 | 3.54% |
| SELL | retest1 | 2024-10-11 11:10:00 | 3039.34 | 2024-10-11 14:15:00 | 3022.79 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-11 11:10:00 | 3039.34 | 2024-10-11 14:35:00 | 3039.34 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 10:45:00 | 3028.57 | 2024-10-14 11:40:00 | 3035.11 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-16 11:15:00 | 2993.09 | 2024-10-16 11:45:00 | 2983.06 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-16 11:15:00 | 2993.09 | 2024-10-16 13:20:00 | 2993.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:35:00 | 2966.82 | 2024-10-17 09:45:00 | 2954.73 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 09:35:00 | 2966.82 | 2024-10-17 10:35:00 | 2960.03 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2024-10-21 09:30:00 | 2886.93 | 2024-10-21 09:45:00 | 2875.14 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-21 09:30:00 | 2886.93 | 2024-10-21 10:00:00 | 2886.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 11:00:00 | 2783.34 | 2024-11-04 11:20:00 | 2793.97 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-06 09:45:00 | 2873.55 | 2024-11-06 09:55:00 | 2863.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-11 11:15:00 | 2847.04 | 2024-11-11 11:55:00 | 2838.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-13 09:40:00 | 2762.11 | 2024-11-13 09:50:00 | 2773.22 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-14 09:40:00 | 2727.69 | 2024-11-14 09:55:00 | 2738.13 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-19 09:30:00 | 2773.70 | 2024-11-19 09:50:00 | 2764.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-03 11:00:00 | 2419.20 | 2024-12-03 11:05:00 | 2432.29 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-03 11:00:00 | 2419.20 | 2024-12-03 13:10:00 | 2429.19 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-10 10:00:00 | 2405.29 | 2024-12-10 10:50:00 | 2410.88 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-11 11:05:00 | 2379.31 | 2024-12-11 11:10:00 | 2384.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-12 10:30:00 | 2411.50 | 2024-12-12 10:35:00 | 2404.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-13 09:40:00 | 2444.31 | 2024-12-13 09:45:00 | 2436.81 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-16 11:10:00 | 2438.64 | 2024-12-16 11:25:00 | 2443.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-17 10:45:00 | 2443.01 | 2024-12-17 10:50:00 | 2437.14 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-20 09:35:00 | 2376.50 | 2024-12-20 09:45:00 | 2370.93 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-24 10:30:00 | 2324.78 | 2024-12-24 10:55:00 | 2316.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-06 09:40:00 | 2444.27 | 2025-01-06 09:45:00 | 2452.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-09 10:30:00 | 2423.13 | 2025-01-09 10:45:00 | 2413.68 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-09 10:30:00 | 2423.13 | 2025-01-09 15:20:00 | 2409.85 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-13 10:55:00 | 2253.76 | 2025-01-13 11:05:00 | 2237.83 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-01-13 10:55:00 | 2253.76 | 2025-01-13 15:20:00 | 2157.83 | TARGET_HIT | 0.50 | 4.26% |
| BUY | retest1 | 2025-01-15 09:40:00 | 2335.78 | 2025-01-15 09:55:00 | 2326.46 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-01-20 10:55:00 | 2343.78 | 2025-01-20 12:40:00 | 2353.49 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-01-20 10:55:00 | 2343.78 | 2025-01-20 15:20:00 | 2361.28 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2025-01-24 10:50:00 | 2300.25 | 2025-01-24 11:25:00 | 2308.75 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-30 10:00:00 | 2278.82 | 2025-01-30 10:05:00 | 2271.93 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-07 10:55:00 | 2270.97 | 2025-02-07 11:15:00 | 2282.57 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-02-07 10:55:00 | 2270.97 | 2025-02-07 12:15:00 | 2270.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 09:40:00 | 2229.14 | 2025-02-12 09:55:00 | 2213.86 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-02-12 09:40:00 | 2229.14 | 2025-02-12 10:15:00 | 2229.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-13 09:55:00 | 2278.00 | 2025-02-13 10:00:00 | 2286.21 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-21 09:40:00 | 2104.27 | 2025-02-21 09:50:00 | 2093.47 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-02-21 09:40:00 | 2104.27 | 2025-02-21 12:30:00 | 2098.45 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-02-25 09:40:00 | 2071.79 | 2025-02-25 09:55:00 | 2064.61 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-27 09:50:00 | 2061.95 | 2025-02-27 10:00:00 | 2053.87 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-02-27 09:50:00 | 2061.95 | 2025-02-27 10:15:00 | 2061.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-28 09:30:00 | 2011.10 | 2025-02-28 09:45:00 | 2019.93 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-03-05 09:45:00 | 2127.63 | 2025-03-05 10:15:00 | 2119.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-18 09:35:00 | 2213.72 | 2025-03-18 09:40:00 | 2207.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-21 10:30:00 | 2293.66 | 2025-03-21 11:00:00 | 2302.84 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-21 10:30:00 | 2293.66 | 2025-03-21 13:35:00 | 2293.66 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 10:10:00 | 2282.60 | 2025-03-26 10:20:00 | 2274.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-28 11:10:00 | 2282.36 | 2025-03-28 11:40:00 | 2270.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-03-28 11:10:00 | 2282.36 | 2025-03-28 15:20:00 | 2235.63 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2025-04-02 10:45:00 | 2283.33 | 2025-04-02 12:10:00 | 2294.47 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-02 10:45:00 | 2283.33 | 2025-04-02 15:20:00 | 2296.71 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-04-04 09:55:00 | 2274.27 | 2025-04-04 10:35:00 | 2283.45 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-09 11:00:00 | 2150.80 | 2025-04-09 11:15:00 | 2158.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-11 09:30:00 | 2261.03 | 2025-04-11 10:05:00 | 2249.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-04-15 10:50:00 | 2327.73 | 2025-04-15 10:55:00 | 2338.22 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-04-15 10:50:00 | 2327.73 | 2025-04-15 13:45:00 | 2331.51 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-04-17 10:30:00 | 2351.00 | 2025-04-17 10:40:00 | 2363.12 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-17 10:30:00 | 2351.00 | 2025-04-17 11:00:00 | 2351.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 10:50:00 | 2386.87 | 2025-04-21 11:15:00 | 2378.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-22 10:40:00 | 2401.41 | 2025-04-22 11:20:00 | 2394.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 09:30:00 | 2373.88 | 2025-04-23 09:40:00 | 2380.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2338.40 | 2025-04-25 09:45:00 | 2326.27 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2338.40 | 2025-04-25 13:55:00 | 2298.46 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2025-04-28 10:15:00 | 2307.76 | 2025-04-28 10:50:00 | 2321.82 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-04-28 10:15:00 | 2307.76 | 2025-04-28 13:05:00 | 2307.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:50:00 | 2288.57 | 2025-05-05 10:05:00 | 2303.47 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-05 09:50:00 | 2288.57 | 2025-05-05 15:20:00 | 2389.97 | TARGET_HIT | 0.50 | 4.43% |
| SELL | retest1 | 2025-05-08 09:30:00 | 2263.75 | 2025-05-08 09:55:00 | 2270.49 | STOP_HIT | 1.00 | -0.30% |
