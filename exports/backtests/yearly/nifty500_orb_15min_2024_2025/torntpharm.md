# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-31 15:25:00 (27421 bars)
- **Last close:** 3547.20
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
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 67
- **Target hits / Stop hits / Partials:** 11 / 67 / 20
- **Avg / median % per leg:** -0.03% / -0.24%
- **Sum % (uncompounded):** -2.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 14 | 28.0% | 6 | 36 | 8 | -0.08% | -4.1% |
| BUY @ 2nd Alert (retest1) | 50 | 14 | 28.0% | 6 | 36 | 8 | -0.08% | -4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 17 | 35.4% | 5 | 31 | 12 | 0.03% | 1.3% |
| SELL @ 2nd Alert (retest1) | 48 | 17 | 35.4% | 5 | 31 | 12 | 0.03% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 31 | 31.6% | 11 | 67 | 20 | -0.03% | -2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:05:00 | 2667.60 | 2656.23 | 0.00 | ORB-long ORB[2635.35,2654.45] vol=3.6x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:20:00 | 2678.48 | 2662.84 | 0.00 | T1 1.5R @ 2678.48 |
| Target hit | 2024-05-16 13:25:00 | 2681.00 | 2681.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 2690.95 | 2699.90 | 0.00 | ORB-short ORB[2698.75,2718.80] vol=3.0x ATR=6.04 |
| Stop hit — per-position SL triggered | 2024-05-17 11:00:00 | 2696.99 | 2700.42 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:15:00 | 2675.80 | 2683.78 | 0.00 | ORB-short ORB[2684.50,2702.90] vol=2.3x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:25:00 | 2667.33 | 2682.13 | 0.00 | T1 1.5R @ 2667.33 |
| Stop hit — per-position SL triggered | 2024-05-22 11:45:00 | 2675.80 | 2680.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:10:00 | 2657.05 | 2664.33 | 0.00 | ORB-short ORB[2665.40,2684.75] vol=4.6x ATR=7.33 |
| Stop hit — per-position SL triggered | 2024-05-29 10:55:00 | 2664.38 | 2662.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:00:00 | 2696.90 | 2667.66 | 0.00 | ORB-long ORB[2648.00,2670.00] vol=1.6x ATR=7.79 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 2689.11 | 2670.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 2898.35 | 2891.94 | 0.00 | ORB-long ORB[2873.00,2892.00] vol=1.8x ATR=7.17 |
| Stop hit — per-position SL triggered | 2024-06-14 10:00:00 | 2891.18 | 2891.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 2916.95 | 2900.97 | 0.00 | ORB-long ORB[2884.80,2903.40] vol=1.7x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:35:00 | 2930.29 | 2914.10 | 0.00 | T1 1.5R @ 2930.29 |
| Target hit | 2024-06-19 09:55:00 | 2930.45 | 2945.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:15:00 | 2855.55 | 2835.85 | 0.00 | ORB-long ORB[2808.05,2837.90] vol=2.8x ATR=7.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:35:00 | 2866.60 | 2842.86 | 0.00 | T1 1.5R @ 2866.60 |
| Target hit | 2024-06-24 12:50:00 | 2861.20 | 2863.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:55:00 | 2814.70 | 2821.63 | 0.00 | ORB-short ORB[2822.60,2843.20] vol=2.6x ATR=6.27 |
| Stop hit — per-position SL triggered | 2024-06-26 10:30:00 | 2820.97 | 2819.76 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:00:00 | 2814.20 | 2809.00 | 0.00 | ORB-long ORB[2792.10,2812.00] vol=11.3x ATR=7.10 |
| Stop hit — per-position SL triggered | 2024-07-01 10:05:00 | 2807.10 | 2809.06 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 2817.90 | 2829.60 | 0.00 | ORB-short ORB[2824.45,2856.40] vol=1.8x ATR=7.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:00:00 | 2806.72 | 2819.10 | 0.00 | T1 1.5R @ 2806.72 |
| Stop hit — per-position SL triggered | 2024-07-04 11:05:00 | 2817.90 | 2819.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 2912.25 | 2897.97 | 0.00 | ORB-long ORB[2874.70,2908.80] vol=2.1x ATR=9.48 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 2902.77 | 2904.47 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 2874.90 | 2888.20 | 0.00 | ORB-short ORB[2891.10,2919.50] vol=3.6x ATR=7.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 12:00:00 | 2863.17 | 2885.17 | 0.00 | T1 1.5R @ 2863.17 |
| Stop hit — per-position SL triggered | 2024-07-08 15:20:00 | 2893.95 | 2873.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:55:00 | 2938.95 | 2945.79 | 0.00 | ORB-short ORB[2943.50,2977.95] vol=4.7x ATR=6.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:20:00 | 2928.63 | 2943.61 | 0.00 | T1 1.5R @ 2928.63 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 2938.95 | 2943.20 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:45:00 | 2999.00 | 2970.46 | 0.00 | ORB-long ORB[2945.65,2971.50] vol=3.3x ATR=7.41 |
| Stop hit — per-position SL triggered | 2024-07-16 09:50:00 | 2991.59 | 2972.70 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:15:00 | 2967.35 | 2982.66 | 0.00 | ORB-short ORB[2970.10,2998.50] vol=1.7x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:35:00 | 2955.23 | 2981.66 | 0.00 | T1 1.5R @ 2955.23 |
| Stop hit — per-position SL triggered | 2024-07-18 11:45:00 | 2967.35 | 2980.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:55:00 | 3083.00 | 3059.46 | 0.00 | ORB-long ORB[3028.65,3058.95] vol=3.9x ATR=12.77 |
| Stop hit — per-position SL triggered | 2024-07-23 10:00:00 | 3070.23 | 3063.88 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:10:00 | 3142.40 | 3168.71 | 0.00 | ORB-short ORB[3162.05,3208.20] vol=1.5x ATR=11.51 |
| Stop hit — per-position SL triggered | 2024-07-29 11:20:00 | 3153.91 | 3167.32 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 3154.80 | 3137.00 | 0.00 | ORB-long ORB[3122.20,3138.00] vol=1.8x ATR=8.98 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 3145.82 | 3140.68 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 3197.30 | 3191.05 | 0.00 | ORB-long ORB[3160.05,3196.30] vol=1.7x ATR=10.27 |
| Stop hit — per-position SL triggered | 2024-08-01 11:00:00 | 3187.03 | 3192.23 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 09:30:00 | 3199.75 | 3187.68 | 0.00 | ORB-long ORB[3173.05,3196.05] vol=1.7x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 10:10:00 | 3214.97 | 3195.77 | 0.00 | T1 1.5R @ 3214.97 |
| Target hit | 2024-08-06 10:30:00 | 3202.60 | 3204.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:15:00 | 3270.85 | 3256.53 | 0.00 | ORB-long ORB[3230.05,3266.95] vol=3.4x ATR=9.17 |
| Stop hit — per-position SL triggered | 2024-08-07 11:55:00 | 3261.68 | 3259.97 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 3320.00 | 3336.16 | 0.00 | ORB-short ORB[3322.95,3348.85] vol=1.9x ATR=11.41 |
| Stop hit — per-position SL triggered | 2024-08-13 09:45:00 | 3331.41 | 3332.53 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:35:00 | 3350.50 | 3357.63 | 0.00 | ORB-short ORB[3351.35,3363.30] vol=2.9x ATR=6.88 |
| Stop hit — per-position SL triggered | 2024-08-21 10:40:00 | 3357.38 | 3357.88 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:15:00 | 3358.00 | 3365.57 | 0.00 | ORB-short ORB[3364.65,3378.45] vol=1.7x ATR=7.75 |
| Stop hit — per-position SL triggered | 2024-08-23 10:20:00 | 3365.75 | 3365.33 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:05:00 | 3369.30 | 3354.33 | 0.00 | ORB-long ORB[3311.45,3353.80] vol=1.7x ATR=10.30 |
| Stop hit — per-position SL triggered | 2024-08-26 10:20:00 | 3359.00 | 3356.28 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:20:00 | 3507.55 | 3478.19 | 0.00 | ORB-long ORB[3447.05,3484.50] vol=1.8x ATR=13.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:50:00 | 3528.20 | 3490.16 | 0.00 | T1 1.5R @ 3528.20 |
| Target hit | 2024-08-30 13:05:00 | 3520.60 | 3521.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 3484.20 | 3461.57 | 0.00 | ORB-long ORB[3435.55,3471.00] vol=2.2x ATR=10.42 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 3473.78 | 3468.90 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:45:00 | 3487.35 | 3469.88 | 0.00 | ORB-long ORB[3441.50,3463.10] vol=2.5x ATR=9.96 |
| Stop hit — per-position SL triggered | 2024-09-05 10:00:00 | 3477.39 | 3477.88 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 3507.85 | 3490.68 | 0.00 | ORB-long ORB[3452.70,3499.00] vol=1.7x ATR=11.84 |
| Stop hit — per-position SL triggered | 2024-09-12 09:40:00 | 3496.01 | 3493.74 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:10:00 | 3450.00 | 3459.74 | 0.00 | ORB-short ORB[3465.85,3498.80] vol=2.2x ATR=7.23 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 3457.23 | 3459.37 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:15:00 | 3482.05 | 3460.66 | 0.00 | ORB-long ORB[3441.05,3463.85] vol=2.1x ATR=7.54 |
| Stop hit — per-position SL triggered | 2024-09-16 11:20:00 | 3474.51 | 3460.84 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 3469.50 | 3467.10 | 0.00 | ORB-long ORB[3454.20,3468.90] vol=2.2x ATR=8.61 |
| Stop hit — per-position SL triggered | 2024-09-17 09:35:00 | 3460.89 | 3466.12 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 3346.00 | 3366.39 | 0.00 | ORB-short ORB[3348.05,3394.85] vol=1.6x ATR=11.14 |
| Stop hit — per-position SL triggered | 2024-09-19 09:55:00 | 3357.14 | 3364.93 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 3489.10 | 3478.01 | 0.00 | ORB-long ORB[3457.85,3487.00] vol=1.8x ATR=7.91 |
| Stop hit — per-position SL triggered | 2024-09-24 09:40:00 | 3481.19 | 3478.44 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 3437.80 | 3456.00 | 0.00 | ORB-short ORB[3450.00,3492.00] vol=2.1x ATR=10.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 12:15:00 | 3422.75 | 3448.13 | 0.00 | T1 1.5R @ 3422.75 |
| Target hit | 2024-09-25 15:20:00 | 3426.00 | 3435.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-09-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:50:00 | 3398.75 | 3423.02 | 0.00 | ORB-short ORB[3420.05,3449.70] vol=2.2x ATR=7.50 |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 3406.25 | 3419.71 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:05:00 | 3436.15 | 3425.69 | 0.00 | ORB-long ORB[3396.00,3422.95] vol=3.0x ATR=11.00 |
| Stop hit — per-position SL triggered | 2024-09-27 10:10:00 | 3425.15 | 3426.00 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 11:00:00 | 3391.80 | 3359.01 | 0.00 | ORB-long ORB[3333.40,3364.85] vol=1.6x ATR=11.11 |
| Stop hit — per-position SL triggered | 2024-10-03 13:50:00 | 3380.69 | 3382.70 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 09:30:00 | 3511.50 | 3494.85 | 0.00 | ORB-long ORB[3474.05,3509.90] vol=1.6x ATR=14.47 |
| Stop hit — per-position SL triggered | 2024-10-07 09:45:00 | 3497.03 | 3498.55 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:35:00 | 3444.00 | 3417.06 | 0.00 | ORB-long ORB[3380.90,3426.90] vol=1.9x ATR=11.96 |
| Stop hit — per-position SL triggered | 2024-10-08 10:50:00 | 3432.04 | 3421.94 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 3492.00 | 3474.03 | 0.00 | ORB-long ORB[3443.00,3483.00] vol=1.8x ATR=12.37 |
| Stop hit — per-position SL triggered | 2024-10-11 10:05:00 | 3479.63 | 3480.59 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:05:00 | 3461.40 | 3474.00 | 0.00 | ORB-short ORB[3469.75,3510.30] vol=1.9x ATR=10.93 |
| Stop hit — per-position SL triggered | 2024-10-14 11:25:00 | 3472.33 | 3473.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 3524.80 | 3513.11 | 0.00 | ORB-long ORB[3480.10,3521.80] vol=2.7x ATR=8.42 |
| Stop hit — per-position SL triggered | 2024-10-15 09:40:00 | 3516.38 | 3515.69 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:05:00 | 3372.25 | 3352.80 | 0.00 | ORB-long ORB[3325.10,3369.70] vol=2.1x ATR=12.73 |
| Stop hit — per-position SL triggered | 2024-10-23 10:10:00 | 3359.52 | 3355.98 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 09:40:00 | 3188.25 | 3197.59 | 0.00 | ORB-short ORB[3191.55,3226.45] vol=1.6x ATR=10.78 |
| Stop hit — per-position SL triggered | 2024-11-04 09:45:00 | 3199.03 | 3197.01 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:00:00 | 3156.25 | 3184.30 | 0.00 | ORB-short ORB[3201.10,3239.10] vol=1.9x ATR=9.75 |
| Stop hit — per-position SL triggered | 2024-11-07 11:05:00 | 3166.00 | 3183.40 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:10:00 | 3123.20 | 3104.74 | 0.00 | ORB-long ORB[3080.10,3122.65] vol=2.0x ATR=9.16 |
| Stop hit — per-position SL triggered | 2024-11-21 11:30:00 | 3114.04 | 3107.88 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:50:00 | 3137.40 | 3130.67 | 0.00 | ORB-long ORB[3102.05,3137.30] vol=4.2x ATR=9.48 |
| Stop hit — per-position SL triggered | 2024-11-22 09:55:00 | 3127.92 | 3130.87 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 3293.30 | 3254.77 | 0.00 | ORB-long ORB[3200.00,3236.90] vol=1.9x ATR=9.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:20:00 | 3307.85 | 3269.76 | 0.00 | T1 1.5R @ 3307.85 |
| Target hit | 2024-11-29 15:20:00 | 3328.10 | 3307.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 3377.40 | 3366.77 | 0.00 | ORB-long ORB[3341.60,3369.00] vol=1.7x ATR=8.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:25:00 | 3389.91 | 3370.49 | 0.00 | T1 1.5R @ 3389.91 |
| Stop hit — per-position SL triggered | 2024-12-03 11:40:00 | 3377.40 | 3371.32 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:55:00 | 3326.50 | 3336.21 | 0.00 | ORB-short ORB[3343.30,3366.00] vol=5.7x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:20:00 | 3313.77 | 3331.58 | 0.00 | T1 1.5R @ 3313.77 |
| Target hit | 2024-12-04 11:05:00 | 3325.40 | 3324.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2024-12-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:50:00 | 3299.90 | 3317.24 | 0.00 | ORB-short ORB[3321.00,3340.85] vol=2.8x ATR=7.98 |
| Stop hit — per-position SL triggered | 2024-12-13 11:00:00 | 3307.88 | 3316.69 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:45:00 | 3330.90 | 3342.92 | 0.00 | ORB-short ORB[3333.80,3374.55] vol=1.8x ATR=9.21 |
| Stop hit — per-position SL triggered | 2024-12-16 11:00:00 | 3340.11 | 3342.46 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:45:00 | 3385.80 | 3393.14 | 0.00 | ORB-short ORB[3388.55,3424.85] vol=5.7x ATR=9.57 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 3395.37 | 3390.99 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:35:00 | 3416.20 | 3421.94 | 0.00 | ORB-short ORB[3417.25,3446.05] vol=2.1x ATR=10.43 |
| Stop hit — per-position SL triggered | 2024-12-27 09:45:00 | 3426.63 | 3424.13 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:35:00 | 3377.30 | 3386.61 | 0.00 | ORB-short ORB[3395.00,3440.00] vol=2.5x ATR=8.14 |
| Stop hit — per-position SL triggered | 2025-01-02 10:40:00 | 3385.44 | 3386.81 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:00:00 | 3380.35 | 3390.41 | 0.00 | ORB-short ORB[3390.00,3418.55] vol=2.3x ATR=9.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:45:00 | 3366.01 | 3385.45 | 0.00 | T1 1.5R @ 3366.01 |
| Stop hit — per-position SL triggered | 2025-01-06 12:20:00 | 3380.35 | 3381.66 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 3427.90 | 3407.41 | 0.00 | ORB-long ORB[3351.55,3392.00] vol=1.7x ATR=12.25 |
| Stop hit — per-position SL triggered | 2025-01-07 12:00:00 | 3415.65 | 3412.94 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:35:00 | 3383.90 | 3392.05 | 0.00 | ORB-short ORB[3387.20,3417.05] vol=1.7x ATR=8.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:20:00 | 3370.49 | 3386.67 | 0.00 | T1 1.5R @ 3370.49 |
| Target hit | 2025-01-08 15:20:00 | 3326.40 | 3349.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 3134.25 | 3143.80 | 0.00 | ORB-short ORB[3143.25,3184.75] vol=2.5x ATR=10.16 |
| Stop hit — per-position SL triggered | 2025-01-22 10:10:00 | 3144.41 | 3141.31 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:30:00 | 3191.45 | 3187.15 | 0.00 | ORB-long ORB[3157.20,3179.60] vol=4.8x ATR=8.02 |
| Stop hit — per-position SL triggered | 2025-01-23 10:35:00 | 3183.43 | 3187.00 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:20:00 | 3262.40 | 3304.06 | 0.00 | ORB-short ORB[3310.60,3344.00] vol=1.7x ATR=16.32 |
| Stop hit — per-position SL triggered | 2025-01-28 10:25:00 | 3278.72 | 3302.63 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:05:00 | 3371.75 | 3343.83 | 0.00 | ORB-long ORB[3300.00,3349.00] vol=2.2x ATR=10.42 |
| Stop hit — per-position SL triggered | 2025-01-29 11:10:00 | 3361.33 | 3344.43 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 11:10:00 | 3084.05 | 3096.15 | 0.00 | ORB-short ORB[3088.05,3128.95] vol=2.0x ATR=7.20 |
| Stop hit — per-position SL triggered | 2025-02-20 12:00:00 | 3091.25 | 3091.23 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:45:00 | 3008.50 | 3001.44 | 0.00 | ORB-long ORB[2974.05,3002.75] vol=2.0x ATR=8.87 |
| Stop hit — per-position SL triggered | 2025-03-06 11:25:00 | 2999.63 | 3002.87 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:55:00 | 3077.85 | 3063.43 | 0.00 | ORB-long ORB[3046.45,3069.00] vol=3.6x ATR=9.07 |
| Stop hit — per-position SL triggered | 2025-03-07 11:00:00 | 3068.78 | 3068.98 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:20:00 | 3131.55 | 3112.04 | 0.00 | ORB-long ORB[3056.80,3101.20] vol=3.5x ATR=10.01 |
| Stop hit — per-position SL triggered | 2025-03-10 10:30:00 | 3121.54 | 3117.47 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:40:00 | 3087.10 | 3073.50 | 0.00 | ORB-long ORB[3028.00,3067.50] vol=8.4x ATR=9.12 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 3077.98 | 3075.84 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:50:00 | 3071.25 | 3078.66 | 0.00 | ORB-short ORB[3076.95,3109.85] vol=4.1x ATR=7.26 |
| Stop hit — per-position SL triggered | 2025-03-13 11:10:00 | 3078.51 | 3075.58 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 3246.55 | 3233.19 | 0.00 | ORB-long ORB[3211.45,3240.00] vol=1.6x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:25:00 | 3259.65 | 3245.51 | 0.00 | T1 1.5R @ 3259.65 |
| Stop hit — per-position SL triggered | 2025-03-21 13:10:00 | 3246.55 | 3251.17 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 3211.25 | 3226.31 | 0.00 | ORB-short ORB[3221.55,3253.55] vol=3.2x ATR=12.47 |
| Stop hit — per-position SL triggered | 2025-03-26 09:40:00 | 3223.72 | 3225.60 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-03-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:10:00 | 3177.55 | 3191.62 | 0.00 | ORB-short ORB[3185.85,3209.45] vol=4.3x ATR=11.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:15:00 | 3160.76 | 3189.61 | 0.00 | T1 1.5R @ 3160.76 |
| Stop hit — per-position SL triggered | 2025-03-27 10:40:00 | 3177.55 | 3183.82 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:30:00 | 3285.90 | 3270.74 | 0.00 | ORB-long ORB[3230.10,3269.00] vol=1.8x ATR=10.46 |
| Stop hit — per-position SL triggered | 2025-04-23 10:55:00 | 3275.44 | 3271.93 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 3269.40 | 3296.88 | 0.00 | ORB-short ORB[3316.60,3349.10] vol=2.3x ATR=11.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:05:00 | 3252.51 | 3287.70 | 0.00 | T1 1.5R @ 3252.51 |
| Target hit | 2025-04-25 12:30:00 | 3242.10 | 3241.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2025-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:50:00 | 3286.50 | 3305.76 | 0.00 | ORB-short ORB[3288.10,3322.00] vol=2.4x ATR=10.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:10:00 | 3271.42 | 3301.11 | 0.00 | T1 1.5R @ 3271.42 |
| Target hit | 2025-05-02 15:20:00 | 3253.30 | 3268.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 3291.90 | 3276.75 | 0.00 | ORB-long ORB[3256.10,3283.50] vol=2.2x ATR=11.74 |
| Stop hit — per-position SL triggered | 2025-05-05 10:35:00 | 3280.16 | 3277.78 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:55:00 | 3244.60 | 3260.43 | 0.00 | ORB-short ORB[3246.10,3287.00] vol=2.0x ATR=10.93 |
| Stop hit — per-position SL triggered | 2025-05-06 10:30:00 | 3255.53 | 3256.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:05:00 | 2667.60 | 2024-05-16 10:20:00 | 2678.48 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-05-16 10:05:00 | 2667.60 | 2024-05-16 13:25:00 | 2681.00 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-17 10:50:00 | 2690.95 | 2024-05-17 11:00:00 | 2696.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-22 11:15:00 | 2675.80 | 2024-05-22 11:25:00 | 2667.33 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-22 11:15:00 | 2675.80 | 2024-05-22 11:45:00 | 2675.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-29 10:10:00 | 2657.05 | 2024-05-29 10:55:00 | 2664.38 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-31 11:00:00 | 2696.90 | 2024-05-31 11:15:00 | 2689.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-14 09:55:00 | 2898.35 | 2024-06-14 10:00:00 | 2891.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-19 09:30:00 | 2916.95 | 2024-06-19 09:35:00 | 2930.29 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-06-19 09:30:00 | 2916.95 | 2024-06-19 09:55:00 | 2930.45 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-06-24 11:15:00 | 2855.55 | 2024-06-24 11:35:00 | 2866.60 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-24 11:15:00 | 2855.55 | 2024-06-24 12:50:00 | 2861.20 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-06-26 09:55:00 | 2814.70 | 2024-06-26 10:30:00 | 2820.97 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-01 10:00:00 | 2814.20 | 2024-07-01 10:05:00 | 2807.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-04 09:45:00 | 2817.90 | 2024-07-04 11:00:00 | 2806.72 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-04 09:45:00 | 2817.90 | 2024-07-04 11:05:00 | 2817.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 09:35:00 | 2912.25 | 2024-07-05 09:50:00 | 2902.77 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-08 11:10:00 | 2874.90 | 2024-07-08 12:00:00 | 2863.17 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-08 11:10:00 | 2874.90 | 2024-07-08 15:20:00 | 2893.95 | STOP_HIT | 0.50 | -0.66% |
| SELL | retest1 | 2024-07-11 10:55:00 | 2938.95 | 2024-07-11 11:20:00 | 2928.63 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-11 10:55:00 | 2938.95 | 2024-07-11 11:40:00 | 2938.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:45:00 | 2999.00 | 2024-07-16 09:50:00 | 2991.59 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-18 11:15:00 | 2967.35 | 2024-07-18 11:35:00 | 2955.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-18 11:15:00 | 2967.35 | 2024-07-18 11:45:00 | 2967.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 09:55:00 | 3083.00 | 2024-07-23 10:00:00 | 3070.23 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-29 11:10:00 | 3142.40 | 2024-07-29 11:20:00 | 3153.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-31 09:30:00 | 3154.80 | 2024-07-31 09:35:00 | 3145.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-01 10:15:00 | 3197.30 | 2024-08-01 11:00:00 | 3187.03 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-06 09:30:00 | 3199.75 | 2024-08-06 10:10:00 | 3214.97 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-06 09:30:00 | 3199.75 | 2024-08-06 10:30:00 | 3202.60 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-08-07 11:15:00 | 3270.85 | 2024-08-07 11:55:00 | 3261.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-13 09:35:00 | 3320.00 | 2024-08-13 09:45:00 | 3331.41 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-21 10:35:00 | 3350.50 | 2024-08-21 10:40:00 | 3357.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-23 10:15:00 | 3358.00 | 2024-08-23 10:20:00 | 3365.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-26 10:05:00 | 3369.30 | 2024-08-26 10:20:00 | 3359.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-30 10:20:00 | 3507.55 | 2024-08-30 10:50:00 | 3528.20 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-30 10:20:00 | 3507.55 | 2024-08-30 13:05:00 | 3520.60 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-03 09:35:00 | 3484.20 | 2024-09-03 09:50:00 | 3473.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-05 09:45:00 | 3487.35 | 2024-09-05 10:00:00 | 3477.39 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-12 09:35:00 | 3507.85 | 2024-09-12 09:40:00 | 3496.01 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-13 11:10:00 | 3450.00 | 2024-09-13 11:20:00 | 3457.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-16 11:15:00 | 3482.05 | 2024-09-16 11:20:00 | 3474.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-17 09:30:00 | 3469.50 | 2024-09-17 09:35:00 | 3460.89 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-19 09:50:00 | 3346.00 | 2024-09-19 09:55:00 | 3357.14 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-24 09:30:00 | 3489.10 | 2024-09-24 09:40:00 | 3481.19 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-25 11:00:00 | 3437.80 | 2024-09-25 12:15:00 | 3422.75 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-25 11:00:00 | 3437.80 | 2024-09-25 15:20:00 | 3426.00 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-26 10:50:00 | 3398.75 | 2024-09-26 11:15:00 | 3406.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-27 10:05:00 | 3436.15 | 2024-09-27 10:10:00 | 3425.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-03 11:00:00 | 3391.80 | 2024-10-03 13:50:00 | 3380.69 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-07 09:30:00 | 3511.50 | 2024-10-07 09:45:00 | 3497.03 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-08 10:35:00 | 3444.00 | 2024-10-08 10:50:00 | 3432.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-11 09:40:00 | 3492.00 | 2024-10-11 10:05:00 | 3479.63 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-14 11:05:00 | 3461.40 | 2024-10-14 11:25:00 | 3472.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-15 09:30:00 | 3524.80 | 2024-10-15 09:40:00 | 3516.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-23 10:05:00 | 3372.25 | 2024-10-23 10:10:00 | 3359.52 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-04 09:40:00 | 3188.25 | 2024-11-04 09:45:00 | 3199.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-07 11:00:00 | 3156.25 | 2024-11-07 11:05:00 | 3166.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-21 11:10:00 | 3123.20 | 2024-11-21 11:30:00 | 3114.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-22 09:50:00 | 3137.40 | 2024-11-22 09:55:00 | 3127.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-29 10:55:00 | 3293.30 | 2024-11-29 11:20:00 | 3307.85 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-11-29 10:55:00 | 3293.30 | 2024-11-29 15:20:00 | 3328.10 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-12-03 10:40:00 | 3377.40 | 2024-12-03 11:25:00 | 3389.91 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-03 10:40:00 | 3377.40 | 2024-12-03 11:40:00 | 3377.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 09:55:00 | 3326.50 | 2024-12-04 10:20:00 | 3313.77 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-04 09:55:00 | 3326.50 | 2024-12-04 11:05:00 | 3325.40 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2024-12-13 10:50:00 | 3299.90 | 2024-12-13 11:00:00 | 3307.88 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-16 10:45:00 | 3330.90 | 2024-12-16 11:00:00 | 3340.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-26 09:45:00 | 3385.80 | 2024-12-26 11:00:00 | 3395.37 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-27 09:35:00 | 3416.20 | 2024-12-27 09:45:00 | 3426.63 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-02 10:35:00 | 3377.30 | 2025-01-02 10:40:00 | 3385.44 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-06 11:00:00 | 3380.35 | 2025-01-06 11:45:00 | 3366.01 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-06 11:00:00 | 3380.35 | 2025-01-06 12:20:00 | 3380.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-07 10:55:00 | 3427.90 | 2025-01-07 12:00:00 | 3415.65 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-08 09:35:00 | 3383.90 | 2025-01-08 10:20:00 | 3370.49 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-01-08 09:35:00 | 3383.90 | 2025-01-08 15:20:00 | 3326.40 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2025-01-22 09:50:00 | 3134.25 | 2025-01-22 10:10:00 | 3144.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-23 10:30:00 | 3191.45 | 2025-01-23 10:35:00 | 3183.43 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-28 10:20:00 | 3262.40 | 2025-01-28 10:25:00 | 3278.72 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-29 11:05:00 | 3371.75 | 2025-01-29 11:10:00 | 3361.33 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-20 11:10:00 | 3084.05 | 2025-02-20 12:00:00 | 3091.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-06 10:45:00 | 3008.50 | 2025-03-06 11:25:00 | 2999.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-07 09:55:00 | 3077.85 | 2025-03-07 11:00:00 | 3068.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-10 10:20:00 | 3131.55 | 2025-03-10 10:30:00 | 3121.54 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-11 10:40:00 | 3087.10 | 2025-03-11 11:15:00 | 3077.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-13 10:50:00 | 3071.25 | 2025-03-13 11:10:00 | 3078.51 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-21 09:35:00 | 3246.55 | 2025-03-21 11:25:00 | 3259.65 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-21 09:35:00 | 3246.55 | 2025-03-21 13:10:00 | 3246.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-26 09:35:00 | 3211.25 | 2025-03-26 09:40:00 | 3223.72 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-27 10:10:00 | 3177.55 | 2025-03-27 10:15:00 | 3160.76 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-03-27 10:10:00 | 3177.55 | 2025-03-27 10:40:00 | 3177.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-23 10:30:00 | 3285.90 | 2025-04-23 10:55:00 | 3275.44 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-25 09:50:00 | 3269.40 | 2025-04-25 10:05:00 | 3252.51 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 09:50:00 | 3269.40 | 2025-04-25 12:30:00 | 3242.10 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-05-02 10:50:00 | 3286.50 | 2025-05-02 11:10:00 | 3271.42 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-05-02 10:50:00 | 3286.50 | 2025-05-02 15:20:00 | 3253.30 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-05-05 10:15:00 | 3291.90 | 2025-05-05 10:35:00 | 3280.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-05-06 09:55:00 | 3244.60 | 2025-05-06 10:30:00 | 3255.53 | STOP_HIT | 1.00 | -0.34% |
