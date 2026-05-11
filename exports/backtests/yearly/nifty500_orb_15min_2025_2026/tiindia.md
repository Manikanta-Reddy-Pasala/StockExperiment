# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 3032.70
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
| ENTRY1 | 88 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 21 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 67
- **Target hits / Stop hits / Partials:** 21 / 67 / 39
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 30.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 30 | 49.2% | 10 | 31 | 20 | 0.30% | 18.3% |
| BUY @ 2nd Alert (retest1) | 61 | 30 | 49.2% | 10 | 31 | 20 | 0.30% | 18.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 30 | 45.5% | 11 | 36 | 19 | 0.19% | 12.3% |
| SELL @ 2nd Alert (retest1) | 66 | 30 | 45.5% | 11 | 36 | 19 | 0.19% | 12.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 60 | 47.2% | 21 | 67 | 39 | 0.24% | 30.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-21 09:30:00 | 3000.40 | 3021.56 | 0.00 | ORB-short ORB[3007.10,3048.30] vol=1.5x ATR=13.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 09:55:00 | 2979.94 | 3010.71 | 0.00 | T1 1.5R @ 2979.94 |
| Stop hit — per-position SL triggered | 2025-05-21 10:25:00 | 3000.40 | 3006.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 10:15:00 | 3033.00 | 3022.16 | 0.00 | ORB-long ORB[2996.50,3031.00] vol=1.8x ATR=11.50 |
| Stop hit — per-position SL triggered | 2025-05-22 10:25:00 | 3021.50 | 3022.29 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:15:00 | 3037.60 | 3041.35 | 0.00 | ORB-short ORB[3038.20,3069.70] vol=4.8x ATR=11.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 11:35:00 | 3020.84 | 3039.24 | 0.00 | T1 1.5R @ 3020.84 |
| Stop hit — per-position SL triggered | 2025-05-26 11:40:00 | 3037.60 | 3039.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:30:00 | 3010.10 | 3023.93 | 0.00 | ORB-short ORB[3040.00,3059.50] vol=2.7x ATR=9.94 |
| Stop hit — per-position SL triggered | 2025-05-27 11:25:00 | 3020.04 | 3019.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:55:00 | 3085.80 | 3070.53 | 0.00 | ORB-long ORB[3054.50,3077.00] vol=3.2x ATR=9.29 |
| Stop hit — per-position SL triggered | 2025-05-30 10:05:00 | 3076.51 | 3072.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 3055.10 | 3068.33 | 0.00 | ORB-short ORB[3067.10,3099.00] vol=2.4x ATR=8.24 |
| Stop hit — per-position SL triggered | 2025-06-06 10:20:00 | 3063.34 | 3067.79 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:45:00 | 3074.10 | 3097.76 | 0.00 | ORB-short ORB[3082.40,3119.90] vol=2.1x ATR=9.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:00:00 | 3060.08 | 3093.08 | 0.00 | T1 1.5R @ 3060.08 |
| Stop hit — per-position SL triggered | 2025-06-11 11:35:00 | 3074.10 | 3089.46 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:00:00 | 2988.00 | 3012.89 | 0.00 | ORB-short ORB[3010.00,3051.90] vol=2.2x ATR=7.90 |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2995.90 | 3010.59 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:40:00 | 2968.90 | 2951.72 | 0.00 | ORB-long ORB[2936.50,2967.40] vol=2.2x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 10:45:00 | 2981.25 | 2955.74 | 0.00 | T1 1.5R @ 2981.25 |
| Target hit | 2025-06-16 14:55:00 | 2989.10 | 2989.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-06-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:10:00 | 2950.20 | 2977.30 | 0.00 | ORB-short ORB[2977.00,3020.00] vol=1.7x ATR=10.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 11:15:00 | 2934.78 | 2959.94 | 0.00 | T1 1.5R @ 2934.78 |
| Target hit | 2025-06-17 15:20:00 | 2904.70 | 2923.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:05:00 | 2870.20 | 2862.22 | 0.00 | ORB-long ORB[2840.20,2867.20] vol=2.0x ATR=10.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 10:30:00 | 2886.35 | 2866.39 | 0.00 | T1 1.5R @ 2886.35 |
| Target hit | 2025-06-20 15:20:00 | 2933.80 | 2919.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 3081.70 | 3092.89 | 0.00 | ORB-short ORB[3083.10,3114.00] vol=2.3x ATR=6.95 |
| Stop hit — per-position SL triggered | 2025-06-30 11:25:00 | 3088.65 | 3092.70 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:10:00 | 3063.80 | 3070.66 | 0.00 | ORB-short ORB[3067.30,3094.50] vol=2.3x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:40:00 | 3052.15 | 3067.56 | 0.00 | T1 1.5R @ 3052.15 |
| Target hit | 2025-07-02 15:20:00 | 2990.40 | 3023.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 11:15:00 | 2945.00 | 2964.81 | 0.00 | ORB-short ORB[2966.50,3002.30] vol=1.8x ATR=7.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:45:00 | 2933.13 | 2950.37 | 0.00 | T1 1.5R @ 2933.13 |
| Stop hit — per-position SL triggered | 2025-07-03 12:45:00 | 2945.00 | 2943.76 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:10:00 | 2954.80 | 2966.64 | 0.00 | ORB-short ORB[2958.00,2980.50] vol=2.9x ATR=8.27 |
| Stop hit — per-position SL triggered | 2025-07-04 10:25:00 | 2963.07 | 2966.40 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 3012.20 | 3002.33 | 0.00 | ORB-long ORB[2969.00,3007.10] vol=1.6x ATR=8.96 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 3003.24 | 3001.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 2938.60 | 2953.80 | 0.00 | ORB-short ORB[2947.70,2965.90] vol=1.5x ATR=9.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:55:00 | 2924.83 | 2945.43 | 0.00 | T1 1.5R @ 2924.83 |
| Target hit | 2025-07-08 12:35:00 | 2920.50 | 2912.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 2855.00 | 2866.95 | 0.00 | ORB-short ORB[2869.50,2908.00] vol=3.3x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 2842.88 | 2854.79 | 0.00 | T1 1.5R @ 2842.88 |
| Stop hit — per-position SL triggered | 2025-07-11 11:50:00 | 2855.00 | 2853.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:55:00 | 2899.60 | 2872.73 | 0.00 | ORB-long ORB[2868.00,2888.80] vol=1.9x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 12:20:00 | 2912.13 | 2879.24 | 0.00 | T1 1.5R @ 2912.13 |
| Target hit | 2025-07-15 15:20:00 | 2929.50 | 2901.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:00:00 | 2958.80 | 2939.52 | 0.00 | ORB-long ORB[2901.30,2941.50] vol=1.9x ATR=8.61 |
| Stop hit — per-position SL triggered | 2025-07-21 10:25:00 | 2950.19 | 2943.77 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 2987.20 | 2973.00 | 0.00 | ORB-long ORB[2943.80,2977.00] vol=2.6x ATR=7.50 |
| Stop hit — per-position SL triggered | 2025-07-22 09:35:00 | 2979.70 | 2974.75 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:40:00 | 2975.60 | 2960.16 | 0.00 | ORB-long ORB[2942.10,2967.60] vol=4.9x ATR=7.61 |
| Stop hit — per-position SL triggered | 2025-07-23 10:55:00 | 2967.99 | 2961.69 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:10:00 | 2901.80 | 2925.52 | 0.00 | ORB-short ORB[2920.30,2963.20] vol=1.9x ATR=10.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:20:00 | 2886.55 | 2907.71 | 0.00 | T1 1.5R @ 2886.55 |
| Target hit | 2025-07-25 15:20:00 | 2874.80 | 2901.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-07-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:45:00 | 2831.10 | 2812.94 | 0.00 | ORB-long ORB[2795.30,2823.50] vol=1.7x ATR=9.75 |
| Stop hit — per-position SL triggered | 2025-07-29 10:20:00 | 2821.35 | 2819.10 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 2891.20 | 2881.59 | 0.00 | ORB-long ORB[2850.00,2885.60] vol=7.2x ATR=10.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 09:55:00 | 2907.61 | 2882.10 | 0.00 | T1 1.5R @ 2907.61 |
| Stop hit — per-position SL triggered | 2025-07-30 12:20:00 | 2891.20 | 2885.81 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:10:00 | 2818.50 | 2827.29 | 0.00 | ORB-short ORB[2822.10,2855.40] vol=2.9x ATR=9.63 |
| Stop hit — per-position SL triggered | 2025-07-31 10:55:00 | 2828.13 | 2824.31 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 2871.50 | 2881.28 | 0.00 | ORB-short ORB[2888.20,2926.80] vol=14.4x ATR=11.28 |
| Stop hit — per-position SL triggered | 2025-08-06 09:50:00 | 2882.78 | 2880.66 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 2926.30 | 2914.53 | 0.00 | ORB-long ORB[2878.50,2912.60] vol=3.2x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 13:00:00 | 2936.15 | 2916.66 | 0.00 | T1 1.5R @ 2936.15 |
| Target hit | 2025-08-11 15:20:00 | 3000.40 | 2968.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-08-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:00:00 | 3044.00 | 3057.94 | 0.00 | ORB-short ORB[3050.20,3072.50] vol=2.2x ATR=6.43 |
| Stop hit — per-position SL triggered | 2025-08-14 11:05:00 | 3050.43 | 3057.66 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 3094.90 | 3086.52 | 0.00 | ORB-long ORB[3049.80,3093.60] vol=1.5x ATR=10.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:00:00 | 3110.83 | 3091.37 | 0.00 | T1 1.5R @ 3110.83 |
| Target hit | 2025-08-19 15:20:00 | 3150.50 | 3138.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-08-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:05:00 | 3176.80 | 3155.75 | 0.00 | ORB-long ORB[3143.10,3167.30] vol=5.8x ATR=8.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 11:10:00 | 3189.00 | 3173.24 | 0.00 | T1 1.5R @ 3189.00 |
| Target hit | 2025-08-20 13:25:00 | 3201.10 | 3202.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 3056.10 | 3069.91 | 0.00 | ORB-short ORB[3067.90,3088.90] vol=2.8x ATR=9.05 |
| Stop hit — per-position SL triggered | 2025-08-25 10:10:00 | 3065.15 | 3067.31 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:25:00 | 3042.00 | 3056.03 | 0.00 | ORB-short ORB[3045.60,3068.00] vol=1.8x ATR=8.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:35:00 | 3028.72 | 3053.01 | 0.00 | T1 1.5R @ 3028.72 |
| Target hit | 2025-09-03 14:30:00 | 3033.50 | 3028.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:30:00 | 3032.40 | 3023.04 | 0.00 | ORB-long ORB[3001.70,3025.40] vol=2.6x ATR=10.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:35:00 | 3047.42 | 3031.99 | 0.00 | T1 1.5R @ 3047.42 |
| Stop hit — per-position SL triggered | 2025-09-05 09:40:00 | 3032.40 | 3032.07 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 3410.10 | 3392.86 | 0.00 | ORB-long ORB[3366.00,3403.70] vol=1.5x ATR=11.38 |
| Stop hit — per-position SL triggered | 2025-09-17 09:45:00 | 3398.72 | 3394.39 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 3368.00 | 3377.59 | 0.00 | ORB-short ORB[3372.50,3403.70] vol=2.1x ATR=8.55 |
| Stop hit — per-position SL triggered | 2025-09-18 14:10:00 | 3376.55 | 3369.58 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:00:00 | 3231.80 | 3254.29 | 0.00 | ORB-short ORB[3245.90,3287.00] vol=1.7x ATR=10.95 |
| Stop hit — per-position SL triggered | 2025-09-25 10:10:00 | 3242.75 | 3244.57 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:50:00 | 3163.40 | 3178.41 | 0.00 | ORB-short ORB[3183.00,3210.90] vol=1.5x ATR=10.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:05:00 | 3148.29 | 3175.83 | 0.00 | T1 1.5R @ 3148.29 |
| Target hit | 2025-09-26 15:20:00 | 3135.00 | 3142.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:55:00 | 3101.30 | 3108.84 | 0.00 | ORB-short ORB[3108.60,3138.70] vol=1.7x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:10:00 | 3092.51 | 3107.37 | 0.00 | T1 1.5R @ 3092.51 |
| Target hit | 2025-09-30 13:25:00 | 3098.50 | 3098.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2025-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:55:00 | 3168.40 | 3135.89 | 0.00 | ORB-long ORB[3119.00,3151.60] vol=1.7x ATR=8.12 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 3160.28 | 3141.16 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:40:00 | 3200.00 | 3185.07 | 0.00 | ORB-long ORB[3150.00,3195.70] vol=2.8x ATR=8.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:15:00 | 3213.21 | 3195.83 | 0.00 | T1 1.5R @ 3213.21 |
| Stop hit — per-position SL triggered | 2025-10-07 10:20:00 | 3200.00 | 3197.40 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:10:00 | 3182.70 | 3209.08 | 0.00 | ORB-short ORB[3209.00,3241.00] vol=1.7x ATR=8.57 |
| Stop hit — per-position SL triggered | 2025-10-08 10:35:00 | 3191.27 | 3202.27 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 3112.00 | 3127.26 | 0.00 | ORB-short ORB[3121.10,3158.90] vol=2.8x ATR=10.13 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 3122.13 | 3126.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:10:00 | 3119.00 | 3109.11 | 0.00 | ORB-long ORB[3088.20,3106.00] vol=1.7x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 3128.01 | 3111.84 | 0.00 | T1 1.5R @ 3128.01 |
| Target hit | 2025-10-15 15:20:00 | 3159.30 | 3138.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 3193.20 | 3179.92 | 0.00 | ORB-long ORB[3159.10,3185.00] vol=1.5x ATR=8.76 |
| Stop hit — per-position SL triggered | 2025-10-16 09:50:00 | 3184.44 | 3184.87 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:10:00 | 3184.80 | 3154.19 | 0.00 | ORB-long ORB[3113.10,3157.30] vol=4.0x ATR=10.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:15:00 | 3201.06 | 3164.46 | 0.00 | T1 1.5R @ 3201.06 |
| Stop hit — per-position SL triggered | 2025-10-23 11:05:00 | 3184.80 | 3175.07 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 3151.10 | 3139.95 | 0.00 | ORB-long ORB[3112.60,3145.90] vol=3.1x ATR=7.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:45:00 | 3162.60 | 3146.99 | 0.00 | T1 1.5R @ 3162.60 |
| Stop hit — per-position SL triggered | 2025-10-28 10:05:00 | 3151.10 | 3151.89 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 3145.00 | 3129.85 | 0.00 | ORB-long ORB[3111.00,3136.50] vol=2.1x ATR=7.45 |
| Stop hit — per-position SL triggered | 2025-10-29 10:50:00 | 3137.55 | 3133.78 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:40:00 | 3124.20 | 3136.42 | 0.00 | ORB-short ORB[3137.30,3169.80] vol=2.2x ATR=8.22 |
| Stop hit — per-position SL triggered | 2025-10-30 11:10:00 | 3132.42 | 3134.08 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:40:00 | 3075.50 | 3070.81 | 0.00 | ORB-long ORB[3050.00,3070.10] vol=1.5x ATR=9.31 |
| Stop hit — per-position SL triggered | 2025-10-31 10:10:00 | 3066.19 | 3071.42 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:50:00 | 3031.30 | 3020.56 | 0.00 | ORB-long ORB[2994.50,3023.90] vol=2.4x ATR=10.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:10:00 | 3047.20 | 3027.84 | 0.00 | T1 1.5R @ 3047.20 |
| Stop hit — per-position SL triggered | 2025-11-03 10:20:00 | 3031.30 | 3028.34 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 3032.00 | 3005.30 | 0.00 | ORB-long ORB[2972.10,3015.00] vol=1.5x ATR=13.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:50:00 | 3052.20 | 3024.49 | 0.00 | T1 1.5R @ 3052.20 |
| Stop hit — per-position SL triggered | 2025-11-06 10:05:00 | 3032.00 | 3026.15 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:50:00 | 2926.40 | 2933.37 | 0.00 | ORB-short ORB[2940.00,2977.40] vol=2.5x ATR=11.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:00:00 | 2909.56 | 2928.69 | 0.00 | T1 1.5R @ 2909.56 |
| Stop hit — per-position SL triggered | 2025-11-07 10:35:00 | 2926.40 | 2927.46 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:20:00 | 2958.00 | 2976.29 | 0.00 | ORB-short ORB[2973.40,3000.00] vol=1.6x ATR=9.73 |
| Stop hit — per-position SL triggered | 2025-11-11 10:25:00 | 2967.73 | 2975.48 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 3023.00 | 2996.92 | 0.00 | ORB-long ORB[2975.20,3010.10] vol=1.9x ATR=9.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:30:00 | 3037.23 | 3004.42 | 0.00 | T1 1.5R @ 3037.23 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 3023.00 | 3020.91 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 3057.40 | 3040.90 | 0.00 | ORB-long ORB[3024.60,3044.50] vol=2.5x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-11-13 09:35:00 | 3048.21 | 3041.92 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:55:00 | 3039.10 | 3046.45 | 0.00 | ORB-short ORB[3050.70,3082.60] vol=1.5x ATR=7.20 |
| Stop hit — per-position SL triggered | 2025-11-18 11:55:00 | 3046.30 | 3048.22 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 2954.60 | 2966.24 | 0.00 | ORB-short ORB[2962.80,3000.00] vol=3.2x ATR=9.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:55:00 | 2940.35 | 2956.15 | 0.00 | T1 1.5R @ 2940.35 |
| Target hit | 2025-11-21 15:20:00 | 2879.20 | 2926.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:45:00 | 2772.70 | 2783.41 | 0.00 | ORB-short ORB[2777.50,2793.40] vol=2.1x ATR=6.96 |
| Stop hit — per-position SL triggered | 2025-12-02 10:40:00 | 2779.66 | 2779.99 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:45:00 | 2665.60 | 2674.92 | 0.00 | ORB-short ORB[2672.80,2710.50] vol=4.2x ATR=6.45 |
| Stop hit — per-position SL triggered | 2025-12-05 12:20:00 | 2672.05 | 2667.41 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 11:15:00 | 2593.00 | 2614.65 | 0.00 | ORB-short ORB[2600.00,2625.80] vol=3.4x ATR=8.08 |
| Stop hit — per-position SL triggered | 2025-12-11 11:35:00 | 2601.08 | 2613.54 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 2657.80 | 2629.18 | 0.00 | ORB-long ORB[2604.50,2639.90] vol=3.0x ATR=8.65 |
| Stop hit — per-position SL triggered | 2025-12-17 09:45:00 | 2649.15 | 2635.25 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:15:00 | 2593.00 | 2600.52 | 0.00 | ORB-short ORB[2599.20,2629.10] vol=3.0x ATR=7.37 |
| Stop hit — per-position SL triggered | 2025-12-18 10:45:00 | 2600.37 | 2599.74 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:25:00 | 2635.60 | 2618.42 | 0.00 | ORB-long ORB[2610.30,2620.00] vol=2.1x ATR=7.17 |
| Stop hit — per-position SL triggered | 2026-01-01 12:20:00 | 2628.43 | 2629.39 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:45:00 | 2603.10 | 2611.45 | 0.00 | ORB-short ORB[2609.40,2630.30] vol=2.4x ATR=6.34 |
| Stop hit — per-position SL triggered | 2026-01-02 10:00:00 | 2609.44 | 2611.05 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:10:00 | 2553.00 | 2569.35 | 0.00 | ORB-short ORB[2585.00,2614.90] vol=2.6x ATR=6.69 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 2559.69 | 2567.08 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:00:00 | 2551.70 | 2545.60 | 0.00 | ORB-long ORB[2523.20,2541.20] vol=3.1x ATR=7.40 |
| Stop hit — per-position SL triggered | 2026-01-06 10:20:00 | 2544.30 | 2545.83 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 2560.10 | 2553.54 | 0.00 | ORB-long ORB[2541.00,2559.50] vol=1.9x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:15:00 | 2570.57 | 2555.86 | 0.00 | T1 1.5R @ 2570.57 |
| Stop hit — per-position SL triggered | 2026-01-07 10:30:00 | 2560.10 | 2556.31 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 2514.60 | 2530.92 | 0.00 | ORB-short ORB[2521.30,2548.50] vol=1.7x ATR=6.62 |
| Stop hit — per-position SL triggered | 2026-01-08 11:40:00 | 2521.22 | 2525.55 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:30:00 | 2358.30 | 2362.75 | 0.00 | ORB-short ORB[2370.50,2387.30] vol=3.1x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:45:00 | 2348.75 | 2360.21 | 0.00 | T1 1.5R @ 2348.75 |
| Stop hit — per-position SL triggered | 2026-01-14 11:55:00 | 2358.30 | 2350.55 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 09:50:00 | 2287.20 | 2305.36 | 0.00 | ORB-short ORB[2306.30,2330.40] vol=1.9x ATR=10.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:40:00 | 2272.16 | 2289.83 | 0.00 | T1 1.5R @ 2272.16 |
| Target hit | 2026-01-21 12:00:00 | 2280.50 | 2277.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2026-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:55:00 | 2348.20 | 2323.28 | 0.00 | ORB-long ORB[2298.50,2326.00] vol=2.2x ATR=6.43 |
| Stop hit — per-position SL triggered | 2026-01-30 11:25:00 | 2341.77 | 2327.12 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:40:00 | 2426.40 | 2394.12 | 0.00 | ORB-long ORB[2333.80,2363.80] vol=2.9x ATR=11.80 |
| Stop hit — per-position SL triggered | 2026-02-01 10:45:00 | 2414.60 | 2395.97 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:30:00 | 2394.40 | 2378.12 | 0.00 | ORB-long ORB[2356.50,2391.10] vol=1.8x ATR=10.23 |
| Stop hit — per-position SL triggered | 2026-02-10 10:35:00 | 2384.17 | 2378.38 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 2460.00 | 2437.08 | 0.00 | ORB-long ORB[2414.20,2446.00] vol=3.3x ATR=7.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:00:00 | 2471.24 | 2443.82 | 0.00 | T1 1.5R @ 2471.24 |
| Target hit | 2026-02-12 15:20:00 | 2506.00 | 2485.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 2512.30 | 2488.53 | 0.00 | ORB-long ORB[2477.20,2496.00] vol=1.8x ATR=9.36 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 2502.94 | 2489.27 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 2434.00 | 2443.53 | 0.00 | ORB-short ORB[2445.00,2477.60] vol=4.5x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 2423.21 | 2439.19 | 0.00 | T1 1.5R @ 2423.21 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 2434.00 | 2437.47 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 2556.30 | 2579.42 | 0.00 | ORB-short ORB[2570.00,2599.10] vol=2.0x ATR=11.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:45:00 | 2539.42 | 2575.96 | 0.00 | T1 1.5R @ 2539.42 |
| Target hit | 2026-02-23 15:20:00 | 2540.10 | 2559.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 2612.60 | 2595.67 | 0.00 | ORB-long ORB[2575.00,2610.00] vol=4.4x ATR=7.79 |
| Stop hit — per-position SL triggered | 2026-02-25 12:35:00 | 2604.81 | 2602.52 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 2661.30 | 2635.03 | 0.00 | ORB-long ORB[2606.50,2617.90] vol=4.2x ATR=10.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:25:00 | 2676.97 | 2649.88 | 0.00 | T1 1.5R @ 2676.97 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 2661.30 | 2652.21 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-03-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:55:00 | 2414.00 | 2430.48 | 0.00 | ORB-short ORB[2433.30,2459.10] vol=3.1x ATR=10.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 2397.98 | 2424.89 | 0.00 | T1 1.5R @ 2397.98 |
| Target hit | 2026-03-16 11:15:00 | 2396.10 | 2391.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 82 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 2515.30 | 2479.17 | 0.00 | ORB-long ORB[2447.60,2478.60] vol=4.0x ATR=10.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:20:00 | 2530.37 | 2495.59 | 0.00 | T1 1.5R @ 2530.37 |
| Target hit | 2026-03-18 15:20:00 | 2556.00 | 2527.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 2737.80 | 2750.38 | 0.00 | ORB-short ORB[2742.60,2768.50] vol=7.0x ATR=8.80 |
| Stop hit — per-position SL triggered | 2026-04-15 11:20:00 | 2746.60 | 2749.96 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 2739.10 | 2750.85 | 0.00 | ORB-short ORB[2744.10,2782.40] vol=1.6x ATR=11.41 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 2750.51 | 2749.63 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 2961.80 | 2966.43 | 0.00 | ORB-short ORB[2970.00,3004.30] vol=5.1x ATR=8.96 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 2970.76 | 2972.31 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-05-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:25:00 | 2928.80 | 2939.81 | 0.00 | ORB-short ORB[2956.00,2978.60] vol=1.6x ATR=11.23 |
| Stop hit — per-position SL triggered | 2026-05-04 10:50:00 | 2940.03 | 2939.37 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-05-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:30:00 | 2898.50 | 2910.71 | 0.00 | ORB-short ORB[2905.10,2920.00] vol=1.9x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 2906.78 | 2907.04 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 2980.30 | 2965.94 | 0.00 | ORB-long ORB[2928.50,2965.00] vol=2.0x ATR=14.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 13:35:00 | 3001.43 | 2978.40 | 0.00 | T1 1.5R @ 3001.43 |
| Target hit | 2026-05-07 15:20:00 | 3021.90 | 2992.04 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-21 09:30:00 | 3000.40 | 2025-05-21 09:55:00 | 2979.94 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-05-21 09:30:00 | 3000.40 | 2025-05-21 10:25:00 | 3000.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-22 10:15:00 | 3033.00 | 2025-05-22 10:25:00 | 3021.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-26 10:15:00 | 3037.60 | 2025-05-26 11:35:00 | 3020.84 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-05-26 10:15:00 | 3037.60 | 2025-05-26 11:40:00 | 3037.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 10:30:00 | 3010.10 | 2025-05-27 11:25:00 | 3020.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-30 09:55:00 | 3085.80 | 2025-05-30 10:05:00 | 3076.51 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-06 10:15:00 | 3055.10 | 2025-06-06 10:20:00 | 3063.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-11 10:45:00 | 3074.10 | 2025-06-11 11:00:00 | 3060.08 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-06-11 10:45:00 | 3074.10 | 2025-06-11 11:35:00 | 3074.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 11:00:00 | 2988.00 | 2025-06-12 11:15:00 | 2995.90 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-16 10:40:00 | 2968.90 | 2025-06-16 10:45:00 | 2981.25 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-16 10:40:00 | 2968.90 | 2025-06-16 14:55:00 | 2989.10 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-06-17 10:10:00 | 2950.20 | 2025-06-17 11:15:00 | 2934.78 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-17 10:10:00 | 2950.20 | 2025-06-17 15:20:00 | 2904.70 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2025-06-20 10:05:00 | 2870.20 | 2025-06-20 10:30:00 | 2886.35 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-20 10:05:00 | 2870.20 | 2025-06-20 15:20:00 | 2933.80 | TARGET_HIT | 0.50 | 2.22% |
| SELL | retest1 | 2025-06-30 11:15:00 | 3081.70 | 2025-06-30 11:25:00 | 3088.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-02 10:10:00 | 3063.80 | 2025-07-02 10:40:00 | 3052.15 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-02 10:10:00 | 3063.80 | 2025-07-02 15:20:00 | 2990.40 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2025-07-03 11:15:00 | 2945.00 | 2025-07-03 11:45:00 | 2933.13 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-03 11:15:00 | 2945.00 | 2025-07-03 12:45:00 | 2945.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 10:10:00 | 2954.80 | 2025-07-04 10:25:00 | 2963.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-07 09:35:00 | 3012.20 | 2025-07-07 09:40:00 | 3003.24 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-08 09:35:00 | 2938.60 | 2025-07-08 09:55:00 | 2924.83 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-08 09:35:00 | 2938.60 | 2025-07-08 12:35:00 | 2920.50 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2025-07-11 09:40:00 | 2855.00 | 2025-07-11 11:00:00 | 2842.88 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-11 09:40:00 | 2855.00 | 2025-07-11 11:50:00 | 2855.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:55:00 | 2899.60 | 2025-07-15 12:20:00 | 2912.13 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-07-15 10:55:00 | 2899.60 | 2025-07-15 15:20:00 | 2929.50 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2025-07-21 10:00:00 | 2958.80 | 2025-07-21 10:25:00 | 2950.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-22 09:30:00 | 2987.20 | 2025-07-22 09:35:00 | 2979.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-23 10:40:00 | 2975.60 | 2025-07-23 10:55:00 | 2967.99 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-25 10:10:00 | 2901.80 | 2025-07-25 14:20:00 | 2886.55 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-25 10:10:00 | 2901.80 | 2025-07-25 15:20:00 | 2874.80 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2025-07-29 09:45:00 | 2831.10 | 2025-07-29 10:20:00 | 2821.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-30 09:45:00 | 2891.20 | 2025-07-30 09:55:00 | 2907.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-07-30 09:45:00 | 2891.20 | 2025-07-30 12:20:00 | 2891.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-31 10:10:00 | 2818.50 | 2025-07-31 10:55:00 | 2828.13 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-06 09:30:00 | 2871.50 | 2025-08-06 09:50:00 | 2882.78 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-11 11:10:00 | 2926.30 | 2025-08-11 13:00:00 | 2936.15 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-11 11:10:00 | 2926.30 | 2025-08-11 15:20:00 | 3000.40 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2025-08-14 11:00:00 | 3044.00 | 2025-08-14 11:05:00 | 3050.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-19 09:50:00 | 3094.90 | 2025-08-19 10:00:00 | 3110.83 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-19 09:50:00 | 3094.90 | 2025-08-19 15:20:00 | 3150.50 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2025-08-20 11:05:00 | 3176.80 | 2025-08-20 11:10:00 | 3189.00 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-08-20 11:05:00 | 3176.80 | 2025-08-20 13:25:00 | 3201.10 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2025-08-25 09:50:00 | 3056.10 | 2025-08-25 10:10:00 | 3065.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-03 10:25:00 | 3042.00 | 2025-09-03 10:35:00 | 3028.72 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-03 10:25:00 | 3042.00 | 2025-09-03 14:30:00 | 3033.50 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-05 09:30:00 | 3032.40 | 2025-09-05 09:35:00 | 3047.42 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-09-05 09:30:00 | 3032.40 | 2025-09-05 09:40:00 | 3032.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 09:35:00 | 3410.10 | 2025-09-17 09:45:00 | 3398.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-18 09:35:00 | 3368.00 | 2025-09-18 14:10:00 | 3376.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-25 10:00:00 | 3231.80 | 2025-09-25 10:10:00 | 3242.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-26 10:50:00 | 3163.40 | 2025-09-26 11:05:00 | 3148.29 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-26 10:50:00 | 3163.40 | 2025-09-26 15:20:00 | 3135.00 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-09-30 10:55:00 | 3101.30 | 2025-09-30 11:10:00 | 3092.51 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-30 10:55:00 | 3101.30 | 2025-09-30 13:25:00 | 3098.50 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-10-06 10:55:00 | 3168.40 | 2025-10-06 11:15:00 | 3160.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-07 09:40:00 | 3200.00 | 2025-10-07 10:15:00 | 3213.21 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-10-07 09:40:00 | 3200.00 | 2025-10-07 10:20:00 | 3200.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:10:00 | 3182.70 | 2025-10-08 10:35:00 | 3191.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-13 09:30:00 | 3112.00 | 2025-10-13 09:35:00 | 3122.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-15 10:10:00 | 3119.00 | 2025-10-15 10:15:00 | 3128.01 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-15 10:10:00 | 3119.00 | 2025-10-15 15:20:00 | 3159.30 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-10-16 09:35:00 | 3193.20 | 2025-10-16 09:50:00 | 3184.44 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-23 10:10:00 | 3184.80 | 2025-10-23 10:15:00 | 3201.06 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-23 10:10:00 | 3184.80 | 2025-10-23 11:05:00 | 3184.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:30:00 | 3151.10 | 2025-10-28 09:45:00 | 3162.60 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-28 09:30:00 | 3151.10 | 2025-10-28 10:05:00 | 3151.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:25:00 | 3145.00 | 2025-10-29 10:50:00 | 3137.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-30 10:40:00 | 3124.20 | 2025-10-30 11:10:00 | 3132.42 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-31 09:40:00 | 3075.50 | 2025-10-31 10:10:00 | 3066.19 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-03 09:50:00 | 3031.30 | 2025-11-03 10:10:00 | 3047.20 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-03 09:50:00 | 3031.30 | 2025-11-03 10:20:00 | 3031.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-06 09:30:00 | 3032.00 | 2025-11-06 09:50:00 | 3052.20 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-11-06 09:30:00 | 3032.00 | 2025-11-06 10:05:00 | 3032.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:50:00 | 2926.40 | 2025-11-07 10:00:00 | 2909.56 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-07 09:50:00 | 2926.40 | 2025-11-07 10:35:00 | 2926.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:20:00 | 2958.00 | 2025-11-11 10:25:00 | 2967.73 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-12 10:00:00 | 3023.00 | 2025-11-12 10:30:00 | 3037.23 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-11-12 10:00:00 | 3023.00 | 2025-11-12 13:15:00 | 3023.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:30:00 | 3057.40 | 2025-11-13 09:35:00 | 3048.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-18 10:55:00 | 3039.10 | 2025-11-18 11:55:00 | 3046.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-21 09:30:00 | 2954.60 | 2025-11-21 09:55:00 | 2940.35 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-11-21 09:30:00 | 2954.60 | 2025-11-21 15:20:00 | 2879.20 | TARGET_HIT | 0.50 | 2.55% |
| SELL | retest1 | 2025-12-02 09:45:00 | 2772.70 | 2025-12-02 10:40:00 | 2779.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-05 09:45:00 | 2665.60 | 2025-12-05 12:20:00 | 2672.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-11 11:15:00 | 2593.00 | 2025-12-11 11:35:00 | 2601.08 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-17 09:30:00 | 2657.80 | 2025-12-17 09:45:00 | 2649.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-18 10:15:00 | 2593.00 | 2025-12-18 10:45:00 | 2600.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-01 10:25:00 | 2635.60 | 2026-01-01 12:20:00 | 2628.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-02 09:45:00 | 2603.10 | 2026-01-02 10:00:00 | 2609.44 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-05 10:10:00 | 2553.00 | 2026-01-05 10:15:00 | 2559.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-06 10:00:00 | 2551.70 | 2026-01-06 10:20:00 | 2544.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-07 10:05:00 | 2560.10 | 2026-01-07 10:15:00 | 2570.57 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-07 10:05:00 | 2560.10 | 2026-01-07 10:30:00 | 2560.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:15:00 | 2514.60 | 2026-01-08 11:40:00 | 2521.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-14 10:30:00 | 2358.30 | 2026-01-14 10:45:00 | 2348.75 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-14 10:30:00 | 2358.30 | 2026-01-14 11:55:00 | 2358.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 09:50:00 | 2287.20 | 2026-01-21 10:40:00 | 2272.16 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-01-21 09:50:00 | 2287.20 | 2026-01-21 12:00:00 | 2280.50 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-30 10:55:00 | 2348.20 | 2026-01-30 11:25:00 | 2341.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-01 10:40:00 | 2426.40 | 2026-02-01 10:45:00 | 2414.60 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-02-10 10:30:00 | 2394.40 | 2026-02-10 10:35:00 | 2384.17 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-12 10:50:00 | 2460.00 | 2026-02-12 11:00:00 | 2471.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-12 10:50:00 | 2460.00 | 2026-02-12 15:20:00 | 2506.00 | TARGET_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2026-02-17 09:35:00 | 2512.30 | 2026-02-17 09:40:00 | 2502.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 2434.00 | 2026-02-18 11:10:00 | 2423.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-18 10:55:00 | 2434.00 | 2026-02-18 11:25:00 | 2434.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 2556.30 | 2026-02-23 11:45:00 | 2539.42 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-23 11:00:00 | 2556.30 | 2026-02-23 15:20:00 | 2540.10 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-25 11:05:00 | 2612.60 | 2026-02-25 12:35:00 | 2604.81 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-26 10:20:00 | 2661.30 | 2026-02-26 10:25:00 | 2676.97 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-26 10:20:00 | 2661.30 | 2026-02-26 10:30:00 | 2661.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 09:55:00 | 2414.00 | 2026-03-16 10:15:00 | 2397.98 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-16 09:55:00 | 2414.00 | 2026-03-16 11:15:00 | 2396.10 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2515.30 | 2026-03-18 11:20:00 | 2530.37 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2515.30 | 2026-03-18 15:20:00 | 2556.00 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2026-04-15 10:50:00 | 2737.80 | 2026-04-15 11:20:00 | 2746.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-17 09:50:00 | 2739.10 | 2026-04-17 09:55:00 | 2750.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-28 10:10:00 | 2961.80 | 2026-04-28 10:15:00 | 2970.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-04 10:25:00 | 2928.80 | 2026-05-04 10:50:00 | 2940.03 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-05 10:30:00 | 2898.50 | 2026-05-05 11:35:00 | 2906.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-07 10:10:00 | 2980.30 | 2026-05-07 13:35:00 | 3001.43 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-05-07 10:10:00 | 2980.30 | 2026-05-07 15:20:00 | 3021.90 | TARGET_HIT | 0.50 | 1.40% |
