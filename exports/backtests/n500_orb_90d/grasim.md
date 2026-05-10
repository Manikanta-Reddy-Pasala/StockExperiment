# Grasim Industries Ltd. (GRASIM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2965.00
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
| ENTRY1 | 26 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 22
- **Target hits / Stop hits / Partials:** 4 / 22 / 10
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 2.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.05% | 1.2% |
| BUY @ 2nd Alert (retest1) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.05% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.14% | 1.5% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.14% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 36 | 14 | 38.9% | 4 | 22 | 10 | 0.07% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 2895.60 | 2871.88 | 0.00 | ORB-long ORB[2846.50,2868.50] vol=3.4x ATR=9.07 |
| Stop hit — per-position SL triggered | 2026-02-09 11:40:00 | 2886.53 | 2880.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 2911.00 | 2918.64 | 0.00 | ORB-short ORB[2911.20,2934.00] vol=1.5x ATR=6.98 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 2917.98 | 2918.12 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 2891.70 | 2887.56 | 0.00 | ORB-long ORB[2861.30,2878.10] vol=2.8x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:30:00 | 2900.80 | 2891.25 | 0.00 | T1 1.5R @ 2900.80 |
| Target hit | 2026-02-16 15:20:00 | 2911.00 | 2900.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 2920.00 | 2909.72 | 0.00 | ORB-long ORB[2900.00,2913.20] vol=2.6x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:55:00 | 2927.44 | 2912.08 | 0.00 | T1 1.5R @ 2927.44 |
| Stop hit — per-position SL triggered | 2026-02-18 13:05:00 | 2920.00 | 2923.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 2914.50 | 2934.96 | 0.00 | ORB-short ORB[2927.90,2946.50] vol=2.8x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:35:00 | 2906.21 | 2932.69 | 0.00 | T1 1.5R @ 2906.21 |
| Target hit | 2026-02-19 15:20:00 | 2857.20 | 2892.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 2849.00 | 2859.51 | 0.00 | ORB-short ORB[2850.20,2885.00] vol=1.6x ATR=7.10 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 2856.10 | 2856.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 2909.00 | 2901.45 | 0.00 | ORB-long ORB[2883.50,2901.20] vol=5.9x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 2904.20 | 2902.46 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 2889.30 | 2882.55 | 0.00 | ORB-long ORB[2873.20,2887.20] vol=1.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 2884.52 | 2882.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 2808.30 | 2831.35 | 0.00 | ORB-short ORB[2839.10,2864.60] vol=1.9x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:45:00 | 2798.33 | 2823.42 | 0.00 | T1 1.5R @ 2798.33 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 2808.30 | 2812.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 2720.00 | 2716.29 | 0.00 | ORB-long ORB[2687.20,2718.40] vol=1.9x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:15:00 | 2732.19 | 2718.15 | 0.00 | T1 1.5R @ 2732.19 |
| Target hit | 2026-03-06 14:10:00 | 2738.10 | 2739.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 2725.00 | 2712.83 | 0.00 | ORB-long ORB[2701.00,2724.10] vol=2.2x ATR=8.89 |
| Stop hit — per-position SL triggered | 2026-03-10 10:55:00 | 2716.11 | 2714.67 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 2749.80 | 2742.56 | 0.00 | ORB-long ORB[2726.60,2743.90] vol=4.3x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:10:00 | 2758.83 | 2750.14 | 0.00 | T1 1.5R @ 2758.83 |
| Stop hit — per-position SL triggered | 2026-03-11 12:05:00 | 2749.80 | 2751.44 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 2636.80 | 2640.52 | 0.00 | ORB-short ORB[2644.00,2664.00] vol=2.7x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:40:00 | 2625.02 | 2638.77 | 0.00 | T1 1.5R @ 2625.02 |
| Target hit | 2026-03-13 10:50:00 | 2634.10 | 2632.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 2718.10 | 2712.00 | 0.00 | ORB-long ORB[2686.70,2713.00] vol=3.2x ATR=6.74 |
| Stop hit — per-position SL triggered | 2026-03-18 13:05:00 | 2711.36 | 2714.59 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 2535.00 | 2550.96 | 0.00 | ORB-short ORB[2565.50,2597.60] vol=2.1x ATR=10.61 |
| Stop hit — per-position SL triggered | 2026-03-24 11:35:00 | 2545.61 | 2545.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 2560.90 | 2574.34 | 0.00 | ORB-short ORB[2582.30,2620.10] vol=3.1x ATR=10.59 |
| Stop hit — per-position SL triggered | 2026-03-30 11:25:00 | 2571.49 | 2573.43 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 11:05:00 | 2539.50 | 2551.47 | 0.00 | ORB-short ORB[2542.40,2568.20] vol=1.7x ATR=8.16 |
| Stop hit — per-position SL triggered | 2026-04-06 11:25:00 | 2547.66 | 2548.90 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 2784.80 | 2779.12 | 0.00 | ORB-long ORB[2755.00,2780.00] vol=2.5x ATR=7.18 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 2777.62 | 2779.88 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 2798.90 | 2772.14 | 0.00 | ORB-long ORB[2763.10,2788.70] vol=1.9x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:50:00 | 2809.67 | 2778.43 | 0.00 | T1 1.5R @ 2809.67 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 2798.90 | 2782.72 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:35:00 | 2766.00 | 2757.49 | 0.00 | ORB-long ORB[2742.60,2762.30] vol=2.9x ATR=7.01 |
| Stop hit — per-position SL triggered | 2026-04-24 10:45:00 | 2758.99 | 2757.68 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 2782.00 | 2771.62 | 0.00 | ORB-long ORB[2752.20,2779.80] vol=3.5x ATR=8.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 13:15:00 | 2794.59 | 2777.58 | 0.00 | T1 1.5R @ 2794.59 |
| Stop hit — per-position SL triggered | 2026-04-27 14:25:00 | 2782.00 | 2782.76 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 2814.00 | 2802.48 | 0.00 | ORB-long ORB[2777.00,2799.00] vol=1.8x ATR=5.68 |
| Stop hit — per-position SL triggered | 2026-04-29 10:55:00 | 2808.32 | 2802.89 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-05-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:50:00 | 2851.80 | 2831.07 | 0.00 | ORB-long ORB[2805.00,2834.00] vol=2.1x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-05-04 10:55:00 | 2844.89 | 2831.59 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 2851.20 | 2848.24 | 0.00 | ORB-long ORB[2823.60,2845.80] vol=1.9x ATR=7.02 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 2844.18 | 2848.07 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 2964.60 | 2939.35 | 0.00 | ORB-long ORB[2900.00,2939.50] vol=2.1x ATR=9.79 |
| Stop hit — per-position SL triggered | 2026-05-07 10:00:00 | 2954.81 | 2950.80 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 2968.00 | 2955.84 | 0.00 | ORB-long ORB[2936.50,2961.40] vol=2.4x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:40:00 | 2980.36 | 2958.97 | 0.00 | T1 1.5R @ 2980.36 |
| Stop hit — per-position SL triggered | 2026-05-08 10:45:00 | 2968.00 | 2959.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 2895.60 | 2026-02-09 11:40:00 | 2886.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-13 10:50:00 | 2911.00 | 2026-02-13 11:00:00 | 2917.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 10:50:00 | 2891.70 | 2026-02-16 12:30:00 | 2900.80 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-16 10:50:00 | 2891.70 | 2026-02-16 15:20:00 | 2911.00 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-18 10:50:00 | 2920.00 | 2026-02-18 10:55:00 | 2927.44 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-02-18 10:50:00 | 2920.00 | 2026-02-18 13:05:00 | 2920.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 2914.50 | 2026-02-19 11:35:00 | 2906.21 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-19 11:15:00 | 2914.50 | 2026-02-19 15:20:00 | 2857.20 | TARGET_HIT | 0.50 | 1.97% |
| SELL | retest1 | 2026-02-24 10:45:00 | 2849.00 | 2026-02-24 11:00:00 | 2856.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-25 10:50:00 | 2909.00 | 2026-02-25 11:05:00 | 2904.20 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-02-26 10:50:00 | 2889.30 | 2026-02-26 10:55:00 | 2884.52 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-27 10:40:00 | 2808.30 | 2026-02-27 10:45:00 | 2798.33 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 10:40:00 | 2808.30 | 2026-02-27 11:05:00 | 2808.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 09:55:00 | 2720.00 | 2026-03-06 10:15:00 | 2732.19 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-06 09:55:00 | 2720.00 | 2026-03-06 14:10:00 | 2738.10 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-10 10:40:00 | 2725.00 | 2026-03-10 10:55:00 | 2716.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-11 10:35:00 | 2749.80 | 2026-03-11 11:10:00 | 2758.83 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-11 10:35:00 | 2749.80 | 2026-03-11 12:05:00 | 2749.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 2636.80 | 2026-03-13 10:40:00 | 2625.02 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-13 10:25:00 | 2636.80 | 2026-03-13 10:50:00 | 2634.10 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-03-18 11:15:00 | 2718.10 | 2026-03-18 13:05:00 | 2711.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-24 10:45:00 | 2535.00 | 2026-03-24 11:35:00 | 2545.61 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-30 11:10:00 | 2560.90 | 2026-03-30 11:25:00 | 2571.49 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-06 11:05:00 | 2539.50 | 2026-04-06 11:25:00 | 2547.66 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 09:35:00 | 2784.80 | 2026-04-21 09:50:00 | 2777.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-22 10:35:00 | 2798.90 | 2026-04-22 10:50:00 | 2809.67 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-22 10:35:00 | 2798.90 | 2026-04-22 11:05:00 | 2798.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 10:35:00 | 2766.00 | 2026-04-24 10:45:00 | 2758.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-27 10:30:00 | 2782.00 | 2026-04-27 13:15:00 | 2794.59 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-27 10:30:00 | 2782.00 | 2026-04-27 14:25:00 | 2782.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:50:00 | 2814.00 | 2026-04-29 10:55:00 | 2808.32 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-05-04 10:50:00 | 2851.80 | 2026-05-04 10:55:00 | 2844.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-05 11:00:00 | 2851.20 | 2026-05-05 11:15:00 | 2844.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-07 09:35:00 | 2964.60 | 2026-05-07 10:00:00 | 2954.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-08 10:30:00 | 2968.00 | 2026-05-08 10:40:00 | 2980.36 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-05-08 10:30:00 | 2968.00 | 2026-05-08 10:45:00 | 2968.00 | STOP_HIT | 0.50 | 0.00% |
