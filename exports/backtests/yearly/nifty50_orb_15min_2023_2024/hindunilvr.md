# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-11-29 15:25:00 (28792 bars)
- **Last close:** 2451.31
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
| ENTRY1 | 113 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 22 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 91
- **Target hits / Stop hits / Partials:** 22 / 91 / 41
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 11.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 23 | 34.8% | 6 | 43 | 17 | 0.04% | 2.3% |
| BUY @ 2nd Alert (retest1) | 66 | 23 | 34.8% | 6 | 43 | 17 | 0.04% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 88 | 40 | 45.5% | 16 | 48 | 24 | 0.11% | 9.3% |
| SELL @ 2nd Alert (retest1) | 88 | 40 | 45.5% | 16 | 48 | 24 | 0.11% | 9.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 154 | 63 | 40.9% | 22 | 91 | 41 | 0.08% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:10:00 | 2578.55 | 2558.49 | 0.00 | ORB-long ORB[2544.17,2561.48] vol=2.7x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 11:30:00 | 2586.47 | 2564.05 | 0.00 | T1 1.5R @ 2586.47 |
| Stop hit — per-position SL triggered | 2023-05-12 14:15:00 | 2578.55 | 2577.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:45:00 | 2608.89 | 2616.11 | 0.00 | ORB-short ORB[2614.60,2625.61] vol=2.9x ATR=5.39 |
| Stop hit — per-position SL triggered | 2023-05-18 09:50:00 | 2614.28 | 2615.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:55:00 | 2561.14 | 2569.84 | 0.00 | ORB-short ORB[2573.33,2593.15] vol=2.3x ATR=4.66 |
| Stop hit — per-position SL triggered | 2023-05-19 11:00:00 | 2565.80 | 2569.62 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 11:05:00 | 2605.45 | 2600.12 | 0.00 | ORB-long ORB[2580.22,2602.50] vol=2.2x ATR=5.52 |
| Stop hit — per-position SL triggered | 2023-05-22 11:10:00 | 2599.93 | 2600.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:15:00 | 2594.88 | 2582.44 | 0.00 | ORB-long ORB[2572.30,2584.64] vol=1.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-05-24 11:30:00 | 2590.75 | 2583.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:10:00 | 2558.82 | 2568.92 | 0.00 | ORB-short ORB[2567.38,2583.27] vol=2.8x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 12:15:00 | 2552.81 | 2565.88 | 0.00 | T1 1.5R @ 2552.81 |
| Target hit | 2023-05-25 14:55:00 | 2554.00 | 2552.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2023-06-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:35:00 | 2676.32 | 2668.86 | 0.00 | ORB-long ORB[2647.70,2671.65] vol=2.2x ATR=4.96 |
| Stop hit — per-position SL triggered | 2023-06-02 09:50:00 | 2671.36 | 2671.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:45:00 | 2644.16 | 2649.40 | 0.00 | ORB-short ORB[2646.17,2663.68] vol=2.3x ATR=4.31 |
| Stop hit — per-position SL triggered | 2023-06-06 10:50:00 | 2648.47 | 2648.98 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 11:10:00 | 2680.95 | 2665.51 | 0.00 | ORB-long ORB[2635.60,2670.67] vol=1.6x ATR=5.40 |
| Stop hit — per-position SL triggered | 2023-06-07 11:25:00 | 2675.55 | 2667.07 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 11:10:00 | 2652.57 | 2664.59 | 0.00 | ORB-short ORB[2654.24,2674.31] vol=1.7x ATR=4.98 |
| Stop hit — per-position SL triggered | 2023-06-08 11:20:00 | 2657.55 | 2663.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:40:00 | 2619.12 | 2630.87 | 0.00 | ORB-short ORB[2627.34,2645.04] vol=1.5x ATR=6.42 |
| Stop hit — per-position SL triggered | 2023-06-09 09:50:00 | 2625.54 | 2629.68 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:55:00 | 2641.94 | 2621.41 | 0.00 | ORB-long ORB[2608.70,2630.98] vol=1.6x ATR=7.29 |
| Stop hit — per-position SL triggered | 2023-06-14 11:15:00 | 2634.65 | 2623.15 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 11:05:00 | 2653.45 | 2662.87 | 0.00 | ORB-short ORB[2653.94,2673.13] vol=1.7x ATR=5.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 11:45:00 | 2644.83 | 2659.53 | 0.00 | T1 1.5R @ 2644.83 |
| Target hit | 2023-06-15 15:20:00 | 2648.04 | 2650.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 11:15:00 | 2668.31 | 2655.41 | 0.00 | ORB-long ORB[2644.40,2657.93] vol=1.6x ATR=5.34 |
| Stop hit — per-position SL triggered | 2023-06-16 11:20:00 | 2662.97 | 2655.65 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 10:50:00 | 2647.85 | 2652.30 | 0.00 | ORB-short ORB[2650.11,2667.13] vol=1.8x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 12:05:00 | 2640.98 | 2649.78 | 0.00 | T1 1.5R @ 2640.98 |
| Target hit | 2023-06-19 14:30:00 | 2646.32 | 2646.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2023-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:25:00 | 2606.83 | 2615.59 | 0.00 | ORB-short ORB[2615.48,2636.63] vol=1.7x ATR=5.10 |
| Stop hit — per-position SL triggered | 2023-06-20 11:25:00 | 2611.93 | 2613.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 11:05:00 | 2609.68 | 2617.50 | 0.00 | ORB-short ORB[2611.65,2632.89] vol=1.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2023-06-22 12:25:00 | 2613.64 | 2615.47 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 11:15:00 | 2594.92 | 2604.04 | 0.00 | ORB-short ORB[2597.19,2618.14] vol=2.4x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-06-27 11:20:00 | 2598.60 | 2603.95 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-06-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:50:00 | 2612.09 | 2622.91 | 0.00 | ORB-short ORB[2620.79,2636.24] vol=2.5x ATR=4.72 |
| Stop hit — per-position SL triggered | 2023-06-30 11:00:00 | 2616.81 | 2622.04 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 11:15:00 | 2622.42 | 2628.65 | 0.00 | ORB-short ORB[2628.52,2644.55] vol=2.1x ATR=4.35 |
| Stop hit — per-position SL triggered | 2023-07-03 13:10:00 | 2626.77 | 2623.75 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 10:45:00 | 2646.52 | 2647.19 | 0.00 | ORB-short ORB[2646.62,2659.99] vol=2.3x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 11:30:00 | 2637.88 | 2646.47 | 0.00 | T1 1.5R @ 2637.88 |
| Stop hit — per-position SL triggered | 2023-07-04 12:05:00 | 2646.52 | 2645.86 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:50:00 | 2690.34 | 2676.47 | 0.00 | ORB-long ORB[2652.08,2684.44] vol=2.2x ATR=6.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:55:00 | 2699.41 | 2679.74 | 0.00 | T1 1.5R @ 2699.41 |
| Stop hit — per-position SL triggered | 2023-07-05 11:00:00 | 2690.34 | 2680.36 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:05:00 | 2638.70 | 2629.14 | 0.00 | ORB-long ORB[2602.79,2623.35] vol=1.8x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:20:00 | 2646.88 | 2634.29 | 0.00 | T1 1.5R @ 2646.88 |
| Target hit | 2023-07-11 12:45:00 | 2646.47 | 2648.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2023-07-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:35:00 | 2629.70 | 2618.15 | 0.00 | ORB-long ORB[2607.76,2623.45] vol=1.6x ATR=4.85 |
| Stop hit — per-position SL triggered | 2023-07-14 10:45:00 | 2624.85 | 2619.31 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:05:00 | 2637.12 | 2645.14 | 0.00 | ORB-short ORB[2637.91,2656.40] vol=1.9x ATR=4.55 |
| Stop hit — per-position SL triggered | 2023-07-18 13:45:00 | 2641.67 | 2640.18 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:35:00 | 2629.40 | 2640.16 | 0.00 | ORB-short ORB[2638.25,2652.86] vol=1.5x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:40:00 | 2623.24 | 2638.47 | 0.00 | T1 1.5R @ 2623.24 |
| Target hit | 2023-07-19 12:55:00 | 2626.89 | 2626.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2023-07-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:55:00 | 2634.17 | 2626.85 | 0.00 | ORB-long ORB[2614.79,2632.45] vol=2.3x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 11:55:00 | 2642.21 | 2630.31 | 0.00 | T1 1.5R @ 2642.21 |
| Target hit | 2023-07-20 15:20:00 | 2658.62 | 2646.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2023-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:35:00 | 2542.40 | 2540.14 | 0.00 | ORB-long ORB[2530.30,2542.25] vol=1.6x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:25:00 | 2548.80 | 2542.07 | 0.00 | T1 1.5R @ 2548.80 |
| Target hit | 2023-07-26 12:35:00 | 2544.27 | 2545.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2023-08-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 11:05:00 | 2515.05 | 2503.87 | 0.00 | ORB-long ORB[2494.59,2509.34] vol=2.1x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-08-02 11:10:00 | 2511.11 | 2504.47 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 09:40:00 | 2499.31 | 2505.22 | 0.00 | ORB-short ORB[2499.70,2517.17] vol=1.5x ATR=4.40 |
| Stop hit — per-position SL triggered | 2023-08-04 11:10:00 | 2503.71 | 2502.83 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:55:00 | 2523.02 | 2525.82 | 0.00 | ORB-short ORB[2527.25,2542.99] vol=3.6x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 11:15:00 | 2517.75 | 2524.83 | 0.00 | T1 1.5R @ 2517.75 |
| Stop hit — per-position SL triggered | 2023-08-08 13:00:00 | 2523.02 | 2522.20 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:25:00 | 2493.61 | 2497.52 | 0.00 | ORB-short ORB[2500.74,2509.74] vol=4.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:30:00 | 2488.96 | 2495.69 | 0.00 | T1 1.5R @ 2488.96 |
| Stop hit — per-position SL triggered | 2023-08-10 10:35:00 | 2493.61 | 2495.63 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 10:00:00 | 2486.13 | 2466.37 | 0.00 | ORB-long ORB[2447.47,2465.18] vol=1.9x ATR=6.41 |
| Stop hit — per-position SL triggered | 2023-08-14 10:05:00 | 2479.72 | 2467.61 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 10:05:00 | 2502.46 | 2497.65 | 0.00 | ORB-long ORB[2483.18,2497.25] vol=3.7x ATR=4.91 |
| Stop hit — per-position SL triggered | 2023-08-16 10:50:00 | 2497.55 | 2499.64 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:55:00 | 2526.36 | 2512.73 | 0.00 | ORB-long ORB[2504.92,2516.18] vol=2.4x ATR=4.75 |
| Stop hit — per-position SL triggered | 2023-08-21 12:15:00 | 2521.61 | 2518.78 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:15:00 | 2498.52 | 2505.50 | 0.00 | ORB-short ORB[2502.71,2517.12] vol=1.6x ATR=3.47 |
| Stop hit — per-position SL triggered | 2023-08-29 10:30:00 | 2501.99 | 2504.01 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-08-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:25:00 | 2494.54 | 2489.58 | 0.00 | ORB-long ORB[2478.85,2492.28] vol=2.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2023-08-30 10:35:00 | 2491.26 | 2489.75 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 11:10:00 | 2455.24 | 2458.87 | 0.00 | ORB-short ORB[2459.28,2472.95] vol=1.7x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 2457.69 | 2458.80 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:35:00 | 2463.11 | 2467.25 | 0.00 | ORB-short ORB[2465.08,2480.67] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-09-07 11:15:00 | 2465.98 | 2466.64 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:50:00 | 2487.06 | 2482.69 | 0.00 | ORB-long ORB[2468.08,2483.62] vol=1.7x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 11:25:00 | 2492.02 | 2484.15 | 0.00 | T1 1.5R @ 2492.02 |
| Stop hit — per-position SL triggered | 2023-09-11 13:25:00 | 2487.06 | 2487.59 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:55:00 | 2467.88 | 2476.64 | 0.00 | ORB-short ORB[2471.03,2507.38] vol=2.0x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 10:20:00 | 2460.53 | 2471.89 | 0.00 | T1 1.5R @ 2460.53 |
| Target hit | 2023-09-12 15:20:00 | 2458.14 | 2460.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 2471.72 | 2467.52 | 0.00 | ORB-long ORB[2460.80,2470.93] vol=1.8x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-09-14 09:35:00 | 2468.70 | 2468.02 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 10:50:00 | 2423.42 | 2429.42 | 0.00 | ORB-short ORB[2424.75,2439.50] vol=5.2x ATR=3.09 |
| Stop hit — per-position SL triggered | 2023-09-18 11:05:00 | 2426.51 | 2428.14 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-09-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:50:00 | 2410.98 | 2421.06 | 0.00 | ORB-short ORB[2418.90,2431.88] vol=3.3x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 10:10:00 | 2405.04 | 2415.49 | 0.00 | T1 1.5R @ 2405.04 |
| Stop hit — per-position SL triggered | 2023-09-21 12:20:00 | 2410.98 | 2411.36 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 11:05:00 | 2422.19 | 2431.40 | 0.00 | ORB-short ORB[2434.59,2458.88] vol=2.2x ATR=3.34 |
| Stop hit — per-position SL triggered | 2023-09-25 11:10:00 | 2425.53 | 2430.61 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:20:00 | 2424.16 | 2425.46 | 0.00 | ORB-short ORB[2424.21,2435.67] vol=1.6x ATR=2.77 |
| Stop hit — per-position SL triggered | 2023-09-26 11:25:00 | 2426.93 | 2425.04 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 11:05:00 | 2447.28 | 2436.65 | 0.00 | ORB-long ORB[2429.82,2442.16] vol=4.3x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 11:10:00 | 2452.45 | 2438.59 | 0.00 | T1 1.5R @ 2452.45 |
| Target hit | 2023-09-27 15:20:00 | 2461.93 | 2452.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:55:00 | 2455.78 | 2448.97 | 0.00 | ORB-long ORB[2425.88,2450.47] vol=1.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2023-10-03 11:35:00 | 2452.17 | 2450.99 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:40:00 | 2452.88 | 2445.53 | 0.00 | ORB-long ORB[2424.90,2450.32] vol=1.5x ATR=5.65 |
| Stop hit — per-position SL triggered | 2023-10-04 09:55:00 | 2447.23 | 2447.31 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:45:00 | 2472.31 | 2457.60 | 0.00 | ORB-long ORB[2440.49,2461.54] vol=1.8x ATR=5.53 |
| Stop hit — per-position SL triggered | 2023-10-09 09:55:00 | 2466.78 | 2459.14 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 2496.80 | 2488.14 | 0.00 | ORB-long ORB[2475.16,2491.49] vol=2.6x ATR=3.72 |
| Stop hit — per-position SL triggered | 2023-10-11 09:35:00 | 2493.08 | 2488.95 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:55:00 | 2505.16 | 2510.43 | 0.00 | ORB-short ORB[2506.30,2518.20] vol=2.7x ATR=3.87 |
| Stop hit — per-position SL triggered | 2023-10-13 11:25:00 | 2509.03 | 2509.30 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:55:00 | 2500.98 | 2508.74 | 0.00 | ORB-short ORB[2504.43,2529.26] vol=1.6x ATR=3.75 |
| Stop hit — per-position SL triggered | 2023-10-18 11:10:00 | 2504.73 | 2508.29 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-10-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 11:00:00 | 2506.39 | 2497.19 | 0.00 | ORB-long ORB[2483.77,2501.18] vol=2.1x ATR=4.74 |
| Stop hit — per-position SL triggered | 2023-10-19 15:10:00 | 2501.65 | 2504.17 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-10-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 10:40:00 | 2428.54 | 2436.84 | 0.00 | ORB-short ORB[2429.86,2449.10] vol=2.2x ATR=3.82 |
| Stop hit — per-position SL triggered | 2023-10-31 10:50:00 | 2432.36 | 2436.28 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 11:00:00 | 2462.42 | 2454.15 | 0.00 | ORB-long ORB[2445.21,2458.19] vol=1.5x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-11-03 11:20:00 | 2458.68 | 2456.20 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:35:00 | 2453.77 | 2461.74 | 0.00 | ORB-short ORB[2465.08,2480.77] vol=2.1x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:30:00 | 2448.77 | 2456.67 | 0.00 | T1 1.5R @ 2448.77 |
| Target hit | 2023-11-09 15:20:00 | 2437.59 | 2444.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 2462.57 | 2447.58 | 0.00 | ORB-long ORB[2434.93,2443.44] vol=2.4x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-11-16 11:10:00 | 2459.33 | 2449.22 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:40:00 | 2474.33 | 2452.96 | 0.00 | ORB-long ORB[2424.75,2461.14] vol=1.6x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 10:20:00 | 2484.15 | 2464.86 | 0.00 | T1 1.5R @ 2484.15 |
| Stop hit — per-position SL triggered | 2023-11-17 10:45:00 | 2474.33 | 2465.91 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 11:05:00 | 2475.41 | 2472.54 | 0.00 | ORB-long ORB[2464.44,2475.36] vol=1.5x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-11-22 12:05:00 | 2472.87 | 2472.78 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:35:00 | 2504.03 | 2494.74 | 0.00 | ORB-long ORB[2475.06,2498.03] vol=1.6x ATR=4.08 |
| Stop hit — per-position SL triggered | 2023-11-30 09:45:00 | 2499.95 | 2497.46 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:55:00 | 2520.51 | 2511.96 | 0.00 | ORB-long ORB[2500.49,2516.08] vol=2.1x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 11:55:00 | 2527.11 | 2515.76 | 0.00 | T1 1.5R @ 2527.11 |
| Target hit | 2023-12-01 14:35:00 | 2523.12 | 2523.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — SELL (started 2023-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 10:55:00 | 2552.38 | 2563.41 | 0.00 | ORB-short ORB[2559.81,2593.94] vol=2.6x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:00:00 | 2546.41 | 2559.43 | 0.00 | T1 1.5R @ 2546.41 |
| Target hit | 2023-12-05 15:20:00 | 2521.20 | 2540.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2023-12-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:50:00 | 2536.69 | 2532.26 | 0.00 | ORB-long ORB[2523.17,2534.23] vol=2.1x ATR=4.47 |
| Stop hit — per-position SL triggered | 2023-12-06 13:10:00 | 2532.22 | 2534.35 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:35:00 | 2465.37 | 2473.08 | 0.00 | ORB-short ORB[2466.41,2488.64] vol=1.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-12-11 11:00:00 | 2468.79 | 2471.01 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 11:15:00 | 2471.42 | 2474.59 | 0.00 | ORB-short ORB[2474.92,2485.64] vol=1.8x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 14:10:00 | 2467.32 | 2472.89 | 0.00 | T1 1.5R @ 2467.32 |
| Stop hit — per-position SL triggered | 2023-12-15 15:00:00 | 2471.42 | 2472.62 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:15:00 | 2496.56 | 2493.83 | 0.00 | ORB-long ORB[2473.93,2493.56] vol=2.0x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 11:50:00 | 2501.28 | 2494.20 | 0.00 | T1 1.5R @ 2501.28 |
| Stop hit — per-position SL triggered | 2023-12-18 11:55:00 | 2496.56 | 2494.29 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 2507.62 | 2501.74 | 0.00 | ORB-long ORB[2493.61,2505.12] vol=1.7x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-12-19 10:25:00 | 2503.68 | 2506.11 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:35:00 | 2515.84 | 2520.66 | 0.00 | ORB-short ORB[2518.89,2530.94] vol=2.3x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-12-20 11:05:00 | 2519.64 | 2519.26 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:40:00 | 2523.12 | 2510.30 | 0.00 | ORB-long ORB[2494.88,2507.28] vol=1.6x ATR=5.14 |
| Stop hit — per-position SL triggered | 2023-12-21 11:05:00 | 2517.98 | 2514.00 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:15:00 | 2606.73 | 2600.96 | 0.00 | ORB-long ORB[2578.45,2594.09] vol=2.4x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:25:00 | 2615.21 | 2603.09 | 0.00 | T1 1.5R @ 2615.21 |
| Stop hit — per-position SL triggered | 2023-12-29 10:35:00 | 2606.73 | 2605.19 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 2590.01 | 2597.17 | 0.00 | ORB-short ORB[2590.99,2610.61] vol=1.6x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:30:00 | 2581.70 | 2593.82 | 0.00 | T1 1.5R @ 2581.70 |
| Target hit | 2024-01-02 15:20:00 | 2572.15 | 2577.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2024-01-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 11:00:00 | 2577.22 | 2570.12 | 0.00 | ORB-long ORB[2558.43,2573.82] vol=2.4x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-01-04 11:15:00 | 2573.34 | 2570.54 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:30:00 | 2529.22 | 2534.15 | 0.00 | ORB-short ORB[2536.49,2546.72] vol=2.6x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 12:05:00 | 2523.46 | 2530.34 | 0.00 | T1 1.5R @ 2523.46 |
| Target hit | 2024-01-11 15:20:00 | 2493.61 | 2515.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-01-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:25:00 | 2502.90 | 2506.76 | 0.00 | ORB-short ORB[2503.44,2529.36] vol=7.8x ATR=4.44 |
| Stop hit — per-position SL triggered | 2024-01-17 11:05:00 | 2507.34 | 2505.46 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:25:00 | 2504.38 | 2512.23 | 0.00 | ORB-short ORB[2508.90,2524.59] vol=1.8x ATR=5.98 |
| Stop hit — per-position SL triggered | 2024-01-19 12:40:00 | 2510.36 | 2506.23 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 11:10:00 | 2442.46 | 2456.97 | 0.00 | ORB-short ORB[2452.34,2481.80] vol=1.8x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 14:40:00 | 2433.21 | 2449.02 | 0.00 | T1 1.5R @ 2433.21 |
| Target hit | 2024-01-20 15:20:00 | 2426.72 | 2445.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2024-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:40:00 | 2398.93 | 2411.88 | 0.00 | ORB-short ORB[2407.19,2440.00] vol=1.8x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:00:00 | 2388.94 | 2407.55 | 0.00 | T1 1.5R @ 2388.94 |
| Stop hit — per-position SL triggered | 2024-01-23 10:10:00 | 2398.93 | 2406.75 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:30:00 | 2442.46 | 2426.85 | 0.00 | ORB-long ORB[2407.34,2431.63] vol=1.5x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:00:00 | 2451.40 | 2436.23 | 0.00 | T1 1.5R @ 2451.40 |
| Stop hit — per-position SL triggered | 2024-01-30 10:30:00 | 2442.46 | 2438.38 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 11:15:00 | 2460.16 | 2454.25 | 0.00 | ORB-long ORB[2430.75,2443.73] vol=1.9x ATR=5.74 |
| Stop hit — per-position SL triggered | 2024-02-01 11:55:00 | 2454.42 | 2455.44 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:00:00 | 2382.84 | 2382.93 | 0.00 | ORB-short ORB[2387.37,2400.06] vol=1.9x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-02-07 12:30:00 | 2386.22 | 2382.82 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:10:00 | 2377.48 | 2383.95 | 0.00 | ORB-short ORB[2386.39,2401.93] vol=1.6x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 10:15:00 | 2371.31 | 2381.85 | 0.00 | T1 1.5R @ 2371.31 |
| Target hit | 2024-02-08 14:45:00 | 2370.65 | 2368.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 83 — BUY (started 2024-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-09 11:00:00 | 2386.68 | 2379.93 | 0.00 | ORB-long ORB[2366.81,2381.42] vol=2.0x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-02-09 11:05:00 | 2382.23 | 2380.63 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 11:00:00 | 2371.63 | 2374.00 | 0.00 | ORB-short ORB[2372.66,2390.17] vol=2.6x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 12:00:00 | 2365.77 | 2372.02 | 0.00 | T1 1.5R @ 2365.77 |
| Target hit | 2024-02-12 15:20:00 | 2346.06 | 2356.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2024-02-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:45:00 | 2341.92 | 2348.89 | 0.00 | ORB-short ORB[2346.06,2354.91] vol=1.5x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-02-14 11:50:00 | 2346.05 | 2343.66 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:00:00 | 2336.37 | 2349.26 | 0.00 | ORB-short ORB[2347.73,2380.78] vol=2.0x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-02-22 13:55:00 | 2342.91 | 2337.58 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 11:15:00 | 2359.14 | 2360.10 | 0.00 | ORB-short ORB[2360.81,2369.61] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-02-27 11:25:00 | 2362.36 | 2360.14 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 2366.71 | 2372.90 | 0.00 | ORB-short ORB[2367.50,2377.83] vol=1.9x ATR=3.97 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 2370.68 | 2372.62 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:15:00 | 2359.14 | 2368.54 | 0.00 | ORB-short ORB[2369.07,2389.14] vol=2.2x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-02-29 10:20:00 | 2363.63 | 2368.09 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 11:15:00 | 2387.42 | 2383.75 | 0.00 | ORB-long ORB[2364.75,2380.04] vol=2.1x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 11:35:00 | 2394.25 | 2384.39 | 0.00 | T1 1.5R @ 2394.25 |
| Stop hit — per-position SL triggered | 2024-03-01 12:00:00 | 2387.42 | 2385.04 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 11:00:00 | 2356.09 | 2363.85 | 0.00 | ORB-short ORB[2365.58,2380.48] vol=1.9x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-03-05 11:20:00 | 2359.36 | 2363.25 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 11:10:00 | 2319.30 | 2322.84 | 0.00 | ORB-short ORB[2321.17,2336.22] vol=1.7x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:30:00 | 2314.45 | 2321.49 | 0.00 | T1 1.5R @ 2314.45 |
| Target hit | 2024-03-13 15:20:00 | 2275.77 | 2297.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — SELL (started 2024-03-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:00:00 | 2280.74 | 2285.78 | 0.00 | ORB-short ORB[2287.04,2298.79] vol=2.3x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-03-15 11:10:00 | 2284.77 | 2285.49 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:00:00 | 2269.82 | 2273.25 | 0.00 | ORB-short ORB[2272.28,2287.04] vol=1.6x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 15:05:00 | 2263.62 | 2269.83 | 0.00 | T1 1.5R @ 2263.62 |
| Target hit | 2024-03-18 15:20:00 | 2260.33 | 2269.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — SELL (started 2024-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:55:00 | 2240.85 | 2251.95 | 0.00 | ORB-short ORB[2245.97,2266.28] vol=1.8x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:15:00 | 2233.80 | 2247.33 | 0.00 | T1 1.5R @ 2233.80 |
| Target hit | 2024-03-19 14:55:00 | 2231.95 | 2230.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 96 — SELL (started 2024-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 11:00:00 | 2211.29 | 2217.16 | 0.00 | ORB-short ORB[2213.26,2228.46] vol=1.5x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-03-20 11:05:00 | 2215.60 | 2217.14 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 11:00:00 | 2248.43 | 2243.44 | 0.00 | ORB-long ORB[2225.06,2241.00] vol=1.6x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-04-01 11:05:00 | 2244.84 | 2243.61 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-04-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 10:40:00 | 2244.98 | 2248.51 | 0.00 | ORB-short ORB[2245.28,2260.48] vol=1.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 11:00:00 | 2240.52 | 2247.81 | 0.00 | T1 1.5R @ 2240.52 |
| Stop hit — per-position SL triggered | 2024-04-02 11:15:00 | 2244.98 | 2247.61 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 2214.69 | 2224.67 | 0.00 | ORB-short ORB[2228.01,2237.65] vol=1.8x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 2218.97 | 2224.23 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-04-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:55:00 | 2235.88 | 2231.91 | 0.00 | ORB-long ORB[2221.23,2231.21] vol=1.8x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-04-05 11:20:00 | 2232.19 | 2232.60 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 09:40:00 | 2245.23 | 2236.71 | 0.00 | ORB-long ORB[2231.36,2239.82] vol=1.5x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:50:00 | 2250.63 | 2240.20 | 0.00 | T1 1.5R @ 2250.63 |
| Stop hit — per-position SL triggered | 2024-04-08 10:05:00 | 2245.23 | 2241.21 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-04-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 11:10:00 | 2223.34 | 2228.21 | 0.00 | ORB-short ORB[2225.31,2239.72] vol=4.7x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-04-10 11:20:00 | 2226.01 | 2228.10 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 2210.95 | 2218.21 | 0.00 | ORB-short ORB[2216.60,2227.87] vol=1.6x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 10:05:00 | 2205.71 | 2214.60 | 0.00 | T1 1.5R @ 2205.71 |
| Stop hit — per-position SL triggered | 2024-04-12 10:45:00 | 2210.95 | 2212.42 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-15 10:00:00 | 2159.90 | 2166.97 | 0.00 | ORB-short ORB[2168.01,2191.37] vol=1.7x ATR=5.56 |
| Target hit | 2024-04-15 15:20:00 | 2158.76 | 2159.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 105 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:15:00 | 2168.99 | 2154.08 | 0.00 | ORB-long ORB[2136.58,2159.16] vol=2.0x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 12:00:00 | 2174.50 | 2157.24 | 0.00 | T1 1.5R @ 2174.50 |
| Stop hit — per-position SL triggered | 2024-04-16 13:25:00 | 2168.99 | 2160.55 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2024-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:55:00 | 2223.10 | 2218.92 | 0.00 | ORB-long ORB[2211.98,2219.90] vol=2.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-04-23 10:20:00 | 2220.14 | 2219.54 | 0.00 | SL hit |

### Cycle 107 — SELL (started 2024-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 11:10:00 | 2214.74 | 2219.57 | 0.00 | ORB-short ORB[2215.77,2231.31] vol=1.6x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 11:15:00 | 2209.49 | 2219.00 | 0.00 | T1 1.5R @ 2209.49 |
| Stop hit — per-position SL triggered | 2024-04-24 11:25:00 | 2214.74 | 2218.39 | 0.00 | SL hit |

### Cycle 108 — SELL (started 2024-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:35:00 | 2185.72 | 2192.21 | 0.00 | ORB-short ORB[2189.26,2203.28] vol=4.0x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-04-25 09:45:00 | 2191.31 | 2190.11 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2024-04-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 10:45:00 | 2210.70 | 2207.20 | 0.00 | ORB-long ORB[2191.72,2208.34] vol=1.9x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 10:50:00 | 2216.79 | 2207.73 | 0.00 | T1 1.5R @ 2216.79 |
| Stop hit — per-position SL triggered | 2024-04-26 11:05:00 | 2210.70 | 2208.35 | 0.00 | SL hit |

### Cycle 110 — SELL (started 2024-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:05:00 | 2186.80 | 2189.10 | 0.00 | ORB-short ORB[2187.73,2197.52] vol=2.3x ATR=4.14 |
| Stop hit — per-position SL triggered | 2024-04-29 10:25:00 | 2190.94 | 2188.94 | 0.00 | SL hit |

### Cycle 111 — BUY (started 2024-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:00:00 | 2198.16 | 2195.11 | 0.00 | ORB-long ORB[2188.67,2196.05] vol=1.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-04-30 10:25:00 | 2195.29 | 2196.09 | 0.00 | SL hit |

### Cycle 112 — BUY (started 2024-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 11:05:00 | 2194.23 | 2185.77 | 0.00 | ORB-long ORB[2179.67,2192.60] vol=2.3x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 11:50:00 | 2200.49 | 2190.36 | 0.00 | T1 1.5R @ 2200.49 |
| Target hit | 2024-05-06 15:20:00 | 2220.15 | 2202.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 113 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 11:15:00 | 2294.07 | 2309.50 | 0.00 | ORB-short ORB[2312.61,2333.51] vol=1.8x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-05-08 11:20:00 | 2299.28 | 2309.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:10:00 | 2578.55 | 2023-05-12 11:30:00 | 2586.47 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-05-12 11:10:00 | 2578.55 | 2023-05-12 14:15:00 | 2578.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-18 09:45:00 | 2608.89 | 2023-05-18 09:50:00 | 2614.28 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-19 10:55:00 | 2561.14 | 2023-05-19 11:00:00 | 2565.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-05-22 11:05:00 | 2605.45 | 2023-05-22 11:10:00 | 2599.93 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-05-24 11:15:00 | 2594.88 | 2023-05-24 11:30:00 | 2590.75 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-05-25 11:10:00 | 2558.82 | 2023-05-25 12:15:00 | 2552.81 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-05-25 11:10:00 | 2558.82 | 2023-05-25 14:55:00 | 2554.00 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2023-06-02 09:35:00 | 2676.32 | 2023-06-02 09:50:00 | 2671.36 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-06 10:45:00 | 2644.16 | 2023-06-06 10:50:00 | 2648.47 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-07 11:10:00 | 2680.95 | 2023-06-07 11:25:00 | 2675.55 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-08 11:10:00 | 2652.57 | 2023-06-08 11:20:00 | 2657.55 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-09 09:40:00 | 2619.12 | 2023-06-09 09:50:00 | 2625.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-14 10:55:00 | 2641.94 | 2023-06-14 11:15:00 | 2634.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-15 11:05:00 | 2653.45 | 2023-06-15 11:45:00 | 2644.83 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-06-15 11:05:00 | 2653.45 | 2023-06-15 15:20:00 | 2648.04 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-16 11:15:00 | 2668.31 | 2023-06-16 11:20:00 | 2662.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-19 10:50:00 | 2647.85 | 2023-06-19 12:05:00 | 2640.98 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-19 10:50:00 | 2647.85 | 2023-06-19 14:30:00 | 2646.32 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2023-06-20 10:25:00 | 2606.83 | 2023-06-20 11:25:00 | 2611.93 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-22 11:05:00 | 2609.68 | 2023-06-22 12:25:00 | 2613.64 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-27 11:15:00 | 2594.92 | 2023-06-27 11:20:00 | 2598.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-30 10:50:00 | 2612.09 | 2023-06-30 11:00:00 | 2616.81 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-03 11:15:00 | 2622.42 | 2023-07-03 13:10:00 | 2626.77 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-04 10:45:00 | 2646.52 | 2023-07-04 11:30:00 | 2637.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-07-04 10:45:00 | 2646.52 | 2023-07-04 12:05:00 | 2646.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 10:50:00 | 2690.34 | 2023-07-05 10:55:00 | 2699.41 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-05 10:50:00 | 2690.34 | 2023-07-05 11:00:00 | 2690.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 10:05:00 | 2638.70 | 2023-07-11 10:20:00 | 2646.88 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-07-11 10:05:00 | 2638.70 | 2023-07-11 12:45:00 | 2646.47 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-14 10:35:00 | 2629.70 | 2023-07-14 10:45:00 | 2624.85 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-18 11:05:00 | 2637.12 | 2023-07-18 13:45:00 | 2641.67 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-19 10:35:00 | 2629.40 | 2023-07-19 10:40:00 | 2623.24 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-07-19 10:35:00 | 2629.40 | 2023-07-19 12:55:00 | 2626.89 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-07-20 10:55:00 | 2634.17 | 2023-07-20 11:55:00 | 2642.21 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-07-20 10:55:00 | 2634.17 | 2023-07-20 15:20:00 | 2658.62 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2023-07-26 09:35:00 | 2542.40 | 2023-07-26 10:25:00 | 2548.80 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-07-26 09:35:00 | 2542.40 | 2023-07-26 12:35:00 | 2544.27 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2023-08-02 11:05:00 | 2515.05 | 2023-08-02 11:10:00 | 2511.11 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-04 09:40:00 | 2499.31 | 2023-08-04 11:10:00 | 2503.71 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-08 10:55:00 | 2523.02 | 2023-08-08 11:15:00 | 2517.75 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-08-08 10:55:00 | 2523.02 | 2023-08-08 13:00:00 | 2523.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-10 10:25:00 | 2493.61 | 2023-08-10 10:30:00 | 2488.96 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2023-08-10 10:25:00 | 2493.61 | 2023-08-10 10:35:00 | 2493.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-14 10:00:00 | 2486.13 | 2023-08-14 10:05:00 | 2479.72 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-16 10:05:00 | 2502.46 | 2023-08-16 10:50:00 | 2497.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-21 10:55:00 | 2526.36 | 2023-08-21 12:15:00 | 2521.61 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-08-29 10:15:00 | 2498.52 | 2023-08-29 10:30:00 | 2501.99 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-08-30 10:25:00 | 2494.54 | 2023-08-30 10:35:00 | 2491.26 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-09-04 11:10:00 | 2455.24 | 2023-09-04 11:15:00 | 2457.69 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2023-09-07 10:35:00 | 2463.11 | 2023-09-07 11:15:00 | 2465.98 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-09-11 10:50:00 | 2487.06 | 2023-09-11 11:25:00 | 2492.02 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-09-11 10:50:00 | 2487.06 | 2023-09-11 13:25:00 | 2487.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:55:00 | 2467.88 | 2023-09-12 10:20:00 | 2460.53 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-09-12 09:55:00 | 2467.88 | 2023-09-12 15:20:00 | 2458.14 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-14 09:30:00 | 2471.72 | 2023-09-14 09:35:00 | 2468.70 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-09-18 10:50:00 | 2423.42 | 2023-09-18 11:05:00 | 2426.51 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-09-21 09:50:00 | 2410.98 | 2023-09-21 10:10:00 | 2405.04 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-09-21 09:50:00 | 2410.98 | 2023-09-21 12:20:00 | 2410.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-25 11:05:00 | 2422.19 | 2023-09-25 11:10:00 | 2425.53 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-09-26 10:20:00 | 2424.16 | 2023-09-26 11:25:00 | 2426.93 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2023-09-27 11:05:00 | 2447.28 | 2023-09-27 11:10:00 | 2452.45 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-09-27 11:05:00 | 2447.28 | 2023-09-27 15:20:00 | 2461.93 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2023-10-03 10:55:00 | 2455.78 | 2023-10-03 11:35:00 | 2452.17 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-10-04 09:40:00 | 2452.88 | 2023-10-04 09:55:00 | 2447.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-09 09:45:00 | 2472.31 | 2023-10-09 09:55:00 | 2466.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-10-11 09:30:00 | 2496.80 | 2023-10-11 09:35:00 | 2493.08 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-10-13 10:55:00 | 2505.16 | 2023-10-13 11:25:00 | 2509.03 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-10-18 10:55:00 | 2500.98 | 2023-10-18 11:10:00 | 2504.73 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-10-19 11:00:00 | 2506.39 | 2023-10-19 15:10:00 | 2501.65 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-31 10:40:00 | 2428.54 | 2023-10-31 10:50:00 | 2432.36 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-03 11:00:00 | 2462.42 | 2023-11-03 11:20:00 | 2458.68 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-11-09 10:35:00 | 2453.77 | 2023-11-09 11:30:00 | 2448.77 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-09 10:35:00 | 2453.77 | 2023-11-09 15:20:00 | 2437.59 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2023-11-16 11:00:00 | 2462.57 | 2023-11-16 11:10:00 | 2459.33 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-11-17 09:40:00 | 2474.33 | 2023-11-17 10:20:00 | 2484.15 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-11-17 09:40:00 | 2474.33 | 2023-11-17 10:45:00 | 2474.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 11:05:00 | 2475.41 | 2023-11-22 12:05:00 | 2472.87 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest1 | 2023-11-30 09:35:00 | 2504.03 | 2023-11-30 09:45:00 | 2499.95 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-12-01 10:55:00 | 2520.51 | 2023-12-01 11:55:00 | 2527.11 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-12-01 10:55:00 | 2520.51 | 2023-12-01 14:35:00 | 2523.12 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-12-05 10:55:00 | 2552.38 | 2023-12-05 11:00:00 | 2546.41 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-12-05 10:55:00 | 2552.38 | 2023-12-05 15:20:00 | 2521.20 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2023-12-06 10:50:00 | 2536.69 | 2023-12-06 13:10:00 | 2532.22 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-12-11 10:35:00 | 2465.37 | 2023-12-11 11:00:00 | 2468.79 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-12-15 11:15:00 | 2471.42 | 2023-12-15 14:10:00 | 2467.32 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2023-12-15 11:15:00 | 2471.42 | 2023-12-15 15:00:00 | 2471.42 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-18 11:15:00 | 2496.56 | 2023-12-18 11:50:00 | 2501.28 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2023-12-18 11:15:00 | 2496.56 | 2023-12-18 11:55:00 | 2496.56 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-19 09:35:00 | 2507.62 | 2023-12-19 10:25:00 | 2503.68 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-12-20 10:35:00 | 2515.84 | 2023-12-20 11:05:00 | 2519.64 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-12-21 10:40:00 | 2523.12 | 2023-12-21 11:05:00 | 2517.98 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-29 10:15:00 | 2606.73 | 2023-12-29 10:25:00 | 2615.21 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-29 10:15:00 | 2606.73 | 2023-12-29 10:35:00 | 2606.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 09:55:00 | 2590.01 | 2024-01-02 10:30:00 | 2581.70 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-01-02 09:55:00 | 2590.01 | 2024-01-02 15:20:00 | 2572.15 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2024-01-04 11:00:00 | 2577.22 | 2024-01-04 11:15:00 | 2573.34 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-01-11 10:30:00 | 2529.22 | 2024-01-11 12:05:00 | 2523.46 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-01-11 10:30:00 | 2529.22 | 2024-01-11 15:20:00 | 2493.61 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2024-01-17 10:25:00 | 2502.90 | 2024-01-17 11:05:00 | 2507.34 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-19 10:25:00 | 2504.38 | 2024-01-19 12:40:00 | 2510.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-20 11:10:00 | 2442.46 | 2024-01-20 14:40:00 | 2433.21 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-01-20 11:10:00 | 2442.46 | 2024-01-20 15:20:00 | 2426.72 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-01-23 09:40:00 | 2398.93 | 2024-01-23 10:00:00 | 2388.94 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-23 09:40:00 | 2398.93 | 2024-01-23 10:10:00 | 2398.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-30 09:30:00 | 2442.46 | 2024-01-30 10:00:00 | 2451.40 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-01-30 09:30:00 | 2442.46 | 2024-01-30 10:30:00 | 2442.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-01 11:15:00 | 2460.16 | 2024-02-01 11:55:00 | 2454.42 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-07 11:00:00 | 2382.84 | 2024-02-07 12:30:00 | 2386.22 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-02-08 10:10:00 | 2377.48 | 2024-02-08 10:15:00 | 2371.31 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-02-08 10:10:00 | 2377.48 | 2024-02-08 14:45:00 | 2370.65 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-02-09 11:00:00 | 2386.68 | 2024-02-09 11:05:00 | 2382.23 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-12 11:00:00 | 2371.63 | 2024-02-12 12:00:00 | 2365.77 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-12 11:00:00 | 2371.63 | 2024-02-12 15:20:00 | 2346.06 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2024-02-14 09:45:00 | 2341.92 | 2024-02-14 11:50:00 | 2346.05 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-02-22 10:00:00 | 2336.37 | 2024-02-22 13:55:00 | 2342.91 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-27 11:15:00 | 2359.14 | 2024-02-27 11:25:00 | 2362.36 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-02-28 10:50:00 | 2366.71 | 2024-02-28 11:00:00 | 2370.68 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-02-29 10:15:00 | 2359.14 | 2024-02-29 10:20:00 | 2363.63 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-03-01 11:15:00 | 2387.42 | 2024-03-01 11:35:00 | 2394.25 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-03-01 11:15:00 | 2387.42 | 2024-03-01 12:00:00 | 2387.42 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-05 11:00:00 | 2356.09 | 2024-03-05 11:20:00 | 2359.36 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-03-13 11:10:00 | 2319.30 | 2024-03-13 11:30:00 | 2314.45 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2024-03-13 11:10:00 | 2319.30 | 2024-03-13 15:20:00 | 2275.77 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2024-03-15 11:00:00 | 2280.74 | 2024-03-15 11:10:00 | 2284.77 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-03-18 11:00:00 | 2269.82 | 2024-03-18 15:05:00 | 2263.62 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-03-18 11:00:00 | 2269.82 | 2024-03-18 15:20:00 | 2260.33 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-03-19 09:55:00 | 2240.85 | 2024-03-19 10:15:00 | 2233.80 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-03-19 09:55:00 | 2240.85 | 2024-03-19 14:55:00 | 2231.95 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-03-20 11:00:00 | 2211.29 | 2024-03-20 11:05:00 | 2215.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-04-01 11:00:00 | 2248.43 | 2024-04-01 11:05:00 | 2244.84 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-04-02 10:40:00 | 2244.98 | 2024-04-02 11:00:00 | 2240.52 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2024-04-02 10:40:00 | 2244.98 | 2024-04-02 11:15:00 | 2244.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 10:50:00 | 2214.69 | 2024-04-04 10:55:00 | 2218.97 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-04-05 10:55:00 | 2235.88 | 2024-04-05 11:20:00 | 2232.19 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-04-08 09:40:00 | 2245.23 | 2024-04-08 09:50:00 | 2250.63 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-04-08 09:40:00 | 2245.23 | 2024-04-08 10:05:00 | 2245.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-10 11:10:00 | 2223.34 | 2024-04-10 11:20:00 | 2226.01 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2024-04-12 09:40:00 | 2210.95 | 2024-04-12 10:05:00 | 2205.71 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-04-12 09:40:00 | 2210.95 | 2024-04-12 10:45:00 | 2210.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-15 10:00:00 | 2159.90 | 2024-04-15 15:20:00 | 2158.76 | TARGET_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2024-04-16 11:15:00 | 2168.99 | 2024-04-16 12:00:00 | 2174.50 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-04-16 11:15:00 | 2168.99 | 2024-04-16 13:25:00 | 2168.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 09:55:00 | 2223.10 | 2024-04-23 10:20:00 | 2220.14 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2024-04-24 11:10:00 | 2214.74 | 2024-04-24 11:15:00 | 2209.49 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-04-24 11:10:00 | 2214.74 | 2024-04-24 11:25:00 | 2214.74 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-25 09:35:00 | 2185.72 | 2024-04-25 09:45:00 | 2191.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-26 10:45:00 | 2210.70 | 2024-04-26 10:50:00 | 2216.79 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-04-26 10:45:00 | 2210.70 | 2024-04-26 11:05:00 | 2210.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-29 10:05:00 | 2186.80 | 2024-04-29 10:25:00 | 2190.94 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-04-30 10:00:00 | 2198.16 | 2024-04-30 10:25:00 | 2195.29 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2024-05-06 11:05:00 | 2194.23 | 2024-05-06 11:50:00 | 2200.49 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-05-06 11:05:00 | 2194.23 | 2024-05-06 15:20:00 | 2220.15 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2024-05-08 11:15:00 | 2294.07 | 2024-05-08 11:20:00 | 2299.28 | STOP_HIT | 1.00 | -0.23% |
