# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2214.50
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 12 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 102 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 61
- **Target hits / Stop hits / Partials:** 12 / 61 / 29
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 11.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 20 | 37.7% | 5 | 33 | 15 | 0.08% | 4.4% |
| BUY @ 2nd Alert (retest1) | 53 | 20 | 37.7% | 5 | 33 | 15 | 0.08% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 21 | 42.9% | 7 | 28 | 14 | 0.15% | 7.4% |
| SELL @ 2nd Alert (retest1) | 49 | 21 | 42.9% | 7 | 28 | 14 | 0.15% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 102 | 41 | 40.2% | 12 | 61 | 29 | 0.12% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 2599.90 | 2583.50 | 0.00 | ORB-long ORB[2560.10,2588.00] vol=1.7x ATR=9.09 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 2590.81 | 2586.20 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 2577.00 | 2564.80 | 0.00 | ORB-long ORB[2542.10,2576.90] vol=2.1x ATR=8.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:45:00 | 2589.48 | 2571.02 | 0.00 | T1 1.5R @ 2589.48 |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 2577.00 | 2578.40 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:35:00 | 2500.80 | 2503.82 | 0.00 | ORB-short ORB[2501.10,2533.70] vol=1.9x ATR=7.59 |
| Stop hit — per-position SL triggered | 2025-06-03 10:50:00 | 2508.39 | 2503.90 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:40:00 | 2543.60 | 2527.63 | 0.00 | ORB-long ORB[2506.00,2541.70] vol=1.5x ATR=6.26 |
| Stop hit — per-position SL triggered | 2025-06-04 10:50:00 | 2537.34 | 2528.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:50:00 | 2480.70 | 2489.48 | 0.00 | ORB-short ORB[2487.60,2513.80] vol=2.9x ATR=8.13 |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 2488.83 | 2489.04 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 2515.90 | 2518.64 | 0.00 | ORB-short ORB[2516.20,2539.00] vol=2.8x ATR=7.71 |
| Stop hit — per-position SL triggered | 2025-06-06 09:45:00 | 2523.61 | 2518.93 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 2653.40 | 2635.11 | 0.00 | ORB-long ORB[2605.10,2641.60] vol=2.8x ATR=8.46 |
| Stop hit — per-position SL triggered | 2025-06-10 09:45:00 | 2644.94 | 2644.03 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:45:00 | 2669.70 | 2683.07 | 0.00 | ORB-short ORB[2680.30,2704.00] vol=2.1x ATR=8.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:50:00 | 2657.62 | 2679.49 | 0.00 | T1 1.5R @ 2657.62 |
| Target hit | 2025-06-19 11:20:00 | 2654.40 | 2650.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:15:00 | 2728.00 | 2701.56 | 0.00 | ORB-long ORB[2682.50,2715.50] vol=2.3x ATR=10.16 |
| Stop hit — per-position SL triggered | 2025-06-24 10:20:00 | 2717.84 | 2702.83 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:55:00 | 2720.20 | 2706.96 | 0.00 | ORB-long ORB[2688.10,2715.00] vol=1.8x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:55:00 | 2731.48 | 2715.06 | 0.00 | T1 1.5R @ 2731.48 |
| Target hit | 2025-06-25 15:20:00 | 2750.30 | 2736.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:45:00 | 2859.60 | 2821.18 | 0.00 | ORB-long ORB[2805.70,2834.80] vol=2.7x ATR=11.23 |
| Stop hit — per-position SL triggered | 2025-06-30 10:50:00 | 2848.37 | 2827.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 2881.60 | 2873.57 | 0.00 | ORB-long ORB[2845.40,2879.00] vol=3.8x ATR=9.41 |
| Stop hit — per-position SL triggered | 2025-07-01 09:35:00 | 2872.19 | 2871.92 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 11:10:00 | 2918.90 | 2902.07 | 0.00 | ORB-long ORB[2882.90,2910.90] vol=1.7x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-07-04 11:20:00 | 2912.42 | 2902.80 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:00:00 | 2876.40 | 2845.92 | 0.00 | ORB-long ORB[2823.10,2850.10] vol=2.1x ATR=9.58 |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 2866.82 | 2848.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 10:35:00 | 2791.60 | 2806.84 | 0.00 | ORB-short ORB[2793.60,2832.60] vol=1.9x ATR=7.63 |
| Stop hit — per-position SL triggered | 2025-07-21 10:55:00 | 2799.23 | 2804.34 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:10:00 | 2705.10 | 2714.32 | 0.00 | ORB-short ORB[2723.20,2746.90] vol=1.8x ATR=7.53 |
| Stop hit — per-position SL triggered | 2025-08-05 12:00:00 | 2712.63 | 2711.72 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 2656.70 | 2674.12 | 0.00 | ORB-short ORB[2685.00,2716.50] vol=1.8x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 2663.81 | 2672.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:10:00 | 2662.30 | 2679.38 | 0.00 | ORB-short ORB[2680.00,2699.00] vol=1.9x ATR=8.03 |
| Stop hit — per-position SL triggered | 2025-08-08 11:00:00 | 2670.33 | 2674.03 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:05:00 | 2700.00 | 2708.88 | 0.00 | ORB-short ORB[2700.50,2734.20] vol=4.0x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-08-14 10:30:00 | 2706.64 | 2709.12 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:00:00 | 2758.00 | 2733.90 | 0.00 | ORB-long ORB[2708.40,2744.70] vol=1.8x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:15:00 | 2771.93 | 2738.63 | 0.00 | T1 1.5R @ 2771.93 |
| Target hit | 2025-08-20 15:20:00 | 2834.10 | 2796.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-08-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:40:00 | 2871.70 | 2846.37 | 0.00 | ORB-long ORB[2815.10,2845.00] vol=1.8x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:25:00 | 2881.82 | 2856.35 | 0.00 | T1 1.5R @ 2881.82 |
| Stop hit — per-position SL triggered | 2025-08-21 12:25:00 | 2871.70 | 2862.60 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 2863.10 | 2848.22 | 0.00 | ORB-long ORB[2836.50,2854.40] vol=2.1x ATR=6.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:15:00 | 2873.40 | 2851.72 | 0.00 | T1 1.5R @ 2873.40 |
| Stop hit — per-position SL triggered | 2025-08-22 12:25:00 | 2863.10 | 2863.09 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:20:00 | 2929.30 | 2918.81 | 0.00 | ORB-long ORB[2891.00,2928.50] vol=3.5x ATR=9.22 |
| Stop hit — per-position SL triggered | 2025-08-26 10:30:00 | 2920.08 | 2918.94 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:50:00 | 2930.50 | 2910.84 | 0.00 | ORB-long ORB[2886.90,2920.10] vol=2.2x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:45:00 | 2940.42 | 2921.34 | 0.00 | T1 1.5R @ 2940.42 |
| Stop hit — per-position SL triggered | 2025-09-02 12:40:00 | 2930.50 | 2923.97 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:05:00 | 2866.30 | 2888.20 | 0.00 | ORB-short ORB[2895.50,2919.00] vol=1.5x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:15:00 | 2853.94 | 2880.64 | 0.00 | T1 1.5R @ 2853.94 |
| Stop hit — per-position SL triggered | 2025-09-03 10:40:00 | 2866.30 | 2872.14 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 2781.20 | 2840.26 | 0.00 | ORB-short ORB[2849.10,2872.90] vol=3.1x ATR=13.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 2761.05 | 2826.90 | 0.00 | T1 1.5R @ 2761.05 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 2781.20 | 2820.24 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:00:00 | 2959.00 | 2946.14 | 0.00 | ORB-long ORB[2930.10,2955.10] vol=1.6x ATR=7.56 |
| Stop hit — per-position SL triggered | 2025-09-16 10:40:00 | 2951.44 | 2949.27 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:15:00 | 2670.00 | 2661.98 | 0.00 | ORB-long ORB[2635.20,2668.40] vol=1.6x ATR=7.17 |
| Stop hit — per-position SL triggered | 2025-09-29 11:20:00 | 2662.83 | 2662.07 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 2717.40 | 2704.25 | 0.00 | ORB-long ORB[2680.00,2701.90] vol=2.0x ATR=6.50 |
| Stop hit — per-position SL triggered | 2025-10-03 12:20:00 | 2710.90 | 2712.44 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:15:00 | 2774.60 | 2757.73 | 0.00 | ORB-long ORB[2722.00,2746.90] vol=4.3x ATR=7.93 |
| Stop hit — per-position SL triggered | 2025-10-06 10:25:00 | 2766.67 | 2760.56 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:40:00 | 2749.80 | 2759.93 | 0.00 | ORB-short ORB[2751.90,2775.00] vol=2.1x ATR=6.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:55:00 | 2740.34 | 2755.73 | 0.00 | T1 1.5R @ 2740.34 |
| Stop hit — per-position SL triggered | 2025-10-13 12:25:00 | 2749.80 | 2749.94 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:05:00 | 2744.80 | 2766.32 | 0.00 | ORB-short ORB[2750.70,2785.00] vol=1.9x ATR=6.76 |
| Stop hit — per-position SL triggered | 2025-10-14 11:40:00 | 2751.56 | 2762.80 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:25:00 | 2756.40 | 2762.49 | 0.00 | ORB-short ORB[2762.70,2797.40] vol=1.5x ATR=6.92 |
| Stop hit — per-position SL triggered | 2025-10-16 10:45:00 | 2763.32 | 2762.16 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 2723.60 | 2743.55 | 0.00 | ORB-short ORB[2746.00,2775.00] vol=5.1x ATR=9.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 2709.82 | 2732.50 | 0.00 | T1 1.5R @ 2709.82 |
| Stop hit — per-position SL triggered | 2025-10-17 10:10:00 | 2723.60 | 2726.00 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:10:00 | 2717.60 | 2702.76 | 0.00 | ORB-long ORB[2687.70,2714.90] vol=2.4x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 12:00:00 | 2726.98 | 2706.83 | 0.00 | T1 1.5R @ 2726.98 |
| Stop hit — per-position SL triggered | 2025-10-20 12:20:00 | 2717.60 | 2707.39 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:55:00 | 2876.30 | 2859.86 | 0.00 | ORB-long ORB[2835.00,2866.50] vol=1.6x ATR=10.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:05:00 | 2891.58 | 2865.78 | 0.00 | T1 1.5R @ 2891.58 |
| Stop hit — per-position SL triggered | 2025-10-27 10:20:00 | 2876.30 | 2867.93 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:10:00 | 2821.30 | 2822.88 | 0.00 | ORB-short ORB[2821.60,2849.00] vol=5.9x ATR=7.94 |
| Stop hit — per-position SL triggered | 2025-10-29 11:35:00 | 2829.24 | 2822.39 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:10:00 | 2763.10 | 2790.25 | 0.00 | ORB-short ORB[2780.50,2812.50] vol=1.6x ATR=6.58 |
| Stop hit — per-position SL triggered | 2025-11-04 13:35:00 | 2769.68 | 2780.91 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:15:00 | 2766.40 | 2744.09 | 0.00 | ORB-long ORB[2726.60,2753.50] vol=1.5x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:20:00 | 2778.18 | 2750.05 | 0.00 | T1 1.5R @ 2778.18 |
| Target hit | 2025-11-10 12:05:00 | 2776.20 | 2778.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2025-11-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:20:00 | 2764.20 | 2770.69 | 0.00 | ORB-short ORB[2775.00,2798.90] vol=3.1x ATR=8.05 |
| Stop hit — per-position SL triggered | 2025-11-11 11:05:00 | 2772.25 | 2770.60 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:55:00 | 2838.00 | 2819.13 | 0.00 | ORB-long ORB[2801.10,2835.90] vol=1.6x ATR=7.08 |
| Stop hit — per-position SL triggered | 2025-11-13 11:05:00 | 2830.92 | 2823.49 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:45:00 | 2770.00 | 2771.15 | 0.00 | ORB-short ORB[2780.00,2814.00] vol=2.4x ATR=7.98 |
| Stop hit — per-position SL triggered | 2025-11-14 10:55:00 | 2777.98 | 2771.46 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:10:00 | 2746.40 | 2727.83 | 0.00 | ORB-long ORB[2702.90,2736.80] vol=1.6x ATR=8.26 |
| Stop hit — per-position SL triggered | 2025-11-20 10:30:00 | 2738.14 | 2730.67 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 2741.60 | 2730.22 | 0.00 | ORB-long ORB[2714.50,2736.20] vol=5.5x ATR=8.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:10:00 | 2754.87 | 2738.94 | 0.00 | T1 1.5R @ 2754.87 |
| Stop hit — per-position SL triggered | 2025-11-21 10:25:00 | 2741.60 | 2740.15 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:55:00 | 2789.80 | 2800.78 | 0.00 | ORB-short ORB[2790.30,2807.80] vol=2.2x ATR=6.23 |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 2796.03 | 2799.70 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2845.70 | 2836.62 | 0.00 | ORB-long ORB[2812.00,2842.30] vol=1.5x ATR=7.16 |
| Stop hit — per-position SL triggered | 2025-12-02 09:40:00 | 2838.54 | 2838.02 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 2868.10 | 2852.26 | 0.00 | ORB-long ORB[2829.30,2865.30] vol=8.0x ATR=7.45 |
| Stop hit — per-position SL triggered | 2025-12-03 11:20:00 | 2860.65 | 2852.75 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:45:00 | 2897.70 | 2885.38 | 0.00 | ORB-long ORB[2865.00,2894.00] vol=2.0x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:50:00 | 2909.06 | 2896.74 | 0.00 | T1 1.5R @ 2909.06 |
| Target hit | 2025-12-04 12:45:00 | 2913.00 | 2914.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:55:00 | 2963.10 | 2938.72 | 0.00 | ORB-long ORB[2911.90,2939.00] vol=1.9x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-12-05 11:05:00 | 2956.23 | 2940.13 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 2919.90 | 2924.76 | 0.00 | ORB-short ORB[2929.20,2968.30] vol=3.1x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 2909.52 | 2923.20 | 0.00 | T1 1.5R @ 2909.52 |
| Target hit | 2025-12-08 15:00:00 | 2900.10 | 2895.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — SELL (started 2025-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:05:00 | 2846.50 | 2862.71 | 0.00 | ORB-short ORB[2875.40,2910.50] vol=1.7x ATR=9.68 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 2856.18 | 2860.81 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:15:00 | 2897.90 | 2871.25 | 0.00 | ORB-long ORB[2848.90,2881.10] vol=2.2x ATR=7.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:05:00 | 2909.64 | 2886.53 | 0.00 | T1 1.5R @ 2909.64 |
| Stop hit — per-position SL triggered | 2025-12-11 13:00:00 | 2897.90 | 2890.63 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:50:00 | 2902.30 | 2882.90 | 0.00 | ORB-long ORB[2874.80,2899.00] vol=2.1x ATR=7.95 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 2894.35 | 2884.99 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:05:00 | 2907.00 | 2926.34 | 0.00 | ORB-short ORB[2922.10,2945.00] vol=3.6x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:25:00 | 2897.78 | 2924.81 | 0.00 | T1 1.5R @ 2897.78 |
| Stop hit — per-position SL triggered | 2025-12-24 13:00:00 | 2907.00 | 2918.94 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:00:00 | 2880.50 | 2885.71 | 0.00 | ORB-short ORB[2893.30,2913.00] vol=2.5x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:10:00 | 2870.46 | 2884.04 | 0.00 | T1 1.5R @ 2870.46 |
| Target hit | 2025-12-26 15:20:00 | 2851.00 | 2864.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:15:00 | 2830.40 | 2854.33 | 0.00 | ORB-short ORB[2842.60,2868.70] vol=2.6x ATR=7.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:15:00 | 2818.86 | 2848.42 | 0.00 | T1 1.5R @ 2818.86 |
| Target hit | 2025-12-29 15:20:00 | 2797.30 | 2831.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 2784.00 | 2789.92 | 0.00 | ORB-short ORB[2785.10,2818.30] vol=1.6x ATR=6.15 |
| Stop hit — per-position SL triggered | 2025-12-31 12:30:00 | 2790.15 | 2787.97 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 2818.10 | 2807.35 | 0.00 | ORB-long ORB[2788.90,2805.40] vol=3.0x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:25:00 | 2824.58 | 2809.48 | 0.00 | T1 1.5R @ 2824.58 |
| Stop hit — per-position SL triggered | 2026-01-01 12:05:00 | 2818.10 | 2810.92 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:55:00 | 2763.10 | 2782.98 | 0.00 | ORB-short ORB[2786.20,2817.00] vol=1.9x ATR=7.73 |
| Stop hit — per-position SL triggered | 2026-01-05 10:00:00 | 2770.83 | 2782.07 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 2822.80 | 2809.90 | 0.00 | ORB-long ORB[2798.00,2818.50] vol=1.9x ATR=6.78 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 2816.02 | 2810.70 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 2820.50 | 2849.58 | 0.00 | ORB-short ORB[2857.20,2891.00] vol=1.6x ATR=8.78 |
| Stop hit — per-position SL triggered | 2026-01-08 11:05:00 | 2829.28 | 2841.57 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:00:00 | 2813.70 | 2835.31 | 0.00 | ORB-short ORB[2847.10,2882.90] vol=2.9x ATR=9.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:05:00 | 2799.26 | 2830.59 | 0.00 | T1 1.5R @ 2799.26 |
| Stop hit — per-position SL triggered | 2026-01-14 13:10:00 | 2813.70 | 2816.81 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:35:00 | 2882.30 | 2860.79 | 0.00 | ORB-long ORB[2848.90,2880.50] vol=1.5x ATR=11.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:45:00 | 2899.97 | 2868.05 | 0.00 | T1 1.5R @ 2899.97 |
| Target hit | 2026-01-19 15:20:00 | 2885.00 | 2883.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2026-01-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 11:00:00 | 2792.80 | 2778.97 | 0.00 | ORB-long ORB[2752.80,2785.90] vol=1.7x ATR=8.83 |
| Stop hit — per-position SL triggered | 2026-01-27 13:00:00 | 2783.97 | 2783.57 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:05:00 | 2664.80 | 2668.60 | 0.00 | ORB-short ORB[2665.20,2683.10] vol=1.7x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 11:30:00 | 2651.84 | 2665.83 | 0.00 | T1 1.5R @ 2651.84 |
| Target hit | 2026-02-05 14:30:00 | 2649.70 | 2649.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 2636.60 | 2629.99 | 0.00 | ORB-long ORB[2600.70,2633.90] vol=1.5x ATR=7.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:45:00 | 2648.34 | 2634.10 | 0.00 | T1 1.5R @ 2648.34 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 2636.60 | 2641.23 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:10:00 | 2433.00 | 2436.81 | 0.00 | ORB-short ORB[2436.60,2467.50] vol=1.6x ATR=9.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 2418.73 | 2432.33 | 0.00 | T1 1.5R @ 2418.73 |
| Target hit | 2026-02-19 15:20:00 | 2372.00 | 2406.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 2235.30 | 2253.25 | 0.00 | ORB-short ORB[2266.70,2299.00] vol=3.1x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 12:20:00 | 2223.13 | 2242.41 | 0.00 | T1 1.5R @ 2223.13 |
| Target hit | 2026-03-05 14:40:00 | 2226.00 | 2224.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2026-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:55:00 | 2188.90 | 2172.49 | 0.00 | ORB-long ORB[2157.00,2184.90] vol=2.1x ATR=7.38 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 2181.52 | 2173.44 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 2085.50 | 2100.68 | 0.00 | ORB-short ORB[2090.00,2118.70] vol=1.8x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2093.78 | 2099.60 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:00:00 | 2070.90 | 2081.77 | 0.00 | ORB-short ORB[2080.20,2109.70] vol=1.9x ATR=9.06 |
| Stop hit — per-position SL triggered | 2026-03-20 10:10:00 | 2079.96 | 2081.50 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 2292.00 | 2324.65 | 0.00 | ORB-short ORB[2327.50,2360.20] vol=3.0x ATR=7.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 2280.18 | 2322.11 | 0.00 | T1 1.5R @ 2280.18 |
| Stop hit — per-position SL triggered | 2026-04-23 11:35:00 | 2292.00 | 2320.65 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 2308.30 | 2290.11 | 0.00 | ORB-long ORB[2268.70,2289.20] vol=2.9x ATR=8.71 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 2299.59 | 2292.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 2599.90 | 2025-05-15 09:35:00 | 2590.81 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-23 09:30:00 | 2577.00 | 2025-05-23 09:45:00 | 2589.48 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-23 09:30:00 | 2577.00 | 2025-05-23 10:15:00 | 2577.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 10:35:00 | 2500.80 | 2025-06-03 10:50:00 | 2508.39 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-04 10:40:00 | 2543.60 | 2025-06-04 10:50:00 | 2537.34 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-05 10:50:00 | 2480.70 | 2025-06-05 11:15:00 | 2488.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-06 09:30:00 | 2515.90 | 2025-06-06 09:45:00 | 2523.61 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-10 09:30:00 | 2653.40 | 2025-06-10 09:45:00 | 2644.94 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-19 09:45:00 | 2669.70 | 2025-06-19 09:50:00 | 2657.62 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-19 09:45:00 | 2669.70 | 2025-06-19 11:20:00 | 2654.40 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-06-24 10:15:00 | 2728.00 | 2025-06-24 10:20:00 | 2717.84 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-25 09:55:00 | 2720.20 | 2025-06-25 10:55:00 | 2731.48 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-25 09:55:00 | 2720.20 | 2025-06-25 15:20:00 | 2750.30 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2025-06-30 10:45:00 | 2859.60 | 2025-06-30 10:50:00 | 2848.37 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-01 09:30:00 | 2881.60 | 2025-07-01 09:35:00 | 2872.19 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-04 11:10:00 | 2918.90 | 2025-07-04 11:20:00 | 2912.42 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-15 11:00:00 | 2876.40 | 2025-07-15 11:15:00 | 2866.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-21 10:35:00 | 2791.60 | 2025-07-21 10:55:00 | 2799.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-05 11:10:00 | 2705.10 | 2025-08-05 12:00:00 | 2712.63 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-06 11:00:00 | 2656.70 | 2025-08-06 11:15:00 | 2663.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-08 10:10:00 | 2662.30 | 2025-08-08 11:00:00 | 2670.33 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-14 10:05:00 | 2700.00 | 2025-08-14 10:30:00 | 2706.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-20 10:00:00 | 2758.00 | 2025-08-20 10:15:00 | 2771.93 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-20 10:00:00 | 2758.00 | 2025-08-20 15:20:00 | 2834.10 | TARGET_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2025-08-21 10:40:00 | 2871.70 | 2025-08-21 11:25:00 | 2881.82 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-08-21 10:40:00 | 2871.70 | 2025-08-21 12:25:00 | 2871.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 11:00:00 | 2863.10 | 2025-08-22 11:15:00 | 2873.40 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-08-22 11:00:00 | 2863.10 | 2025-08-22 12:25:00 | 2863.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-26 10:20:00 | 2929.30 | 2025-08-26 10:30:00 | 2920.08 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-02 10:50:00 | 2930.50 | 2025-09-02 11:45:00 | 2940.42 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-02 10:50:00 | 2930.50 | 2025-09-02 12:40:00 | 2930.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-03 10:05:00 | 2866.30 | 2025-09-03 10:15:00 | 2853.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-03 10:05:00 | 2866.30 | 2025-09-03 10:40:00 | 2866.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:10:00 | 2781.20 | 2025-09-05 10:15:00 | 2761.05 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-09-05 10:10:00 | 2781.20 | 2025-09-05 10:20:00 | 2781.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 10:00:00 | 2959.00 | 2025-09-16 10:40:00 | 2951.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-29 11:15:00 | 2670.00 | 2025-09-29 11:20:00 | 2662.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-03 10:55:00 | 2717.40 | 2025-10-03 12:20:00 | 2710.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-06 10:15:00 | 2774.60 | 2025-10-06 10:25:00 | 2766.67 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-13 10:40:00 | 2749.80 | 2025-10-13 10:55:00 | 2740.34 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-13 10:40:00 | 2749.80 | 2025-10-13 12:25:00 | 2749.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:05:00 | 2744.80 | 2025-10-14 11:40:00 | 2751.56 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-16 10:25:00 | 2756.40 | 2025-10-16 10:45:00 | 2763.32 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-17 09:30:00 | 2723.60 | 2025-10-17 09:45:00 | 2709.82 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-17 09:30:00 | 2723.60 | 2025-10-17 10:10:00 | 2723.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 11:10:00 | 2717.60 | 2025-10-20 12:00:00 | 2726.98 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-20 11:10:00 | 2717.60 | 2025-10-20 12:20:00 | 2717.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:55:00 | 2876.30 | 2025-10-27 10:05:00 | 2891.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-27 09:55:00 | 2876.30 | 2025-10-27 10:20:00 | 2876.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 11:10:00 | 2821.30 | 2025-10-29 11:35:00 | 2829.24 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-04 11:10:00 | 2763.10 | 2025-11-04 13:35:00 | 2769.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-10 10:15:00 | 2766.40 | 2025-11-10 10:20:00 | 2778.18 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-10 10:15:00 | 2766.40 | 2025-11-10 12:05:00 | 2776.20 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-11 10:20:00 | 2764.20 | 2025-11-11 11:05:00 | 2772.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-13 10:55:00 | 2838.00 | 2025-11-13 11:05:00 | 2830.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-14 10:45:00 | 2770.00 | 2025-11-14 10:55:00 | 2777.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-20 10:10:00 | 2746.40 | 2025-11-20 10:30:00 | 2738.14 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-21 09:45:00 | 2741.60 | 2025-11-21 10:10:00 | 2754.87 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-11-21 09:45:00 | 2741.60 | 2025-11-21 10:25:00 | 2741.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-28 10:55:00 | 2789.80 | 2025-11-28 11:15:00 | 2796.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-02 09:30:00 | 2845.70 | 2025-12-02 09:40:00 | 2838.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-03 11:10:00 | 2868.10 | 2025-12-03 11:20:00 | 2860.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-04 09:45:00 | 2897.70 | 2025-12-04 09:50:00 | 2909.06 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-04 09:45:00 | 2897.70 | 2025-12-04 12:45:00 | 2913.00 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-05 10:55:00 | 2963.10 | 2025-12-05 11:05:00 | 2956.23 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-08 11:10:00 | 2919.90 | 2025-12-08 11:15:00 | 2909.52 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-08 11:10:00 | 2919.90 | 2025-12-08 15:00:00 | 2900.10 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-12-09 10:05:00 | 2846.50 | 2025-12-09 10:15:00 | 2856.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-11 11:15:00 | 2897.90 | 2025-12-11 12:05:00 | 2909.64 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-12-11 11:15:00 | 2897.90 | 2025-12-11 13:00:00 | 2897.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-15 10:50:00 | 2902.30 | 2025-12-15 11:05:00 | 2894.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-24 11:05:00 | 2907.00 | 2025-12-24 11:25:00 | 2897.78 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-24 11:05:00 | 2907.00 | 2025-12-24 13:00:00 | 2907.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 11:00:00 | 2880.50 | 2025-12-26 11:10:00 | 2870.46 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-26 11:00:00 | 2880.50 | 2025-12-26 15:20:00 | 2851.00 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2025-12-29 11:15:00 | 2830.40 | 2025-12-29 12:15:00 | 2818.86 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-29 11:15:00 | 2830.40 | 2025-12-29 15:20:00 | 2797.30 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2025-12-31 10:55:00 | 2784.00 | 2025-12-31 12:30:00 | 2790.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-01 11:15:00 | 2818.10 | 2026-01-01 11:25:00 | 2824.58 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2026-01-01 11:15:00 | 2818.10 | 2026-01-01 12:05:00 | 2818.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-05 09:55:00 | 2763.10 | 2026-01-05 10:00:00 | 2770.83 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-06 11:05:00 | 2822.80 | 2026-01-06 11:15:00 | 2816.02 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-08 10:45:00 | 2820.50 | 2026-01-08 11:05:00 | 2829.28 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-14 11:00:00 | 2813.70 | 2026-01-14 11:05:00 | 2799.26 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-14 11:00:00 | 2813.70 | 2026-01-14 13:10:00 | 2813.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 10:35:00 | 2882.30 | 2026-01-19 10:45:00 | 2899.97 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-01-19 10:35:00 | 2882.30 | 2026-01-19 15:20:00 | 2885.00 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2026-01-27 11:00:00 | 2792.80 | 2026-01-27 13:00:00 | 2783.97 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-05 11:05:00 | 2664.80 | 2026-02-05 11:30:00 | 2651.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-05 11:05:00 | 2664.80 | 2026-02-05 14:30:00 | 2649.70 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-10 10:05:00 | 2636.60 | 2026-02-10 10:45:00 | 2648.34 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 10:05:00 | 2636.60 | 2026-02-10 11:10:00 | 2636.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:10:00 | 2433.00 | 2026-02-19 11:15:00 | 2418.73 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-19 10:10:00 | 2433.00 | 2026-02-19 15:20:00 | 2372.00 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2026-03-05 10:50:00 | 2235.30 | 2026-03-05 12:20:00 | 2223.13 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-05 10:50:00 | 2235.30 | 2026-03-05 14:40:00 | 2226.00 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-12 10:55:00 | 2188.90 | 2026-03-12 11:00:00 | 2181.52 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-16 11:00:00 | 2085.50 | 2026-03-16 11:15:00 | 2093.78 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-20 10:00:00 | 2070.90 | 2026-03-20 10:10:00 | 2079.96 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-23 11:10:00 | 2292.00 | 2026-04-23 11:20:00 | 2280.18 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-23 11:10:00 | 2292.00 | 2026-04-23 11:35:00 | 2292.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:50:00 | 2308.30 | 2026-04-29 09:55:00 | 2299.59 | STOP_HIT | 1.00 | -0.38% |
