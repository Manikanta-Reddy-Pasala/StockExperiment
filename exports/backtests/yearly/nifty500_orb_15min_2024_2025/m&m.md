# Mahindra & Mahindra Ltd. (M&M)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 3331.50
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 19 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 44
- **Target hits / Stop hits / Partials:** 19 / 44 / 34
- **Avg / median % per leg:** 0.27% / 0.26%
- **Sum % (uncompounded):** 26.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 32 | 58.2% | 12 | 23 | 20 | 0.29% | 15.8% |
| BUY @ 2nd Alert (retest1) | 55 | 32 | 58.2% | 12 | 23 | 20 | 0.29% | 15.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 21 | 50.0% | 7 | 21 | 14 | 0.25% | 10.4% |
| SELL @ 2nd Alert (retest1) | 42 | 21 | 50.0% | 7 | 21 | 14 | 0.25% | 10.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 97 | 53 | 54.6% | 19 | 44 | 34 | 0.27% | 26.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:05:00 | 2287.50 | 2272.42 | 0.00 | ORB-long ORB[2256.00,2286.95] vol=1.9x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:15:00 | 2296.22 | 2274.35 | 0.00 | T1 1.5R @ 2296.22 |
| Target hit | 2024-05-15 15:20:00 | 2301.00 | 2291.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 2498.00 | 2507.31 | 0.00 | ORB-short ORB[2498.05,2521.70] vol=1.7x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 2505.73 | 2506.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:15:00 | 2818.25 | 2829.83 | 0.00 | ORB-short ORB[2828.50,2848.60] vol=3.8x ATR=7.75 |
| Stop hit — per-position SL triggered | 2024-06-12 10:25:00 | 2826.00 | 2827.88 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 2999.25 | 2977.31 | 0.00 | ORB-long ORB[2956.65,2986.70] vol=1.8x ATR=11.46 |
| Stop hit — per-position SL triggered | 2024-06-18 10:30:00 | 2987.79 | 2988.07 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 2943.95 | 2926.70 | 0.00 | ORB-long ORB[2910.00,2927.90] vol=2.5x ATR=9.17 |
| Stop hit — per-position SL triggered | 2024-06-25 09:40:00 | 2934.78 | 2930.43 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 2809.45 | 2823.89 | 0.00 | ORB-short ORB[2812.15,2850.80] vol=2.1x ATR=8.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:40:00 | 2796.90 | 2817.41 | 0.00 | T1 1.5R @ 2796.90 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 2809.45 | 2816.37 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 2843.60 | 2853.23 | 0.00 | ORB-short ORB[2843.90,2886.00] vol=1.7x ATR=9.33 |
| Stop hit — per-position SL triggered | 2024-07-05 10:20:00 | 2852.93 | 2852.51 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:45:00 | 2912.60 | 2898.53 | 0.00 | ORB-long ORB[2871.00,2904.10] vol=2.4x ATR=8.98 |
| Stop hit — per-position SL triggered | 2024-07-08 09:50:00 | 2903.62 | 2898.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 11:00:00 | 2765.30 | 2754.37 | 0.00 | ORB-long ORB[2742.10,2764.00] vol=1.6x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:30:00 | 2776.07 | 2758.61 | 0.00 | T1 1.5R @ 2776.07 |
| Target hit | 2024-07-18 15:20:00 | 2820.00 | 2788.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-08-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 11:05:00 | 2772.60 | 2788.52 | 0.00 | ORB-short ORB[2780.90,2810.30] vol=1.9x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 12:00:00 | 2761.53 | 2783.53 | 0.00 | T1 1.5R @ 2761.53 |
| Target hit | 2024-08-02 15:20:00 | 2758.75 | 2765.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:00:00 | 2678.35 | 2668.19 | 0.00 | ORB-long ORB[2656.50,2670.55] vol=5.9x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:10:00 | 2689.14 | 2669.16 | 0.00 | T1 1.5R @ 2689.14 |
| Target hit | 2024-08-08 13:05:00 | 2681.10 | 2684.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 11:00:00 | 2715.15 | 2727.65 | 0.00 | ORB-short ORB[2724.20,2757.00] vol=4.8x ATR=7.43 |
| Stop hit — per-position SL triggered | 2024-08-12 11:10:00 | 2722.58 | 2726.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:45:00 | 2766.00 | 2749.81 | 0.00 | ORB-long ORB[2715.30,2756.25] vol=2.3x ATR=7.68 |
| Stop hit — per-position SL triggered | 2024-08-14 11:30:00 | 2758.32 | 2753.18 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:00:00 | 2750.00 | 2763.09 | 0.00 | ORB-short ORB[2756.85,2783.20] vol=1.5x ATR=5.51 |
| Stop hit — per-position SL triggered | 2024-08-20 11:10:00 | 2755.51 | 2762.55 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 2757.90 | 2762.52 | 0.00 | ORB-short ORB[2758.65,2780.95] vol=4.9x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-08-21 11:25:00 | 2761.46 | 2761.29 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 2767.70 | 2782.00 | 0.00 | ORB-short ORB[2773.45,2801.65] vol=2.7x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:45:00 | 2760.48 | 2778.93 | 0.00 | T1 1.5R @ 2760.48 |
| Stop hit — per-position SL triggered | 2024-08-29 12:25:00 | 2767.70 | 2775.74 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:30:00 | 2797.25 | 2779.44 | 0.00 | ORB-long ORB[2765.85,2780.00] vol=2.0x ATR=7.97 |
| Stop hit — per-position SL triggered | 2024-08-30 10:40:00 | 2789.28 | 2780.70 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 2733.00 | 2739.75 | 0.00 | ORB-short ORB[2738.15,2760.00] vol=2.5x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:30:00 | 2726.65 | 2737.92 | 0.00 | T1 1.5R @ 2726.65 |
| Stop hit — per-position SL triggered | 2024-09-05 10:45:00 | 2733.00 | 2737.04 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:50:00 | 2685.35 | 2711.37 | 0.00 | ORB-short ORB[2709.00,2735.95] vol=2.4x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 2693.08 | 2707.47 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:05:00 | 2681.75 | 2691.84 | 0.00 | ORB-short ORB[2690.00,2705.65] vol=2.3x ATR=5.85 |
| Stop hit — per-position SL triggered | 2024-09-10 11:10:00 | 2687.60 | 2687.72 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:40:00 | 2701.05 | 2693.73 | 0.00 | ORB-long ORB[2673.00,2691.90] vol=5.3x ATR=6.64 |
| Stop hit — per-position SL triggered | 2024-09-11 09:45:00 | 2694.41 | 2693.83 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:40:00 | 2775.40 | 2762.22 | 0.00 | ORB-long ORB[2745.30,2773.45] vol=1.8x ATR=6.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:50:00 | 2785.06 | 2768.71 | 0.00 | T1 1.5R @ 2785.06 |
| Stop hit — per-position SL triggered | 2024-09-17 12:40:00 | 2775.40 | 2772.04 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:40:00 | 2811.00 | 2798.85 | 0.00 | ORB-long ORB[2777.65,2797.85] vol=1.6x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:25:00 | 2820.02 | 2809.14 | 0.00 | T1 1.5R @ 2820.02 |
| Target hit | 2024-09-18 13:05:00 | 2816.25 | 2820.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 2826.50 | 2817.03 | 0.00 | ORB-long ORB[2808.80,2822.95] vol=1.8x ATR=7.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:45:00 | 2837.83 | 2821.69 | 0.00 | T1 1.5R @ 2837.83 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 2826.50 | 2822.87 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 3073.70 | 3060.50 | 0.00 | ORB-long ORB[3020.00,3063.10] vol=1.9x ATR=10.42 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 3063.28 | 3060.94 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 3002.70 | 3023.40 | 0.00 | ORB-short ORB[3020.85,3046.15] vol=1.6x ATR=13.12 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 3015.82 | 3022.44 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 3154.70 | 3143.70 | 0.00 | ORB-long ORB[3115.00,3154.00] vol=1.9x ATR=7.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:10:00 | 3166.67 | 3147.74 | 0.00 | T1 1.5R @ 3166.67 |
| Stop hit — per-position SL triggered | 2024-10-14 10:20:00 | 3154.70 | 3148.13 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 2900.50 | 2927.10 | 0.00 | ORB-short ORB[2926.90,2963.95] vol=1.8x ATR=10.43 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 2910.93 | 2921.96 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:00:00 | 2967.45 | 2980.04 | 0.00 | ORB-short ORB[2978.25,3008.95] vol=2.1x ATR=10.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:05:00 | 2952.31 | 2976.79 | 0.00 | T1 1.5R @ 2952.31 |
| Target hit | 2024-10-22 11:00:00 | 2962.20 | 2960.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2024-10-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:45:00 | 2759.90 | 2780.16 | 0.00 | ORB-short ORB[2785.00,2825.05] vol=3.2x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:00:00 | 2743.25 | 2775.67 | 0.00 | T1 1.5R @ 2743.25 |
| Target hit | 2024-10-25 15:05:00 | 2719.70 | 2713.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:10:00 | 2713.95 | 2691.92 | 0.00 | ORB-long ORB[2688.05,2708.90] vol=2.1x ATR=8.22 |
| Stop hit — per-position SL triggered | 2024-10-31 11:20:00 | 2705.73 | 2694.82 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:15:00 | 3011.50 | 3027.11 | 0.00 | ORB-short ORB[3038.95,3068.10] vol=1.6x ATR=9.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:55:00 | 2997.06 | 3024.75 | 0.00 | T1 1.5R @ 2997.06 |
| Target hit | 2024-11-26 14:55:00 | 3001.00 | 3000.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 2934.10 | 2970.45 | 0.00 | ORB-short ORB[2972.20,2999.95] vol=1.7x ATR=11.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:25:00 | 2917.13 | 2958.73 | 0.00 | T1 1.5R @ 2917.13 |
| Stop hit — per-position SL triggered | 2024-11-28 11:50:00 | 2934.10 | 2955.13 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:15:00 | 3048.10 | 3034.37 | 0.00 | ORB-long ORB[3003.00,3045.70] vol=2.4x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 3040.37 | 3034.77 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:35:00 | 3090.15 | 3073.04 | 0.00 | ORB-long ORB[3047.80,3083.50] vol=1.7x ATR=9.56 |
| Stop hit — per-position SL triggered | 2024-12-11 09:45:00 | 3080.59 | 3074.85 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:45:00 | 3067.65 | 3075.93 | 0.00 | ORB-short ORB[3069.00,3084.95] vol=1.8x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:00:00 | 3058.34 | 3073.82 | 0.00 | T1 1.5R @ 3058.34 |
| Stop hit — per-position SL triggered | 2024-12-12 11:10:00 | 3067.65 | 3073.04 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 3020.20 | 3024.22 | 0.00 | ORB-short ORB[3038.00,3062.90] vol=1.5x ATR=9.68 |
| Stop hit — per-position SL triggered | 2024-12-13 11:25:00 | 3029.88 | 3024.50 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:55:00 | 2938.10 | 2929.30 | 0.00 | ORB-long ORB[2900.80,2921.00] vol=2.1x ATR=7.97 |
| Stop hit — per-position SL triggered | 2024-12-24 13:40:00 | 2930.13 | 2933.03 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 2964.75 | 2949.40 | 0.00 | ORB-long ORB[2923.35,2958.00] vol=1.6x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:25:00 | 2977.91 | 2953.30 | 0.00 | T1 1.5R @ 2977.91 |
| Stop hit — per-position SL triggered | 2024-12-26 13:30:00 | 2964.75 | 2961.21 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 3037.00 | 3014.94 | 0.00 | ORB-long ORB[2978.05,3021.00] vol=2.1x ATR=8.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:00:00 | 3050.22 | 3021.22 | 0.00 | T1 1.5R @ 3050.22 |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 3037.00 | 3033.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:45:00 | 3045.00 | 3015.12 | 0.00 | ORB-long ORB[2995.20,3018.55] vol=2.3x ATR=9.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:50:00 | 3058.61 | 3020.96 | 0.00 | T1 1.5R @ 3058.61 |
| Target hit | 2025-01-01 15:20:00 | 3082.40 | 3057.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2025-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:00:00 | 3118.00 | 3100.11 | 0.00 | ORB-long ORB[3085.05,3107.00] vol=1.5x ATR=8.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:05:00 | 3131.42 | 3105.61 | 0.00 | T1 1.5R @ 3131.42 |
| Target hit | 2025-01-02 15:20:00 | 3208.00 | 3185.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:05:00 | 3133.30 | 3196.03 | 0.00 | ORB-short ORB[3193.00,3235.00] vol=1.6x ATR=13.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 3113.36 | 3193.01 | 0.00 | T1 1.5R @ 3113.36 |
| Stop hit — per-position SL triggered | 2025-01-06 12:40:00 | 3133.30 | 3166.49 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 3129.75 | 3116.51 | 0.00 | ORB-long ORB[3085.35,3123.15] vol=2.6x ATR=11.35 |
| Stop hit — per-position SL triggered | 2025-01-09 11:05:00 | 3118.40 | 3117.64 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 3057.15 | 3034.40 | 0.00 | ORB-long ORB[2999.65,3044.20] vol=1.5x ATR=9.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 11:10:00 | 3071.91 | 3037.80 | 0.00 | T1 1.5R @ 3071.91 |
| Stop hit — per-position SL triggered | 2025-01-14 11:30:00 | 3057.15 | 3041.61 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:05:00 | 2884.05 | 2899.47 | 0.00 | ORB-short ORB[2902.70,2935.95] vol=1.8x ATR=9.33 |
| Stop hit — per-position SL triggered | 2025-01-20 10:20:00 | 2893.38 | 2896.83 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 2846.25 | 2880.52 | 0.00 | ORB-short ORB[2888.10,2905.00] vol=2.7x ATR=9.94 |
| Stop hit — per-position SL triggered | 2025-01-21 10:25:00 | 2856.19 | 2878.18 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:00:00 | 2871.00 | 2870.72 | 0.00 | ORB-long ORB[2810.00,2847.75] vol=1.6x ATR=10.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 12:30:00 | 2886.85 | 2871.92 | 0.00 | T1 1.5R @ 2886.85 |
| Target hit | 2025-01-23 15:20:00 | 2884.00 | 2881.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 2855.00 | 2875.73 | 0.00 | ORB-short ORB[2875.00,2896.00] vol=1.9x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-01-24 11:10:00 | 2864.19 | 2865.65 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:50:00 | 2948.00 | 2935.90 | 0.00 | ORB-long ORB[2919.00,2942.00] vol=1.7x ATR=8.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:20:00 | 2961.46 | 2942.86 | 0.00 | T1 1.5R @ 2961.46 |
| Target hit | 2025-01-30 14:00:00 | 2970.00 | 2972.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 3047.25 | 3023.35 | 0.00 | ORB-long ORB[2990.00,3019.35] vol=1.8x ATR=11.27 |
| Stop hit — per-position SL triggered | 2025-02-01 09:40:00 | 3035.98 | 3031.57 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:25:00 | 2975.70 | 2984.66 | 0.00 | ORB-short ORB[2977.75,3020.00] vol=1.5x ATR=10.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:35:00 | 2960.04 | 2982.75 | 0.00 | T1 1.5R @ 2960.04 |
| Target hit | 2025-02-14 14:40:00 | 2951.05 | 2949.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2025-02-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:50:00 | 2738.10 | 2780.73 | 0.00 | ORB-short ORB[2775.00,2815.20] vol=2.0x ATR=11.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:05:00 | 2720.85 | 2766.43 | 0.00 | T1 1.5R @ 2720.85 |
| Target hit | 2025-02-21 15:20:00 | 2667.35 | 2693.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 2769.55 | 2750.58 | 0.00 | ORB-long ORB[2728.15,2769.00] vol=1.7x ATR=9.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:25:00 | 2783.31 | 2760.14 | 0.00 | T1 1.5R @ 2783.31 |
| Target hit | 2025-02-25 15:10:00 | 2777.60 | 2778.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2025-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:00:00 | 2753.10 | 2747.04 | 0.00 | ORB-long ORB[2715.05,2743.95] vol=2.1x ATR=8.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:10:00 | 2765.56 | 2747.73 | 0.00 | T1 1.5R @ 2765.56 |
| Target hit | 2025-03-18 15:20:00 | 2791.55 | 2769.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2025-03-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:00:00 | 2865.40 | 2847.06 | 0.00 | ORB-long ORB[2827.00,2859.70] vol=1.7x ATR=7.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:30:00 | 2875.92 | 2850.62 | 0.00 | T1 1.5R @ 2875.92 |
| Stop hit — per-position SL triggered | 2025-03-21 12:05:00 | 2865.40 | 2859.85 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:45:00 | 2773.35 | 2761.31 | 0.00 | ORB-long ORB[2741.20,2762.00] vol=1.8x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 12:40:00 | 2785.88 | 2767.00 | 0.00 | T1 1.5R @ 2785.88 |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 2773.35 | 2770.83 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 11:15:00 | 2638.45 | 2628.74 | 0.00 | ORB-long ORB[2600.25,2627.95] vol=1.7x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-04-03 12:20:00 | 2632.95 | 2631.01 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 10:45:00 | 2565.75 | 2545.20 | 0.00 | ORB-long ORB[2521.40,2555.00] vol=1.9x ATR=11.71 |
| Stop hit — per-position SL triggered | 2025-04-09 11:00:00 | 2554.04 | 2546.67 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:35:00 | 2731.00 | 2700.02 | 0.00 | ORB-long ORB[2665.90,2699.90] vol=1.5x ATR=8.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:40:00 | 2743.99 | 2703.51 | 0.00 | T1 1.5R @ 2743.99 |
| Target hit | 2025-04-21 15:20:00 | 2766.00 | 2741.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:45:00 | 2796.10 | 2776.81 | 0.00 | ORB-long ORB[2756.00,2776.20] vol=2.1x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:20:00 | 2806.90 | 2784.76 | 0.00 | T1 1.5R @ 2806.90 |
| Target hit | 2025-04-22 15:20:00 | 2819.30 | 2799.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 11:15:00 | 2889.20 | 2898.97 | 0.00 | ORB-short ORB[2891.70,2928.00] vol=2.6x ATR=7.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 12:10:00 | 2878.39 | 2896.84 | 0.00 | T1 1.5R @ 2878.39 |
| Stop hit — per-position SL triggered | 2025-04-24 13:25:00 | 2889.20 | 2891.38 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:50:00 | 3089.00 | 3105.37 | 0.00 | ORB-short ORB[3115.60,3137.00] vol=2.5x ATR=10.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:35:00 | 3073.82 | 3099.67 | 0.00 | T1 1.5R @ 3073.82 |
| Target hit | 2025-05-08 15:20:00 | 3022.20 | 3059.46 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:05:00 | 2287.50 | 2024-05-15 11:15:00 | 2296.22 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-05-15 11:05:00 | 2287.50 | 2024-05-15 15:20:00 | 2301.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2024-05-22 09:40:00 | 2498.00 | 2024-05-22 09:45:00 | 2505.73 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-12 10:15:00 | 2818.25 | 2024-06-12 10:25:00 | 2826.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-18 09:35:00 | 2999.25 | 2024-06-18 10:30:00 | 2987.79 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-25 09:30:00 | 2943.95 | 2024-06-25 09:40:00 | 2934.78 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-27 09:30:00 | 2809.45 | 2024-06-27 09:40:00 | 2796.90 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-27 09:30:00 | 2809.45 | 2024-06-27 09:45:00 | 2809.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:50:00 | 2843.60 | 2024-07-05 10:20:00 | 2852.93 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-08 09:45:00 | 2912.60 | 2024-07-08 09:50:00 | 2903.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-18 11:00:00 | 2765.30 | 2024-07-18 11:30:00 | 2776.07 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-18 11:00:00 | 2765.30 | 2024-07-18 15:20:00 | 2820.00 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2024-08-02 11:05:00 | 2772.60 | 2024-08-02 12:00:00 | 2761.53 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-02 11:05:00 | 2772.60 | 2024-08-02 15:20:00 | 2758.75 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-08 11:00:00 | 2678.35 | 2024-08-08 11:10:00 | 2689.14 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-08 11:00:00 | 2678.35 | 2024-08-08 13:05:00 | 2681.10 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-08-12 11:00:00 | 2715.15 | 2024-08-12 11:10:00 | 2722.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-14 10:45:00 | 2766.00 | 2024-08-14 11:30:00 | 2758.32 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-20 11:00:00 | 2750.00 | 2024-08-20 11:10:00 | 2755.51 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-21 11:00:00 | 2757.90 | 2024-08-21 11:25:00 | 2761.46 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2024-08-29 11:10:00 | 2767.70 | 2024-08-29 11:45:00 | 2760.48 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-08-29 11:10:00 | 2767.70 | 2024-08-29 12:25:00 | 2767.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:30:00 | 2797.25 | 2024-08-30 10:40:00 | 2789.28 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-05 10:15:00 | 2733.00 | 2024-09-05 10:30:00 | 2726.65 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-09-05 10:15:00 | 2733.00 | 2024-09-05 10:45:00 | 2733.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:50:00 | 2685.35 | 2024-09-06 11:40:00 | 2693.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-10 10:05:00 | 2681.75 | 2024-09-10 11:10:00 | 2687.60 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-11 09:40:00 | 2701.05 | 2024-09-11 09:45:00 | 2694.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-17 10:40:00 | 2775.40 | 2024-09-17 11:50:00 | 2785.06 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-17 10:40:00 | 2775.40 | 2024-09-17 12:40:00 | 2775.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 09:40:00 | 2811.00 | 2024-09-18 10:25:00 | 2820.02 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-09-18 09:40:00 | 2811.00 | 2024-09-18 13:05:00 | 2816.25 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2024-09-19 09:30:00 | 2826.50 | 2024-09-19 09:45:00 | 2837.83 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-19 09:30:00 | 2826.50 | 2024-09-19 09:50:00 | 2826.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 11:00:00 | 3073.70 | 2024-09-24 11:05:00 | 3063.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-07 11:05:00 | 3002.70 | 2024-10-07 11:15:00 | 3015.82 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-14 09:45:00 | 3154.70 | 2024-10-14 10:10:00 | 3166.67 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-10-14 09:45:00 | 3154.70 | 2024-10-14 10:20:00 | 3154.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:45:00 | 2900.50 | 2024-10-21 09:55:00 | 2910.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-22 10:00:00 | 2967.45 | 2024-10-22 10:05:00 | 2952.31 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-22 10:00:00 | 2967.45 | 2024-10-22 11:00:00 | 2962.20 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-10-25 09:45:00 | 2759.90 | 2024-10-25 10:00:00 | 2743.25 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-25 09:45:00 | 2759.90 | 2024-10-25 15:05:00 | 2719.70 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2024-10-31 11:10:00 | 2713.95 | 2024-10-31 11:20:00 | 2705.73 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-26 11:15:00 | 3011.50 | 2024-11-26 11:55:00 | 2997.06 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-26 11:15:00 | 3011.50 | 2024-11-26 14:55:00 | 3001.00 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2024-11-28 10:35:00 | 2934.10 | 2024-11-28 11:25:00 | 2917.13 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-28 10:35:00 | 2934.10 | 2024-11-28 11:50:00 | 2934.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 11:15:00 | 3048.10 | 2024-12-04 11:55:00 | 3040.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-11 09:35:00 | 3090.15 | 2024-12-11 09:45:00 | 3080.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-12 10:45:00 | 3067.65 | 2024-12-12 11:00:00 | 3058.34 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-12 10:45:00 | 3067.65 | 2024-12-12 11:10:00 | 3067.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:10:00 | 3020.20 | 2024-12-13 11:25:00 | 3029.88 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-24 10:55:00 | 2938.10 | 2024-12-24 13:40:00 | 2930.13 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-26 10:55:00 | 2964.75 | 2024-12-26 11:25:00 | 2977.91 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-26 10:55:00 | 2964.75 | 2024-12-26 13:30:00 | 2964.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:55:00 | 3037.00 | 2024-12-27 10:00:00 | 3050.22 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-27 09:55:00 | 3037.00 | 2024-12-27 11:15:00 | 3037.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:45:00 | 3045.00 | 2025-01-01 10:50:00 | 3058.61 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-01 10:45:00 | 3045.00 | 2025-01-01 15:20:00 | 3082.40 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-01-02 10:00:00 | 3118.00 | 2025-01-02 10:05:00 | 3131.42 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-02 10:00:00 | 3118.00 | 2025-01-02 15:20:00 | 3208.00 | TARGET_HIT | 0.50 | 2.89% |
| SELL | retest1 | 2025-01-06 11:05:00 | 3133.30 | 2025-01-06 11:10:00 | 3113.36 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-01-06 11:05:00 | 3133.30 | 2025-01-06 12:40:00 | 3133.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 10:45:00 | 3129.75 | 2025-01-09 11:05:00 | 3118.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-14 11:00:00 | 3057.15 | 2025-01-14 11:10:00 | 3071.91 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-14 11:00:00 | 3057.15 | 2025-01-14 11:30:00 | 3057.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 10:05:00 | 2884.05 | 2025-01-20 10:20:00 | 2893.38 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-21 10:20:00 | 2846.25 | 2025-01-21 10:25:00 | 2856.19 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-23 11:00:00 | 2871.00 | 2025-01-23 12:30:00 | 2886.85 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-01-23 11:00:00 | 2871.00 | 2025-01-23 15:20:00 | 2884.00 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-24 09:45:00 | 2855.00 | 2025-01-24 11:10:00 | 2864.19 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-30 09:50:00 | 2948.00 | 2025-01-30 10:20:00 | 2961.46 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-30 09:50:00 | 2948.00 | 2025-01-30 14:00:00 | 2970.00 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2025-02-01 09:30:00 | 3047.25 | 2025-02-01 09:40:00 | 3035.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-02-14 10:25:00 | 2975.70 | 2025-02-14 10:35:00 | 2960.04 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-02-14 10:25:00 | 2975.70 | 2025-02-14 14:40:00 | 2951.05 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-02-21 09:50:00 | 2738.10 | 2025-02-21 10:05:00 | 2720.85 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-02-21 09:50:00 | 2738.10 | 2025-02-21 15:20:00 | 2667.35 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2025-02-25 09:30:00 | 2769.55 | 2025-02-25 10:25:00 | 2783.31 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-02-25 09:30:00 | 2769.55 | 2025-02-25 15:10:00 | 2777.60 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-03-18 11:00:00 | 2753.10 | 2025-03-18 11:10:00 | 2765.56 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-18 11:00:00 | 2753.10 | 2025-03-18 15:20:00 | 2791.55 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2025-03-21 11:00:00 | 2865.40 | 2025-03-21 11:30:00 | 2875.92 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-21 11:00:00 | 2865.40 | 2025-03-21 12:05:00 | 2865.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 10:45:00 | 2773.35 | 2025-03-26 12:40:00 | 2785.88 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-26 10:45:00 | 2773.35 | 2025-03-26 14:15:00 | 2773.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-03 11:15:00 | 2638.45 | 2025-04-03 12:20:00 | 2632.95 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-04-09 10:45:00 | 2565.75 | 2025-04-09 11:00:00 | 2554.04 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-21 10:35:00 | 2731.00 | 2025-04-21 10:40:00 | 2743.99 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-21 10:35:00 | 2731.00 | 2025-04-21 15:20:00 | 2766.00 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-04-22 10:45:00 | 2796.10 | 2025-04-22 11:20:00 | 2806.90 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-22 10:45:00 | 2796.10 | 2025-04-22 15:20:00 | 2819.30 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-04-24 11:15:00 | 2889.20 | 2025-04-24 12:10:00 | 2878.39 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-04-24 11:15:00 | 2889.20 | 2025-04-24 13:25:00 | 2889.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-08 09:50:00 | 3089.00 | 2025-05-08 10:35:00 | 3073.82 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-08 09:50:00 | 3089.00 | 2025-05-08 15:20:00 | 3022.20 | TARGET_HIT | 0.50 | 2.16% |
