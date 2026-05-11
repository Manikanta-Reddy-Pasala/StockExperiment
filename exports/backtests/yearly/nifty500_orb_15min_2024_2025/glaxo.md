# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 2480.40
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 10 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 55
- **Target hits / Stop hits / Partials:** 10 / 55 / 27
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 16.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 11 | 34.4% | 2 | 21 | 9 | 0.12% | 3.7% |
| BUY @ 2nd Alert (retest1) | 32 | 11 | 34.4% | 2 | 21 | 9 | 0.12% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 26 | 43.3% | 8 | 34 | 18 | 0.22% | 13.0% |
| SELL @ 2nd Alert (retest1) | 60 | 26 | 43.3% | 8 | 34 | 18 | 0.22% | 13.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 92 | 37 | 40.2% | 10 | 55 | 27 | 0.18% | 16.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:40:00 | 1967.95 | 1984.14 | 0.00 | ORB-short ORB[1975.00,2002.00] vol=1.9x ATR=11.61 |
| Stop hit — per-position SL triggered | 2024-05-13 10:50:00 | 1979.56 | 1983.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 2000.00 | 2009.20 | 0.00 | ORB-short ORB[2005.80,2020.00] vol=2.8x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:05:00 | 1990.14 | 2003.39 | 0.00 | T1 1.5R @ 1990.14 |
| Stop hit — per-position SL triggered | 2024-05-14 11:35:00 | 2000.00 | 2000.84 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:45:00 | 1980.00 | 1988.94 | 0.00 | ORB-short ORB[1985.15,1999.35] vol=2.1x ATR=6.12 |
| Stop hit — per-position SL triggered | 2024-05-15 11:40:00 | 1986.12 | 1981.24 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:30:00 | 2051.05 | 2041.00 | 0.00 | ORB-long ORB[2026.15,2042.60] vol=4.0x ATR=8.15 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 2042.90 | 2041.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:55:00 | 2300.00 | 2345.97 | 0.00 | ORB-short ORB[2342.90,2369.00] vol=3.0x ATR=11.94 |
| Stop hit — per-position SL triggered | 2024-05-22 11:10:00 | 2311.94 | 2335.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 10:05:00 | 2602.45 | 2579.58 | 0.00 | ORB-long ORB[2546.25,2579.90] vol=2.0x ATR=15.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:10:00 | 2624.98 | 2585.80 | 0.00 | T1 1.5R @ 2624.98 |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 2602.45 | 2587.77 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:30:00 | 2654.75 | 2646.07 | 0.00 | ORB-long ORB[2630.85,2654.00] vol=2.2x ATR=7.34 |
| Stop hit — per-position SL triggered | 2024-06-13 10:35:00 | 2647.41 | 2646.70 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 2594.00 | 2571.17 | 0.00 | ORB-long ORB[2539.95,2577.85] vol=4.3x ATR=10.75 |
| Stop hit — per-position SL triggered | 2024-06-21 09:35:00 | 2583.25 | 2571.96 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:25:00 | 2644.90 | 2616.26 | 0.00 | ORB-long ORB[2593.90,2624.90] vol=1.6x ATR=7.85 |
| Stop hit — per-position SL triggered | 2024-06-27 10:30:00 | 2637.05 | 2619.68 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 11:05:00 | 2633.80 | 2656.35 | 0.00 | ORB-short ORB[2651.00,2674.35] vol=3.0x ATR=7.40 |
| Stop hit — per-position SL triggered | 2024-07-01 11:10:00 | 2641.20 | 2655.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:05:00 | 2620.00 | 2607.14 | 0.00 | ORB-long ORB[2582.05,2619.90] vol=1.7x ATR=8.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:10:00 | 2633.16 | 2612.16 | 0.00 | T1 1.5R @ 2633.16 |
| Stop hit — per-position SL triggered | 2024-07-04 11:50:00 | 2620.00 | 2620.71 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:05:00 | 2573.90 | 2585.01 | 0.00 | ORB-short ORB[2578.50,2606.95] vol=2.5x ATR=8.53 |
| Stop hit — per-position SL triggered | 2024-07-05 11:35:00 | 2582.43 | 2584.26 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:40:00 | 2512.00 | 2529.81 | 0.00 | ORB-short ORB[2527.75,2549.30] vol=1.6x ATR=7.87 |
| Stop hit — per-position SL triggered | 2024-07-11 10:55:00 | 2519.87 | 2528.52 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:30:00 | 2596.25 | 2578.69 | 0.00 | ORB-long ORB[2551.00,2587.00] vol=3.0x ATR=11.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:10:00 | 2613.08 | 2588.63 | 0.00 | T1 1.5R @ 2613.08 |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 2596.25 | 2590.10 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:45:00 | 2726.20 | 2706.35 | 0.00 | ORB-long ORB[2694.30,2720.70] vol=2.0x ATR=8.14 |
| Stop hit — per-position SL triggered | 2024-07-26 11:25:00 | 2718.06 | 2709.25 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:45:00 | 2767.70 | 2747.79 | 0.00 | ORB-long ORB[2697.05,2728.90] vol=3.6x ATR=11.37 |
| Stop hit — per-position SL triggered | 2024-07-31 09:50:00 | 2756.33 | 2749.32 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:15:00 | 2913.00 | 2882.49 | 0.00 | ORB-long ORB[2834.65,2876.45] vol=3.3x ATR=12.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:35:00 | 2931.96 | 2899.15 | 0.00 | T1 1.5R @ 2931.96 |
| Stop hit — per-position SL triggered | 2024-08-08 10:45:00 | 2913.00 | 2902.65 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 2893.90 | 2913.60 | 0.00 | ORB-short ORB[2900.20,2937.70] vol=2.1x ATR=9.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:45:00 | 2880.29 | 2907.25 | 0.00 | T1 1.5R @ 2880.29 |
| Stop hit — per-position SL triggered | 2024-08-20 10:30:00 | 2893.90 | 2889.66 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:50:00 | 2771.65 | 2785.94 | 0.00 | ORB-short ORB[2775.00,2805.00] vol=2.1x ATR=9.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 15:00:00 | 2757.16 | 2772.39 | 0.00 | T1 1.5R @ 2757.16 |
| Target hit | 2024-08-30 15:20:00 | 2742.00 | 2767.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:50:00 | 2876.25 | 2850.55 | 0.00 | ORB-long ORB[2828.05,2866.15] vol=1.8x ATR=8.81 |
| Stop hit — per-position SL triggered | 2024-09-05 09:55:00 | 2867.44 | 2853.31 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 2787.65 | 2796.09 | 0.00 | ORB-short ORB[2797.15,2838.30] vol=2.4x ATR=10.00 |
| Stop hit — per-position SL triggered | 2024-09-06 11:45:00 | 2797.65 | 2792.82 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 2805.20 | 2813.07 | 0.00 | ORB-short ORB[2810.30,2845.00] vol=3.2x ATR=10.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 2790.03 | 2803.30 | 0.00 | T1 1.5R @ 2790.03 |
| Stop hit — per-position SL triggered | 2024-09-17 14:40:00 | 2805.20 | 2781.60 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 2811.85 | 2801.68 | 0.00 | ORB-long ORB[2778.15,2794.25] vol=2.6x ATR=7.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:05:00 | 2823.48 | 2804.14 | 0.00 | T1 1.5R @ 2823.48 |
| Stop hit — per-position SL triggered | 2024-09-18 13:00:00 | 2811.85 | 2825.52 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 2774.55 | 2781.80 | 0.00 | ORB-short ORB[2777.70,2799.95] vol=1.6x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:50:00 | 2765.56 | 2774.54 | 0.00 | T1 1.5R @ 2765.56 |
| Target hit | 2024-09-19 15:20:00 | 2698.90 | 2717.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2024-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 09:30:00 | 2698.25 | 2708.11 | 0.00 | ORB-short ORB[2702.65,2735.95] vol=1.7x ATR=11.43 |
| Stop hit — per-position SL triggered | 2024-09-23 10:55:00 | 2709.68 | 2699.46 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:15:00 | 2700.30 | 2710.25 | 0.00 | ORB-short ORB[2702.15,2735.00] vol=1.9x ATR=8.01 |
| Stop hit — per-position SL triggered | 2024-09-25 10:50:00 | 2708.31 | 2708.33 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 2718.05 | 2728.98 | 0.00 | ORB-short ORB[2737.25,2763.00] vol=2.2x ATR=7.93 |
| Stop hit — per-position SL triggered | 2024-10-01 11:50:00 | 2725.98 | 2726.95 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 2738.95 | 2722.34 | 0.00 | ORB-long ORB[2695.00,2727.00] vol=1.7x ATR=10.89 |
| Stop hit — per-position SL triggered | 2024-10-03 09:45:00 | 2728.06 | 2723.01 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:40:00 | 2758.65 | 2735.47 | 0.00 | ORB-long ORB[2706.05,2746.90] vol=1.6x ATR=9.60 |
| Stop hit — per-position SL triggered | 2024-10-04 14:20:00 | 2749.05 | 2747.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 11:00:00 | 2570.70 | 2582.38 | 0.00 | ORB-short ORB[2587.55,2622.75] vol=1.8x ATR=10.97 |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 2581.67 | 2582.26 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:20:00 | 2612.55 | 2627.82 | 0.00 | ORB-short ORB[2621.95,2658.95] vol=5.1x ATR=12.04 |
| Stop hit — per-position SL triggered | 2024-11-05 10:35:00 | 2624.59 | 2623.29 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 09:35:00 | 2576.60 | 2588.46 | 0.00 | ORB-short ORB[2582.20,2620.00] vol=1.6x ATR=8.21 |
| Stop hit — per-position SL triggered | 2024-11-08 09:45:00 | 2584.81 | 2587.74 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 2483.00 | 2510.20 | 0.00 | ORB-short ORB[2515.60,2540.10] vol=1.7x ATR=11.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:20:00 | 2465.24 | 2502.33 | 0.00 | T1 1.5R @ 2465.24 |
| Stop hit — per-position SL triggered | 2024-11-13 10:30:00 | 2483.00 | 2501.08 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:15:00 | 2443.55 | 2431.92 | 0.00 | ORB-long ORB[2414.05,2439.30] vol=1.9x ATR=7.72 |
| Stop hit — per-position SL triggered | 2024-11-19 10:20:00 | 2435.83 | 2432.29 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:30:00 | 2404.85 | 2422.41 | 0.00 | ORB-short ORB[2420.10,2449.60] vol=1.9x ATR=11.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:35:00 | 2387.22 | 2416.50 | 0.00 | T1 1.5R @ 2387.22 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 2404.85 | 2414.23 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 10:55:00 | 2424.40 | 2441.79 | 0.00 | ORB-short ORB[2453.75,2486.85] vol=1.5x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:05:00 | 2412.36 | 2437.72 | 0.00 | T1 1.5R @ 2412.36 |
| Target hit | 2024-11-25 15:20:00 | 2336.00 | 2377.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:05:00 | 2322.00 | 2336.21 | 0.00 | ORB-short ORB[2342.40,2366.15] vol=2.8x ATR=7.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:50:00 | 2310.45 | 2331.63 | 0.00 | T1 1.5R @ 2310.45 |
| Stop hit — per-position SL triggered | 2024-11-26 12:10:00 | 2322.00 | 2325.55 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:45:00 | 2385.00 | 2398.05 | 0.00 | ORB-short ORB[2390.05,2413.40] vol=1.9x ATR=10.30 |
| Stop hit — per-position SL triggered | 2024-11-27 09:50:00 | 2395.30 | 2397.31 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:55:00 | 2417.45 | 2428.02 | 0.00 | ORB-short ORB[2423.00,2452.10] vol=5.2x ATR=7.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:25:00 | 2405.87 | 2423.53 | 0.00 | T1 1.5R @ 2405.87 |
| Target hit | 2024-12-02 14:30:00 | 2412.85 | 2410.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:15:00 | 2456.50 | 2432.00 | 0.00 | ORB-long ORB[2401.55,2419.00] vol=2.0x ATR=8.82 |
| Stop hit — per-position SL triggered | 2024-12-03 10:25:00 | 2447.68 | 2433.84 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 2367.85 | 2381.32 | 0.00 | ORB-short ORB[2385.00,2407.80] vol=2.9x ATR=5.45 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 2373.30 | 2377.77 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:35:00 | 2356.65 | 2369.24 | 0.00 | ORB-short ORB[2365.00,2399.00] vol=1.6x ATR=6.53 |
| Stop hit — per-position SL triggered | 2024-12-06 10:40:00 | 2363.18 | 2369.11 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:15:00 | 2329.50 | 2338.92 | 0.00 | ORB-short ORB[2340.00,2363.75] vol=4.2x ATR=5.43 |
| Stop hit — per-position SL triggered | 2024-12-09 12:10:00 | 2334.93 | 2336.65 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 2316.05 | 2324.14 | 0.00 | ORB-short ORB[2323.05,2356.30] vol=1.9x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:55:00 | 2308.38 | 2321.32 | 0.00 | T1 1.5R @ 2308.38 |
| Target hit | 2024-12-12 15:20:00 | 2304.50 | 2309.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 2245.00 | 2257.37 | 0.00 | ORB-short ORB[2255.55,2277.75] vol=2.5x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-12-16 09:45:00 | 2251.84 | 2255.92 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:05:00 | 2309.70 | 2290.03 | 0.00 | ORB-long ORB[2270.10,2285.00] vol=2.1x ATR=9.48 |
| Stop hit — per-position SL triggered | 2024-12-18 10:10:00 | 2300.22 | 2291.60 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 10:45:00 | 2264.85 | 2271.62 | 0.00 | ORB-short ORB[2266.00,2287.95] vol=1.6x ATR=5.85 |
| Stop hit — per-position SL triggered | 2024-12-24 11:35:00 | 2270.70 | 2267.99 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 2259.20 | 2256.51 | 0.00 | ORB-long ORB[2241.00,2256.80] vol=5.9x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 2255.27 | 2255.48 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 2234.70 | 2240.53 | 0.00 | ORB-short ORB[2237.45,2258.10] vol=3.6x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:30:00 | 2227.21 | 2238.85 | 0.00 | T1 1.5R @ 2227.21 |
| Stop hit — per-position SL triggered | 2025-01-02 11:40:00 | 2234.70 | 2238.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:05:00 | 2218.70 | 2228.15 | 0.00 | ORB-short ORB[2232.05,2248.95] vol=1.5x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:15:00 | 2208.86 | 2226.79 | 0.00 | T1 1.5R @ 2208.86 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 2218.70 | 2225.88 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:10:00 | 2202.10 | 2224.87 | 0.00 | ORB-short ORB[2222.80,2241.55] vol=1.7x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:40:00 | 2193.14 | 2218.59 | 0.00 | T1 1.5R @ 2193.14 |
| Stop hit — per-position SL triggered | 2025-01-08 13:15:00 | 2202.10 | 2210.29 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 10:50:00 | 2074.10 | 2082.91 | 0.00 | ORB-short ORB[2086.00,2116.40] vol=1.6x ATR=7.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:15:00 | 2062.37 | 2079.81 | 0.00 | T1 1.5R @ 2062.37 |
| Target hit | 2025-01-15 15:20:00 | 2044.20 | 2063.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 09:45:00 | 2039.25 | 2052.44 | 0.00 | ORB-short ORB[2051.00,2069.20] vol=2.0x ATR=9.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:05:00 | 2024.49 | 2044.83 | 0.00 | T1 1.5R @ 2024.49 |
| Stop hit — per-position SL triggered | 2025-01-16 14:40:00 | 2039.25 | 2035.21 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:55:00 | 2052.90 | 2066.99 | 0.00 | ORB-short ORB[2065.00,2094.00] vol=7.3x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-01-20 11:00:00 | 2059.88 | 2064.85 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:10:00 | 2077.85 | 2087.83 | 0.00 | ORB-short ORB[2082.20,2113.30] vol=2.5x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:15:00 | 2067.95 | 2087.30 | 0.00 | T1 1.5R @ 2067.95 |
| Target hit | 2025-01-24 14:10:00 | 2059.45 | 2058.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — SELL (started 2025-01-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 11:05:00 | 2036.60 | 2044.67 | 0.00 | ORB-short ORB[2063.80,2094.15] vol=11.7x ATR=6.51 |
| Stop hit — per-position SL triggered | 2025-01-27 11:15:00 | 2043.11 | 2044.52 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1983.85 | 1988.86 | 0.00 | ORB-short ORB[1984.45,2009.95] vol=3.9x ATR=6.34 |
| Stop hit — per-position SL triggered | 2025-01-30 10:20:00 | 1990.19 | 1988.95 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:50:00 | 2026.45 | 2041.30 | 0.00 | ORB-short ORB[2030.00,2059.45] vol=5.0x ATR=10.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 13:15:00 | 2011.35 | 2031.76 | 0.00 | T1 1.5R @ 2011.35 |
| Target hit | 2025-02-13 15:20:00 | 1995.20 | 2015.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:50:00 | 2696.75 | 2676.37 | 0.00 | ORB-long ORB[2658.60,2694.15] vol=1.9x ATR=13.94 |
| Stop hit — per-position SL triggered | 2025-03-12 10:05:00 | 2682.81 | 2679.43 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 11:05:00 | 2644.70 | 2666.53 | 0.00 | ORB-short ORB[2673.10,2695.10] vol=2.6x ATR=7.70 |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 2652.40 | 2665.83 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:10:00 | 2730.40 | 2713.95 | 0.00 | ORB-long ORB[2677.20,2717.55] vol=1.7x ATR=11.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:50:00 | 2748.23 | 2727.08 | 0.00 | T1 1.5R @ 2748.23 |
| Target hit | 2025-03-18 15:20:00 | 2798.70 | 2758.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-03-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:20:00 | 2879.20 | 2856.93 | 0.00 | ORB-long ORB[2832.05,2866.95] vol=2.0x ATR=13.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:45:00 | 2898.87 | 2861.94 | 0.00 | T1 1.5R @ 2898.87 |
| Stop hit — per-position SL triggered | 2025-03-20 11:15:00 | 2879.20 | 2865.97 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 2778.00 | 2800.67 | 0.00 | ORB-short ORB[2793.00,2834.00] vol=2.5x ATR=9.62 |
| Stop hit — per-position SL triggered | 2025-03-27 10:50:00 | 2787.62 | 2799.59 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:05:00 | 2910.20 | 2896.18 | 0.00 | ORB-long ORB[2871.20,2903.00] vol=3.1x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:25:00 | 2922.39 | 2900.27 | 0.00 | T1 1.5R @ 2922.39 |
| Stop hit — per-position SL triggered | 2025-04-16 11:35:00 | 2910.20 | 2901.38 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:50:00 | 2865.90 | 2838.39 | 0.00 | ORB-long ORB[2821.00,2861.90] vol=1.6x ATR=11.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:15:00 | 2882.62 | 2846.19 | 0.00 | T1 1.5R @ 2882.62 |
| Target hit | 2025-04-29 15:15:00 | 2885.90 | 2889.39 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:40:00 | 1967.95 | 2024-05-13 10:50:00 | 1979.56 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-05-14 09:35:00 | 2000.00 | 2024-05-14 10:05:00 | 1990.14 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-14 09:35:00 | 2000.00 | 2024-05-14 11:35:00 | 2000.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-15 09:45:00 | 1980.00 | 2024-05-15 11:40:00 | 1986.12 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-17 09:30:00 | 2051.05 | 2024-05-17 09:40:00 | 2042.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-22 10:55:00 | 2300.00 | 2024-05-22 11:10:00 | 2311.94 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-05-31 10:05:00 | 2602.45 | 2024-05-31 10:10:00 | 2624.98 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-05-31 10:05:00 | 2602.45 | 2024-05-31 10:15:00 | 2602.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 10:30:00 | 2654.75 | 2024-06-13 10:35:00 | 2647.41 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-21 09:30:00 | 2594.00 | 2024-06-21 09:35:00 | 2583.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-06-27 10:25:00 | 2644.90 | 2024-06-27 10:30:00 | 2637.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-01 11:05:00 | 2633.80 | 2024-07-01 11:10:00 | 2641.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-04 11:05:00 | 2620.00 | 2024-07-04 11:10:00 | 2633.16 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-04 11:05:00 | 2620.00 | 2024-07-04 11:50:00 | 2620.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 11:05:00 | 2573.90 | 2024-07-05 11:35:00 | 2582.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-11 10:40:00 | 2512.00 | 2024-07-11 10:55:00 | 2519.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-22 10:30:00 | 2596.25 | 2024-07-22 11:10:00 | 2613.08 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-07-22 10:30:00 | 2596.25 | 2024-07-22 11:15:00 | 2596.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:45:00 | 2726.20 | 2024-07-26 11:25:00 | 2718.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-31 09:45:00 | 2767.70 | 2024-07-31 09:50:00 | 2756.33 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-08-08 10:15:00 | 2913.00 | 2024-08-08 10:35:00 | 2931.96 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-08-08 10:15:00 | 2913.00 | 2024-08-08 10:45:00 | 2913.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 09:30:00 | 2893.90 | 2024-08-20 09:45:00 | 2880.29 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-08-20 09:30:00 | 2893.90 | 2024-08-20 10:30:00 | 2893.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 10:50:00 | 2771.65 | 2024-08-30 15:00:00 | 2757.16 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-30 10:50:00 | 2771.65 | 2024-08-30 15:20:00 | 2742.00 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2024-09-05 09:50:00 | 2876.25 | 2024-09-05 09:55:00 | 2867.44 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-06 10:30:00 | 2787.65 | 2024-09-06 11:45:00 | 2797.65 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-17 09:30:00 | 2805.20 | 2024-09-17 09:55:00 | 2790.03 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-17 09:30:00 | 2805.20 | 2024-09-17 14:40:00 | 2805.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 10:50:00 | 2811.85 | 2024-09-18 11:05:00 | 2823.48 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-18 10:50:00 | 2811.85 | 2024-09-18 13:00:00 | 2811.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:30:00 | 2774.55 | 2024-09-19 09:50:00 | 2765.56 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-09-19 09:30:00 | 2774.55 | 2024-09-19 15:20:00 | 2698.90 | TARGET_HIT | 0.50 | 2.73% |
| SELL | retest1 | 2024-09-23 09:30:00 | 2698.25 | 2024-09-23 10:55:00 | 2709.68 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-09-25 10:15:00 | 2700.30 | 2024-09-25 10:50:00 | 2708.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-01 11:10:00 | 2718.05 | 2024-10-01 11:50:00 | 2725.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-03 09:35:00 | 2738.95 | 2024-10-03 09:45:00 | 2728.06 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-04 10:40:00 | 2758.65 | 2024-10-04 14:20:00 | 2749.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-28 11:00:00 | 2570.70 | 2024-10-28 11:15:00 | 2581.67 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-05 10:20:00 | 2612.55 | 2024-11-05 10:35:00 | 2624.59 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-11-08 09:35:00 | 2576.60 | 2024-11-08 09:45:00 | 2584.81 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-13 09:45:00 | 2483.00 | 2024-11-13 10:20:00 | 2465.24 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-11-13 09:45:00 | 2483.00 | 2024-11-13 10:30:00 | 2483.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:15:00 | 2443.55 | 2024-11-19 10:20:00 | 2435.83 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-21 09:30:00 | 2404.85 | 2024-11-21 09:35:00 | 2387.22 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-11-21 09:30:00 | 2404.85 | 2024-11-21 09:40:00 | 2404.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-25 10:55:00 | 2424.40 | 2024-11-25 11:05:00 | 2412.36 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-25 10:55:00 | 2424.40 | 2024-11-25 15:20:00 | 2336.00 | TARGET_HIT | 0.50 | 3.65% |
| SELL | retest1 | 2024-11-26 11:05:00 | 2322.00 | 2024-11-26 11:50:00 | 2310.45 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-26 11:05:00 | 2322.00 | 2024-11-26 12:10:00 | 2322.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 09:45:00 | 2385.00 | 2024-11-27 09:50:00 | 2395.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-02 10:55:00 | 2417.45 | 2024-12-02 11:25:00 | 2405.87 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-02 10:55:00 | 2417.45 | 2024-12-02 14:30:00 | 2412.85 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2024-12-03 10:15:00 | 2456.50 | 2024-12-03 10:25:00 | 2447.68 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2367.85 | 2024-12-05 12:05:00 | 2373.30 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-06 10:35:00 | 2356.65 | 2024-12-06 10:40:00 | 2363.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-09 11:15:00 | 2329.50 | 2024-12-09 12:10:00 | 2334.93 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-12 11:00:00 | 2316.05 | 2024-12-12 11:55:00 | 2308.38 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-12 11:00:00 | 2316.05 | 2024-12-12 15:20:00 | 2304.50 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-16 09:35:00 | 2245.00 | 2024-12-16 09:45:00 | 2251.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-18 10:05:00 | 2309.70 | 2024-12-18 10:10:00 | 2300.22 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-24 10:45:00 | 2264.85 | 2024-12-24 11:35:00 | 2270.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-27 11:05:00 | 2259.20 | 2024-12-27 11:10:00 | 2255.27 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-01-02 10:45:00 | 2234.70 | 2025-01-02 11:30:00 | 2227.21 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-02 10:45:00 | 2234.70 | 2025-01-02 11:40:00 | 2234.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:05:00 | 2218.70 | 2025-01-06 11:15:00 | 2208.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-06 11:05:00 | 2218.70 | 2025-01-06 11:30:00 | 2218.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 11:10:00 | 2202.10 | 2025-01-08 11:40:00 | 2193.14 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-01-08 11:10:00 | 2202.10 | 2025-01-08 13:15:00 | 2202.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-15 10:50:00 | 2074.10 | 2025-01-15 11:15:00 | 2062.37 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-15 10:50:00 | 2074.10 | 2025-01-15 15:20:00 | 2044.20 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2025-01-16 09:45:00 | 2039.25 | 2025-01-16 11:05:00 | 2024.49 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-01-16 09:45:00 | 2039.25 | 2025-01-16 14:40:00 | 2039.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 10:55:00 | 2052.90 | 2025-01-20 11:00:00 | 2059.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-24 11:10:00 | 2077.85 | 2025-01-24 11:15:00 | 2067.95 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-24 11:10:00 | 2077.85 | 2025-01-24 14:10:00 | 2059.45 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2025-01-27 11:05:00 | 2036.60 | 2025-01-27 11:15:00 | 2043.11 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-30 10:15:00 | 1983.85 | 2025-01-30 10:20:00 | 1990.19 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-13 10:50:00 | 2026.45 | 2025-02-13 13:15:00 | 2011.35 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2025-02-13 10:50:00 | 2026.45 | 2025-02-13 15:20:00 | 1995.20 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2025-03-12 09:50:00 | 2696.75 | 2025-03-12 10:05:00 | 2682.81 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-03-13 11:05:00 | 2644.70 | 2025-03-13 11:15:00 | 2652.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-18 10:10:00 | 2730.40 | 2025-03-18 11:50:00 | 2748.23 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-03-18 10:10:00 | 2730.40 | 2025-03-18 15:20:00 | 2798.70 | TARGET_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2025-03-20 10:20:00 | 2879.20 | 2025-03-20 10:45:00 | 2898.87 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-03-20 10:20:00 | 2879.20 | 2025-03-20 11:15:00 | 2879.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-27 10:45:00 | 2778.00 | 2025-03-27 10:50:00 | 2787.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-16 11:05:00 | 2910.20 | 2025-04-16 11:25:00 | 2922.39 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-04-16 11:05:00 | 2910.20 | 2025-04-16 11:35:00 | 2910.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-29 10:50:00 | 2865.90 | 2025-04-29 11:15:00 | 2882.62 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-04-29 10:50:00 | 2865.90 | 2025-04-29 15:15:00 | 2885.90 | TARGET_HIT | 0.50 | 0.70% |
