# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
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
| ENTRY1 | 111 |
| ENTRY2 | 0 |
| PARTIAL | 45 |
| TARGET_HIT | 22 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 156 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 89
- **Target hits / Stop hits / Partials:** 22 / 89 / 45
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 16.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 31 | 35.6% | 9 | 56 | 22 | 0.02% | 1.7% |
| BUY @ 2nd Alert (retest1) | 87 | 31 | 35.6% | 9 | 56 | 22 | 0.02% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 36 | 52.2% | 13 | 33 | 23 | 0.22% | 15.2% |
| SELL @ 2nd Alert (retest1) | 69 | 36 | 52.2% | 13 | 33 | 23 | 0.22% | 15.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 156 | 67 | 42.9% | 22 | 89 | 45 | 0.11% | 16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:50:00 | 2732.80 | 2716.77 | 0.00 | ORB-long ORB[2671.10,2709.00] vol=1.9x ATR=14.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 11:15:00 | 2755.12 | 2724.98 | 0.00 | T1 1.5R @ 2755.12 |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 2732.80 | 2731.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 11:10:00 | 2710.40 | 2716.05 | 0.00 | ORB-short ORB[2713.00,2748.00] vol=5.0x ATR=6.28 |
| Stop hit — per-position SL triggered | 2025-05-13 11:15:00 | 2716.68 | 2716.02 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:35:00 | 2772.70 | 2737.40 | 0.00 | ORB-long ORB[2720.60,2741.20] vol=3.0x ATR=9.07 |
| Stop hit — per-position SL triggered | 2025-05-15 11:10:00 | 2763.63 | 2748.82 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 11:00:00 | 2778.50 | 2786.28 | 0.00 | ORB-short ORB[2780.50,2820.10] vol=1.7x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 13:10:00 | 2769.75 | 2782.12 | 0.00 | T1 1.5R @ 2769.75 |
| Target hit | 2025-05-19 15:20:00 | 2723.40 | 2752.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:30:00 | 2665.10 | 2677.68 | 0.00 | ORB-short ORB[2675.00,2702.30] vol=1.7x ATR=7.41 |
| Stop hit — per-position SL triggered | 2025-05-22 09:35:00 | 2672.51 | 2676.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:40:00 | 2593.60 | 2603.32 | 0.00 | ORB-short ORB[2596.90,2620.80] vol=1.8x ATR=5.37 |
| Stop hit — per-position SL triggered | 2025-05-28 11:20:00 | 2598.97 | 2601.04 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 2564.00 | 2581.82 | 0.00 | ORB-short ORB[2572.70,2589.90] vol=1.9x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-05-29 11:05:00 | 2569.35 | 2581.38 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:00:00 | 2573.90 | 2583.82 | 0.00 | ORB-short ORB[2574.70,2604.90] vol=1.5x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 11:40:00 | 2565.07 | 2582.44 | 0.00 | T1 1.5R @ 2565.07 |
| Target hit | 2025-05-30 15:20:00 | 2539.90 | 2554.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:55:00 | 2516.30 | 2522.33 | 0.00 | ORB-short ORB[2520.00,2549.60] vol=2.0x ATR=5.82 |
| Stop hit — per-position SL triggered | 2025-06-02 11:45:00 | 2522.12 | 2521.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:55:00 | 2556.20 | 2536.25 | 0.00 | ORB-long ORB[2521.10,2544.30] vol=2.6x ATR=6.84 |
| Stop hit — per-position SL triggered | 2025-06-03 11:10:00 | 2549.36 | 2539.24 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 2574.00 | 2558.21 | 0.00 | ORB-long ORB[2538.90,2559.90] vol=1.7x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:45:00 | 2582.91 | 2571.91 | 0.00 | T1 1.5R @ 2582.91 |
| Target hit | 2025-06-05 10:30:00 | 2579.40 | 2579.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2025-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:05:00 | 2570.10 | 2553.86 | 0.00 | ORB-long ORB[2545.10,2565.90] vol=1.8x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-06-06 12:35:00 | 2564.32 | 2560.06 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:45:00 | 2593.00 | 2583.82 | 0.00 | ORB-long ORB[2574.70,2587.90] vol=2.0x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:55:00 | 2601.42 | 2589.39 | 0.00 | T1 1.5R @ 2601.42 |
| Target hit | 2025-06-09 10:55:00 | 2600.00 | 2600.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2025-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:35:00 | 2702.70 | 2690.92 | 0.00 | ORB-long ORB[2670.00,2692.50] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-06-19 09:45:00 | 2697.00 | 2694.32 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:45:00 | 2718.80 | 2704.81 | 0.00 | ORB-long ORB[2683.00,2716.00] vol=1.6x ATR=7.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:00:00 | 2729.61 | 2708.72 | 0.00 | T1 1.5R @ 2729.61 |
| Stop hit — per-position SL triggered | 2025-06-20 12:05:00 | 2718.80 | 2713.91 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:45:00 | 2766.40 | 2753.89 | 0.00 | ORB-long ORB[2737.30,2757.00] vol=2.3x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:00:00 | 2776.98 | 2757.46 | 0.00 | T1 1.5R @ 2776.98 |
| Target hit | 2025-06-24 13:45:00 | 2775.70 | 2777.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2025-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:55:00 | 2848.00 | 2848.71 | 0.00 | ORB-short ORB[2851.90,2870.40] vol=4.9x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:00:00 | 2840.87 | 2848.04 | 0.00 | T1 1.5R @ 2840.87 |
| Stop hit — per-position SL triggered | 2025-07-02 11:25:00 | 2848.00 | 2847.82 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 09:40:00 | 2800.00 | 2804.14 | 0.00 | ORB-short ORB[2803.60,2820.00] vol=2.3x ATR=6.10 |
| Stop hit — per-position SL triggered | 2025-07-04 10:00:00 | 2806.10 | 2801.60 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:45:00 | 2793.50 | 2806.81 | 0.00 | ORB-short ORB[2802.90,2817.50] vol=2.3x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:50:00 | 2787.16 | 2801.69 | 0.00 | T1 1.5R @ 2787.16 |
| Target hit | 2025-07-07 15:20:00 | 2779.60 | 2789.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:55:00 | 2803.00 | 2795.81 | 0.00 | ORB-long ORB[2778.20,2790.50] vol=1.7x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:05:00 | 2809.54 | 2800.50 | 0.00 | T1 1.5R @ 2809.54 |
| Target hit | 2025-07-08 11:35:00 | 2807.20 | 2809.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 2795.00 | 2803.62 | 0.00 | ORB-short ORB[2797.90,2809.80] vol=3.3x ATR=3.95 |
| Stop hit — per-position SL triggered | 2025-07-10 11:20:00 | 2798.95 | 2803.04 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:45:00 | 2806.00 | 2785.26 | 0.00 | ORB-long ORB[2771.00,2784.20] vol=2.0x ATR=7.17 |
| Stop hit — per-position SL triggered | 2025-07-11 09:55:00 | 2798.83 | 2787.97 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 2794.80 | 2778.63 | 0.00 | ORB-long ORB[2766.50,2780.90] vol=1.9x ATR=6.02 |
| Stop hit — per-position SL triggered | 2025-07-14 09:50:00 | 2788.78 | 2787.12 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:10:00 | 2748.50 | 2751.88 | 0.00 | ORB-short ORB[2748.80,2773.20] vol=2.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-07-17 11:35:00 | 2752.44 | 2751.41 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:55:00 | 2748.00 | 2751.57 | 0.00 | ORB-short ORB[2751.00,2774.00] vol=4.1x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 2739.05 | 2749.32 | 0.00 | T1 1.5R @ 2739.05 |
| Target hit | 2025-07-18 13:40:00 | 2732.50 | 2730.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2025-07-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:40:00 | 2731.70 | 2737.74 | 0.00 | ORB-short ORB[2738.10,2759.00] vol=4.8x ATR=6.32 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 2738.02 | 2733.90 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 2690.00 | 2696.74 | 0.00 | ORB-short ORB[2701.50,2723.90] vol=9.4x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:25:00 | 2683.29 | 2694.63 | 0.00 | T1 1.5R @ 2683.29 |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 2690.00 | 2688.24 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:05:00 | 2710.90 | 2717.07 | 0.00 | ORB-short ORB[2714.90,2735.00] vol=1.8x ATR=6.59 |
| Stop hit — per-position SL triggered | 2025-07-25 10:10:00 | 2717.49 | 2717.21 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-07-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:30:00 | 2748.00 | 2736.52 | 0.00 | ORB-long ORB[2697.00,2730.30] vol=1.6x ATR=8.36 |
| Stop hit — per-position SL triggered | 2025-07-28 10:40:00 | 2739.64 | 2737.35 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:55:00 | 2720.00 | 2723.69 | 0.00 | ORB-short ORB[2727.50,2739.20] vol=4.1x ATR=6.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 11:25:00 | 2710.67 | 2722.16 | 0.00 | T1 1.5R @ 2710.67 |
| Stop hit — per-position SL triggered | 2025-07-29 13:25:00 | 2720.00 | 2718.64 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 2765.50 | 2755.14 | 0.00 | ORB-long ORB[2733.10,2750.60] vol=1.5x ATR=7.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:05:00 | 2776.34 | 2763.08 | 0.00 | T1 1.5R @ 2776.34 |
| Target hit | 2025-07-30 13:35:00 | 2778.60 | 2778.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2025-08-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:50:00 | 2809.20 | 2804.08 | 0.00 | ORB-long ORB[2788.00,2804.90] vol=4.9x ATR=7.60 |
| Stop hit — per-position SL triggered | 2025-08-05 11:55:00 | 2801.60 | 2805.19 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 2767.60 | 2782.96 | 0.00 | ORB-short ORB[2788.00,2800.00] vol=2.1x ATR=5.63 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 2773.23 | 2779.13 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:40:00 | 2747.90 | 2760.40 | 0.00 | ORB-short ORB[2749.10,2767.90] vol=1.5x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:00:00 | 2738.83 | 2757.87 | 0.00 | T1 1.5R @ 2738.83 |
| Target hit | 2025-08-07 15:00:00 | 2742.10 | 2740.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2025-08-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:30:00 | 2708.00 | 2727.81 | 0.00 | ORB-short ORB[2735.00,2761.90] vol=1.7x ATR=7.60 |
| Stop hit — per-position SL triggered | 2025-08-12 11:00:00 | 2715.60 | 2724.39 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-08-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:40:00 | 2743.20 | 2753.51 | 0.00 | ORB-short ORB[2745.00,2766.00] vol=1.8x ATR=5.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:05:00 | 2734.33 | 2750.58 | 0.00 | T1 1.5R @ 2734.33 |
| Stop hit — per-position SL triggered | 2025-08-13 13:00:00 | 2743.20 | 2743.78 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-08-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:50:00 | 2827.00 | 2833.81 | 0.00 | ORB-short ORB[2830.50,2854.00] vol=2.3x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:55:00 | 2819.41 | 2831.03 | 0.00 | T1 1.5R @ 2819.41 |
| Target hit | 2025-08-19 14:30:00 | 2819.80 | 2818.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2025-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:35:00 | 2828.90 | 2820.18 | 0.00 | ORB-long ORB[2807.60,2828.30] vol=2.4x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 11:05:00 | 2836.87 | 2823.61 | 0.00 | T1 1.5R @ 2836.87 |
| Target hit | 2025-08-20 15:20:00 | 2864.80 | 2850.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:15:00 | 2887.90 | 2874.71 | 0.00 | ORB-long ORB[2855.00,2882.00] vol=3.0x ATR=7.47 |
| Stop hit — per-position SL triggered | 2025-08-21 12:20:00 | 2880.43 | 2881.94 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:00:00 | 2805.80 | 2796.22 | 0.00 | ORB-long ORB[2772.10,2789.00] vol=1.8x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 2801.07 | 2797.25 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:30:00 | 2815.20 | 2810.79 | 0.00 | ORB-long ORB[2788.00,2814.50] vol=3.3x ATR=5.13 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 2810.07 | 2811.14 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:05:00 | 2813.10 | 2801.22 | 0.00 | ORB-long ORB[2792.10,2804.90] vol=4.7x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-09-12 12:20:00 | 2809.23 | 2808.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 2899.00 | 2886.15 | 0.00 | ORB-long ORB[2860.00,2889.30] vol=4.9x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:05:00 | 2906.59 | 2889.44 | 0.00 | T1 1.5R @ 2906.59 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 2899.00 | 2893.59 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:05:00 | 2820.80 | 2838.99 | 0.00 | ORB-short ORB[2845.60,2873.80] vol=1.7x ATR=6.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:35:00 | 2811.18 | 2829.12 | 0.00 | T1 1.5R @ 2811.18 |
| Stop hit — per-position SL triggered | 2025-09-23 12:30:00 | 2820.80 | 2826.45 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 11:00:00 | 2829.30 | 2816.58 | 0.00 | ORB-long ORB[2800.90,2824.70] vol=3.1x ATR=5.95 |
| Stop hit — per-position SL triggered | 2025-09-24 11:05:00 | 2823.35 | 2817.05 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:50:00 | 2831.00 | 2813.70 | 0.00 | ORB-long ORB[2796.80,2814.20] vol=1.5x ATR=8.21 |
| Stop hit — per-position SL triggered | 2025-09-25 10:05:00 | 2822.79 | 2817.32 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 2740.50 | 2749.16 | 0.00 | ORB-short ORB[2752.60,2767.90] vol=1.6x ATR=5.65 |
| Stop hit — per-position SL triggered | 2025-10-01 11:40:00 | 2746.15 | 2744.92 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 2819.00 | 2812.27 | 0.00 | ORB-long ORB[2788.30,2817.70] vol=1.9x ATR=6.78 |
| Stop hit — per-position SL triggered | 2025-10-10 09:40:00 | 2812.22 | 2813.17 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:55:00 | 2783.40 | 2791.46 | 0.00 | ORB-short ORB[2798.00,2808.00] vol=1.6x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:05:00 | 2777.42 | 2789.18 | 0.00 | T1 1.5R @ 2777.42 |
| Target hit | 2025-10-14 14:25:00 | 2776.00 | 2775.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2025-10-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:20:00 | 2799.30 | 2793.17 | 0.00 | ORB-long ORB[2774.20,2794.30] vol=2.2x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 2794.75 | 2796.24 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:25:00 | 2844.20 | 2835.92 | 0.00 | ORB-long ORB[2817.10,2832.20] vol=1.5x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-10-16 10:40:00 | 2838.92 | 2837.03 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 2881.60 | 2861.61 | 0.00 | ORB-long ORB[2851.10,2864.40] vol=2.7x ATR=6.30 |
| Stop hit — per-position SL triggered | 2025-10-17 11:40:00 | 2875.30 | 2865.03 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:55:00 | 2872.30 | 2857.92 | 0.00 | ORB-long ORB[2842.70,2863.00] vol=1.5x ATR=6.90 |
| Stop hit — per-position SL triggered | 2025-10-20 11:20:00 | 2865.40 | 2859.06 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 2894.00 | 2885.27 | 0.00 | ORB-long ORB[2870.30,2889.50] vol=2.2x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:50:00 | 2903.68 | 2895.19 | 0.00 | T1 1.5R @ 2903.68 |
| Target hit | 2025-10-23 11:05:00 | 2895.70 | 2898.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — SELL (started 2025-10-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:20:00 | 2849.70 | 2860.08 | 0.00 | ORB-short ORB[2852.70,2877.00] vol=2.0x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:05:00 | 2838.90 | 2854.57 | 0.00 | T1 1.5R @ 2838.90 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 2849.70 | 2848.52 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:05:00 | 2875.90 | 2870.03 | 0.00 | ORB-long ORB[2847.60,2863.40] vol=1.8x ATR=6.90 |
| Stop hit — per-position SL triggered | 2025-10-27 10:10:00 | 2869.00 | 2869.87 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:55:00 | 2918.60 | 2936.96 | 0.00 | ORB-short ORB[2937.40,2950.80] vol=2.8x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:20:00 | 2909.20 | 2926.13 | 0.00 | T1 1.5R @ 2909.20 |
| Stop hit — per-position SL triggered | 2025-10-31 11:25:00 | 2918.60 | 2925.97 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 2910.70 | 2904.14 | 0.00 | ORB-long ORB[2883.10,2904.20] vol=2.1x ATR=7.57 |
| Stop hit — per-position SL triggered | 2025-11-03 11:00:00 | 2903.13 | 2904.21 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:50:00 | 2756.00 | 2774.97 | 0.00 | ORB-short ORB[2759.70,2800.00] vol=2.0x ATR=13.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:05:00 | 2735.64 | 2769.46 | 0.00 | T1 1.5R @ 2735.64 |
| Target hit | 2025-11-06 15:20:00 | 2699.40 | 2731.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 2764.00 | 2748.92 | 0.00 | ORB-long ORB[2729.30,2747.70] vol=2.1x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:10:00 | 2771.38 | 2752.13 | 0.00 | T1 1.5R @ 2771.38 |
| Stop hit — per-position SL triggered | 2025-11-10 15:05:00 | 2764.00 | 2764.36 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:55:00 | 2749.10 | 2756.44 | 0.00 | ORB-short ORB[2752.80,2780.10] vol=4.1x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 2754.15 | 2755.75 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-11-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:35:00 | 2780.30 | 2771.62 | 0.00 | ORB-long ORB[2761.50,2777.20] vol=2.3x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 12:10:00 | 2789.46 | 2777.34 | 0.00 | T1 1.5R @ 2789.46 |
| Stop hit — per-position SL triggered | 2025-11-14 12:45:00 | 2780.30 | 2777.98 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-11-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 11:00:00 | 2761.00 | 2771.36 | 0.00 | ORB-short ORB[2766.90,2790.30] vol=2.3x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 2764.66 | 2770.40 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:50:00 | 2737.30 | 2726.72 | 0.00 | ORB-long ORB[2710.00,2728.70] vol=1.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2025-12-04 11:35:00 | 2733.07 | 2731.81 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 2763.30 | 2753.46 | 0.00 | ORB-long ORB[2737.00,2763.00] vol=2.5x ATR=6.15 |
| Stop hit — per-position SL triggered | 2025-12-08 11:40:00 | 2757.15 | 2755.05 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:35:00 | 2758.20 | 2746.45 | 0.00 | ORB-long ORB[2726.50,2745.80] vol=1.5x ATR=7.14 |
| Stop hit — per-position SL triggered | 2025-12-09 11:40:00 | 2751.06 | 2751.74 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 2759.70 | 2767.60 | 0.00 | ORB-short ORB[2760.60,2779.80] vol=2.4x ATR=6.08 |
| Stop hit — per-position SL triggered | 2025-12-10 11:10:00 | 2765.78 | 2767.04 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:05:00 | 2774.10 | 2762.51 | 0.00 | ORB-long ORB[2745.00,2757.00] vol=1.8x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:30:00 | 2782.02 | 2765.00 | 0.00 | T1 1.5R @ 2782.02 |
| Stop hit — per-position SL triggered | 2025-12-11 12:20:00 | 2774.10 | 2766.94 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:40:00 | 2837.90 | 2832.93 | 0.00 | ORB-long ORB[2811.00,2829.00] vol=2.1x ATR=5.51 |
| Stop hit — per-position SL triggered | 2025-12-23 11:40:00 | 2832.39 | 2833.67 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 2843.90 | 2836.93 | 0.00 | ORB-long ORB[2824.30,2839.90] vol=2.8x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:10:00 | 2850.38 | 2838.85 | 0.00 | T1 1.5R @ 2850.38 |
| Stop hit — per-position SL triggered | 2025-12-24 12:05:00 | 2843.90 | 2846.12 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:20:00 | 2837.10 | 2828.85 | 0.00 | ORB-long ORB[2820.30,2835.00] vol=1.7x ATR=5.32 |
| Stop hit — per-position SL triggered | 2025-12-26 10:30:00 | 2831.78 | 2829.68 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 2829.80 | 2821.54 | 0.00 | ORB-long ORB[2812.50,2829.40] vol=3.1x ATR=5.95 |
| Stop hit — per-position SL triggered | 2025-12-29 09:35:00 | 2823.85 | 2822.03 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:50:00 | 2875.50 | 2871.31 | 0.00 | ORB-long ORB[2850.60,2874.00] vol=2.2x ATR=5.51 |
| Stop hit — per-position SL triggered | 2026-01-02 10:55:00 | 2869.99 | 2871.32 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:50:00 | 2891.00 | 2883.20 | 0.00 | ORB-long ORB[2865.80,2889.20] vol=1.9x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-01-05 10:55:00 | 2885.95 | 2883.30 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 2853.40 | 2861.73 | 0.00 | ORB-short ORB[2859.10,2871.70] vol=1.5x ATR=6.29 |
| Stop hit — per-position SL triggered | 2026-01-06 11:30:00 | 2859.69 | 2856.65 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 2801.00 | 2815.10 | 0.00 | ORB-short ORB[2809.60,2830.00] vol=2.2x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:25:00 | 2794.01 | 2813.12 | 0.00 | T1 1.5R @ 2794.01 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 2801.00 | 2812.16 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:40:00 | 2746.00 | 2751.27 | 0.00 | ORB-short ORB[2747.70,2774.00] vol=3.2x ATR=7.03 |
| Stop hit — per-position SL triggered | 2026-01-12 10:45:00 | 2753.03 | 2752.88 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:15:00 | 2792.70 | 2799.71 | 0.00 | ORB-short ORB[2799.00,2814.80] vol=4.5x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:40:00 | 2783.24 | 2794.88 | 0.00 | T1 1.5R @ 2783.24 |
| Target hit | 2026-01-13 15:20:00 | 2774.80 | 2782.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:05:00 | 2815.00 | 2788.08 | 0.00 | ORB-long ORB[2755.10,2774.50] vol=1.6x ATR=6.41 |
| Stop hit — per-position SL triggered | 2026-01-14 11:35:00 | 2808.59 | 2791.63 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-01-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:40:00 | 2774.20 | 2776.97 | 0.00 | ORB-short ORB[2774.70,2790.80] vol=4.6x ATR=5.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:00:00 | 2766.36 | 2775.51 | 0.00 | T1 1.5R @ 2766.36 |
| Target hit | 2026-01-20 15:20:00 | 2709.30 | 2750.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2026-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 11:00:00 | 2776.50 | 2776.13 | 0.00 | ORB-long ORB[2742.50,2775.00] vol=2.1x ATR=7.24 |
| Stop hit — per-position SL triggered | 2026-01-22 11:45:00 | 2769.26 | 2776.48 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:35:00 | 2822.80 | 2806.11 | 0.00 | ORB-long ORB[2773.00,2815.00] vol=1.9x ATR=9.88 |
| Stop hit — per-position SL triggered | 2026-01-27 10:30:00 | 2812.92 | 2815.58 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-01-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:05:00 | 2863.30 | 2871.39 | 0.00 | ORB-short ORB[2866.00,2891.20] vol=2.3x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 12:00:00 | 2853.44 | 2867.41 | 0.00 | T1 1.5R @ 2853.44 |
| Target hit | 2026-01-28 15:20:00 | 2837.60 | 2855.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 2760.60 | 2738.34 | 0.00 | ORB-long ORB[2720.10,2747.10] vol=2.1x ATR=13.67 |
| Stop hit — per-position SL triggered | 2026-02-02 09:40:00 | 2746.93 | 2740.88 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 2844.50 | 2860.77 | 0.00 | ORB-short ORB[2862.50,2879.00] vol=2.4x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 11:10:00 | 2835.15 | 2859.17 | 0.00 | T1 1.5R @ 2835.15 |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 2844.50 | 2858.84 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 2895.60 | 2871.88 | 0.00 | ORB-long ORB[2846.50,2868.50] vol=3.4x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 2904.83 | 2878.33 | 0.00 | T1 1.5R @ 2904.83 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 2895.60 | 2879.71 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 2911.00 | 2918.64 | 0.00 | ORB-short ORB[2911.20,2934.00] vol=1.5x ATR=6.98 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 2917.98 | 2918.12 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 2891.70 | 2887.56 | 0.00 | ORB-long ORB[2861.30,2878.10] vol=2.8x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:30:00 | 2900.80 | 2891.25 | 0.00 | T1 1.5R @ 2900.80 |
| Target hit | 2026-02-16 15:20:00 | 2911.00 | 2900.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 2920.00 | 2909.72 | 0.00 | ORB-long ORB[2900.00,2913.20] vol=2.6x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:55:00 | 2927.44 | 2912.08 | 0.00 | T1 1.5R @ 2927.44 |
| Stop hit — per-position SL triggered | 2026-02-18 13:05:00 | 2920.00 | 2923.53 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 2914.50 | 2934.96 | 0.00 | ORB-short ORB[2927.90,2946.50] vol=2.8x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:35:00 | 2906.21 | 2932.69 | 0.00 | T1 1.5R @ 2906.21 |
| Target hit | 2026-02-19 15:20:00 | 2857.20 | 2892.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 2849.00 | 2859.51 | 0.00 | ORB-short ORB[2850.20,2885.00] vol=1.6x ATR=7.10 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 2856.10 | 2856.01 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 2909.00 | 2901.45 | 0.00 | ORB-long ORB[2883.50,2901.20] vol=5.9x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 2904.20 | 2902.46 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 2889.30 | 2882.55 | 0.00 | ORB-long ORB[2873.20,2887.20] vol=1.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 2884.52 | 2882.55 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 2808.30 | 2831.35 | 0.00 | ORB-short ORB[2839.10,2864.60] vol=1.9x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:45:00 | 2798.33 | 2823.42 | 0.00 | T1 1.5R @ 2798.33 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 2808.30 | 2812.01 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 2720.00 | 2716.29 | 0.00 | ORB-long ORB[2687.20,2718.40] vol=1.9x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:15:00 | 2732.19 | 2718.15 | 0.00 | T1 1.5R @ 2732.19 |
| Target hit | 2026-03-06 14:10:00 | 2738.10 | 2739.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 96 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 2725.00 | 2712.83 | 0.00 | ORB-long ORB[2701.00,2724.10] vol=2.2x ATR=8.89 |
| Stop hit — per-position SL triggered | 2026-03-10 10:55:00 | 2716.11 | 2714.67 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 2749.80 | 2742.56 | 0.00 | ORB-long ORB[2726.60,2743.90] vol=4.3x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:10:00 | 2758.83 | 2750.14 | 0.00 | T1 1.5R @ 2758.83 |
| Stop hit — per-position SL triggered | 2026-03-11 12:05:00 | 2749.80 | 2751.44 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 2636.80 | 2640.52 | 0.00 | ORB-short ORB[2644.00,2664.00] vol=2.7x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:40:00 | 2625.02 | 2638.77 | 0.00 | T1 1.5R @ 2625.02 |
| Target hit | 2026-03-13 10:50:00 | 2634.10 | 2632.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 99 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 2718.10 | 2712.00 | 0.00 | ORB-long ORB[2686.70,2713.00] vol=3.2x ATR=6.74 |
| Stop hit — per-position SL triggered | 2026-03-18 13:05:00 | 2711.36 | 2714.59 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 2535.00 | 2550.96 | 0.00 | ORB-short ORB[2565.50,2597.60] vol=2.1x ATR=10.61 |
| Stop hit — per-position SL triggered | 2026-03-24 11:35:00 | 2545.61 | 2545.00 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 2560.90 | 2574.34 | 0.00 | ORB-short ORB[2582.30,2620.10] vol=3.1x ATR=10.59 |
| Stop hit — per-position SL triggered | 2026-03-30 11:25:00 | 2571.49 | 2573.43 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2026-04-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 11:05:00 | 2539.50 | 2551.47 | 0.00 | ORB-short ORB[2542.40,2568.20] vol=1.7x ATR=8.16 |
| Stop hit — per-position SL triggered | 2026-04-06 11:25:00 | 2547.66 | 2548.90 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 2784.80 | 2779.12 | 0.00 | ORB-long ORB[2755.00,2780.00] vol=2.5x ATR=7.18 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 2777.62 | 2779.88 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 2798.90 | 2772.14 | 0.00 | ORB-long ORB[2763.10,2788.70] vol=1.9x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:50:00 | 2809.67 | 2778.43 | 0.00 | T1 1.5R @ 2809.67 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 2798.90 | 2782.72 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2026-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:35:00 | 2766.00 | 2757.49 | 0.00 | ORB-long ORB[2742.60,2762.30] vol=2.9x ATR=7.01 |
| Stop hit — per-position SL triggered | 2026-04-24 10:45:00 | 2758.99 | 2757.68 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 2782.00 | 2771.62 | 0.00 | ORB-long ORB[2752.20,2779.80] vol=3.5x ATR=8.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 13:15:00 | 2794.59 | 2777.58 | 0.00 | T1 1.5R @ 2794.59 |
| Stop hit — per-position SL triggered | 2026-04-27 14:25:00 | 2782.00 | 2782.76 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 2814.00 | 2802.48 | 0.00 | ORB-long ORB[2777.00,2799.00] vol=1.8x ATR=5.68 |
| Stop hit — per-position SL triggered | 2026-04-29 10:55:00 | 2808.32 | 2802.89 | 0.00 | SL hit |

### Cycle 108 — BUY (started 2026-05-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:50:00 | 2851.80 | 2831.07 | 0.00 | ORB-long ORB[2805.00,2834.00] vol=2.1x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-05-04 10:55:00 | 2844.89 | 2831.59 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 2851.20 | 2848.24 | 0.00 | ORB-long ORB[2823.60,2845.80] vol=1.9x ATR=7.02 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 2844.18 | 2848.07 | 0.00 | SL hit |

### Cycle 110 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 2964.60 | 2939.35 | 0.00 | ORB-long ORB[2900.00,2939.50] vol=2.1x ATR=9.79 |
| Stop hit — per-position SL triggered | 2026-05-07 10:00:00 | 2954.81 | 2950.80 | 0.00 | SL hit |

### Cycle 111 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 2968.00 | 2955.84 | 0.00 | ORB-long ORB[2936.50,2961.40] vol=2.4x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:40:00 | 2980.36 | 2958.97 | 0.00 | T1 1.5R @ 2980.36 |
| Stop hit — per-position SL triggered | 2026-05-08 10:45:00 | 2968.00 | 2959.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:50:00 | 2732.80 | 2025-05-12 11:15:00 | 2755.12 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-05-12 10:50:00 | 2732.80 | 2025-05-12 12:15:00 | 2732.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-13 11:10:00 | 2710.40 | 2025-05-13 11:15:00 | 2716.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-15 10:35:00 | 2772.70 | 2025-05-15 11:10:00 | 2763.63 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-19 11:00:00 | 2778.50 | 2025-05-19 13:10:00 | 2769.75 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-05-19 11:00:00 | 2778.50 | 2025-05-19 15:20:00 | 2723.40 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2025-05-22 09:30:00 | 2665.10 | 2025-05-22 09:35:00 | 2672.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-28 10:40:00 | 2593.60 | 2025-05-28 11:20:00 | 2598.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-29 11:00:00 | 2564.00 | 2025-05-29 11:05:00 | 2569.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-30 11:00:00 | 2573.90 | 2025-05-30 11:40:00 | 2565.07 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-30 11:00:00 | 2573.90 | 2025-05-30 15:20:00 | 2539.90 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2025-06-02 10:55:00 | 2516.30 | 2025-06-02 11:45:00 | 2522.12 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-03 10:55:00 | 2556.20 | 2025-06-03 11:10:00 | 2549.36 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-05 09:30:00 | 2574.00 | 2025-06-05 09:45:00 | 2582.91 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-06-05 09:30:00 | 2574.00 | 2025-06-05 10:30:00 | 2579.40 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-06-06 11:05:00 | 2570.10 | 2025-06-06 12:35:00 | 2564.32 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-09 09:45:00 | 2593.00 | 2025-06-09 09:55:00 | 2601.42 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-09 09:45:00 | 2593.00 | 2025-06-09 10:55:00 | 2600.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-19 09:35:00 | 2702.70 | 2025-06-19 09:45:00 | 2697.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-20 10:45:00 | 2718.80 | 2025-06-20 11:00:00 | 2729.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-06-20 10:45:00 | 2718.80 | 2025-06-20 12:05:00 | 2718.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 10:45:00 | 2766.40 | 2025-06-24 11:00:00 | 2776.98 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-24 10:45:00 | 2766.40 | 2025-06-24 13:45:00 | 2775.70 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-02 10:55:00 | 2848.00 | 2025-07-02 11:00:00 | 2840.87 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-02 10:55:00 | 2848.00 | 2025-07-02 11:25:00 | 2848.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 09:40:00 | 2800.00 | 2025-07-04 10:00:00 | 2806.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-07 10:45:00 | 2793.50 | 2025-07-07 11:50:00 | 2787.16 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-07 10:45:00 | 2793.50 | 2025-07-07 15:20:00 | 2779.60 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-07-08 09:55:00 | 2803.00 | 2025-07-08 10:05:00 | 2809.54 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-07-08 09:55:00 | 2803.00 | 2025-07-08 11:35:00 | 2807.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-07-10 11:10:00 | 2795.00 | 2025-07-10 11:20:00 | 2798.95 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-11 09:45:00 | 2806.00 | 2025-07-11 09:55:00 | 2798.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-14 09:40:00 | 2794.80 | 2025-07-14 09:50:00 | 2788.78 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-17 11:10:00 | 2748.50 | 2025-07-17 11:35:00 | 2752.44 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-18 09:55:00 | 2748.00 | 2025-07-18 10:15:00 | 2739.05 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-18 09:55:00 | 2748.00 | 2025-07-18 13:40:00 | 2732.50 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-22 10:40:00 | 2731.70 | 2025-07-22 11:15:00 | 2738.02 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-24 11:15:00 | 2690.00 | 2025-07-24 11:25:00 | 2683.29 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-24 11:15:00 | 2690.00 | 2025-07-24 12:15:00 | 2690.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 10:05:00 | 2710.90 | 2025-07-25 10:10:00 | 2717.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-28 10:30:00 | 2748.00 | 2025-07-28 10:40:00 | 2739.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-29 10:55:00 | 2720.00 | 2025-07-29 11:25:00 | 2710.67 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-29 10:55:00 | 2720.00 | 2025-07-29 13:25:00 | 2720.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 09:45:00 | 2765.50 | 2025-07-30 10:05:00 | 2776.34 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-30 09:45:00 | 2765.50 | 2025-07-30 13:35:00 | 2778.60 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-05 10:50:00 | 2809.20 | 2025-08-05 11:55:00 | 2801.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-06 11:00:00 | 2767.60 | 2025-08-06 12:15:00 | 2773.23 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-07 10:40:00 | 2747.90 | 2025-08-07 11:00:00 | 2738.83 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-07 10:40:00 | 2747.90 | 2025-08-07 15:00:00 | 2742.10 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-08-12 10:30:00 | 2708.00 | 2025-08-12 11:00:00 | 2715.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-13 10:40:00 | 2743.20 | 2025-08-13 11:05:00 | 2734.33 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-13 10:40:00 | 2743.20 | 2025-08-13 13:00:00 | 2743.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-19 10:50:00 | 2827.00 | 2025-08-19 11:55:00 | 2819.41 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-08-19 10:50:00 | 2827.00 | 2025-08-19 14:30:00 | 2819.80 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-08-20 10:35:00 | 2828.90 | 2025-08-20 11:05:00 | 2836.87 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-08-20 10:35:00 | 2828.90 | 2025-08-20 15:20:00 | 2864.80 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-08-21 10:15:00 | 2887.90 | 2025-08-21 12:20:00 | 2880.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-03 11:00:00 | 2805.80 | 2025-09-03 11:15:00 | 2801.07 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-10 09:30:00 | 2815.20 | 2025-09-10 09:45:00 | 2810.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-12 11:05:00 | 2813.10 | 2025-09-12 12:20:00 | 2809.23 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-22 11:00:00 | 2899.00 | 2025-09-22 11:05:00 | 2906.59 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-09-22 11:00:00 | 2899.00 | 2025-09-22 11:15:00 | 2899.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 11:05:00 | 2820.80 | 2025-09-23 11:35:00 | 2811.18 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-23 11:05:00 | 2820.80 | 2025-09-23 12:30:00 | 2820.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-24 11:00:00 | 2829.30 | 2025-09-24 11:05:00 | 2823.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-25 09:50:00 | 2831.00 | 2025-09-25 10:05:00 | 2822.79 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-01 11:00:00 | 2740.50 | 2025-10-01 11:40:00 | 2746.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-10 09:30:00 | 2819.00 | 2025-10-10 09:40:00 | 2812.22 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-14 10:55:00 | 2783.40 | 2025-10-14 11:05:00 | 2777.42 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-10-14 10:55:00 | 2783.40 | 2025-10-14 14:25:00 | 2776.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-10-15 10:20:00 | 2799.30 | 2025-10-15 11:15:00 | 2794.75 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-16 10:25:00 | 2844.20 | 2025-10-16 10:40:00 | 2838.92 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-17 11:05:00 | 2881.60 | 2025-10-17 11:40:00 | 2875.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-20 10:55:00 | 2872.30 | 2025-10-20 11:20:00 | 2865.40 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-23 09:30:00 | 2894.00 | 2025-10-23 09:50:00 | 2903.68 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-23 09:30:00 | 2894.00 | 2025-10-23 11:05:00 | 2895.70 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-10-24 10:20:00 | 2849.70 | 2025-10-24 11:05:00 | 2838.90 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-24 10:20:00 | 2849.70 | 2025-10-24 12:15:00 | 2849.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 10:05:00 | 2875.90 | 2025-10-27 10:10:00 | 2869.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-31 10:55:00 | 2918.60 | 2025-10-31 11:20:00 | 2909.20 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-31 10:55:00 | 2918.60 | 2025-10-31 11:25:00 | 2918.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:50:00 | 2910.70 | 2025-11-03 11:00:00 | 2903.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-06 09:50:00 | 2756.00 | 2025-11-06 10:05:00 | 2735.64 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-11-06 09:50:00 | 2756.00 | 2025-11-06 15:20:00 | 2699.40 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2025-11-10 11:00:00 | 2764.00 | 2025-11-10 11:10:00 | 2771.38 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-11-10 11:00:00 | 2764.00 | 2025-11-10 15:05:00 | 2764.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:55:00 | 2749.10 | 2025-11-11 11:00:00 | 2754.15 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-14 10:35:00 | 2780.30 | 2025-11-14 12:10:00 | 2789.46 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-14 10:35:00 | 2780.30 | 2025-11-14 12:45:00 | 2780.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-18 11:00:00 | 2761.00 | 2025-11-18 11:15:00 | 2764.66 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-04 10:50:00 | 2737.30 | 2025-12-04 11:35:00 | 2733.07 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-08 11:10:00 | 2763.30 | 2025-12-08 11:40:00 | 2757.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-09 10:35:00 | 2758.20 | 2025-12-09 11:40:00 | 2751.06 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-10 11:00:00 | 2759.70 | 2025-12-10 11:10:00 | 2765.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-11 11:05:00 | 2774.10 | 2025-12-11 11:30:00 | 2782.02 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-11 11:05:00 | 2774.10 | 2025-12-11 12:20:00 | 2774.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 10:40:00 | 2837.90 | 2025-12-23 11:40:00 | 2832.39 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-24 10:55:00 | 2843.90 | 2025-12-24 11:10:00 | 2850.38 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-12-24 10:55:00 | 2843.90 | 2025-12-24 12:05:00 | 2843.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 10:20:00 | 2837.10 | 2025-12-26 10:30:00 | 2831.78 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-29 09:30:00 | 2829.80 | 2025-12-29 09:35:00 | 2823.85 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 10:50:00 | 2875.50 | 2026-01-02 10:55:00 | 2869.99 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-05 10:50:00 | 2891.00 | 2026-01-05 10:55:00 | 2885.95 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-01-06 10:15:00 | 2853.40 | 2026-01-06 11:30:00 | 2859.69 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-08 11:10:00 | 2801.00 | 2026-01-08 11:25:00 | 2794.01 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-01-08 11:10:00 | 2801.00 | 2026-01-08 11:35:00 | 2801.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-12 10:40:00 | 2746.00 | 2026-01-12 10:45:00 | 2753.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-13 11:15:00 | 2792.70 | 2026-01-13 11:40:00 | 2783.24 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-13 11:15:00 | 2792.70 | 2026-01-13 15:20:00 | 2774.80 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-01-14 11:05:00 | 2815.00 | 2026-01-14 11:35:00 | 2808.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-20 10:40:00 | 2774.20 | 2026-01-20 12:00:00 | 2766.36 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-01-20 10:40:00 | 2774.20 | 2026-01-20 15:20:00 | 2709.30 | TARGET_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2026-01-22 11:00:00 | 2776.50 | 2026-01-22 11:45:00 | 2769.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-27 09:35:00 | 2822.80 | 2026-01-27 10:30:00 | 2812.92 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-28 11:05:00 | 2863.30 | 2026-01-28 12:00:00 | 2853.44 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-28 11:05:00 | 2863.30 | 2026-01-28 15:20:00 | 2837.60 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-02-02 09:30:00 | 2760.60 | 2026-02-02 09:40:00 | 2746.93 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-02-06 11:00:00 | 2844.50 | 2026-02-06 11:10:00 | 2835.15 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-06 11:00:00 | 2844.50 | 2026-02-06 11:15:00 | 2844.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 10:55:00 | 2895.60 | 2026-02-09 11:20:00 | 2904.83 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-09 10:55:00 | 2895.60 | 2026-02-09 11:30:00 | 2895.60 | STOP_HIT | 0.50 | 0.00% |
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
