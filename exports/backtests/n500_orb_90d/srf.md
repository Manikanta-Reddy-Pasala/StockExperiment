# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2778.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 16
- **Target hits / Stop hits / Partials:** 7 / 16 / 11
- **Avg / median % per leg:** 0.33% / 0.36%
- **Sum % (uncompounded):** 11.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.22% | 4.2% |
| BUY @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.22% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 9 | 60.0% | 4 | 6 | 5 | 0.46% | 6.9% |
| SELL @ 2nd Alert (retest1) | 15 | 9 | 60.0% | 4 | 6 | 5 | 0.46% | 6.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 18 | 52.9% | 7 | 16 | 11 | 0.33% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:50:00 | 2968.50 | 2928.33 | 0.00 | ORB-long ORB[2905.60,2949.50] vol=1.7x ATR=17.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:20:00 | 2994.78 | 2944.28 | 0.00 | T1 1.5R @ 2994.78 |
| Target hit | 2026-02-09 15:20:00 | 2985.10 | 2974.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 2981.90 | 2982.84 | 0.00 | ORB-short ORB[2992.20,3014.90] vol=2.6x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 2970.53 | 2982.22 | 0.00 | T1 1.5R @ 2970.53 |
| Target hit | 2026-02-10 11:40:00 | 2981.30 | 2981.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:55:00 | 2904.00 | 2908.26 | 0.00 | ORB-short ORB[2905.00,2937.40] vol=4.7x ATR=7.99 |
| Stop hit — per-position SL triggered | 2026-02-12 10:30:00 | 2911.99 | 2907.44 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 2862.70 | 2852.14 | 0.00 | ORB-long ORB[2835.00,2848.90] vol=2.1x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 2856.39 | 2859.45 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 2665.00 | 2685.33 | 0.00 | ORB-short ORB[2688.00,2718.90] vol=2.5x ATR=6.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 12:05:00 | 2654.68 | 2680.64 | 0.00 | T1 1.5R @ 2654.68 |
| Target hit | 2026-02-23 15:20:00 | 2640.70 | 2658.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 2588.10 | 2598.81 | 0.00 | ORB-short ORB[2593.80,2632.00] vol=2.1x ATR=8.41 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 2596.51 | 2598.54 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 2603.20 | 2590.19 | 0.00 | ORB-long ORB[2575.40,2594.10] vol=2.1x ATR=7.24 |
| Stop hit — per-position SL triggered | 2026-02-25 11:50:00 | 2595.96 | 2595.78 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 2621.20 | 2619.84 | 0.00 | ORB-long ORB[2606.70,2620.80] vol=4.4x ATR=5.90 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 2615.30 | 2620.42 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:10:00 | 2558.00 | 2551.74 | 0.00 | ORB-long ORB[2541.10,2556.00] vol=5.3x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 2553.02 | 2552.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 2602.40 | 2588.80 | 0.00 | ORB-long ORB[2555.10,2592.60] vol=1.9x ATR=10.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:05:00 | 2617.42 | 2599.86 | 0.00 | T1 1.5R @ 2617.42 |
| Target hit | 2026-03-06 15:20:00 | 2625.20 | 2618.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 2578.60 | 2587.95 | 0.00 | ORB-short ORB[2585.20,2608.50] vol=1.9x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:00:00 | 2569.20 | 2583.99 | 0.00 | T1 1.5R @ 2569.20 |
| Target hit | 2026-03-11 15:20:00 | 2490.50 | 2523.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:10:00 | 2543.00 | 2574.42 | 0.00 | ORB-short ORB[2600.30,2623.50] vol=3.3x ATR=11.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 2526.41 | 2570.98 | 0.00 | T1 1.5R @ 2526.41 |
| Target hit | 2026-03-13 15:20:00 | 2497.90 | 2528.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 2475.90 | 2489.13 | 0.00 | ORB-short ORB[2484.90,2504.80] vol=1.5x ATR=8.51 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 2484.41 | 2485.84 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 2469.50 | 2456.33 | 0.00 | ORB-long ORB[2441.10,2463.80] vol=1.5x ATR=7.93 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 2461.57 | 2457.76 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:45:00 | 2392.10 | 2410.70 | 0.00 | ORB-short ORB[2408.10,2440.00] vol=3.6x ATR=8.69 |
| Stop hit — per-position SL triggered | 2026-03-23 12:35:00 | 2400.79 | 2401.81 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 2494.40 | 2487.16 | 0.00 | ORB-long ORB[2466.80,2494.00] vol=1.8x ATR=6.57 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 2487.83 | 2489.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 2490.30 | 2471.56 | 0.00 | ORB-long ORB[2452.00,2477.60] vol=1.6x ATR=6.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:05:00 | 2499.67 | 2476.35 | 0.00 | T1 1.5R @ 2499.67 |
| Stop hit — per-position SL triggered | 2026-04-22 13:20:00 | 2490.30 | 2484.56 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 2506.20 | 2484.92 | 0.00 | ORB-long ORB[2466.00,2488.60] vol=2.2x ATR=7.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:30:00 | 2516.88 | 2504.30 | 0.00 | T1 1.5R @ 2516.88 |
| Target hit | 2026-04-23 15:20:00 | 2540.10 | 2527.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:15:00 | 2535.50 | 2521.74 | 0.00 | ORB-long ORB[2503.90,2525.00] vol=1.8x ATR=8.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:25:00 | 2547.80 | 2527.91 | 0.00 | T1 1.5R @ 2547.80 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 2535.50 | 2529.26 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 2585.40 | 2569.64 | 0.00 | ORB-long ORB[2547.50,2576.50] vol=3.6x ATR=9.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:20:00 | 2599.12 | 2580.91 | 0.00 | T1 1.5R @ 2599.12 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 2585.40 | 2584.00 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:25:00 | 2492.00 | 2512.08 | 0.00 | ORB-short ORB[2513.40,2543.50] vol=2.6x ATR=6.83 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 2498.83 | 2509.59 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 2557.20 | 2537.56 | 0.00 | ORB-long ORB[2520.40,2549.90] vol=2.2x ATR=8.80 |
| Stop hit — per-position SL triggered | 2026-05-04 12:50:00 | 2548.40 | 2545.20 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:20:00 | 2522.00 | 2543.56 | 0.00 | ORB-short ORB[2535.30,2567.80] vol=2.3x ATR=9.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:45:00 | 2507.92 | 2532.27 | 0.00 | T1 1.5R @ 2507.92 |
| Stop hit — per-position SL triggered | 2026-05-05 12:05:00 | 2522.00 | 2527.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:50:00 | 2968.50 | 2026-02-09 12:20:00 | 2994.78 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2026-02-09 10:50:00 | 2968.50 | 2026-02-09 15:20:00 | 2985.10 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-10 10:55:00 | 2981.90 | 2026-02-10 11:10:00 | 2970.53 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-10 10:55:00 | 2981.90 | 2026-02-10 11:40:00 | 2981.30 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2026-02-12 09:55:00 | 2904.00 | 2026-02-12 10:30:00 | 2911.99 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-17 09:50:00 | 2862.70 | 2026-02-17 10:30:00 | 2856.39 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-23 11:10:00 | 2665.00 | 2026-02-23 12:05:00 | 2654.68 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-23 11:10:00 | 2665.00 | 2026-02-23 15:20:00 | 2640.70 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-02-24 09:35:00 | 2588.10 | 2026-02-24 09:40:00 | 2596.51 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-25 10:55:00 | 2603.20 | 2026-02-25 11:50:00 | 2595.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-26 10:05:00 | 2621.20 | 2026-02-26 11:35:00 | 2615.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-05 11:10:00 | 2558.00 | 2026-03-05 11:25:00 | 2553.02 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-03-06 09:30:00 | 2602.40 | 2026-03-06 10:05:00 | 2617.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-06 09:30:00 | 2602.40 | 2026-03-06 15:20:00 | 2625.20 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2026-03-11 09:55:00 | 2578.60 | 2026-03-11 10:00:00 | 2569.20 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-11 09:55:00 | 2578.60 | 2026-03-11 15:20:00 | 2490.50 | TARGET_HIT | 0.50 | 3.42% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2543.00 | 2026-03-13 10:20:00 | 2526.41 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2543.00 | 2026-03-13 15:20:00 | 2497.90 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2026-03-16 10:35:00 | 2475.90 | 2026-03-16 11:00:00 | 2484.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 10:25:00 | 2469.50 | 2026-03-17 10:30:00 | 2461.57 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-23 10:45:00 | 2392.10 | 2026-03-23 12:35:00 | 2400.79 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 09:35:00 | 2494.40 | 2026-04-21 09:50:00 | 2487.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-22 10:55:00 | 2490.30 | 2026-04-22 11:05:00 | 2499.67 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-22 10:55:00 | 2490.30 | 2026-04-22 13:20:00 | 2490.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:00:00 | 2506.20 | 2026-04-23 11:30:00 | 2516.88 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-23 10:00:00 | 2506.20 | 2026-04-23 15:20:00 | 2540.10 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2026-04-27 10:15:00 | 2535.50 | 2026-04-27 10:25:00 | 2547.80 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-27 10:15:00 | 2535.50 | 2026-04-27 10:45:00 | 2535.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:55:00 | 2585.40 | 2026-04-29 10:20:00 | 2599.12 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-29 09:55:00 | 2585.40 | 2026-04-29 10:50:00 | 2585.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:25:00 | 2492.00 | 2026-04-30 11:05:00 | 2498.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-04 11:00:00 | 2557.20 | 2026-05-04 12:50:00 | 2548.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-05 10:20:00 | 2522.00 | 2026-05-05 11:45:00 | 2507.92 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-05-05 10:20:00 | 2522.00 | 2026-05-05 12:05:00 | 2522.00 | STOP_HIT | 0.50 | 0.00% |
