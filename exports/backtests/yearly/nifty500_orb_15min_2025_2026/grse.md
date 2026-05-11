# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 3043.00
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
| ENTRY1 | 28 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 5 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 23
- **Target hits / Stop hits / Partials:** 5 / 23 / 13
- **Avg / median % per leg:** 0.34% / 0.00%
- **Sum % (uncompounded):** 13.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.30% | 7.6% |
| BUY @ 2nd Alert (retest1) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.30% | 7.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.39% | 6.2% |
| SELL @ 2nd Alert (retest1) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.39% | 6.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 41 | 18 | 43.9% | 5 | 23 | 13 | 0.34% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:50:00 | 3055.60 | 3033.26 | 0.00 | ORB-long ORB[3004.90,3050.00] vol=1.7x ATR=14.26 |
| Stop hit — per-position SL triggered | 2025-07-01 09:55:00 | 3041.34 | 3034.35 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 2649.10 | 2660.42 | 0.00 | ORB-short ORB[2650.50,2686.00] vol=3.1x ATR=9.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:40:00 | 2634.33 | 2655.48 | 0.00 | T1 1.5R @ 2634.33 |
| Target hit | 2025-07-16 12:15:00 | 2643.50 | 2641.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 2656.40 | 2629.23 | 0.00 | ORB-long ORB[2611.90,2644.00] vol=2.2x ATR=11.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:45:00 | 2673.43 | 2639.42 | 0.00 | T1 1.5R @ 2673.43 |
| Stop hit — per-position SL triggered | 2025-08-18 09:55:00 | 2656.40 | 2644.45 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 2576.90 | 2609.07 | 0.00 | ORB-short ORB[2601.10,2635.90] vol=2.2x ATR=13.03 |
| Stop hit — per-position SL triggered | 2025-08-20 10:00:00 | 2589.93 | 2598.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:15:00 | 2615.20 | 2580.17 | 0.00 | ORB-long ORB[2555.30,2588.00] vol=5.2x ATR=11.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:30:00 | 2631.73 | 2592.96 | 0.00 | T1 1.5R @ 2631.73 |
| Stop hit — per-position SL triggered | 2025-08-22 10:35:00 | 2615.20 | 2594.03 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-09-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:50:00 | 2404.80 | 2376.88 | 0.00 | ORB-long ORB[2355.00,2388.60] vol=3.9x ATR=11.89 |
| Stop hit — per-position SL triggered | 2025-09-09 09:55:00 | 2392.91 | 2378.63 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 2405.00 | 2384.06 | 0.00 | ORB-long ORB[2360.20,2385.20] vol=1.9x ATR=9.67 |
| Stop hit — per-position SL triggered | 2025-09-10 11:35:00 | 2395.33 | 2388.98 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:45:00 | 2397.60 | 2373.73 | 0.00 | ORB-long ORB[2357.50,2373.60] vol=2.1x ATR=9.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:50:00 | 2412.11 | 2385.07 | 0.00 | T1 1.5R @ 2412.11 |
| Target hit | 2025-09-12 15:20:00 | 2586.40 | 2510.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-10-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:20:00 | 2685.00 | 2715.55 | 0.00 | ORB-short ORB[2700.30,2729.00] vol=1.7x ATR=11.65 |
| Stop hit — per-position SL triggered | 2025-10-08 10:25:00 | 2696.65 | 2714.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:00:00 | 2711.00 | 2698.70 | 0.00 | ORB-long ORB[2684.10,2710.40] vol=2.3x ATR=13.03 |
| Stop hit — per-position SL triggered | 2025-10-09 10:10:00 | 2697.97 | 2699.16 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 2609.20 | 2618.64 | 0.00 | ORB-short ORB[2612.70,2629.90] vol=1.7x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:55:00 | 2596.85 | 2612.79 | 0.00 | T1 1.5R @ 2596.85 |
| Target hit | 2025-10-14 15:20:00 | 2553.70 | 2586.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:35:00 | 2594.20 | 2569.73 | 0.00 | ORB-long ORB[2555.00,2579.00] vol=3.1x ATR=11.68 |
| Stop hit — per-position SL triggered | 2025-10-15 10:40:00 | 2582.52 | 2570.54 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 2626.40 | 2612.19 | 0.00 | ORB-long ORB[2590.10,2622.10] vol=3.9x ATR=9.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:35:00 | 2641.25 | 2617.95 | 0.00 | T1 1.5R @ 2641.25 |
| Target hit | 2025-10-17 10:20:00 | 2635.10 | 2636.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2025-10-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:35:00 | 2626.20 | 2616.12 | 0.00 | ORB-long ORB[2598.00,2625.40] vol=2.4x ATR=10.08 |
| Stop hit — per-position SL triggered | 2025-10-20 09:40:00 | 2616.12 | 2616.57 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 2558.00 | 2569.51 | 0.00 | ORB-short ORB[2563.50,2584.40] vol=1.8x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:20:00 | 2548.20 | 2564.18 | 0.00 | T1 1.5R @ 2548.20 |
| Stop hit — per-position SL triggered | 2025-10-30 11:05:00 | 2558.00 | 2564.04 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-11-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:50:00 | 2764.00 | 2747.00 | 0.00 | ORB-long ORB[2725.00,2760.00] vol=1.6x ATR=9.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:00:00 | 2777.89 | 2748.95 | 0.00 | T1 1.5R @ 2777.89 |
| Stop hit — per-position SL triggered | 2025-11-26 11:10:00 | 2764.00 | 2750.39 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 2788.20 | 2773.41 | 0.00 | ORB-long ORB[2755.90,2786.40] vol=3.6x ATR=8.63 |
| Stop hit — per-position SL triggered | 2025-11-27 09:35:00 | 2779.57 | 2774.37 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 2580.10 | 2562.95 | 0.00 | ORB-long ORB[2540.00,2576.60] vol=2.1x ATR=10.63 |
| Stop hit — per-position SL triggered | 2025-12-04 09:50:00 | 2569.47 | 2566.85 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 2538.20 | 2556.04 | 0.00 | ORB-short ORB[2545.20,2574.70] vol=2.0x ATR=8.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:35:00 | 2525.21 | 2547.96 | 0.00 | T1 1.5R @ 2525.21 |
| Target hit | 2025-12-05 15:20:00 | 2470.30 | 2498.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:30:00 | 2319.80 | 2328.46 | 0.00 | ORB-short ORB[2320.50,2347.60] vol=1.8x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 09:40:00 | 2312.03 | 2324.54 | 0.00 | T1 1.5R @ 2312.03 |
| Stop hit — per-position SL triggered | 2025-12-16 09:45:00 | 2319.80 | 2323.43 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 2561.00 | 2537.39 | 0.00 | ORB-long ORB[2511.00,2548.00] vol=3.7x ATR=11.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 09:35:00 | 2578.39 | 2545.80 | 0.00 | T1 1.5R @ 2578.39 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 2561.00 | 2547.72 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:10:00 | 2538.90 | 2510.17 | 0.00 | ORB-long ORB[2491.10,2529.00] vol=2.4x ATR=11.51 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 2527.39 | 2512.16 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:00:00 | 2290.00 | 2308.77 | 0.00 | ORB-short ORB[2295.40,2328.00] vol=2.0x ATR=10.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 12:00:00 | 2274.37 | 2303.78 | 0.00 | T1 1.5R @ 2274.37 |
| Stop hit — per-position SL triggered | 2026-01-22 12:20:00 | 2290.00 | 2302.96 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 2494.80 | 2475.01 | 0.00 | ORB-long ORB[2455.30,2483.60] vol=3.6x ATR=9.55 |
| Stop hit — per-position SL triggered | 2026-02-12 09:45:00 | 2485.25 | 2480.15 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2449.00 | 2464.81 | 0.00 | ORB-short ORB[2456.60,2488.00] vol=1.8x ATR=9.99 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 2458.99 | 2461.07 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2407.90 | 2420.73 | 0.00 | ORB-short ORB[2410.10,2440.00] vol=1.9x ATR=9.37 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 2417.27 | 2415.65 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 2455.00 | 2436.49 | 0.00 | ORB-long ORB[2418.40,2438.90] vol=3.0x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:35:00 | 2467.02 | 2447.22 | 0.00 | T1 1.5R @ 2467.02 |
| Stop hit — per-position SL triggered | 2026-02-26 09:40:00 | 2455.00 | 2447.95 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2392.20 | 2370.65 | 0.00 | ORB-long ORB[2350.00,2377.00] vol=2.5x ATR=11.35 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 2380.85 | 2376.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-01 09:50:00 | 3055.60 | 2025-07-01 09:55:00 | 3041.34 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-07-16 09:30:00 | 2649.10 | 2025-07-16 09:40:00 | 2634.33 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-16 09:30:00 | 2649.10 | 2025-07-16 12:15:00 | 2643.50 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-08-18 09:40:00 | 2656.40 | 2025-08-18 09:45:00 | 2673.43 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-08-18 09:40:00 | 2656.40 | 2025-08-18 09:55:00 | 2656.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-20 09:30:00 | 2576.90 | 2025-08-20 10:00:00 | 2589.93 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-08-22 10:15:00 | 2615.20 | 2025-08-22 10:30:00 | 2631.73 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-08-22 10:15:00 | 2615.20 | 2025-08-22 10:35:00 | 2615.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-09 09:50:00 | 2404.80 | 2025-09-09 09:55:00 | 2392.91 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-09-10 11:00:00 | 2405.00 | 2025-09-10 11:35:00 | 2395.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-09-12 09:45:00 | 2397.60 | 2025-09-12 09:50:00 | 2412.11 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-09-12 09:45:00 | 2397.60 | 2025-09-12 15:20:00 | 2586.40 | TARGET_HIT | 0.50 | 7.87% |
| SELL | retest1 | 2025-10-08 10:20:00 | 2685.00 | 2025-10-08 10:25:00 | 2696.65 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-09 10:00:00 | 2711.00 | 2025-10-09 10:10:00 | 2697.97 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-10-14 09:40:00 | 2609.20 | 2025-10-14 09:55:00 | 2596.85 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-10-14 09:40:00 | 2609.20 | 2025-10-14 15:20:00 | 2553.70 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2025-10-15 10:35:00 | 2594.20 | 2025-10-15 10:40:00 | 2582.52 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-10-17 09:30:00 | 2626.40 | 2025-10-17 09:35:00 | 2641.25 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-17 09:30:00 | 2626.40 | 2025-10-17 10:20:00 | 2635.10 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-20 09:35:00 | 2626.20 | 2025-10-20 09:40:00 | 2616.12 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-30 09:35:00 | 2558.00 | 2025-10-30 10:20:00 | 2548.20 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-30 09:35:00 | 2558.00 | 2025-10-30 11:05:00 | 2558.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 10:50:00 | 2764.00 | 2025-11-26 11:00:00 | 2777.89 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-26 10:50:00 | 2764.00 | 2025-11-26 11:10:00 | 2764.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-27 09:30:00 | 2788.20 | 2025-11-27 09:35:00 | 2779.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-04 09:30:00 | 2580.10 | 2025-12-04 09:50:00 | 2569.47 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-05 09:30:00 | 2538.20 | 2025-12-05 09:35:00 | 2525.21 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-05 09:30:00 | 2538.20 | 2025-12-05 15:20:00 | 2470.30 | TARGET_HIT | 0.50 | 2.68% |
| SELL | retest1 | 2025-12-16 09:30:00 | 2319.80 | 2025-12-16 09:40:00 | 2312.03 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-16 09:30:00 | 2319.80 | 2025-12-16 09:45:00 | 2319.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-29 09:30:00 | 2561.00 | 2025-12-29 09:35:00 | 2578.39 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-12-29 09:30:00 | 2561.00 | 2025-12-29 09:40:00 | 2561.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 11:10:00 | 2538.90 | 2026-01-06 11:20:00 | 2527.39 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-01-22 11:00:00 | 2290.00 | 2026-01-22 12:00:00 | 2274.37 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-01-22 11:00:00 | 2290.00 | 2026-01-22 12:20:00 | 2290.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 09:35:00 | 2494.80 | 2026-02-12 09:45:00 | 2485.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-13 09:30:00 | 2449.00 | 2026-02-13 09:40:00 | 2458.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-24 09:30:00 | 2407.90 | 2026-02-24 10:00:00 | 2417.27 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2455.00 | 2026-02-26 09:35:00 | 2467.02 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2455.00 | 2026-02-26 09:40:00 | 2455.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2392.20 | 2026-03-18 09:55:00 | 2380.85 | STOP_HIT | 1.00 | -0.47% |
