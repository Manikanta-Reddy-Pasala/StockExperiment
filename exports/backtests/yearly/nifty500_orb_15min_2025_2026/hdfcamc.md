# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2843.90
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 19 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 66
- **Target hits / Stop hits / Partials:** 19 / 66 / 34
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 15.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 29 | 40.8% | 9 | 42 | 20 | 0.15% | 10.7% |
| BUY @ 2nd Alert (retest1) | 71 | 29 | 40.8% | 9 | 42 | 20 | 0.15% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 24 | 50.0% | 10 | 24 | 14 | 0.10% | 4.8% |
| SELL @ 2nd Alert (retest1) | 48 | 24 | 50.0% | 10 | 24 | 14 | 0.10% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 119 | 53 | 44.5% | 19 | 66 | 34 | 0.13% | 15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:50:00 | 2273.90 | 2270.28 | 0.00 | ORB-long ORB[2248.60,2272.55] vol=4.9x ATR=7.00 |
| Stop hit — per-position SL triggered | 2025-05-13 11:45:00 | 2266.90 | 2272.32 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:35:00 | 2265.70 | 2258.75 | 0.00 | ORB-long ORB[2241.65,2261.45] vol=2.4x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 12:10:00 | 2273.61 | 2264.67 | 0.00 | T1 1.5R @ 2273.61 |
| Stop hit — per-position SL triggered | 2025-05-14 12:50:00 | 2265.70 | 2265.11 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:55:00 | 2305.00 | 2298.25 | 0.00 | ORB-long ORB[2270.90,2299.50] vol=1.7x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 10:10:00 | 2314.81 | 2300.60 | 0.00 | T1 1.5R @ 2314.81 |
| Target hit | 2025-05-15 15:20:00 | 2353.00 | 2339.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:40:00 | 2382.05 | 2377.17 | 0.00 | ORB-long ORB[2355.05,2376.05] vol=10.1x ATR=5.29 |
| Stop hit — per-position SL triggered | 2025-05-16 10:55:00 | 2376.76 | 2377.37 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:35:00 | 2428.10 | 2415.30 | 0.00 | ORB-long ORB[2391.20,2426.40] vol=2.3x ATR=8.23 |
| Stop hit — per-position SL triggered | 2025-05-22 11:20:00 | 2419.87 | 2423.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:05:00 | 2406.80 | 2400.28 | 0.00 | ORB-long ORB[2385.05,2401.60] vol=1.8x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:10:00 | 2414.95 | 2403.49 | 0.00 | T1 1.5R @ 2414.95 |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 2406.80 | 2403.61 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:15:00 | 2390.20 | 2397.55 | 0.00 | ORB-short ORB[2398.05,2422.50] vol=2.7x ATR=5.99 |
| Stop hit — per-position SL triggered | 2025-05-27 10:35:00 | 2396.19 | 2397.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 2422.50 | 2415.80 | 0.00 | ORB-long ORB[2400.00,2416.20] vol=2.2x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:50:00 | 2429.18 | 2417.37 | 0.00 | T1 1.5R @ 2429.18 |
| Target hit | 2025-05-28 12:30:00 | 2426.00 | 2427.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:15:00 | 2367.55 | 2382.10 | 0.00 | ORB-short ORB[2380.60,2410.70] vol=1.8x ATR=5.83 |
| Stop hit — per-position SL triggered | 2025-06-04 11:20:00 | 2373.38 | 2381.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:00:00 | 2404.35 | 2385.05 | 0.00 | ORB-long ORB[2367.10,2390.90] vol=2.0x ATR=6.24 |
| Stop hit — per-position SL triggered | 2025-06-05 10:35:00 | 2398.11 | 2391.45 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 2447.05 | 2431.67 | 0.00 | ORB-long ORB[2410.50,2443.55] vol=1.5x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:40:00 | 2462.27 | 2440.20 | 0.00 | T1 1.5R @ 2462.27 |
| Target hit | 2025-06-06 15:20:00 | 2561.40 | 2524.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:40:00 | 2583.20 | 2569.14 | 0.00 | ORB-long ORB[2544.60,2582.50] vol=1.6x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:05:00 | 2593.69 | 2579.88 | 0.00 | T1 1.5R @ 2593.69 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 2583.20 | 2581.63 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:55:00 | 2538.65 | 2550.82 | 0.00 | ORB-short ORB[2542.55,2570.00] vol=7.0x ATR=7.42 |
| Stop hit — per-position SL triggered | 2025-06-12 11:05:00 | 2546.07 | 2549.80 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:20:00 | 2501.50 | 2482.91 | 0.00 | ORB-long ORB[2464.05,2485.85] vol=1.8x ATR=9.16 |
| Stop hit — per-position SL triggered | 2025-06-13 10:35:00 | 2492.34 | 2487.59 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:05:00 | 2463.85 | 2481.92 | 0.00 | ORB-short ORB[2467.00,2499.40] vol=1.9x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 2454.99 | 2475.42 | 0.00 | T1 1.5R @ 2454.99 |
| Target hit | 2025-06-19 15:20:00 | 2441.70 | 2456.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:30:00 | 2465.00 | 2451.92 | 0.00 | ORB-long ORB[2434.25,2457.15] vol=2.5x ATR=8.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:30:00 | 2477.90 | 2457.70 | 0.00 | T1 1.5R @ 2477.90 |
| Stop hit — per-position SL triggered | 2025-06-20 12:20:00 | 2465.00 | 2460.86 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:15:00 | 2491.65 | 2475.15 | 0.00 | ORB-long ORB[2460.50,2486.25] vol=2.6x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-06-23 11:20:00 | 2484.54 | 2476.39 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-06-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:40:00 | 2522.50 | 2514.66 | 0.00 | ORB-long ORB[2491.85,2519.30] vol=2.6x ATR=8.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:00:00 | 2534.90 | 2520.13 | 0.00 | T1 1.5R @ 2534.90 |
| Target hit | 2025-06-24 13:30:00 | 2532.55 | 2533.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2025-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:40:00 | 2555.60 | 2544.83 | 0.00 | ORB-long ORB[2532.35,2550.05] vol=2.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2025-06-25 10:05:00 | 2549.27 | 2549.08 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 2561.45 | 2546.50 | 0.00 | ORB-long ORB[2530.70,2552.50] vol=1.6x ATR=6.39 |
| Stop hit — per-position SL triggered | 2025-06-26 09:40:00 | 2555.06 | 2548.05 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2584.15 | 2554.72 | 0.00 | ORB-long ORB[2529.05,2549.40] vol=5.7x ATR=6.57 |
| Stop hit — per-position SL triggered | 2025-06-27 12:05:00 | 2577.58 | 2565.57 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-06-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:35:00 | 2593.25 | 2586.56 | 0.00 | ORB-long ORB[2573.70,2589.00] vol=1.7x ATR=6.77 |
| Stop hit — per-position SL triggered | 2025-06-30 10:50:00 | 2586.48 | 2587.47 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:45:00 | 2579.50 | 2596.82 | 0.00 | ORB-short ORB[2592.25,2610.00] vol=2.0x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:00:00 | 2569.81 | 2594.13 | 0.00 | T1 1.5R @ 2569.81 |
| Target hit | 2025-07-01 15:20:00 | 2570.50 | 2576.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:40:00 | 2551.25 | 2545.79 | 0.00 | ORB-long ORB[2532.00,2550.00] vol=10.5x ATR=6.54 |
| Stop hit — per-position SL triggered | 2025-07-03 11:50:00 | 2544.71 | 2549.80 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 11:00:00 | 2551.50 | 2530.52 | 0.00 | ORB-long ORB[2492.00,2516.75] vol=2.4x ATR=7.81 |
| Stop hit — per-position SL triggered | 2025-07-07 11:05:00 | 2543.69 | 2533.55 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:00:00 | 2567.75 | 2589.35 | 0.00 | ORB-short ORB[2585.25,2610.00] vol=2.8x ATR=6.58 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 2574.33 | 2588.30 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:40:00 | 2597.25 | 2579.76 | 0.00 | ORB-long ORB[2555.25,2575.00] vol=1.7x ATR=6.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:55:00 | 2607.43 | 2583.91 | 0.00 | T1 1.5R @ 2607.43 |
| Target hit | 2025-07-15 12:40:00 | 2610.75 | 2611.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 2665.00 | 2660.96 | 0.00 | ORB-long ORB[2639.25,2657.25] vol=1.9x ATR=9.30 |
| Stop hit — per-position SL triggered | 2025-07-16 10:00:00 | 2655.70 | 2661.97 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-07-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:40:00 | 2806.25 | 2789.95 | 0.00 | ORB-long ORB[2769.50,2800.50] vol=5.0x ATR=7.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:50:00 | 2817.23 | 2793.27 | 0.00 | T1 1.5R @ 2817.23 |
| Stop hit — per-position SL triggered | 2025-07-21 10:55:00 | 2806.25 | 2793.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:55:00 | 2781.75 | 2797.64 | 0.00 | ORB-short ORB[2788.75,2825.00] vol=1.7x ATR=6.45 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 2788.20 | 2795.94 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:10:00 | 2780.00 | 2782.75 | 0.00 | ORB-short ORB[2785.25,2801.50] vol=1.5x ATR=4.63 |
| Stop hit — per-position SL triggered | 2025-07-24 15:20:00 | 2780.25 | 2781.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 2784.50 | 2776.82 | 0.00 | ORB-long ORB[2758.75,2778.75] vol=4.0x ATR=8.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:45:00 | 2797.94 | 2778.37 | 0.00 | T1 1.5R @ 2797.94 |
| Target hit | 2025-07-29 14:00:00 | 2806.25 | 2806.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 2817.00 | 2834.16 | 0.00 | ORB-short ORB[2824.50,2838.50] vol=1.6x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:20:00 | 2807.58 | 2827.62 | 0.00 | T1 1.5R @ 2807.58 |
| Target hit | 2025-08-07 14:10:00 | 2814.00 | 2813.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2025-08-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:20:00 | 2762.50 | 2776.45 | 0.00 | ORB-short ORB[2775.75,2807.25] vol=3.0x ATR=8.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:05:00 | 2749.91 | 2772.01 | 0.00 | T1 1.5R @ 2749.91 |
| Target hit | 2025-08-11 12:50:00 | 2762.00 | 2761.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:15:00 | 2747.50 | 2764.42 | 0.00 | ORB-short ORB[2760.50,2790.00] vol=2.4x ATR=6.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 14:15:00 | 2737.18 | 2758.60 | 0.00 | T1 1.5R @ 2737.18 |
| Target hit | 2025-08-12 15:20:00 | 2732.00 | 2753.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-08-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:10:00 | 2722.50 | 2736.88 | 0.00 | ORB-short ORB[2737.50,2765.00] vol=4.1x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-08-14 10:20:00 | 2728.86 | 2736.10 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 2823.75 | 2804.04 | 0.00 | ORB-long ORB[2757.75,2800.00] vol=2.3x ATR=9.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:05:00 | 2838.15 | 2813.66 | 0.00 | T1 1.5R @ 2838.15 |
| Target hit | 2025-08-18 15:20:00 | 2859.00 | 2840.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 11:00:00 | 2825.00 | 2838.31 | 0.00 | ORB-short ORB[2841.00,2858.50] vol=1.6x ATR=5.00 |
| Stop hit — per-position SL triggered | 2025-08-20 11:10:00 | 2830.00 | 2837.90 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 2946.75 | 2935.64 | 0.00 | ORB-long ORB[2913.25,2929.00] vol=3.9x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 2940.28 | 2935.92 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-08-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:50:00 | 2913.75 | 2922.66 | 0.00 | ORB-short ORB[2918.25,2961.00] vol=4.1x ATR=7.70 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 2921.45 | 2922.52 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-08-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 10:20:00 | 2826.50 | 2842.10 | 0.00 | ORB-short ORB[2833.50,2872.50] vol=3.5x ATR=9.88 |
| Stop hit — per-position SL triggered | 2025-08-28 10:35:00 | 2836.38 | 2839.61 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 2813.25 | 2792.31 | 0.00 | ORB-long ORB[2768.75,2803.00] vol=2.3x ATR=7.85 |
| Stop hit — per-position SL triggered | 2025-09-03 09:35:00 | 2805.40 | 2794.90 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:35:00 | 2825.75 | 2842.04 | 0.00 | ORB-short ORB[2842.50,2864.00] vol=1.8x ATR=6.11 |
| Stop hit — per-position SL triggered | 2025-09-09 10:40:00 | 2831.86 | 2841.33 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-09-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:25:00 | 2897.50 | 2871.66 | 0.00 | ORB-long ORB[2835.25,2869.25] vol=1.7x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:55:00 | 2907.47 | 2877.83 | 0.00 | T1 1.5R @ 2907.47 |
| Stop hit — per-position SL triggered | 2025-09-10 11:25:00 | 2897.50 | 2885.42 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 10:20:00 | 2869.75 | 2874.26 | 0.00 | ORB-short ORB[2878.50,2895.50] vol=1.8x ATR=5.75 |
| Stop hit — per-position SL triggered | 2025-09-15 11:55:00 | 2875.50 | 2872.67 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:40:00 | 2941.00 | 2935.15 | 0.00 | ORB-long ORB[2920.50,2939.00] vol=3.0x ATR=4.82 |
| Stop hit — per-position SL triggered | 2025-09-19 11:05:00 | 2936.18 | 2935.77 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-09-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:20:00 | 2902.75 | 2916.13 | 0.00 | ORB-short ORB[2925.50,2950.00] vol=3.6x ATR=6.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:25:00 | 2893.00 | 2914.95 | 0.00 | T1 1.5R @ 2893.00 |
| Stop hit — per-position SL triggered | 2025-09-24 12:35:00 | 2902.75 | 2900.08 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-09-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 11:00:00 | 2776.50 | 2788.35 | 0.00 | ORB-short ORB[2784.00,2814.50] vol=3.8x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:20:00 | 2765.43 | 2781.50 | 0.00 | T1 1.5R @ 2765.43 |
| Target hit | 2025-09-30 15:05:00 | 2762.75 | 2761.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2025-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:40:00 | 2805.75 | 2810.75 | 0.00 | ORB-short ORB[2808.25,2844.25] vol=1.6x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:50:00 | 2795.70 | 2809.65 | 0.00 | T1 1.5R @ 2795.70 |
| Target hit | 2025-10-07 13:40:00 | 2792.50 | 2789.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2025-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:05:00 | 2852.75 | 2836.46 | 0.00 | ORB-long ORB[2807.50,2838.50] vol=2.9x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:00:00 | 2864.92 | 2845.10 | 0.00 | T1 1.5R @ 2864.92 |
| Stop hit — per-position SL triggered | 2025-10-15 14:50:00 | 2852.75 | 2852.03 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:25:00 | 2934.25 | 2904.18 | 0.00 | ORB-long ORB[2877.50,2909.00] vol=4.9x ATR=12.31 |
| Stop hit — per-position SL triggered | 2025-10-17 10:40:00 | 2921.94 | 2908.17 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:10:00 | 2777.00 | 2782.04 | 0.00 | ORB-short ORB[2778.50,2794.75] vol=2.8x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-10-28 11:20:00 | 2781.67 | 2782.15 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:10:00 | 2699.50 | 2675.30 | 0.00 | ORB-long ORB[2655.00,2695.25] vol=2.8x ATR=6.42 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 2693.08 | 2676.12 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 2733.50 | 2720.08 | 0.00 | ORB-long ORB[2697.50,2726.50] vol=1.6x ATR=6.72 |
| Stop hit — per-position SL triggered | 2025-11-04 10:00:00 | 2726.78 | 2722.88 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:05:00 | 2697.25 | 2674.12 | 0.00 | ORB-long ORB[2655.25,2690.75] vol=3.3x ATR=8.51 |
| Stop hit — per-position SL triggered | 2025-11-07 10:20:00 | 2688.74 | 2676.15 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 2714.00 | 2730.87 | 0.00 | ORB-short ORB[2725.50,2745.00] vol=2.0x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 11:15:00 | 2705.73 | 2723.56 | 0.00 | T1 1.5R @ 2705.73 |
| Stop hit — per-position SL triggered | 2025-11-14 11:30:00 | 2714.00 | 2719.47 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:05:00 | 2682.00 | 2684.61 | 0.00 | ORB-short ORB[2687.75,2706.75] vol=2.1x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-11-21 11:50:00 | 2685.51 | 2684.27 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:35:00 | 2719.50 | 2704.80 | 0.00 | ORB-long ORB[2698.50,2713.50] vol=2.7x ATR=6.08 |
| Stop hit — per-position SL triggered | 2025-11-24 11:05:00 | 2713.42 | 2709.40 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:45:00 | 2690.90 | 2684.65 | 0.00 | ORB-long ORB[2673.30,2685.60] vol=1.6x ATR=5.56 |
| Stop hit — per-position SL triggered | 2025-12-01 09:55:00 | 2685.34 | 2684.82 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 11:05:00 | 2585.60 | 2597.93 | 0.00 | ORB-short ORB[2594.10,2609.00] vol=1.6x ATR=5.96 |
| Stop hit — per-position SL triggered | 2025-12-04 11:35:00 | 2591.56 | 2595.62 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:55:00 | 2595.30 | 2587.78 | 0.00 | ORB-long ORB[2560.90,2593.50] vol=1.9x ATR=6.44 |
| Stop hit — per-position SL triggered | 2025-12-08 11:10:00 | 2588.86 | 2587.85 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:35:00 | 2608.60 | 2593.59 | 0.00 | ORB-long ORB[2559.00,2595.50] vol=1.8x ATR=8.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:45:00 | 2621.98 | 2603.57 | 0.00 | T1 1.5R @ 2621.98 |
| Stop hit — per-position SL triggered | 2025-12-10 09:50:00 | 2608.60 | 2606.90 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 09:30:00 | 2637.40 | 2627.14 | 0.00 | ORB-long ORB[2600.30,2636.70] vol=1.8x ATR=7.72 |
| Stop hit — per-position SL triggered | 2025-12-11 09:35:00 | 2629.68 | 2627.79 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:05:00 | 2620.20 | 2639.74 | 0.00 | ORB-short ORB[2649.20,2669.80] vol=2.2x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-12-15 11:20:00 | 2625.82 | 2637.18 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:50:00 | 2580.70 | 2584.50 | 0.00 | ORB-short ORB[2582.80,2608.00] vol=2.0x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:00:00 | 2573.40 | 2583.66 | 0.00 | T1 1.5R @ 2573.40 |
| Stop hit — per-position SL triggered | 2025-12-16 12:20:00 | 2580.70 | 2577.64 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:00:00 | 2570.30 | 2579.65 | 0.00 | ORB-short ORB[2581.30,2601.70] vol=1.8x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2576.37 | 2579.20 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:35:00 | 2697.00 | 2686.48 | 0.00 | ORB-long ORB[2665.20,2688.20] vol=1.6x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-12-23 09:40:00 | 2690.53 | 2687.08 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:40:00 | 2719.50 | 2715.72 | 0.00 | ORB-long ORB[2700.00,2717.70] vol=1.7x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-12-24 10:00:00 | 2714.93 | 2717.55 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:10:00 | 2666.90 | 2665.54 | 0.00 | ORB-long ORB[2639.90,2662.30] vol=2.1x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 12:10:00 | 2674.46 | 2665.97 | 0.00 | T1 1.5R @ 2674.46 |
| Stop hit — per-position SL triggered | 2025-12-31 15:00:00 | 2666.90 | 2669.78 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 2649.40 | 2649.71 | 0.00 | ORB-short ORB[2656.20,2682.00] vol=4.0x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:30:00 | 2641.96 | 2649.38 | 0.00 | T1 1.5R @ 2641.96 |
| Target hit | 2026-01-01 13:45:00 | 2647.00 | 2644.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2026-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 2606.00 | 2612.20 | 0.00 | ORB-short ORB[2610.50,2625.20] vol=2.7x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:50:00 | 2599.07 | 2611.22 | 0.00 | T1 1.5R @ 2599.07 |
| Target hit | 2026-01-08 14:45:00 | 2584.00 | 2582.66 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2026-01-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:20:00 | 2479.60 | 2488.89 | 0.00 | ORB-short ORB[2495.00,2512.50] vol=2.0x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:25:00 | 2469.89 | 2482.91 | 0.00 | T1 1.5R @ 2469.89 |
| Stop hit — per-position SL triggered | 2026-01-13 13:10:00 | 2479.60 | 2478.44 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:35:00 | 2535.00 | 2523.06 | 0.00 | ORB-long ORB[2487.10,2512.50] vol=6.1x ATR=8.14 |
| Stop hit — per-position SL triggered | 2026-01-14 11:10:00 | 2526.86 | 2525.55 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:50:00 | 2470.00 | 2502.59 | 0.00 | ORB-short ORB[2499.20,2533.60] vol=2.4x ATR=9.42 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 2479.42 | 2497.17 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:55:00 | 2531.00 | 2501.51 | 0.00 | ORB-long ORB[2484.00,2518.10] vol=3.4x ATR=8.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:05:00 | 2543.98 | 2515.99 | 0.00 | T1 1.5R @ 2543.98 |
| Stop hit — per-position SL triggered | 2026-02-01 11:50:00 | 2531.00 | 2522.90 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:05:00 | 2759.80 | 2756.81 | 0.00 | ORB-long ORB[2720.90,2754.70] vol=4.4x ATR=8.45 |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 2751.35 | 2756.81 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 2781.50 | 2762.34 | 0.00 | ORB-long ORB[2735.00,2761.00] vol=2.3x ATR=7.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 2792.39 | 2767.11 | 0.00 | T1 1.5R @ 2792.39 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 2781.50 | 2769.55 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 2864.80 | 2848.22 | 0.00 | ORB-long ORB[2823.20,2852.50] vol=1.7x ATR=6.36 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 2858.44 | 2849.36 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 2715.20 | 2711.86 | 0.00 | ORB-long ORB[2682.80,2711.30] vol=3.5x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:00:00 | 2725.76 | 2715.52 | 0.00 | T1 1.5R @ 2725.76 |
| Target hit | 2026-02-25 12:45:00 | 2722.50 | 2728.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2026-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:50:00 | 2442.10 | 2434.45 | 0.00 | ORB-long ORB[2408.40,2440.00] vol=1.5x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:30:00 | 2456.04 | 2437.58 | 0.00 | T1 1.5R @ 2456.04 |
| Target hit | 2026-03-12 14:25:00 | 2453.30 | 2453.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 2395.60 | 2382.89 | 0.00 | ORB-long ORB[2355.50,2382.00] vol=3.0x ATR=9.53 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 2386.07 | 2383.66 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:20:00 | 2299.00 | 2327.48 | 0.00 | ORB-short ORB[2329.30,2361.30] vol=1.8x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:40:00 | 2282.53 | 2314.42 | 0.00 | T1 1.5R @ 2282.53 |
| Target hit | 2026-03-23 14:10:00 | 2285.00 | 2283.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 83 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 2509.00 | 2523.01 | 0.00 | ORB-short ORB[2516.20,2542.00] vol=1.6x ATR=11.71 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 2520.71 | 2514.01 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 2771.20 | 2778.73 | 0.00 | ORB-short ORB[2773.00,2811.40] vol=1.8x ATR=8.08 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 2779.28 | 2777.32 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 2801.00 | 2780.50 | 0.00 | ORB-long ORB[2762.60,2794.90] vol=3.3x ATR=8.33 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 2792.67 | 2782.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:50:00 | 2273.90 | 2025-05-13 11:45:00 | 2266.90 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-14 10:35:00 | 2265.70 | 2025-05-14 12:10:00 | 2273.61 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-14 10:35:00 | 2265.70 | 2025-05-14 12:50:00 | 2265.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 09:55:00 | 2305.00 | 2025-05-15 10:10:00 | 2314.81 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-15 09:55:00 | 2305.00 | 2025-05-15 15:20:00 | 2353.00 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2025-05-16 10:40:00 | 2382.05 | 2025-05-16 10:55:00 | 2376.76 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-22 09:35:00 | 2428.10 | 2025-05-22 11:20:00 | 2419.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-23 11:05:00 | 2406.80 | 2025-05-23 12:10:00 | 2414.95 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-23 11:05:00 | 2406.80 | 2025-05-23 12:15:00 | 2406.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 10:15:00 | 2390.20 | 2025-05-27 10:35:00 | 2396.19 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-28 10:45:00 | 2422.50 | 2025-05-28 10:50:00 | 2429.18 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-05-28 10:45:00 | 2422.50 | 2025-05-28 12:30:00 | 2426.00 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-06-04 11:15:00 | 2367.55 | 2025-06-04 11:20:00 | 2373.38 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-05 10:00:00 | 2404.35 | 2025-06-05 10:35:00 | 2398.11 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-06 09:30:00 | 2447.05 | 2025-06-06 09:40:00 | 2462.27 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-06 09:30:00 | 2447.05 | 2025-06-06 15:20:00 | 2561.40 | TARGET_HIT | 0.50 | 4.67% |
| BUY | retest1 | 2025-06-11 10:40:00 | 2583.20 | 2025-06-11 11:05:00 | 2593.69 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-11 10:40:00 | 2583.20 | 2025-06-11 12:15:00 | 2583.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 10:55:00 | 2538.65 | 2025-06-12 11:05:00 | 2546.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-13 10:20:00 | 2501.50 | 2025-06-13 10:35:00 | 2492.34 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-19 11:05:00 | 2463.85 | 2025-06-19 12:15:00 | 2454.99 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-19 11:05:00 | 2463.85 | 2025-06-19 15:20:00 | 2441.70 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2025-06-20 10:30:00 | 2465.00 | 2025-06-20 11:30:00 | 2477.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-06-20 10:30:00 | 2465.00 | 2025-06-20 12:20:00 | 2465.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-23 11:15:00 | 2491.65 | 2025-06-23 11:20:00 | 2484.54 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-24 09:40:00 | 2522.50 | 2025-06-24 10:00:00 | 2534.90 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-24 09:40:00 | 2522.50 | 2025-06-24 13:30:00 | 2532.55 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-06-25 09:40:00 | 2555.60 | 2025-06-25 10:05:00 | 2549.27 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-26 09:30:00 | 2561.45 | 2025-06-26 09:40:00 | 2555.06 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-27 11:15:00 | 2584.15 | 2025-06-27 12:05:00 | 2577.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-30 10:35:00 | 2593.25 | 2025-06-30 10:50:00 | 2586.48 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-01 10:45:00 | 2579.50 | 2025-07-01 11:00:00 | 2569.81 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-01 10:45:00 | 2579.50 | 2025-07-01 15:20:00 | 2570.50 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-03 09:40:00 | 2551.25 | 2025-07-03 11:50:00 | 2544.71 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-07 11:00:00 | 2551.50 | 2025-07-07 11:05:00 | 2543.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-11 11:00:00 | 2567.75 | 2025-07-11 11:10:00 | 2574.33 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-15 10:40:00 | 2597.25 | 2025-07-15 10:55:00 | 2607.43 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-15 10:40:00 | 2597.25 | 2025-07-15 12:40:00 | 2610.75 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2025-07-16 09:40:00 | 2665.00 | 2025-07-16 10:00:00 | 2655.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-21 10:40:00 | 2806.25 | 2025-07-21 10:50:00 | 2817.23 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-21 10:40:00 | 2806.25 | 2025-07-21 10:55:00 | 2806.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 10:55:00 | 2781.75 | 2025-07-22 11:15:00 | 2788.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-24 11:10:00 | 2780.00 | 2025-07-24 15:20:00 | 2780.25 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest1 | 2025-07-29 09:40:00 | 2784.50 | 2025-07-29 09:45:00 | 2797.94 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-29 09:40:00 | 2784.50 | 2025-07-29 14:00:00 | 2806.25 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-08-07 11:05:00 | 2817.00 | 2025-08-07 11:20:00 | 2807.58 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-07 11:05:00 | 2817.00 | 2025-08-07 14:10:00 | 2814.00 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-08-11 10:20:00 | 2762.50 | 2025-08-11 11:05:00 | 2749.91 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-11 10:20:00 | 2762.50 | 2025-08-11 12:50:00 | 2762.00 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2025-08-12 11:15:00 | 2747.50 | 2025-08-12 14:15:00 | 2737.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-12 11:15:00 | 2747.50 | 2025-08-12 15:20:00 | 2732.00 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-08-14 10:10:00 | 2722.50 | 2025-08-14 10:20:00 | 2728.86 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-18 09:40:00 | 2823.75 | 2025-08-18 10:05:00 | 2838.15 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-18 09:40:00 | 2823.75 | 2025-08-18 15:20:00 | 2859.00 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2025-08-20 11:00:00 | 2825.00 | 2025-08-20 11:10:00 | 2830.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-25 09:50:00 | 2946.75 | 2025-08-25 09:55:00 | 2940.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-26 09:50:00 | 2913.75 | 2025-08-26 09:55:00 | 2921.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-28 10:20:00 | 2826.50 | 2025-08-28 10:35:00 | 2836.38 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-03 09:30:00 | 2813.25 | 2025-09-03 09:35:00 | 2805.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-09 10:35:00 | 2825.75 | 2025-09-09 10:40:00 | 2831.86 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-10 10:25:00 | 2897.50 | 2025-09-10 10:55:00 | 2907.47 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-10 10:25:00 | 2897.50 | 2025-09-10 11:25:00 | 2897.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-15 10:20:00 | 2869.75 | 2025-09-15 11:55:00 | 2875.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-19 10:40:00 | 2941.00 | 2025-09-19 11:05:00 | 2936.18 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-24 10:20:00 | 2902.75 | 2025-09-24 10:25:00 | 2893.00 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-24 10:20:00 | 2902.75 | 2025-09-24 12:35:00 | 2902.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 11:00:00 | 2776.50 | 2025-09-30 11:20:00 | 2765.43 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-30 11:00:00 | 2776.50 | 2025-09-30 15:05:00 | 2762.75 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2025-10-07 10:40:00 | 2805.75 | 2025-10-07 10:50:00 | 2795.70 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-07 10:40:00 | 2805.75 | 2025-10-07 13:40:00 | 2792.50 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-15 10:05:00 | 2852.75 | 2025-10-15 12:00:00 | 2864.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-15 10:05:00 | 2852.75 | 2025-10-15 14:50:00 | 2852.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 10:25:00 | 2934.25 | 2025-10-17 10:40:00 | 2921.94 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-10-28 11:10:00 | 2777.00 | 2025-10-28 11:20:00 | 2781.67 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-03 11:10:00 | 2699.50 | 2025-11-03 11:15:00 | 2693.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-04 09:45:00 | 2733.50 | 2025-11-04 10:00:00 | 2726.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-07 10:05:00 | 2697.25 | 2025-11-07 10:20:00 | 2688.74 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-14 10:50:00 | 2714.00 | 2025-11-14 11:15:00 | 2705.73 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-14 10:50:00 | 2714.00 | 2025-11-14 11:30:00 | 2714.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 11:05:00 | 2682.00 | 2025-11-21 11:50:00 | 2685.51 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-11-24 10:35:00 | 2719.50 | 2025-11-24 11:05:00 | 2713.42 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-01 09:45:00 | 2690.90 | 2025-12-01 09:55:00 | 2685.34 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-04 11:05:00 | 2585.60 | 2025-12-04 11:35:00 | 2591.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-08 10:55:00 | 2595.30 | 2025-12-08 11:10:00 | 2588.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-10 09:35:00 | 2608.60 | 2025-12-10 09:45:00 | 2621.98 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-10 09:35:00 | 2608.60 | 2025-12-10 09:50:00 | 2608.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 09:30:00 | 2637.40 | 2025-12-11 09:35:00 | 2629.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-15 11:05:00 | 2620.20 | 2025-12-15 11:20:00 | 2625.82 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-16 10:50:00 | 2580.70 | 2025-12-16 11:00:00 | 2573.40 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-16 10:50:00 | 2580.70 | 2025-12-16 12:20:00 | 2580.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-17 11:00:00 | 2570.30 | 2025-12-17 11:15:00 | 2576.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-23 09:35:00 | 2697.00 | 2025-12-23 09:40:00 | 2690.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-24 09:40:00 | 2719.50 | 2025-12-24 10:00:00 | 2714.93 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-31 11:10:00 | 2666.90 | 2025-12-31 12:10:00 | 2674.46 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-31 11:10:00 | 2666.90 | 2025-12-31 15:00:00 | 2666.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 11:10:00 | 2649.40 | 2026-01-01 11:30:00 | 2641.96 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-01-01 11:10:00 | 2649.40 | 2026-01-01 13:45:00 | 2647.00 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-01-08 10:45:00 | 2606.00 | 2026-01-08 10:50:00 | 2599.07 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-01-08 10:45:00 | 2606.00 | 2026-01-08 14:45:00 | 2584.00 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2026-01-13 10:20:00 | 2479.60 | 2026-01-13 11:25:00 | 2469.89 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-13 10:20:00 | 2479.60 | 2026-01-13 13:10:00 | 2479.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-14 10:35:00 | 2535.00 | 2026-01-14 11:10:00 | 2526.86 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-21 10:50:00 | 2470.00 | 2026-01-21 11:00:00 | 2479.42 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-01 10:55:00 | 2531.00 | 2026-02-01 11:05:00 | 2543.98 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-01 10:55:00 | 2531.00 | 2026-02-01 11:50:00 | 2531.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-05 10:05:00 | 2759.80 | 2026-02-05 10:15:00 | 2751.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-10 09:40:00 | 2781.50 | 2026-02-10 09:45:00 | 2792.39 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-10 09:40:00 | 2781.50 | 2026-02-10 09:50:00 | 2781.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:25:00 | 2864.80 | 2026-02-17 10:30:00 | 2858.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 10:45:00 | 2715.20 | 2026-02-25 11:00:00 | 2725.76 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-25 10:45:00 | 2715.20 | 2026-02-25 12:45:00 | 2722.50 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-03-12 10:50:00 | 2442.10 | 2026-03-12 11:30:00 | 2456.04 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-12 10:50:00 | 2442.10 | 2026-03-12 14:25:00 | 2453.30 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-17 10:25:00 | 2395.60 | 2026-03-17 10:35:00 | 2386.07 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-23 10:20:00 | 2299.00 | 2026-03-23 11:40:00 | 2282.53 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-03-23 10:20:00 | 2299.00 | 2026-03-23 14:10:00 | 2285.00 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-09 09:30:00 | 2509.00 | 2026-04-09 10:15:00 | 2520.71 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-22 10:40:00 | 2771.20 | 2026-04-22 11:05:00 | 2779.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 10:45:00 | 2801.00 | 2026-04-29 11:00:00 | 2792.67 | STOP_HIT | 1.00 | -0.30% |
