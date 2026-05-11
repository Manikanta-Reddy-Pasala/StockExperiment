# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2340.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 15 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 62
- **Target hits / Stop hits / Partials:** 15 / 62 / 31
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 15.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 19 | 39.6% | 6 | 29 | 13 | 0.12% | 6.0% |
| BUY @ 2nd Alert (retest1) | 48 | 19 | 39.6% | 6 | 29 | 13 | 0.12% | 6.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 27 | 45.0% | 9 | 33 | 18 | 0.16% | 9.5% |
| SELL @ 2nd Alert (retest1) | 60 | 27 | 45.0% | 9 | 33 | 18 | 0.16% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 46 | 42.6% | 15 | 62 | 31 | 0.14% | 15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 2400.90 | 2398.71 | 0.00 | ORB-long ORB[2370.00,2399.00] vol=6.3x ATR=10.34 |
| Stop hit — per-position SL triggered | 2025-05-30 10:50:00 | 2390.56 | 2401.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 2473.80 | 2459.33 | 0.00 | ORB-long ORB[2436.10,2465.30] vol=1.7x ATR=8.43 |
| Stop hit — per-position SL triggered | 2025-06-03 09:45:00 | 2465.37 | 2460.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:15:00 | 2438.10 | 2466.87 | 0.00 | ORB-short ORB[2472.30,2492.20] vol=2.3x ATR=7.73 |
| Stop hit — per-position SL triggered | 2025-06-04 10:30:00 | 2445.83 | 2464.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:50:00 | 2414.00 | 2425.46 | 0.00 | ORB-short ORB[2423.80,2444.40] vol=3.0x ATR=7.18 |
| Stop hit — per-position SL triggered | 2025-06-06 09:55:00 | 2421.18 | 2424.90 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 2425.00 | 2436.26 | 0.00 | ORB-short ORB[2432.10,2449.90] vol=2.1x ATR=6.59 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 2431.59 | 2435.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:15:00 | 2403.00 | 2411.51 | 0.00 | ORB-short ORB[2410.00,2438.80] vol=1.6x ATR=4.48 |
| Stop hit — per-position SL triggered | 2025-06-10 13:00:00 | 2407.48 | 2409.88 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:15:00 | 2366.10 | 2383.07 | 0.00 | ORB-short ORB[2381.00,2409.90] vol=2.4x ATR=5.94 |
| Stop hit — per-position SL triggered | 2025-06-12 11:50:00 | 2372.04 | 2381.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:35:00 | 2499.50 | 2476.46 | 0.00 | ORB-long ORB[2450.00,2483.90] vol=2.4x ATR=9.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:50:00 | 2514.07 | 2488.88 | 0.00 | T1 1.5R @ 2514.07 |
| Stop hit — per-position SL triggered | 2025-06-24 11:55:00 | 2499.50 | 2494.60 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 09:35:00 | 2467.00 | 2482.11 | 0.00 | ORB-short ORB[2471.10,2499.00] vol=2.0x ATR=9.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 09:55:00 | 2452.91 | 2475.68 | 0.00 | T1 1.5R @ 2452.91 |
| Stop hit — per-position SL triggered | 2025-06-30 10:20:00 | 2467.00 | 2477.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:35:00 | 2488.10 | 2501.68 | 0.00 | ORB-short ORB[2497.50,2534.90] vol=1.9x ATR=10.15 |
| Target hit | 2025-07-01 15:20:00 | 2486.90 | 2492.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:35:00 | 2465.10 | 2477.51 | 0.00 | ORB-short ORB[2477.00,2493.60] vol=2.7x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:45:00 | 2455.51 | 2470.09 | 0.00 | T1 1.5R @ 2455.51 |
| Target hit | 2025-07-02 15:20:00 | 2451.80 | 2447.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:45:00 | 2438.00 | 2451.83 | 0.00 | ORB-short ORB[2444.60,2467.80] vol=2.1x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:05:00 | 2428.80 | 2441.36 | 0.00 | T1 1.5R @ 2428.80 |
| Stop hit — per-position SL triggered | 2025-07-08 10:10:00 | 2438.00 | 2440.39 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:50:00 | 2407.00 | 2415.99 | 0.00 | ORB-short ORB[2410.00,2443.00] vol=1.8x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:00:00 | 2396.84 | 2408.45 | 0.00 | T1 1.5R @ 2396.84 |
| Stop hit — per-position SL triggered | 2025-07-10 10:05:00 | 2407.00 | 2408.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 2375.00 | 2387.99 | 0.00 | ORB-short ORB[2387.50,2410.00] vol=1.7x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 2378.81 | 2386.75 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:50:00 | 2396.90 | 2399.37 | 0.00 | ORB-short ORB[2410.40,2429.90] vol=7.3x ATR=7.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:05:00 | 2384.97 | 2396.47 | 0.00 | T1 1.5R @ 2384.97 |
| Target hit | 2025-07-16 15:20:00 | 2388.20 | 2392.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:00:00 | 2408.50 | 2396.80 | 0.00 | ORB-long ORB[2374.90,2405.90] vol=1.7x ATR=6.53 |
| Stop hit — per-position SL triggered | 2025-07-17 10:10:00 | 2401.97 | 2397.14 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 2377.50 | 2390.57 | 0.00 | ORB-short ORB[2386.30,2400.00] vol=1.9x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 2382.69 | 2387.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:05:00 | 2356.60 | 2373.37 | 0.00 | ORB-short ORB[2366.30,2396.90] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-07-22 11:10:00 | 2361.38 | 2373.17 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:15:00 | 2389.50 | 2379.12 | 0.00 | ORB-long ORB[2352.30,2376.10] vol=2.1x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-07-24 11:20:00 | 2382.84 | 2383.31 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 11:05:00 | 2335.50 | 2312.68 | 0.00 | ORB-long ORB[2300.10,2323.70] vol=3.2x ATR=7.80 |
| Stop hit — per-position SL triggered | 2025-08-04 11:10:00 | 2327.70 | 2318.03 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 2468.90 | 2478.15 | 0.00 | ORB-short ORB[2476.00,2509.90] vol=3.2x ATR=6.86 |
| Stop hit — per-position SL triggered | 2025-08-19 13:20:00 | 2475.76 | 2473.94 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 2454.90 | 2465.10 | 0.00 | ORB-short ORB[2462.00,2493.00] vol=2.5x ATR=6.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 2444.96 | 2459.80 | 0.00 | T1 1.5R @ 2444.96 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 2454.90 | 2459.24 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:35:00 | 2425.40 | 2438.67 | 0.00 | ORB-short ORB[2431.30,2464.90] vol=1.5x ATR=6.63 |
| Stop hit — per-position SL triggered | 2025-08-29 09:40:00 | 2432.03 | 2437.72 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:50:00 | 2446.40 | 2435.06 | 0.00 | ORB-long ORB[2422.50,2442.20] vol=1.5x ATR=5.79 |
| Stop hit — per-position SL triggered | 2025-09-01 10:00:00 | 2440.61 | 2435.96 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 2472.10 | 2463.44 | 0.00 | ORB-long ORB[2446.10,2470.00] vol=3.6x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:55:00 | 2482.30 | 2473.71 | 0.00 | T1 1.5R @ 2482.30 |
| Target hit | 2025-09-02 10:40:00 | 2478.10 | 2479.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 2507.90 | 2489.41 | 0.00 | ORB-long ORB[2462.60,2493.30] vol=3.9x ATR=8.50 |
| Stop hit — per-position SL triggered | 2025-09-03 09:50:00 | 2499.40 | 2490.97 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:15:00 | 2626.40 | 2602.95 | 0.00 | ORB-long ORB[2581.00,2601.70] vol=3.5x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:30:00 | 2641.63 | 2614.13 | 0.00 | T1 1.5R @ 2641.63 |
| Stop hit — per-position SL triggered | 2025-09-15 10:50:00 | 2626.40 | 2616.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:00:00 | 2615.00 | 2630.46 | 0.00 | ORB-short ORB[2618.00,2650.00] vol=2.5x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 13:05:00 | 2606.30 | 2625.44 | 0.00 | T1 1.5R @ 2606.30 |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 2615.00 | 2622.92 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:30:00 | 2639.60 | 2628.60 | 0.00 | ORB-long ORB[2609.00,2638.90] vol=1.5x ATR=9.44 |
| Stop hit — per-position SL triggered | 2025-09-22 09:40:00 | 2630.16 | 2629.50 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:20:00 | 2605.10 | 2609.15 | 0.00 | ORB-short ORB[2608.50,2624.00] vol=1.7x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-09-23 10:25:00 | 2611.17 | 2609.45 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:00:00 | 2526.00 | 2491.83 | 0.00 | ORB-long ORB[2468.40,2494.90] vol=1.6x ATR=10.74 |
| Stop hit — per-position SL triggered | 2025-09-29 10:45:00 | 2515.26 | 2504.11 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:45:00 | 2489.30 | 2503.39 | 0.00 | ORB-short ORB[2508.00,2522.00] vol=1.7x ATR=6.74 |
| Stop hit — per-position SL triggered | 2025-09-30 10:50:00 | 2496.04 | 2502.70 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 2477.90 | 2488.99 | 0.00 | ORB-short ORB[2480.30,2504.50] vol=2.8x ATR=6.32 |
| Stop hit — per-position SL triggered | 2025-10-01 11:10:00 | 2484.22 | 2488.75 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:00:00 | 2481.20 | 2494.69 | 0.00 | ORB-short ORB[2490.40,2516.40] vol=1.8x ATR=5.48 |
| Stop hit — per-position SL triggered | 2025-10-06 11:50:00 | 2486.68 | 2493.60 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:50:00 | 2551.10 | 2569.09 | 0.00 | ORB-short ORB[2570.60,2591.90] vol=1.7x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:25:00 | 2542.71 | 2564.87 | 0.00 | T1 1.5R @ 2542.71 |
| Stop hit — per-position SL triggered | 2025-10-09 13:50:00 | 2551.10 | 2557.95 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 10:05:00 | 2543.40 | 2552.30 | 0.00 | ORB-short ORB[2544.30,2568.40] vol=1.6x ATR=7.98 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 2551.38 | 2552.77 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:10:00 | 2493.00 | 2500.37 | 0.00 | ORB-short ORB[2493.10,2519.60] vol=2.5x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 2482.36 | 2497.17 | 0.00 | T1 1.5R @ 2482.36 |
| Target hit | 2025-10-15 13:45:00 | 2486.90 | 2486.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2025-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:55:00 | 2503.10 | 2471.04 | 0.00 | ORB-long ORB[2453.40,2490.20] vol=2.7x ATR=7.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:15:00 | 2515.08 | 2476.73 | 0.00 | T1 1.5R @ 2515.08 |
| Stop hit — per-position SL triggered | 2025-11-10 13:40:00 | 2503.10 | 2491.24 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 2551.70 | 2561.20 | 0.00 | ORB-short ORB[2553.30,2571.00] vol=1.5x ATR=7.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:40:00 | 2540.20 | 2556.34 | 0.00 | T1 1.5R @ 2540.20 |
| Stop hit — per-position SL triggered | 2025-11-12 11:05:00 | 2551.70 | 2553.80 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:40:00 | 2523.90 | 2516.60 | 0.00 | ORB-long ORB[2505.90,2519.40] vol=2.1x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 10:50:00 | 2531.40 | 2517.35 | 0.00 | T1 1.5R @ 2531.40 |
| Target hit | 2025-11-14 15:20:00 | 2621.00 | 2595.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-11-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:55:00 | 2585.00 | 2597.46 | 0.00 | ORB-short ORB[2593.00,2614.60] vol=1.8x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:10:00 | 2575.17 | 2593.36 | 0.00 | T1 1.5R @ 2575.17 |
| Target hit | 2025-11-18 11:55:00 | 2580.90 | 2580.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2025-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:30:00 | 2499.00 | 2504.21 | 0.00 | ORB-short ORB[2500.20,2525.10] vol=1.6x ATR=7.01 |
| Stop hit — per-position SL triggered | 2025-11-25 12:05:00 | 2506.01 | 2501.06 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:35:00 | 2525.20 | 2532.14 | 0.00 | ORB-short ORB[2530.10,2554.70] vol=2.1x ATR=6.75 |
| Stop hit — per-position SL triggered | 2025-11-27 09:50:00 | 2531.95 | 2532.98 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 2557.50 | 2546.71 | 0.00 | ORB-long ORB[2530.70,2555.00] vol=1.6x ATR=8.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:40:00 | 2570.00 | 2556.49 | 0.00 | T1 1.5R @ 2570.00 |
| Target hit | 2025-12-10 10:15:00 | 2574.20 | 2574.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:15:00 | 2527.80 | 2510.88 | 0.00 | ORB-long ORB[2492.00,2514.10] vol=2.3x ATR=6.22 |
| Stop hit — per-position SL triggered | 2025-12-11 11:50:00 | 2521.58 | 2513.08 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:25:00 | 2570.00 | 2552.36 | 0.00 | ORB-long ORB[2533.00,2551.90] vol=5.0x ATR=7.43 |
| Stop hit — per-position SL triggered | 2025-12-12 10:40:00 | 2562.57 | 2556.63 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:20:00 | 2596.30 | 2572.84 | 0.00 | ORB-long ORB[2550.00,2578.30] vol=2.0x ATR=7.76 |
| Stop hit — per-position SL triggered | 2025-12-15 11:45:00 | 2588.54 | 2586.66 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:50:00 | 2562.30 | 2573.44 | 0.00 | ORB-short ORB[2566.10,2600.00] vol=1.6x ATR=5.55 |
| Stop hit — per-position SL triggered | 2025-12-23 11:55:00 | 2567.85 | 2571.53 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 2564.80 | 2551.14 | 0.00 | ORB-long ORB[2531.70,2550.40] vol=4.9x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-12-31 11:10:00 | 2559.58 | 2551.61 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 2590.10 | 2606.18 | 0.00 | ORB-short ORB[2601.30,2630.00] vol=3.1x ATR=4.94 |
| Stop hit — per-position SL triggered | 2026-01-01 11:30:00 | 2595.04 | 2605.65 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:45:00 | 2661.00 | 2669.36 | 0.00 | ORB-short ORB[2665.10,2699.00] vol=15.0x ATR=10.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:00:00 | 2645.85 | 2668.66 | 0.00 | T1 1.5R @ 2645.85 |
| Stop hit — per-position SL triggered | 2026-01-05 11:55:00 | 2661.00 | 2667.55 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:40:00 | 2591.80 | 2608.90 | 0.00 | ORB-short ORB[2610.00,2636.60] vol=2.7x ATR=10.87 |
| Stop hit — per-position SL triggered | 2026-01-07 10:05:00 | 2602.67 | 2603.80 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 2576.90 | 2593.84 | 0.00 | ORB-short ORB[2589.50,2606.90] vol=2.3x ATR=7.28 |
| Stop hit — per-position SL triggered | 2026-01-08 11:10:00 | 2584.18 | 2593.39 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 11:15:00 | 2523.60 | 2519.20 | 0.00 | ORB-long ORB[2487.60,2519.90] vol=1.6x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:40:00 | 2534.02 | 2520.26 | 0.00 | T1 1.5R @ 2534.02 |
| Stop hit — per-position SL triggered | 2026-01-16 11:50:00 | 2523.60 | 2520.36 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:30:00 | 2434.80 | 2419.76 | 0.00 | ORB-long ORB[2390.80,2425.50] vol=1.6x ATR=6.81 |
| Stop hit — per-position SL triggered | 2026-01-23 10:45:00 | 2427.99 | 2421.81 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:35:00 | 2327.70 | 2338.64 | 0.00 | ORB-short ORB[2335.00,2351.00] vol=2.2x ATR=10.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:20:00 | 2311.33 | 2330.92 | 0.00 | T1 1.5R @ 2311.33 |
| Target hit | 2026-01-29 15:20:00 | 2291.60 | 2297.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 2323.00 | 2294.66 | 0.00 | ORB-long ORB[2271.90,2298.90] vol=1.6x ATR=11.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:35:00 | 2339.86 | 2325.78 | 0.00 | T1 1.5R @ 2339.86 |
| Target hit | 2026-01-30 15:20:00 | 2380.80 | 2360.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2026-02-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 09:40:00 | 2393.70 | 2400.92 | 0.00 | ORB-short ORB[2394.20,2419.00] vol=2.2x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:40:00 | 2379.76 | 2392.83 | 0.00 | T1 1.5R @ 2379.76 |
| Stop hit — per-position SL triggered | 2026-02-01 12:10:00 | 2393.70 | 2391.50 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:10:00 | 2430.00 | 2408.73 | 0.00 | ORB-long ORB[2387.40,2413.10] vol=4.6x ATR=9.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:40:00 | 2444.02 | 2418.38 | 0.00 | T1 1.5R @ 2444.02 |
| Target hit | 2026-02-04 15:20:00 | 2441.10 | 2444.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2451.90 | 2434.56 | 0.00 | ORB-long ORB[2412.40,2438.90] vol=5.6x ATR=6.88 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 2445.02 | 2436.77 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 2344.50 | 2358.97 | 0.00 | ORB-short ORB[2351.10,2373.90] vol=3.2x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 2349.26 | 2356.36 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 2374.90 | 2370.05 | 0.00 | ORB-long ORB[2360.00,2374.40] vol=2.1x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 2384.17 | 2375.38 | 0.00 | T1 1.5R @ 2384.17 |
| Stop hit — per-position SL triggered | 2026-02-18 10:05:00 | 2374.90 | 2374.60 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 2304.30 | 2312.99 | 0.00 | ORB-short ORB[2308.20,2329.10] vol=1.5x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 2295.30 | 2308.53 | 0.00 | T1 1.5R @ 2295.30 |
| Stop hit — per-position SL triggered | 2026-02-20 10:30:00 | 2304.30 | 2300.92 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 2202.40 | 2218.22 | 0.00 | ORB-short ORB[2215.70,2239.90] vol=3.4x ATR=8.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:25:00 | 2189.43 | 2213.25 | 0.00 | T1 1.5R @ 2189.43 |
| Target hit | 2026-03-05 15:05:00 | 2163.30 | 2158.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2122.00 | 2132.42 | 0.00 | ORB-short ORB[2129.00,2150.00] vol=1.9x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:20:00 | 2113.10 | 2128.39 | 0.00 | T1 1.5R @ 2113.10 |
| Target hit | 2026-03-06 15:20:00 | 2094.00 | 2114.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2026-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:55:00 | 2113.60 | 2080.59 | 0.00 | ORB-long ORB[2068.40,2095.00] vol=2.6x ATR=8.13 |
| Stop hit — per-position SL triggered | 2026-03-10 11:00:00 | 2105.47 | 2081.21 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 2334.90 | 2261.90 | 0.00 | ORB-long ORB[2069.10,2103.00] vol=5.9x ATR=31.42 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 2303.48 | 2282.85 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-03-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:55:00 | 2197.40 | 2176.96 | 0.00 | ORB-long ORB[2153.00,2179.90] vol=1.6x ATR=10.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 2212.75 | 2190.81 | 0.00 | T1 1.5R @ 2212.75 |
| Stop hit — per-position SL triggered | 2026-03-18 10:20:00 | 2197.40 | 2192.79 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 2408.90 | 2419.31 | 0.00 | ORB-short ORB[2410.30,2434.50] vol=5.7x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:10:00 | 2400.62 | 2415.44 | 0.00 | T1 1.5R @ 2400.62 |
| Target hit | 2026-04-15 15:20:00 | 2381.00 | 2400.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 2425.00 | 2406.37 | 0.00 | ORB-long ORB[2383.10,2409.60] vol=5.2x ATR=8.73 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 2416.27 | 2407.74 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 2386.30 | 2377.07 | 0.00 | ORB-long ORB[2367.00,2382.70] vol=2.6x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:10:00 | 2393.09 | 2386.08 | 0.00 | T1 1.5R @ 2393.09 |
| Target hit | 2026-04-21 15:10:00 | 2387.10 | 2395.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 2324.80 | 2303.76 | 0.00 | ORB-long ORB[2290.20,2306.00] vol=1.7x ATR=7.56 |
| Stop hit — per-position SL triggered | 2026-04-29 10:25:00 | 2317.24 | 2306.08 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 2320.00 | 2310.61 | 0.00 | ORB-long ORB[2292.00,2311.90] vol=2.8x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 2329.87 | 2318.04 | 0.00 | T1 1.5R @ 2329.87 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 2320.00 | 2319.44 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-05-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:25:00 | 2304.00 | 2312.67 | 0.00 | ORB-short ORB[2307.00,2339.90] vol=2.0x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-05-05 10:30:00 | 2308.99 | 2312.00 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 2379.70 | 2364.16 | 0.00 | ORB-long ORB[2340.00,2365.90] vol=3.9x ATR=10.26 |
| Stop hit — per-position SL triggered | 2026-05-06 09:40:00 | 2369.44 | 2364.88 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 2374.70 | 2365.93 | 0.00 | ORB-long ORB[2350.40,2368.40] vol=3.9x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 2367.79 | 2365.87 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 2370.20 | 2351.36 | 0.00 | ORB-long ORB[2331.10,2354.00] vol=2.4x ATR=7.81 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 2362.39 | 2354.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-30 09:30:00 | 2400.90 | 2025-05-30 10:50:00 | 2390.56 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-03 09:35:00 | 2473.80 | 2025-06-03 09:45:00 | 2465.37 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-04 10:15:00 | 2438.10 | 2025-06-04 10:30:00 | 2445.83 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-06 09:50:00 | 2414.00 | 2025-06-06 09:55:00 | 2421.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-09 09:35:00 | 2425.00 | 2025-06-09 09:45:00 | 2431.59 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-10 11:15:00 | 2403.00 | 2025-06-10 13:00:00 | 2407.48 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-06-12 11:15:00 | 2366.10 | 2025-06-12 11:50:00 | 2372.04 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-24 10:35:00 | 2499.50 | 2025-06-24 10:50:00 | 2514.07 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-24 10:35:00 | 2499.50 | 2025-06-24 11:55:00 | 2499.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-30 09:35:00 | 2467.00 | 2025-06-30 09:55:00 | 2452.91 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-06-30 09:35:00 | 2467.00 | 2025-06-30 10:20:00 | 2467.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 09:35:00 | 2488.10 | 2025-07-01 15:20:00 | 2486.90 | TARGET_HIT | 1.00 | 0.05% |
| SELL | retest1 | 2025-07-02 09:35:00 | 2465.10 | 2025-07-02 09:45:00 | 2455.51 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-02 09:35:00 | 2465.10 | 2025-07-02 15:20:00 | 2451.80 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-07-08 09:45:00 | 2438.00 | 2025-07-08 10:05:00 | 2428.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-08 09:45:00 | 2438.00 | 2025-07-08 10:10:00 | 2438.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 09:50:00 | 2407.00 | 2025-07-10 10:00:00 | 2396.84 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-10 09:50:00 | 2407.00 | 2025-07-10 10:05:00 | 2407.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:45:00 | 2375.00 | 2025-07-11 11:10:00 | 2378.81 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-16 10:50:00 | 2396.90 | 2025-07-16 11:05:00 | 2384.97 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-16 10:50:00 | 2396.90 | 2025-07-16 15:20:00 | 2388.20 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-17 10:00:00 | 2408.50 | 2025-07-17 10:10:00 | 2401.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-18 10:15:00 | 2377.50 | 2025-07-18 10:50:00 | 2382.69 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-22 11:05:00 | 2356.60 | 2025-07-22 11:10:00 | 2361.38 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-24 10:15:00 | 2389.50 | 2025-07-24 11:20:00 | 2382.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-04 11:05:00 | 2335.50 | 2025-08-04 11:10:00 | 2327.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-19 11:15:00 | 2468.90 | 2025-08-19 13:20:00 | 2475.76 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-26 09:30:00 | 2454.90 | 2025-08-26 09:35:00 | 2444.96 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-26 09:30:00 | 2454.90 | 2025-08-26 09:45:00 | 2454.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:35:00 | 2425.40 | 2025-08-29 09:40:00 | 2432.03 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-01 09:50:00 | 2446.40 | 2025-09-01 10:00:00 | 2440.61 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-02 09:45:00 | 2472.10 | 2025-09-02 09:55:00 | 2482.30 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-02 09:45:00 | 2472.10 | 2025-09-02 10:40:00 | 2478.10 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-03 09:45:00 | 2507.90 | 2025-09-03 09:50:00 | 2499.40 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-15 10:15:00 | 2626.40 | 2025-09-15 10:30:00 | 2641.63 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-09-15 10:15:00 | 2626.40 | 2025-09-15 10:50:00 | 2626.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 11:00:00 | 2615.00 | 2025-09-16 13:05:00 | 2606.30 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-16 11:00:00 | 2615.00 | 2025-09-16 13:15:00 | 2615.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:30:00 | 2639.60 | 2025-09-22 09:40:00 | 2630.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-23 10:20:00 | 2605.10 | 2025-09-23 10:25:00 | 2611.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-29 10:00:00 | 2526.00 | 2025-09-29 10:45:00 | 2515.26 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-09-30 10:45:00 | 2489.30 | 2025-09-30 10:50:00 | 2496.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-01 11:00:00 | 2477.90 | 2025-10-01 11:10:00 | 2484.22 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-06 11:00:00 | 2481.20 | 2025-10-06 11:50:00 | 2486.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-09 10:50:00 | 2551.10 | 2025-10-09 11:25:00 | 2542.71 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-09 10:50:00 | 2551.10 | 2025-10-09 13:50:00 | 2551.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-10 10:05:00 | 2543.40 | 2025-10-10 10:15:00 | 2551.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-15 10:10:00 | 2493.00 | 2025-10-15 10:15:00 | 2482.36 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-15 10:10:00 | 2493.00 | 2025-10-15 13:45:00 | 2486.90 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-10 10:55:00 | 2503.10 | 2025-11-10 11:15:00 | 2515.08 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-11-10 10:55:00 | 2503.10 | 2025-11-10 13:40:00 | 2503.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 10:00:00 | 2551.70 | 2025-11-12 10:40:00 | 2540.20 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-12 10:00:00 | 2551.70 | 2025-11-12 11:05:00 | 2551.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 10:40:00 | 2523.90 | 2025-11-14 10:50:00 | 2531.40 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-11-14 10:40:00 | 2523.90 | 2025-11-14 15:20:00 | 2621.00 | TARGET_HIT | 0.50 | 3.85% |
| SELL | retest1 | 2025-11-18 09:55:00 | 2585.00 | 2025-11-18 10:10:00 | 2575.17 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-18 09:55:00 | 2585.00 | 2025-11-18 11:55:00 | 2580.90 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-11-25 09:30:00 | 2499.00 | 2025-11-25 12:05:00 | 2506.01 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-27 09:35:00 | 2525.20 | 2025-11-27 09:50:00 | 2531.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-10 09:30:00 | 2557.50 | 2025-12-10 09:40:00 | 2570.00 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-10 09:30:00 | 2557.50 | 2025-12-10 10:15:00 | 2574.20 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2025-12-11 11:15:00 | 2527.80 | 2025-12-11 11:50:00 | 2521.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-12 10:25:00 | 2570.00 | 2025-12-12 10:40:00 | 2562.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-15 10:20:00 | 2596.30 | 2025-12-15 11:45:00 | 2588.54 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-23 10:50:00 | 2562.30 | 2025-12-23 11:55:00 | 2567.85 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-31 10:50:00 | 2564.80 | 2025-12-31 11:10:00 | 2559.58 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-01 11:15:00 | 2590.10 | 2026-01-01 11:30:00 | 2595.04 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-05 10:45:00 | 2661.00 | 2026-01-05 11:00:00 | 2645.85 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-01-05 10:45:00 | 2661.00 | 2026-01-05 11:55:00 | 2661.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:40:00 | 2591.80 | 2026-01-07 10:05:00 | 2602.67 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-01-08 10:55:00 | 2576.90 | 2026-01-08 11:10:00 | 2584.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-16 11:15:00 | 2523.60 | 2026-01-16 11:40:00 | 2534.02 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-16 11:15:00 | 2523.60 | 2026-01-16 11:50:00 | 2523.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:30:00 | 2434.80 | 2026-01-23 10:45:00 | 2427.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-29 09:35:00 | 2327.70 | 2026-01-29 10:20:00 | 2311.33 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-01-29 09:35:00 | 2327.70 | 2026-01-29 15:20:00 | 2291.60 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2026-01-30 09:30:00 | 2323.00 | 2026-01-30 09:35:00 | 2339.86 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-01-30 09:30:00 | 2323.00 | 2026-01-30 15:20:00 | 2380.80 | TARGET_HIT | 0.50 | 2.49% |
| SELL | retest1 | 2026-02-01 09:40:00 | 2393.70 | 2026-02-01 11:40:00 | 2379.76 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-01 09:40:00 | 2393.70 | 2026-02-01 12:10:00 | 2393.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-04 10:10:00 | 2430.00 | 2026-02-04 10:40:00 | 2444.02 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-04 10:10:00 | 2430.00 | 2026-02-04 15:20:00 | 2441.10 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-10 11:00:00 | 2451.90 | 2026-02-10 11:10:00 | 2445.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-17 10:45:00 | 2344.50 | 2026-02-17 11:30:00 | 2349.26 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 09:35:00 | 2374.90 | 2026-02-18 09:50:00 | 2384.17 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-18 09:35:00 | 2374.90 | 2026-02-18 10:05:00 | 2374.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-20 09:45:00 | 2304.30 | 2026-02-20 10:00:00 | 2295.30 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-20 09:45:00 | 2304.30 | 2026-02-20 10:30:00 | 2304.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:15:00 | 2202.40 | 2026-03-05 10:25:00 | 2189.43 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-05 10:15:00 | 2202.40 | 2026-03-05 15:05:00 | 2163.30 | TARGET_HIT | 0.50 | 1.78% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2122.00 | 2026-03-06 12:20:00 | 2113.10 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2122.00 | 2026-03-06 15:20:00 | 2094.00 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2026-03-10 10:55:00 | 2113.60 | 2026-03-10 11:00:00 | 2105.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-12 10:20:00 | 2334.90 | 2026-03-12 10:35:00 | 2303.48 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2026-03-18 09:55:00 | 2197.40 | 2026-03-18 10:15:00 | 2212.75 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-18 09:55:00 | 2197.40 | 2026-03-18 10:20:00 | 2197.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 11:05:00 | 2408.90 | 2026-04-15 11:10:00 | 2400.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-15 11:05:00 | 2408.90 | 2026-04-15 15:20:00 | 2381.00 | TARGET_HIT | 0.50 | 1.16% |
| BUY | retest1 | 2026-04-17 09:50:00 | 2425.00 | 2026-04-17 09:55:00 | 2416.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 10:40:00 | 2386.30 | 2026-04-21 12:10:00 | 2393.09 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-04-21 10:40:00 | 2386.30 | 2026-04-21 15:10:00 | 2387.10 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2026-04-29 10:10:00 | 2324.80 | 2026-04-29 10:25:00 | 2317.24 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-04 09:35:00 | 2320.00 | 2026-05-04 09:55:00 | 2329.87 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-05-04 09:35:00 | 2320.00 | 2026-05-04 10:10:00 | 2320.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:25:00 | 2304.00 | 2026-05-05 10:30:00 | 2308.99 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-06 09:30:00 | 2379.70 | 2026-05-06 09:40:00 | 2369.44 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-07 09:40:00 | 2374.70 | 2026-05-07 09:50:00 | 2367.79 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-08 10:00:00 | 2370.20 | 2026-05-08 10:10:00 | 2362.39 | STOP_HIT | 1.00 | -0.33% |
