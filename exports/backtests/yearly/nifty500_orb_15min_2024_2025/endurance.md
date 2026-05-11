# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-01-03 15:25:00 (12108 bars)
- **Last close:** 2200.95
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 38
- **Target hits / Stop hits / Partials:** 5 / 38 / 11
- **Avg / median % per leg:** -0.01% / -0.28%
- **Sum % (uncompounded):** -0.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 2 | 17 | 3 | -0.03% | -0.6% |
| BUY @ 2nd Alert (retest1) | 22 | 5 | 22.7% | 2 | 17 | 3 | -0.03% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 11 | 34.4% | 3 | 21 | 8 | 0.01% | 0.2% |
| SELL @ 2nd Alert (retest1) | 32 | 11 | 34.4% | 3 | 21 | 8 | 0.01% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 16 | 29.6% | 5 | 38 | 11 | -0.01% | -0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:30:00 | 2188.70 | 2192.24 | 0.00 | ORB-short ORB[2196.00,2220.00] vol=7.0x ATR=8.81 |
| Stop hit — per-position SL triggered | 2024-05-27 11:10:00 | 2197.51 | 2190.81 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:35:00 | 2178.00 | 2197.23 | 0.00 | ORB-short ORB[2192.00,2224.45] vol=3.0x ATR=8.54 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 2186.54 | 2196.34 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:55:00 | 2449.85 | 2462.53 | 0.00 | ORB-short ORB[2463.60,2490.75] vol=1.6x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 2456.70 | 2463.66 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:15:00 | 2699.65 | 2677.79 | 0.00 | ORB-long ORB[2637.30,2676.85] vol=1.5x ATR=13.54 |
| Stop hit — per-position SL triggered | 2024-06-24 12:05:00 | 2686.11 | 2684.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 2738.10 | 2721.72 | 0.00 | ORB-long ORB[2681.30,2722.00] vol=4.1x ATR=14.91 |
| Stop hit — per-position SL triggered | 2024-06-25 13:10:00 | 2723.19 | 2740.65 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:40:00 | 2713.15 | 2709.06 | 0.00 | ORB-long ORB[2685.00,2712.50] vol=1.8x ATR=11.98 |
| Stop hit — per-position SL triggered | 2024-07-01 10:10:00 | 2701.17 | 2709.59 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 11:00:00 | 2608.20 | 2615.27 | 0.00 | ORB-short ORB[2609.30,2639.75] vol=1.6x ATR=7.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 12:30:00 | 2596.80 | 2612.25 | 0.00 | T1 1.5R @ 2596.80 |
| Stop hit — per-position SL triggered | 2024-07-15 13:00:00 | 2608.20 | 2609.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:20:00 | 2614.85 | 2607.92 | 0.00 | ORB-long ORB[2585.05,2613.25] vol=1.6x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:35:00 | 2626.64 | 2609.11 | 0.00 | T1 1.5R @ 2626.64 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 2614.85 | 2609.98 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:55:00 | 2553.40 | 2563.40 | 0.00 | ORB-short ORB[2572.20,2601.55] vol=5.5x ATR=7.63 |
| Stop hit — per-position SL triggered | 2024-07-18 11:05:00 | 2561.03 | 2562.53 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 2557.20 | 2581.53 | 0.00 | ORB-short ORB[2591.00,2628.10] vol=1.6x ATR=10.13 |
| Stop hit — per-position SL triggered | 2024-08-01 11:45:00 | 2567.33 | 2572.54 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:40:00 | 2548.05 | 2534.40 | 0.00 | ORB-long ORB[2503.60,2530.35] vol=2.6x ATR=12.77 |
| Stop hit — per-position SL triggered | 2024-08-08 10:45:00 | 2535.28 | 2535.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:40:00 | 2528.75 | 2520.31 | 0.00 | ORB-long ORB[2485.55,2522.80] vol=2.0x ATR=7.34 |
| Stop hit — per-position SL triggered | 2024-08-16 13:05:00 | 2521.41 | 2525.16 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:45:00 | 2459.05 | 2489.74 | 0.00 | ORB-short ORB[2503.00,2534.40] vol=1.6x ATR=9.26 |
| Stop hit — per-position SL triggered | 2024-08-21 12:35:00 | 2468.31 | 2473.09 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:40:00 | 2594.95 | 2606.19 | 0.00 | ORB-short ORB[2598.00,2636.15] vol=2.3x ATR=9.61 |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 2604.56 | 2603.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-09-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:40:00 | 2504.35 | 2496.92 | 0.00 | ORB-long ORB[2470.45,2499.40] vol=1.7x ATR=11.33 |
| Target hit | 2024-09-03 15:20:00 | 2505.35 | 2503.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 2527.80 | 2522.61 | 0.00 | ORB-long ORB[2487.45,2525.00] vol=5.6x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:35:00 | 2537.68 | 2523.30 | 0.00 | T1 1.5R @ 2537.68 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 2527.80 | 2526.49 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:15:00 | 2534.90 | 2550.14 | 0.00 | ORB-short ORB[2542.00,2568.65] vol=1.8x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:25:00 | 2526.47 | 2547.86 | 0.00 | T1 1.5R @ 2526.47 |
| Target hit | 2024-09-05 15:20:00 | 2508.25 | 2521.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 2451.90 | 2462.89 | 0.00 | ORB-short ORB[2455.00,2480.00] vol=1.7x ATR=10.11 |
| Stop hit — per-position SL triggered | 2024-09-12 11:40:00 | 2462.01 | 2460.03 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 2360.55 | 2370.05 | 0.00 | ORB-short ORB[2362.00,2381.05] vol=4.3x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:15:00 | 2352.69 | 2367.96 | 0.00 | T1 1.5R @ 2352.69 |
| Stop hit — per-position SL triggered | 2024-09-27 11:45:00 | 2360.55 | 2366.47 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:50:00 | 2415.25 | 2404.41 | 0.00 | ORB-long ORB[2355.00,2390.70] vol=4.0x ATR=12.01 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 2403.24 | 2405.84 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 2215.15 | 2195.88 | 0.00 | ORB-long ORB[2158.25,2181.50] vol=2.2x ATR=6.32 |
| Stop hit — per-position SL triggered | 2024-10-08 11:20:00 | 2208.83 | 2196.60 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:40:00 | 2356.95 | 2341.95 | 0.00 | ORB-long ORB[2290.85,2320.00] vol=17.7x ATR=8.15 |
| Stop hit — per-position SL triggered | 2024-10-10 10:45:00 | 2348.80 | 2341.90 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:35:00 | 2334.15 | 2321.18 | 0.00 | ORB-long ORB[2310.00,2329.00] vol=5.8x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:45:00 | 2343.86 | 2321.43 | 0.00 | T1 1.5R @ 2343.86 |
| Target hit | 2024-10-16 15:20:00 | 2433.60 | 2382.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-10-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:55:00 | 2372.60 | 2348.15 | 0.00 | ORB-long ORB[2323.55,2354.20] vol=1.7x ATR=10.11 |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 2362.49 | 2350.43 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:35:00 | 2321.50 | 2326.58 | 0.00 | ORB-short ORB[2329.00,2359.55] vol=15.8x ATR=7.77 |
| Stop hit — per-position SL triggered | 2024-10-29 10:40:00 | 2329.27 | 2329.08 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:05:00 | 2387.75 | 2397.83 | 0.00 | ORB-short ORB[2394.20,2418.00] vol=10.7x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:50:00 | 2376.45 | 2396.00 | 0.00 | T1 1.5R @ 2376.45 |
| Stop hit — per-position SL triggered | 2024-11-06 13:25:00 | 2387.75 | 2391.27 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:35:00 | 2351.60 | 2368.13 | 0.00 | ORB-short ORB[2362.95,2395.90] vol=1.9x ATR=9.84 |
| Stop hit — per-position SL triggered | 2024-11-18 09:40:00 | 2361.44 | 2365.47 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-11-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:30:00 | 2375.45 | 2368.58 | 0.00 | ORB-long ORB[2345.50,2372.15] vol=2.3x ATR=7.96 |
| Stop hit — per-position SL triggered | 2024-11-22 11:20:00 | 2367.49 | 2371.19 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:00:00 | 2382.30 | 2384.80 | 0.00 | ORB-short ORB[2392.25,2412.60] vol=3.3x ATR=7.54 |
| Stop hit — per-position SL triggered | 2024-11-26 11:10:00 | 2389.84 | 2385.29 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:05:00 | 2392.30 | 2404.30 | 0.00 | ORB-short ORB[2396.30,2413.00] vol=4.0x ATR=8.35 |
| Stop hit — per-position SL triggered | 2024-11-27 10:20:00 | 2400.65 | 2403.61 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:50:00 | 2348.00 | 2361.43 | 0.00 | ORB-short ORB[2360.05,2385.00] vol=4.6x ATR=6.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 15:00:00 | 2337.72 | 2350.77 | 0.00 | T1 1.5R @ 2337.72 |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 2348.00 | 2350.61 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:55:00 | 2335.35 | 2349.04 | 0.00 | ORB-short ORB[2339.50,2367.65] vol=2.1x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-12-06 10:30:00 | 2342.60 | 2338.12 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:35:00 | 2327.40 | 2341.16 | 0.00 | ORB-short ORB[2339.40,2357.90] vol=4.2x ATR=5.85 |
| Stop hit — per-position SL triggered | 2024-12-09 10:45:00 | 2333.25 | 2340.63 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 2323.55 | 2331.50 | 0.00 | ORB-short ORB[2330.05,2347.25] vol=3.1x ATR=4.26 |
| Stop hit — per-position SL triggered | 2024-12-12 09:55:00 | 2327.81 | 2328.57 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:15:00 | 2276.05 | 2284.21 | 0.00 | ORB-short ORB[2277.80,2307.50] vol=2.3x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-12-16 13:10:00 | 2281.81 | 2281.43 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 2275.00 | 2286.70 | 0.00 | ORB-short ORB[2286.00,2301.10] vol=2.3x ATR=6.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:40:00 | 2264.61 | 2278.78 | 0.00 | T1 1.5R @ 2264.61 |
| Target hit | 2024-12-17 14:30:00 | 2258.00 | 2257.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2024-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:55:00 | 2221.00 | 2226.67 | 0.00 | ORB-short ORB[2224.35,2245.00] vol=6.3x ATR=7.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:50:00 | 2209.14 | 2220.68 | 0.00 | T1 1.5R @ 2209.14 |
| Target hit | 2024-12-18 14:25:00 | 2214.90 | 2214.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2024-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:50:00 | 2233.10 | 2211.44 | 0.00 | ORB-long ORB[2194.60,2219.10] vol=1.6x ATR=8.72 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 2224.38 | 2215.08 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:55:00 | 2162.45 | 2148.56 | 0.00 | ORB-long ORB[2135.65,2155.10] vol=2.1x ATR=6.31 |
| Stop hit — per-position SL triggered | 2024-12-24 11:00:00 | 2156.14 | 2148.78 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 2153.00 | 2166.64 | 0.00 | ORB-short ORB[2158.20,2190.45] vol=2.3x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:45:00 | 2143.58 | 2159.20 | 0.00 | T1 1.5R @ 2143.58 |
| Stop hit — per-position SL triggered | 2024-12-26 10:00:00 | 2153.00 | 2149.69 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:45:00 | 2170.15 | 2154.60 | 0.00 | ORB-long ORB[2125.50,2157.55] vol=5.0x ATR=10.59 |
| Stop hit — per-position SL triggered | 2024-12-31 11:40:00 | 2159.56 | 2160.01 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:40:00 | 2160.00 | 2153.20 | 0.00 | ORB-long ORB[2130.00,2154.95] vol=4.7x ATR=7.83 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 2152.17 | 2154.61 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 2224.55 | 2213.38 | 0.00 | ORB-long ORB[2185.80,2218.95] vol=2.2x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-01-03 10:10:00 | 2215.36 | 2214.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-27 10:30:00 | 2188.70 | 2024-05-27 11:10:00 | 2197.51 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-31 10:35:00 | 2178.00 | 2024-05-31 10:45:00 | 2186.54 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-06-12 09:55:00 | 2449.85 | 2024-06-12 10:05:00 | 2456.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-24 11:15:00 | 2699.65 | 2024-06-24 12:05:00 | 2686.11 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-06-25 09:30:00 | 2738.10 | 2024-06-25 13:10:00 | 2723.19 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-01 09:40:00 | 2713.15 | 2024-07-01 10:10:00 | 2701.17 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-15 11:00:00 | 2608.20 | 2024-07-15 12:30:00 | 2596.80 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-15 11:00:00 | 2608.20 | 2024-07-15 13:00:00 | 2608.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:20:00 | 2614.85 | 2024-07-16 10:35:00 | 2626.64 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-16 10:20:00 | 2614.85 | 2024-07-16 10:45:00 | 2614.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 10:55:00 | 2553.40 | 2024-07-18 11:05:00 | 2561.03 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-01 10:15:00 | 2557.20 | 2024-08-01 11:45:00 | 2567.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-08 10:40:00 | 2548.05 | 2024-08-08 10:45:00 | 2535.28 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-08-16 10:40:00 | 2528.75 | 2024-08-16 13:05:00 | 2521.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-21 10:45:00 | 2459.05 | 2024-08-21 12:35:00 | 2468.31 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-23 09:40:00 | 2594.95 | 2024-08-23 10:15:00 | 2604.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-03 09:40:00 | 2504.35 | 2024-09-03 15:20:00 | 2505.35 | TARGET_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2024-09-04 09:30:00 | 2527.80 | 2024-09-04 09:35:00 | 2537.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-04 09:30:00 | 2527.80 | 2024-09-04 10:15:00 | 2527.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 11:15:00 | 2534.90 | 2024-09-05 11:25:00 | 2526.47 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-05 11:15:00 | 2534.90 | 2024-09-05 15:20:00 | 2508.25 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2024-09-12 09:30:00 | 2451.90 | 2024-09-12 11:40:00 | 2462.01 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-27 11:05:00 | 2360.55 | 2024-09-27 11:15:00 | 2352.69 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-27 11:05:00 | 2360.55 | 2024-09-27 11:45:00 | 2360.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-30 09:50:00 | 2415.25 | 2024-09-30 09:55:00 | 2403.24 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-08 11:05:00 | 2215.15 | 2024-10-08 11:20:00 | 2208.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-10 10:40:00 | 2356.95 | 2024-10-10 10:45:00 | 2348.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-16 10:35:00 | 2334.15 | 2024-10-16 10:45:00 | 2343.86 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-16 10:35:00 | 2334.15 | 2024-10-16 15:20:00 | 2433.60 | TARGET_HIT | 0.50 | 4.26% |
| BUY | retest1 | 2024-10-23 10:55:00 | 2372.60 | 2024-10-23 11:15:00 | 2362.49 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-29 10:35:00 | 2321.50 | 2024-10-29 10:40:00 | 2329.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-06 11:05:00 | 2387.75 | 2024-11-06 11:50:00 | 2376.45 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-11-06 11:05:00 | 2387.75 | 2024-11-06 13:25:00 | 2387.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-18 09:35:00 | 2351.60 | 2024-11-18 09:40:00 | 2361.44 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-11-22 10:30:00 | 2375.45 | 2024-11-22 11:20:00 | 2367.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-26 11:00:00 | 2382.30 | 2024-11-26 11:10:00 | 2389.84 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-27 10:05:00 | 2392.30 | 2024-11-27 10:20:00 | 2400.65 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-05 09:50:00 | 2348.00 | 2024-12-05 15:00:00 | 2337.72 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-05 09:50:00 | 2348.00 | 2024-12-05 15:15:00 | 2348.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 09:55:00 | 2335.35 | 2024-12-06 10:30:00 | 2342.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-09 10:35:00 | 2327.40 | 2024-12-09 10:45:00 | 2333.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-12 09:40:00 | 2323.55 | 2024-12-12 09:55:00 | 2327.81 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-16 11:15:00 | 2276.05 | 2024-12-16 13:10:00 | 2281.81 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-17 09:30:00 | 2275.00 | 2024-12-17 09:40:00 | 2264.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-17 09:30:00 | 2275.00 | 2024-12-17 14:30:00 | 2258.00 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-12-18 09:55:00 | 2221.00 | 2024-12-18 10:50:00 | 2209.14 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-18 09:55:00 | 2221.00 | 2024-12-18 14:25:00 | 2214.90 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2024-12-19 09:50:00 | 2233.10 | 2024-12-19 10:00:00 | 2224.38 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-24 10:55:00 | 2162.45 | 2024-12-24 11:00:00 | 2156.14 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-26 09:30:00 | 2153.00 | 2024-12-26 09:45:00 | 2143.58 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-26 09:30:00 | 2153.00 | 2024-12-26 10:00:00 | 2153.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 10:45:00 | 2170.15 | 2024-12-31 11:40:00 | 2159.56 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-01 10:40:00 | 2160.00 | 2025-01-01 11:10:00 | 2152.17 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-03 10:05:00 | 2224.55 | 2025-01-03 10:10:00 | 2215.36 | STOP_HIT | 1.00 | -0.41% |
