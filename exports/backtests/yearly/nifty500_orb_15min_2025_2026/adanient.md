# Adani Enterprises Ltd. (ADANIENT)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2502.00
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 14 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 77
- **Target hits / Stop hits / Partials:** 14 / 77 / 41
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 22.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 32 | 49.2% | 10 | 33 | 22 | 0.26% | 17.2% |
| BUY @ 2nd Alert (retest1) | 65 | 32 | 49.2% | 10 | 33 | 22 | 0.26% | 17.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 67 | 23 | 34.3% | 4 | 44 | 19 | 0.07% | 4.9% |
| SELL @ 2nd Alert (retest1) | 67 | 23 | 34.3% | 4 | 44 | 19 | 0.07% | 4.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 55 | 41.7% | 14 | 77 | 41 | 0.17% | 22.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 11:00:00 | 2359.05 | 2348.50 | 0.00 | ORB-long ORB[2331.61,2354.78] vol=2.0x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:30:00 | 2372.38 | 2351.95 | 0.00 | T1 1.5R @ 2372.38 |
| Stop hit — per-position SL triggered | 2025-05-13 12:50:00 | 2359.05 | 2354.48 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:35:00 | 2423.23 | 2393.75 | 0.00 | ORB-long ORB[2382.12,2398.22] vol=2.8x ATR=8.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 10:45:00 | 2435.30 | 2406.57 | 0.00 | T1 1.5R @ 2435.30 |
| Stop hit — per-position SL triggered | 2025-05-15 11:05:00 | 2423.23 | 2412.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:35:00 | 2484.98 | 2469.44 | 0.00 | ORB-long ORB[2457.94,2478.97] vol=2.0x ATR=7.57 |
| Stop hit — per-position SL triggered | 2025-05-20 09:40:00 | 2477.41 | 2470.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 2447.95 | 2432.98 | 0.00 | ORB-long ORB[2414.50,2435.35] vol=3.3x ATR=7.19 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 2440.76 | 2436.52 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:15:00 | 2474.61 | 2461.38 | 0.00 | ORB-long ORB[2456.09,2472.09] vol=1.6x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:20:00 | 2482.57 | 2463.14 | 0.00 | T1 1.5R @ 2482.57 |
| Stop hit — per-position SL triggered | 2025-05-27 13:05:00 | 2474.61 | 2471.58 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:40:00 | 2376.98 | 2390.38 | 0.00 | ORB-short ORB[2384.74,2406.26] vol=1.6x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:50:00 | 2367.57 | 2387.73 | 0.00 | T1 1.5R @ 2367.57 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 2376.98 | 2386.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 2432.73 | 2422.88 | 0.00 | ORB-long ORB[2411.59,2431.27] vol=1.5x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:35:00 | 2441.50 | 2425.34 | 0.00 | T1 1.5R @ 2441.50 |
| Stop hit — per-position SL triggered | 2025-06-05 09:40:00 | 2432.73 | 2426.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 2447.37 | 2439.02 | 0.00 | ORB-long ORB[2427.59,2445.91] vol=1.8x ATR=6.25 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 2441.12 | 2439.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 2449.99 | 2461.25 | 0.00 | ORB-short ORB[2457.64,2479.65] vol=3.1x ATR=6.46 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 2456.45 | 2458.51 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:50:00 | 2485.47 | 2501.11 | 0.00 | ORB-short ORB[2493.90,2517.17] vol=2.2x ATR=6.41 |
| Stop hit — per-position SL triggered | 2025-06-12 11:00:00 | 2491.88 | 2500.20 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2395.21 | 2416.21 | 0.00 | ORB-short ORB[2406.46,2434.96] vol=2.2x ATR=7.55 |
| Stop hit — per-position SL triggered | 2025-06-16 09:40:00 | 2402.76 | 2410.30 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 2372.33 | 2357.54 | 0.00 | ORB-long ORB[2353.23,2369.32] vol=1.9x ATR=8.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:40:00 | 2385.54 | 2365.95 | 0.00 | T1 1.5R @ 2385.54 |
| Stop hit — per-position SL triggered | 2025-06-20 12:25:00 | 2372.33 | 2367.84 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:10:00 | 2400.06 | 2377.18 | 0.00 | ORB-long ORB[2347.12,2381.06] vol=2.2x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 13:20:00 | 2411.63 | 2388.82 | 0.00 | T1 1.5R @ 2411.63 |
| Stop hit — per-position SL triggered | 2025-06-23 15:05:00 | 2400.06 | 2391.97 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:00:00 | 2454.15 | 2440.16 | 0.00 | ORB-long ORB[2426.62,2452.02] vol=1.8x ATR=9.33 |
| Stop hit — per-position SL triggered | 2025-06-24 10:10:00 | 2444.82 | 2440.99 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 2533.26 | 2524.55 | 0.00 | ORB-long ORB[2503.21,2526.38] vol=2.4x ATR=7.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:40:00 | 2545.15 | 2529.80 | 0.00 | T1 1.5R @ 2545.15 |
| Stop hit — per-position SL triggered | 2025-06-27 10:05:00 | 2533.26 | 2540.87 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 2535.78 | 2551.62 | 0.00 | ORB-short ORB[2543.06,2563.32] vol=1.6x ATR=6.73 |
| Stop hit — per-position SL triggered | 2025-07-01 11:00:00 | 2542.51 | 2550.95 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 2497.30 | 2508.83 | 0.00 | ORB-short ORB[2500.50,2520.66] vol=2.0x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 2502.33 | 2508.05 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 2499.33 | 2512.04 | 0.00 | ORB-short ORB[2505.44,2524.54] vol=2.4x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-07-10 11:10:00 | 2503.37 | 2511.76 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:15:00 | 2484.21 | 2494.71 | 0.00 | ORB-short ORB[2491.58,2512.81] vol=1.6x ATR=5.09 |
| Stop hit — per-position SL triggered | 2025-07-11 11:40:00 | 2489.30 | 2494.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:55:00 | 2494.58 | 2482.36 | 0.00 | ORB-long ORB[2468.41,2485.76] vol=2.0x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:20:00 | 2506.15 | 2489.24 | 0.00 | T1 1.5R @ 2506.15 |
| Target hit | 2025-07-14 12:50:00 | 2494.78 | 2497.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2025-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:50:00 | 2520.18 | 2508.66 | 0.00 | ORB-long ORB[2503.31,2512.91] vol=2.6x ATR=5.68 |
| Stop hit — per-position SL triggered | 2025-07-15 11:35:00 | 2514.50 | 2513.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:15:00 | 2526.67 | 2530.71 | 0.00 | ORB-short ORB[2535.20,2544.12] vol=2.3x ATR=4.86 |
| Stop hit — per-position SL triggered | 2025-07-24 10:30:00 | 2531.53 | 2530.57 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:45:00 | 2515.23 | 2524.27 | 0.00 | ORB-short ORB[2515.33,2536.95] vol=2.4x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:05:00 | 2505.32 | 2520.76 | 0.00 | T1 1.5R @ 2505.32 |
| Target hit | 2025-07-25 15:20:00 | 2473.83 | 2490.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2025-07-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:50:00 | 2457.16 | 2465.55 | 0.00 | ORB-short ORB[2460.65,2474.71] vol=1.6x ATR=6.34 |
| Stop hit — per-position SL triggered | 2025-07-30 09:55:00 | 2463.50 | 2465.41 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:35:00 | 2414.70 | 2427.89 | 0.00 | ORB-short ORB[2423.71,2443.10] vol=2.3x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:10:00 | 2405.25 | 2419.30 | 0.00 | T1 1.5R @ 2405.25 |
| Stop hit — per-position SL triggered | 2025-07-31 11:00:00 | 2414.70 | 2415.31 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:45:00 | 2256.38 | 2277.00 | 0.00 | ORB-short ORB[2281.10,2297.68] vol=1.5x ATR=6.39 |
| Stop hit — per-position SL triggered | 2025-08-05 10:35:00 | 2262.77 | 2266.30 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:35:00 | 2243.49 | 2258.62 | 0.00 | ORB-short ORB[2254.05,2276.25] vol=1.5x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:55:00 | 2233.90 | 2252.27 | 0.00 | T1 1.5R @ 2233.90 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 2243.49 | 2247.03 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:05:00 | 2192.01 | 2207.37 | 0.00 | ORB-short ORB[2204.12,2224.97] vol=2.7x ATR=6.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:40:00 | 2181.87 | 2201.46 | 0.00 | T1 1.5R @ 2181.87 |
| Target hit | 2025-08-07 14:15:00 | 2173.78 | 2173.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2025-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:10:00 | 2135.29 | 2161.71 | 0.00 | ORB-short ORB[2166.90,2191.91] vol=2.2x ATR=9.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:35:00 | 2121.62 | 2147.87 | 0.00 | T1 1.5R @ 2121.62 |
| Target hit | 2025-08-08 15:20:00 | 2112.70 | 2118.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-08-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:40:00 | 2156.23 | 2141.97 | 0.00 | ORB-long ORB[2116.39,2145.96] vol=1.6x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 10:50:00 | 2166.20 | 2144.19 | 0.00 | T1 1.5R @ 2166.20 |
| Target hit | 2025-08-11 15:20:00 | 2209.46 | 2188.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 2222.25 | 2230.00 | 0.00 | ORB-short ORB[2225.84,2244.26] vol=2.2x ATR=6.94 |
| Stop hit — per-position SL triggered | 2025-08-13 09:45:00 | 2229.19 | 2229.11 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 2202.09 | 2218.46 | 0.00 | ORB-short ORB[2208.58,2233.69] vol=1.7x ATR=6.60 |
| Stop hit — per-position SL triggered | 2025-08-14 09:40:00 | 2208.69 | 2214.46 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:45:00 | 2292.06 | 2261.18 | 0.00 | ORB-long ORB[2243.49,2268.50] vol=2.1x ATR=7.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:05:00 | 2302.68 | 2271.76 | 0.00 | T1 1.5R @ 2302.68 |
| Target hit | 2025-08-19 15:20:00 | 2316.97 | 2295.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 2321.92 | 2313.73 | 0.00 | ORB-long ORB[2299.72,2318.43] vol=1.9x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 11:25:00 | 2330.28 | 2316.16 | 0.00 | T1 1.5R @ 2330.28 |
| Stop hit — per-position SL triggered | 2025-08-20 11:35:00 | 2321.92 | 2316.38 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 2279.55 | 2287.88 | 0.00 | ORB-short ORB[2281.78,2303.79] vol=2.3x ATR=5.43 |
| Stop hit — per-position SL triggered | 2025-08-22 09:35:00 | 2284.98 | 2287.07 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 09:55:00 | 2226.91 | 2207.16 | 0.00 | ORB-long ORB[2186.19,2210.81] vol=1.5x ATR=7.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:35:00 | 2237.85 | 2213.51 | 0.00 | T1 1.5R @ 2237.85 |
| Target hit | 2025-08-28 12:15:00 | 2230.20 | 2232.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:30:00 | 2185.22 | 2201.01 | 0.00 | ORB-short ORB[2193.07,2219.83] vol=1.9x ATR=7.10 |
| Stop hit — per-position SL triggered | 2025-08-29 09:35:00 | 2192.32 | 2198.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:35:00 | 2196.66 | 2208.10 | 0.00 | ORB-short ORB[2203.15,2219.05] vol=1.8x ATR=5.07 |
| Stop hit — per-position SL triggered | 2025-09-03 11:00:00 | 2201.73 | 2206.56 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 2203.54 | 2213.36 | 0.00 | ORB-short ORB[2204.71,2222.45] vol=1.6x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 2195.97 | 2211.51 | 0.00 | T1 1.5R @ 2195.97 |
| Stop hit — per-position SL triggered | 2025-09-05 10:25:00 | 2203.54 | 2209.55 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:55:00 | 2296.71 | 2284.43 | 0.00 | ORB-long ORB[2268.79,2289.83] vol=1.9x ATR=5.52 |
| Stop hit — per-position SL triggered | 2025-09-11 10:00:00 | 2291.19 | 2285.98 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 11:00:00 | 2344.80 | 2334.20 | 0.00 | ORB-long ORB[2326.76,2340.82] vol=6.2x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:05:00 | 2351.07 | 2341.30 | 0.00 | T1 1.5R @ 2351.07 |
| Stop hit — per-position SL triggered | 2025-09-18 11:40:00 | 2344.80 | 2344.54 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:30:00 | 2560.99 | 2549.16 | 0.00 | ORB-long ORB[2538.50,2558.47] vol=1.8x ATR=8.37 |
| Stop hit — per-position SL triggered | 2025-09-25 10:40:00 | 2552.62 | 2549.78 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:20:00 | 2419.83 | 2437.55 | 0.00 | ORB-short ORB[2433.41,2457.06] vol=2.7x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:30:00 | 2409.08 | 2435.39 | 0.00 | T1 1.5R @ 2409.08 |
| Stop hit — per-position SL triggered | 2025-10-01 11:55:00 | 2419.83 | 2427.40 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:05:00 | 2517.17 | 2508.84 | 0.00 | ORB-long ORB[2486.15,2515.81] vol=1.9x ATR=11.51 |
| Stop hit — per-position SL triggered | 2025-10-03 10:40:00 | 2505.66 | 2509.72 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:50:00 | 2491.48 | 2498.43 | 0.00 | ORB-short ORB[2501.27,2520.47] vol=1.7x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-10-06 11:20:00 | 2497.46 | 2497.30 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 2530.36 | 2520.62 | 0.00 | ORB-long ORB[2500.01,2528.90] vol=2.7x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-10-07 09:40:00 | 2523.70 | 2523.51 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:20:00 | 2449.31 | 2468.51 | 0.00 | ORB-short ORB[2460.65,2486.73] vol=1.6x ATR=7.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:55:00 | 2437.40 | 2462.29 | 0.00 | T1 1.5R @ 2437.40 |
| Stop hit — per-position SL triggered | 2025-10-08 13:35:00 | 2449.31 | 2449.48 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 2476.65 | 2470.84 | 0.00 | ORB-long ORB[2465.40,2475.97] vol=1.5x ATR=5.34 |
| Stop hit — per-position SL triggered | 2025-10-10 12:05:00 | 2471.31 | 2474.59 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:50:00 | 2436.22 | 2446.66 | 0.00 | ORB-short ORB[2440.68,2464.53] vol=1.6x ATR=8.02 |
| Stop hit — per-position SL triggered | 2025-10-13 10:10:00 | 2444.24 | 2444.19 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:30:00 | 2447.66 | 2453.09 | 0.00 | ORB-short ORB[2448.92,2461.43] vol=1.7x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:45:00 | 2441.05 | 2449.67 | 0.00 | T1 1.5R @ 2441.05 |
| Stop hit — per-position SL triggered | 2025-10-14 09:50:00 | 2447.66 | 2449.53 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 2489.06 | 2476.28 | 0.00 | ORB-long ORB[2470.34,2485.57] vol=4.6x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:35:00 | 2497.20 | 2481.79 | 0.00 | T1 1.5R @ 2497.20 |
| Stop hit — per-position SL triggered | 2025-10-17 11:45:00 | 2489.06 | 2482.34 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 2434.38 | 2426.52 | 0.00 | ORB-long ORB[2414.02,2430.40] vol=2.0x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-10-28 09:45:00 | 2430.47 | 2429.34 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 2448.92 | 2435.85 | 0.00 | ORB-long ORB[2419.35,2441.16] vol=1.7x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:30:00 | 2458.23 | 2440.10 | 0.00 | T1 1.5R @ 2458.23 |
| Target hit | 2025-10-29 12:10:00 | 2480.04 | 2482.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2025-10-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:25:00 | 2434.09 | 2449.88 | 0.00 | ORB-short ORB[2439.22,2459.87] vol=1.6x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:55:00 | 2424.04 | 2443.66 | 0.00 | T1 1.5R @ 2424.04 |
| Stop hit — per-position SL triggered | 2025-10-31 11:25:00 | 2434.09 | 2441.69 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:55:00 | 2381.15 | 2396.47 | 0.00 | ORB-short ORB[2395.60,2412.95] vol=1.9x ATR=6.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:20:00 | 2370.78 | 2393.77 | 0.00 | T1 1.5R @ 2370.78 |
| Stop hit — per-position SL triggered | 2025-11-04 13:05:00 | 2381.15 | 2382.15 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 09:45:00 | 2259.87 | 2241.27 | 0.00 | ORB-long ORB[2221.48,2247.17] vol=1.5x ATR=8.54 |
| Stop hit — per-position SL triggered | 2025-11-07 09:55:00 | 2251.33 | 2245.84 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:00:00 | 2272.67 | 2282.40 | 0.00 | ORB-short ORB[2284.40,2300.30] vol=1.9x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-11-11 12:20:00 | 2277.38 | 2279.42 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 2340.14 | 2308.61 | 0.00 | ORB-long ORB[2288.95,2321.24] vol=3.5x ATR=9.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:40:00 | 2354.99 | 2316.95 | 0.00 | T1 1.5R @ 2354.99 |
| Target hit | 2025-11-12 15:20:00 | 2410.04 | 2393.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-11-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:45:00 | 2457.45 | 2430.15 | 0.00 | ORB-long ORB[2405.29,2436.12] vol=3.3x ATR=9.23 |
| Stop hit — per-position SL triggered | 2025-11-14 09:50:00 | 2448.22 | 2431.77 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 2289.00 | 2264.83 | 0.00 | ORB-long ORB[2247.80,2273.90] vol=1.6x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:00:00 | 2302.08 | 2274.09 | 0.00 | T1 1.5R @ 2302.08 |
| Target hit | 2025-11-28 14:05:00 | 2296.90 | 2298.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2025-12-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 11:00:00 | 2226.50 | 2214.43 | 0.00 | ORB-long ORB[2210.10,2222.00] vol=1.6x ATR=5.27 |
| Stop hit — per-position SL triggered | 2025-12-05 11:35:00 | 2221.23 | 2217.06 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 2227.10 | 2206.22 | 0.00 | ORB-long ORB[2196.50,2218.80] vol=1.8x ATR=8.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 12:10:00 | 2240.33 | 2210.98 | 0.00 | T1 1.5R @ 2240.33 |
| Stop hit — per-position SL triggered | 2025-12-09 14:50:00 | 2227.10 | 2223.07 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:35:00 | 2248.60 | 2261.67 | 0.00 | ORB-short ORB[2258.70,2276.80] vol=1.7x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-12-16 09:40:00 | 2253.39 | 2260.09 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 09:50:00 | 2235.00 | 2243.10 | 0.00 | ORB-short ORB[2235.80,2250.00] vol=2.8x ATR=5.60 |
| Stop hit — per-position SL triggered | 2025-12-19 10:00:00 | 2240.60 | 2241.38 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 2270.50 | 2260.29 | 0.00 | ORB-long ORB[2245.90,2267.00] vol=2.2x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-12-22 11:10:00 | 2265.95 | 2260.57 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:50:00 | 2237.00 | 2230.57 | 0.00 | ORB-long ORB[2219.00,2235.70] vol=2.4x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-12-29 10:00:00 | 2232.26 | 2231.20 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 2209.70 | 2203.42 | 0.00 | ORB-long ORB[2196.80,2207.70] vol=1.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-12-30 09:45:00 | 2205.36 | 2203.50 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:35:00 | 2258.20 | 2250.99 | 0.00 | ORB-long ORB[2241.00,2249.70] vol=2.3x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:40:00 | 2265.26 | 2257.77 | 0.00 | T1 1.5R @ 2265.26 |
| Target hit | 2026-01-01 10:15:00 | 2263.50 | 2270.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — BUY (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 2272.60 | 2266.71 | 0.00 | ORB-long ORB[2259.00,2269.90] vol=1.8x ATR=4.45 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 2268.15 | 2266.90 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:40:00 | 2243.90 | 2252.82 | 0.00 | ORB-short ORB[2252.00,2265.90] vol=1.8x ATR=4.94 |
| Stop hit — per-position SL triggered | 2026-01-07 09:55:00 | 2248.84 | 2251.87 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:40:00 | 2246.90 | 2256.77 | 0.00 | ORB-short ORB[2256.20,2275.90] vol=1.8x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:00:00 | 2238.54 | 2253.90 | 0.00 | T1 1.5R @ 2238.54 |
| Target hit | 2026-01-08 15:20:00 | 2210.90 | 2231.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2026-01-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:50:00 | 2175.00 | 2164.90 | 0.00 | ORB-long ORB[2160.40,2169.80] vol=2.0x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-01-16 09:55:00 | 2169.60 | 2165.84 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:35:00 | 2129.70 | 2142.72 | 0.00 | ORB-short ORB[2140.00,2156.80] vol=1.7x ATR=6.29 |
| Stop hit — per-position SL triggered | 2026-01-19 10:55:00 | 2135.99 | 2141.51 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 2113.80 | 2124.57 | 0.00 | ORB-short ORB[2117.20,2139.90] vol=1.5x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:35:00 | 2105.25 | 2121.46 | 0.00 | T1 1.5R @ 2105.25 |
| Stop hit — per-position SL triggered | 2026-01-20 10:00:00 | 2113.80 | 2116.24 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 2013.00 | 2043.98 | 0.00 | ORB-short ORB[2047.10,2068.20] vol=1.8x ATR=9.22 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 2022.22 | 2043.25 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:10:00 | 2213.70 | 2225.43 | 0.00 | ORB-short ORB[2220.60,2238.00] vol=2.5x ATR=6.83 |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 2220.53 | 2224.94 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 2242.40 | 2246.56 | 0.00 | ORB-short ORB[2244.10,2259.00] vol=1.5x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:55:00 | 2233.69 | 2245.19 | 0.00 | T1 1.5R @ 2233.69 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 2242.40 | 2245.19 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:35:00 | 2211.60 | 2216.84 | 0.00 | ORB-short ORB[2216.00,2229.50] vol=1.6x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 2204.56 | 2215.05 | 0.00 | T1 1.5R @ 2204.56 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 2211.60 | 2212.91 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 2191.10 | 2187.72 | 0.00 | ORB-long ORB[2170.00,2181.90] vol=4.3x ATR=7.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 2202.22 | 2188.70 | 0.00 | T1 1.5R @ 2202.22 |
| Target hit | 2026-02-17 15:20:00 | 2244.00 | 2217.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 2195.80 | 2220.27 | 0.00 | ORB-short ORB[2228.10,2254.00] vol=1.7x ATR=6.56 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 2202.36 | 2213.64 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 2172.80 | 2194.03 | 0.00 | ORB-short ORB[2197.50,2218.90] vol=1.7x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 2164.45 | 2188.19 | 0.00 | T1 1.5R @ 2164.45 |
| Stop hit — per-position SL triggered | 2026-02-19 11:20:00 | 2172.80 | 2187.37 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:35:00 | 2193.00 | 2179.47 | 0.00 | ORB-long ORB[2155.30,2184.00] vol=2.9x ATR=7.95 |
| Stop hit — per-position SL triggered | 2026-02-23 10:40:00 | 2185.05 | 2188.71 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 2181.20 | 2197.73 | 0.00 | ORB-short ORB[2188.40,2211.30] vol=2.3x ATR=5.58 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 2186.78 | 2197.24 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 2214.50 | 2220.41 | 0.00 | ORB-short ORB[2217.60,2232.40] vol=1.6x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:00:00 | 2206.84 | 2219.19 | 0.00 | T1 1.5R @ 2206.84 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 2214.50 | 2219.15 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 2194.30 | 2205.18 | 0.00 | ORB-short ORB[2202.10,2216.40] vol=1.5x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 2185.43 | 2200.56 | 0.00 | T1 1.5R @ 2185.43 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 2194.30 | 2199.95 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2064.10 | 2078.18 | 0.00 | ORB-short ORB[2065.10,2088.50] vol=2.4x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-03-06 11:50:00 | 2070.09 | 2073.87 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 1990.20 | 2010.61 | 0.00 | ORB-short ORB[2009.10,2030.00] vol=1.9x ATR=8.61 |
| Stop hit — per-position SL triggered | 2026-03-10 14:10:00 | 1998.81 | 1996.55 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:35:00 | 2004.00 | 1970.23 | 0.00 | ORB-long ORB[1947.10,1966.60] vol=1.5x ATR=8.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:55:00 | 2016.74 | 1977.74 | 0.00 | T1 1.5R @ 2016.74 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 2004.00 | 1986.40 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2164.00 | 2171.97 | 0.00 | ORB-short ORB[2166.20,2186.00] vol=2.8x ATR=7.61 |
| Stop hit — per-position SL triggered | 2026-04-16 09:45:00 | 2171.61 | 2172.11 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:20:00 | 2258.90 | 2248.74 | 0.00 | ORB-long ORB[2221.80,2254.90] vol=2.6x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:50:00 | 2271.43 | 2250.95 | 0.00 | T1 1.5R @ 2271.43 |
| Target hit | 2026-04-23 15:20:00 | 2300.00 | 2284.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 2437.90 | 2426.08 | 0.00 | ORB-long ORB[2404.20,2431.70] vol=2.9x ATR=9.17 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 2428.73 | 2426.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 11:00:00 | 2359.05 | 2025-05-13 11:30:00 | 2372.38 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-13 11:00:00 | 2359.05 | 2025-05-13 12:50:00 | 2359.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:35:00 | 2423.23 | 2025-05-15 10:45:00 | 2435.30 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-05-15 10:35:00 | 2423.23 | 2025-05-15 11:05:00 | 2423.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-20 09:35:00 | 2484.98 | 2025-05-20 09:40:00 | 2477.41 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-23 10:45:00 | 2447.95 | 2025-05-23 11:45:00 | 2440.76 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-27 11:15:00 | 2474.61 | 2025-05-27 11:20:00 | 2482.57 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-05-27 11:15:00 | 2474.61 | 2025-05-27 13:05:00 | 2474.61 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:40:00 | 2376.98 | 2025-06-04 09:50:00 | 2367.57 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-04 09:40:00 | 2376.98 | 2025-06-04 10:00:00 | 2376.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:30:00 | 2432.73 | 2025-06-05 09:35:00 | 2441.50 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-06-05 09:30:00 | 2432.73 | 2025-06-05 09:40:00 | 2432.73 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 10:05:00 | 2447.37 | 2025-06-06 10:10:00 | 2441.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-09 09:30:00 | 2449.99 | 2025-06-09 09:45:00 | 2456.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-12 10:50:00 | 2485.47 | 2025-06-12 11:00:00 | 2491.88 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-16 09:30:00 | 2395.21 | 2025-06-16 09:40:00 | 2402.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-20 10:15:00 | 2372.33 | 2025-06-20 11:40:00 | 2385.54 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-20 10:15:00 | 2372.33 | 2025-06-20 12:25:00 | 2372.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-23 11:10:00 | 2400.06 | 2025-06-23 13:20:00 | 2411.63 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-23 11:10:00 | 2400.06 | 2025-06-23 15:05:00 | 2400.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 10:00:00 | 2454.15 | 2025-06-24 10:10:00 | 2444.82 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-27 09:35:00 | 2533.26 | 2025-06-27 09:40:00 | 2545.15 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-27 09:35:00 | 2533.26 | 2025-06-27 10:05:00 | 2533.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:55:00 | 2535.78 | 2025-07-01 11:00:00 | 2542.51 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-08 11:05:00 | 2497.30 | 2025-07-08 11:15:00 | 2502.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-10 11:00:00 | 2499.33 | 2025-07-10 11:10:00 | 2503.37 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-11 11:15:00 | 2484.21 | 2025-07-11 11:40:00 | 2489.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-14 09:55:00 | 2494.58 | 2025-07-14 10:20:00 | 2506.15 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-14 09:55:00 | 2494.58 | 2025-07-14 12:50:00 | 2494.78 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-07-15 10:50:00 | 2520.18 | 2025-07-15 11:35:00 | 2514.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-24 10:15:00 | 2526.67 | 2025-07-24 10:30:00 | 2531.53 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-25 09:45:00 | 2515.23 | 2025-07-25 10:05:00 | 2505.32 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-25 09:45:00 | 2515.23 | 2025-07-25 15:20:00 | 2473.83 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2025-07-30 09:50:00 | 2457.16 | 2025-07-30 09:55:00 | 2463.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-31 09:35:00 | 2414.70 | 2025-07-31 10:10:00 | 2405.25 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-31 09:35:00 | 2414.70 | 2025-07-31 11:00:00 | 2414.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-05 09:45:00 | 2256.38 | 2025-08-05 10:35:00 | 2262.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-06 10:35:00 | 2243.49 | 2025-08-06 10:55:00 | 2233.90 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-08-06 10:35:00 | 2243.49 | 2025-08-06 12:15:00 | 2243.49 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 10:05:00 | 2192.01 | 2025-08-07 10:40:00 | 2181.87 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-07 10:05:00 | 2192.01 | 2025-08-07 14:15:00 | 2173.78 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-08-08 10:10:00 | 2135.29 | 2025-08-08 10:35:00 | 2121.62 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-08-08 10:10:00 | 2135.29 | 2025-08-08 15:20:00 | 2112.70 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2025-08-11 10:40:00 | 2156.23 | 2025-08-11 10:50:00 | 2166.20 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-11 10:40:00 | 2156.23 | 2025-08-11 15:20:00 | 2209.46 | TARGET_HIT | 0.50 | 2.47% |
| SELL | retest1 | 2025-08-13 09:30:00 | 2222.25 | 2025-08-13 09:45:00 | 2229.19 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-14 09:30:00 | 2202.09 | 2025-08-14 09:40:00 | 2208.69 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-19 10:45:00 | 2292.06 | 2025-08-19 11:05:00 | 2302.68 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-19 10:45:00 | 2292.06 | 2025-08-19 15:20:00 | 2316.97 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2025-08-20 10:55:00 | 2321.92 | 2025-08-20 11:25:00 | 2330.28 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-08-20 10:55:00 | 2321.92 | 2025-08-20 11:35:00 | 2321.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 09:30:00 | 2279.55 | 2025-08-22 09:35:00 | 2284.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-28 09:55:00 | 2226.91 | 2025-08-28 10:35:00 | 2237.85 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-08-28 09:55:00 | 2226.91 | 2025-08-28 12:15:00 | 2230.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-08-29 09:30:00 | 2185.22 | 2025-08-29 09:35:00 | 2192.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-03 10:35:00 | 2196.66 | 2025-09-03 11:00:00 | 2201.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-05 10:10:00 | 2203.54 | 2025-09-05 10:15:00 | 2195.97 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-05 10:10:00 | 2203.54 | 2025-09-05 10:25:00 | 2203.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 09:55:00 | 2296.71 | 2025-09-11 10:00:00 | 2291.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-18 11:00:00 | 2344.80 | 2025-09-18 11:05:00 | 2351.07 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-09-18 11:00:00 | 2344.80 | 2025-09-18 11:40:00 | 2344.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-25 10:30:00 | 2560.99 | 2025-09-25 10:40:00 | 2552.62 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-01 10:20:00 | 2419.83 | 2025-10-01 10:30:00 | 2409.08 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-01 10:20:00 | 2419.83 | 2025-10-01 11:55:00 | 2419.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 10:05:00 | 2517.17 | 2025-10-03 10:40:00 | 2505.66 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-10-06 10:50:00 | 2491.48 | 2025-10-06 11:20:00 | 2497.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-07 09:30:00 | 2530.36 | 2025-10-07 09:40:00 | 2523.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-08 10:20:00 | 2449.31 | 2025-10-08 10:55:00 | 2437.40 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-10-08 10:20:00 | 2449.31 | 2025-10-08 13:35:00 | 2449.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:40:00 | 2476.65 | 2025-10-10 12:05:00 | 2471.31 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-13 09:50:00 | 2436.22 | 2025-10-13 10:10:00 | 2444.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-14 09:30:00 | 2447.66 | 2025-10-14 09:45:00 | 2441.05 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-14 09:30:00 | 2447.66 | 2025-10-14 09:50:00 | 2447.66 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 11:05:00 | 2489.06 | 2025-10-17 11:35:00 | 2497.20 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-17 11:05:00 | 2489.06 | 2025-10-17 11:45:00 | 2489.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:30:00 | 2434.38 | 2025-10-28 09:45:00 | 2430.47 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-29 10:25:00 | 2448.92 | 2025-10-29 10:30:00 | 2458.23 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-29 10:25:00 | 2448.92 | 2025-10-29 12:10:00 | 2480.04 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2025-10-31 10:25:00 | 2434.09 | 2025-10-31 10:55:00 | 2424.04 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-31 10:25:00 | 2434.09 | 2025-10-31 11:25:00 | 2434.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:55:00 | 2381.15 | 2025-11-04 11:20:00 | 2370.78 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-11-04 10:55:00 | 2381.15 | 2025-11-04 13:05:00 | 2381.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 09:45:00 | 2259.87 | 2025-11-07 09:55:00 | 2251.33 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-11-11 11:00:00 | 2272.67 | 2025-11-11 12:20:00 | 2277.38 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-12 09:35:00 | 2340.14 | 2025-11-12 09:40:00 | 2354.99 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-11-12 09:35:00 | 2340.14 | 2025-11-12 15:20:00 | 2410.04 | TARGET_HIT | 0.50 | 2.99% |
| BUY | retest1 | 2025-11-14 09:45:00 | 2457.45 | 2025-11-14 09:50:00 | 2448.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-11-28 09:45:00 | 2289.00 | 2025-11-28 10:00:00 | 2302.08 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-11-28 09:45:00 | 2289.00 | 2025-11-28 14:05:00 | 2296.90 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-05 11:00:00 | 2226.50 | 2025-12-05 11:35:00 | 2221.23 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-09 11:15:00 | 2227.10 | 2025-12-09 12:10:00 | 2240.33 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-12-09 11:15:00 | 2227.10 | 2025-12-09 14:50:00 | 2227.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 09:35:00 | 2248.60 | 2025-12-16 09:40:00 | 2253.39 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-19 09:50:00 | 2235.00 | 2025-12-19 10:00:00 | 2240.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-22 11:00:00 | 2270.50 | 2025-12-22 11:10:00 | 2265.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-29 09:50:00 | 2237.00 | 2025-12-29 10:00:00 | 2232.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-30 09:40:00 | 2209.70 | 2025-12-30 09:45:00 | 2205.36 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-01 09:35:00 | 2258.20 | 2026-01-01 09:40:00 | 2265.26 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-01 09:35:00 | 2258.20 | 2026-01-01 10:15:00 | 2263.50 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2026-01-02 10:10:00 | 2272.60 | 2026-01-02 10:15:00 | 2268.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-07 09:40:00 | 2243.90 | 2026-01-07 09:55:00 | 2248.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-08 10:40:00 | 2246.90 | 2026-01-08 11:00:00 | 2238.54 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-08 10:40:00 | 2246.90 | 2026-01-08 15:20:00 | 2210.90 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2026-01-16 09:50:00 | 2175.00 | 2026-01-16 09:55:00 | 2169.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-19 10:35:00 | 2129.70 | 2026-01-19 10:55:00 | 2135.99 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-20 09:30:00 | 2113.80 | 2026-01-20 09:35:00 | 2105.25 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-20 09:30:00 | 2113.80 | 2026-01-20 10:00:00 | 2113.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 10:55:00 | 2013.00 | 2026-01-21 11:00:00 | 2022.22 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-06 10:10:00 | 2213.70 | 2026-02-06 10:15:00 | 2220.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-10 09:45:00 | 2242.40 | 2026-02-10 09:55:00 | 2233.69 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-10 09:45:00 | 2242.40 | 2026-02-10 10:00:00 | 2242.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:35:00 | 2211.60 | 2026-02-12 11:15:00 | 2204.56 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-12 10:35:00 | 2211.60 | 2026-02-12 12:30:00 | 2211.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:40:00 | 2191.10 | 2026-02-17 11:15:00 | 2202.22 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-17 10:40:00 | 2191.10 | 2026-02-17 15:20:00 | 2244.00 | TARGET_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-02-18 10:35:00 | 2195.80 | 2026-02-18 11:00:00 | 2202.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:45:00 | 2172.80 | 2026-02-19 11:15:00 | 2164.45 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 10:45:00 | 2172.80 | 2026-02-19 11:20:00 | 2172.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 09:35:00 | 2193.00 | 2026-02-23 10:40:00 | 2185.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-25 11:15:00 | 2181.20 | 2026-02-25 11:20:00 | 2186.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-26 10:45:00 | 2214.50 | 2026-02-26 11:00:00 | 2206.84 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-26 10:45:00 | 2214.50 | 2026-02-26 11:05:00 | 2214.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2194.30 | 2026-02-27 10:15:00 | 2185.43 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2194.30 | 2026-02-27 10:25:00 | 2194.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2064.10 | 2026-03-06 11:50:00 | 2070.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 09:35:00 | 1990.20 | 2026-03-10 14:10:00 | 1998.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-12 10:35:00 | 2004.00 | 2026-03-12 10:55:00 | 2016.74 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-12 10:35:00 | 2004.00 | 2026-03-12 11:20:00 | 2004.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:30:00 | 2164.00 | 2026-04-16 09:45:00 | 2171.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-23 10:20:00 | 2258.90 | 2026-04-23 10:50:00 | 2271.43 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-23 10:20:00 | 2258.90 | 2026-04-23 15:20:00 | 2300.00 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2026-04-29 10:40:00 | 2437.90 | 2026-04-29 10:45:00 | 2428.73 | STOP_HIT | 1.00 | -0.38% |
