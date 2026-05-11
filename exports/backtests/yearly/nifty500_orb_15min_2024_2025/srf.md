# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-12-06 15:25:00 (10683 bars)
- **Last close:** 2298.45
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
| PARTIAL | 28 |
| TARGET_HIT | 15 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 48
- **Target hits / Stop hits / Partials:** 15 / 48 / 28
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 16.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 20 | 42.6% | 7 | 27 | 13 | 0.13% | 6.0% |
| BUY @ 2nd Alert (retest1) | 47 | 20 | 42.6% | 7 | 27 | 13 | 0.13% | 6.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 23 | 52.3% | 8 | 21 | 15 | 0.25% | 10.8% |
| SELL @ 2nd Alert (retest1) | 44 | 23 | 52.3% | 8 | 21 | 15 | 0.25% | 10.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 91 | 43 | 47.3% | 15 | 48 | 28 | 0.18% | 16.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:15:00 | 2217.30 | 2221.63 | 0.00 | ORB-short ORB[2220.00,2249.05] vol=1.8x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-05-14 10:20:00 | 2223.80 | 2221.72 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 2310.00 | 2297.80 | 0.00 | ORB-long ORB[2282.10,2304.95] vol=1.8x ATR=9.27 |
| Stop hit — per-position SL triggered | 2024-05-15 12:30:00 | 2300.73 | 2309.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:40:00 | 2285.80 | 2289.40 | 0.00 | ORB-short ORB[2288.50,2309.90] vol=2.0x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:50:00 | 2277.74 | 2288.19 | 0.00 | T1 1.5R @ 2277.74 |
| Target hit | 2024-05-16 15:20:00 | 2270.95 | 2271.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 2266.10 | 2277.36 | 0.00 | ORB-short ORB[2271.30,2286.35] vol=1.9x ATR=5.44 |
| Stop hit — per-position SL triggered | 2024-05-17 11:00:00 | 2271.54 | 2276.66 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:30:00 | 2292.65 | 2283.64 | 0.00 | ORB-long ORB[2265.15,2291.45] vol=2.9x ATR=6.96 |
| Stop hit — per-position SL triggered | 2024-05-21 10:40:00 | 2285.69 | 2284.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:15:00 | 2299.50 | 2306.84 | 0.00 | ORB-short ORB[2310.25,2324.90] vol=2.7x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:30:00 | 2290.55 | 2304.43 | 0.00 | T1 1.5R @ 2290.55 |
| Target hit | 2024-05-28 15:20:00 | 2282.35 | 2289.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-05-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:05:00 | 2260.40 | 2270.81 | 0.00 | ORB-short ORB[2261.05,2281.85] vol=1.7x ATR=5.91 |
| Stop hit — per-position SL triggered | 2024-05-29 10:10:00 | 2266.31 | 2270.36 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 2302.75 | 2292.84 | 0.00 | ORB-long ORB[2279.00,2299.95] vol=2.1x ATR=7.26 |
| Stop hit — per-position SL triggered | 2024-06-06 11:20:00 | 2295.49 | 2298.19 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:45:00 | 2323.75 | 2315.17 | 0.00 | ORB-long ORB[2295.60,2322.00] vol=1.9x ATR=8.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:45:00 | 2336.32 | 2322.15 | 0.00 | T1 1.5R @ 2336.32 |
| Target hit | 2024-06-10 15:20:00 | 2352.10 | 2346.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 11:05:00 | 2391.05 | 2395.77 | 0.00 | ORB-short ORB[2398.00,2409.00] vol=2.1x ATR=4.86 |
| Stop hit — per-position SL triggered | 2024-06-18 11:30:00 | 2395.91 | 2395.18 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:00:00 | 2414.80 | 2421.45 | 0.00 | ORB-short ORB[2415.80,2440.00] vol=3.8x ATR=7.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:10:00 | 2402.91 | 2419.67 | 0.00 | T1 1.5R @ 2402.91 |
| Stop hit — per-position SL triggered | 2024-06-19 11:55:00 | 2414.80 | 2413.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 2402.05 | 2389.85 | 0.00 | ORB-long ORB[2380.00,2398.80] vol=2.5x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:35:00 | 2411.75 | 2392.75 | 0.00 | T1 1.5R @ 2411.75 |
| Stop hit — per-position SL triggered | 2024-06-26 12:05:00 | 2402.05 | 2395.08 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:35:00 | 2416.20 | 2410.29 | 0.00 | ORB-long ORB[2395.00,2408.85] vol=2.6x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:40:00 | 2423.78 | 2413.16 | 0.00 | T1 1.5R @ 2423.78 |
| Target hit | 2024-06-27 10:40:00 | 2428.35 | 2432.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:00:00 | 2438.75 | 2430.40 | 0.00 | ORB-long ORB[2415.70,2430.65] vol=1.5x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:15:00 | 2448.57 | 2431.71 | 0.00 | T1 1.5R @ 2448.57 |
| Stop hit — per-position SL triggered | 2024-07-01 11:25:00 | 2438.75 | 2434.37 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:45:00 | 2428.50 | 2440.85 | 0.00 | ORB-short ORB[2430.20,2465.00] vol=4.2x ATR=8.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:15:00 | 2416.06 | 2439.40 | 0.00 | T1 1.5R @ 2416.06 |
| Target hit | 2024-07-02 15:20:00 | 2402.00 | 2408.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-07-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:30:00 | 2390.50 | 2403.02 | 0.00 | ORB-short ORB[2395.00,2407.90] vol=2.3x ATR=6.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:20:00 | 2380.21 | 2396.82 | 0.00 | T1 1.5R @ 2380.21 |
| Target hit | 2024-07-03 15:20:00 | 2382.65 | 2384.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-07-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:05:00 | 2402.45 | 2388.26 | 0.00 | ORB-long ORB[2372.80,2390.20] vol=1.6x ATR=4.97 |
| Stop hit — per-position SL triggered | 2024-07-05 11:35:00 | 2397.48 | 2393.96 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:25:00 | 2413.55 | 2401.06 | 0.00 | ORB-long ORB[2395.00,2405.95] vol=1.7x ATR=6.64 |
| Stop hit — per-position SL triggered | 2024-07-08 10:30:00 | 2406.91 | 2402.48 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 2374.40 | 2379.56 | 0.00 | ORB-short ORB[2380.00,2399.00] vol=3.0x ATR=6.09 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 2380.49 | 2379.61 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:45:00 | 2390.00 | 2398.20 | 0.00 | ORB-short ORB[2397.35,2412.95] vol=1.7x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:45:00 | 2379.53 | 2393.97 | 0.00 | T1 1.5R @ 2379.53 |
| Stop hit — per-position SL triggered | 2024-07-11 11:50:00 | 2390.00 | 2388.75 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 2401.95 | 2397.64 | 0.00 | ORB-long ORB[2390.05,2399.00] vol=3.1x ATR=4.84 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 2397.11 | 2397.72 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:40:00 | 2384.35 | 2395.49 | 0.00 | ORB-short ORB[2391.50,2405.55] vol=2.1x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:25:00 | 2377.83 | 2390.43 | 0.00 | T1 1.5R @ 2377.83 |
| Stop hit — per-position SL triggered | 2024-07-16 12:10:00 | 2384.35 | 2388.46 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:30:00 | 2380.25 | 2381.78 | 0.00 | ORB-short ORB[2385.00,2410.00] vol=1.8x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:50:00 | 2371.08 | 2380.16 | 0.00 | T1 1.5R @ 2371.08 |
| Target hit | 2024-07-19 12:10:00 | 2375.00 | 2373.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 11:00:00 | 2346.55 | 2330.52 | 0.00 | ORB-long ORB[2309.30,2335.25] vol=1.7x ATR=8.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:35:00 | 2359.31 | 2333.62 | 0.00 | T1 1.5R @ 2359.31 |
| Target hit | 2024-07-22 15:20:00 | 2368.00 | 2352.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 2391.25 | 2367.32 | 0.00 | ORB-long ORB[2347.35,2381.20] vol=1.6x ATR=9.95 |
| Stop hit — per-position SL triggered | 2024-07-23 10:35:00 | 2381.30 | 2368.32 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:05:00 | 2377.80 | 2375.72 | 0.00 | ORB-long ORB[2342.30,2365.15] vol=11.1x ATR=7.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:25:00 | 2389.24 | 2376.12 | 0.00 | T1 1.5R @ 2389.24 |
| Target hit | 2024-07-25 15:20:00 | 2400.00 | 2389.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-07-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:10:00 | 2434.15 | 2425.20 | 0.00 | ORB-long ORB[2400.00,2427.10] vol=2.0x ATR=6.47 |
| Stop hit — per-position SL triggered | 2024-07-26 11:20:00 | 2427.68 | 2426.34 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-07-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:50:00 | 2490.00 | 2476.74 | 0.00 | ORB-long ORB[2466.15,2480.90] vol=2.1x ATR=7.65 |
| Stop hit — per-position SL triggered | 2024-07-29 10:30:00 | 2482.35 | 2482.71 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:35:00 | 2600.05 | 2579.71 | 0.00 | ORB-long ORB[2557.30,2588.15] vol=2.9x ATR=10.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:40:00 | 2615.92 | 2592.83 | 0.00 | T1 1.5R @ 2615.92 |
| Target hit | 2024-07-31 12:35:00 | 2621.05 | 2635.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — BUY (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 2536.90 | 2524.01 | 0.00 | ORB-long ORB[2488.95,2524.50] vol=1.9x ATR=8.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:35:00 | 2549.92 | 2528.55 | 0.00 | T1 1.5R @ 2549.92 |
| Stop hit — per-position SL triggered | 2024-08-07 10:50:00 | 2536.90 | 2529.72 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:40:00 | 2585.85 | 2568.86 | 0.00 | ORB-long ORB[2538.00,2572.00] vol=2.3x ATR=8.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:50:00 | 2598.82 | 2577.12 | 0.00 | T1 1.5R @ 2598.82 |
| Stop hit — per-position SL triggered | 2024-08-09 10:05:00 | 2585.85 | 2581.60 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:40:00 | 2581.95 | 2556.44 | 0.00 | ORB-long ORB[2525.35,2553.65] vol=1.6x ATR=10.75 |
| Stop hit — per-position SL triggered | 2024-08-12 09:45:00 | 2571.20 | 2559.94 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 2462.45 | 2467.96 | 0.00 | ORB-short ORB[2466.05,2487.90] vol=2.0x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 2467.41 | 2464.48 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:45:00 | 2462.10 | 2469.91 | 0.00 | ORB-short ORB[2465.00,2487.30] vol=1.9x ATR=5.34 |
| Stop hit — per-position SL triggered | 2024-08-21 11:00:00 | 2467.44 | 2469.22 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 2510.00 | 2509.66 | 0.00 | ORB-long ORB[2486.65,2507.95] vol=4.9x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:55:00 | 2518.26 | 2510.39 | 0.00 | T1 1.5R @ 2518.26 |
| Target hit | 2024-08-26 10:40:00 | 2511.05 | 2511.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 2582.15 | 2564.91 | 0.00 | ORB-long ORB[2545.30,2575.00] vol=2.4x ATR=8.55 |
| Stop hit — per-position SL triggered | 2024-08-27 09:35:00 | 2573.60 | 2566.61 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 2526.50 | 2532.78 | 0.00 | ORB-short ORB[2526.60,2538.00] vol=1.9x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:35:00 | 2519.09 | 2529.46 | 0.00 | T1 1.5R @ 2519.09 |
| Stop hit — per-position SL triggered | 2024-08-29 14:50:00 | 2526.50 | 2522.22 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 2568.75 | 2557.34 | 0.00 | ORB-long ORB[2539.00,2559.60] vol=2.6x ATR=7.54 |
| Stop hit — per-position SL triggered | 2024-08-30 09:35:00 | 2561.21 | 2558.15 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 2615.00 | 2597.78 | 0.00 | ORB-long ORB[2575.45,2604.50] vol=3.8x ATR=8.36 |
| Stop hit — per-position SL triggered | 2024-09-03 09:35:00 | 2606.64 | 2601.23 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 2623.35 | 2614.01 | 0.00 | ORB-long ORB[2603.00,2613.70] vol=2.6x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:40:00 | 2631.59 | 2623.78 | 0.00 | T1 1.5R @ 2631.59 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 2623.35 | 2625.26 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:00:00 | 2554.55 | 2581.82 | 0.00 | ORB-short ORB[2606.05,2618.55] vol=2.1x ATR=8.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:15:00 | 2541.15 | 2573.06 | 0.00 | T1 1.5R @ 2541.15 |
| Target hit | 2024-09-06 15:20:00 | 2507.45 | 2537.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-09-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:35:00 | 2562.55 | 2555.61 | 0.00 | ORB-long ORB[2525.00,2551.75] vol=4.8x ATR=6.03 |
| Stop hit — per-position SL triggered | 2024-09-10 10:40:00 | 2556.52 | 2555.70 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:45:00 | 2423.05 | 2431.40 | 0.00 | ORB-short ORB[2436.60,2460.65] vol=1.9x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 2428.80 | 2427.62 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-09-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:25:00 | 2377.10 | 2389.86 | 0.00 | ORB-short ORB[2391.30,2411.15] vol=2.2x ATR=7.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:40:00 | 2366.10 | 2387.70 | 0.00 | T1 1.5R @ 2366.10 |
| Stop hit — per-position SL triggered | 2024-09-19 10:45:00 | 2377.10 | 2387.14 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 2485.00 | 2476.19 | 0.00 | ORB-long ORB[2450.15,2481.70] vol=3.7x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:10:00 | 2493.52 | 2477.66 | 0.00 | T1 1.5R @ 2493.52 |
| Stop hit — per-position SL triggered | 2024-09-27 11:45:00 | 2485.00 | 2479.04 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:50:00 | 2335.70 | 2350.82 | 0.00 | ORB-short ORB[2350.00,2366.95] vol=2.2x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:00:00 | 2323.11 | 2347.12 | 0.00 | T1 1.5R @ 2323.11 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 2335.70 | 2332.03 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:45:00 | 2351.30 | 2339.10 | 0.00 | ORB-long ORB[2321.60,2342.90] vol=2.2x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-10-09 11:40:00 | 2344.76 | 2340.93 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 2352.80 | 2343.96 | 0.00 | ORB-long ORB[2336.40,2349.30] vol=1.7x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-10-16 09:45:00 | 2347.21 | 2345.53 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:40:00 | 2282.00 | 2277.25 | 0.00 | ORB-long ORB[2259.95,2279.00] vol=1.8x ATR=9.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 11:00:00 | 2295.71 | 2279.43 | 0.00 | T1 1.5R @ 2295.71 |
| Target hit | 2024-10-18 15:20:00 | 2325.30 | 2305.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-10-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:40:00 | 2288.85 | 2301.09 | 0.00 | ORB-short ORB[2311.05,2336.00] vol=1.6x ATR=8.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 10:45:00 | 2275.39 | 2299.05 | 0.00 | T1 1.5R @ 2275.39 |
| Stop hit — per-position SL triggered | 2024-10-21 11:25:00 | 2288.85 | 2296.64 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 11:00:00 | 2246.05 | 2265.38 | 0.00 | ORB-short ORB[2258.05,2287.75] vol=1.8x ATR=9.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:35:00 | 2231.37 | 2259.68 | 0.00 | T1 1.5R @ 2231.37 |
| Target hit | 2024-10-22 15:20:00 | 2187.05 | 2198.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 2229.10 | 2239.48 | 0.00 | ORB-short ORB[2232.60,2257.55] vol=1.5x ATR=10.77 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 2239.87 | 2238.94 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-10-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:40:00 | 2279.40 | 2263.57 | 0.00 | ORB-long ORB[2244.15,2262.70] vol=1.9x ATR=7.43 |
| Stop hit — per-position SL triggered | 2024-10-30 11:00:00 | 2271.97 | 2265.26 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:35:00 | 2231.75 | 2239.47 | 0.00 | ORB-short ORB[2240.35,2267.20] vol=3.1x ATR=7.17 |
| Stop hit — per-position SL triggered | 2024-10-31 09:40:00 | 2238.92 | 2239.80 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:55:00 | 2373.20 | 2364.59 | 0.00 | ORB-long ORB[2336.55,2367.70] vol=2.7x ATR=8.16 |
| Stop hit — per-position SL triggered | 2024-11-07 10:00:00 | 2365.04 | 2364.63 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:40:00 | 2288.50 | 2295.77 | 0.00 | ORB-short ORB[2291.60,2308.95] vol=1.5x ATR=6.53 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 2295.03 | 2294.57 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 2207.00 | 2223.19 | 0.00 | ORB-short ORB[2226.40,2247.00] vol=2.0x ATR=9.13 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 2216.13 | 2223.26 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:40:00 | 2214.70 | 2227.98 | 0.00 | ORB-short ORB[2223.00,2242.30] vol=1.5x ATR=6.80 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 2221.50 | 2227.40 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-11-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 11:00:00 | 2220.40 | 2205.20 | 0.00 | ORB-long ORB[2185.00,2199.00] vol=2.9x ATR=5.69 |
| Stop hit — per-position SL triggered | 2024-11-19 12:45:00 | 2214.71 | 2209.43 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:55:00 | 2248.10 | 2237.11 | 0.00 | ORB-long ORB[2223.60,2246.40] vol=2.7x ATR=6.29 |
| Stop hit — per-position SL triggered | 2024-11-26 11:50:00 | 2241.81 | 2241.09 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 2329.05 | 2322.35 | 0.00 | ORB-long ORB[2311.30,2324.95] vol=2.8x ATR=5.80 |
| Stop hit — per-position SL triggered | 2024-12-04 09:50:00 | 2323.25 | 2324.56 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 2313.10 | 2318.36 | 0.00 | ORB-short ORB[2318.25,2338.05] vol=1.6x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-12-05 12:00:00 | 2319.16 | 2316.94 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:05:00 | 2311.00 | 2315.53 | 0.00 | ORB-short ORB[2315.00,2336.00] vol=2.6x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:25:00 | 2303.43 | 2314.69 | 0.00 | T1 1.5R @ 2303.43 |
| Target hit | 2024-12-06 15:20:00 | 2296.15 | 2299.29 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:15:00 | 2217.30 | 2024-05-14 10:20:00 | 2223.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-15 09:35:00 | 2310.00 | 2024-05-15 12:30:00 | 2300.73 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-16 10:40:00 | 2285.80 | 2024-05-16 10:50:00 | 2277.74 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-16 10:40:00 | 2285.80 | 2024-05-16 15:20:00 | 2270.95 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-05-17 10:50:00 | 2266.10 | 2024-05-17 11:00:00 | 2271.54 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-21 10:30:00 | 2292.65 | 2024-05-21 10:40:00 | 2285.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-28 10:15:00 | 2299.50 | 2024-05-28 10:30:00 | 2290.55 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-28 10:15:00 | 2299.50 | 2024-05-28 15:20:00 | 2282.35 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-05-29 10:05:00 | 2260.40 | 2024-05-29 10:10:00 | 2266.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-06 10:05:00 | 2302.75 | 2024-06-06 11:20:00 | 2295.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-10 09:45:00 | 2323.75 | 2024-06-10 10:45:00 | 2336.32 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-10 09:45:00 | 2323.75 | 2024-06-10 15:20:00 | 2352.10 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2024-06-18 11:05:00 | 2391.05 | 2024-06-18 11:30:00 | 2395.91 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-06-19 10:00:00 | 2414.80 | 2024-06-19 10:10:00 | 2402.91 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-06-19 10:00:00 | 2414.80 | 2024-06-19 11:55:00 | 2414.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 11:00:00 | 2402.05 | 2024-06-26 11:35:00 | 2411.75 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-26 11:00:00 | 2402.05 | 2024-06-26 12:05:00 | 2402.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 09:35:00 | 2416.20 | 2024-06-27 09:40:00 | 2423.78 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-06-27 09:35:00 | 2416.20 | 2024-06-27 10:40:00 | 2428.35 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-01 11:00:00 | 2438.75 | 2024-07-01 11:15:00 | 2448.57 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-01 11:00:00 | 2438.75 | 2024-07-01 11:25:00 | 2438.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 10:45:00 | 2428.50 | 2024-07-02 11:15:00 | 2416.06 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-02 10:45:00 | 2428.50 | 2024-07-02 15:20:00 | 2402.00 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2024-07-03 10:30:00 | 2390.50 | 2024-07-03 11:20:00 | 2380.21 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-03 10:30:00 | 2390.50 | 2024-07-03 15:20:00 | 2382.65 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2024-07-05 11:05:00 | 2402.45 | 2024-07-05 11:35:00 | 2397.48 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-08 10:25:00 | 2413.55 | 2024-07-08 10:30:00 | 2406.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-10 10:10:00 | 2374.40 | 2024-07-10 10:15:00 | 2380.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-11 09:45:00 | 2390.00 | 2024-07-11 10:45:00 | 2379.53 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-11 09:45:00 | 2390.00 | 2024-07-11 11:50:00 | 2390.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 09:30:00 | 2401.95 | 2024-07-12 09:35:00 | 2397.11 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-16 10:40:00 | 2384.35 | 2024-07-16 11:25:00 | 2377.83 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-07-16 10:40:00 | 2384.35 | 2024-07-16 12:10:00 | 2384.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:30:00 | 2380.25 | 2024-07-19 10:50:00 | 2371.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-19 10:30:00 | 2380.25 | 2024-07-19 12:10:00 | 2375.00 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-22 11:00:00 | 2346.55 | 2024-07-22 11:35:00 | 2359.31 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-22 11:00:00 | 2346.55 | 2024-07-22 15:20:00 | 2368.00 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2024-07-23 10:30:00 | 2391.25 | 2024-07-23 10:35:00 | 2381.30 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-25 10:05:00 | 2377.80 | 2024-07-25 10:25:00 | 2389.24 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-25 10:05:00 | 2377.80 | 2024-07-25 15:20:00 | 2400.00 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-07-26 11:10:00 | 2434.15 | 2024-07-26 11:20:00 | 2427.68 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-29 09:50:00 | 2490.00 | 2024-07-29 10:30:00 | 2482.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-31 09:35:00 | 2600.05 | 2024-07-31 09:40:00 | 2615.92 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-31 09:35:00 | 2600.05 | 2024-07-31 12:35:00 | 2621.05 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2024-08-07 10:25:00 | 2536.90 | 2024-08-07 10:35:00 | 2549.92 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-07 10:25:00 | 2536.90 | 2024-08-07 10:50:00 | 2536.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 09:40:00 | 2585.85 | 2024-08-09 09:50:00 | 2598.82 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-09 09:40:00 | 2585.85 | 2024-08-09 10:05:00 | 2585.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 09:40:00 | 2581.95 | 2024-08-12 09:45:00 | 2571.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-20 09:45:00 | 2462.45 | 2024-08-20 10:15:00 | 2467.41 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-21 10:45:00 | 2462.10 | 2024-08-21 11:00:00 | 2467.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-26 09:45:00 | 2510.00 | 2024-08-26 09:55:00 | 2518.26 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-26 09:45:00 | 2510.00 | 2024-08-26 10:40:00 | 2511.05 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-08-27 09:30:00 | 2582.15 | 2024-08-27 09:35:00 | 2573.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-29 10:55:00 | 2526.50 | 2024-08-29 11:35:00 | 2519.09 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-08-29 10:55:00 | 2526.50 | 2024-08-29 14:50:00 | 2526.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 09:30:00 | 2568.75 | 2024-08-30 09:35:00 | 2561.21 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-03 09:30:00 | 2615.00 | 2024-09-03 09:35:00 | 2606.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-05 09:30:00 | 2623.35 | 2024-09-05 09:40:00 | 2631.59 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-09-05 09:30:00 | 2623.35 | 2024-09-05 09:50:00 | 2623.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:00:00 | 2554.55 | 2024-09-06 10:15:00 | 2541.15 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-06 10:00:00 | 2554.55 | 2024-09-06 15:20:00 | 2507.45 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2024-09-10 10:35:00 | 2562.55 | 2024-09-10 10:40:00 | 2556.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-17 09:45:00 | 2423.05 | 2024-09-17 10:00:00 | 2428.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-19 10:25:00 | 2377.10 | 2024-09-19 10:40:00 | 2366.10 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-19 10:25:00 | 2377.10 | 2024-09-19 10:45:00 | 2377.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 11:00:00 | 2485.00 | 2024-09-27 11:10:00 | 2493.52 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-27 11:00:00 | 2485.00 | 2024-09-27 11:45:00 | 2485.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 09:50:00 | 2335.70 | 2024-10-07 10:00:00 | 2323.11 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-07 09:50:00 | 2335.70 | 2024-10-07 11:20:00 | 2335.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:45:00 | 2351.30 | 2024-10-09 11:40:00 | 2344.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-16 09:30:00 | 2352.80 | 2024-10-16 09:45:00 | 2347.21 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-18 10:40:00 | 2282.00 | 2024-10-18 11:00:00 | 2295.71 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-10-18 10:40:00 | 2282.00 | 2024-10-18 15:20:00 | 2325.30 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2024-10-21 10:40:00 | 2288.85 | 2024-10-21 10:45:00 | 2275.39 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-10-21 10:40:00 | 2288.85 | 2024-10-21 11:25:00 | 2288.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 11:00:00 | 2246.05 | 2024-10-22 11:35:00 | 2231.37 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-22 11:00:00 | 2246.05 | 2024-10-22 15:20:00 | 2187.05 | TARGET_HIT | 0.50 | 2.63% |
| SELL | retest1 | 2024-10-25 10:55:00 | 2229.10 | 2024-10-25 11:15:00 | 2239.87 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-10-30 10:40:00 | 2279.40 | 2024-10-30 11:00:00 | 2271.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-31 09:35:00 | 2231.75 | 2024-10-31 09:40:00 | 2238.92 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-07 09:55:00 | 2373.20 | 2024-11-07 10:00:00 | 2365.04 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-12 10:40:00 | 2288.50 | 2024-11-12 11:20:00 | 2295.03 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-13 09:45:00 | 2207.00 | 2024-11-13 09:50:00 | 2216.13 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-11-18 09:40:00 | 2214.70 | 2024-11-18 09:45:00 | 2221.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-19 11:00:00 | 2220.40 | 2024-11-19 12:45:00 | 2214.71 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-11-26 10:55:00 | 2248.10 | 2024-11-26 11:50:00 | 2241.81 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-04 09:40:00 | 2329.05 | 2024-12-04 09:50:00 | 2323.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2313.10 | 2024-12-05 12:00:00 | 2319.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-06 11:05:00 | 2311.00 | 2024-12-06 11:25:00 | 2303.43 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-06 11:05:00 | 2311.00 | 2024-12-06 15:20:00 | 2296.15 | TARGET_HIT | 0.50 | 0.64% |
