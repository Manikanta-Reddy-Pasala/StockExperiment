# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-04-24 15:25:00 (53255 bars)
- **Last close:** 2281.60
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
| ENTRY1 | 102 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 8 |
| STOP_HIT | 94 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 134 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 94
- **Target hits / Stop hits / Partials:** 8 / 94 / 32
- **Avg / median % per leg:** 0.01% / -0.19%
- **Sum % (uncompounded):** 1.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 26 | 34.2% | 5 | 50 | 21 | 0.05% | 3.9% |
| BUY @ 2nd Alert (retest1) | 76 | 26 | 34.2% | 5 | 50 | 21 | 0.05% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 14 | 24.1% | 3 | 44 | 11 | -0.04% | -2.4% |
| SELL @ 2nd Alert (retest1) | 58 | 14 | 24.1% | 3 | 44 | 11 | -0.04% | -2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 134 | 40 | 29.9% | 8 | 94 | 32 | 0.01% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:05:00 | 1826.17 | 1833.98 | 0.00 | ORB-short ORB[1827.48,1850.84] vol=1.9x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 11:15:00 | 1818.16 | 1831.59 | 0.00 | T1 1.5R @ 1818.16 |
| Stop hit — per-position SL triggered | 2023-05-17 12:10:00 | 1826.17 | 1826.63 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 11:00:00 | 1841.63 | 1846.64 | 0.00 | ORB-short ORB[1842.99,1865.29] vol=2.8x ATR=5.39 |
| Stop hit — per-position SL triggered | 2023-05-18 11:05:00 | 1847.02 | 1846.61 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 10:20:00 | 2455.32 | 2481.43 | 0.00 | ORB-short ORB[2477.03,2504.66] vol=1.6x ATR=10.81 |
| Stop hit — per-position SL triggered | 2023-05-29 10:55:00 | 2466.13 | 2477.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 11:00:00 | 2442.52 | 2460.52 | 0.00 | ORB-short ORB[2450.86,2470.34] vol=2.7x ATR=7.05 |
| Stop hit — per-position SL triggered | 2023-05-30 11:45:00 | 2449.57 | 2457.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:05:00 | 2423.71 | 2438.37 | 0.00 | ORB-short ORB[2433.41,2461.43] vol=1.5x ATR=6.31 |
| Stop hit — per-position SL triggered | 2023-06-02 11:50:00 | 2430.02 | 2436.42 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:55:00 | 2405.29 | 2390.49 | 0.00 | ORB-long ORB[2371.41,2398.99] vol=2.1x ATR=10.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 10:05:00 | 2421.67 | 2397.70 | 0.00 | T1 1.5R @ 2421.67 |
| Stop hit — per-position SL triggered | 2023-06-05 12:05:00 | 2405.29 | 2405.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 2391.72 | 2382.84 | 0.00 | ORB-long ORB[2365.11,2384.93] vol=3.2x ATR=7.08 |
| Stop hit — per-position SL triggered | 2023-06-08 10:05:00 | 2384.64 | 2386.39 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 11:00:00 | 2367.14 | 2379.02 | 0.00 | ORB-short ORB[2371.36,2399.33] vol=2.5x ATR=6.07 |
| Stop hit — per-position SL triggered | 2023-06-12 11:10:00 | 2373.21 | 2378.62 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:35:00 | 2417.90 | 2404.83 | 0.00 | ORB-long ORB[2387.74,2414.02] vol=3.1x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 09:45:00 | 2428.93 | 2413.28 | 0.00 | T1 1.5R @ 2428.93 |
| Stop hit — per-position SL triggered | 2023-06-15 09:50:00 | 2417.90 | 2413.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:00:00 | 2326.42 | 2341.43 | 0.00 | ORB-short ORB[2343.44,2365.54] vol=1.5x ATR=6.22 |
| Stop hit — per-position SL triggered | 2023-06-21 15:20:00 | 2336.36 | 2329.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-06-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:10:00 | 2367.48 | 2350.14 | 0.00 | ORB-long ORB[2330.11,2361.42] vol=2.7x ATR=7.32 |
| Stop hit — per-position SL triggered | 2023-06-22 10:15:00 | 2360.16 | 2354.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:35:00 | 2341.69 | 2324.93 | 0.00 | ORB-long ORB[2314.16,2337.91] vol=2.0x ATR=8.05 |
| Stop hit — per-position SL triggered | 2023-07-03 10:40:00 | 2333.64 | 2325.61 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:10:00 | 2328.70 | 2314.63 | 0.00 | ORB-long ORB[2300.64,2325.36] vol=2.5x ATR=6.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 10:15:00 | 2338.97 | 2317.98 | 0.00 | T1 1.5R @ 2338.97 |
| Stop hit — per-position SL triggered | 2023-07-04 10:30:00 | 2328.70 | 2321.51 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:50:00 | 2313.19 | 2324.58 | 0.00 | ORB-short ORB[2313.92,2331.61] vol=1.8x ATR=5.38 |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 2318.57 | 2323.42 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:55:00 | 2323.86 | 2303.24 | 0.00 | ORB-long ORB[2289.05,2319.78] vol=3.9x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:00:00 | 2335.24 | 2305.96 | 0.00 | T1 1.5R @ 2335.24 |
| Stop hit — per-position SL triggered | 2023-07-10 11:05:00 | 2323.86 | 2307.50 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:50:00 | 2324.68 | 2339.42 | 0.00 | ORB-short ORB[2331.71,2357.79] vol=1.8x ATR=7.43 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 2332.11 | 2338.85 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 10:20:00 | 2270.05 | 2288.90 | 0.00 | ORB-short ORB[2284.11,2306.60] vol=1.8x ATR=6.99 |
| Stop hit — per-position SL triggered | 2023-07-14 10:25:00 | 2277.04 | 2287.72 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:40:00 | 2352.79 | 2345.05 | 0.00 | ORB-long ORB[2335.05,2351.00] vol=2.4x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:55:00 | 2363.20 | 2348.04 | 0.00 | T1 1.5R @ 2363.20 |
| Target hit | 2023-07-18 11:10:00 | 2370.20 | 2387.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 2332.68 | 2344.80 | 0.00 | ORB-short ORB[2341.31,2354.88] vol=3.6x ATR=5.60 |
| Stop hit — per-position SL triggered | 2023-07-20 09:55:00 | 2338.28 | 2341.59 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:30:00 | 2352.94 | 2342.69 | 0.00 | ORB-long ORB[2322.21,2349.01] vol=2.1x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:45:00 | 2361.85 | 2348.86 | 0.00 | T1 1.5R @ 2361.85 |
| Stop hit — per-position SL triggered | 2023-07-21 10:10:00 | 2352.94 | 2351.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:00:00 | 2356.04 | 2348.39 | 0.00 | ORB-long ORB[2338.45,2355.85] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2023-07-24 10:15:00 | 2350.34 | 2349.40 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:40:00 | 2356.82 | 2353.36 | 0.00 | ORB-long ORB[2346.15,2356.33] vol=2.2x ATR=5.37 |
| Stop hit — per-position SL triggered | 2023-07-25 09:50:00 | 2351.45 | 2353.82 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:55:00 | 2386.68 | 2374.42 | 0.00 | ORB-long ORB[2356.91,2377.86] vol=2.0x ATR=8.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:15:00 | 2399.22 | 2382.01 | 0.00 | T1 1.5R @ 2399.22 |
| Stop hit — per-position SL triggered | 2023-07-28 10:30:00 | 2386.68 | 2383.12 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:10:00 | 2412.08 | 2404.00 | 0.00 | ORB-long ORB[2392.69,2409.17] vol=2.0x ATR=8.01 |
| Stop hit — per-position SL triggered | 2023-07-31 10:45:00 | 2404.07 | 2408.13 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:30:00 | 2465.30 | 2454.87 | 0.00 | ORB-long ORB[2433.41,2464.43] vol=1.7x ATR=7.70 |
| Stop hit — per-position SL triggered | 2023-08-10 09:35:00 | 2457.60 | 2455.07 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 10:45:00 | 2382.75 | 2369.34 | 0.00 | ORB-long ORB[2351.00,2376.98] vol=1.6x ATR=8.58 |
| Stop hit — per-position SL triggered | 2023-08-16 10:50:00 | 2374.17 | 2369.72 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:10:00 | 2389.34 | 2384.53 | 0.00 | ORB-long ORB[2367.53,2385.37] vol=1.6x ATR=6.66 |
| Stop hit — per-position SL triggered | 2023-08-17 10:25:00 | 2382.68 | 2385.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 11:10:00 | 2451.68 | 2437.37 | 0.00 | ORB-long ORB[2416.34,2450.23] vol=1.8x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 11:35:00 | 2461.25 | 2441.60 | 0.00 | T1 1.5R @ 2461.25 |
| Stop hit — per-position SL triggered | 2023-08-30 13:20:00 | 2451.68 | 2448.88 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 2393.95 | 2398.14 | 0.00 | ORB-short ORB[2397.54,2412.03] vol=1.9x ATR=5.95 |
| Stop hit — per-position SL triggered | 2023-09-05 09:50:00 | 2399.90 | 2397.88 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:55:00 | 2422.84 | 2414.32 | 0.00 | ORB-long ORB[2397.54,2422.79] vol=1.8x ATR=6.07 |
| Stop hit — per-position SL triggered | 2023-09-06 10:30:00 | 2416.77 | 2417.72 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:35:00 | 2443.01 | 2437.69 | 0.00 | ORB-long ORB[2429.82,2442.13] vol=1.6x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 11:35:00 | 2450.33 | 2441.91 | 0.00 | T1 1.5R @ 2450.33 |
| Target hit | 2023-09-08 14:55:00 | 2446.98 | 2449.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2023-09-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:00:00 | 2460.26 | 2447.52 | 0.00 | ORB-long ORB[2437.48,2453.28] vol=2.0x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 10:35:00 | 2469.63 | 2456.29 | 0.00 | T1 1.5R @ 2469.63 |
| Stop hit — per-position SL triggered | 2023-09-15 10:50:00 | 2460.26 | 2457.45 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 09:30:00 | 2448.43 | 2440.62 | 0.00 | ORB-long ORB[2415.76,2446.93] vol=3.0x ATR=7.85 |
| Stop hit — per-position SL triggered | 2023-09-20 09:35:00 | 2440.58 | 2440.67 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 2414.26 | 2402.60 | 0.00 | ORB-long ORB[2385.90,2414.02] vol=1.6x ATR=7.05 |
| Stop hit — per-position SL triggered | 2023-09-21 10:05:00 | 2407.21 | 2408.20 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:20:00 | 2418.67 | 2408.79 | 0.00 | ORB-long ORB[2399.57,2414.99] vol=1.7x ATR=5.93 |
| Stop hit — per-position SL triggered | 2023-09-26 10:30:00 | 2412.74 | 2409.94 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 11:15:00 | 2407.04 | 2386.37 | 0.00 | ORB-long ORB[2370.49,2391.86] vol=2.1x ATR=4.87 |
| Stop hit — per-position SL triggered | 2023-09-27 11:25:00 | 2402.17 | 2389.15 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 10:10:00 | 2428.66 | 2418.26 | 0.00 | ORB-long ORB[2404.32,2419.79] vol=2.0x ATR=5.45 |
| Stop hit — per-position SL triggered | 2023-09-28 10:25:00 | 2423.21 | 2420.21 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-29 10:35:00 | 2353.86 | 2375.17 | 0.00 | ORB-short ORB[2386.92,2403.89] vol=1.8x ATR=7.67 |
| Stop hit — per-position SL triggered | 2023-09-29 10:50:00 | 2361.53 | 2372.64 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-10-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 10:45:00 | 2356.58 | 2334.03 | 0.00 | ORB-long ORB[2319.25,2341.31] vol=3.7x ATR=8.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 10:55:00 | 2368.80 | 2338.42 | 0.00 | T1 1.5R @ 2368.80 |
| Target hit | 2023-10-04 15:20:00 | 2389.64 | 2376.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:30:00 | 2408.20 | 2395.44 | 0.00 | ORB-long ORB[2368.45,2403.11] vol=2.3x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 11:05:00 | 2420.87 | 2405.97 | 0.00 | T1 1.5R @ 2420.87 |
| Stop hit — per-position SL triggered | 2023-10-10 11:50:00 | 2408.20 | 2406.76 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 09:45:00 | 2362.00 | 2391.19 | 0.00 | ORB-short ORB[2391.23,2418.87] vol=2.5x ATR=10.07 |
| Stop hit — per-position SL triggered | 2023-10-13 13:50:00 | 2372.07 | 2368.31 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:35:00 | 2385.18 | 2377.38 | 0.00 | ORB-long ORB[2365.88,2384.88] vol=1.7x ATR=5.13 |
| Stop hit — per-position SL triggered | 2023-10-17 10:40:00 | 2380.05 | 2377.56 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 2341.31 | 2355.81 | 0.00 | ORB-short ORB[2352.31,2367.24] vol=4.3x ATR=4.34 |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 2345.65 | 2355.48 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 09:35:00 | 2316.34 | 2325.17 | 0.00 | ORB-short ORB[2321.92,2338.74] vol=2.1x ATR=5.41 |
| Stop hit — per-position SL triggered | 2023-10-23 09:40:00 | 2321.75 | 2324.25 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 11:15:00 | 2215.42 | 2211.32 | 0.00 | ORB-long ORB[2181.34,2207.37] vol=1.8x ATR=7.21 |
| Stop hit — per-position SL triggered | 2023-10-30 11:45:00 | 2208.21 | 2211.61 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:30:00 | 2218.23 | 2232.15 | 0.00 | ORB-short ORB[2225.40,2249.21] vol=2.1x ATR=6.99 |
| Stop hit — per-position SL triggered | 2023-10-31 09:35:00 | 2225.22 | 2231.51 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:00:00 | 2216.19 | 2225.20 | 0.00 | ORB-short ORB[2217.75,2234.47] vol=1.6x ATR=6.21 |
| Stop hit — per-position SL triggered | 2023-11-01 10:05:00 | 2222.40 | 2224.85 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 11:10:00 | 2169.22 | 2180.48 | 0.00 | ORB-short ORB[2173.68,2190.07] vol=3.0x ATR=4.91 |
| Stop hit — per-position SL triggered | 2023-11-06 12:55:00 | 2174.13 | 2177.10 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:20:00 | 2190.41 | 2182.69 | 0.00 | ORB-long ORB[2174.89,2188.56] vol=1.5x ATR=4.27 |
| Stop hit — per-position SL triggered | 2023-11-07 10:25:00 | 2186.14 | 2182.88 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:50:00 | 2187.35 | 2179.74 | 0.00 | ORB-long ORB[2166.80,2186.14] vol=2.1x ATR=4.99 |
| Stop hit — per-position SL triggered | 2023-11-08 10:10:00 | 2182.36 | 2180.68 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:15:00 | 2134.32 | 2142.37 | 0.00 | ORB-short ORB[2138.73,2155.84] vol=1.8x ATR=4.84 |
| Stop hit — per-position SL triggered | 2023-11-13 12:20:00 | 2139.16 | 2137.98 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-15 11:10:00 | 2155.21 | 2165.62 | 0.00 | ORB-short ORB[2161.95,2181.29] vol=2.3x ATR=3.86 |
| Stop hit — per-position SL triggered | 2023-11-15 13:15:00 | 2159.07 | 2161.65 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 09:30:00 | 2131.46 | 2145.04 | 0.00 | ORB-short ORB[2141.45,2164.67] vol=3.7x ATR=4.94 |
| Stop hit — per-position SL triggered | 2023-11-16 09:35:00 | 2136.40 | 2142.43 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:50:00 | 2159.04 | 2150.54 | 0.00 | ORB-long ORB[2133.84,2154.34] vol=1.8x ATR=5.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 10:20:00 | 2167.97 | 2154.71 | 0.00 | T1 1.5R @ 2167.97 |
| Stop hit — per-position SL triggered | 2023-11-17 10:45:00 | 2159.04 | 2155.47 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:30:00 | 2132.29 | 2138.98 | 0.00 | ORB-short ORB[2135.10,2150.80] vol=2.9x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 09:45:00 | 2125.24 | 2132.55 | 0.00 | T1 1.5R @ 2125.24 |
| Stop hit — per-position SL triggered | 2023-11-20 10:10:00 | 2132.29 | 2130.21 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:00:00 | 2100.15 | 2113.00 | 0.00 | ORB-short ORB[2109.45,2123.17] vol=2.4x ATR=5.04 |
| Stop hit — per-position SL triggered | 2023-11-23 12:20:00 | 2105.19 | 2110.56 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 2119.25 | 2113.08 | 0.00 | ORB-long ORB[2103.78,2117.69] vol=2.2x ATR=4.55 |
| Stop hit — per-position SL triggered | 2023-11-24 09:40:00 | 2114.70 | 2113.79 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-11-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 10:40:00 | 2332.82 | 2368.34 | 0.00 | ORB-short ORB[2367.48,2402.04] vol=2.0x ATR=12.35 |
| Stop hit — per-position SL triggered | 2023-11-29 10:50:00 | 2345.17 | 2365.86 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:25:00 | 2320.95 | 2302.38 | 0.00 | ORB-long ORB[2295.01,2314.16] vol=2.6x ATR=10.05 |
| Stop hit — per-position SL triggered | 2023-12-01 10:30:00 | 2310.90 | 2302.98 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 11:00:00 | 2769.48 | 2747.79 | 0.00 | ORB-long ORB[2730.07,2766.91] vol=2.2x ATR=12.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 11:05:00 | 2788.50 | 2763.57 | 0.00 | T1 1.5R @ 2788.50 |
| Target hit | 2023-12-11 13:45:00 | 2781.74 | 2782.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 2760.85 | 2772.72 | 0.00 | ORB-short ORB[2765.94,2790.86] vol=1.8x ATR=8.30 |
| Stop hit — per-position SL triggered | 2023-12-13 09:40:00 | 2769.15 | 2770.71 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:45:00 | 2840.64 | 2818.17 | 0.00 | ORB-long ORB[2804.72,2833.80] vol=3.4x ATR=10.45 |
| Stop hit — per-position SL triggered | 2023-12-14 10:50:00 | 2830.19 | 2819.22 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:55:00 | 2852.22 | 2827.18 | 0.00 | ORB-long ORB[2805.59,2825.03] vol=4.1x ATR=9.30 |
| Stop hit — per-position SL triggered | 2023-12-15 10:05:00 | 2842.92 | 2836.84 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 2854.50 | 2877.75 | 0.00 | ORB-short ORB[2871.32,2902.98] vol=3.5x ATR=8.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 10:05:00 | 2841.30 | 2865.83 | 0.00 | T1 1.5R @ 2841.30 |
| Stop hit — per-position SL triggered | 2023-12-19 10:20:00 | 2854.50 | 2861.32 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:55:00 | 2771.76 | 2744.90 | 0.00 | ORB-long ORB[2714.65,2748.49] vol=2.5x ATR=9.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 10:05:00 | 2786.15 | 2759.42 | 0.00 | T1 1.5R @ 2786.15 |
| Stop hit — per-position SL triggered | 2023-12-26 11:00:00 | 2771.76 | 2769.50 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 2767.83 | 2784.43 | 0.00 | ORB-short ORB[2772.73,2806.47] vol=1.6x ATR=10.21 |
| Stop hit — per-position SL triggered | 2023-12-27 09:35:00 | 2778.04 | 2783.56 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 11:15:00 | 2742.53 | 2754.91 | 0.00 | ORB-short ORB[2763.03,2779.85] vol=2.2x ATR=5.71 |
| Stop hit — per-position SL triggered | 2023-12-28 11:35:00 | 2748.24 | 2753.56 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 11:00:00 | 2760.75 | 2740.63 | 0.00 | ORB-long ORB[2717.32,2743.16] vol=2.0x ATR=8.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:15:00 | 2774.10 | 2743.96 | 0.00 | T1 1.5R @ 2774.10 |
| Stop hit — per-position SL triggered | 2023-12-29 11:25:00 | 2760.75 | 2745.85 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:35:00 | 2798.81 | 2771.98 | 0.00 | ORB-long ORB[2762.01,2777.53] vol=4.6x ATR=7.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 10:40:00 | 2810.50 | 2783.01 | 0.00 | T1 1.5R @ 2810.50 |
| Target hit | 2024-01-01 15:10:00 | 2822.17 | 2825.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — SELL (started 2024-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:40:00 | 2770.40 | 2798.41 | 0.00 | ORB-short ORB[2800.75,2832.84] vol=1.6x ATR=11.34 |
| Stop hit — per-position SL triggered | 2024-01-02 10:50:00 | 2781.74 | 2796.85 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:55:00 | 2941.61 | 2923.21 | 0.00 | ORB-long ORB[2904.72,2932.69] vol=2.7x ATR=10.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:15:00 | 2957.00 | 2933.45 | 0.00 | T1 1.5R @ 2957.00 |
| Stop hit — per-position SL triggered | 2024-01-05 10:30:00 | 2941.61 | 2934.81 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:00:00 | 2958.24 | 2926.16 | 0.00 | ORB-long ORB[2889.07,2921.06] vol=5.6x ATR=13.14 |
| Stop hit — per-position SL triggered | 2024-01-09 10:45:00 | 2945.10 | 2940.33 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:35:00 | 2966.62 | 2944.49 | 0.00 | ORB-long ORB[2927.89,2953.05] vol=3.0x ATR=11.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 09:40:00 | 2983.88 | 2960.02 | 0.00 | T1 1.5R @ 2983.88 |
| Stop hit — per-position SL triggered | 2024-01-10 09:50:00 | 2966.62 | 2962.38 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 09:55:00 | 2985.04 | 2994.43 | 0.00 | ORB-short ORB[2992.02,3009.77] vol=1.8x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:15:00 | 2975.78 | 2991.00 | 0.00 | T1 1.5R @ 2975.78 |
| Target hit | 2024-01-16 15:20:00 | 2962.75 | 2972.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 2859.98 | 2880.16 | 0.00 | ORB-short ORB[2866.96,2893.91] vol=2.1x ATR=11.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 2843.18 | 2872.46 | 0.00 | T1 1.5R @ 2843.18 |
| Target hit | 2024-01-18 11:30:00 | 2839.52 | 2839.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2024-01-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:00:00 | 2814.61 | 2832.31 | 0.00 | ORB-short ORB[2834.10,2850.29] vol=1.5x ATR=7.94 |
| Stop hit — per-position SL triggered | 2024-01-20 10:15:00 | 2822.55 | 2827.87 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:35:00 | 2764.29 | 2799.01 | 0.00 | ORB-short ORB[2810.10,2837.05] vol=2.3x ATR=10.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:50:00 | 2748.00 | 2794.07 | 0.00 | T1 1.5R @ 2748.00 |
| Stop hit — per-position SL triggered | 2024-01-25 13:40:00 | 2764.29 | 2763.04 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:05:00 | 3030.76 | 3039.27 | 0.00 | ORB-short ORB[3031.58,3060.62] vol=3.1x ATR=9.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 11:15:00 | 3016.74 | 3036.37 | 0.00 | T1 1.5R @ 3016.74 |
| Stop hit — per-position SL triggered | 2024-02-01 11:40:00 | 3030.76 | 3033.65 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 10:25:00 | 3099.44 | 3080.74 | 0.00 | ORB-long ORB[3055.28,3085.53] vol=2.1x ATR=10.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 10:45:00 | 3115.68 | 3090.56 | 0.00 | T1 1.5R @ 3115.68 |
| Stop hit — per-position SL triggered | 2024-02-05 12:20:00 | 3099.44 | 3098.64 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:40:00 | 3123.15 | 3085.33 | 0.00 | ORB-long ORB[3063.57,3092.03] vol=4.1x ATR=11.31 |
| Stop hit — per-position SL triggered | 2024-02-06 11:00:00 | 3111.84 | 3094.79 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:55:00 | 3149.76 | 3127.30 | 0.00 | ORB-long ORB[3114.96,3141.13] vol=3.2x ATR=10.34 |
| Stop hit — per-position SL triggered | 2024-02-07 10:00:00 | 3139.42 | 3137.82 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 09:35:00 | 3105.26 | 3123.84 | 0.00 | ORB-short ORB[3118.49,3146.56] vol=1.7x ATR=9.68 |
| Stop hit — per-position SL triggered | 2024-02-08 09:50:00 | 3114.94 | 3120.89 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:45:00 | 3099.25 | 3123.32 | 0.00 | ORB-short ORB[3114.96,3143.02] vol=1.6x ATR=11.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:05:00 | 3082.49 | 3114.22 | 0.00 | T1 1.5R @ 3082.49 |
| Stop hit — per-position SL triggered | 2024-02-12 11:10:00 | 3099.25 | 3103.70 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:35:00 | 3171.14 | 3157.87 | 0.00 | ORB-long ORB[3131.44,3160.52] vol=2.1x ATR=8.55 |
| Stop hit — per-position SL triggered | 2024-02-21 10:40:00 | 3162.59 | 3158.15 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:55:00 | 3118.49 | 3130.01 | 0.00 | ORB-short ORB[3119.17,3144.86] vol=1.7x ATR=8.46 |
| Stop hit — per-position SL triggered | 2024-02-22 11:55:00 | 3126.95 | 3128.37 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 10:35:00 | 3132.65 | 3153.80 | 0.00 | ORB-short ORB[3156.06,3172.83] vol=2.1x ATR=8.34 |
| Stop hit — per-position SL triggered | 2024-02-23 10:40:00 | 3140.99 | 3153.12 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 3169.25 | 3196.58 | 0.00 | ORB-short ORB[3196.39,3217.53] vol=1.6x ATR=7.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:55:00 | 3157.42 | 3193.35 | 0.00 | T1 1.5R @ 3157.42 |
| Stop hit — per-position SL triggered | 2024-02-28 11:20:00 | 3169.25 | 3183.50 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-02-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:10:00 | 3154.70 | 3131.69 | 0.00 | ORB-long ORB[3107.54,3131.92] vol=2.2x ATR=10.19 |
| Stop hit — per-position SL triggered | 2024-02-29 11:20:00 | 3144.51 | 3132.70 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-03-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-02 09:50:00 | 3234.44 | 3229.50 | 0.00 | ORB-long ORB[3220.14,3232.26] vol=1.8x ATR=7.20 |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 3233.23 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 90 — SELL (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 3226.11 | 3238.58 | 0.00 | ORB-short ORB[3230.28,3247.77] vol=1.9x ATR=7.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:40:00 | 3214.62 | 3236.03 | 0.00 | T1 1.5R @ 3214.62 |
| Stop hit — per-position SL triggered | 2024-03-04 09:55:00 | 3226.11 | 3234.07 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 11:15:00 | 3198.33 | 3217.57 | 0.00 | ORB-short ORB[3203.76,3232.26] vol=1.7x ATR=6.37 |
| Stop hit — per-position SL triggered | 2024-03-05 11:35:00 | 3204.70 | 3216.61 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 3183.01 | 3195.18 | 0.00 | ORB-short ORB[3185.82,3221.60] vol=1.8x ATR=9.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:45:00 | 3168.66 | 3186.68 | 0.00 | T1 1.5R @ 3168.66 |
| Target hit | 2024-03-06 14:35:00 | 3127.36 | 3126.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 93 — SELL (started 2024-03-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 09:40:00 | 3015.83 | 3022.74 | 0.00 | ORB-short ORB[3019.99,3038.37] vol=1.7x ATR=7.80 |
| Stop hit — per-position SL triggered | 2024-03-28 09:55:00 | 3023.63 | 3020.95 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 11:05:00 | 3152.81 | 3134.63 | 0.00 | ORB-long ORB[3116.89,3148.98] vol=1.7x ATR=10.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 11:15:00 | 3168.23 | 3141.83 | 0.00 | T1 1.5R @ 3168.23 |
| Stop hit — per-position SL triggered | 2024-04-01 11:30:00 | 3152.81 | 3143.55 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 09:40:00 | 3163.57 | 3155.17 | 0.00 | ORB-long ORB[3141.13,3162.31] vol=3.0x ATR=9.66 |
| Stop hit — per-position SL triggered | 2024-04-04 09:50:00 | 3153.91 | 3155.57 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-04-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:35:00 | 3089.31 | 3098.97 | 0.00 | ORB-short ORB[3092.66,3119.41] vol=1.8x ATR=10.28 |
| Stop hit — per-position SL triggered | 2024-04-05 09:45:00 | 3099.59 | 3098.32 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-04-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 10:05:00 | 3041.03 | 3031.57 | 0.00 | ORB-long ORB[3020.58,3039.34] vol=2.0x ATR=7.96 |
| Stop hit — per-position SL triggered | 2024-04-18 10:15:00 | 3033.07 | 3031.79 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:30:00 | 2938.80 | 2946.88 | 0.00 | ORB-short ORB[2940.64,2959.84] vol=1.7x ATR=6.20 |
| Stop hit — per-position SL triggered | 2024-04-25 09:40:00 | 2945.00 | 2945.64 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:35:00 | 3006.57 | 2985.35 | 0.00 | ORB-long ORB[2968.81,3000.56] vol=2.5x ATR=8.93 |
| Stop hit — per-position SL triggered | 2024-04-30 09:50:00 | 2997.64 | 2990.90 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:50:00 | 3000.27 | 2984.21 | 0.00 | ORB-long ORB[2967.59,2997.65] vol=1.7x ATR=8.77 |
| Stop hit — per-position SL triggered | 2024-05-02 11:10:00 | 2991.50 | 2985.37 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:50:00 | 2756.73 | 2791.98 | 0.00 | ORB-short ORB[2779.17,2814.41] vol=1.9x ATR=9.86 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 2766.59 | 2791.07 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:10:00 | 2800.50 | 2770.67 | 0.00 | ORB-long ORB[2733.95,2770.74] vol=1.8x ATR=11.51 |
| Stop hit — per-position SL triggered | 2024-05-08 10:15:00 | 2788.99 | 2771.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 11:05:00 | 1826.17 | 2023-05-17 11:15:00 | 1818.16 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-05-17 11:05:00 | 1826.17 | 2023-05-17 12:10:00 | 1826.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-18 11:00:00 | 1841.63 | 2023-05-18 11:05:00 | 1847.02 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-05-29 10:20:00 | 2455.32 | 2023-05-29 10:55:00 | 2466.13 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-05-30 11:00:00 | 2442.52 | 2023-05-30 11:45:00 | 2449.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-02 11:05:00 | 2423.71 | 2023-06-02 11:50:00 | 2430.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-05 09:55:00 | 2405.29 | 2023-06-05 10:05:00 | 2421.67 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-06-05 09:55:00 | 2405.29 | 2023-06-05 12:05:00 | 2405.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 09:35:00 | 2391.72 | 2023-06-08 10:05:00 | 2384.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-12 11:00:00 | 2367.14 | 2023-06-12 11:10:00 | 2373.21 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-15 09:35:00 | 2417.90 | 2023-06-15 09:45:00 | 2428.93 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-06-15 09:35:00 | 2417.90 | 2023-06-15 09:50:00 | 2417.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-21 11:00:00 | 2326.42 | 2023-06-21 15:20:00 | 2336.36 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-06-22 10:10:00 | 2367.48 | 2023-06-22 10:15:00 | 2360.16 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-03 10:35:00 | 2341.69 | 2023-07-03 10:40:00 | 2333.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-04 10:10:00 | 2328.70 | 2023-07-04 10:15:00 | 2338.97 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-04 10:10:00 | 2328.70 | 2023-07-04 10:30:00 | 2328.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-07 10:50:00 | 2313.19 | 2023-07-07 11:15:00 | 2318.57 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-10 10:55:00 | 2323.86 | 2023-07-10 11:00:00 | 2335.24 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-07-10 10:55:00 | 2323.86 | 2023-07-10 11:05:00 | 2323.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-12 09:50:00 | 2324.68 | 2023-07-12 09:55:00 | 2332.11 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-07-14 10:20:00 | 2270.05 | 2023-07-14 10:25:00 | 2277.04 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-18 09:40:00 | 2352.79 | 2023-07-18 09:55:00 | 2363.20 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-18 09:40:00 | 2352.79 | 2023-07-18 11:10:00 | 2370.20 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2023-07-20 09:40:00 | 2332.68 | 2023-07-20 09:55:00 | 2338.28 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-21 09:30:00 | 2352.94 | 2023-07-21 09:45:00 | 2361.85 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-21 09:30:00 | 2352.94 | 2023-07-21 10:10:00 | 2352.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-24 10:00:00 | 2356.04 | 2023-07-24 10:15:00 | 2350.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-25 09:40:00 | 2356.82 | 2023-07-25 09:50:00 | 2351.45 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-28 09:55:00 | 2386.68 | 2023-07-28 10:15:00 | 2399.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-28 09:55:00 | 2386.68 | 2023-07-28 10:30:00 | 2386.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-31 10:10:00 | 2412.08 | 2023-07-31 10:45:00 | 2404.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-08-10 09:30:00 | 2465.30 | 2023-08-10 09:35:00 | 2457.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-16 10:45:00 | 2382.75 | 2023-08-16 10:50:00 | 2374.17 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-17 10:10:00 | 2389.34 | 2023-08-17 10:25:00 | 2382.68 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-30 11:10:00 | 2451.68 | 2023-08-30 11:35:00 | 2461.25 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-08-30 11:10:00 | 2451.68 | 2023-08-30 13:20:00 | 2451.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-05 09:40:00 | 2393.95 | 2023-09-05 09:50:00 | 2399.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-06 09:55:00 | 2422.84 | 2023-09-06 10:30:00 | 2416.77 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-08 10:35:00 | 2443.01 | 2023-09-08 11:35:00 | 2450.33 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-08 10:35:00 | 2443.01 | 2023-09-08 14:55:00 | 2446.98 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-09-15 10:00:00 | 2460.26 | 2023-09-15 10:35:00 | 2469.63 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-09-15 10:00:00 | 2460.26 | 2023-09-15 10:50:00 | 2460.26 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-20 09:30:00 | 2448.43 | 2023-09-20 09:35:00 | 2440.58 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-21 09:30:00 | 2414.26 | 2023-09-21 10:05:00 | 2407.21 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-26 10:20:00 | 2418.67 | 2023-09-26 10:30:00 | 2412.74 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-27 11:15:00 | 2407.04 | 2023-09-27 11:25:00 | 2402.17 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-28 10:10:00 | 2428.66 | 2023-09-28 10:25:00 | 2423.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-29 10:35:00 | 2353.86 | 2023-09-29 10:50:00 | 2361.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-10-04 10:45:00 | 2356.58 | 2023-10-04 10:55:00 | 2368.80 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-10-04 10:45:00 | 2356.58 | 2023-10-04 15:20:00 | 2389.64 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2023-10-10 09:30:00 | 2408.20 | 2023-10-10 11:05:00 | 2420.87 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-10-10 09:30:00 | 2408.20 | 2023-10-10 11:50:00 | 2408.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-13 09:45:00 | 2362.00 | 2023-10-13 13:50:00 | 2372.07 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-10-17 10:35:00 | 2385.18 | 2023-10-17 10:40:00 | 2380.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-18 11:10:00 | 2341.31 | 2023-10-18 11:15:00 | 2345.65 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-23 09:35:00 | 2316.34 | 2023-10-23 09:40:00 | 2321.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-30 11:15:00 | 2215.42 | 2023-10-30 11:45:00 | 2208.21 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-10-31 09:30:00 | 2218.23 | 2023-10-31 09:35:00 | 2225.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-01 10:00:00 | 2216.19 | 2023-11-01 10:05:00 | 2222.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-06 11:10:00 | 2169.22 | 2023-11-06 12:55:00 | 2174.13 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-07 10:20:00 | 2190.41 | 2023-11-07 10:25:00 | 2186.14 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-08 09:50:00 | 2187.35 | 2023-11-08 10:10:00 | 2182.36 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-11-13 10:15:00 | 2134.32 | 2023-11-13 12:20:00 | 2139.16 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-11-15 11:10:00 | 2155.21 | 2023-11-15 13:15:00 | 2159.07 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-16 09:30:00 | 2131.46 | 2023-11-16 09:35:00 | 2136.40 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-17 09:50:00 | 2159.04 | 2023-11-17 10:20:00 | 2167.97 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-11-17 09:50:00 | 2159.04 | 2023-11-17 10:45:00 | 2159.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 09:30:00 | 2132.29 | 2023-11-20 09:45:00 | 2125.24 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-11-20 09:30:00 | 2132.29 | 2023-11-20 10:10:00 | 2132.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 11:00:00 | 2100.15 | 2023-11-23 12:20:00 | 2105.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-24 09:30:00 | 2119.25 | 2023-11-24 09:40:00 | 2114.70 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-29 10:40:00 | 2332.82 | 2023-11-29 10:50:00 | 2345.17 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-12-01 10:25:00 | 2320.95 | 2023-12-01 10:30:00 | 2310.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-12-11 11:00:00 | 2769.48 | 2023-12-11 11:05:00 | 2788.50 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2023-12-11 11:00:00 | 2769.48 | 2023-12-11 13:45:00 | 2781.74 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-13 09:30:00 | 2760.85 | 2023-12-13 09:40:00 | 2769.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-14 10:45:00 | 2840.64 | 2023-12-14 10:50:00 | 2830.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-15 09:55:00 | 2852.22 | 2023-12-15 10:05:00 | 2842.92 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-19 09:35:00 | 2854.50 | 2023-12-19 10:05:00 | 2841.30 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-12-19 09:35:00 | 2854.50 | 2023-12-19 10:20:00 | 2854.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 09:55:00 | 2771.76 | 2023-12-26 10:05:00 | 2786.15 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-12-26 09:55:00 | 2771.76 | 2023-12-26 11:00:00 | 2771.76 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 09:30:00 | 2767.83 | 2023-12-27 09:35:00 | 2778.04 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-12-28 11:15:00 | 2742.53 | 2023-12-28 11:35:00 | 2748.24 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-29 11:00:00 | 2760.75 | 2023-12-29 11:15:00 | 2774.10 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-12-29 11:00:00 | 2760.75 | 2023-12-29 11:25:00 | 2760.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-01 10:35:00 | 2798.81 | 2024-01-01 10:40:00 | 2810.50 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-01-01 10:35:00 | 2798.81 | 2024-01-01 15:10:00 | 2822.17 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-01-02 10:40:00 | 2770.40 | 2024-01-02 10:50:00 | 2781.74 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-05 09:55:00 | 2941.61 | 2024-01-05 10:15:00 | 2957.00 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-01-05 09:55:00 | 2941.61 | 2024-01-05 10:30:00 | 2941.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-09 10:00:00 | 2958.24 | 2024-01-09 10:45:00 | 2945.10 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-01-10 09:35:00 | 2966.62 | 2024-01-10 09:40:00 | 2983.88 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-01-10 09:35:00 | 2966.62 | 2024-01-10 09:50:00 | 2966.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-16 09:55:00 | 2985.04 | 2024-01-16 10:15:00 | 2975.78 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-01-16 09:55:00 | 2985.04 | 2024-01-16 15:20:00 | 2962.75 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-01-18 09:35:00 | 2859.98 | 2024-01-18 09:45:00 | 2843.18 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-01-18 09:35:00 | 2859.98 | 2024-01-18 11:30:00 | 2839.52 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2024-01-20 10:00:00 | 2814.61 | 2024-01-20 10:15:00 | 2822.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-25 10:35:00 | 2764.29 | 2024-01-25 10:50:00 | 2748.00 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-01-25 10:35:00 | 2764.29 | 2024-01-25 13:40:00 | 2764.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-01 11:05:00 | 3030.76 | 2024-02-01 11:15:00 | 3016.74 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-02-01 11:05:00 | 3030.76 | 2024-02-01 11:40:00 | 3030.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-05 10:25:00 | 3099.44 | 2024-02-05 10:45:00 | 3115.68 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-02-05 10:25:00 | 3099.44 | 2024-02-05 12:20:00 | 3099.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-06 10:40:00 | 3123.15 | 2024-02-06 11:00:00 | 3111.84 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-02-07 09:55:00 | 3149.76 | 2024-02-07 10:00:00 | 3139.42 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-08 09:35:00 | 3105.26 | 2024-02-08 09:50:00 | 3114.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-12 09:45:00 | 3099.25 | 2024-02-12 10:05:00 | 3082.49 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-02-12 09:45:00 | 3099.25 | 2024-02-12 11:10:00 | 3099.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 10:35:00 | 3171.14 | 2024-02-21 10:40:00 | 3162.59 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-22 10:55:00 | 3118.49 | 2024-02-22 11:55:00 | 3126.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-23 10:35:00 | 3132.65 | 2024-02-23 10:40:00 | 3140.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-28 10:50:00 | 3169.25 | 2024-02-28 10:55:00 | 3157.42 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-02-28 10:50:00 | 3169.25 | 2024-02-28 11:20:00 | 3169.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-29 11:10:00 | 3154.70 | 2024-02-29 11:20:00 | 3144.51 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-02 09:50:00 | 3234.44 | 2024-03-04 09:15:00 | 3233.23 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest1 | 2024-03-04 09:30:00 | 3226.11 | 2024-03-04 09:40:00 | 3214.62 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-03-04 09:30:00 | 3226.11 | 2024-03-04 09:55:00 | 3226.11 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-05 11:15:00 | 3198.33 | 2024-03-05 11:35:00 | 3204.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-03-06 09:30:00 | 3183.01 | 2024-03-06 09:45:00 | 3168.66 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-03-06 09:30:00 | 3183.01 | 2024-03-06 14:35:00 | 3127.36 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2024-03-28 09:40:00 | 3015.83 | 2024-03-28 09:55:00 | 3023.63 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-01 11:05:00 | 3152.81 | 2024-04-01 11:15:00 | 3168.23 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-04-01 11:05:00 | 3152.81 | 2024-04-01 11:30:00 | 3152.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-04 09:40:00 | 3163.57 | 2024-04-04 09:50:00 | 3153.91 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-05 09:35:00 | 3089.31 | 2024-04-05 09:45:00 | 3099.59 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-18 10:05:00 | 3041.03 | 2024-04-18 10:15:00 | 3033.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-25 09:30:00 | 2938.80 | 2024-04-25 09:40:00 | 2945.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-30 09:35:00 | 3006.57 | 2024-04-30 09:50:00 | 2997.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-02 10:50:00 | 3000.27 | 2024-05-02 11:10:00 | 2991.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-07 10:50:00 | 2756.73 | 2024-05-07 10:55:00 | 2766.59 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-08 10:10:00 | 2800.50 | 2024-05-08 10:15:00 | 2788.99 | STOP_HIT | 1.00 | -0.41% |
