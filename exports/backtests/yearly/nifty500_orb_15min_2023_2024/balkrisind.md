# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2024-02-06 09:15:00 → 2026-05-08 15:25:00 (41617 bars)
- **Last close:** 2265.10
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
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 24
- **Target hits / Stop hits / Partials:** 4 / 24 / 7
- **Avg / median % per leg:** -0.02% / -0.24%
- **Sum % (uncompounded):** -0.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.03% | 0.5% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.03% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 6 | 28.6% | 2 | 15 | 4 | -0.05% | -1.1% |
| SELL @ 2nd Alert (retest1) | 21 | 6 | 28.6% | 2 | 15 | 4 | -0.05% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 35 | 11 | 31.4% | 4 | 24 | 7 | -0.02% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 10:35:00 | 2362.50 | 2375.78 | 0.00 | ORB-short ORB[2387.40,2404.00] vol=1.7x ATR=6.96 |
| Stop hit — per-position SL triggered | 2024-02-07 10:45:00 | 2369.46 | 2374.76 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-02-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:05:00 | 2325.85 | 2339.79 | 0.00 | ORB-short ORB[2338.80,2365.65] vol=2.0x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-02-08 11:45:00 | 2331.60 | 2337.62 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-02-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:20:00 | 2288.80 | 2301.87 | 0.00 | ORB-short ORB[2301.05,2330.00] vol=5.4x ATR=8.53 |
| Stop hit — per-position SL triggered | 2024-02-09 10:40:00 | 2297.33 | 2300.32 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 11:15:00 | 2314.90 | 2309.12 | 0.00 | ORB-long ORB[2285.60,2314.00] vol=2.1x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-02-14 11:35:00 | 2308.06 | 2309.54 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:30:00 | 2307.00 | 2313.76 | 0.00 | ORB-short ORB[2308.00,2327.50] vol=1.7x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 09:35:00 | 2297.95 | 2304.95 | 0.00 | T1 1.5R @ 2297.95 |
| Target hit | 2024-02-20 12:30:00 | 2298.05 | 2297.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2024-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 10:00:00 | 2317.90 | 2309.09 | 0.00 | ORB-long ORB[2286.50,2314.00] vol=2.0x ATR=6.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 10:35:00 | 2328.19 | 2314.41 | 0.00 | T1 1.5R @ 2328.19 |
| Target hit | 2024-02-23 15:15:00 | 2332.95 | 2334.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:45:00 | 2292.10 | 2311.81 | 0.00 | ORB-short ORB[2307.10,2333.00] vol=2.1x ATR=8.80 |
| Stop hit — per-position SL triggered | 2024-02-26 09:50:00 | 2300.90 | 2301.33 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 2269.95 | 2278.82 | 0.00 | ORB-short ORB[2280.00,2299.75] vol=2.4x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-02-28 11:10:00 | 2274.58 | 2278.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-03-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 09:35:00 | 2221.45 | 2226.50 | 0.00 | ORB-short ORB[2224.40,2248.00] vol=1.8x ATR=9.46 |
| Stop hit — per-position SL triggered | 2024-03-01 10:40:00 | 2230.91 | 2221.74 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 2250.80 | 2258.96 | 0.00 | ORB-short ORB[2251.00,2269.80] vol=3.5x ATR=6.69 |
| Stop hit — per-position SL triggered | 2024-03-04 10:10:00 | 2257.49 | 2256.13 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:00:00 | 2244.90 | 2255.94 | 0.00 | ORB-short ORB[2259.50,2287.00] vol=2.1x ATR=7.85 |
| Stop hit — per-position SL triggered | 2024-03-06 10:15:00 | 2252.75 | 2255.11 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 10:10:00 | 2240.05 | 2245.36 | 0.00 | ORB-short ORB[2240.45,2263.85] vol=7.7x ATR=8.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:15:00 | 2227.67 | 2229.40 | 0.00 | T1 1.5R @ 2227.67 |
| Target hit | 2024-03-11 10:30:00 | 2222.55 | 2221.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2024-03-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 09:45:00 | 2281.40 | 2294.09 | 0.00 | ORB-short ORB[2291.65,2312.95] vol=1.8x ATR=7.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 10:05:00 | 2270.51 | 2290.11 | 0.00 | T1 1.5R @ 2270.51 |
| Stop hit — per-position SL triggered | 2024-03-22 10:10:00 | 2281.40 | 2289.78 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:50:00 | 2292.95 | 2287.82 | 0.00 | ORB-long ORB[2255.40,2285.60] vol=1.5x ATR=5.68 |
| Stop hit — per-position SL triggered | 2024-03-27 11:00:00 | 2287.27 | 2288.39 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 10:15:00 | 2311.75 | 2322.94 | 0.00 | ORB-short ORB[2316.90,2337.55] vol=1.5x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-04-01 11:10:00 | 2319.61 | 2317.12 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-04-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:05:00 | 2309.05 | 2301.84 | 0.00 | ORB-long ORB[2284.00,2301.00] vol=8.5x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:15:00 | 2318.13 | 2302.92 | 0.00 | T1 1.5R @ 2318.13 |
| Target hit | 2024-04-02 15:20:00 | 2329.75 | 2319.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-04-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 10:20:00 | 2350.00 | 2336.04 | 0.00 | ORB-long ORB[2310.55,2334.90] vol=1.8x ATR=5.10 |
| Stop hit — per-position SL triggered | 2024-04-03 10:30:00 | 2344.90 | 2336.48 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-04-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:35:00 | 2435.10 | 2416.23 | 0.00 | ORB-long ORB[2400.05,2430.95] vol=2.9x ATR=8.03 |
| Stop hit — per-position SL triggered | 2024-04-08 11:15:00 | 2427.07 | 2422.94 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 11:00:00 | 2426.05 | 2445.65 | 0.00 | ORB-short ORB[2438.25,2453.00] vol=2.0x ATR=9.07 |
| Stop hit — per-position SL triggered | 2024-04-09 12:40:00 | 2435.12 | 2436.89 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 2466.95 | 2454.38 | 0.00 | ORB-long ORB[2434.00,2460.25] vol=5.0x ATR=7.27 |
| Stop hit — per-position SL triggered | 2024-04-10 10:40:00 | 2459.68 | 2454.33 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:10:00 | 2389.30 | 2381.85 | 0.00 | ORB-long ORB[2362.95,2387.45] vol=1.5x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-04-16 11:20:00 | 2383.55 | 2382.36 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-04-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:40:00 | 2372.20 | 2368.65 | 0.00 | ORB-long ORB[2351.00,2370.25] vol=3.1x ATR=6.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 09:50:00 | 2382.45 | 2371.49 | 0.00 | T1 1.5R @ 2382.45 |
| Stop hit — per-position SL triggered | 2024-04-18 10:30:00 | 2372.20 | 2373.96 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 10:30:00 | 2316.20 | 2324.16 | 0.00 | ORB-short ORB[2335.00,2349.95] vol=2.1x ATR=5.53 |
| Stop hit — per-position SL triggered | 2024-04-24 10:40:00 | 2321.73 | 2323.55 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-04-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 11:05:00 | 2318.00 | 2323.19 | 0.00 | ORB-short ORB[2319.20,2350.00] vol=1.7x ATR=4.84 |
| Stop hit — per-position SL triggered | 2024-04-25 11:20:00 | 2322.84 | 2322.67 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:30:00 | 2370.00 | 2375.37 | 0.00 | ORB-short ORB[2375.00,2410.00] vol=7.8x ATR=7.14 |
| Stop hit — per-position SL triggered | 2024-04-29 11:45:00 | 2377.14 | 2372.26 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-05-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:45:00 | 2437.55 | 2453.50 | 0.00 | ORB-short ORB[2460.55,2474.50] vol=2.0x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 12:55:00 | 2428.11 | 2440.43 | 0.00 | T1 1.5R @ 2428.11 |
| Stop hit — per-position SL triggered | 2024-05-03 13:05:00 | 2437.55 | 2440.23 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-05-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:45:00 | 2487.70 | 2475.63 | 0.00 | ORB-long ORB[2461.25,2479.00] vol=2.9x ATR=8.20 |
| Stop hit — per-position SL triggered | 2024-05-09 09:50:00 | 2479.50 | 2476.86 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:15:00 | 2460.45 | 2452.84 | 0.00 | ORB-long ORB[2424.05,2450.00] vol=6.8x ATR=9.14 |
| Stop hit — per-position SL triggered | 2024-05-10 10:20:00 | 2451.31 | 2452.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-02-07 10:35:00 | 2362.50 | 2024-02-07 10:45:00 | 2369.46 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-08 11:05:00 | 2325.85 | 2024-02-08 11:45:00 | 2331.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-09 10:20:00 | 2288.80 | 2024-02-09 10:40:00 | 2297.33 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-02-14 11:15:00 | 2314.90 | 2024-02-14 11:35:00 | 2308.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-20 09:30:00 | 2307.00 | 2024-02-20 09:35:00 | 2297.95 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-20 09:30:00 | 2307.00 | 2024-02-20 12:30:00 | 2298.05 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-02-23 10:00:00 | 2317.90 | 2024-02-23 10:35:00 | 2328.19 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-23 10:00:00 | 2317.90 | 2024-02-23 15:15:00 | 2332.95 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-02-26 09:45:00 | 2292.10 | 2024-02-26 09:50:00 | 2300.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-28 10:50:00 | 2269.95 | 2024-02-28 11:10:00 | 2274.58 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-03-01 09:35:00 | 2221.45 | 2024-03-01 10:40:00 | 2230.91 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-03-04 09:30:00 | 2250.80 | 2024-03-04 10:10:00 | 2257.49 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-06 10:00:00 | 2244.90 | 2024-03-06 10:15:00 | 2252.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-11 10:10:00 | 2240.05 | 2024-03-11 10:15:00 | 2227.67 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-03-11 10:10:00 | 2240.05 | 2024-03-11 10:30:00 | 2222.55 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-03-22 09:45:00 | 2281.40 | 2024-03-22 10:05:00 | 2270.51 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-03-22 09:45:00 | 2281.40 | 2024-03-22 10:10:00 | 2281.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 10:50:00 | 2292.95 | 2024-03-27 11:00:00 | 2287.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-01 10:15:00 | 2311.75 | 2024-04-01 11:10:00 | 2319.61 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-04-02 10:05:00 | 2309.05 | 2024-04-02 10:15:00 | 2318.13 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-04-02 10:05:00 | 2309.05 | 2024-04-02 15:20:00 | 2329.75 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-04-03 10:20:00 | 2350.00 | 2024-04-03 10:30:00 | 2344.90 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-08 10:35:00 | 2435.10 | 2024-04-08 11:15:00 | 2427.07 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-09 11:00:00 | 2426.05 | 2024-04-09 12:40:00 | 2435.12 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-10 10:35:00 | 2466.95 | 2024-04-10 10:40:00 | 2459.68 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-16 11:10:00 | 2389.30 | 2024-04-16 11:20:00 | 2383.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-18 09:40:00 | 2372.20 | 2024-04-18 09:50:00 | 2382.45 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-04-18 09:40:00 | 2372.20 | 2024-04-18 10:30:00 | 2372.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-24 10:30:00 | 2316.20 | 2024-04-24 10:40:00 | 2321.73 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-25 11:05:00 | 2318.00 | 2024-04-25 11:20:00 | 2322.84 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-04-29 10:30:00 | 2370.00 | 2024-04-29 11:45:00 | 2377.14 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-03 10:45:00 | 2437.55 | 2024-05-03 12:55:00 | 2428.11 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-03 10:45:00 | 2437.55 | 2024-05-03 13:05:00 | 2437.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-09 09:45:00 | 2487.70 | 2024-05-09 09:50:00 | 2479.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-10 10:15:00 | 2460.45 | 2024-05-10 10:20:00 | 2451.31 | STOP_HIT | 1.00 | -0.37% |
