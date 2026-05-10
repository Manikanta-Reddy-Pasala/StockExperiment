# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 8
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.12% | -1.9% |
| BUY @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.12% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.56% | 5.6% |
| SELL @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.56% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 12 | 46.2% | 4 | 14 | 8 | 0.14% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2451.90 | 2434.56 | 0.00 | ORB-long ORB[2412.40,2438.90] vol=5.6x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 2444.99 | 2436.77 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 2344.50 | 2358.97 | 0.00 | ORB-short ORB[2351.10,2373.90] vol=3.2x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 2349.26 | 2356.36 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 2374.90 | 2370.05 | 0.00 | ORB-long ORB[2360.00,2374.40] vol=2.1x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 2384.17 | 2375.38 | 0.00 | T1 1.5R @ 2384.17 |
| Stop hit — per-position SL triggered | 2026-02-18 10:05:00 | 2374.90 | 2374.60 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 2304.30 | 2312.99 | 0.00 | ORB-short ORB[2308.20,2329.10] vol=1.5x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 2295.30 | 2308.53 | 0.00 | T1 1.5R @ 2295.30 |
| Stop hit — per-position SL triggered | 2026-02-20 10:30:00 | 2304.30 | 2300.92 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 2202.40 | 2218.22 | 0.00 | ORB-short ORB[2215.70,2239.90] vol=3.4x ATR=8.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:25:00 | 2189.43 | 2213.25 | 0.00 | T1 1.5R @ 2189.43 |
| Target hit | 2026-03-05 15:05:00 | 2163.30 | 2158.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2122.00 | 2132.42 | 0.00 | ORB-short ORB[2129.00,2150.00] vol=1.9x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:20:00 | 2113.10 | 2128.39 | 0.00 | T1 1.5R @ 2113.10 |
| Target hit | 2026-03-06 15:20:00 | 2094.00 | 2114.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:55:00 | 2113.60 | 2080.59 | 0.00 | ORB-long ORB[2068.40,2095.00] vol=2.6x ATR=8.13 |
| Stop hit — per-position SL triggered | 2026-03-10 11:00:00 | 2105.47 | 2081.21 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 2334.90 | 2261.90 | 0.00 | ORB-long ORB[2069.10,2103.00] vol=5.9x ATR=31.42 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 2303.48 | 2282.85 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:55:00 | 2197.40 | 2176.96 | 0.00 | ORB-long ORB[2153.00,2179.90] vol=1.6x ATR=10.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 2212.75 | 2190.81 | 0.00 | T1 1.5R @ 2212.75 |
| Stop hit — per-position SL triggered | 2026-03-18 10:20:00 | 2197.40 | 2192.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 2408.90 | 2419.31 | 0.00 | ORB-short ORB[2410.30,2434.50] vol=5.7x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:10:00 | 2400.62 | 2415.44 | 0.00 | T1 1.5R @ 2400.62 |
| Target hit | 2026-04-15 15:20:00 | 2381.00 | 2400.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 2425.00 | 2406.37 | 0.00 | ORB-long ORB[2383.10,2409.60] vol=5.2x ATR=8.73 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 2416.27 | 2407.74 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 2386.30 | 2377.07 | 0.00 | ORB-long ORB[2367.00,2382.70] vol=2.6x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:10:00 | 2393.09 | 2386.08 | 0.00 | T1 1.5R @ 2393.09 |
| Target hit | 2026-04-21 15:10:00 | 2387.10 | 2395.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 2324.80 | 2303.76 | 0.00 | ORB-long ORB[2290.20,2306.00] vol=1.7x ATR=7.56 |
| Stop hit — per-position SL triggered | 2026-04-29 10:25:00 | 2317.24 | 2306.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 2320.00 | 2310.61 | 0.00 | ORB-long ORB[2292.00,2311.90] vol=2.8x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 2329.87 | 2318.04 | 0.00 | T1 1.5R @ 2329.87 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 2320.00 | 2319.44 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:25:00 | 2304.00 | 2312.67 | 0.00 | ORB-short ORB[2307.00,2339.90] vol=2.0x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-05-05 10:30:00 | 2308.99 | 2312.00 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 2379.70 | 2364.16 | 0.00 | ORB-long ORB[2340.00,2365.90] vol=3.9x ATR=10.26 |
| Stop hit — per-position SL triggered | 2026-05-06 09:40:00 | 2369.44 | 2364.88 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 2374.70 | 2365.93 | 0.00 | ORB-long ORB[2350.40,2368.40] vol=3.9x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 2367.79 | 2365.87 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 2370.20 | 2351.36 | 0.00 | ORB-long ORB[2331.10,2354.00] vol=2.4x ATR=7.81 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 2362.39 | 2354.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 2451.90 | 2026-02-10 11:10:00 | 2444.99 | STOP_HIT | 1.00 | -0.28% |
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
