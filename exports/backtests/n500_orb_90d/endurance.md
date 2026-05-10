# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2530.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 8
- **Avg / median % per leg:** 0.20% / 0.15%
- **Sum % (uncompounded):** 4.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.13% | 1.1% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.13% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.23% | 3.6% |
| SELL @ 2nd Alert (retest1) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.23% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 12 | 50.0% | 4 | 12 | 8 | 0.20% | 4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 2557.60 | 2547.18 | 0.00 | ORB-long ORB[2520.10,2538.40] vol=1.9x ATR=7.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 2569.34 | 2549.71 | 0.00 | T1 1.5R @ 2569.34 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 2557.60 | 2550.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 2488.00 | 2499.55 | 0.00 | ORB-short ORB[2494.90,2525.30] vol=1.7x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:30:00 | 2479.45 | 2493.22 | 0.00 | T1 1.5R @ 2479.45 |
| Target hit | 2026-02-18 15:20:00 | 2482.80 | 2479.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 2460.00 | 2443.05 | 0.00 | ORB-long ORB[2429.60,2449.40] vol=2.3x ATR=7.68 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 2452.32 | 2443.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 2620.00 | 2629.96 | 0.00 | ORB-short ORB[2622.00,2652.80] vol=4.5x ATR=11.59 |
| Stop hit — per-position SL triggered | 2026-02-27 11:30:00 | 2631.59 | 2625.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:45:00 | 2523.80 | 2514.73 | 0.00 | ORB-long ORB[2503.40,2521.90] vol=3.2x ATR=12.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:00:00 | 2542.36 | 2525.12 | 0.00 | T1 1.5R @ 2542.36 |
| Target hit | 2026-03-05 10:20:00 | 2527.70 | 2528.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-03-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:50:00 | 2486.00 | 2491.36 | 0.00 | ORB-short ORB[2495.40,2520.30] vol=1.5x ATR=8.61 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 2494.61 | 2488.31 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 2422.80 | 2426.73 | 0.00 | ORB-short ORB[2424.60,2439.90] vol=2.2x ATR=10.75 |
| Stop hit — per-position SL triggered | 2026-03-10 09:55:00 | 2433.55 | 2424.63 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 2503.80 | 2485.07 | 0.00 | ORB-long ORB[2454.90,2488.00] vol=2.9x ATR=9.44 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 2494.36 | 2492.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:50:00 | 2368.50 | 2384.47 | 0.00 | ORB-short ORB[2385.00,2416.20] vol=4.0x ATR=9.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:55:00 | 2353.98 | 2376.58 | 0.00 | T1 1.5R @ 2353.98 |
| Target hit | 2026-03-17 15:20:00 | 2325.40 | 2354.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:50:00 | 2315.60 | 2326.30 | 0.00 | ORB-short ORB[2320.10,2353.40] vol=2.2x ATR=6.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:35:00 | 2305.23 | 2324.44 | 0.00 | T1 1.5R @ 2305.23 |
| Target hit | 2026-03-19 15:20:00 | 2277.90 | 2299.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:40:00 | 2185.00 | 2190.33 | 0.00 | ORB-short ORB[2186.30,2218.40] vol=2.3x ATR=6.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:45:00 | 2175.29 | 2187.56 | 0.00 | T1 1.5R @ 2175.29 |
| Stop hit — per-position SL triggered | 2026-04-07 10:55:00 | 2185.00 | 2185.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 2407.50 | 2414.22 | 0.00 | ORB-short ORB[2407.70,2436.70] vol=10.4x ATR=9.18 |
| Stop hit — per-position SL triggered | 2026-04-15 10:10:00 | 2416.68 | 2413.94 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 2396.20 | 2408.84 | 0.00 | ORB-short ORB[2399.20,2418.20] vol=3.8x ATR=7.56 |
| Stop hit — per-position SL triggered | 2026-04-17 10:50:00 | 2403.76 | 2407.89 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 2380.30 | 2375.02 | 0.00 | ORB-long ORB[2354.10,2376.60] vol=2.3x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:25:00 | 2390.30 | 2380.29 | 0.00 | T1 1.5R @ 2390.30 |
| Stop hit — per-position SL triggered | 2026-04-21 14:50:00 | 2380.30 | 2386.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 2328.10 | 2340.53 | 0.00 | ORB-short ORB[2344.70,2371.00] vol=1.5x ATR=9.22 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 2337.32 | 2339.20 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 2348.70 | 2373.59 | 0.00 | ORB-short ORB[2368.80,2397.70] vol=2.1x ATR=6.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:05:00 | 2338.46 | 2369.10 | 0.00 | T1 1.5R @ 2338.46 |
| Stop hit — per-position SL triggered | 2026-05-06 11:40:00 | 2348.70 | 2363.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:20:00 | 2557.60 | 2026-02-11 10:30:00 | 2569.34 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-11 10:20:00 | 2557.60 | 2026-02-11 10:35:00 | 2557.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 2488.00 | 2026-02-18 11:30:00 | 2479.45 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-18 09:55:00 | 2488.00 | 2026-02-18 15:20:00 | 2482.80 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-02-20 10:45:00 | 2460.00 | 2026-02-20 11:00:00 | 2452.32 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 09:55:00 | 2620.00 | 2026-02-27 11:30:00 | 2631.59 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-05 09:45:00 | 2523.80 | 2026-03-05 10:00:00 | 2542.36 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-05 09:45:00 | 2523.80 | 2026-03-05 10:20:00 | 2527.70 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-03-06 09:50:00 | 2486.00 | 2026-03-06 11:20:00 | 2494.61 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-10 09:30:00 | 2422.80 | 2026-03-10 09:55:00 | 2433.55 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-11 09:45:00 | 2503.80 | 2026-03-11 10:35:00 | 2494.36 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-17 10:50:00 | 2368.50 | 2026-03-17 11:55:00 | 2353.98 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-17 10:50:00 | 2368.50 | 2026-03-17 15:20:00 | 2325.40 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2026-03-19 10:50:00 | 2315.60 | 2026-03-19 11:35:00 | 2305.23 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-19 10:50:00 | 2315.60 | 2026-03-19 15:20:00 | 2277.90 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2026-04-07 10:40:00 | 2185.00 | 2026-04-07 10:45:00 | 2175.29 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-07 10:40:00 | 2185.00 | 2026-04-07 10:55:00 | 2185.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 09:50:00 | 2407.50 | 2026-04-15 10:10:00 | 2416.68 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-17 10:40:00 | 2396.20 | 2026-04-17 10:50:00 | 2403.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 10:35:00 | 2380.30 | 2026-04-21 11:25:00 | 2390.30 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-21 10:35:00 | 2380.30 | 2026-04-21 14:50:00 | 2380.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:35:00 | 2328.10 | 2026-05-05 11:05:00 | 2337.32 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2348.70 | 2026-05-06 11:05:00 | 2338.46 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2348.70 | 2026-05-06 11:40:00 | 2348.70 | STOP_HIT | 0.50 | 0.00% |
