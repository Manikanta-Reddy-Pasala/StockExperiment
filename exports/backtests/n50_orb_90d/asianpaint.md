# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2600.00
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
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 10
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 4.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.19% | 2.2% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.19% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 7 | 43.8% | 1 | 9 | 6 | 0.14% | 2.2% |
| SELL @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 1 | 9 | 6 | 0.14% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 13 | 46.4% | 3 | 15 | 10 | 0.16% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 2405.10 | 2408.23 | 0.00 | ORB-short ORB[2407.30,2420.00] vol=6.1x ATR=5.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 2397.34 | 2406.95 | 0.00 | T1 1.5R @ 2397.34 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 2405.10 | 2406.50 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 2383.90 | 2389.25 | 0.00 | ORB-short ORB[2393.60,2405.70] vol=4.1x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 2388.70 | 2389.11 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2396.00 | 2400.37 | 0.00 | ORB-short ORB[2400.30,2411.80] vol=1.8x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:35:00 | 2389.28 | 2399.09 | 0.00 | T1 1.5R @ 2389.28 |
| Target hit | 2026-02-13 15:20:00 | 2365.20 | 2381.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 2379.30 | 2368.74 | 0.00 | ORB-long ORB[2356.80,2372.60] vol=1.6x ATR=4.74 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 2374.56 | 2369.36 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 2432.90 | 2434.14 | 0.00 | ORB-short ORB[2433.50,2448.20] vol=1.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:20:00 | 2426.70 | 2433.62 | 0.00 | T1 1.5R @ 2426.70 |
| Stop hit — per-position SL triggered | 2026-02-18 12:55:00 | 2432.90 | 2428.90 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 2400.00 | 2409.80 | 0.00 | ORB-short ORB[2401.00,2428.00] vol=1.8x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 2405.05 | 2409.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 2404.00 | 2412.70 | 0.00 | ORB-short ORB[2410.00,2420.30] vol=1.8x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 2398.32 | 2410.72 | 0.00 | T1 1.5R @ 2398.32 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 2404.00 | 2408.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 2369.70 | 2375.86 | 0.00 | ORB-short ORB[2371.40,2397.60] vol=1.5x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:00:00 | 2362.26 | 2371.64 | 0.00 | T1 1.5R @ 2362.26 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 2369.70 | 2371.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 2257.00 | 2269.22 | 0.00 | ORB-short ORB[2271.40,2302.10] vol=1.8x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-03-05 10:35:00 | 2262.84 | 2268.77 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 2271.50 | 2286.77 | 0.00 | ORB-short ORB[2285.30,2302.00] vol=2.8x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 2277.85 | 2286.03 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 2234.60 | 2219.65 | 0.00 | ORB-long ORB[2201.00,2223.60] vol=1.5x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:45:00 | 2245.39 | 2223.95 | 0.00 | T1 1.5R @ 2245.39 |
| Stop hit — per-position SL triggered | 2026-03-12 11:50:00 | 2234.60 | 2230.28 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 2220.20 | 2213.30 | 0.00 | ORB-long ORB[2204.10,2219.60] vol=2.2x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-03-20 11:00:00 | 2215.43 | 2213.53 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:10:00 | 2248.40 | 2241.31 | 0.00 | ORB-long ORB[2222.00,2242.90] vol=1.9x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:40:00 | 2258.18 | 2244.92 | 0.00 | T1 1.5R @ 2258.18 |
| Target hit | 2026-03-25 15:20:00 | 2270.00 | 2264.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 2473.50 | 2455.27 | 0.00 | ORB-long ORB[2417.50,2446.90] vol=1.7x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:50:00 | 2482.15 | 2458.63 | 0.00 | T1 1.5R @ 2482.15 |
| Stop hit — per-position SL triggered | 2026-04-17 12:50:00 | 2473.50 | 2463.45 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 2544.40 | 2537.14 | 0.00 | ORB-long ORB[2513.70,2537.90] vol=4.2x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 2555.15 | 2539.40 | 0.00 | T1 1.5R @ 2555.15 |
| Target hit | 2026-04-21 12:05:00 | 2558.80 | 2560.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:25:00 | 2494.00 | 2483.79 | 0.00 | ORB-long ORB[2458.00,2492.10] vol=2.4x ATR=6.81 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 2487.19 | 2486.20 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:10:00 | 2397.20 | 2413.14 | 0.00 | ORB-short ORB[2408.70,2429.70] vol=2.2x ATR=6.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:20:00 | 2387.02 | 2409.71 | 0.00 | T1 1.5R @ 2387.02 |
| Stop hit — per-position SL triggered | 2026-04-30 11:35:00 | 2397.20 | 2407.72 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:40:00 | 2481.30 | 2453.11 | 0.00 | ORB-long ORB[2440.40,2469.80] vol=2.3x ATR=6.95 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 2474.35 | 2454.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:30:00 | 2405.10 | 2026-02-10 09:35:00 | 2397.34 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 09:30:00 | 2405.10 | 2026-02-10 09:45:00 | 2405.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:10:00 | 2383.90 | 2026-02-11 10:35:00 | 2388.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 11:15:00 | 2396.00 | 2026-02-13 11:35:00 | 2389.28 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-13 11:15:00 | 2396.00 | 2026-02-13 15:20:00 | 2365.20 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-02-16 11:15:00 | 2379.30 | 2026-02-16 11:30:00 | 2374.56 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-18 11:00:00 | 2432.90 | 2026-02-18 11:20:00 | 2426.70 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-18 11:00:00 | 2432.90 | 2026-02-18 12:55:00 | 2432.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:10:00 | 2400.00 | 2026-02-19 11:25:00 | 2405.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-26 10:55:00 | 2404.00 | 2026-02-26 11:05:00 | 2398.32 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-26 10:55:00 | 2404.00 | 2026-02-26 11:25:00 | 2404.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:40:00 | 2369.70 | 2026-02-27 11:00:00 | 2362.26 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-27 09:40:00 | 2369.70 | 2026-02-27 11:25:00 | 2369.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:25:00 | 2257.00 | 2026-03-05 10:35:00 | 2262.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-11 10:55:00 | 2271.50 | 2026-03-11 11:00:00 | 2277.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-12 10:20:00 | 2234.60 | 2026-03-12 10:45:00 | 2245.39 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-12 10:20:00 | 2234.60 | 2026-03-12 11:50:00 | 2234.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:50:00 | 2220.20 | 2026-03-20 11:00:00 | 2215.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-25 10:10:00 | 2248.40 | 2026-03-25 10:40:00 | 2258.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-25 10:10:00 | 2248.40 | 2026-03-25 15:20:00 | 2270.00 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2026-04-17 11:15:00 | 2473.50 | 2026-04-17 11:50:00 | 2482.15 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-17 11:15:00 | 2473.50 | 2026-04-17 12:50:00 | 2473.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:40:00 | 2544.40 | 2026-04-21 09:45:00 | 2555.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-21 09:40:00 | 2544.40 | 2026-04-21 12:05:00 | 2558.80 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-28 10:25:00 | 2494.00 | 2026-04-28 11:00:00 | 2487.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-30 11:10:00 | 2397.20 | 2026-04-30 11:20:00 | 2387.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-30 11:10:00 | 2397.20 | 2026-04-30 11:35:00 | 2397.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:40:00 | 2481.30 | 2026-05-06 10:45:00 | 2474.35 | STOP_HIT | 1.00 | -0.28% |
