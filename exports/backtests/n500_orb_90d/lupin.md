# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2373.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 17
- **Target hits / Stop hits / Partials:** 3 / 17 / 10
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 6.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.30% | 5.0% |
| BUY @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.30% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.12% | 1.6% |
| SELL @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.12% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 13 | 43.3% | 3 | 17 | 10 | 0.22% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 2192.50 | 2211.48 | 0.00 | ORB-short ORB[2201.00,2218.00] vol=3.0x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 2184.84 | 2206.41 | 0.00 | T1 1.5R @ 2184.84 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 2192.50 | 2201.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:30:00 | 2206.40 | 2223.16 | 0.00 | ORB-short ORB[2218.00,2234.00] vol=3.3x ATR=5.95 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 2212.35 | 2221.18 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 2227.70 | 2220.32 | 0.00 | ORB-long ORB[2210.50,2224.20] vol=2.1x ATR=5.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:40:00 | 2236.51 | 2222.86 | 0.00 | T1 1.5R @ 2236.51 |
| Target hit | 2026-02-23 11:55:00 | 2235.70 | 2239.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 2276.80 | 2264.27 | 0.00 | ORB-long ORB[2255.70,2270.00] vol=2.2x ATR=4.73 |
| Stop hit — per-position SL triggered | 2026-02-25 11:25:00 | 2272.07 | 2265.32 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 2304.60 | 2318.62 | 0.00 | ORB-short ORB[2316.70,2336.30] vol=1.5x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:00:00 | 2297.87 | 2315.65 | 0.00 | T1 1.5R @ 2297.87 |
| Stop hit — per-position SL triggered | 2026-02-27 12:25:00 | 2304.60 | 2314.35 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 2295.10 | 2287.49 | 0.00 | ORB-long ORB[2265.10,2293.70] vol=1.7x ATR=7.36 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 2287.74 | 2289.16 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:50:00 | 2290.70 | 2299.55 | 0.00 | ORB-short ORB[2300.00,2325.00] vol=2.3x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:55:00 | 2280.69 | 2296.36 | 0.00 | T1 1.5R @ 2280.69 |
| Stop hit — per-position SL triggered | 2026-03-24 11:00:00 | 2290.70 | 2295.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 2358.60 | 2346.49 | 0.00 | ORB-long ORB[2323.40,2354.70] vol=1.7x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:05:00 | 2367.03 | 2349.30 | 0.00 | T1 1.5R @ 2367.03 |
| Stop hit — per-position SL triggered | 2026-03-27 11:10:00 | 2358.60 | 2350.06 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:05:00 | 2304.00 | 2313.07 | 0.00 | ORB-short ORB[2306.10,2330.00] vol=1.9x ATR=6.18 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 2310.18 | 2309.30 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:50:00 | 2290.00 | 2285.24 | 0.00 | ORB-long ORB[2246.90,2271.50] vol=1.9x ATR=7.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:15:00 | 2301.12 | 2286.69 | 0.00 | T1 1.5R @ 2301.12 |
| Stop hit — per-position SL triggered | 2026-04-07 12:45:00 | 2290.00 | 2291.64 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 2308.70 | 2319.79 | 0.00 | ORB-short ORB[2309.00,2339.00] vol=1.7x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:45:00 | 2297.95 | 2310.77 | 0.00 | T1 1.5R @ 2297.95 |
| Target hit | 2026-04-08 12:05:00 | 2278.90 | 2274.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2026-04-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:35:00 | 2285.60 | 2294.58 | 0.00 | ORB-short ORB[2289.70,2308.70] vol=1.8x ATR=6.09 |
| Stop hit — per-position SL triggered | 2026-04-09 09:40:00 | 2291.69 | 2294.19 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2335.40 | 2322.12 | 0.00 | ORB-long ORB[2301.10,2329.10] vol=1.6x ATR=6.67 |
| Stop hit — per-position SL triggered | 2026-04-13 11:10:00 | 2328.73 | 2324.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 2324.10 | 2329.04 | 0.00 | ORB-short ORB[2327.90,2340.20] vol=1.9x ATR=3.58 |
| Stop hit — per-position SL triggered | 2026-04-21 10:45:00 | 2327.68 | 2328.70 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:00:00 | 2313.80 | 2302.12 | 0.00 | ORB-long ORB[2285.10,2313.70] vol=3.7x ATR=5.22 |
| Stop hit — per-position SL triggered | 2026-04-22 11:45:00 | 2308.58 | 2304.13 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 2353.60 | 2333.53 | 0.00 | ORB-long ORB[2293.90,2327.00] vol=2.2x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:50:00 | 2364.90 | 2342.66 | 0.00 | T1 1.5R @ 2364.90 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 2353.60 | 2343.92 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:10:00 | 2307.00 | 2326.23 | 0.00 | ORB-short ORB[2318.00,2351.50] vol=1.6x ATR=7.18 |
| Stop hit — per-position SL triggered | 2026-04-24 10:30:00 | 2314.18 | 2321.07 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 2328.50 | 2317.29 | 0.00 | ORB-long ORB[2305.00,2323.20] vol=3.8x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-04-29 12:00:00 | 2323.29 | 2319.38 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:05:00 | 2344.60 | 2336.01 | 0.00 | ORB-long ORB[2304.10,2324.00] vol=2.4x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:20:00 | 2354.02 | 2341.15 | 0.00 | T1 1.5R @ 2354.02 |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 2344.60 | 2343.21 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 2362.90 | 2359.77 | 0.00 | ORB-long ORB[2350.40,2362.80] vol=4.0x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:35:00 | 2372.22 | 2364.04 | 0.00 | T1 1.5R @ 2372.22 |
| Target hit | 2026-05-06 15:20:00 | 2443.60 | 2415.83 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 2192.50 | 2026-02-12 11:20:00 | 2184.84 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-12 11:15:00 | 2192.50 | 2026-02-12 11:30:00 | 2192.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:30:00 | 2206.40 | 2026-02-19 10:40:00 | 2212.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-23 09:30:00 | 2227.70 | 2026-02-23 09:40:00 | 2236.51 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-23 09:30:00 | 2227.70 | 2026-02-23 11:55:00 | 2235.70 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-25 11:05:00 | 2276.80 | 2026-02-25 11:25:00 | 2272.07 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 11:15:00 | 2304.60 | 2026-02-27 12:00:00 | 2297.87 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-27 11:15:00 | 2304.60 | 2026-02-27 12:25:00 | 2304.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:30:00 | 2295.10 | 2026-03-20 09:40:00 | 2287.74 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-24 10:50:00 | 2290.70 | 2026-03-24 10:55:00 | 2280.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-24 10:50:00 | 2290.70 | 2026-03-24 11:00:00 | 2290.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 11:00:00 | 2358.60 | 2026-03-27 11:05:00 | 2367.03 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-27 11:00:00 | 2358.60 | 2026-03-27 11:10:00 | 2358.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 11:05:00 | 2304.00 | 2026-03-30 12:00:00 | 2310.18 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-07 10:50:00 | 2290.00 | 2026-04-07 11:15:00 | 2301.12 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-07 10:50:00 | 2290.00 | 2026-04-07 12:45:00 | 2290.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-08 09:35:00 | 2308.70 | 2026-04-08 09:45:00 | 2297.95 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-08 09:35:00 | 2308.70 | 2026-04-08 12:05:00 | 2278.90 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2026-04-09 09:35:00 | 2285.60 | 2026-04-09 09:40:00 | 2291.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-13 10:45:00 | 2335.40 | 2026-04-13 11:10:00 | 2328.73 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-21 10:40:00 | 2324.10 | 2026-04-21 10:45:00 | 2327.68 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-04-22 11:00:00 | 2313.80 | 2026-04-22 11:45:00 | 2308.58 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-23 09:40:00 | 2353.60 | 2026-04-23 09:50:00 | 2364.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-23 09:40:00 | 2353.60 | 2026-04-23 09:55:00 | 2353.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:10:00 | 2307.00 | 2026-04-24 10:30:00 | 2314.18 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-29 11:10:00 | 2328.50 | 2026-04-29 12:00:00 | 2323.29 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-04 11:05:00 | 2344.60 | 2026-05-04 11:20:00 | 2354.02 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-05-04 11:05:00 | 2344.60 | 2026-05-04 12:15:00 | 2344.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:30:00 | 2362.90 | 2026-05-06 09:35:00 | 2372.22 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-05-06 09:30:00 | 2362.90 | 2026-05-06 15:20:00 | 2443.60 | TARGET_HIT | 0.50 | 3.42% |
