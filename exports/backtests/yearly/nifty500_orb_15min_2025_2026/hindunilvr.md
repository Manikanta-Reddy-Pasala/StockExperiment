# Hindustan Unilever Ltd. (HINDUNILVR)

## Backtest Summary

- **Window:** 2025-12-08 09:15:00 → 2026-05-08 15:25:00 (4800 bars)
- **Last close:** 2286.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 16
- **Target hits / Stop hits / Partials:** 5 / 16 / 9
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 3.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.25% | 4.5% |
| BUY @ 2nd Alert (retest1) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.25% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 2 | 8 | 2 | -0.05% | -0.7% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 2 | 8 | 2 | -0.05% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 14 | 46.7% | 5 | 16 | 9 | 0.13% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 2331.40 | 2336.64 | 0.00 | ORB-short ORB[2333.80,2355.10] vol=2.1x ATR=6.18 |
| Stop hit — per-position SL triggered | 2025-12-08 12:10:00 | 2337.58 | 2334.86 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:15:00 | 2300.60 | 2308.00 | 0.00 | ORB-short ORB[2305.20,2316.90] vol=3.6x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-12-10 11:20:00 | 2304.48 | 2307.95 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-12-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:05:00 | 2256.40 | 2279.53 | 0.00 | ORB-short ORB[2298.00,2309.00] vol=5.6x ATR=6.89 |
| Stop hit — per-position SL triggered | 2025-12-12 10:10:00 | 2263.29 | 2276.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-12-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:45:00 | 2296.40 | 2292.20 | 0.00 | ORB-long ORB[2280.70,2289.30] vol=3.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-12-23 10:55:00 | 2293.52 | 2292.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-12-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:30:00 | 2312.30 | 2310.59 | 0.00 | ORB-long ORB[2286.80,2309.00] vol=2.4x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 12:20:00 | 2317.75 | 2311.67 | 0.00 | T1 1.5R @ 2317.75 |
| Stop hit — per-position SL triggered | 2025-12-31 14:30:00 | 2312.30 | 2313.90 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 2303.80 | 2308.39 | 0.00 | ORB-short ORB[2310.50,2322.00] vol=2.0x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-01-01 11:25:00 | 2306.88 | 2307.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-01-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:50:00 | 2370.60 | 2351.30 | 0.00 | ORB-long ORB[2336.40,2358.00] vol=2.0x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:05:00 | 2379.30 | 2358.90 | 0.00 | T1 1.5R @ 2379.30 |
| Target hit | 2026-01-05 14:00:00 | 2374.20 | 2377.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-01-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:00:00 | 2396.20 | 2407.24 | 0.00 | ORB-short ORB[2406.60,2422.20] vol=3.0x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:35:00 | 2388.67 | 2403.22 | 0.00 | T1 1.5R @ 2388.67 |
| Target hit | 2026-01-13 14:50:00 | 2386.50 | 2384.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2026-01-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:10:00 | 2351.40 | 2369.12 | 0.00 | ORB-short ORB[2376.30,2398.90] vol=2.1x ATR=4.82 |
| Stop hit — per-position SL triggered | 2026-01-14 11:45:00 | 2356.22 | 2365.38 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:40:00 | 2427.90 | 2419.09 | 0.00 | ORB-long ORB[2410.40,2424.40] vol=1.7x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2436.63 | 2426.37 | 0.00 | T1 1.5R @ 2436.63 |
| Stop hit — per-position SL triggered | 2026-01-20 10:30:00 | 2427.90 | 2427.16 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 2422.20 | 2406.27 | 0.00 | ORB-long ORB[2376.80,2400.00] vol=1.8x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:55:00 | 2431.40 | 2411.43 | 0.00 | T1 1.5R @ 2431.40 |
| Stop hit — per-position SL triggered | 2026-01-23 12:50:00 | 2422.20 | 2416.62 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 2380.00 | 2367.55 | 0.00 | ORB-long ORB[2357.00,2375.00] vol=2.8x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 2374.34 | 2368.91 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 11:10:00 | 2309.70 | 2326.59 | 0.00 | ORB-short ORB[2325.10,2356.10] vol=4.5x ATR=7.07 |
| Stop hit — per-position SL triggered | 2026-02-02 11:30:00 | 2316.77 | 2322.58 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 2164.60 | 2152.48 | 0.00 | ORB-long ORB[2121.40,2152.80] vol=1.7x ATR=6.04 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 2158.56 | 2154.43 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 2147.50 | 2148.65 | 0.00 | ORB-short ORB[2148.80,2163.70] vol=7.6x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-04-16 10:35:00 | 2152.26 | 2148.70 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:55:00 | 2181.50 | 2161.08 | 0.00 | ORB-long ORB[2133.00,2157.90] vol=2.8x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 2190.88 | 2175.47 | 0.00 | T1 1.5R @ 2190.88 |
| Target hit | 2026-04-17 15:20:00 | 2241.60 | 2226.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 2266.60 | 2255.55 | 0.00 | ORB-long ORB[2227.20,2245.60] vol=1.7x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:55:00 | 2273.97 | 2257.13 | 0.00 | T1 1.5R @ 2273.97 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 2266.60 | 2258.20 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 2329.80 | 2353.45 | 0.00 | ORB-short ORB[2365.10,2388.60] vol=1.6x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:20:00 | 2321.48 | 2347.11 | 0.00 | T1 1.5R @ 2321.48 |
| Target hit | 2026-04-24 14:45:00 | 2326.60 | 2322.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 2296.50 | 2315.13 | 0.00 | ORB-short ORB[2309.10,2334.40] vol=1.9x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-04-28 10:10:00 | 2302.49 | 2314.09 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 2314.90 | 2304.96 | 0.00 | ORB-long ORB[2291.00,2312.50] vol=4.2x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:30:00 | 2322.01 | 2310.04 | 0.00 | T1 1.5R @ 2322.01 |
| Target hit | 2026-04-29 14:20:00 | 2323.30 | 2323.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 2352.00 | 2321.33 | 0.00 | ORB-long ORB[2281.30,2314.60] vol=7.9x ATR=12.28 |
| Stop hit — per-position SL triggered | 2026-04-30 10:35:00 | 2339.72 | 2332.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-12-08 11:05:00 | 2331.40 | 2025-12-08 12:10:00 | 2337.58 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-10 11:15:00 | 2300.60 | 2025-12-10 11:20:00 | 2304.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-12 10:05:00 | 2256.40 | 2025-12-12 10:10:00 | 2263.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-23 10:45:00 | 2296.40 | 2025-12-23 10:55:00 | 2293.52 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-31 10:30:00 | 2312.30 | 2025-12-31 12:20:00 | 2317.75 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-31 10:30:00 | 2312.30 | 2025-12-31 14:30:00 | 2312.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 11:05:00 | 2303.80 | 2026-01-01 11:25:00 | 2306.88 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2026-01-05 09:50:00 | 2370.60 | 2026-01-05 10:05:00 | 2379.30 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-05 09:50:00 | 2370.60 | 2026-01-05 14:00:00 | 2374.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-01-13 11:00:00 | 2396.20 | 2026-01-13 11:35:00 | 2388.67 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-01-13 11:00:00 | 2396.20 | 2026-01-13 14:50:00 | 2386.50 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-14 11:10:00 | 2351.40 | 2026-01-14 11:45:00 | 2356.22 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-20 09:40:00 | 2427.90 | 2026-01-20 10:15:00 | 2436.63 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-20 09:40:00 | 2427.90 | 2026-01-20 10:30:00 | 2427.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 11:05:00 | 2422.20 | 2026-01-23 11:55:00 | 2431.40 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-23 11:05:00 | 2422.20 | 2026-01-23 12:50:00 | 2422.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:05:00 | 2380.00 | 2026-02-01 11:10:00 | 2374.34 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-02 11:10:00 | 2309.70 | 2026-02-02 11:30:00 | 2316.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-10 10:45:00 | 2164.60 | 2026-04-10 10:50:00 | 2158.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-16 10:15:00 | 2147.50 | 2026-04-16 10:35:00 | 2152.26 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-17 09:55:00 | 2181.50 | 2026-04-17 10:00:00 | 2190.88 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-17 09:55:00 | 2181.50 | 2026-04-17 15:20:00 | 2241.60 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2026-04-21 10:45:00 | 2266.60 | 2026-04-21 10:55:00 | 2273.97 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-21 10:45:00 | 2266.60 | 2026-04-21 11:00:00 | 2266.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:55:00 | 2329.80 | 2026-04-24 11:20:00 | 2321.48 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-24 10:55:00 | 2329.80 | 2026-04-24 14:45:00 | 2326.60 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-04-28 10:05:00 | 2296.50 | 2026-04-28 10:10:00 | 2302.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-29 11:15:00 | 2314.90 | 2026-04-29 11:30:00 | 2322.01 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-04-29 11:15:00 | 2314.90 | 2026-04-29 14:20:00 | 2323.30 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-30 10:30:00 | 2352.00 | 2026-04-30 10:35:00 | 2339.72 | STOP_HIT | 1.00 | -0.52% |
