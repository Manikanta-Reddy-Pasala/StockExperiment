# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 12
- **Target hits / Stop hits / Partials:** 7 / 12 / 11
- **Avg / median % per leg:** 0.24% / 0.28%
- **Sum % (uncompounded):** 7.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.33% | 4.6% |
| BUY @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.33% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.16% | 2.6% |
| SELL @ 2nd Alert (retest1) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.16% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 18 | 60.0% | 7 | 12 | 11 | 0.24% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 2464.40 | 2458.79 | 0.00 | ORB-long ORB[2450.30,2463.60] vol=1.5x ATR=4.62 |
| Stop hit — per-position SL triggered | 2026-02-11 09:40:00 | 2459.78 | 2459.67 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 2437.90 | 2456.96 | 0.00 | ORB-short ORB[2455.60,2476.50] vol=6.8x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:25:00 | 2425.19 | 2448.39 | 0.00 | T1 1.5R @ 2425.19 |
| Stop hit — per-position SL triggered | 2026-02-12 10:35:00 | 2437.90 | 2438.48 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 2310.20 | 2301.99 | 0.00 | ORB-long ORB[2278.30,2292.20] vol=3.8x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:00:00 | 2316.85 | 2303.55 | 0.00 | T1 1.5R @ 2316.85 |
| Target hit | 2026-02-20 13:40:00 | 2313.00 | 2316.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 2344.70 | 2336.35 | 0.00 | ORB-long ORB[2320.80,2339.80] vol=1.6x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 2340.02 | 2337.26 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 2358.70 | 2368.51 | 0.00 | ORB-short ORB[2363.40,2379.00] vol=2.0x ATR=4.41 |
| Stop hit — per-position SL triggered | 2026-02-25 11:35:00 | 2363.11 | 2366.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:20:00 | 2236.90 | 2245.36 | 0.00 | ORB-short ORB[2248.00,2277.80] vol=1.7x ATR=6.46 |
| Stop hit — per-position SL triggered | 2026-03-05 12:05:00 | 2243.36 | 2240.91 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 2196.90 | 2204.04 | 0.00 | ORB-short ORB[2202.00,2224.00] vol=3.6x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 12:10:00 | 2189.92 | 2200.88 | 0.00 | T1 1.5R @ 2189.92 |
| Target hit | 2026-03-10 14:10:00 | 2195.80 | 2194.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 2157.00 | 2169.35 | 0.00 | ORB-short ORB[2159.00,2172.90] vol=3.7x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:35:00 | 2148.59 | 2162.64 | 0.00 | T1 1.5R @ 2148.59 |
| Stop hit — per-position SL triggered | 2026-03-17 13:05:00 | 2157.00 | 2159.64 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 2156.60 | 2160.56 | 0.00 | ORB-short ORB[2157.50,2175.00] vol=6.8x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:10:00 | 2150.49 | 2159.35 | 0.00 | T1 1.5R @ 2150.49 |
| Target hit | 2026-03-18 15:20:00 | 2132.90 | 2144.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 2119.60 | 2111.00 | 0.00 | ORB-long ORB[2086.00,2115.30] vol=2.0x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:10:00 | 2128.98 | 2114.68 | 0.00 | T1 1.5R @ 2128.98 |
| Target hit | 2026-03-25 14:25:00 | 2135.10 | 2135.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:50:00 | 2099.70 | 2104.77 | 0.00 | ORB-short ORB[2102.30,2127.00] vol=2.8x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:30:00 | 2092.00 | 2102.79 | 0.00 | T1 1.5R @ 2092.00 |
| Stop hit — per-position SL triggered | 2026-03-27 12:05:00 | 2099.70 | 2102.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 2164.60 | 2152.48 | 0.00 | ORB-long ORB[2121.40,2152.80] vol=1.7x ATR=6.04 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 2158.56 | 2154.43 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 2147.50 | 2148.65 | 0.00 | ORB-short ORB[2148.80,2163.70] vol=7.6x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-04-16 10:35:00 | 2152.26 | 2148.70 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:55:00 | 2181.50 | 2161.08 | 0.00 | ORB-long ORB[2133.00,2157.90] vol=2.8x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 2190.88 | 2175.47 | 0.00 | T1 1.5R @ 2190.88 |
| Target hit | 2026-04-17 15:20:00 | 2241.60 | 2226.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 2266.60 | 2255.55 | 0.00 | ORB-long ORB[2227.20,2245.60] vol=1.7x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:55:00 | 2273.97 | 2257.13 | 0.00 | T1 1.5R @ 2273.97 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 2266.60 | 2258.20 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 2329.80 | 2353.45 | 0.00 | ORB-short ORB[2365.10,2388.60] vol=1.6x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:20:00 | 2321.48 | 2347.11 | 0.00 | T1 1.5R @ 2321.48 |
| Target hit | 2026-04-24 14:45:00 | 2326.60 | 2322.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 2296.50 | 2315.13 | 0.00 | ORB-short ORB[2309.10,2334.40] vol=1.9x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-04-28 10:10:00 | 2302.49 | 2314.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 2314.90 | 2304.96 | 0.00 | ORB-long ORB[2291.00,2312.50] vol=4.2x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:30:00 | 2322.01 | 2310.04 | 0.00 | T1 1.5R @ 2322.01 |
| Target hit | 2026-04-29 14:20:00 | 2323.30 | 2323.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 2352.00 | 2321.33 | 0.00 | ORB-long ORB[2281.30,2314.60] vol=7.9x ATR=12.28 |
| Stop hit — per-position SL triggered | 2026-04-30 10:35:00 | 2339.72 | 2332.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:30:00 | 2464.40 | 2026-02-11 09:40:00 | 2459.78 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-12 10:20:00 | 2437.90 | 2026-02-12 10:25:00 | 2425.19 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-12 10:20:00 | 2437.90 | 2026-02-12 10:35:00 | 2437.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:35:00 | 2310.20 | 2026-02-20 11:00:00 | 2316.85 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-20 10:35:00 | 2310.20 | 2026-02-20 13:40:00 | 2313.00 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-02-23 10:55:00 | 2344.70 | 2026-02-23 11:05:00 | 2340.02 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 10:55:00 | 2358.70 | 2026-02-25 11:35:00 | 2363.11 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-03-05 10:20:00 | 2236.90 | 2026-03-05 12:05:00 | 2243.36 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 11:15:00 | 2196.90 | 2026-03-10 12:10:00 | 2189.92 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-03-10 11:15:00 | 2196.90 | 2026-03-10 14:10:00 | 2195.80 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2026-03-17 10:55:00 | 2157.00 | 2026-03-17 11:35:00 | 2148.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-17 10:55:00 | 2157.00 | 2026-03-17 13:05:00 | 2157.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-18 11:05:00 | 2156.60 | 2026-03-18 11:10:00 | 2150.49 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-03-18 11:05:00 | 2156.60 | 2026-03-18 15:20:00 | 2132.90 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2026-03-25 09:45:00 | 2119.60 | 2026-03-25 10:10:00 | 2128.98 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-25 09:45:00 | 2119.60 | 2026-03-25 14:25:00 | 2135.10 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-27 10:50:00 | 2099.70 | 2026-03-27 11:30:00 | 2092.00 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-27 10:50:00 | 2099.70 | 2026-03-27 12:05:00 | 2099.70 | STOP_HIT | 0.50 | 0.00% |
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
