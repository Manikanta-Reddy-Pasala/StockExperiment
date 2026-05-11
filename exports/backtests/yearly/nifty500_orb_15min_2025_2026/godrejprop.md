# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1874.80
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
| ENTRY1 | 76 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 12 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 64
- **Target hits / Stop hits / Partials:** 12 / 64 / 33
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 17.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 20 | 38.5% | 3 | 32 | 17 | 0.16% | 8.2% |
| BUY @ 2nd Alert (retest1) | 52 | 20 | 38.5% | 3 | 32 | 17 | 0.16% | 8.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 25 | 43.9% | 9 | 32 | 16 | 0.17% | 9.5% |
| SELL @ 2nd Alert (retest1) | 57 | 25 | 43.9% | 9 | 32 | 16 | 0.17% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 45 | 41.3% | 12 | 64 | 33 | 0.16% | 17.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:20:00 | 2124.40 | 2111.45 | 0.00 | ORB-long ORB[2087.20,2114.00] vol=4.9x ATR=7.29 |
| Stop hit — per-position SL triggered | 2025-05-14 10:25:00 | 2117.11 | 2111.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:35:00 | 2211.50 | 2197.85 | 0.00 | ORB-long ORB[2180.00,2203.60] vol=2.3x ATR=6.52 |
| Stop hit — per-position SL triggered | 2025-05-26 09:40:00 | 2204.98 | 2198.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:50:00 | 2214.70 | 2234.71 | 0.00 | ORB-short ORB[2230.90,2253.90] vol=1.6x ATR=9.01 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 2223.71 | 2229.84 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 2267.40 | 2261.31 | 0.00 | ORB-long ORB[2249.60,2267.00] vol=1.9x ATR=7.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:40:00 | 2277.97 | 2265.27 | 0.00 | T1 1.5R @ 2277.97 |
| Stop hit — per-position SL triggered | 2025-05-28 09:55:00 | 2267.40 | 2268.25 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:15:00 | 2209.00 | 2228.82 | 0.00 | ORB-short ORB[2239.40,2259.70] vol=3.1x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:30:00 | 2196.81 | 2225.24 | 0.00 | T1 1.5R @ 2196.81 |
| Stop hit — per-position SL triggered | 2025-05-30 10:55:00 | 2209.00 | 2220.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 2260.30 | 2279.05 | 0.00 | ORB-short ORB[2275.00,2300.00] vol=2.0x ATR=9.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:45:00 | 2246.29 | 2271.26 | 0.00 | T1 1.5R @ 2246.29 |
| Stop hit — per-position SL triggered | 2025-06-04 09:55:00 | 2260.30 | 2268.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:40:00 | 2291.90 | 2266.33 | 0.00 | ORB-long ORB[2249.60,2279.90] vol=3.7x ATR=7.66 |
| Stop hit — per-position SL triggered | 2025-06-05 10:50:00 | 2284.24 | 2271.07 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:30:00 | 2448.20 | 2461.56 | 0.00 | ORB-short ORB[2455.30,2473.90] vol=1.6x ATR=6.35 |
| Stop hit — per-position SL triggered | 2025-06-19 10:45:00 | 2454.55 | 2458.31 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:30:00 | 2430.80 | 2409.89 | 0.00 | ORB-long ORB[2378.90,2402.80] vol=1.5x ATR=8.77 |
| Stop hit — per-position SL triggered | 2025-06-20 10:45:00 | 2422.03 | 2411.71 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:00:00 | 2371.90 | 2384.50 | 0.00 | ORB-short ORB[2385.60,2410.40] vol=1.9x ATR=6.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:15:00 | 2361.80 | 2382.71 | 0.00 | T1 1.5R @ 2361.80 |
| Stop hit — per-position SL triggered | 2025-06-26 12:30:00 | 2371.90 | 2376.95 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 09:40:00 | 2373.90 | 2389.44 | 0.00 | ORB-short ORB[2384.00,2408.00] vol=1.5x ATR=7.72 |
| Stop hit — per-position SL triggered | 2025-06-27 10:50:00 | 2381.62 | 2381.06 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 2293.20 | 2279.17 | 0.00 | ORB-long ORB[2263.40,2284.90] vol=1.9x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 09:40:00 | 2301.70 | 2283.94 | 0.00 | T1 1.5R @ 2301.70 |
| Stop hit — per-position SL triggered | 2025-07-04 11:35:00 | 2293.20 | 2298.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 2268.00 | 2278.92 | 0.00 | ORB-short ORB[2273.30,2295.90] vol=2.4x ATR=8.07 |
| Stop hit — per-position SL triggered | 2025-07-08 09:40:00 | 2276.07 | 2278.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:00:00 | 2270.20 | 2262.25 | 0.00 | ORB-long ORB[2245.50,2268.60] vol=1.6x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-07-10 10:50:00 | 2264.73 | 2264.97 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:25:00 | 2239.30 | 2243.99 | 0.00 | ORB-short ORB[2240.00,2261.40] vol=1.6x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:30:00 | 2232.40 | 2242.67 | 0.00 | T1 1.5R @ 2232.40 |
| Target hit | 2025-07-11 15:20:00 | 2200.40 | 2225.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 10:25:00 | 2251.20 | 2253.64 | 0.00 | ORB-short ORB[2256.00,2271.60] vol=6.4x ATR=6.60 |
| Stop hit — per-position SL triggered | 2025-07-15 10:30:00 | 2257.80 | 2253.87 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:30:00 | 2358.90 | 2373.84 | 0.00 | ORB-short ORB[2363.10,2384.50] vol=1.7x ATR=6.81 |
| Stop hit — per-position SL triggered | 2025-07-21 09:50:00 | 2365.71 | 2370.22 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:00:00 | 2373.20 | 2374.18 | 0.00 | ORB-short ORB[2384.00,2407.90] vol=1.7x ATR=7.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:55:00 | 2361.45 | 2370.45 | 0.00 | T1 1.5R @ 2361.45 |
| Target hit | 2025-07-22 14:00:00 | 2372.00 | 2368.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-07-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:05:00 | 2311.80 | 2321.61 | 0.00 | ORB-short ORB[2312.30,2339.90] vol=1.7x ATR=5.88 |
| Stop hit — per-position SL triggered | 2025-07-24 12:10:00 | 2317.68 | 2314.86 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:35:00 | 2127.40 | 2143.01 | 0.00 | ORB-short ORB[2139.00,2166.70] vol=2.0x ATR=6.56 |
| Stop hit — per-position SL triggered | 2025-07-30 10:50:00 | 2133.96 | 2142.02 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:30:00 | 2006.20 | 2031.36 | 0.00 | ORB-short ORB[2044.40,2072.70] vol=1.8x ATR=8.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:45:00 | 1993.56 | 2018.53 | 0.00 | T1 1.5R @ 1993.56 |
| Stop hit — per-position SL triggered | 2025-08-08 11:10:00 | 2006.20 | 2015.31 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:00:00 | 1954.90 | 1960.58 | 0.00 | ORB-short ORB[1957.70,1973.90] vol=2.6x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:40:00 | 1947.70 | 1958.18 | 0.00 | T1 1.5R @ 1947.70 |
| Target hit | 2025-08-14 15:20:00 | 1936.00 | 1947.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 1978.30 | 1968.24 | 0.00 | ORB-long ORB[1953.90,1974.00] vol=3.3x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:40:00 | 1987.55 | 1973.00 | 0.00 | T1 1.5R @ 1987.55 |
| Target hit | 2025-08-18 12:10:00 | 2000.90 | 2003.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:40:00 | 1980.50 | 1992.30 | 0.00 | ORB-short ORB[1991.00,2015.10] vol=2.0x ATR=5.85 |
| Stop hit — per-position SL triggered | 2025-08-19 09:55:00 | 1986.35 | 1989.60 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 11:00:00 | 2090.40 | 2079.13 | 0.00 | ORB-long ORB[2062.00,2079.80] vol=2.1x ATR=5.48 |
| Stop hit — per-position SL triggered | 2025-08-25 11:10:00 | 2084.92 | 2079.90 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 2029.00 | 2047.07 | 0.00 | ORB-short ORB[2041.00,2070.00] vol=2.1x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 2035.98 | 2040.34 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 2002.00 | 1988.28 | 0.00 | ORB-long ORB[1972.00,1989.50] vol=3.1x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:05:00 | 2011.31 | 1997.16 | 0.00 | T1 1.5R @ 2011.31 |
| Stop hit — per-position SL triggered | 2025-09-03 10:45:00 | 2002.00 | 2000.18 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:45:00 | 2045.70 | 2034.21 | 0.00 | ORB-long ORB[2016.00,2041.60] vol=4.8x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 09:50:00 | 2054.88 | 2037.75 | 0.00 | T1 1.5R @ 2054.88 |
| Stop hit — per-position SL triggered | 2025-09-04 10:10:00 | 2045.70 | 2040.66 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 1990.00 | 2005.93 | 0.00 | ORB-short ORB[2003.00,2015.70] vol=2.1x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:20:00 | 1981.69 | 2004.43 | 0.00 | T1 1.5R @ 1981.69 |
| Target hit | 2025-09-05 15:20:00 | 1963.80 | 1969.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:55:00 | 1975.10 | 1958.41 | 0.00 | ORB-long ORB[1945.60,1972.40] vol=2.0x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:00:00 | 1981.53 | 1959.96 | 0.00 | T1 1.5R @ 1981.53 |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1975.10 | 1973.58 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:50:00 | 2015.20 | 2007.06 | 0.00 | ORB-long ORB[1993.90,2012.30] vol=1.8x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-09-11 09:55:00 | 2009.33 | 2007.32 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 09:55:00 | 1996.30 | 2002.89 | 0.00 | ORB-short ORB[1997.10,2013.60] vol=3.1x ATR=5.86 |
| Stop hit — per-position SL triggered | 2025-09-12 10:00:00 | 2002.16 | 2002.93 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 2049.00 | 2041.07 | 0.00 | ORB-long ORB[2028.20,2044.90] vol=1.9x ATR=5.94 |
| Stop hit — per-position SL triggered | 2025-09-16 10:50:00 | 2043.06 | 2046.54 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 2103.50 | 2089.36 | 0.00 | ORB-long ORB[2084.90,2094.00] vol=5.9x ATR=4.69 |
| Stop hit — per-position SL triggered | 2025-09-18 11:25:00 | 2098.81 | 2090.13 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:55:00 | 2081.50 | 2094.69 | 0.00 | ORB-short ORB[2093.00,2115.30] vol=1.7x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:35:00 | 2072.21 | 2088.39 | 0.00 | T1 1.5R @ 2072.21 |
| Target hit | 2025-09-24 15:20:00 | 2022.00 | 2043.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:00:00 | 1986.60 | 2005.57 | 0.00 | ORB-short ORB[1999.40,2022.10] vol=2.2x ATR=5.53 |
| Stop hit — per-position SL triggered | 2025-09-25 11:20:00 | 1992.13 | 2003.78 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:15:00 | 2038.80 | 2055.10 | 0.00 | ORB-short ORB[2046.80,2065.30] vol=1.7x ATR=5.44 |
| Stop hit — per-position SL triggered | 2025-10-07 11:45:00 | 2044.24 | 2053.53 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 2066.70 | 2054.09 | 0.00 | ORB-long ORB[2036.00,2056.10] vol=3.3x ATR=7.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:35:00 | 2077.61 | 2059.96 | 0.00 | T1 1.5R @ 2077.61 |
| Stop hit — per-position SL triggered | 2025-10-10 09:45:00 | 2066.70 | 2063.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 2058.10 | 2072.17 | 0.00 | ORB-short ORB[2061.10,2090.00] vol=1.6x ATR=8.30 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 2066.40 | 2070.28 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 2060.60 | 2080.90 | 0.00 | ORB-short ORB[2086.30,2105.00] vol=1.6x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 12:45:00 | 2052.43 | 2074.46 | 0.00 | T1 1.5R @ 2052.43 |
| Stop hit — per-position SL triggered | 2025-10-14 12:55:00 | 2060.60 | 2074.19 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:05:00 | 2163.90 | 2147.81 | 0.00 | ORB-long ORB[2135.60,2157.20] vol=2.2x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:15:00 | 2175.18 | 2159.96 | 0.00 | T1 1.5R @ 2175.18 |
| Target hit | 2025-10-16 15:20:00 | 2205.30 | 2192.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2025-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:35:00 | 2321.00 | 2303.50 | 0.00 | ORB-long ORB[2278.70,2305.00] vol=1.9x ATR=9.21 |
| Stop hit — per-position SL triggered | 2025-10-23 10:50:00 | 2311.79 | 2314.76 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:05:00 | 2336.10 | 2321.68 | 0.00 | ORB-long ORB[2306.70,2333.70] vol=2.6x ATR=8.70 |
| Stop hit — per-position SL triggered | 2025-10-27 10:10:00 | 2327.40 | 2322.24 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 2296.00 | 2306.85 | 0.00 | ORB-short ORB[2297.80,2330.00] vol=2.4x ATR=6.20 |
| Stop hit — per-position SL triggered | 2025-10-28 11:10:00 | 2302.20 | 2306.27 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:55:00 | 2312.00 | 2303.40 | 0.00 | ORB-long ORB[2295.00,2310.90] vol=2.3x ATR=7.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:20:00 | 2323.22 | 2307.82 | 0.00 | T1 1.5R @ 2323.22 |
| Stop hit — per-position SL triggered | 2025-11-04 11:05:00 | 2312.00 | 2311.04 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:25:00 | 2164.20 | 2149.74 | 0.00 | ORB-long ORB[2131.10,2152.90] vol=5.9x ATR=8.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:20:00 | 2176.93 | 2153.21 | 0.00 | T1 1.5R @ 2176.93 |
| Stop hit — per-position SL triggered | 2025-11-10 12:05:00 | 2164.20 | 2154.50 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:50:00 | 2139.00 | 2151.36 | 0.00 | ORB-short ORB[2156.90,2184.70] vol=2.5x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-11-11 11:40:00 | 2144.05 | 2148.24 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:45:00 | 2216.00 | 2201.98 | 0.00 | ORB-long ORB[2187.10,2202.90] vol=3.9x ATR=7.57 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 2208.43 | 2212.51 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 2117.40 | 2132.87 | 0.00 | ORB-short ORB[2122.70,2149.70] vol=2.1x ATR=6.51 |
| Stop hit — per-position SL triggered | 2025-11-19 11:50:00 | 2123.91 | 2128.97 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 2119.80 | 2136.44 | 0.00 | ORB-short ORB[2130.50,2152.80] vol=1.9x ATR=6.59 |
| Stop hit — per-position SL triggered | 2025-11-20 10:45:00 | 2126.39 | 2128.75 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:45:00 | 2101.00 | 2090.13 | 0.00 | ORB-long ORB[2080.60,2095.00] vol=1.7x ATR=6.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 10:05:00 | 2110.47 | 2095.51 | 0.00 | T1 1.5R @ 2110.47 |
| Stop hit — per-position SL triggered | 2025-11-26 10:20:00 | 2101.00 | 2096.46 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:35:00 | 2110.30 | 2115.78 | 0.00 | ORB-short ORB[2111.40,2127.90] vol=1.7x ATR=4.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:40:00 | 2102.83 | 2114.10 | 0.00 | T1 1.5R @ 2102.83 |
| Target hit | 2025-12-01 14:30:00 | 2104.50 | 2104.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2025-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:05:00 | 2066.50 | 2081.29 | 0.00 | ORB-short ORB[2086.60,2097.70] vol=1.6x ATR=5.85 |
| Stop hit — per-position SL triggered | 2025-12-03 10:25:00 | 2072.35 | 2075.45 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 2079.90 | 2073.09 | 0.00 | ORB-long ORB[2056.00,2078.80] vol=1.8x ATR=5.57 |
| Stop hit — per-position SL triggered | 2025-12-04 09:35:00 | 2074.33 | 2073.62 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 2093.70 | 2077.77 | 0.00 | ORB-long ORB[2069.30,2086.00] vol=2.6x ATR=7.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:05:00 | 2104.61 | 2083.98 | 0.00 | T1 1.5R @ 2104.61 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 2093.70 | 2092.87 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:35:00 | 1988.30 | 1973.49 | 0.00 | ORB-long ORB[1962.20,1976.20] vol=2.3x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:50:00 | 1999.49 | 1979.71 | 0.00 | T1 1.5R @ 1999.49 |
| Target hit | 2025-12-09 15:20:00 | 2039.90 | 2023.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2018.30 | 2032.76 | 0.00 | ORB-short ORB[2024.70,2049.20] vol=1.9x ATR=5.97 |
| Stop hit — per-position SL triggered | 2025-12-10 11:20:00 | 2024.27 | 2028.83 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:40:00 | 2049.40 | 2056.98 | 0.00 | ORB-short ORB[2055.40,2079.80] vol=5.4x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-12-15 11:45:00 | 2054.90 | 2053.27 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 2041.40 | 2049.62 | 0.00 | ORB-short ORB[2048.10,2077.00] vol=2.2x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:35:00 | 2035.44 | 2047.93 | 0.00 | T1 1.5R @ 2035.44 |
| Target hit | 2025-12-16 15:20:00 | 2020.00 | 2034.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 2011.00 | 1998.71 | 0.00 | ORB-long ORB[1978.50,1995.00] vol=1.8x ATR=4.32 |
| Stop hit — per-position SL triggered | 2025-12-31 11:10:00 | 2006.68 | 2000.57 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 2053.30 | 2038.10 | 0.00 | ORB-long ORB[2018.00,2034.40] vol=1.7x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:55:00 | 2060.57 | 2041.15 | 0.00 | T1 1.5R @ 2060.57 |
| Stop hit — per-position SL triggered | 2026-01-02 11:40:00 | 2053.30 | 2047.42 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 2080.70 | 2071.65 | 0.00 | ORB-long ORB[2057.70,2076.70] vol=2.4x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-01-05 09:40:00 | 2075.61 | 2074.41 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 2137.00 | 2117.18 | 0.00 | ORB-long ORB[2106.00,2130.00] vol=3.1x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:15:00 | 2148.09 | 2124.50 | 0.00 | T1 1.5R @ 2148.09 |
| Stop hit — per-position SL triggered | 2026-01-07 10:20:00 | 2137.00 | 2125.44 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:05:00 | 1772.10 | 1784.21 | 0.00 | ORB-short ORB[1784.50,1804.70] vol=1.6x ATR=9.00 |
| Stop hit — per-position SL triggered | 2026-01-20 10:10:00 | 1781.10 | 1783.99 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 1821.40 | 1818.55 | 0.00 | ORB-long ORB[1801.50,1820.60] vol=3.9x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:45:00 | 1831.00 | 1819.75 | 0.00 | T1 1.5R @ 1831.00 |
| Stop hit — per-position SL triggered | 2026-02-10 13:00:00 | 1821.40 | 1821.07 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:00:00 | 1818.40 | 1820.28 | 0.00 | ORB-short ORB[1819.80,1829.90] vol=1.9x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1823.12 | 1820.05 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 1869.50 | 1874.56 | 0.00 | ORB-short ORB[1869.70,1887.00] vol=6.8x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 1875.85 | 1874.37 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 1803.00 | 1807.93 | 0.00 | ORB-short ORB[1806.30,1824.90] vol=2.4x ATR=7.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:00:00 | 1791.73 | 1805.96 | 0.00 | T1 1.5R @ 1791.73 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 1803.00 | 1805.52 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1826.10 | 1834.23 | 0.00 | ORB-short ORB[1827.50,1848.90] vol=2.2x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:20:00 | 1816.40 | 1832.65 | 0.00 | T1 1.5R @ 1816.40 |
| Target hit | 2026-02-23 14:40:00 | 1821.10 | 1820.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 1804.30 | 1807.05 | 0.00 | ORB-short ORB[1805.10,1831.20] vol=3.9x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 1795.91 | 1804.84 | 0.00 | T1 1.5R @ 1795.91 |
| Target hit | 2026-02-24 15:20:00 | 1777.00 | 1778.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1786.00 | 1794.93 | 0.00 | ORB-short ORB[1788.00,1810.00] vol=1.5x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 1778.17 | 1792.97 | 0.00 | T1 1.5R @ 1778.17 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 1786.00 | 1788.53 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 1578.30 | 1588.74 | 0.00 | ORB-short ORB[1586.90,1606.00] vol=1.9x ATR=6.50 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 1584.80 | 1588.04 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1601.80 | 1583.07 | 0.00 | ORB-long ORB[1562.70,1586.00] vol=1.7x ATR=8.45 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 1593.35 | 1587.82 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 1813.10 | 1805.49 | 0.00 | ORB-long ORB[1796.10,1812.70] vol=1.5x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 1807.26 | 1805.82 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 1843.90 | 1835.01 | 0.00 | ORB-long ORB[1822.70,1839.30] vol=2.7x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1838.85 | 1835.16 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 1861.50 | 1849.29 | 0.00 | ORB-long ORB[1831.00,1853.90] vol=2.8x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 1872.26 | 1855.53 | 0.00 | T1 1.5R @ 1872.26 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 1861.50 | 1857.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:20:00 | 2124.40 | 2025-05-14 10:25:00 | 2117.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-26 09:35:00 | 2211.50 | 2025-05-26 09:40:00 | 2204.98 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-27 09:50:00 | 2214.70 | 2025-05-27 10:10:00 | 2223.71 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-05-28 09:30:00 | 2267.40 | 2025-05-28 09:40:00 | 2277.97 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-05-28 09:30:00 | 2267.40 | 2025-05-28 09:55:00 | 2267.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 10:15:00 | 2209.00 | 2025-05-30 10:30:00 | 2196.81 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-05-30 10:15:00 | 2209.00 | 2025-05-30 10:55:00 | 2209.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 2260.30 | 2025-06-04 09:45:00 | 2246.29 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-06-04 09:35:00 | 2260.30 | 2025-06-04 09:55:00 | 2260.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 10:40:00 | 2291.90 | 2025-06-05 10:50:00 | 2284.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-19 10:30:00 | 2448.20 | 2025-06-19 10:45:00 | 2454.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-20 10:30:00 | 2430.80 | 2025-06-20 10:45:00 | 2422.03 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-06-26 11:00:00 | 2371.90 | 2025-06-26 11:15:00 | 2361.80 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-26 11:00:00 | 2371.90 | 2025-06-26 12:30:00 | 2371.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-27 09:40:00 | 2373.90 | 2025-06-27 10:50:00 | 2381.62 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-04 09:30:00 | 2293.20 | 2025-07-04 09:40:00 | 2301.70 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-04 09:30:00 | 2293.20 | 2025-07-04 11:35:00 | 2293.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 09:35:00 | 2268.00 | 2025-07-08 09:40:00 | 2276.07 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-10 10:00:00 | 2270.20 | 2025-07-10 10:50:00 | 2264.73 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-11 10:25:00 | 2239.30 | 2025-07-11 10:30:00 | 2232.40 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-11 10:25:00 | 2239.30 | 2025-07-11 15:20:00 | 2200.40 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2025-07-15 10:25:00 | 2251.20 | 2025-07-15 10:30:00 | 2257.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-21 09:30:00 | 2358.90 | 2025-07-21 09:50:00 | 2365.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-22 10:00:00 | 2373.20 | 2025-07-22 11:55:00 | 2361.45 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-22 10:00:00 | 2373.20 | 2025-07-22 14:00:00 | 2372.00 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-07-24 11:05:00 | 2311.80 | 2025-07-24 12:10:00 | 2317.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-30 10:35:00 | 2127.40 | 2025-07-30 10:50:00 | 2133.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-08 10:30:00 | 2006.20 | 2025-08-08 10:45:00 | 1993.56 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-08-08 10:30:00 | 2006.20 | 2025-08-08 11:10:00 | 2006.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 11:00:00 | 1954.90 | 2025-08-14 11:40:00 | 1947.70 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-14 11:00:00 | 1954.90 | 2025-08-14 15:20:00 | 1936.00 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-08-18 09:30:00 | 1978.30 | 2025-08-18 09:40:00 | 1987.55 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-18 09:30:00 | 1978.30 | 2025-08-18 12:10:00 | 2000.90 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-08-19 09:40:00 | 1980.50 | 2025-08-19 09:55:00 | 1986.35 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-25 11:00:00 | 2090.40 | 2025-08-25 11:10:00 | 2084.92 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-26 09:35:00 | 2029.00 | 2025-08-26 09:55:00 | 2035.98 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-03 09:45:00 | 2002.00 | 2025-09-03 10:05:00 | 2011.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-09-03 09:45:00 | 2002.00 | 2025-09-03 10:45:00 | 2002.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-04 09:45:00 | 2045.70 | 2025-09-04 09:50:00 | 2054.88 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-04 09:45:00 | 2045.70 | 2025-09-04 10:10:00 | 2045.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:15:00 | 1990.00 | 2025-09-05 10:20:00 | 1981.69 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-09-05 10:15:00 | 1990.00 | 2025-09-05 15:20:00 | 1963.80 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-09-10 10:55:00 | 1975.10 | 2025-09-10 11:00:00 | 1981.53 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-10 10:55:00 | 1975.10 | 2025-09-10 13:15:00 | 1975.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 09:50:00 | 2015.20 | 2025-09-11 09:55:00 | 2009.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-12 09:55:00 | 1996.30 | 2025-09-12 10:00:00 | 2002.16 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-16 09:30:00 | 2049.00 | 2025-09-16 10:50:00 | 2043.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-18 11:15:00 | 2103.50 | 2025-09-18 11:25:00 | 2098.81 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-24 09:55:00 | 2081.50 | 2025-09-24 10:35:00 | 2072.21 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-09-24 09:55:00 | 2081.50 | 2025-09-24 15:20:00 | 2022.00 | TARGET_HIT | 0.50 | 2.86% |
| SELL | retest1 | 2025-09-25 11:00:00 | 1986.60 | 2025-09-25 11:20:00 | 1992.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-07 11:15:00 | 2038.80 | 2025-10-07 11:45:00 | 2044.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-10 09:30:00 | 2066.70 | 2025-10-10 09:35:00 | 2077.61 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-10 09:30:00 | 2066.70 | 2025-10-10 09:45:00 | 2066.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 09:30:00 | 2058.10 | 2025-10-13 09:35:00 | 2066.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-14 11:15:00 | 2060.60 | 2025-10-14 12:45:00 | 2052.43 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-14 11:15:00 | 2060.60 | 2025-10-14 12:55:00 | 2060.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 10:05:00 | 2163.90 | 2025-10-16 11:15:00 | 2175.18 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-16 10:05:00 | 2163.90 | 2025-10-16 15:20:00 | 2205.30 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2025-10-23 09:35:00 | 2321.00 | 2025-10-23 10:50:00 | 2311.79 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-10-27 10:05:00 | 2336.10 | 2025-10-27 10:10:00 | 2327.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-28 10:50:00 | 2296.00 | 2025-10-28 11:10:00 | 2302.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-04 09:55:00 | 2312.00 | 2025-11-04 10:20:00 | 2323.22 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-11-04 09:55:00 | 2312.00 | 2025-11-04 11:05:00 | 2312.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:25:00 | 2164.20 | 2025-11-10 11:20:00 | 2176.93 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-11-10 10:25:00 | 2164.20 | 2025-11-10 12:05:00 | 2164.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:50:00 | 2139.00 | 2025-11-11 11:40:00 | 2144.05 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-13 09:45:00 | 2216.00 | 2025-11-13 10:15:00 | 2208.43 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-19 10:55:00 | 2117.40 | 2025-11-19 11:50:00 | 2123.91 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-20 09:35:00 | 2119.80 | 2025-11-20 10:45:00 | 2126.39 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-26 09:45:00 | 2101.00 | 2025-11-26 10:05:00 | 2110.47 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-26 09:45:00 | 2101.00 | 2025-11-26 10:20:00 | 2101.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:35:00 | 2110.30 | 2025-12-01 11:40:00 | 2102.83 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-01 10:35:00 | 2110.30 | 2025-12-01 14:30:00 | 2104.50 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-03 10:05:00 | 2066.50 | 2025-12-03 10:25:00 | 2072.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-04 09:30:00 | 2079.90 | 2025-12-04 09:35:00 | 2074.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-05 10:00:00 | 2093.70 | 2025-12-05 10:05:00 | 2104.61 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-05 10:00:00 | 2093.70 | 2025-12-05 10:25:00 | 2093.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-09 10:35:00 | 1988.30 | 2025-12-09 10:50:00 | 1999.49 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-09 10:35:00 | 1988.30 | 2025-12-09 15:20:00 | 2039.90 | TARGET_HIT | 0.50 | 2.60% |
| SELL | retest1 | 2025-12-10 10:45:00 | 2018.30 | 2025-12-10 11:20:00 | 2024.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-15 10:40:00 | 2049.40 | 2025-12-15 11:45:00 | 2054.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-16 11:00:00 | 2041.40 | 2025-12-16 11:35:00 | 2035.44 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-16 11:00:00 | 2041.40 | 2025-12-16 15:20:00 | 2020.00 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-12-31 10:55:00 | 2011.00 | 2025-12-31 11:10:00 | 2006.68 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 10:40:00 | 2053.30 | 2026-01-02 10:55:00 | 2060.57 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-01-02 10:40:00 | 2053.30 | 2026-01-02 11:40:00 | 2053.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 09:30:00 | 2080.70 | 2026-01-05 09:40:00 | 2075.61 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-07 10:05:00 | 2137.00 | 2026-01-07 10:15:00 | 2148.09 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-01-07 10:05:00 | 2137.00 | 2026-01-07 10:20:00 | 2137.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 10:05:00 | 1772.10 | 2026-01-20 10:10:00 | 1781.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-02-10 10:15:00 | 1821.40 | 2026-02-10 11:45:00 | 1831.00 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-10 10:15:00 | 1821.40 | 2026-02-10 13:00:00 | 1821.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:00:00 | 1818.40 | 2026-02-11 10:15:00 | 1823.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 09:40:00 | 1869.50 | 2026-02-19 09:50:00 | 1875.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-20 10:45:00 | 1803.00 | 2026-02-20 11:00:00 | 1791.73 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-02-20 10:45:00 | 1803.00 | 2026-02-20 11:30:00 | 1803.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1826.10 | 2026-02-23 11:20:00 | 1816.40 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1826.10 | 2026-02-23 14:40:00 | 1821.10 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1804.30 | 2026-02-24 11:45:00 | 1795.91 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1804.30 | 2026-02-24 15:20:00 | 1777.00 | TARGET_HIT | 0.50 | 1.51% |
| SELL | retest1 | 2026-02-26 10:45:00 | 1786.00 | 2026-02-26 10:55:00 | 1778.17 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-26 10:45:00 | 1786.00 | 2026-02-26 11:25:00 | 1786.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 1578.30 | 2026-03-13 10:30:00 | 1584.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1601.80 | 2026-03-20 09:50:00 | 1593.35 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-22 10:35:00 | 1813.10 | 2026-04-22 10:45:00 | 1807.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 11:15:00 | 1843.90 | 2026-04-28 11:20:00 | 1838.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1861.50 | 2026-04-29 10:00:00 | 1872.26 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1861.50 | 2026-04-29 10:45:00 | 1861.50 | STOP_HIT | 0.50 | 0.00% |
