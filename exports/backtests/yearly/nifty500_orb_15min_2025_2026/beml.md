# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (18238 bars)
- **Last close:** 1872.00
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 14 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 61
- **Target hits / Stop hits / Partials:** 14 / 61 / 24
- **Avg / median % per leg:** 0.10% / -0.27%
- **Sum % (uncompounded):** 9.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 15 | 26.8% | 5 | 41 | 10 | -0.04% | -2.3% |
| BUY @ 2nd Alert (retest1) | 56 | 15 | 26.8% | 5 | 41 | 10 | -0.04% | -2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 23 | 53.5% | 9 | 20 | 14 | 0.28% | 11.8% |
| SELL @ 2nd Alert (retest1) | 43 | 23 | 53.5% | 9 | 20 | 14 | 0.28% | 11.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 99 | 38 | 38.4% | 14 | 61 | 24 | 0.10% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 1667.10 | 1675.75 | 0.00 | ORB-short ORB[1668.00,1687.40] vol=1.8x ATR=7.80 |
| Stop hit — per-position SL triggered | 2025-05-14 09:50:00 | 1674.90 | 1674.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 11:00:00 | 1707.45 | 1685.16 | 0.00 | ORB-long ORB[1677.50,1701.00] vol=3.2x ATR=7.86 |
| Stop hit — per-position SL triggered | 2025-05-15 11:05:00 | 1699.59 | 1686.77 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 11:10:00 | 1899.25 | 1870.79 | 0.00 | ORB-long ORB[1850.50,1877.55] vol=7.7x ATR=9.20 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 1890.05 | 1872.95 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:45:00 | 2137.50 | 2123.54 | 0.00 | ORB-long ORB[2103.00,2129.45] vol=2.6x ATR=10.58 |
| Stop hit — per-position SL triggered | 2025-05-30 09:50:00 | 2126.92 | 2123.78 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 2161.85 | 2131.56 | 0.00 | ORB-long ORB[2111.00,2140.00] vol=4.7x ATR=11.84 |
| Stop hit — per-position SL triggered | 2025-06-03 09:40:00 | 2150.01 | 2134.60 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:50:00 | 2215.70 | 2201.35 | 0.00 | ORB-long ORB[2177.50,2210.00] vol=3.4x ATR=10.48 |
| Stop hit — per-position SL triggered | 2025-06-10 10:00:00 | 2205.22 | 2204.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2120.00 | 2147.39 | 0.00 | ORB-short ORB[2139.40,2170.00] vol=1.9x ATR=9.98 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 2129.98 | 2144.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:40:00 | 2215.75 | 2200.60 | 0.00 | ORB-long ORB[2177.50,2202.50] vol=6.4x ATR=13.45 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 2202.30 | 2201.55 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:40:00 | 2263.80 | 2241.48 | 0.00 | ORB-long ORB[2221.40,2254.50] vol=2.9x ATR=11.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:45:00 | 2281.24 | 2252.23 | 0.00 | T1 1.5R @ 2281.24 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 2263.80 | 2253.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 2249.35 | 2238.72 | 0.00 | ORB-long ORB[2224.50,2246.90] vol=2.7x ATR=8.48 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 2240.87 | 2239.21 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:35:00 | 2260.95 | 2248.31 | 0.00 | ORB-long ORB[2229.05,2257.50] vol=4.2x ATR=9.09 |
| Stop hit — per-position SL triggered | 2025-07-01 09:40:00 | 2251.86 | 2248.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:50:00 | 2203.45 | 2180.73 | 0.00 | ORB-long ORB[2167.10,2192.50] vol=2.5x ATR=8.75 |
| Stop hit — per-position SL triggered | 2025-07-03 09:55:00 | 2194.70 | 2183.13 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:55:00 | 2299.25 | 2282.07 | 0.00 | ORB-long ORB[2266.00,2298.70] vol=2.0x ATR=10.76 |
| Stop hit — per-position SL triggered | 2025-07-07 10:00:00 | 2288.49 | 2282.57 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:40:00 | 2355.00 | 2337.85 | 0.00 | ORB-long ORB[2320.15,2347.50] vol=1.9x ATR=9.90 |
| Stop hit — per-position SL triggered | 2025-07-10 09:50:00 | 2345.10 | 2341.49 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:55:00 | 2277.95 | 2301.12 | 0.00 | ORB-short ORB[2297.50,2320.50] vol=2.4x ATR=7.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:05:00 | 2266.46 | 2294.02 | 0.00 | T1 1.5R @ 2266.46 |
| Stop hit — per-position SL triggered | 2025-07-17 10:30:00 | 2277.95 | 2288.85 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:55:00 | 2245.00 | 2261.73 | 0.00 | ORB-short ORB[2270.00,2294.00] vol=2.5x ATR=6.96 |
| Stop hit — per-position SL triggered | 2025-07-18 11:10:00 | 2251.96 | 2260.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:35:00 | 2122.05 | 2132.23 | 0.00 | ORB-short ORB[2125.50,2146.25] vol=1.5x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:10:00 | 2111.86 | 2126.85 | 0.00 | T1 1.5R @ 2111.86 |
| Target hit | 2025-07-25 15:20:00 | 2073.70 | 2099.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-08-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:30:00 | 1974.50 | 1961.39 | 0.00 | ORB-long ORB[1945.00,1969.45] vol=1.5x ATR=6.94 |
| Stop hit — per-position SL triggered | 2025-08-07 09:40:00 | 1967.56 | 1962.53 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:10:00 | 2005.75 | 1986.01 | 0.00 | ORB-long ORB[1967.75,1993.50] vol=2.4x ATR=11.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:15:00 | 2022.74 | 1993.47 | 0.00 | T1 1.5R @ 2022.74 |
| Target hit | 2025-08-13 12:10:00 | 2018.15 | 2018.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2025-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:50:00 | 2010.15 | 2024.77 | 0.00 | ORB-short ORB[2024.15,2047.75] vol=1.8x ATR=6.78 |
| Stop hit — per-position SL triggered | 2025-08-14 13:00:00 | 2016.93 | 2022.03 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:25:00 | 2058.00 | 2043.69 | 0.00 | ORB-long ORB[2027.55,2054.90] vol=1.9x ATR=7.25 |
| Stop hit — per-position SL triggered | 2025-08-18 10:30:00 | 2050.75 | 2044.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:00:00 | 2082.50 | 2067.73 | 0.00 | ORB-long ORB[2057.00,2077.05] vol=2.4x ATR=6.44 |
| Stop hit — per-position SL triggered | 2025-08-20 10:10:00 | 2076.06 | 2068.78 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:45:00 | 2068.50 | 2059.56 | 0.00 | ORB-long ORB[2047.50,2067.50] vol=1.6x ATR=6.09 |
| Stop hit — per-position SL triggered | 2025-08-21 10:10:00 | 2062.41 | 2062.38 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:15:00 | 2035.05 | 2023.10 | 0.00 | ORB-long ORB[2005.75,2029.00] vol=3.3x ATR=7.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:20:00 | 2046.36 | 2024.87 | 0.00 | T1 1.5R @ 2046.36 |
| Stop hit — per-position SL triggered | 2025-08-22 10:25:00 | 2035.05 | 2025.66 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:30:00 | 1931.00 | 1918.18 | 0.00 | ORB-long ORB[1911.25,1928.45] vol=1.5x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:10:00 | 1940.15 | 1923.52 | 0.00 | T1 1.5R @ 1940.15 |
| Stop hit — per-position SL triggered | 2025-08-29 12:25:00 | 1931.00 | 1928.16 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 1948.15 | 1937.41 | 0.00 | ORB-long ORB[1927.50,1947.00] vol=2.8x ATR=7.24 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 1940.91 | 1938.17 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:50:00 | 2011.65 | 1989.78 | 0.00 | ORB-long ORB[1970.10,1989.00] vol=3.4x ATR=9.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:55:00 | 2025.42 | 2013.55 | 0.00 | T1 1.5R @ 2025.42 |
| Target hit | 2025-09-05 12:05:00 | 2037.70 | 2044.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 2062.50 | 2050.33 | 0.00 | ORB-long ORB[2037.50,2059.40] vol=2.1x ATR=8.42 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 2054.08 | 2056.32 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 2055.40 | 2038.66 | 0.00 | ORB-long ORB[2025.55,2053.50] vol=3.2x ATR=7.89 |
| Stop hit — per-position SL triggered | 2025-09-09 09:35:00 | 2047.51 | 2039.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 2052.40 | 2041.18 | 0.00 | ORB-long ORB[2028.20,2042.20] vol=4.5x ATR=6.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:35:00 | 2062.69 | 2054.18 | 0.00 | T1 1.5R @ 2062.69 |
| Target hit | 2025-09-12 10:05:00 | 2084.95 | 2095.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 2199.00 | 2186.62 | 0.00 | ORB-long ORB[2168.95,2192.90] vol=2.5x ATR=7.57 |
| Stop hit — per-position SL triggered | 2025-09-17 09:55:00 | 2191.43 | 2190.35 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:30:00 | 2198.50 | 2188.20 | 0.00 | ORB-long ORB[2168.10,2192.45] vol=1.7x ATR=6.13 |
| Stop hit — per-position SL triggered | 2025-09-19 10:45:00 | 2192.37 | 2188.72 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:10:00 | 2081.20 | 2122.85 | 0.00 | ORB-short ORB[2114.25,2142.00] vol=2.0x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:20:00 | 2069.04 | 2118.73 | 0.00 | T1 1.5R @ 2069.04 |
| Target hit | 2025-09-29 15:20:00 | 2031.85 | 2074.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:15:00 | 2072.35 | 2060.66 | 0.00 | ORB-long ORB[2033.00,2057.50] vol=2.2x ATR=7.44 |
| Stop hit — per-position SL triggered | 2025-09-30 11:25:00 | 2064.91 | 2060.89 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:30:00 | 2156.05 | 2163.05 | 0.00 | ORB-short ORB[2156.55,2172.15] vol=1.9x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:20:00 | 2144.98 | 2158.98 | 0.00 | T1 1.5R @ 2144.98 |
| Target hit | 2025-10-08 13:35:00 | 2148.80 | 2147.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — BUY (started 2025-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:00:00 | 2142.50 | 2133.06 | 0.00 | ORB-long ORB[2122.50,2139.50] vol=2.7x ATR=7.35 |
| Stop hit — per-position SL triggered | 2025-10-09 10:50:00 | 2135.15 | 2136.73 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:45:00 | 2191.40 | 2178.78 | 0.00 | ORB-long ORB[2165.05,2186.45] vol=2.3x ATR=9.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:50:00 | 2204.96 | 2187.37 | 0.00 | T1 1.5R @ 2204.96 |
| Stop hit — per-position SL triggered | 2025-10-10 10:00:00 | 2191.40 | 2189.17 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:45:00 | 2222.00 | 2207.58 | 0.00 | ORB-long ORB[2185.35,2213.85] vol=4.3x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-10-14 09:50:00 | 2212.81 | 2207.94 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:40:00 | 2198.15 | 2183.37 | 0.00 | ORB-long ORB[2167.50,2184.00] vol=2.3x ATR=7.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:45:00 | 2209.24 | 2191.37 | 0.00 | T1 1.5R @ 2209.24 |
| Target hit | 2025-10-15 11:45:00 | 2228.60 | 2232.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 2264.65 | 2253.93 | 0.00 | ORB-long ORB[2240.05,2260.65] vol=2.9x ATR=9.44 |
| Stop hit — per-position SL triggered | 2025-10-16 09:35:00 | 2255.21 | 2254.26 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 2239.90 | 2228.15 | 0.00 | ORB-long ORB[2209.50,2229.25] vol=4.1x ATR=6.65 |
| Stop hit — per-position SL triggered | 2025-10-17 09:45:00 | 2233.25 | 2232.16 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:00:00 | 2187.30 | 2199.58 | 0.00 | ORB-short ORB[2187.55,2219.95] vol=1.5x ATR=7.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:10:00 | 2175.55 | 2195.93 | 0.00 | T1 1.5R @ 2175.55 |
| Target hit | 2025-10-20 15:20:00 | 2172.55 | 2180.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 2195.05 | 2205.36 | 0.00 | ORB-short ORB[2195.20,2217.90] vol=1.6x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:50:00 | 2187.07 | 2200.85 | 0.00 | T1 1.5R @ 2187.07 |
| Target hit | 2025-10-28 15:20:00 | 2163.35 | 2174.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 09:35:00 | 2157.50 | 2165.88 | 0.00 | ORB-short ORB[2159.00,2174.00] vol=1.5x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:40:00 | 2148.25 | 2161.81 | 0.00 | T1 1.5R @ 2148.25 |
| Target hit | 2025-10-29 14:25:00 | 2154.00 | 2149.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 2181.00 | 2166.95 | 0.00 | ORB-long ORB[2158.45,2179.95] vol=4.2x ATR=5.81 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 2175.19 | 2167.42 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:35:00 | 2196.70 | 2183.79 | 0.00 | ORB-long ORB[2164.00,2193.70] vol=2.7x ATR=9.13 |
| Stop hit — per-position SL triggered | 2025-11-03 09:40:00 | 2187.57 | 2184.36 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 2146.80 | 2171.33 | 0.00 | ORB-short ORB[2171.00,2191.00] vol=1.7x ATR=5.83 |
| Stop hit — per-position SL triggered | 2025-11-04 11:45:00 | 2152.63 | 2168.68 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 2045.60 | 2025.51 | 0.00 | ORB-long ORB[2001.10,2028.00] vol=2.9x ATR=9.02 |
| Stop hit — per-position SL triggered | 2025-11-10 09:40:00 | 2036.58 | 2027.69 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:10:00 | 2035.00 | 2024.22 | 0.00 | ORB-long ORB[2018.20,2030.50] vol=1.7x ATR=7.31 |
| Stop hit — per-position SL triggered | 2025-11-12 10:25:00 | 2027.69 | 2025.83 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:55:00 | 1994.10 | 2006.24 | 0.00 | ORB-short ORB[2009.20,2026.80] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:35:00 | 1987.23 | 2002.99 | 0.00 | T1 1.5R @ 1987.23 |
| Stop hit — per-position SL triggered | 2025-11-13 12:00:00 | 1994.10 | 2002.17 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1946.60 | 1952.58 | 0.00 | ORB-short ORB[1947.30,1962.40] vol=1.8x ATR=5.85 |
| Stop hit — per-position SL triggered | 2025-11-20 09:50:00 | 1952.45 | 1951.47 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 1866.20 | 1886.75 | 0.00 | ORB-short ORB[1876.80,1899.10] vol=4.1x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 12:15:00 | 1856.78 | 1881.89 | 0.00 | T1 1.5R @ 1856.78 |
| Target hit | 2025-11-27 15:20:00 | 1842.80 | 1866.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:45:00 | 1800.90 | 1810.04 | 0.00 | ORB-short ORB[1810.40,1829.10] vol=3.0x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:15:00 | 1792.89 | 1804.96 | 0.00 | T1 1.5R @ 1792.89 |
| Target hit | 2025-12-02 15:20:00 | 1793.80 | 1797.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1783.00 | 1793.01 | 0.00 | ORB-short ORB[1791.00,1801.70] vol=1.5x ATR=4.84 |
| Stop hit — per-position SL triggered | 2025-12-03 09:40:00 | 1787.84 | 1791.26 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:40:00 | 1741.00 | 1759.59 | 0.00 | ORB-short ORB[1759.00,1779.80] vol=3.2x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:50:00 | 1731.00 | 1753.98 | 0.00 | T1 1.5R @ 1731.00 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 1741.00 | 1752.02 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:25:00 | 1712.80 | 1694.62 | 0.00 | ORB-long ORB[1677.70,1694.00] vol=1.9x ATR=7.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 12:00:00 | 1723.72 | 1703.31 | 0.00 | T1 1.5R @ 1723.72 |
| Target hit | 2025-12-15 15:15:00 | 1735.20 | 1735.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2025-12-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:00:00 | 1654.40 | 1665.04 | 0.00 | ORB-short ORB[1666.10,1682.30] vol=1.6x ATR=5.44 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 1659.84 | 1664.61 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:10:00 | 1823.00 | 1837.78 | 0.00 | ORB-short ORB[1837.80,1853.50] vol=2.2x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 1828.87 | 1837.39 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:25:00 | 1873.20 | 1863.39 | 0.00 | ORB-long ORB[1847.70,1870.00] vol=2.8x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:35:00 | 1881.06 | 1865.99 | 0.00 | T1 1.5R @ 1881.06 |
| Stop hit — per-position SL triggered | 2026-01-02 11:50:00 | 1873.20 | 1869.53 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 1872.50 | 1877.94 | 0.00 | ORB-short ORB[1873.00,1892.90] vol=1.6x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-01-06 09:45:00 | 1878.12 | 1877.78 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1787.80 | 1801.55 | 0.00 | ORB-short ORB[1795.10,1813.90] vol=1.7x ATR=8.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:45:00 | 1774.92 | 1791.60 | 0.00 | T1 1.5R @ 1774.92 |
| Target hit | 2026-01-13 15:20:00 | 1774.30 | 1778.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 1743.90 | 1752.91 | 0.00 | ORB-short ORB[1746.30,1771.90] vol=2.1x ATR=7.84 |
| Stop hit — per-position SL triggered | 2026-01-20 09:40:00 | 1751.74 | 1752.21 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 09:35:00 | 1683.50 | 1703.83 | 0.00 | ORB-short ORB[1693.80,1719.10] vol=2.7x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:45:00 | 1666.86 | 1692.68 | 0.00 | T1 1.5R @ 1666.86 |
| Stop hit — per-position SL triggered | 2026-01-21 12:10:00 | 1683.50 | 1686.04 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:40:00 | 1721.00 | 1733.41 | 0.00 | ORB-short ORB[1727.10,1743.00] vol=2.5x ATR=6.48 |
| Stop hit — per-position SL triggered | 2026-02-06 09:45:00 | 1727.48 | 1732.51 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1771.40 | 1759.76 | 0.00 | ORB-long ORB[1746.00,1770.00] vol=5.1x ATR=7.21 |
| Stop hit — per-position SL triggered | 2026-02-11 11:40:00 | 1764.19 | 1763.18 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1744.90 | 1748.76 | 0.00 | ORB-short ORB[1746.00,1767.40] vol=1.5x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 1751.04 | 1748.75 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1748.50 | 1743.39 | 0.00 | ORB-long ORB[1733.00,1748.10] vol=2.4x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 1743.52 | 1743.88 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1742.10 | 1756.75 | 0.00 | ORB-short ORB[1752.00,1774.40] vol=1.6x ATR=5.38 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 1747.48 | 1755.67 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:55:00 | 1744.90 | 1720.25 | 0.00 | ORB-long ORB[1701.00,1725.00] vol=2.2x ATR=8.66 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 1736.24 | 1721.43 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:10:00 | 1696.00 | 1709.73 | 0.00 | ORB-short ORB[1700.60,1720.80] vol=2.0x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:35:00 | 1688.20 | 1705.85 | 0.00 | T1 1.5R @ 1688.20 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 1696.00 | 1702.88 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 1729.80 | 1715.30 | 0.00 | ORB-long ORB[1703.10,1724.30] vol=2.1x ATR=6.93 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 1722.87 | 1716.50 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 1683.00 | 1664.85 | 0.00 | ORB-long ORB[1647.50,1669.90] vol=1.9x ATR=8.58 |
| Stop hit — per-position SL triggered | 2026-04-15 10:20:00 | 1674.42 | 1667.04 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1772.00 | 1758.99 | 0.00 | ORB-long ORB[1738.70,1764.90] vol=2.0x ATR=8.36 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 1763.64 | 1765.05 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:20:00 | 1773.80 | 1797.63 | 0.00 | ORB-short ORB[1796.80,1823.10] vol=2.1x ATR=7.46 |
| Stop hit — per-position SL triggered | 2026-04-30 10:30:00 | 1781.26 | 1797.28 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 1859.00 | 1844.72 | 0.00 | ORB-long ORB[1826.00,1847.00] vol=4.0x ATR=8.31 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 1850.69 | 1846.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 09:35:00 | 1667.10 | 2025-05-14 09:50:00 | 1674.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-05-15 11:00:00 | 1707.45 | 2025-05-15 11:05:00 | 1699.59 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-05-26 11:10:00 | 1899.25 | 2025-05-26 11:15:00 | 1890.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-30 09:45:00 | 2137.50 | 2025-05-30 09:50:00 | 2126.92 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-06-03 09:35:00 | 2161.85 | 2025-06-03 09:40:00 | 2150.01 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-06-10 09:50:00 | 2215.70 | 2025-06-10 10:00:00 | 2205.22 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-06-16 09:30:00 | 2120.00 | 2025-06-16 09:35:00 | 2129.98 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-06-17 09:40:00 | 2215.75 | 2025-06-17 09:45:00 | 2202.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-06-19 09:40:00 | 2263.80 | 2025-06-19 09:45:00 | 2281.24 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-06-19 09:40:00 | 2263.80 | 2025-06-19 09:50:00 | 2263.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:30:00 | 2249.35 | 2025-06-27 09:40:00 | 2240.87 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-07-01 09:35:00 | 2260.95 | 2025-07-01 09:40:00 | 2251.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-07-03 09:50:00 | 2203.45 | 2025-07-03 09:55:00 | 2194.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-07-07 09:55:00 | 2299.25 | 2025-07-07 10:00:00 | 2288.49 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-07-10 09:40:00 | 2355.00 | 2025-07-10 09:50:00 | 2345.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-07-17 09:55:00 | 2277.95 | 2025-07-17 10:05:00 | 2266.46 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-17 09:55:00 | 2277.95 | 2025-07-17 10:30:00 | 2277.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:55:00 | 2245.00 | 2025-07-18 11:10:00 | 2251.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-25 09:35:00 | 2122.05 | 2025-07-25 10:10:00 | 2111.86 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-07-25 09:35:00 | 2122.05 | 2025-07-25 15:20:00 | 2073.70 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2025-08-07 09:30:00 | 1974.50 | 2025-08-07 09:40:00 | 1967.56 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-13 10:10:00 | 2005.75 | 2025-08-13 10:15:00 | 2022.74 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2025-08-13 10:10:00 | 2005.75 | 2025-08-13 12:10:00 | 2018.15 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2025-08-14 10:50:00 | 2010.15 | 2025-08-14 13:00:00 | 2016.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-18 10:25:00 | 2058.00 | 2025-08-18 10:30:00 | 2050.75 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-20 10:00:00 | 2082.50 | 2025-08-20 10:10:00 | 2076.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-21 09:45:00 | 2068.50 | 2025-08-21 10:10:00 | 2062.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-22 10:15:00 | 2035.05 | 2025-08-22 10:20:00 | 2046.36 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-08-22 10:15:00 | 2035.05 | 2025-08-22 10:25:00 | 2035.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-29 10:30:00 | 1931.00 | 2025-08-29 11:10:00 | 1940.15 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-29 10:30:00 | 1931.00 | 2025-08-29 12:25:00 | 1931.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 09:45:00 | 1948.15 | 2025-09-01 09:55:00 | 1940.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-05 09:50:00 | 2011.65 | 2025-09-05 09:55:00 | 2025.42 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-09-05 09:50:00 | 2011.65 | 2025-09-05 12:05:00 | 2037.70 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-09-08 09:30:00 | 2062.50 | 2025-09-08 10:15:00 | 2054.08 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-09-09 09:30:00 | 2055.40 | 2025-09-09 09:35:00 | 2047.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-12 09:30:00 | 2052.40 | 2025-09-12 09:35:00 | 2062.69 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-09-12 09:30:00 | 2052.40 | 2025-09-12 10:05:00 | 2084.95 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-09-17 09:45:00 | 2199.00 | 2025-09-17 09:55:00 | 2191.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-19 10:30:00 | 2198.50 | 2025-09-19 10:45:00 | 2192.37 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-29 11:10:00 | 2081.20 | 2025-09-29 11:20:00 | 2069.04 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-09-29 11:10:00 | 2081.20 | 2025-09-29 15:20:00 | 2031.85 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2025-09-30 11:15:00 | 2072.35 | 2025-09-30 11:25:00 | 2064.91 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-08 09:30:00 | 2156.05 | 2025-10-08 10:20:00 | 2144.98 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-08 09:30:00 | 2156.05 | 2025-10-08 13:35:00 | 2148.80 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-09 10:00:00 | 2142.50 | 2025-10-09 10:50:00 | 2135.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-10 09:45:00 | 2191.40 | 2025-10-10 09:50:00 | 2204.96 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-10-10 09:45:00 | 2191.40 | 2025-10-10 10:00:00 | 2191.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-14 09:45:00 | 2222.00 | 2025-10-14 09:50:00 | 2212.81 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-10-15 09:40:00 | 2198.15 | 2025-10-15 09:45:00 | 2209.24 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-15 09:40:00 | 2198.15 | 2025-10-15 11:45:00 | 2228.60 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2025-10-16 09:30:00 | 2264.65 | 2025-10-16 09:35:00 | 2255.21 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-10-17 09:30:00 | 2239.90 | 2025-10-17 09:45:00 | 2233.25 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-20 10:00:00 | 2187.30 | 2025-10-20 10:10:00 | 2175.55 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-10-20 10:00:00 | 2187.30 | 2025-10-20 15:20:00 | 2172.55 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2025-10-28 09:30:00 | 2195.05 | 2025-10-28 09:50:00 | 2187.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-28 09:30:00 | 2195.05 | 2025-10-28 15:20:00 | 2163.35 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2025-10-29 09:35:00 | 2157.50 | 2025-10-29 09:40:00 | 2148.25 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-29 09:35:00 | 2157.50 | 2025-10-29 14:25:00 | 2154.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-10-30 11:10:00 | 2181.00 | 2025-10-30 11:15:00 | 2175.19 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-03 09:35:00 | 2196.70 | 2025-11-03 09:40:00 | 2187.57 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-11-04 11:15:00 | 2146.80 | 2025-11-04 11:45:00 | 2152.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-10 09:35:00 | 2045.60 | 2025-11-10 09:40:00 | 2036.58 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-11-12 10:10:00 | 2035.00 | 2025-11-12 10:25:00 | 2027.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-11-13 10:55:00 | 1994.10 | 2025-11-13 11:35:00 | 1987.23 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-13 10:55:00 | 1994.10 | 2025-11-13 12:00:00 | 1994.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-20 09:30:00 | 1946.60 | 2025-11-20 09:50:00 | 1952.45 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-27 11:15:00 | 1866.20 | 2025-11-27 12:15:00 | 1856.78 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-11-27 11:15:00 | 1866.20 | 2025-11-27 15:20:00 | 1842.80 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2025-12-02 09:45:00 | 1800.90 | 2025-12-02 10:15:00 | 1792.89 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-02 09:45:00 | 1800.90 | 2025-12-02 15:20:00 | 1793.80 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1783.00 | 2025-12-03 09:40:00 | 1787.84 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-05 09:40:00 | 1741.00 | 2025-12-05 09:50:00 | 1731.00 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-12-05 09:40:00 | 1741.00 | 2025-12-05 10:00:00 | 1741.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-15 10:25:00 | 1712.80 | 2025-12-15 12:00:00 | 1723.72 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-12-15 10:25:00 | 1712.80 | 2025-12-15 15:15:00 | 1735.20 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-12-18 10:00:00 | 1654.40 | 2025-12-18 10:05:00 | 1659.84 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-30 11:10:00 | 1823.00 | 2025-12-30 11:15:00 | 1828.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-02 10:25:00 | 1873.20 | 2026-01-02 10:35:00 | 1881.06 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-02 10:25:00 | 1873.20 | 2026-01-02 11:50:00 | 1873.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 09:40:00 | 1872.50 | 2026-01-06 09:45:00 | 1878.12 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-13 09:30:00 | 1787.80 | 2026-01-13 11:45:00 | 1774.92 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-01-13 09:30:00 | 1787.80 | 2026-01-13 15:20:00 | 1774.30 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-01-20 09:35:00 | 1743.90 | 2026-01-20 09:40:00 | 1751.74 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-01-21 09:35:00 | 1683.50 | 2026-01-21 10:45:00 | 1666.86 | PARTIAL | 0.50 | 0.99% |
| SELL | retest1 | 2026-01-21 09:35:00 | 1683.50 | 2026-01-21 12:10:00 | 1683.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 09:40:00 | 1721.00 | 2026-02-06 09:45:00 | 1727.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-11 11:05:00 | 1771.40 | 2026-02-11 11:40:00 | 1764.19 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-12 10:45:00 | 1744.90 | 2026-02-12 11:00:00 | 1751.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-18 09:45:00 | 1748.50 | 2026-02-18 09:50:00 | 1743.52 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 09:30:00 | 1742.10 | 2026-02-19 09:35:00 | 1747.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 09:55:00 | 1744.90 | 2026-02-20 10:00:00 | 1736.24 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-02-23 10:10:00 | 1696.00 | 2026-02-23 10:35:00 | 1688.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-23 10:10:00 | 1696.00 | 2026-02-23 11:35:00 | 1696.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:55:00 | 1729.80 | 2026-02-25 10:00:00 | 1722.87 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-15 10:05:00 | 1683.00 | 2026-04-15 10:20:00 | 1674.42 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1772.00 | 2026-04-21 10:25:00 | 1763.64 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-30 10:20:00 | 1773.80 | 2026-04-30 10:30:00 | 1781.26 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-05 09:35:00 | 1859.00 | 2026-05-05 09:40:00 | 1850.69 | STOP_HIT | 1.00 | -0.45% |
