# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-11-06 15:25:00 (9183 bars)
- **Last close:** 2247.50
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 8 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 31
- **Target hits / Stop hits / Partials:** 8 / 31 / 19
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 11.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 17 | 50.0% | 6 | 17 | 11 | 0.24% | 8.1% |
| BUY @ 2nd Alert (retest1) | 34 | 17 | 50.0% | 6 | 17 | 11 | 0.24% | 8.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 10 | 41.7% | 2 | 14 | 8 | 0.13% | 3.0% |
| SELL @ 2nd Alert (retest1) | 24 | 10 | 41.7% | 2 | 14 | 8 | 0.13% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 58 | 27 | 46.6% | 8 | 31 | 19 | 0.19% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 1861.00 | 1850.79 | 0.00 | ORB-long ORB[1838.10,1848.20] vol=2.3x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-05-15 11:05:00 | 1856.46 | 1851.75 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 1909.00 | 1896.87 | 0.00 | ORB-long ORB[1882.53,1899.95] vol=3.2x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:10:00 | 1917.76 | 1901.79 | 0.00 | T1 1.5R @ 1917.76 |
| Stop hit — per-position SL triggered | 2024-05-16 10:20:00 | 1909.00 | 1903.23 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:45:00 | 1906.23 | 1917.70 | 0.00 | ORB-short ORB[1915.00,1937.43] vol=1.6x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:20:00 | 1897.73 | 1913.21 | 0.00 | T1 1.5R @ 1897.73 |
| Stop hit — per-position SL triggered | 2024-05-17 10:35:00 | 1906.23 | 1911.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 1885.00 | 1892.03 | 0.00 | ORB-short ORB[1892.63,1907.85] vol=2.7x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-05-23 11:50:00 | 1888.93 | 1888.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1905.25 | 1917.48 | 0.00 | ORB-short ORB[1910.08,1936.00] vol=1.8x ATR=7.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:50:00 | 1893.77 | 1912.36 | 0.00 | T1 1.5R @ 1893.77 |
| Target hit | 2024-05-30 15:20:00 | 1878.98 | 1887.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:10:00 | 1905.20 | 1895.30 | 0.00 | ORB-long ORB[1882.00,1901.38] vol=3.5x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 13:05:00 | 1912.78 | 1901.29 | 0.00 | T1 1.5R @ 1912.78 |
| Target hit | 2024-05-31 15:20:00 | 1946.10 | 1929.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:00:00 | 1901.83 | 1920.14 | 0.00 | ORB-short ORB[1916.00,1944.20] vol=1.9x ATR=11.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:35:00 | 1885.25 | 1912.12 | 0.00 | T1 1.5R @ 1885.25 |
| Target hit | 2024-06-10 12:40:00 | 1899.90 | 1899.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 1964.75 | 1945.41 | 0.00 | ORB-long ORB[1908.43,1934.43] vol=1.6x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:20:00 | 1974.12 | 1949.33 | 0.00 | T1 1.5R @ 1974.12 |
| Target hit | 2024-06-12 15:20:00 | 2000.55 | 1978.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 11:10:00 | 1948.08 | 1959.70 | 0.00 | ORB-short ORB[1948.55,1974.18] vol=4.9x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-06-21 12:50:00 | 1953.41 | 1955.78 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:45:00 | 1963.03 | 1934.42 | 0.00 | ORB-long ORB[1916.65,1942.50] vol=2.3x ATR=9.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 10:55:00 | 1977.77 | 1940.03 | 0.00 | T1 1.5R @ 1977.77 |
| Stop hit — per-position SL triggered | 2024-06-24 11:40:00 | 1963.03 | 1945.85 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:00:00 | 2031.53 | 2015.36 | 0.00 | ORB-long ORB[1997.30,2021.58] vol=2.1x ATR=6.90 |
| Stop hit — per-position SL triggered | 2024-06-27 10:25:00 | 2024.63 | 2017.11 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:00:00 | 2043.50 | 2021.67 | 0.00 | ORB-long ORB[1991.33,2012.08] vol=2.7x ATR=5.51 |
| Stop hit — per-position SL triggered | 2024-07-01 11:30:00 | 2037.99 | 2024.81 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 2099.10 | 2112.51 | 0.00 | ORB-short ORB[2107.00,2126.10] vol=1.5x ATR=6.90 |
| Stop hit — per-position SL triggered | 2024-07-08 09:55:00 | 2106.00 | 2111.01 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 2071.35 | 2075.89 | 0.00 | ORB-short ORB[2075.13,2100.98] vol=5.9x ATR=9.11 |
| Stop hit — per-position SL triggered | 2024-07-15 11:35:00 | 2080.46 | 2076.23 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:50:00 | 2099.00 | 2092.21 | 0.00 | ORB-long ORB[2052.60,2079.38] vol=2.5x ATR=6.62 |
| Stop hit — per-position SL triggered | 2024-07-29 11:10:00 | 2092.38 | 2093.68 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 2073.95 | 2070.11 | 0.00 | ORB-long ORB[2050.28,2069.68] vol=5.4x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-07-31 09:50:00 | 2067.37 | 2069.59 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:40:00 | 2103.85 | 2089.98 | 0.00 | ORB-long ORB[2060.15,2082.50] vol=1.6x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-08-01 10:45:00 | 2097.60 | 2090.54 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 2041.20 | 2026.17 | 0.00 | ORB-long ORB[2010.50,2034.00] vol=2.2x ATR=6.61 |
| Stop hit — per-position SL triggered | 2024-08-07 13:10:00 | 2034.59 | 2037.77 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 2057.70 | 2052.89 | 0.00 | ORB-long ORB[2027.70,2049.85] vol=1.9x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:15:00 | 2065.01 | 2054.38 | 0.00 | T1 1.5R @ 2065.01 |
| Target hit | 2024-08-08 12:45:00 | 2070.03 | 2071.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2024-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:55:00 | 2086.18 | 2071.03 | 0.00 | ORB-long ORB[2056.15,2084.00] vol=2.4x ATR=6.04 |
| Stop hit — per-position SL triggered | 2024-08-09 11:05:00 | 2080.14 | 2072.29 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:05:00 | 2101.75 | 2111.34 | 0.00 | ORB-short ORB[2105.00,2127.60] vol=1.9x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:45:00 | 2091.62 | 2107.69 | 0.00 | T1 1.5R @ 2091.62 |
| Stop hit — per-position SL triggered | 2024-08-13 13:10:00 | 2101.75 | 2106.70 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:40:00 | 2121.00 | 2111.31 | 0.00 | ORB-long ORB[2087.85,2109.95] vol=3.1x ATR=7.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:50:00 | 2132.85 | 2115.40 | 0.00 | T1 1.5R @ 2132.85 |
| Target hit | 2024-08-20 15:20:00 | 2149.95 | 2138.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-08-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:40:00 | 2205.75 | 2195.28 | 0.00 | ORB-long ORB[2182.50,2198.90] vol=2.5x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:55:00 | 2213.56 | 2198.89 | 0.00 | T1 1.5R @ 2213.56 |
| Target hit | 2024-08-22 15:20:00 | 2208.18 | 2212.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 2238.03 | 2221.60 | 0.00 | ORB-long ORB[2193.63,2212.38] vol=2.6x ATR=8.57 |
| Stop hit — per-position SL triggered | 2024-08-27 10:55:00 | 2229.46 | 2222.00 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 2243.75 | 2250.42 | 0.00 | ORB-short ORB[2250.50,2269.85] vol=3.0x ATR=6.51 |
| Stop hit — per-position SL triggered | 2024-09-06 09:40:00 | 2250.26 | 2250.27 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 10:15:00 | 2191.85 | 2196.89 | 0.00 | ORB-short ORB[2192.80,2209.03] vol=2.1x ATR=5.79 |
| Stop hit — per-position SL triggered | 2024-09-09 12:40:00 | 2197.64 | 2194.89 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:15:00 | 2221.82 | 2216.45 | 0.00 | ORB-long ORB[2188.03,2209.50] vol=3.5x ATR=5.40 |
| Stop hit — per-position SL triggered | 2024-09-11 11:50:00 | 2216.42 | 2217.94 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:55:00 | 2219.00 | 2208.50 | 0.00 | ORB-long ORB[2193.32,2207.95] vol=1.7x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:05:00 | 2224.75 | 2211.15 | 0.00 | T1 1.5R @ 2224.75 |
| Stop hit — per-position SL triggered | 2024-09-13 11:10:00 | 2219.00 | 2211.28 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 2206.20 | 2227.50 | 0.00 | ORB-short ORB[2217.50,2243.00] vol=1.7x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:55:00 | 2195.01 | 2225.75 | 0.00 | T1 1.5R @ 2195.01 |
| Stop hit — per-position SL triggered | 2024-09-19 10:00:00 | 2206.20 | 2225.07 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 2195.00 | 2189.20 | 0.00 | ORB-long ORB[2165.00,2194.38] vol=1.6x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 12:55:00 | 2205.95 | 2192.53 | 0.00 | T1 1.5R @ 2205.95 |
| Target hit | 2024-09-23 15:20:00 | 2208.00 | 2201.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:45:00 | 2237.48 | 2222.55 | 0.00 | ORB-long ORB[2195.00,2211.00] vol=5.3x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:05:00 | 2247.39 | 2227.05 | 0.00 | T1 1.5R @ 2247.39 |
| Stop hit — per-position SL triggered | 2024-09-24 11:10:00 | 2237.48 | 2227.54 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:05:00 | 2218.38 | 2219.51 | 0.00 | ORB-short ORB[2218.45,2245.20] vol=2.0x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:15:00 | 2212.02 | 2218.69 | 0.00 | T1 1.5R @ 2212.02 |
| Stop hit — per-position SL triggered | 2024-09-26 11:30:00 | 2218.38 | 2218.48 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:30:00 | 2229.15 | 2241.91 | 0.00 | ORB-short ORB[2235.80,2262.50] vol=1.9x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:40:00 | 2219.89 | 2237.19 | 0.00 | T1 1.5R @ 2219.89 |
| Stop hit — per-position SL triggered | 2024-09-27 10:20:00 | 2229.15 | 2228.88 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:40:00 | 2180.00 | 2172.59 | 0.00 | ORB-long ORB[2150.88,2179.00] vol=1.6x ATR=5.77 |
| Stop hit — per-position SL triggered | 2024-10-01 10:45:00 | 2174.23 | 2172.66 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:50:00 | 2125.10 | 2110.13 | 0.00 | ORB-long ORB[2093.82,2114.20] vol=1.9x ATR=8.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:55:00 | 2137.27 | 2113.42 | 0.00 | T1 1.5R @ 2137.27 |
| Stop hit — per-position SL triggered | 2024-10-04 10:10:00 | 2125.10 | 2120.11 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 2082.45 | 2099.85 | 0.00 | ORB-short ORB[2090.00,2112.90] vol=2.5x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-10-07 10:40:00 | 2091.07 | 2099.07 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 2176.07 | 2183.06 | 0.00 | ORB-short ORB[2185.07,2209.78] vol=9.7x ATR=7.99 |
| Stop hit — per-position SL triggered | 2024-10-11 11:25:00 | 2184.06 | 2181.84 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:15:00 | 2266.05 | 2254.52 | 0.00 | ORB-long ORB[2238.00,2254.50] vol=1.8x ATR=8.35 |
| Stop hit — per-position SL triggered | 2024-10-15 10:30:00 | 2257.70 | 2258.08 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:35:00 | 2240.07 | 2248.11 | 0.00 | ORB-short ORB[2249.50,2270.43] vol=4.0x ATR=7.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:40:00 | 2228.38 | 2246.03 | 0.00 | T1 1.5R @ 2228.38 |
| Stop hit — per-position SL triggered | 2024-10-24 09:55:00 | 2240.07 | 2244.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:00:00 | 1861.00 | 2024-05-15 11:05:00 | 1856.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-16 09:40:00 | 1909.00 | 2024-05-16 10:10:00 | 1917.76 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-05-16 09:40:00 | 1909.00 | 2024-05-16 10:20:00 | 1909.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-17 09:45:00 | 1906.23 | 2024-05-17 10:20:00 | 1897.73 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-17 09:45:00 | 1906.23 | 2024-05-17 10:35:00 | 1906.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:35:00 | 1885.00 | 2024-05-23 11:50:00 | 1888.93 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1905.25 | 2024-05-30 09:50:00 | 1893.77 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1905.25 | 2024-05-30 15:20:00 | 1878.98 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2024-05-31 11:10:00 | 1905.20 | 2024-05-31 13:05:00 | 1912.78 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-05-31 11:10:00 | 1905.20 | 2024-05-31 15:20:00 | 1946.10 | TARGET_HIT | 0.50 | 2.15% |
| SELL | retest1 | 2024-06-10 10:00:00 | 1901.83 | 2024-06-10 10:35:00 | 1885.25 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2024-06-10 10:00:00 | 1901.83 | 2024-06-10 12:40:00 | 1899.90 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-06-12 11:00:00 | 1964.75 | 2024-06-12 11:20:00 | 1974.12 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-12 11:00:00 | 1964.75 | 2024-06-12 15:20:00 | 2000.55 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2024-06-21 11:10:00 | 1948.08 | 2024-06-21 12:50:00 | 1953.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-24 10:45:00 | 1963.03 | 2024-06-24 10:55:00 | 1977.77 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-06-24 10:45:00 | 1963.03 | 2024-06-24 11:40:00 | 1963.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:00:00 | 2031.53 | 2024-06-27 10:25:00 | 2024.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-01 11:00:00 | 2043.50 | 2024-07-01 11:30:00 | 2037.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-08 09:40:00 | 2099.10 | 2024-07-08 09:55:00 | 2106.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-15 10:50:00 | 2071.35 | 2024-07-15 11:35:00 | 2080.46 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-29 10:50:00 | 2099.00 | 2024-07-29 11:10:00 | 2092.38 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-31 09:40:00 | 2073.95 | 2024-07-31 09:50:00 | 2067.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-01 10:40:00 | 2103.85 | 2024-08-01 10:45:00 | 2097.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-07 10:45:00 | 2041.20 | 2024-08-07 13:10:00 | 2034.59 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-08 10:05:00 | 2057.70 | 2024-08-08 10:15:00 | 2065.01 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-08 10:05:00 | 2057.70 | 2024-08-08 12:45:00 | 2070.03 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-09 10:55:00 | 2086.18 | 2024-08-09 11:05:00 | 2080.14 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-13 11:05:00 | 2101.75 | 2024-08-13 12:45:00 | 2091.62 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-13 11:05:00 | 2101.75 | 2024-08-13 13:10:00 | 2101.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 09:40:00 | 2121.00 | 2024-08-20 09:50:00 | 2132.85 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-20 09:40:00 | 2121.00 | 2024-08-20 15:20:00 | 2149.95 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2024-08-22 10:40:00 | 2205.75 | 2024-08-22 10:55:00 | 2213.56 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-22 10:40:00 | 2205.75 | 2024-08-22 15:20:00 | 2208.18 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-08-27 10:50:00 | 2238.03 | 2024-08-27 10:55:00 | 2229.46 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-06 09:30:00 | 2243.75 | 2024-09-06 09:40:00 | 2250.26 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-09 10:15:00 | 2191.85 | 2024-09-09 12:40:00 | 2197.64 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-11 11:15:00 | 2221.82 | 2024-09-11 11:50:00 | 2216.42 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-13 10:55:00 | 2219.00 | 2024-09-13 11:05:00 | 2224.75 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-09-13 10:55:00 | 2219.00 | 2024-09-13 11:10:00 | 2219.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:50:00 | 2206.20 | 2024-09-19 09:55:00 | 2195.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-19 09:50:00 | 2206.20 | 2024-09-19 10:00:00 | 2206.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 11:00:00 | 2195.00 | 2024-09-23 12:55:00 | 2205.95 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-23 11:00:00 | 2195.00 | 2024-09-23 15:20:00 | 2208.00 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2024-09-24 10:45:00 | 2237.48 | 2024-09-24 11:05:00 | 2247.39 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-24 10:45:00 | 2237.48 | 2024-09-24 11:10:00 | 2237.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-26 11:05:00 | 2218.38 | 2024-09-26 11:15:00 | 2212.02 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-26 11:05:00 | 2218.38 | 2024-09-26 11:30:00 | 2218.38 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 09:30:00 | 2229.15 | 2024-09-27 09:40:00 | 2219.89 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-27 09:30:00 | 2229.15 | 2024-09-27 10:20:00 | 2229.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 10:40:00 | 2180.00 | 2024-10-01 10:45:00 | 2174.23 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-04 09:50:00 | 2125.10 | 2024-10-04 09:55:00 | 2137.27 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-04 09:50:00 | 2125.10 | 2024-10-04 10:10:00 | 2125.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:35:00 | 2082.45 | 2024-10-07 10:40:00 | 2091.07 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-11 09:55:00 | 2176.07 | 2024-10-11 11:25:00 | 2184.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-15 10:15:00 | 2266.05 | 2024-10-15 10:30:00 | 2257.70 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-24 09:35:00 | 2240.07 | 2024-10-24 09:40:00 | 2228.38 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-24 09:35:00 | 2240.07 | 2024-10-24 09:55:00 | 2240.07 | STOP_HIT | 0.50 | 0.00% |
