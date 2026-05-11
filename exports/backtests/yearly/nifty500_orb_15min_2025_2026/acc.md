# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1393.00
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
| ENTRY1 | 100 |
| ENTRY2 | 0 |
| PARTIAL | 48 |
| TARGET_HIT | 23 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 77
- **Target hits / Stop hits / Partials:** 23 / 77 / 48
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 17.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 31 | 44.9% | 10 | 38 | 21 | 0.10% | 7.0% |
| BUY @ 2nd Alert (retest1) | 69 | 31 | 44.9% | 10 | 38 | 21 | 0.10% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 79 | 40 | 50.6% | 13 | 39 | 27 | 0.13% | 10.4% |
| SELL @ 2nd Alert (retest1) | 79 | 40 | 50.6% | 13 | 39 | 27 | 0.13% | 10.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 148 | 71 | 48.0% | 23 | 77 | 48 | 0.12% | 17.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:00:00 | 1942.00 | 1929.57 | 0.00 | ORB-long ORB[1924.00,1937.90] vol=1.9x ATR=5.89 |
| Stop hit — per-position SL triggered | 2025-05-19 10:25:00 | 1936.11 | 1934.20 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:35:00 | 1954.10 | 1941.83 | 0.00 | ORB-long ORB[1927.20,1944.80] vol=4.6x ATR=5.65 |
| Stop hit — per-position SL triggered | 2025-05-20 09:40:00 | 1948.45 | 1943.68 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:05:00 | 1936.60 | 1927.67 | 0.00 | ORB-long ORB[1912.20,1930.00] vol=1.5x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-05-21 11:20:00 | 1931.24 | 1930.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1964.00 | 1953.34 | 0.00 | ORB-long ORB[1938.90,1960.80] vol=1.6x ATR=5.66 |
| Stop hit — per-position SL triggered | 2025-05-23 10:20:00 | 1958.34 | 1958.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:40:00 | 1937.00 | 1944.35 | 0.00 | ORB-short ORB[1947.30,1961.90] vol=3.4x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:55:00 | 1930.53 | 1942.59 | 0.00 | T1 1.5R @ 1930.53 |
| Stop hit — per-position SL triggered | 2025-05-27 11:25:00 | 1937.00 | 1940.44 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:00:00 | 1902.10 | 1913.93 | 0.00 | ORB-short ORB[1913.00,1921.90] vol=1.6x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 11:15:00 | 1896.39 | 1908.16 | 0.00 | T1 1.5R @ 1896.39 |
| Target hit | 2025-05-30 15:20:00 | 1882.00 | 1892.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1881.20 | 1885.50 | 0.00 | ORB-short ORB[1882.50,1895.70] vol=2.3x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:50:00 | 1875.65 | 1882.97 | 0.00 | T1 1.5R @ 1875.65 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 1881.20 | 1882.53 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 1892.80 | 1886.20 | 0.00 | ORB-long ORB[1877.00,1888.90] vol=1.5x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-06-05 12:30:00 | 1889.86 | 1888.53 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 1903.00 | 1908.39 | 0.00 | ORB-short ORB[1904.30,1916.50] vol=1.8x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 1907.08 | 1908.16 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:40:00 | 1894.20 | 1901.51 | 0.00 | ORB-short ORB[1901.00,1912.00] vol=1.7x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:35:00 | 1888.91 | 1898.43 | 0.00 | T1 1.5R @ 1888.91 |
| Target hit | 2025-06-12 15:20:00 | 1863.50 | 1881.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:55:00 | 1818.80 | 1823.32 | 0.00 | ORB-short ORB[1819.10,1833.80] vol=1.9x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 1824.25 | 1822.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:50:00 | 1838.90 | 1830.61 | 0.00 | ORB-long ORB[1824.60,1837.00] vol=2.7x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:15:00 | 1844.57 | 1836.20 | 0.00 | T1 1.5R @ 1844.57 |
| Target hit | 2025-06-24 14:50:00 | 1850.00 | 1850.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1874.60 | 1867.13 | 0.00 | ORB-long ORB[1856.50,1872.30] vol=1.8x ATR=3.61 |
| Stop hit — per-position SL triggered | 2025-06-26 09:35:00 | 1870.99 | 1867.98 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 1919.60 | 1905.54 | 0.00 | ORB-long ORB[1886.40,1909.00] vol=3.1x ATR=6.42 |
| Stop hit — per-position SL triggered | 2025-06-27 09:50:00 | 1913.18 | 1907.61 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 1930.10 | 1928.16 | 0.00 | ORB-long ORB[1922.00,1929.90] vol=2.0x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-07-02 09:45:00 | 1926.60 | 1928.08 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 1966.10 | 1952.95 | 0.00 | ORB-long ORB[1942.00,1955.00] vol=3.3x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:30:00 | 1973.22 | 1960.83 | 0.00 | T1 1.5R @ 1973.22 |
| Stop hit — per-position SL triggered | 2025-07-03 10:35:00 | 1966.10 | 1961.13 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:15:00 | 1970.20 | 1960.77 | 0.00 | ORB-long ORB[1946.60,1959.80] vol=1.5x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-07-04 10:35:00 | 1966.20 | 1963.24 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:30:00 | 1944.00 | 1955.43 | 0.00 | ORB-short ORB[1951.30,1967.00] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-07-07 11:00:00 | 1948.21 | 1953.99 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:05:00 | 1958.10 | 1961.54 | 0.00 | ORB-short ORB[1958.90,1969.70] vol=3.2x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:30:00 | 1953.26 | 1960.55 | 0.00 | T1 1.5R @ 1953.26 |
| Stop hit — per-position SL triggered | 2025-07-08 11:40:00 | 1958.10 | 1958.12 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:05:00 | 1991.00 | 1985.22 | 0.00 | ORB-long ORB[1976.40,1989.30] vol=2.5x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:20:00 | 1996.96 | 1988.01 | 0.00 | T1 1.5R @ 1996.96 |
| Stop hit — per-position SL triggered | 2025-07-09 10:40:00 | 1991.00 | 1988.66 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:15:00 | 1994.00 | 1997.48 | 0.00 | ORB-short ORB[1994.10,2006.60] vol=1.9x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-07-10 13:05:00 | 1997.44 | 1996.52 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 2011.80 | 2005.64 | 0.00 | ORB-long ORB[1991.00,2009.20] vol=1.8x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:40:00 | 2017.40 | 2016.09 | 0.00 | T1 1.5R @ 2017.40 |
| Target hit | 2025-07-11 10:00:00 | 2016.70 | 2018.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:55:00 | 1996.40 | 1988.79 | 0.00 | ORB-long ORB[1980.10,1992.50] vol=1.7x ATR=3.61 |
| Stop hit — per-position SL triggered | 2025-07-15 11:10:00 | 1992.79 | 1989.99 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 1975.90 | 1977.46 | 0.00 | ORB-short ORB[1976.10,1985.00] vol=3.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 1979.01 | 1977.19 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:50:00 | 1977.00 | 1981.16 | 0.00 | ORB-short ORB[1978.00,1989.70] vol=2.8x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1972.05 | 1976.41 | 0.00 | T1 1.5R @ 1972.05 |
| Target hit | 2025-07-18 13:40:00 | 1972.20 | 1971.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2025-07-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 10:30:00 | 1969.50 | 1975.55 | 0.00 | ORB-short ORB[1971.30,1985.00] vol=1.5x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:35:00 | 1963.35 | 1974.46 | 0.00 | T1 1.5R @ 1963.35 |
| Stop hit — per-position SL triggered | 2025-07-21 11:55:00 | 1969.50 | 1971.53 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:10:00 | 1975.00 | 1978.70 | 0.00 | ORB-short ORB[1978.00,1993.60] vol=2.7x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:20:00 | 1968.55 | 1978.21 | 0.00 | T1 1.5R @ 1968.55 |
| Stop hit — per-position SL triggered | 2025-07-22 12:10:00 | 1975.00 | 1977.88 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1939.90 | 1946.41 | 0.00 | ORB-short ORB[1941.10,1967.90] vol=2.7x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:45:00 | 1931.74 | 1941.12 | 0.00 | T1 1.5R @ 1931.74 |
| Target hit | 2025-07-23 11:50:00 | 1932.50 | 1932.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1940.70 | 1949.17 | 0.00 | ORB-short ORB[1946.70,1956.10] vol=1.9x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-07-24 10:40:00 | 1944.78 | 1945.67 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:55:00 | 1836.10 | 1844.56 | 0.00 | ORB-short ORB[1838.40,1857.70] vol=2.1x ATR=5.31 |
| Stop hit — per-position SL triggered | 2025-07-28 10:55:00 | 1841.41 | 1839.48 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:30:00 | 1841.30 | 1835.93 | 0.00 | ORB-long ORB[1825.60,1840.80] vol=1.8x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-07-29 09:45:00 | 1835.68 | 1836.64 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:50:00 | 1787.60 | 1794.16 | 0.00 | ORB-short ORB[1788.60,1804.00] vol=1.6x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-08-05 11:20:00 | 1792.66 | 1790.18 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:10:00 | 1798.90 | 1813.05 | 0.00 | ORB-short ORB[1811.20,1830.40] vol=1.6x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-08-07 10:20:00 | 1803.80 | 1812.19 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:00:00 | 1843.00 | 1825.62 | 0.00 | ORB-long ORB[1811.00,1829.90] vol=2.7x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 11:40:00 | 1852.32 | 1833.81 | 0.00 | T1 1.5R @ 1852.32 |
| Stop hit — per-position SL triggered | 2025-08-18 12:10:00 | 1843.00 | 1835.03 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-08-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:25:00 | 1859.30 | 1850.96 | 0.00 | ORB-long ORB[1843.00,1859.00] vol=2.3x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:30:00 | 1866.57 | 1853.49 | 0.00 | T1 1.5R @ 1866.57 |
| Stop hit — per-position SL triggered | 2025-08-19 12:45:00 | 1859.30 | 1859.86 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 1832.20 | 1843.65 | 0.00 | ORB-short ORB[1844.40,1860.70] vol=2.1x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:30:00 | 1826.88 | 1841.85 | 0.00 | T1 1.5R @ 1826.88 |
| Target hit | 2025-08-22 15:20:00 | 1823.40 | 1831.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:50:00 | 1833.20 | 1826.05 | 0.00 | ORB-long ORB[1818.40,1828.20] vol=2.6x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:20:00 | 1840.31 | 1831.65 | 0.00 | T1 1.5R @ 1840.31 |
| Target hit | 2025-09-03 11:25:00 | 1835.20 | 1836.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — BUY (started 2025-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:05:00 | 1845.00 | 1841.61 | 0.00 | ORB-long ORB[1830.00,1843.40] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 1841.31 | 1841.95 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:45:00 | 1852.10 | 1848.44 | 0.00 | ORB-long ORB[1841.90,1851.00] vol=1.7x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1848.33 | 1849.01 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:20:00 | 1859.90 | 1858.30 | 0.00 | ORB-long ORB[1847.00,1859.10] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-09-15 10:35:00 | 1856.68 | 1858.29 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 1864.80 | 1873.02 | 0.00 | ORB-short ORB[1867.60,1881.70] vol=1.7x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:45:00 | 1858.40 | 1868.78 | 0.00 | T1 1.5R @ 1858.40 |
| Stop hit — per-position SL triggered | 2025-09-17 11:45:00 | 1864.80 | 1867.55 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:30:00 | 1866.90 | 1875.58 | 0.00 | ORB-short ORB[1868.50,1888.00] vol=2.8x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-09-19 10:25:00 | 1872.26 | 1871.71 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:50:00 | 1884.30 | 1879.25 | 0.00 | ORB-long ORB[1870.00,1884.20] vol=1.7x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 10:55:00 | 1890.40 | 1882.05 | 0.00 | T1 1.5R @ 1890.40 |
| Target hit | 2025-09-22 12:15:00 | 1896.00 | 1896.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2025-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:10:00 | 1839.70 | 1847.63 | 0.00 | ORB-short ORB[1846.20,1857.10] vol=1.5x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-09-26 10:45:00 | 1843.97 | 1845.69 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:50:00 | 1848.00 | 1843.23 | 0.00 | ORB-long ORB[1828.00,1845.30] vol=1.6x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-10-03 10:00:00 | 1843.21 | 1844.11 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:40:00 | 1845.00 | 1855.66 | 0.00 | ORB-short ORB[1850.70,1864.00] vol=2.3x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 1848.73 | 1853.55 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:30:00 | 1877.80 | 1870.70 | 0.00 | ORB-long ORB[1853.60,1872.80] vol=2.0x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-10-09 09:40:00 | 1873.25 | 1872.70 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:55:00 | 1862.30 | 1873.03 | 0.00 | ORB-short ORB[1866.20,1880.00] vol=1.8x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-10-13 11:10:00 | 1865.95 | 1872.79 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:25:00 | 1859.10 | 1867.32 | 0.00 | ORB-short ORB[1870.10,1877.70] vol=1.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-10-14 10:40:00 | 1861.88 | 1865.70 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 1855.20 | 1860.89 | 0.00 | ORB-short ORB[1858.20,1865.90] vol=1.6x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:30:00 | 1850.95 | 1858.60 | 0.00 | T1 1.5R @ 1850.95 |
| Stop hit — per-position SL triggered | 2025-10-17 10:55:00 | 1855.20 | 1857.54 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:55:00 | 1831.20 | 1836.03 | 0.00 | ORB-short ORB[1834.60,1847.70] vol=1.7x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:05:00 | 1824.77 | 1831.86 | 0.00 | T1 1.5R @ 1824.77 |
| Stop hit — per-position SL triggered | 2025-10-20 14:50:00 | 1831.20 | 1828.22 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:25:00 | 1852.00 | 1856.75 | 0.00 | ORB-short ORB[1855.00,1862.70] vol=1.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-10-28 11:45:00 | 1854.99 | 1854.18 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:35:00 | 1871.00 | 1866.30 | 0.00 | ORB-long ORB[1856.10,1869.70] vol=2.0x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:40:00 | 1877.11 | 1869.85 | 0.00 | T1 1.5R @ 1877.11 |
| Target hit | 2025-10-29 10:00:00 | 1872.80 | 1875.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2025-10-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:50:00 | 1870.30 | 1879.02 | 0.00 | ORB-short ORB[1872.50,1887.90] vol=3.1x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:55:00 | 1864.97 | 1875.07 | 0.00 | T1 1.5R @ 1864.97 |
| Target hit | 2025-10-30 15:20:00 | 1859.20 | 1865.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-11-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:50:00 | 1845.80 | 1859.06 | 0.00 | ORB-short ORB[1858.00,1879.70] vol=1.6x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 1837.03 | 1851.28 | 0.00 | T1 1.5R @ 1837.03 |
| Target hit | 2025-11-04 15:20:00 | 1830.00 | 1840.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 1820.60 | 1823.51 | 0.00 | ORB-short ORB[1821.40,1832.60] vol=1.8x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-11-07 09:45:00 | 1823.55 | 1823.21 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:50:00 | 1840.90 | 1844.55 | 0.00 | ORB-short ORB[1844.00,1852.00] vol=1.9x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-11-18 10:50:00 | 1842.98 | 1842.51 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:15:00 | 1856.20 | 1850.83 | 0.00 | ORB-long ORB[1842.50,1853.00] vol=3.4x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:20:00 | 1861.69 | 1852.63 | 0.00 | T1 1.5R @ 1861.69 |
| Stop hit — per-position SL triggered | 2025-11-20 10:55:00 | 1856.20 | 1854.67 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:35:00 | 1838.40 | 1843.68 | 0.00 | ORB-short ORB[1845.90,1855.00] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-11-21 11:35:00 | 1841.15 | 1840.25 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:00:00 | 1876.20 | 1870.60 | 0.00 | ORB-long ORB[1856.30,1871.70] vol=2.3x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 1872.42 | 1870.79 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:40:00 | 1851.50 | 1856.13 | 0.00 | ORB-short ORB[1854.00,1868.00] vol=1.6x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-12-01 10:05:00 | 1855.02 | 1854.48 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:55:00 | 1856.70 | 1849.53 | 0.00 | ORB-long ORB[1845.00,1854.30] vol=2.1x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-12-02 11:00:00 | 1853.99 | 1849.83 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1844.00 | 1850.42 | 0.00 | ORB-short ORB[1845.90,1865.60] vol=1.7x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:40:00 | 1837.10 | 1843.58 | 0.00 | T1 1.5R @ 1837.10 |
| Target hit | 2025-12-03 15:20:00 | 1839.00 | 1840.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-12-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 10:25:00 | 1830.00 | 1834.17 | 0.00 | ORB-short ORB[1832.10,1843.90] vol=2.7x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 10:35:00 | 1825.57 | 1831.98 | 0.00 | T1 1.5R @ 1825.57 |
| Target hit | 2025-12-04 15:20:00 | 1817.40 | 1823.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2025-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:30:00 | 1774.90 | 1781.91 | 0.00 | ORB-short ORB[1778.00,1796.40] vol=2.2x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-12-09 10:00:00 | 1779.15 | 1779.92 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 1764.10 | 1760.29 | 0.00 | ORB-long ORB[1753.90,1763.90] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-12-19 09:35:00 | 1760.44 | 1760.54 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1744.20 | 1748.37 | 0.00 | ORB-short ORB[1748.00,1760.10] vol=4.4x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 12:45:00 | 1739.62 | 1746.62 | 0.00 | T1 1.5R @ 1739.62 |
| Target hit | 2025-12-24 15:20:00 | 1739.20 | 1743.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:30:00 | 1738.50 | 1742.41 | 0.00 | ORB-short ORB[1741.00,1752.00] vol=3.2x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:55:00 | 1734.05 | 1740.55 | 0.00 | T1 1.5R @ 1734.05 |
| Target hit | 2025-12-26 14:30:00 | 1735.50 | 1734.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:15:00 | 1724.00 | 1730.78 | 0.00 | ORB-short ORB[1730.50,1738.40] vol=6.3x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-12-29 11:50:00 | 1726.08 | 1729.86 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1727.80 | 1722.48 | 0.00 | ORB-long ORB[1716.10,1727.50] vol=3.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-12-30 11:20:00 | 1725.70 | 1722.70 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1737.00 | 1743.54 | 0.00 | ORB-short ORB[1738.30,1755.40] vol=1.7x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:25:00 | 1732.84 | 1742.10 | 0.00 | T1 1.5R @ 1732.84 |
| Stop hit — per-position SL triggered | 2026-01-08 11:40:00 | 1737.00 | 1741.55 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:30:00 | 1707.90 | 1713.96 | 0.00 | ORB-short ORB[1712.20,1725.80] vol=2.4x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:35:00 | 1701.77 | 1709.94 | 0.00 | T1 1.5R @ 1701.77 |
| Stop hit — per-position SL triggered | 2026-01-09 09:40:00 | 1707.90 | 1709.34 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:40:00 | 1720.00 | 1712.42 | 0.00 | ORB-long ORB[1705.00,1712.80] vol=2.3x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:00:00 | 1725.10 | 1714.63 | 0.00 | T1 1.5R @ 1725.10 |
| Target hit | 2026-01-14 15:20:00 | 1728.60 | 1722.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2026-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1686.50 | 1691.62 | 0.00 | ORB-short ORB[1689.00,1703.80] vol=2.0x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:20:00 | 1677.75 | 1690.78 | 0.00 | T1 1.5R @ 1677.75 |
| Stop hit — per-position SL triggered | 2026-01-21 12:00:00 | 1686.50 | 1688.98 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:55:00 | 1673.60 | 1668.65 | 0.00 | ORB-long ORB[1660.10,1673.40] vol=2.1x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 14:00:00 | 1678.73 | 1671.35 | 0.00 | T1 1.5R @ 1678.73 |
| Stop hit — per-position SL triggered | 2026-01-30 14:25:00 | 1673.60 | 1671.50 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:55:00 | 1654.10 | 1646.42 | 0.00 | ORB-long ORB[1632.60,1654.00] vol=5.3x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:05:00 | 1661.30 | 1648.85 | 0.00 | T1 1.5R @ 1661.30 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1654.10 | 1649.75 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 1639.90 | 1626.08 | 0.00 | ORB-long ORB[1609.00,1632.20] vol=1.8x ATR=7.25 |
| Stop hit — per-position SL triggered | 2026-02-02 10:05:00 | 1632.65 | 1633.17 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:30:00 | 1677.90 | 1672.12 | 0.00 | ORB-long ORB[1656.00,1676.50] vol=1.9x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 12:10:00 | 1683.84 | 1674.68 | 0.00 | T1 1.5R @ 1683.84 |
| Target hit | 2026-02-04 15:20:00 | 1692.60 | 1681.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 1687.50 | 1681.54 | 0.00 | ORB-long ORB[1671.90,1683.70] vol=2.1x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 1692.71 | 1683.06 | 0.00 | T1 1.5R @ 1692.71 |
| Target hit | 2026-02-09 15:20:00 | 1706.50 | 1695.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1696.10 | 1704.38 | 0.00 | ORB-short ORB[1702.20,1713.50] vol=1.7x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 1690.70 | 1702.50 | 0.00 | T1 1.5R @ 1690.70 |
| Stop hit — per-position SL triggered | 2026-02-10 10:45:00 | 1696.10 | 1702.32 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 1684.00 | 1689.63 | 0.00 | ORB-short ORB[1685.90,1703.90] vol=1.9x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 1679.43 | 1688.02 | 0.00 | T1 1.5R @ 1679.43 |
| Stop hit — per-position SL triggered | 2026-02-12 12:20:00 | 1684.00 | 1686.20 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 1642.40 | 1635.74 | 0.00 | ORB-long ORB[1625.70,1636.50] vol=1.7x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 1638.73 | 1636.41 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1617.30 | 1611.04 | 0.00 | ORB-long ORB[1602.00,1614.00] vol=1.8x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 1623.04 | 1614.12 | 0.00 | T1 1.5R @ 1623.04 |
| Stop hit — per-position SL triggered | 2026-02-20 13:10:00 | 1617.30 | 1615.10 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 1629.80 | 1624.50 | 0.00 | ORB-long ORB[1619.00,1628.00] vol=4.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 1626.67 | 1624.86 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 1599.10 | 1604.15 | 0.00 | ORB-short ORB[1603.00,1617.20] vol=3.3x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:00:00 | 1593.85 | 1602.10 | 0.00 | T1 1.5R @ 1593.85 |
| Target hit | 2026-02-27 14:30:00 | 1595.60 | 1595.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 1518.40 | 1521.40 | 0.00 | ORB-short ORB[1520.10,1528.40] vol=2.8x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:35:00 | 1512.42 | 1519.75 | 0.00 | T1 1.5R @ 1512.42 |
| Target hit | 2026-03-05 14:35:00 | 1515.20 | 1515.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 87 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 1525.00 | 1521.44 | 0.00 | ORB-long ORB[1508.40,1524.30] vol=1.8x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:05:00 | 1532.25 | 1524.51 | 0.00 | T1 1.5R @ 1532.25 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 1525.00 | 1525.50 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:15:00 | 1477.00 | 1459.62 | 0.00 | ORB-long ORB[1450.70,1470.00] vol=1.5x ATR=5.44 |
| Stop hit — per-position SL triggered | 2026-03-10 11:10:00 | 1471.56 | 1463.44 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 1410.00 | 1417.02 | 0.00 | ORB-short ORB[1415.00,1430.10] vol=1.9x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-03-13 10:10:00 | 1414.65 | 1416.22 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:10:00 | 1335.80 | 1324.72 | 0.00 | ORB-long ORB[1310.10,1325.90] vol=1.8x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-04-06 11:00:00 | 1330.18 | 1328.13 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-04-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 11:00:00 | 1338.80 | 1348.47 | 0.00 | ORB-short ORB[1344.00,1357.90] vol=2.8x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-07 13:40:00 | 1342.82 | 1344.31 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:30:00 | 1397.30 | 1391.45 | 0.00 | ORB-long ORB[1376.00,1396.80] vol=2.9x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:45:00 | 1406.32 | 1394.20 | 0.00 | T1 1.5R @ 1406.32 |
| Target hit | 2026-04-08 14:25:00 | 1415.60 | 1416.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 1413.90 | 1406.83 | 0.00 | ORB-long ORB[1395.60,1407.50] vol=1.8x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:10:00 | 1419.01 | 1409.35 | 0.00 | T1 1.5R @ 1419.01 |
| Target hit | 2026-04-10 14:35:00 | 1415.60 | 1417.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 94 — SELL (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 1432.00 | 1438.25 | 0.00 | ORB-short ORB[1434.90,1447.10] vol=2.0x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 1436.02 | 1437.84 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 1444.10 | 1437.18 | 0.00 | ORB-long ORB[1425.00,1438.60] vol=2.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 1440.94 | 1437.42 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2026-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:35:00 | 1424.80 | 1430.26 | 0.00 | ORB-short ORB[1425.10,1439.70] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-04-23 12:00:00 | 1427.98 | 1428.87 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 1409.10 | 1411.81 | 0.00 | ORB-short ORB[1409.60,1429.00] vol=2.2x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:40:00 | 1403.13 | 1411.06 | 0.00 | T1 1.5R @ 1403.13 |
| Stop hit — per-position SL triggered | 2026-04-24 13:55:00 | 1409.10 | 1407.67 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 1429.60 | 1423.88 | 0.00 | ORB-long ORB[1416.40,1426.30] vol=2.1x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 1436.81 | 1427.10 | 0.00 | T1 1.5R @ 1436.81 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 1429.60 | 1427.81 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 1425.90 | 1413.10 | 0.00 | ORB-long ORB[1402.00,1420.20] vol=1.8x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:35:00 | 1433.33 | 1416.02 | 0.00 | T1 1.5R @ 1433.33 |
| Stop hit — per-position SL triggered | 2026-05-04 11:55:00 | 1425.90 | 1417.40 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1430.00 | 1423.55 | 0.00 | ORB-long ORB[1409.20,1421.70] vol=1.8x ATR=3.98 |
| Stop hit — per-position SL triggered | 2026-05-08 10:35:00 | 1426.02 | 1425.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 10:00:00 | 1942.00 | 2025-05-19 10:25:00 | 1936.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-20 09:35:00 | 1954.10 | 2025-05-20 09:40:00 | 1948.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-21 10:05:00 | 1936.60 | 2025-05-21 11:20:00 | 1931.24 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1964.00 | 2025-05-23 10:20:00 | 1958.34 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-27 10:40:00 | 1937.00 | 2025-05-27 10:55:00 | 1930.53 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-05-27 10:40:00 | 1937.00 | 2025-05-27 11:25:00 | 1937.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 10:00:00 | 1902.10 | 2025-05-30 11:15:00 | 1896.39 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-05-30 10:00:00 | 1902.10 | 2025-05-30 15:20:00 | 1882.00 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2025-06-03 09:30:00 | 1881.20 | 2025-06-03 09:50:00 | 1875.65 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-03 09:30:00 | 1881.20 | 2025-06-03 10:05:00 | 1881.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 10:45:00 | 1892.80 | 2025-06-05 12:30:00 | 1889.86 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-06-09 09:35:00 | 1903.00 | 2025-06-09 09:45:00 | 1907.08 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-12 10:40:00 | 1894.20 | 2025-06-12 11:35:00 | 1888.91 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-12 10:40:00 | 1894.20 | 2025-06-12 15:20:00 | 1863.50 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2025-06-20 09:55:00 | 1818.80 | 2025-06-20 10:15:00 | 1824.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-24 09:50:00 | 1838.90 | 2025-06-24 10:15:00 | 1844.57 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-06-24 09:50:00 | 1838.90 | 2025-06-24 14:50:00 | 1850.00 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-06-26 09:30:00 | 1874.60 | 2025-06-26 09:35:00 | 1870.99 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-27 09:45:00 | 1919.60 | 2025-06-27 09:50:00 | 1913.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-02 09:40:00 | 1930.10 | 2025-07-02 09:45:00 | 1926.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-03 10:00:00 | 1966.10 | 2025-07-03 10:30:00 | 1973.22 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-03 10:00:00 | 1966.10 | 2025-07-03 10:35:00 | 1966.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 10:15:00 | 1970.20 | 2025-07-04 10:35:00 | 1966.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-07 10:30:00 | 1944.00 | 2025-07-07 11:00:00 | 1948.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-08 10:05:00 | 1958.10 | 2025-07-08 10:30:00 | 1953.26 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-08 10:05:00 | 1958.10 | 2025-07-08 11:40:00 | 1958.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 10:05:00 | 1991.00 | 2025-07-09 10:20:00 | 1996.96 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-09 10:05:00 | 1991.00 | 2025-07-09 10:40:00 | 1991.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 11:15:00 | 1994.00 | 2025-07-10 13:05:00 | 1997.44 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-11 09:30:00 | 2011.80 | 2025-07-11 09:40:00 | 2017.40 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-11 09:30:00 | 2011.80 | 2025-07-11 10:00:00 | 2016.70 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-07-15 10:55:00 | 1996.40 | 2025-07-15 11:10:00 | 1992.79 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-16 09:30:00 | 1975.90 | 2025-07-16 09:40:00 | 1979.01 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-18 09:50:00 | 1977.00 | 2025-07-18 10:15:00 | 1972.05 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-18 09:50:00 | 1977.00 | 2025-07-18 13:40:00 | 1972.20 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-21 10:30:00 | 1969.50 | 2025-07-21 10:35:00 | 1963.35 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-21 10:30:00 | 1969.50 | 2025-07-21 11:55:00 | 1969.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 11:10:00 | 1975.00 | 2025-07-22 11:20:00 | 1968.55 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-22 11:10:00 | 1975.00 | 2025-07-22 12:10:00 | 1975.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1939.90 | 2025-07-23 09:45:00 | 1931.74 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1939.90 | 2025-07-23 11:50:00 | 1932.50 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-24 10:15:00 | 1940.70 | 2025-07-24 10:40:00 | 1944.78 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-28 09:55:00 | 1836.10 | 2025-07-28 10:55:00 | 1841.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-29 09:30:00 | 1841.30 | 2025-07-29 09:45:00 | 1835.68 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-05 09:50:00 | 1787.60 | 2025-08-05 11:20:00 | 1792.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-07 10:10:00 | 1798.90 | 2025-08-07 10:20:00 | 1803.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-18 10:00:00 | 1843.00 | 2025-08-18 11:40:00 | 1852.32 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-18 10:00:00 | 1843.00 | 2025-08-18 12:10:00 | 1843.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:25:00 | 1859.30 | 2025-08-19 10:30:00 | 1866.57 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-19 10:25:00 | 1859.30 | 2025-08-19 12:45:00 | 1859.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 11:00:00 | 1832.20 | 2025-08-22 11:30:00 | 1826.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-22 11:00:00 | 1832.20 | 2025-08-22 15:20:00 | 1823.40 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-03 09:50:00 | 1833.20 | 2025-09-03 10:20:00 | 1840.31 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-03 09:50:00 | 1833.20 | 2025-09-03 11:25:00 | 1835.20 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-09-08 10:05:00 | 1845.00 | 2025-09-08 10:30:00 | 1841.31 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-09 09:45:00 | 1852.10 | 2025-09-09 10:15:00 | 1848.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-15 10:20:00 | 1859.90 | 2025-09-15 10:35:00 | 1856.68 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-17 09:45:00 | 1864.80 | 2025-09-17 10:45:00 | 1858.40 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-17 09:45:00 | 1864.80 | 2025-09-17 11:45:00 | 1864.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-19 09:30:00 | 1866.90 | 2025-09-19 10:25:00 | 1872.26 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-22 10:50:00 | 1884.30 | 2025-09-22 10:55:00 | 1890.40 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-22 10:50:00 | 1884.30 | 2025-09-22 12:15:00 | 1896.00 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2025-09-26 10:10:00 | 1839.70 | 2025-09-26 10:45:00 | 1843.97 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-03 09:50:00 | 1848.00 | 2025-10-03 10:00:00 | 1843.21 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-08 10:40:00 | 1845.00 | 2025-10-08 11:25:00 | 1848.73 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-09 09:30:00 | 1877.80 | 2025-10-09 09:40:00 | 1873.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-13 10:55:00 | 1862.30 | 2025-10-13 11:10:00 | 1865.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-14 10:25:00 | 1859.10 | 2025-10-14 10:40:00 | 1861.88 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-10-17 10:20:00 | 1855.20 | 2025-10-17 10:30:00 | 1850.95 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-10-17 10:20:00 | 1855.20 | 2025-10-17 10:55:00 | 1855.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-20 09:55:00 | 1831.20 | 2025-10-20 11:05:00 | 1824.77 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-20 09:55:00 | 1831.20 | 2025-10-20 14:50:00 | 1831.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:25:00 | 1852.00 | 2025-10-28 11:45:00 | 1854.99 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-29 09:35:00 | 1871.00 | 2025-10-29 09:40:00 | 1877.11 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-29 09:35:00 | 1871.00 | 2025-10-29 10:00:00 | 1872.80 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-10-30 10:50:00 | 1870.30 | 2025-10-30 11:55:00 | 1864.97 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-30 10:50:00 | 1870.30 | 2025-10-30 15:20:00 | 1859.20 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-11-04 09:50:00 | 1845.80 | 2025-11-04 11:35:00 | 1837.03 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-11-04 09:50:00 | 1845.80 | 2025-11-04 15:20:00 | 1830.00 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-11-07 09:35:00 | 1820.60 | 2025-11-07 09:45:00 | 1823.55 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-11-18 09:50:00 | 1840.90 | 2025-11-18 10:50:00 | 1842.98 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-11-20 10:15:00 | 1856.20 | 2025-11-20 10:20:00 | 1861.69 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-11-20 10:15:00 | 1856.20 | 2025-11-20 10:55:00 | 1856.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:35:00 | 1838.40 | 2025-11-21 11:35:00 | 1841.15 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-11-26 11:00:00 | 1876.20 | 2025-11-26 11:15:00 | 1872.42 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-01 09:40:00 | 1851.50 | 2025-12-01 10:05:00 | 1855.02 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-02 10:55:00 | 1856.70 | 2025-12-02 11:00:00 | 1853.99 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1844.00 | 2025-12-03 10:40:00 | 1837.10 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1844.00 | 2025-12-03 15:20:00 | 1839.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-04 10:25:00 | 1830.00 | 2025-12-04 10:35:00 | 1825.57 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-04 10:25:00 | 1830.00 | 2025-12-04 15:20:00 | 1817.40 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-12-09 09:30:00 | 1774.90 | 2025-12-09 10:00:00 | 1779.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-19 09:30:00 | 1764.10 | 2025-12-19 09:35:00 | 1760.44 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-24 10:15:00 | 1744.20 | 2025-12-24 12:45:00 | 1739.62 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-24 10:15:00 | 1744.20 | 2025-12-24 15:20:00 | 1739.20 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-26 09:30:00 | 1738.50 | 2025-12-26 09:55:00 | 1734.05 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-26 09:30:00 | 1738.50 | 2025-12-26 14:30:00 | 1735.50 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-12-29 11:15:00 | 1724.00 | 2025-12-29 11:50:00 | 1726.08 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-12-30 11:15:00 | 1727.80 | 2025-12-30 11:20:00 | 1725.70 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1737.00 | 2026-01-08 11:25:00 | 1732.84 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1737.00 | 2026-01-08 11:40:00 | 1737.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-09 09:30:00 | 1707.90 | 2026-01-09 09:35:00 | 1701.77 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-09 09:30:00 | 1707.90 | 2026-01-09 09:40:00 | 1707.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-14 10:40:00 | 1720.00 | 2026-01-14 11:00:00 | 1725.10 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-01-14 10:40:00 | 1720.00 | 2026-01-14 15:20:00 | 1728.60 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-21 11:00:00 | 1686.50 | 2026-01-21 11:20:00 | 1677.75 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-21 11:00:00 | 1686.50 | 2026-01-21 12:00:00 | 1686.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 10:55:00 | 1673.60 | 2026-01-30 14:00:00 | 1678.73 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-30 10:55:00 | 1673.60 | 2026-01-30 14:25:00 | 1673.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 10:55:00 | 1654.10 | 2026-02-01 11:05:00 | 1661.30 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-01 10:55:00 | 1654.10 | 2026-02-01 11:15:00 | 1654.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-02 09:30:00 | 1639.90 | 2026-02-02 10:05:00 | 1632.65 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-04 10:30:00 | 1677.90 | 2026-02-04 12:10:00 | 1683.84 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-04 10:30:00 | 1677.90 | 2026-02-04 15:20:00 | 1692.60 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-02-09 10:45:00 | 1687.50 | 2026-02-09 11:00:00 | 1692.71 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-09 10:45:00 | 1687.50 | 2026-02-09 15:20:00 | 1706.50 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1696.10 | 2026-02-10 10:40:00 | 1690.70 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1696.10 | 2026-02-10 10:45:00 | 1696.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:50:00 | 1684.00 | 2026-02-12 11:15:00 | 1679.43 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-12 10:50:00 | 1684.00 | 2026-02-12 12:20:00 | 1684.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:50:00 | 1642.40 | 2026-02-17 10:05:00 | 1638.73 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1617.30 | 2026-02-20 12:15:00 | 1623.04 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1617.30 | 2026-02-20 13:10:00 | 1617.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:40:00 | 1629.80 | 2026-02-25 10:45:00 | 1626.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 09:40:00 | 1599.10 | 2026-02-27 10:00:00 | 1593.85 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-27 09:40:00 | 1599.10 | 2026-02-27 14:30:00 | 1595.60 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1518.40 | 2026-03-05 11:35:00 | 1512.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1518.40 | 2026-03-05 14:35:00 | 1515.20 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-03-06 09:30:00 | 1525.00 | 2026-03-06 10:05:00 | 1532.25 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-06 09:30:00 | 1525.00 | 2026-03-06 10:30:00 | 1525.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:15:00 | 1477.00 | 2026-03-10 11:10:00 | 1471.56 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-13 09:50:00 | 1410.00 | 2026-03-13 10:10:00 | 1414.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-06 10:10:00 | 1335.80 | 2026-04-06 11:00:00 | 1330.18 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1338.80 | 2026-04-07 13:40:00 | 1342.82 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-08 09:30:00 | 1397.30 | 2026-04-08 09:45:00 | 1406.32 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-08 09:30:00 | 1397.30 | 2026-04-08 14:25:00 | 1415.60 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1413.90 | 2026-04-10 12:10:00 | 1419.01 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1413.90 | 2026-04-10 14:35:00 | 1415.60 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-04-16 10:35:00 | 1432.00 | 2026-04-16 10:45:00 | 1436.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 10:10:00 | 1444.10 | 2026-04-21 10:15:00 | 1440.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-23 10:35:00 | 1424.80 | 2026-04-23 12:00:00 | 1427.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1409.10 | 2026-04-24 10:40:00 | 1403.13 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1409.10 | 2026-04-24 13:55:00 | 1409.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1429.60 | 2026-04-27 09:45:00 | 1436.81 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1429.60 | 2026-04-27 10:05:00 | 1429.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:00:00 | 1425.90 | 2026-05-04 11:35:00 | 1433.33 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-05-04 11:00:00 | 1425.90 | 2026-05-04 11:55:00 | 1425.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 10:15:00 | 1430.00 | 2026-05-08 10:35:00 | 1426.02 | STOP_HIT | 1.00 | -0.28% |
