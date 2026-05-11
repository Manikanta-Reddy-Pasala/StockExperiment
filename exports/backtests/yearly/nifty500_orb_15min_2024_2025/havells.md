# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1253.00
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
| PARTIAL | 28 |
| TARGET_HIT | 12 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 103 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 63
- **Target hits / Stop hits / Partials:** 12 / 63 / 28
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 9.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 22 | 38.6% | 7 | 35 | 15 | 0.07% | 4.2% |
| BUY @ 2nd Alert (retest1) | 57 | 22 | 38.6% | 7 | 35 | 15 | 0.07% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 18 | 39.1% | 5 | 28 | 13 | 0.11% | 4.9% |
| SELL @ 2nd Alert (retest1) | 46 | 18 | 39.1% | 5 | 28 | 13 | 0.11% | 4.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 103 | 40 | 38.8% | 12 | 63 | 28 | 0.09% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:40:00 | 1888.40 | 1870.09 | 0.00 | ORB-long ORB[1843.75,1870.00] vol=2.0x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 09:45:00 | 1898.97 | 1882.06 | 0.00 | T1 1.5R @ 1898.97 |
| Target hit | 2024-05-24 11:45:00 | 1903.90 | 1904.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-05-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:45:00 | 1912.50 | 1890.68 | 0.00 | ORB-long ORB[1876.70,1899.00] vol=1.6x ATR=6.23 |
| Stop hit — per-position SL triggered | 2024-05-27 11:30:00 | 1906.27 | 1899.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:25:00 | 1881.50 | 1865.78 | 0.00 | ORB-long ORB[1852.05,1865.85] vol=2.1x ATR=5.85 |
| Stop hit — per-position SL triggered | 2024-06-10 10:45:00 | 1875.65 | 1869.58 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 1844.85 | 1853.80 | 0.00 | ORB-short ORB[1848.40,1870.85] vol=1.7x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:05:00 | 1836.01 | 1850.13 | 0.00 | T1 1.5R @ 1836.01 |
| Stop hit — per-position SL triggered | 2024-06-11 10:10:00 | 1844.85 | 1849.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:55:00 | 1869.00 | 1839.03 | 0.00 | ORB-long ORB[1818.05,1834.00] vol=1.8x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 11:00:00 | 1878.18 | 1842.27 | 0.00 | T1 1.5R @ 1878.18 |
| Target hit | 2024-06-21 15:20:00 | 1883.20 | 1884.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:05:00 | 1960.00 | 1943.57 | 0.00 | ORB-long ORB[1915.00,1944.25] vol=2.1x ATR=6.70 |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 1953.30 | 1945.35 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1835.90 | 1829.45 | 0.00 | ORB-long ORB[1814.70,1832.95] vol=1.6x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:35:00 | 1844.48 | 1832.42 | 0.00 | T1 1.5R @ 1844.48 |
| Stop hit — per-position SL triggered | 2024-07-03 09:55:00 | 1835.90 | 1836.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 1897.65 | 1893.61 | 0.00 | ORB-long ORB[1874.00,1891.60] vol=2.8x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:50:00 | 1903.22 | 1895.15 | 0.00 | T1 1.5R @ 1903.22 |
| Stop hit — per-position SL triggered | 2024-07-05 10:55:00 | 1897.65 | 1895.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 1919.95 | 1929.98 | 0.00 | ORB-short ORB[1926.05,1945.00] vol=2.5x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 1910.55 | 1927.95 | 0.00 | T1 1.5R @ 1910.55 |
| Stop hit — per-position SL triggered | 2024-07-10 11:00:00 | 1919.95 | 1923.56 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:20:00 | 1899.65 | 1912.56 | 0.00 | ORB-short ORB[1920.15,1929.00] vol=1.5x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-07-12 10:40:00 | 1904.42 | 1909.26 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:40:00 | 1912.75 | 1924.49 | 0.00 | ORB-short ORB[1913.00,1939.10] vol=2.6x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:25:00 | 1905.15 | 1921.07 | 0.00 | T1 1.5R @ 1905.15 |
| Target hit | 2024-07-15 15:20:00 | 1892.90 | 1906.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 1738.35 | 1750.31 | 0.00 | ORB-short ORB[1754.30,1774.95] vol=2.0x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-07-23 10:40:00 | 1743.56 | 1747.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:40:00 | 1793.40 | 1786.36 | 0.00 | ORB-long ORB[1774.70,1790.00] vol=2.2x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:15:00 | 1799.73 | 1788.42 | 0.00 | T1 1.5R @ 1799.73 |
| Stop hit — per-position SL triggered | 2024-07-25 11:20:00 | 1793.40 | 1788.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:40:00 | 1851.45 | 1843.99 | 0.00 | ORB-long ORB[1838.15,1846.80] vol=1.9x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-07-26 11:20:00 | 1846.49 | 1847.46 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:00:00 | 1857.50 | 1850.01 | 0.00 | ORB-long ORB[1840.05,1855.00] vol=1.6x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-07-29 10:10:00 | 1853.20 | 1852.13 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:50:00 | 1840.60 | 1828.59 | 0.00 | ORB-long ORB[1820.70,1833.60] vol=3.2x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-07-31 10:55:00 | 1836.61 | 1829.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 11:15:00 | 1824.10 | 1811.98 | 0.00 | ORB-long ORB[1803.80,1817.95] vol=3.8x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:30:00 | 1830.28 | 1815.87 | 0.00 | T1 1.5R @ 1830.28 |
| Target hit | 2024-08-13 13:55:00 | 1827.60 | 1828.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 1816.00 | 1818.59 | 0.00 | ORB-short ORB[1818.65,1838.80] vol=1.9x ATR=4.44 |
| Stop hit — per-position SL triggered | 2024-08-14 11:15:00 | 1820.44 | 1818.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:50:00 | 1883.00 | 1863.14 | 0.00 | ORB-long ORB[1854.30,1871.00] vol=2.1x ATR=5.77 |
| Stop hit — per-position SL triggered | 2024-08-16 14:00:00 | 1877.23 | 1869.41 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 11:15:00 | 1904.55 | 1896.28 | 0.00 | ORB-long ORB[1880.35,1889.80] vol=2.4x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-08-26 11:55:00 | 1900.23 | 1896.68 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 1937.00 | 1927.18 | 0.00 | ORB-long ORB[1912.00,1936.20] vol=1.7x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:40:00 | 1944.42 | 1931.44 | 0.00 | T1 1.5R @ 1944.42 |
| Stop hit — per-position SL triggered | 2024-08-27 10:30:00 | 1937.00 | 1940.14 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:55:00 | 1886.65 | 1888.94 | 0.00 | ORB-short ORB[1887.25,1905.00] vol=2.5x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-08-28 11:00:00 | 1890.94 | 1888.98 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 1899.25 | 1907.09 | 0.00 | ORB-short ORB[1900.55,1920.00] vol=2.4x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-09-02 11:05:00 | 1903.56 | 1906.82 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:45:00 | 1902.60 | 1891.77 | 0.00 | ORB-long ORB[1884.50,1894.70] vol=1.7x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-09-03 11:25:00 | 1898.22 | 1895.11 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:40:00 | 1888.45 | 1876.66 | 0.00 | ORB-long ORB[1859.05,1881.70] vol=2.0x ATR=5.16 |
| Stop hit — per-position SL triggered | 2024-09-09 10:55:00 | 1883.29 | 1876.96 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1944.75 | 1933.94 | 0.00 | ORB-long ORB[1924.95,1935.85] vol=2.3x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-09-11 09:45:00 | 1940.41 | 1938.24 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:15:00 | 1986.10 | 1979.60 | 0.00 | ORB-long ORB[1970.30,1981.70] vol=2.1x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 1981.37 | 1980.87 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 2020.65 | 2011.06 | 0.00 | ORB-long ORB[2000.00,2013.95] vol=1.9x ATR=6.22 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 2014.43 | 2012.95 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:20:00 | 1977.45 | 1983.35 | 0.00 | ORB-short ORB[1987.80,1998.85] vol=6.2x ATR=5.96 |
| Stop hit — per-position SL triggered | 2024-09-19 10:25:00 | 1983.41 | 1983.31 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:05:00 | 2062.50 | 2069.85 | 0.00 | ORB-short ORB[2064.85,2087.00] vol=1.9x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:30:00 | 2056.09 | 2068.81 | 0.00 | T1 1.5R @ 2056.09 |
| Stop hit — per-position SL triggered | 2024-09-24 11:50:00 | 2062.50 | 2067.98 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:15:00 | 2050.25 | 2052.02 | 0.00 | ORB-short ORB[2057.15,2072.95] vol=1.9x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:55:00 | 2042.70 | 2051.31 | 0.00 | T1 1.5R @ 2042.70 |
| Target hit | 2024-09-25 13:40:00 | 2049.95 | 2048.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 11:15:00 | 2042.95 | 2031.48 | 0.00 | ORB-long ORB[2012.85,2036.00] vol=2.2x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:30:00 | 2050.92 | 2034.49 | 0.00 | T1 1.5R @ 2050.92 |
| Stop hit — per-position SL triggered | 2024-09-30 13:20:00 | 2042.95 | 2041.71 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:40:00 | 1684.55 | 1704.63 | 0.00 | ORB-short ORB[1717.00,1730.35] vol=1.5x ATR=6.26 |
| Stop hit — per-position SL triggered | 2024-10-25 11:00:00 | 1690.81 | 1700.65 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 11:05:00 | 1677.70 | 1683.94 | 0.00 | ORB-short ORB[1688.55,1709.65] vol=2.0x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 11:25:00 | 1670.49 | 1682.92 | 0.00 | T1 1.5R @ 1670.49 |
| Stop hit — per-position SL triggered | 2024-10-28 12:10:00 | 1677.70 | 1681.63 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 1658.40 | 1664.99 | 0.00 | ORB-short ORB[1660.00,1683.05] vol=1.6x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:05:00 | 1651.28 | 1661.38 | 0.00 | T1 1.5R @ 1651.28 |
| Stop hit — per-position SL triggered | 2024-10-29 10:20:00 | 1658.40 | 1659.96 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:45:00 | 1629.00 | 1635.86 | 0.00 | ORB-short ORB[1638.15,1655.45] vol=1.5x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-11-12 11:05:00 | 1633.38 | 1635.04 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:10:00 | 1626.00 | 1613.67 | 0.00 | ORB-long ORB[1601.85,1624.40] vol=2.4x ATR=6.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 10:30:00 | 1636.14 | 1616.11 | 0.00 | T1 1.5R @ 1636.14 |
| Stop hit — per-position SL triggered | 2024-11-14 11:05:00 | 1626.00 | 1619.74 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:00:00 | 1650.10 | 1636.85 | 0.00 | ORB-long ORB[1618.40,1633.45] vol=1.6x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:50:00 | 1658.01 | 1641.85 | 0.00 | T1 1.5R @ 1658.01 |
| Stop hit — per-position SL triggered | 2024-11-19 11:55:00 | 1650.10 | 1646.55 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:55:00 | 1627.50 | 1621.25 | 0.00 | ORB-long ORB[1611.40,1626.10] vol=1.6x ATR=4.67 |
| Stop hit — per-position SL triggered | 2024-11-21 11:00:00 | 1622.83 | 1621.64 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 11:00:00 | 1720.55 | 1705.98 | 0.00 | ORB-long ORB[1699.55,1714.90] vol=1.6x ATR=5.65 |
| Stop hit — per-position SL triggered | 2024-11-26 12:20:00 | 1714.90 | 1711.47 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:15:00 | 1713.75 | 1728.17 | 0.00 | ORB-short ORB[1721.60,1737.00] vol=1.8x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-11-28 12:25:00 | 1718.48 | 1723.02 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:15:00 | 1730.10 | 1740.49 | 0.00 | ORB-short ORB[1750.65,1763.90] vol=2.0x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:50:00 | 1725.66 | 1738.08 | 0.00 | T1 1.5R @ 1725.66 |
| Stop hit — per-position SL triggered | 2024-12-04 12:05:00 | 1730.10 | 1735.13 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 1735.70 | 1721.21 | 0.00 | ORB-long ORB[1705.95,1722.95] vol=3.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:10:00 | 1741.96 | 1722.69 | 0.00 | T1 1.5R @ 1741.96 |
| Target hit | 2024-12-11 15:20:00 | 1752.10 | 1742.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-12-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:55:00 | 1745.10 | 1763.72 | 0.00 | ORB-short ORB[1763.00,1782.65] vol=2.7x ATR=6.29 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 1751.39 | 1762.87 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:00:00 | 1754.85 | 1759.66 | 0.00 | ORB-short ORB[1756.55,1769.00] vol=1.6x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:45:00 | 1749.09 | 1758.39 | 0.00 | T1 1.5R @ 1749.09 |
| Target hit | 2024-12-17 15:20:00 | 1727.40 | 1742.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1678.65 | 1665.18 | 0.00 | ORB-long ORB[1653.40,1672.30] vol=2.9x ATR=4.67 |
| Stop hit — per-position SL triggered | 2024-12-23 12:00:00 | 1673.98 | 1667.43 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 1679.30 | 1670.11 | 0.00 | ORB-long ORB[1655.70,1667.80] vol=2.0x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:30:00 | 1686.32 | 1671.93 | 0.00 | T1 1.5R @ 1686.32 |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 1679.30 | 1673.30 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 1666.40 | 1670.49 | 0.00 | ORB-short ORB[1670.80,1682.65] vol=1.5x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:15:00 | 1661.28 | 1668.87 | 0.00 | T1 1.5R @ 1661.28 |
| Stop hit — per-position SL triggered | 2024-12-26 12:40:00 | 1666.40 | 1668.02 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1657.30 | 1677.76 | 0.00 | ORB-short ORB[1687.25,1704.70] vol=1.6x ATR=5.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:55:00 | 1648.84 | 1669.64 | 0.00 | T1 1.5R @ 1648.84 |
| Stop hit — per-position SL triggered | 2025-01-06 12:05:00 | 1657.30 | 1668.99 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:45:00 | 1646.00 | 1651.51 | 0.00 | ORB-short ORB[1646.15,1659.25] vol=2.3x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 12:00:00 | 1637.51 | 1647.73 | 0.00 | T1 1.5R @ 1637.51 |
| Target hit | 2025-01-07 15:20:00 | 1633.50 | 1640.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-01-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 09:50:00 | 1515.20 | 1536.57 | 0.00 | ORB-short ORB[1535.00,1555.55] vol=1.5x ATR=7.74 |
| Stop hit — per-position SL triggered | 2025-01-14 10:35:00 | 1522.94 | 1527.45 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:30:00 | 1548.75 | 1539.23 | 0.00 | ORB-long ORB[1525.15,1544.80] vol=2.8x ATR=6.73 |
| Stop hit — per-position SL triggered | 2025-01-16 12:40:00 | 1542.02 | 1544.93 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:10:00 | 1586.20 | 1594.57 | 0.00 | ORB-short ORB[1600.00,1615.00] vol=1.7x ATR=4.05 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 1590.25 | 1594.36 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:50:00 | 1574.35 | 1557.06 | 0.00 | ORB-long ORB[1538.70,1551.90] vol=1.6x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:00:00 | 1583.28 | 1561.91 | 0.00 | T1 1.5R @ 1583.28 |
| Target hit | 2025-01-23 15:20:00 | 1594.60 | 1588.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-01-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:55:00 | 1566.80 | 1580.56 | 0.00 | ORB-short ORB[1584.85,1605.95] vol=2.3x ATR=4.76 |
| Stop hit — per-position SL triggered | 2025-01-24 11:05:00 | 1571.56 | 1578.88 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:55:00 | 1500.10 | 1503.67 | 0.00 | ORB-short ORB[1508.80,1523.20] vol=1.8x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-01-27 11:10:00 | 1505.02 | 1503.41 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:40:00 | 1486.10 | 1492.11 | 0.00 | ORB-short ORB[1498.20,1518.00] vol=4.2x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-01-28 10:45:00 | 1491.80 | 1491.35 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:30:00 | 1515.95 | 1501.50 | 0.00 | ORB-long ORB[1483.90,1501.20] vol=2.3x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-01-29 10:40:00 | 1511.43 | 1504.24 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1559.40 | 1565.77 | 0.00 | ORB-short ORB[1565.00,1574.40] vol=1.6x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-02-01 11:30:00 | 1563.33 | 1565.23 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:05:00 | 1623.70 | 1645.63 | 0.00 | ORB-short ORB[1656.85,1675.25] vol=1.9x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-02-04 11:10:00 | 1630.06 | 1644.67 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:05:00 | 1600.30 | 1606.86 | 0.00 | ORB-short ORB[1602.60,1617.00] vol=1.7x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-02-06 11:45:00 | 1604.02 | 1605.55 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 11:10:00 | 1500.80 | 1512.30 | 0.00 | ORB-short ORB[1512.60,1534.95] vol=2.7x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-02-18 11:35:00 | 1505.19 | 1511.01 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:25:00 | 1513.00 | 1511.63 | 0.00 | ORB-long ORB[1488.25,1510.70] vol=5.7x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 1507.62 | 1512.47 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:00:00 | 1432.95 | 1429.44 | 0.00 | ORB-long ORB[1406.75,1421.45] vol=1.7x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:50:00 | 1440.03 | 1430.52 | 0.00 | T1 1.5R @ 1440.03 |
| Target hit | 2025-03-05 15:20:00 | 1441.80 | 1440.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-03-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:05:00 | 1470.00 | 1460.94 | 0.00 | ORB-long ORB[1450.00,1464.40] vol=2.4x ATR=6.72 |
| Stop hit — per-position SL triggered | 2025-03-06 11:20:00 | 1463.28 | 1465.13 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:15:00 | 1458.70 | 1451.34 | 0.00 | ORB-long ORB[1441.05,1456.00] vol=1.6x ATR=3.61 |
| Stop hit — per-position SL triggered | 2025-03-07 12:00:00 | 1455.09 | 1453.21 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:55:00 | 1497.30 | 1485.57 | 0.00 | ORB-long ORB[1470.00,1492.35] vol=2.1x ATR=5.83 |
| Stop hit — per-position SL triggered | 2025-03-10 10:00:00 | 1491.47 | 1486.05 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 1475.45 | 1460.74 | 0.00 | ORB-long ORB[1449.00,1465.50] vol=1.7x ATR=4.94 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 1470.51 | 1462.85 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 11:00:00 | 1478.40 | 1483.86 | 0.00 | ORB-short ORB[1479.05,1488.95] vol=2.7x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 1481.81 | 1483.74 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:50:00 | 1489.10 | 1480.14 | 0.00 | ORB-long ORB[1470.50,1487.20] vol=2.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2025-03-26 10:55:00 | 1485.25 | 1481.13 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:35:00 | 1505.75 | 1498.03 | 0.00 | ORB-long ORB[1479.50,1501.30] vol=1.5x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-03-27 09:55:00 | 1500.96 | 1499.28 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:40:00 | 1600.20 | 1589.17 | 0.00 | ORB-long ORB[1563.20,1577.10] vol=1.7x ATR=6.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:05:00 | 1609.52 | 1592.94 | 0.00 | T1 1.5R @ 1609.52 |
| Target hit | 2025-04-16 15:20:00 | 1614.70 | 1607.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:35:00 | 1630.70 | 1623.89 | 0.00 | ORB-long ORB[1609.00,1628.10] vol=2.0x ATR=6.05 |
| Stop hit — per-position SL triggered | 2025-04-17 10:05:00 | 1624.65 | 1625.14 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:50:00 | 1625.00 | 1618.43 | 0.00 | ORB-long ORB[1610.60,1621.00] vol=2.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-04-29 11:10:00 | 1621.06 | 1619.09 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 11:00:00 | 1572.80 | 1576.80 | 0.00 | ORB-short ORB[1575.30,1584.60] vol=2.1x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 12:25:00 | 1568.17 | 1574.98 | 0.00 | T1 1.5R @ 1568.17 |
| Target hit | 2025-05-08 15:20:00 | 1534.80 | 1548.13 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-24 09:40:00 | 1888.40 | 2024-05-24 09:45:00 | 1898.97 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-24 09:40:00 | 1888.40 | 2024-05-24 11:45:00 | 1903.90 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-05-27 10:45:00 | 1912.50 | 2024-05-27 11:30:00 | 1906.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-10 10:25:00 | 1881.50 | 2024-06-10 10:45:00 | 1875.65 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-11 09:50:00 | 1844.85 | 2024-06-11 10:05:00 | 1836.01 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-06-11 09:50:00 | 1844.85 | 2024-06-11 10:10:00 | 1844.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 10:55:00 | 1869.00 | 2024-06-21 11:00:00 | 1878.18 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-06-21 10:55:00 | 1869.00 | 2024-06-21 15:20:00 | 1883.20 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-06-25 10:05:00 | 1960.00 | 2024-06-25 10:15:00 | 1953.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1835.90 | 2024-07-03 09:35:00 | 1844.48 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1835.90 | 2024-07-03 09:55:00 | 1835.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 10:45:00 | 1897.65 | 2024-07-05 10:50:00 | 1903.22 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-07-05 10:45:00 | 1897.65 | 2024-07-05 10:55:00 | 1897.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:10:00 | 1919.95 | 2024-07-10 10:20:00 | 1910.55 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-10 10:10:00 | 1919.95 | 2024-07-10 11:00:00 | 1919.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:20:00 | 1899.65 | 2024-07-12 10:40:00 | 1904.42 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-15 10:40:00 | 1912.75 | 2024-07-15 11:25:00 | 1905.15 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-15 10:40:00 | 1912.75 | 2024-07-15 15:20:00 | 1892.90 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2024-07-23 10:30:00 | 1738.35 | 2024-07-23 10:40:00 | 1743.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-25 10:40:00 | 1793.40 | 2024-07-25 11:15:00 | 1799.73 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-25 10:40:00 | 1793.40 | 2024-07-25 11:20:00 | 1793.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:40:00 | 1851.45 | 2024-07-26 11:20:00 | 1846.49 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-29 10:00:00 | 1857.50 | 2024-07-29 10:10:00 | 1853.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-31 10:50:00 | 1840.60 | 2024-07-31 10:55:00 | 1836.61 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-13 11:15:00 | 1824.10 | 2024-08-13 11:30:00 | 1830.28 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-13 11:15:00 | 1824.10 | 2024-08-13 13:55:00 | 1827.60 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-08-14 10:50:00 | 1816.00 | 2024-08-14 11:15:00 | 1820.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-16 10:50:00 | 1883.00 | 2024-08-16 14:00:00 | 1877.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-26 11:15:00 | 1904.55 | 2024-08-26 11:55:00 | 1900.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-27 09:30:00 | 1937.00 | 2024-08-27 09:40:00 | 1944.42 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-27 09:30:00 | 1937.00 | 2024-08-27 10:30:00 | 1937.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 10:55:00 | 1886.65 | 2024-08-28 11:00:00 | 1890.94 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1899.25 | 2024-09-02 11:05:00 | 1903.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-03 10:45:00 | 1902.60 | 2024-09-03 11:25:00 | 1898.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-09 10:40:00 | 1888.45 | 2024-09-09 10:55:00 | 1883.29 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-11 09:35:00 | 1944.75 | 2024-09-11 09:45:00 | 1940.41 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-12 11:15:00 | 1986.10 | 2024-09-12 12:15:00 | 1981.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-13 09:35:00 | 2020.65 | 2024-09-13 09:50:00 | 2014.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-19 10:20:00 | 1977.45 | 2024-09-19 10:25:00 | 1983.41 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-24 11:05:00 | 2062.50 | 2024-09-24 11:30:00 | 2056.09 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-09-24 11:05:00 | 2062.50 | 2024-09-24 11:50:00 | 2062.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 11:15:00 | 2050.25 | 2024-09-25 11:55:00 | 2042.70 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-25 11:15:00 | 2050.25 | 2024-09-25 13:40:00 | 2049.95 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2024-09-30 11:15:00 | 2042.95 | 2024-09-30 11:30:00 | 2050.92 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-30 11:15:00 | 2042.95 | 2024-09-30 13:20:00 | 2042.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:40:00 | 1684.55 | 2024-10-25 11:00:00 | 1690.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-28 11:05:00 | 1677.70 | 2024-10-28 11:25:00 | 1670.49 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-28 11:05:00 | 1677.70 | 2024-10-28 12:10:00 | 1677.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:35:00 | 1658.40 | 2024-10-29 10:05:00 | 1651.28 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-29 09:35:00 | 1658.40 | 2024-10-29 10:20:00 | 1658.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 10:45:00 | 1629.00 | 2024-11-12 11:05:00 | 1633.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-14 10:10:00 | 1626.00 | 2024-11-14 10:30:00 | 1636.14 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-14 10:10:00 | 1626.00 | 2024-11-14 11:05:00 | 1626.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:00:00 | 1650.10 | 2024-11-19 10:50:00 | 1658.01 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-19 10:00:00 | 1650.10 | 2024-11-19 11:55:00 | 1650.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-21 10:55:00 | 1627.50 | 2024-11-21 11:00:00 | 1622.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-26 11:00:00 | 1720.55 | 2024-11-26 12:20:00 | 1714.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-28 11:15:00 | 1713.75 | 2024-11-28 12:25:00 | 1718.48 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-04 11:15:00 | 1730.10 | 2024-12-04 11:50:00 | 1725.66 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-12-04 11:15:00 | 1730.10 | 2024-12-04 12:05:00 | 1730.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:05:00 | 1735.70 | 2024-12-11 11:10:00 | 1741.96 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-12-11 11:05:00 | 1735.70 | 2024-12-11 15:20:00 | 1752.10 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-12-13 10:55:00 | 1745.10 | 2024-12-13 11:10:00 | 1751.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1754.85 | 2024-12-17 11:45:00 | 1749.09 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1754.85 | 2024-12-17 15:20:00 | 1727.40 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2024-12-23 11:15:00 | 1678.65 | 2024-12-23 12:00:00 | 1673.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 10:05:00 | 1679.30 | 2024-12-24 10:30:00 | 1686.32 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-24 10:05:00 | 1679.30 | 2024-12-24 11:15:00 | 1679.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:55:00 | 1666.40 | 2024-12-26 12:15:00 | 1661.28 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-26 10:55:00 | 1666.40 | 2024-12-26 12:40:00 | 1666.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1657.30 | 2025-01-06 11:55:00 | 1648.84 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1657.30 | 2025-01-06 12:05:00 | 1657.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 10:45:00 | 1646.00 | 2025-01-07 12:00:00 | 1637.51 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-07 10:45:00 | 1646.00 | 2025-01-07 15:20:00 | 1633.50 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2025-01-14 09:50:00 | 1515.20 | 2025-01-14 10:35:00 | 1522.94 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-16 10:30:00 | 1548.75 | 2025-01-16 12:40:00 | 1542.02 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-21 11:10:00 | 1586.20 | 2025-01-21 11:15:00 | 1590.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-23 09:50:00 | 1574.35 | 2025-01-23 10:00:00 | 1583.28 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-01-23 09:50:00 | 1574.35 | 2025-01-23 15:20:00 | 1594.60 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2025-01-24 10:55:00 | 1566.80 | 2025-01-24 11:05:00 | 1571.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-27 10:55:00 | 1500.10 | 2025-01-27 11:10:00 | 1505.02 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-28 10:40:00 | 1486.10 | 2025-01-28 10:45:00 | 1491.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-29 10:30:00 | 1515.95 | 2025-01-29 10:40:00 | 1511.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1559.40 | 2025-02-01 11:30:00 | 1563.33 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-04 11:05:00 | 1623.70 | 2025-02-04 11:10:00 | 1630.06 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-06 11:05:00 | 1600.30 | 2025-02-06 11:45:00 | 1604.02 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-18 11:10:00 | 1500.80 | 2025-02-18 11:35:00 | 1505.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-20 10:25:00 | 1513.00 | 2025-02-20 13:15:00 | 1507.62 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-05 11:00:00 | 1432.95 | 2025-03-05 11:50:00 | 1440.03 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-05 11:00:00 | 1432.95 | 2025-03-05 15:20:00 | 1441.80 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-03-06 10:05:00 | 1470.00 | 2025-03-06 11:20:00 | 1463.28 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-07 11:15:00 | 1458.70 | 2025-03-07 12:00:00 | 1455.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-10 09:55:00 | 1497.30 | 2025-03-10 10:00:00 | 1491.47 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-11 11:00:00 | 1475.45 | 2025-03-11 11:15:00 | 1470.51 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-25 11:00:00 | 1478.40 | 2025-03-25 11:15:00 | 1481.81 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-26 10:50:00 | 1489.10 | 2025-03-26 10:55:00 | 1485.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-27 09:35:00 | 1505.75 | 2025-03-27 09:55:00 | 1500.96 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-16 10:40:00 | 1600.20 | 2025-04-16 11:05:00 | 1609.52 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-04-16 10:40:00 | 1600.20 | 2025-04-16 15:20:00 | 1614.70 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2025-04-17 09:35:00 | 1630.70 | 2025-04-17 10:05:00 | 1624.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-29 10:50:00 | 1625.00 | 2025-04-29 11:10:00 | 1621.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-08 11:00:00 | 1572.80 | 2025-05-08 12:25:00 | 1568.17 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-05-08 11:00:00 | 1572.80 | 2025-05-08 15:20:00 | 1534.80 | TARGET_HIT | 0.50 | 2.42% |
