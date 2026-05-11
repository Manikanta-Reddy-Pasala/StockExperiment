# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-07-04 15:25:00 (21279 bars)
- **Last close:** 2370.50
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 21 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 64
- **Target hits / Stop hits / Partials:** 21 / 64 / 40
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 25.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 26 | 44.1% | 9 | 33 | 17 | 0.19% | 11.0% |
| BUY @ 2nd Alert (retest1) | 59 | 26 | 44.1% | 9 | 33 | 17 | 0.19% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 35 | 53.0% | 12 | 31 | 23 | 0.22% | 14.4% |
| SELL @ 2nd Alert (retest1) | 66 | 35 | 53.0% | 12 | 31 | 23 | 0.22% | 14.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 125 | 61 | 48.8% | 21 | 64 | 40 | 0.20% | 25.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:40:00 | 1564.65 | 1553.19 | 0.00 | ORB-long ORB[1543.00,1551.45] vol=1.9x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 10:45:00 | 1572.51 | 1555.27 | 0.00 | T1 1.5R @ 1572.51 |
| Target hit | 2023-05-12 15:05:00 | 1571.40 | 1572.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2023-05-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:40:00 | 1723.25 | 1718.03 | 0.00 | ORB-long ORB[1706.00,1720.00] vol=1.9x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:45:00 | 1728.73 | 1720.41 | 0.00 | T1 1.5R @ 1728.73 |
| Stop hit — per-position SL triggered | 2023-05-25 10:00:00 | 1723.25 | 1723.68 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:00:00 | 1773.65 | 1762.93 | 0.00 | ORB-long ORB[1748.05,1765.00] vol=1.5x ATR=5.16 |
| Stop hit — per-position SL triggered | 2023-05-26 10:05:00 | 1768.49 | 1763.50 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 1844.95 | 1834.42 | 0.00 | ORB-long ORB[1818.00,1841.00] vol=1.9x ATR=6.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 09:40:00 | 1854.07 | 1840.48 | 0.00 | T1 1.5R @ 1854.07 |
| Stop hit — per-position SL triggered | 2023-06-02 09:45:00 | 1844.95 | 1841.48 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 1944.40 | 1934.17 | 0.00 | ORB-long ORB[1918.00,1941.90] vol=2.9x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 09:45:00 | 1951.41 | 1938.40 | 0.00 | T1 1.5R @ 1951.41 |
| Stop hit — per-position SL triggered | 2023-06-08 09:50:00 | 1944.40 | 1938.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:30:00 | 2012.05 | 1995.94 | 0.00 | ORB-long ORB[1983.35,1996.45] vol=2.7x ATR=6.66 |
| Stop hit — per-position SL triggered | 2023-06-12 09:35:00 | 2005.39 | 1998.21 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:35:00 | 1995.50 | 1986.11 | 0.00 | ORB-long ORB[1973.00,1989.90] vol=2.3x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 09:50:00 | 2003.07 | 1994.41 | 0.00 | T1 1.5R @ 2003.07 |
| Target hit | 2023-06-20 11:00:00 | 1999.95 | 2000.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2023-06-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:55:00 | 1989.95 | 2002.33 | 0.00 | ORB-short ORB[1993.60,2013.20] vol=2.2x ATR=4.28 |
| Stop hit — per-position SL triggered | 2023-06-21 11:20:00 | 1994.23 | 1999.26 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-07-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:40:00 | 1966.30 | 1971.75 | 0.00 | ORB-short ORB[1968.00,1992.50] vol=2.8x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 11:05:00 | 1959.16 | 1969.94 | 0.00 | T1 1.5R @ 1959.16 |
| Target hit | 2023-07-03 12:50:00 | 1964.05 | 1963.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2023-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:55:00 | 1959.00 | 1963.27 | 0.00 | ORB-short ORB[1960.00,1973.90] vol=1.7x ATR=4.47 |
| Stop hit — per-position SL triggered | 2023-07-04 10:10:00 | 1963.47 | 1962.89 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 11:00:00 | 1849.85 | 1864.63 | 0.00 | ORB-short ORB[1858.00,1878.50] vol=2.3x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 11:50:00 | 1842.71 | 1860.82 | 0.00 | T1 1.5R @ 1842.71 |
| Target hit | 2023-07-06 15:20:00 | 1840.00 | 1849.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-07-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 09:45:00 | 1815.00 | 1824.55 | 0.00 | ORB-short ORB[1820.50,1841.95] vol=1.6x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:35:00 | 1806.97 | 1818.62 | 0.00 | T1 1.5R @ 1806.97 |
| Target hit | 2023-07-07 15:20:00 | 1786.00 | 1798.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:45:00 | 1806.95 | 1800.26 | 0.00 | ORB-long ORB[1787.55,1802.00] vol=1.6x ATR=4.56 |
| Stop hit — per-position SL triggered | 2023-07-11 10:00:00 | 1802.39 | 1800.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 10:50:00 | 1841.75 | 1844.37 | 0.00 | ORB-short ORB[1842.05,1863.95] vol=3.3x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:45:00 | 1834.79 | 1843.39 | 0.00 | T1 1.5R @ 1834.79 |
| Stop hit — per-position SL triggered | 2023-07-17 12:45:00 | 1841.75 | 1841.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 10:10:00 | 1855.00 | 1848.58 | 0.00 | ORB-long ORB[1837.30,1850.00] vol=3.7x ATR=4.16 |
| Stop hit — per-position SL triggered | 2023-07-18 10:40:00 | 1850.84 | 1850.08 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 09:35:00 | 1906.40 | 1923.52 | 0.00 | ORB-short ORB[1920.00,1948.50] vol=1.8x ATR=8.57 |
| Stop hit — per-position SL triggered | 2023-07-24 09:45:00 | 1914.97 | 1921.73 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:00:00 | 1903.85 | 1899.27 | 0.00 | ORB-long ORB[1888.25,1903.45] vol=1.6x ATR=5.48 |
| Stop hit — per-position SL triggered | 2023-07-25 10:25:00 | 1898.37 | 1900.99 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:55:00 | 1932.00 | 1920.92 | 0.00 | ORB-long ORB[1900.00,1924.95] vol=1.9x ATR=5.67 |
| Stop hit — per-position SL triggered | 2023-07-27 10:05:00 | 1926.33 | 1921.75 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 09:30:00 | 1964.40 | 1970.15 | 0.00 | ORB-short ORB[1964.45,1980.70] vol=2.5x ATR=6.01 |
| Stop hit — per-position SL triggered | 2023-08-09 09:55:00 | 1970.41 | 1968.37 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 11:15:00 | 1925.00 | 1956.29 | 0.00 | ORB-short ORB[1965.85,1987.40] vol=2.1x ATR=7.32 |
| Stop hit — per-position SL triggered | 2023-08-18 11:30:00 | 1932.32 | 1952.92 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:45:00 | 2048.00 | 2040.80 | 0.00 | ORB-long ORB[2027.05,2047.40] vol=1.7x ATR=6.16 |
| Stop hit — per-position SL triggered | 2023-08-24 09:55:00 | 2041.84 | 2041.58 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 11:15:00 | 1896.00 | 1902.55 | 0.00 | ORB-short ORB[1898.00,1912.45] vol=1.6x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 12:20:00 | 1890.98 | 1900.03 | 0.00 | T1 1.5R @ 1890.98 |
| Stop hit — per-position SL triggered | 2023-09-07 15:00:00 | 1896.00 | 1895.55 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-09-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:50:00 | 1874.50 | 1906.00 | 0.00 | ORB-short ORB[1916.00,1935.00] vol=1.7x ATR=7.63 |
| Stop hit — per-position SL triggered | 2023-09-12 09:55:00 | 1882.13 | 1903.94 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-09-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:00:00 | 1864.70 | 1880.20 | 0.00 | ORB-short ORB[1881.00,1896.65] vol=2.4x ATR=5.76 |
| Stop hit — per-position SL triggered | 2023-09-22 10:05:00 | 1870.46 | 1879.43 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-09-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 11:10:00 | 1917.00 | 1924.15 | 0.00 | ORB-short ORB[1920.50,1932.35] vol=1.8x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 11:30:00 | 1911.69 | 1922.07 | 0.00 | T1 1.5R @ 1911.69 |
| Target hit | 2023-09-28 15:20:00 | 1898.00 | 1909.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-10-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 10:00:00 | 1881.35 | 1893.32 | 0.00 | ORB-short ORB[1887.50,1903.55] vol=2.0x ATR=5.67 |
| Stop hit — per-position SL triggered | 2023-10-04 10:25:00 | 1887.02 | 1891.83 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-10-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:40:00 | 1877.00 | 1866.01 | 0.00 | ORB-long ORB[1860.00,1872.45] vol=2.5x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:50:00 | 1883.52 | 1871.88 | 0.00 | T1 1.5R @ 1883.52 |
| Stop hit — per-position SL triggered | 2023-10-06 11:20:00 | 1877.00 | 1875.99 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-10-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 11:05:00 | 1891.05 | 1884.85 | 0.00 | ORB-long ORB[1862.05,1883.00] vol=1.5x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 11:55:00 | 1897.35 | 1887.16 | 0.00 | T1 1.5R @ 1897.35 |
| Target hit | 2023-10-10 15:20:00 | 1911.00 | 1899.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 1943.10 | 1934.57 | 0.00 | ORB-long ORB[1917.00,1941.50] vol=2.2x ATR=5.72 |
| Stop hit — per-position SL triggered | 2023-10-11 09:50:00 | 1937.38 | 1936.56 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 1935.90 | 1942.77 | 0.00 | ORB-short ORB[1936.05,1953.40] vol=1.8x ATR=4.25 |
| Stop hit — per-position SL triggered | 2023-10-12 11:30:00 | 1940.15 | 1942.09 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:30:00 | 1960.00 | 1946.08 | 0.00 | ORB-long ORB[1933.40,1955.25] vol=2.0x ATR=5.79 |
| Stop hit — per-position SL triggered | 2023-10-16 12:55:00 | 1954.21 | 1954.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:40:00 | 1967.25 | 1963.20 | 0.00 | ORB-long ORB[1951.15,1967.00] vol=1.8x ATR=4.81 |
| Stop hit — per-position SL triggered | 2023-10-17 09:45:00 | 1962.44 | 1963.38 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 11:10:00 | 1827.00 | 1816.61 | 0.00 | ORB-long ORB[1795.60,1817.05] vol=3.1x ATR=5.71 |
| Stop hit — per-position SL triggered | 2023-10-27 14:50:00 | 1821.29 | 1822.30 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:25:00 | 1834.00 | 1816.82 | 0.00 | ORB-long ORB[1806.95,1830.00] vol=2.3x ATR=6.18 |
| Stop hit — per-position SL triggered | 2023-10-30 10:40:00 | 1827.82 | 1818.98 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-11-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 11:00:00 | 1844.25 | 1838.71 | 0.00 | ORB-long ORB[1827.00,1840.00] vol=1.6x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:40:00 | 1849.83 | 1843.90 | 0.00 | T1 1.5R @ 1849.83 |
| Stop hit — per-position SL triggered | 2023-11-02 13:25:00 | 1844.25 | 1845.98 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 11:00:00 | 1843.95 | 1848.97 | 0.00 | ORB-short ORB[1845.00,1855.75] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-11-03 11:20:00 | 1847.55 | 1848.34 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 10:35:00 | 1852.50 | 1857.07 | 0.00 | ORB-short ORB[1854.90,1863.55] vol=2.0x ATR=3.77 |
| Stop hit — per-position SL triggered | 2023-11-06 11:15:00 | 1856.27 | 1855.81 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-11-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:50:00 | 1883.45 | 1884.93 | 0.00 | ORB-short ORB[1886.00,1894.90] vol=2.2x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-11-09 11:20:00 | 1886.69 | 1884.89 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:05:00 | 1888.50 | 1883.16 | 0.00 | ORB-long ORB[1875.70,1887.95] vol=2.3x ATR=3.64 |
| Stop hit — per-position SL triggered | 2023-11-16 10:50:00 | 1884.86 | 1883.73 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-11-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 10:35:00 | 1912.70 | 1907.09 | 0.00 | ORB-long ORB[1886.15,1909.25] vol=1.6x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 10:45:00 | 1918.92 | 1911.33 | 0.00 | T1 1.5R @ 1918.92 |
| Target hit | 2023-11-21 15:20:00 | 1942.45 | 1926.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2023-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:30:00 | 1959.55 | 1950.84 | 0.00 | ORB-long ORB[1943.00,1949.95] vol=2.4x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 09:45:00 | 1965.86 | 1955.55 | 0.00 | T1 1.5R @ 1965.86 |
| Stop hit — per-position SL triggered | 2023-11-22 09:55:00 | 1959.55 | 1956.68 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 1938.05 | 1945.78 | 0.00 | ORB-short ORB[1941.30,1963.95] vol=2.9x ATR=5.26 |
| Stop hit — per-position SL triggered | 2023-11-24 10:35:00 | 1943.31 | 1941.33 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 11:05:00 | 1955.60 | 1947.65 | 0.00 | ORB-long ORB[1935.00,1954.75] vol=2.4x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 11:20:00 | 1962.37 | 1950.35 | 0.00 | T1 1.5R @ 1962.37 |
| Target hit | 2023-11-30 13:25:00 | 1964.20 | 1967.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2023-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 11:10:00 | 1985.30 | 1993.69 | 0.00 | ORB-short ORB[1995.00,2003.00] vol=1.7x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:15:00 | 1979.33 | 1992.45 | 0.00 | T1 1.5R @ 1979.33 |
| Target hit | 2023-12-05 15:20:00 | 1977.95 | 1975.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2023-12-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:55:00 | 1973.75 | 1964.91 | 0.00 | ORB-long ORB[1956.30,1973.65] vol=1.7x ATR=5.94 |
| Stop hit — per-position SL triggered | 2023-12-07 10:15:00 | 1967.81 | 1973.57 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:50:00 | 1957.60 | 1964.71 | 0.00 | ORB-short ORB[1971.95,1979.35] vol=1.6x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:55:00 | 1951.54 | 1962.68 | 0.00 | T1 1.5R @ 1951.54 |
| Target hit | 2023-12-08 15:20:00 | 1930.70 | 1941.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:15:00 | 1910.30 | 1916.20 | 0.00 | ORB-short ORB[1923.00,1934.95] vol=5.2x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 11:05:00 | 1901.07 | 1912.88 | 0.00 | T1 1.5R @ 1901.07 |
| Stop hit — per-position SL triggered | 2023-12-13 12:55:00 | 1910.30 | 1909.67 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 09:35:00 | 1915.30 | 1920.42 | 0.00 | ORB-short ORB[1916.20,1939.95] vol=3.2x ATR=6.43 |
| Stop hit — per-position SL triggered | 2023-12-14 10:45:00 | 1921.73 | 1917.90 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-12-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 11:00:00 | 1926.20 | 1930.80 | 0.00 | ORB-short ORB[1936.00,1949.95] vol=9.8x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 14:45:00 | 1919.83 | 1926.32 | 0.00 | T1 1.5R @ 1919.83 |
| Target hit | 2023-12-19 15:20:00 | 1911.00 | 1922.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2023-12-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:35:00 | 1910.40 | 1895.70 | 0.00 | ORB-long ORB[1877.75,1901.00] vol=6.6x ATR=6.22 |
| Stop hit — per-position SL triggered | 2023-12-22 10:45:00 | 1904.18 | 1896.57 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:45:00 | 1927.45 | 1934.72 | 0.00 | ORB-short ORB[1929.65,1946.00] vol=2.0x ATR=5.41 |
| Stop hit — per-position SL triggered | 2023-12-26 11:00:00 | 1932.86 | 1934.04 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-12-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:35:00 | 1903.30 | 1912.63 | 0.00 | ORB-short ORB[1910.00,1920.00] vol=1.6x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 13:10:00 | 1896.76 | 1906.75 | 0.00 | T1 1.5R @ 1896.76 |
| Target hit | 2023-12-28 15:20:00 | 1890.40 | 1891.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 11:05:00 | 1896.75 | 1900.42 | 0.00 | ORB-short ORB[1899.20,1913.95] vol=5.2x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-01-01 11:35:00 | 1900.48 | 1900.18 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:20:00 | 1885.75 | 1896.44 | 0.00 | ORB-short ORB[1897.25,1907.05] vol=1.5x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:30:00 | 1879.48 | 1893.20 | 0.00 | T1 1.5R @ 1879.48 |
| Stop hit — per-position SL triggered | 2024-01-02 10:55:00 | 1885.75 | 1887.45 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-01-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 10:20:00 | 1866.10 | 1877.73 | 0.00 | ORB-short ORB[1875.65,1898.00] vol=2.4x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 10:40:00 | 1857.27 | 1873.77 | 0.00 | T1 1.5R @ 1857.27 |
| Stop hit — per-position SL triggered | 2024-01-03 11:45:00 | 1866.10 | 1869.78 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 10:45:00 | 1856.55 | 1863.65 | 0.00 | ORB-short ORB[1861.05,1872.40] vol=1.6x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 11:10:00 | 1851.55 | 1861.40 | 0.00 | T1 1.5R @ 1851.55 |
| Target hit | 2024-01-04 14:15:00 | 1851.10 | 1851.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:15:00 | 1868.45 | 1863.54 | 0.00 | ORB-long ORB[1860.10,1867.35] vol=1.7x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-01-05 10:40:00 | 1865.04 | 1864.46 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:35:00 | 1829.55 | 1837.29 | 0.00 | ORB-short ORB[1834.00,1844.65] vol=1.6x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:00:00 | 1823.00 | 1833.18 | 0.00 | T1 1.5R @ 1823.00 |
| Stop hit — per-position SL triggered | 2024-01-08 10:05:00 | 1829.55 | 1830.74 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:45:00 | 1813.45 | 1820.52 | 0.00 | ORB-short ORB[1816.00,1829.90] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-01-09 10:05:00 | 1818.78 | 1819.68 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:15:00 | 1811.75 | 1816.91 | 0.00 | ORB-short ORB[1812.40,1829.00] vol=1.5x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:30:00 | 1804.96 | 1814.77 | 0.00 | T1 1.5R @ 1804.96 |
| Stop hit — per-position SL triggered | 2024-01-11 10:40:00 | 1811.75 | 1814.67 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-01-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 10:40:00 | 1797.05 | 1792.84 | 0.00 | ORB-long ORB[1778.00,1791.95] vol=1.8x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-01-17 12:00:00 | 1792.34 | 1794.64 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:30:00 | 1766.85 | 1777.22 | 0.00 | ORB-short ORB[1771.25,1786.95] vol=1.5x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:35:00 | 1758.45 | 1770.55 | 0.00 | T1 1.5R @ 1758.45 |
| Target hit | 2024-01-18 10:20:00 | 1760.65 | 1759.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2024-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:30:00 | 1935.15 | 1918.01 | 0.00 | ORB-long ORB[1889.90,1918.00] vol=4.2x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-02-02 11:05:00 | 1927.37 | 1928.63 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:05:00 | 1876.15 | 1892.00 | 0.00 | ORB-short ORB[1894.25,1911.00] vol=5.2x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-02-08 11:15:00 | 1881.53 | 1890.81 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 10:10:00 | 1865.90 | 1852.36 | 0.00 | ORB-long ORB[1840.25,1860.00] vol=2.3x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:25:00 | 1876.77 | 1859.06 | 0.00 | T1 1.5R @ 1876.77 |
| Target hit | 2024-02-12 15:20:00 | 1933.05 | 1910.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2024-02-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 11:05:00 | 1943.05 | 1957.10 | 0.00 | ORB-short ORB[1952.00,1966.70] vol=1.9x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 11:25:00 | 1935.34 | 1948.54 | 0.00 | T1 1.5R @ 1935.34 |
| Stop hit — per-position SL triggered | 2024-02-21 12:05:00 | 1943.05 | 1942.48 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-22 11:05:00 | 1952.50 | 1940.85 | 0.00 | ORB-long ORB[1927.10,1945.25] vol=1.6x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 11:45:00 | 1961.80 | 1945.20 | 0.00 | T1 1.5R @ 1961.80 |
| Target hit | 2024-02-22 15:20:00 | 1965.85 | 1954.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2024-03-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-02 09:30:00 | 2111.95 | 2102.07 | 0.00 | ORB-long ORB[2074.00,2104.45] vol=3.8x ATR=8.59 |
| Stop hit — per-position SL triggered | 2024-03-02 09:50:00 | 2103.36 | 2107.46 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 2066.25 | 2086.01 | 0.00 | ORB-short ORB[2090.45,2111.00] vol=2.4x ATR=7.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:00:00 | 2054.34 | 2081.36 | 0.00 | T1 1.5R @ 2054.34 |
| Stop hit — per-position SL triggered | 2024-03-06 10:05:00 | 2066.25 | 2080.51 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:55:00 | 2118.65 | 2105.54 | 0.00 | ORB-long ORB[2077.30,2106.85] vol=2.8x ATR=6.92 |
| Stop hit — per-position SL triggered | 2024-03-11 10:35:00 | 2111.73 | 2108.74 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-03-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 09:55:00 | 2044.15 | 2030.59 | 0.00 | ORB-long ORB[2006.80,2035.25] vol=1.9x ATR=11.29 |
| Stop hit — per-position SL triggered | 2024-03-18 10:10:00 | 2032.86 | 2032.93 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 09:30:00 | 1940.90 | 1944.10 | 0.00 | ORB-short ORB[1944.25,1965.00] vol=5.8x ATR=12.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:00:00 | 1921.64 | 1936.37 | 0.00 | T1 1.5R @ 1921.64 |
| Target hit | 2024-03-20 15:20:00 | 1904.85 | 1919.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 10:55:00 | 1987.85 | 1989.60 | 0.00 | ORB-short ORB[1988.30,2010.10] vol=2.8x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-03-28 11:05:00 | 1991.90 | 1989.62 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:30:00 | 2011.65 | 2002.05 | 0.00 | ORB-long ORB[1990.00,2007.95] vol=2.0x ATR=4.86 |
| Stop hit — per-position SL triggered | 2024-04-10 10:40:00 | 2006.79 | 2002.68 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-04-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:50:00 | 2013.20 | 1999.40 | 0.00 | ORB-long ORB[1990.15,2008.00] vol=3.7x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 11:00:00 | 2020.50 | 2002.44 | 0.00 | T1 1.5R @ 2020.50 |
| Stop hit — per-position SL triggered | 2024-04-12 11:15:00 | 2013.20 | 2003.88 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-19 11:15:00 | 1943.20 | 1953.81 | 0.00 | ORB-short ORB[1950.10,1977.45] vol=2.0x ATR=6.24 |
| Stop hit — per-position SL triggered | 2024-04-19 11:25:00 | 1949.44 | 1953.27 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:40:00 | 1972.95 | 1968.52 | 0.00 | ORB-long ORB[1955.00,1972.00] vol=1.6x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-04-23 10:30:00 | 1966.51 | 1968.81 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:45:00 | 2020.90 | 2006.77 | 0.00 | ORB-long ORB[1986.00,2012.00] vol=3.1x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:05:00 | 2030.31 | 2014.43 | 0.00 | T1 1.5R @ 2030.31 |
| Stop hit — per-position SL triggered | 2024-04-25 10:10:00 | 2020.90 | 2015.25 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:15:00 | 2071.50 | 2057.45 | 0.00 | ORB-long ORB[2045.95,2058.20] vol=1.8x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-04-29 10:20:00 | 2066.49 | 2058.91 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 10:50:00 | 2098.00 | 2117.16 | 0.00 | ORB-short ORB[2120.00,2142.00] vol=2.1x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 13:30:00 | 2089.29 | 2111.18 | 0.00 | T1 1.5R @ 2089.29 |
| Target hit | 2024-05-02 15:20:00 | 2063.05 | 2089.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2024-05-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 11:00:00 | 2060.00 | 2072.23 | 0.00 | ORB-short ORB[2068.05,2083.90] vol=1.7x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:30:00 | 2052.22 | 2069.33 | 0.00 | T1 1.5R @ 2052.22 |
| Stop hit — per-position SL triggered | 2024-05-03 12:40:00 | 2060.00 | 2065.21 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 09:40:00 | 2084.80 | 2081.00 | 0.00 | ORB-long ORB[2060.00,2084.00] vol=1.8x ATR=5.71 |
| Stop hit — per-position SL triggered | 2024-05-06 09:45:00 | 2079.09 | 2081.41 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:30:00 | 2079.20 | 2070.43 | 0.00 | ORB-long ORB[2050.35,2072.70] vol=6.4x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 10:45:00 | 2089.61 | 2075.39 | 0.00 | T1 1.5R @ 2089.61 |
| Target hit | 2024-05-08 14:15:00 | 2094.80 | 2095.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 84 — SELL (started 2024-05-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:40:00 | 2083.65 | 2093.49 | 0.00 | ORB-short ORB[2085.00,2103.00] vol=1.5x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:50:00 | 2074.08 | 2089.88 | 0.00 | T1 1.5R @ 2074.08 |
| Stop hit — per-position SL triggered | 2024-05-09 11:00:00 | 2083.65 | 2089.20 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 11:15:00 | 2101.75 | 2083.45 | 0.00 | ORB-long ORB[2060.00,2085.35] vol=3.4x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 13:30:00 | 2112.15 | 2098.51 | 0.00 | T1 1.5R @ 2112.15 |
| Target hit | 2024-05-10 15:20:00 | 2161.00 | 2132.36 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:40:00 | 1564.65 | 2023-05-12 10:45:00 | 1572.51 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-05-12 10:40:00 | 1564.65 | 2023-05-12 15:05:00 | 1571.40 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2023-05-25 09:40:00 | 1723.25 | 2023-05-25 09:45:00 | 1728.73 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-05-25 09:40:00 | 1723.25 | 2023-05-25 10:00:00 | 1723.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-26 10:00:00 | 1773.65 | 2023-05-26 10:05:00 | 1768.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-02 09:30:00 | 1844.95 | 2023-06-02 09:40:00 | 1854.07 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-06-02 09:30:00 | 1844.95 | 2023-06-02 09:45:00 | 1844.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 09:35:00 | 1944.40 | 2023-06-08 09:45:00 | 1951.41 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-06-08 09:35:00 | 1944.40 | 2023-06-08 09:50:00 | 1944.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-12 09:30:00 | 2012.05 | 2023-06-12 09:35:00 | 2005.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-20 09:35:00 | 1995.50 | 2023-06-20 09:50:00 | 2003.07 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-20 09:35:00 | 1995.50 | 2023-06-20 11:00:00 | 1999.95 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2023-06-21 10:55:00 | 1989.95 | 2023-06-21 11:20:00 | 1994.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-03 10:40:00 | 1966.30 | 2023-07-03 11:05:00 | 1959.16 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-07-03 10:40:00 | 1966.30 | 2023-07-03 12:50:00 | 1964.05 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2023-07-04 09:55:00 | 1959.00 | 2023-07-04 10:10:00 | 1963.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-06 11:00:00 | 1849.85 | 2023-07-06 11:50:00 | 1842.71 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-07-06 11:00:00 | 1849.85 | 2023-07-06 15:20:00 | 1840.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2023-07-07 09:45:00 | 1815.00 | 2023-07-07 10:35:00 | 1806.97 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-07-07 09:45:00 | 1815.00 | 2023-07-07 15:20:00 | 1786.00 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2023-07-11 09:45:00 | 1806.95 | 2023-07-11 10:00:00 | 1802.39 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-17 10:50:00 | 1841.75 | 2023-07-17 11:45:00 | 1834.79 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-07-17 10:50:00 | 1841.75 | 2023-07-17 12:45:00 | 1841.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-18 10:10:00 | 1855.00 | 2023-07-18 10:40:00 | 1850.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-24 09:35:00 | 1906.40 | 2023-07-24 09:45:00 | 1914.97 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-07-25 10:00:00 | 1903.85 | 2023-07-25 10:25:00 | 1898.37 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-27 09:55:00 | 1932.00 | 2023-07-27 10:05:00 | 1926.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-09 09:30:00 | 1964.40 | 2023-08-09 09:55:00 | 1970.41 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-18 11:15:00 | 1925.00 | 2023-08-18 11:30:00 | 1932.32 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-08-24 09:45:00 | 2048.00 | 2023-08-24 09:55:00 | 2041.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-07 11:15:00 | 1896.00 | 2023-09-07 12:20:00 | 1890.98 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-09-07 11:15:00 | 1896.00 | 2023-09-07 15:00:00 | 1896.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:50:00 | 1874.50 | 2023-09-12 09:55:00 | 1882.13 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-09-22 10:00:00 | 1864.70 | 2023-09-22 10:05:00 | 1870.46 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-09-28 11:10:00 | 1917.00 | 2023-09-28 11:30:00 | 1911.69 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-09-28 11:10:00 | 1917.00 | 2023-09-28 15:20:00 | 1898.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2023-10-04 10:00:00 | 1881.35 | 2023-10-04 10:25:00 | 1887.02 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-10-06 10:40:00 | 1877.00 | 2023-10-06 10:50:00 | 1883.52 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-10-06 10:40:00 | 1877.00 | 2023-10-06 11:20:00 | 1877.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-10 11:05:00 | 1891.05 | 2023-10-10 11:55:00 | 1897.35 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-10-10 11:05:00 | 1891.05 | 2023-10-10 15:20:00 | 1911.00 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2023-10-11 09:35:00 | 1943.10 | 2023-10-11 09:50:00 | 1937.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-12 11:10:00 | 1935.90 | 2023-10-12 11:30:00 | 1940.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-10-16 10:30:00 | 1960.00 | 2023-10-16 12:55:00 | 1954.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-10-17 09:40:00 | 1967.25 | 2023-10-17 09:45:00 | 1962.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-27 11:10:00 | 1827.00 | 2023-10-27 14:50:00 | 1821.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-30 10:25:00 | 1834.00 | 2023-10-30 10:40:00 | 1827.82 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-11-02 11:00:00 | 1844.25 | 2023-11-02 11:40:00 | 1849.83 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-02 11:00:00 | 1844.25 | 2023-11-02 13:25:00 | 1844.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-03 11:00:00 | 1843.95 | 2023-11-03 11:20:00 | 1847.55 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-06 10:35:00 | 1852.50 | 2023-11-06 11:15:00 | 1856.27 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-09 10:50:00 | 1883.45 | 2023-11-09 11:20:00 | 1886.69 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-11-16 10:05:00 | 1888.50 | 2023-11-16 10:50:00 | 1884.86 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-21 10:35:00 | 1912.70 | 2023-11-21 10:45:00 | 1918.92 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-21 10:35:00 | 1912.70 | 2023-11-21 15:20:00 | 1942.45 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2023-11-22 09:30:00 | 1959.55 | 2023-11-22 09:45:00 | 1965.86 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-11-22 09:30:00 | 1959.55 | 2023-11-22 09:55:00 | 1959.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 09:30:00 | 1938.05 | 2023-11-24 10:35:00 | 1943.31 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-30 11:05:00 | 1955.60 | 2023-11-30 11:20:00 | 1962.37 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-30 11:05:00 | 1955.60 | 2023-11-30 13:25:00 | 1964.20 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-05 11:10:00 | 1985.30 | 2023-12-05 11:15:00 | 1979.33 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-05 11:10:00 | 1985.30 | 2023-12-05 15:20:00 | 1977.95 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-07 09:55:00 | 1973.75 | 2023-12-07 10:15:00 | 1967.81 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-08 10:50:00 | 1957.60 | 2023-12-08 10:55:00 | 1951.54 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-12-08 10:50:00 | 1957.60 | 2023-12-08 15:20:00 | 1930.70 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2023-12-13 10:15:00 | 1910.30 | 2023-12-13 11:05:00 | 1901.07 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-12-13 10:15:00 | 1910.30 | 2023-12-13 12:55:00 | 1910.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-14 09:35:00 | 1915.30 | 2023-12-14 10:45:00 | 1921.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-19 11:00:00 | 1926.20 | 2023-12-19 14:45:00 | 1919.83 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-12-19 11:00:00 | 1926.20 | 2023-12-19 15:20:00 | 1911.00 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2023-12-22 10:35:00 | 1910.40 | 2023-12-22 10:45:00 | 1904.18 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-26 10:45:00 | 1927.45 | 2023-12-26 11:00:00 | 1932.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-28 10:35:00 | 1903.30 | 2023-12-28 13:10:00 | 1896.76 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-12-28 10:35:00 | 1903.30 | 2023-12-28 15:20:00 | 1890.40 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-01-01 11:05:00 | 1896.75 | 2024-01-01 11:35:00 | 1900.48 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-02 10:20:00 | 1885.75 | 2024-01-02 10:30:00 | 1879.48 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-01-02 10:20:00 | 1885.75 | 2024-01-02 10:55:00 | 1885.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-03 10:20:00 | 1866.10 | 2024-01-03 10:40:00 | 1857.27 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-01-03 10:20:00 | 1866.10 | 2024-01-03 11:45:00 | 1866.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-04 10:45:00 | 1856.55 | 2024-01-04 11:10:00 | 1851.55 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-01-04 10:45:00 | 1856.55 | 2024-01-04 14:15:00 | 1851.10 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-01-05 10:15:00 | 1868.45 | 2024-01-05 10:40:00 | 1865.04 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-08 09:35:00 | 1829.55 | 2024-01-08 10:00:00 | 1823.00 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-01-08 09:35:00 | 1829.55 | 2024-01-08 10:05:00 | 1829.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-09 09:45:00 | 1813.45 | 2024-01-09 10:05:00 | 1818.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-11 10:15:00 | 1811.75 | 2024-01-11 10:30:00 | 1804.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-01-11 10:15:00 | 1811.75 | 2024-01-11 10:40:00 | 1811.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-17 10:40:00 | 1797.05 | 2024-01-17 12:00:00 | 1792.34 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-18 09:30:00 | 1766.85 | 2024-01-18 09:35:00 | 1758.45 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-01-18 09:30:00 | 1766.85 | 2024-01-18 10:20:00 | 1760.65 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-02-02 09:30:00 | 1935.15 | 2024-02-02 11:05:00 | 1927.37 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-02-08 11:05:00 | 1876.15 | 2024-02-08 11:15:00 | 1881.53 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-12 10:10:00 | 1865.90 | 2024-02-12 10:25:00 | 1876.77 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-02-12 10:10:00 | 1865.90 | 2024-02-12 15:20:00 | 1933.05 | TARGET_HIT | 0.50 | 3.60% |
| SELL | retest1 | 2024-02-21 11:05:00 | 1943.05 | 2024-02-21 11:25:00 | 1935.34 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-02-21 11:05:00 | 1943.05 | 2024-02-21 12:05:00 | 1943.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-22 11:05:00 | 1952.50 | 2024-02-22 11:45:00 | 1961.80 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-02-22 11:05:00 | 1952.50 | 2024-02-22 15:20:00 | 1965.85 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-03-02 09:30:00 | 2111.95 | 2024-03-02 09:50:00 | 2103.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-06 09:55:00 | 2066.25 | 2024-03-06 10:00:00 | 2054.34 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-03-06 09:55:00 | 2066.25 | 2024-03-06 10:05:00 | 2066.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-11 09:55:00 | 2118.65 | 2024-03-11 10:35:00 | 2111.73 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-18 09:55:00 | 2044.15 | 2024-03-18 10:10:00 | 2032.86 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-03-20 09:30:00 | 1940.90 | 2024-03-20 10:00:00 | 1921.64 | PARTIAL | 0.50 | 0.99% |
| SELL | retest1 | 2024-03-20 09:30:00 | 1940.90 | 2024-03-20 15:20:00 | 1904.85 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2024-03-28 10:55:00 | 1987.85 | 2024-03-28 11:05:00 | 1991.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-04-10 10:30:00 | 2011.65 | 2024-04-10 10:40:00 | 2006.79 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-12 10:50:00 | 2013.20 | 2024-04-12 11:00:00 | 2020.50 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-04-12 10:50:00 | 2013.20 | 2024-04-12 11:15:00 | 2013.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-19 11:15:00 | 1943.20 | 2024-04-19 11:25:00 | 1949.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-23 09:40:00 | 1972.95 | 2024-04-23 10:30:00 | 1966.51 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-25 09:45:00 | 2020.90 | 2024-04-25 10:05:00 | 2030.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-04-25 09:45:00 | 2020.90 | 2024-04-25 10:10:00 | 2020.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-29 10:15:00 | 2071.50 | 2024-04-29 10:20:00 | 2066.49 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-02 10:50:00 | 2098.00 | 2024-05-02 13:30:00 | 2089.29 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-02 10:50:00 | 2098.00 | 2024-05-02 15:20:00 | 2063.05 | TARGET_HIT | 0.50 | 1.67% |
| SELL | retest1 | 2024-05-03 11:00:00 | 2060.00 | 2024-05-03 11:30:00 | 2052.22 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-05-03 11:00:00 | 2060.00 | 2024-05-03 12:40:00 | 2060.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-06 09:40:00 | 2084.80 | 2024-05-06 09:45:00 | 2079.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-08 10:30:00 | 2079.20 | 2024-05-08 10:45:00 | 2089.61 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-05-08 10:30:00 | 2079.20 | 2024-05-08 14:15:00 | 2094.80 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-05-09 10:40:00 | 2083.65 | 2024-05-09 10:50:00 | 2074.08 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-09 10:40:00 | 2083.65 | 2024-05-09 11:00:00 | 2083.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-10 11:15:00 | 2101.75 | 2024-05-10 13:30:00 | 2112.15 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-05-10 11:15:00 | 2101.75 | 2024-05-10 15:20:00 | 2161.00 | TARGET_HIT | 0.50 | 2.82% |
