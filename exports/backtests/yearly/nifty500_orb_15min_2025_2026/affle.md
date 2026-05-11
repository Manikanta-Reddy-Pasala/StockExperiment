# Affle 3i Ltd. (AFFLE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15613 bars)
- **Last close:** 1510.00
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
| PARTIAL | 31 |
| TARGET_HIT | 19 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 56
- **Target hits / Stop hits / Partials:** 19 / 56 / 31
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 19.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 31 | 56.4% | 12 | 24 | 19 | 0.27% | 14.8% |
| BUY @ 2nd Alert (retest1) | 55 | 31 | 56.4% | 12 | 24 | 19 | 0.27% | 14.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 19 | 37.3% | 7 | 32 | 12 | 0.09% | 4.8% |
| SELL @ 2nd Alert (retest1) | 51 | 19 | 37.3% | 7 | 32 | 12 | 0.09% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 50 | 47.2% | 19 | 56 | 31 | 0.19% | 19.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:35:00 | 1681.40 | 1673.29 | 0.00 | ORB-long ORB[1652.80,1672.20] vol=1.8x ATR=6.49 |
| Stop hit — per-position SL triggered | 2025-05-16 10:05:00 | 1674.91 | 1676.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1715.20 | 1707.77 | 0.00 | ORB-long ORB[1691.00,1706.00] vol=4.3x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:45:00 | 1724.38 | 1713.97 | 0.00 | T1 1.5R @ 1724.38 |
| Target hit | 2025-05-23 10:15:00 | 1727.90 | 1728.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2025-05-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:25:00 | 1692.10 | 1700.37 | 0.00 | ORB-short ORB[1695.40,1718.90] vol=1.7x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 13:25:00 | 1682.49 | 1695.97 | 0.00 | T1 1.5R @ 1682.49 |
| Target hit | 2025-05-27 15:20:00 | 1676.60 | 1690.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 1742.00 | 1730.23 | 0.00 | ORB-long ORB[1717.00,1737.00] vol=2.2x ATR=7.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:15:00 | 1752.93 | 1735.59 | 0.00 | T1 1.5R @ 1752.93 |
| Stop hit — per-position SL triggered | 2025-05-29 12:30:00 | 1742.00 | 1743.35 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:10:00 | 1710.90 | 1727.48 | 0.00 | ORB-short ORB[1729.00,1748.90] vol=1.7x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 13:50:00 | 1704.03 | 1721.40 | 0.00 | T1 1.5R @ 1704.03 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 1710.90 | 1718.51 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:55:00 | 1777.70 | 1765.16 | 0.00 | ORB-long ORB[1730.00,1745.00] vol=1.9x ATR=6.84 |
| Stop hit — per-position SL triggered | 2025-06-02 11:35:00 | 1770.86 | 1768.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1819.30 | 1812.50 | 0.00 | ORB-long ORB[1802.00,1818.00] vol=2.0x ATR=6.43 |
| Stop hit — per-position SL triggered | 2025-06-05 09:50:00 | 1812.87 | 1812.68 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:50:00 | 1917.20 | 1923.59 | 0.00 | ORB-short ORB[1920.00,1946.10] vol=6.0x ATR=7.41 |
| Stop hit — per-position SL triggered | 2025-06-12 11:00:00 | 1924.61 | 1922.73 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:05:00 | 1900.00 | 1908.72 | 0.00 | ORB-short ORB[1900.50,1919.90] vol=1.8x ATR=5.21 |
| Stop hit — per-position SL triggered | 2025-06-19 11:30:00 | 1905.21 | 1908.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:50:00 | 1942.10 | 1933.21 | 0.00 | ORB-long ORB[1917.10,1935.00] vol=2.7x ATR=7.61 |
| Stop hit — per-position SL triggered | 2025-06-25 10:25:00 | 1934.49 | 1934.99 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 2010.60 | 1985.42 | 0.00 | ORB-long ORB[1955.70,1980.00] vol=5.8x ATR=8.49 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 2002.11 | 1991.04 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 1986.20 | 2004.30 | 0.00 | ORB-short ORB[1992.00,2014.30] vol=2.4x ATR=6.19 |
| Stop hit — per-position SL triggered | 2025-07-01 12:20:00 | 1992.39 | 2000.81 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 1952.30 | 1966.38 | 0.00 | ORB-short ORB[1965.20,1979.30] vol=2.1x ATR=6.86 |
| Stop hit — per-position SL triggered | 2025-07-02 09:45:00 | 1959.16 | 1965.25 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:10:00 | 2008.00 | 1989.33 | 0.00 | ORB-long ORB[1975.30,1998.00] vol=2.6x ATR=8.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:20:00 | 2020.67 | 2000.47 | 0.00 | T1 1.5R @ 2020.67 |
| Target hit | 2025-07-03 15:15:00 | 2034.00 | 2039.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2025-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:10:00 | 2024.70 | 2037.01 | 0.00 | ORB-short ORB[2031.30,2059.00] vol=3.4x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:20:00 | 2016.37 | 2036.01 | 0.00 | T1 1.5R @ 2016.37 |
| Stop hit — per-position SL triggered | 2025-07-04 11:25:00 | 2024.70 | 2035.36 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 1981.00 | 1990.88 | 0.00 | ORB-short ORB[1985.00,2003.40] vol=1.7x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:10:00 | 1972.16 | 1986.10 | 0.00 | T1 1.5R @ 1972.16 |
| Target hit | 2025-07-08 13:40:00 | 1944.80 | 1940.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 1957.80 | 1966.39 | 0.00 | ORB-short ORB[1964.70,1979.50] vol=1.6x ATR=4.75 |
| Stop hit — per-position SL triggered | 2025-07-16 10:00:00 | 1962.55 | 1964.26 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 1935.70 | 1940.94 | 0.00 | ORB-short ORB[1937.50,1951.00] vol=3.6x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 09:50:00 | 1928.02 | 1939.69 | 0.00 | T1 1.5R @ 1928.02 |
| Target hit | 2025-07-18 15:20:00 | 1903.90 | 1919.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:05:00 | 1878.70 | 1886.94 | 0.00 | ORB-short ORB[1883.10,1903.00] vol=1.8x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-07-23 14:35:00 | 1883.23 | 1883.87 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:30:00 | 1829.40 | 1842.81 | 0.00 | ORB-short ORB[1843.00,1856.60] vol=1.5x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:55:00 | 1819.55 | 1838.67 | 0.00 | T1 1.5R @ 1819.55 |
| Target hit | 2025-07-25 15:20:00 | 1799.20 | 1822.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:00:00 | 1975.00 | 1962.96 | 0.00 | ORB-long ORB[1940.10,1966.70] vol=2.2x ATR=8.37 |
| Stop hit — per-position SL triggered | 2025-08-01 11:45:00 | 1966.63 | 1964.49 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:55:00 | 1956.10 | 1963.49 | 0.00 | ORB-short ORB[1960.00,1989.00] vol=1.5x ATR=8.28 |
| Stop hit — per-position SL triggered | 2025-08-06 12:10:00 | 1964.38 | 1961.01 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:45:00 | 1954.10 | 1973.35 | 0.00 | ORB-short ORB[1970.00,1992.00] vol=1.5x ATR=7.92 |
| Stop hit — per-position SL triggered | 2025-08-11 09:50:00 | 1962.02 | 1972.49 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:35:00 | 1960.20 | 1965.13 | 0.00 | ORB-short ORB[1963.00,1981.90] vol=1.8x ATR=5.93 |
| Stop hit — per-position SL triggered | 2025-08-12 10:00:00 | 1966.13 | 1963.84 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:50:00 | 2001.40 | 1992.83 | 0.00 | ORB-long ORB[1973.10,1993.50] vol=5.0x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-08-13 10:55:00 | 1995.53 | 1992.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:35:00 | 1956.30 | 1944.35 | 0.00 | ORB-long ORB[1931.00,1942.70] vol=2.5x ATR=5.74 |
| Stop hit — per-position SL triggered | 2025-08-21 09:50:00 | 1950.56 | 1946.88 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1930.50 | 1934.06 | 0.00 | ORB-short ORB[1931.00,1955.00] vol=2.2x ATR=4.29 |
| Stop hit — per-position SL triggered | 2025-08-25 11:20:00 | 1934.79 | 1934.13 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:35:00 | 1898.50 | 1888.07 | 0.00 | ORB-long ORB[1872.10,1894.80] vol=2.0x ATR=5.31 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1893.19 | 1893.36 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:55:00 | 1901.30 | 1895.89 | 0.00 | ORB-long ORB[1885.50,1901.00] vol=4.8x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:00:00 | 1908.34 | 1905.19 | 0.00 | T1 1.5R @ 1908.34 |
| Target hit | 2025-09-03 12:50:00 | 1910.20 | 1913.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 2062.50 | 2076.14 | 0.00 | ORB-short ORB[2070.00,2099.00] vol=2.2x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:35:00 | 2053.23 | 2073.17 | 0.00 | T1 1.5R @ 2053.23 |
| Target hit | 2025-09-24 15:20:00 | 2039.30 | 2057.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-10-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:00:00 | 1916.50 | 1927.67 | 0.00 | ORB-short ORB[1930.10,1952.70] vol=1.6x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-10-07 12:35:00 | 1920.62 | 1924.39 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 1917.80 | 1907.67 | 0.00 | ORB-long ORB[1890.00,1914.00] vol=2.5x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 12:25:00 | 1926.83 | 1913.17 | 0.00 | T1 1.5R @ 1926.83 |
| Target hit | 2025-10-08 15:20:00 | 1932.70 | 1927.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:25:00 | 1939.50 | 1947.17 | 0.00 | ORB-short ORB[1944.80,1964.00] vol=1.6x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:35:00 | 1933.54 | 1945.63 | 0.00 | T1 1.5R @ 1933.54 |
| Target hit | 2025-10-14 13:45:00 | 1935.00 | 1933.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:00:00 | 1942.50 | 1926.56 | 0.00 | ORB-long ORB[1906.00,1924.30] vol=1.6x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-10-15 12:10:00 | 1937.27 | 1931.45 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:05:00 | 1960.90 | 1945.49 | 0.00 | ORB-long ORB[1917.30,1938.70] vol=2.0x ATR=8.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:15:00 | 1973.39 | 1950.67 | 0.00 | T1 1.5R @ 1973.39 |
| Stop hit — per-position SL triggered | 2025-10-16 10:20:00 | 1960.90 | 1951.09 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:30:00 | 1924.80 | 1934.54 | 0.00 | ORB-short ORB[1930.70,1945.60] vol=2.0x ATR=6.15 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1930.95 | 1932.15 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 1887.10 | 1892.59 | 0.00 | ORB-short ORB[1895.40,1910.90] vol=1.8x ATR=5.51 |
| Stop hit — per-position SL triggered | 2025-10-20 11:35:00 | 1892.61 | 1891.73 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:55:00 | 1889.10 | 1881.77 | 0.00 | ORB-long ORB[1863.00,1888.80] vol=4.3x ATR=7.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:15:00 | 1900.04 | 1886.84 | 0.00 | T1 1.5R @ 1900.04 |
| Target hit | 2025-10-23 14:40:00 | 1894.10 | 1894.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 1932.00 | 1915.44 | 0.00 | ORB-long ORB[1908.00,1923.90] vol=3.2x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:55:00 | 1941.08 | 1922.55 | 0.00 | T1 1.5R @ 1941.08 |
| Stop hit — per-position SL triggered | 2025-10-30 13:05:00 | 1932.00 | 1926.87 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 1946.90 | 1939.86 | 0.00 | ORB-long ORB[1930.00,1946.60] vol=1.6x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:55:00 | 1954.26 | 1945.02 | 0.00 | T1 1.5R @ 1954.26 |
| Stop hit — per-position SL triggered | 2025-10-31 10:25:00 | 1946.90 | 1948.78 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 1722.40 | 1730.92 | 0.00 | ORB-short ORB[1725.00,1742.60] vol=2.8x ATR=5.41 |
| Stop hit — per-position SL triggered | 2025-11-12 09:40:00 | 1727.81 | 1729.88 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:50:00 | 1753.10 | 1761.37 | 0.00 | ORB-short ORB[1762.00,1785.80] vol=2.6x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:20:00 | 1746.15 | 1758.17 | 0.00 | T1 1.5R @ 1746.15 |
| Stop hit — per-position SL triggered | 2025-11-13 13:05:00 | 1753.10 | 1754.20 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:05:00 | 1704.10 | 1686.45 | 0.00 | ORB-long ORB[1681.60,1696.10] vol=3.8x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-11-19 10:10:00 | 1698.99 | 1688.22 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1677.90 | 1683.87 | 0.00 | ORB-short ORB[1683.00,1698.90] vol=2.8x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:50:00 | 1672.70 | 1680.53 | 0.00 | T1 1.5R @ 1672.70 |
| Stop hit — per-position SL triggered | 2025-11-20 10:05:00 | 1677.90 | 1679.51 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 1725.10 | 1722.31 | 0.00 | ORB-long ORB[1710.70,1722.40] vol=3.2x ATR=3.89 |
| Stop hit — per-position SL triggered | 2025-11-21 09:55:00 | 1721.21 | 1722.62 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:45:00 | 1694.10 | 1702.15 | 0.00 | ORB-short ORB[1705.00,1719.00] vol=2.4x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-11-24 10:55:00 | 1698.20 | 1701.46 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:35:00 | 1695.00 | 1702.59 | 0.00 | ORB-short ORB[1705.40,1715.60] vol=2.7x ATR=4.51 |
| Stop hit — per-position SL triggered | 2025-11-25 10:50:00 | 1699.51 | 1701.62 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 11:00:00 | 1681.30 | 1675.06 | 0.00 | ORB-long ORB[1665.10,1680.00] vol=16.1x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:05:00 | 1689.85 | 1675.48 | 0.00 | T1 1.5R @ 1689.85 |
| Stop hit — per-position SL triggered | 2025-11-27 11:10:00 | 1681.30 | 1676.35 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 1687.80 | 1681.62 | 0.00 | ORB-long ORB[1666.00,1683.20] vol=2.2x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 1682.57 | 1682.08 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:50:00 | 1659.70 | 1670.45 | 0.00 | ORB-short ORB[1671.70,1696.00] vol=3.0x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-12-02 11:20:00 | 1664.15 | 1668.21 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 11:10:00 | 1633.80 | 1637.44 | 0.00 | ORB-short ORB[1637.10,1652.20] vol=2.1x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:20:00 | 1627.38 | 1637.19 | 0.00 | T1 1.5R @ 1627.38 |
| Target hit | 2025-12-05 13:45:00 | 1628.70 | 1627.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2025-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:10:00 | 1655.20 | 1646.60 | 0.00 | ORB-long ORB[1631.10,1649.20] vol=2.3x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:20:00 | 1663.37 | 1649.58 | 0.00 | T1 1.5R @ 1663.37 |
| Stop hit — per-position SL triggered | 2025-12-10 10:55:00 | 1655.20 | 1658.51 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 1641.70 | 1629.84 | 0.00 | ORB-long ORB[1618.20,1630.30] vol=4.5x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:55:00 | 1647.76 | 1635.28 | 0.00 | T1 1.5R @ 1647.76 |
| Target hit | 2025-12-11 15:20:00 | 1668.00 | 1650.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:05:00 | 1676.70 | 1672.78 | 0.00 | ORB-long ORB[1664.80,1675.70] vol=2.3x ATR=3.56 |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 1673.14 | 1674.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:35:00 | 1707.20 | 1690.56 | 0.00 | ORB-long ORB[1669.00,1689.80] vol=1.7x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 1701.42 | 1693.71 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:00:00 | 1703.30 | 1696.01 | 0.00 | ORB-long ORB[1685.00,1701.90] vol=3.1x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 10:05:00 | 1711.62 | 1699.18 | 0.00 | T1 1.5R @ 1711.62 |
| Target hit | 2025-12-16 12:30:00 | 1712.10 | 1712.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2025-12-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:50:00 | 1683.40 | 1697.50 | 0.00 | ORB-short ORB[1698.20,1714.60] vol=1.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-12-17 11:00:00 | 1687.74 | 1696.37 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:55:00 | 1639.80 | 1651.25 | 0.00 | ORB-short ORB[1655.00,1672.10] vol=2.2x ATR=6.83 |
| Stop hit — per-position SL triggered | 2025-12-18 10:20:00 | 1646.63 | 1648.81 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:35:00 | 1721.40 | 1712.41 | 0.00 | ORB-long ORB[1702.60,1715.20] vol=2.2x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:50:00 | 1729.70 | 1716.41 | 0.00 | T1 1.5R @ 1729.70 |
| Target hit | 2025-12-23 15:20:00 | 1765.60 | 1755.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:45:00 | 1771.80 | 1760.51 | 0.00 | ORB-long ORB[1740.20,1762.90] vol=5.6x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:00:00 | 1781.34 | 1765.38 | 0.00 | T1 1.5R @ 1781.34 |
| Target hit | 2025-12-30 11:55:00 | 1783.80 | 1785.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2025-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:10:00 | 1797.60 | 1781.64 | 0.00 | ORB-long ORB[1771.20,1790.20] vol=2.5x ATR=5.71 |
| Stop hit — per-position SL triggered | 2025-12-31 11:55:00 | 1791.89 | 1787.90 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:30:00 | 1784.60 | 1770.40 | 0.00 | ORB-long ORB[1760.00,1782.00] vol=1.5x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-01-02 10:55:00 | 1779.39 | 1773.26 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:40:00 | 1803.20 | 1792.94 | 0.00 | ORB-long ORB[1776.30,1799.90] vol=2.0x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:25:00 | 1812.63 | 1799.56 | 0.00 | T1 1.5R @ 1812.63 |
| Stop hit — per-position SL triggered | 2026-01-05 10:35:00 | 1803.20 | 1800.91 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:05:00 | 1716.00 | 1722.94 | 0.00 | ORB-short ORB[1716.20,1732.60] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-01-14 11:25:00 | 1720.78 | 1722.32 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:05:00 | 1615.70 | 1626.72 | 0.00 | ORB-short ORB[1622.20,1645.00] vol=2.2x ATR=4.41 |
| Stop hit — per-position SL triggered | 2026-01-22 11:35:00 | 1620.11 | 1625.91 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:50:00 | 1541.50 | 1549.63 | 0.00 | ORB-short ORB[1547.00,1567.50] vol=2.1x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:00:00 | 1533.70 | 1545.54 | 0.00 | T1 1.5R @ 1533.70 |
| Stop hit — per-position SL triggered | 2026-01-29 14:30:00 | 1541.50 | 1540.71 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:40:00 | 1557.70 | 1536.48 | 0.00 | ORB-long ORB[1510.00,1526.60] vol=1.8x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:35:00 | 1565.70 | 1543.13 | 0.00 | T1 1.5R @ 1565.70 |
| Target hit | 2026-01-30 14:15:00 | 1560.90 | 1562.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — SELL (started 2026-04-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:20:00 | 1400.00 | 1405.98 | 0.00 | ORB-short ORB[1402.10,1416.70] vol=2.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-04-10 14:05:00 | 1404.30 | 1403.84 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 1413.50 | 1423.71 | 0.00 | ORB-short ORB[1422.30,1440.00] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2026-04-15 11:10:00 | 1417.71 | 1423.35 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1462.90 | 1452.28 | 0.00 | ORB-long ORB[1442.00,1452.00] vol=2.4x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 1470.11 | 1460.33 | 0.00 | T1 1.5R @ 1470.11 |
| Target hit | 2026-04-21 11:00:00 | 1469.00 | 1469.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 1434.40 | 1440.79 | 0.00 | ORB-short ORB[1438.60,1452.90] vol=5.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-04-23 11:30:00 | 1438.00 | 1440.63 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 1428.70 | 1422.84 | 0.00 | ORB-long ORB[1406.80,1426.20] vol=1.7x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 1437.27 | 1425.31 | 0.00 | T1 1.5R @ 1437.27 |
| Target hit | 2026-04-27 14:05:00 | 1444.00 | 1444.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 1430.70 | 1435.72 | 0.00 | ORB-short ORB[1431.20,1448.00] vol=1.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 1435.43 | 1434.81 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1414.10 | 1422.86 | 0.00 | ORB-short ORB[1418.10,1437.70] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-05-04 10:40:00 | 1418.12 | 1422.52 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 1419.40 | 1426.74 | 0.00 | ORB-short ORB[1424.40,1434.60] vol=3.0x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-05-06 11:50:00 | 1422.69 | 1425.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 09:35:00 | 1681.40 | 2025-05-16 10:05:00 | 1674.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1715.20 | 2025-05-23 09:45:00 | 1724.38 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1715.20 | 2025-05-23 10:15:00 | 1727.90 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2025-05-27 10:25:00 | 1692.10 | 2025-05-27 13:25:00 | 1682.49 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-05-27 10:25:00 | 1692.10 | 2025-05-27 15:20:00 | 1676.60 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2025-05-29 11:10:00 | 1742.00 | 2025-05-29 11:15:00 | 1752.93 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-05-29 11:10:00 | 1742.00 | 2025-05-29 12:30:00 | 1742.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 11:10:00 | 1710.90 | 2025-05-30 13:50:00 | 1704.03 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-05-30 11:10:00 | 1710.90 | 2025-05-30 15:15:00 | 1710.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-02 10:55:00 | 1777.70 | 2025-06-02 11:35:00 | 1770.86 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-05 09:30:00 | 1819.30 | 2025-06-05 09:50:00 | 1812.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-12 09:50:00 | 1917.20 | 2025-06-12 11:00:00 | 1924.61 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-06-19 11:05:00 | 1900.00 | 2025-06-19 11:30:00 | 1905.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-25 09:50:00 | 1942.10 | 2025-06-25 10:25:00 | 1934.49 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-27 09:35:00 | 2010.60 | 2025-06-27 09:40:00 | 2002.11 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-07-01 10:50:00 | 1986.20 | 2025-07-01 12:20:00 | 1992.39 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-02 09:40:00 | 1952.30 | 2025-07-02 09:45:00 | 1959.16 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-03 10:10:00 | 2008.00 | 2025-07-03 10:20:00 | 2020.67 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-07-03 10:10:00 | 2008.00 | 2025-07-03 15:15:00 | 2034.00 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2025-07-04 11:10:00 | 2024.70 | 2025-07-04 11:20:00 | 2016.37 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-04 11:10:00 | 2024.70 | 2025-07-04 11:25:00 | 2024.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 09:35:00 | 1981.00 | 2025-07-08 10:10:00 | 1972.16 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-08 09:35:00 | 1981.00 | 2025-07-08 13:40:00 | 1944.80 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2025-07-16 09:40:00 | 1957.80 | 2025-07-16 10:00:00 | 1962.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-18 09:45:00 | 1935.70 | 2025-07-18 09:50:00 | 1928.02 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-18 09:45:00 | 1935.70 | 2025-07-18 15:20:00 | 1903.90 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2025-07-23 11:05:00 | 1878.70 | 2025-07-23 14:35:00 | 1883.23 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-25 10:30:00 | 1829.40 | 2025-07-25 10:55:00 | 1819.55 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-07-25 10:30:00 | 1829.40 | 2025-07-25 15:20:00 | 1799.20 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2025-08-01 11:00:00 | 1975.00 | 2025-08-01 11:45:00 | 1966.63 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-08-06 10:55:00 | 1956.10 | 2025-08-06 12:10:00 | 1964.38 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-08-11 09:45:00 | 1954.10 | 2025-08-11 09:50:00 | 1962.02 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-12 09:35:00 | 1960.20 | 2025-08-12 10:00:00 | 1966.13 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-13 10:50:00 | 2001.40 | 2025-08-13 10:55:00 | 1995.53 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-21 09:35:00 | 1956.30 | 2025-08-21 09:50:00 | 1950.56 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-25 11:15:00 | 1930.50 | 2025-08-25 11:20:00 | 1934.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-08-29 10:35:00 | 1898.50 | 2025-08-29 12:15:00 | 1893.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-03 09:55:00 | 1901.30 | 2025-09-03 10:00:00 | 1908.34 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-03 09:55:00 | 1901.30 | 2025-09-03 12:50:00 | 1910.20 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2025-09-24 11:15:00 | 2062.50 | 2025-09-24 11:35:00 | 2053.23 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-09-24 11:15:00 | 2062.50 | 2025-09-24 15:20:00 | 2039.30 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2025-10-07 11:00:00 | 1916.50 | 2025-10-07 12:35:00 | 1920.62 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-08 11:15:00 | 1917.80 | 2025-10-08 12:25:00 | 1926.83 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-08 11:15:00 | 1917.80 | 2025-10-08 15:20:00 | 1932.70 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-10-14 10:25:00 | 1939.50 | 2025-10-14 10:35:00 | 1933.54 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-14 10:25:00 | 1939.50 | 2025-10-14 13:45:00 | 1935.00 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-10-15 11:00:00 | 1942.50 | 2025-10-15 12:10:00 | 1937.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-16 10:05:00 | 1960.90 | 2025-10-16 10:15:00 | 1973.39 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-16 10:05:00 | 1960.90 | 2025-10-16 10:20:00 | 1960.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 10:30:00 | 1924.80 | 2025-10-17 11:15:00 | 1930.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-20 11:15:00 | 1887.10 | 2025-10-20 11:35:00 | 1892.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-23 09:55:00 | 1889.10 | 2025-10-23 11:15:00 | 1900.04 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-10-23 09:55:00 | 1889.10 | 2025-10-23 14:40:00 | 1894.10 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-30 11:10:00 | 1932.00 | 2025-10-30 11:55:00 | 1941.08 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-30 11:10:00 | 1932.00 | 2025-10-30 13:05:00 | 1932.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 09:30:00 | 1946.90 | 2025-10-31 09:55:00 | 1954.26 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-31 09:30:00 | 1946.90 | 2025-10-31 10:25:00 | 1946.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 09:35:00 | 1722.40 | 2025-11-12 09:40:00 | 1727.81 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-13 10:50:00 | 1753.10 | 2025-11-13 11:20:00 | 1746.15 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-13 10:50:00 | 1753.10 | 2025-11-13 13:05:00 | 1753.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-19 10:05:00 | 1704.10 | 2025-11-19 10:10:00 | 1698.99 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-20 09:30:00 | 1677.90 | 2025-11-20 09:50:00 | 1672.70 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-20 09:30:00 | 1677.90 | 2025-11-20 10:05:00 | 1677.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 09:30:00 | 1725.10 | 2025-11-21 09:55:00 | 1721.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-24 10:45:00 | 1694.10 | 2025-11-24 10:55:00 | 1698.20 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-25 10:35:00 | 1695.00 | 2025-11-25 10:50:00 | 1699.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-27 11:00:00 | 1681.30 | 2025-11-27 11:05:00 | 1689.85 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-11-27 11:00:00 | 1681.30 | 2025-11-27 11:10:00 | 1681.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 09:30:00 | 1687.80 | 2025-11-28 09:35:00 | 1682.57 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-02 10:50:00 | 1659.70 | 2025-12-02 11:20:00 | 1664.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-05 11:10:00 | 1633.80 | 2025-12-05 11:20:00 | 1627.38 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-05 11:10:00 | 1633.80 | 2025-12-05 13:45:00 | 1628.70 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-12-10 10:10:00 | 1655.20 | 2025-12-10 10:20:00 | 1663.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-10 10:10:00 | 1655.20 | 2025-12-10 10:55:00 | 1655.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1641.70 | 2025-12-11 11:55:00 | 1647.76 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1641.70 | 2025-12-11 15:20:00 | 1668.00 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2025-12-12 11:05:00 | 1676.70 | 2025-12-12 12:15:00 | 1673.14 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-15 10:35:00 | 1707.20 | 2025-12-15 11:05:00 | 1701.42 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-16 10:00:00 | 1703.30 | 2025-12-16 10:05:00 | 1711.62 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-16 10:00:00 | 1703.30 | 2025-12-16 12:30:00 | 1712.10 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-12-17 10:50:00 | 1683.40 | 2025-12-17 11:00:00 | 1687.74 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-18 09:55:00 | 1639.80 | 2025-12-18 10:20:00 | 1646.63 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-12-23 09:35:00 | 1721.40 | 2025-12-23 09:50:00 | 1729.70 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-23 09:35:00 | 1721.40 | 2025-12-23 15:20:00 | 1765.60 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2025-12-30 09:45:00 | 1771.80 | 2025-12-30 10:00:00 | 1781.34 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-12-30 09:45:00 | 1771.80 | 2025-12-30 11:55:00 | 1783.80 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-12-31 11:10:00 | 1797.60 | 2025-12-31 11:55:00 | 1791.89 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-02 10:30:00 | 1784.60 | 2026-01-02 10:55:00 | 1779.39 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-05 09:40:00 | 1803.20 | 2026-01-05 10:25:00 | 1812.63 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-01-05 09:40:00 | 1803.20 | 2026-01-05 10:35:00 | 1803.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-14 11:05:00 | 1716.00 | 2026-01-14 11:25:00 | 1720.78 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-22 11:05:00 | 1615.70 | 2026-01-22 11:35:00 | 1620.11 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-29 10:50:00 | 1541.50 | 2026-01-29 12:00:00 | 1533.70 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-29 10:50:00 | 1541.50 | 2026-01-29 14:30:00 | 1541.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 10:40:00 | 1557.70 | 2026-01-30 11:35:00 | 1565.70 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-01-30 10:40:00 | 1557.70 | 2026-01-30 14:15:00 | 1560.90 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-04-10 10:20:00 | 1400.00 | 2026-04-10 14:05:00 | 1404.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1413.50 | 2026-04-15 11:10:00 | 1417.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1462.90 | 2026-04-21 09:45:00 | 1470.11 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1462.90 | 2026-04-21 11:00:00 | 1469.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1434.40 | 2026-04-23 11:30:00 | 1438.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1428.70 | 2026-04-27 09:45:00 | 1437.27 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1428.70 | 2026-04-27 14:05:00 | 1444.00 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2026-04-29 10:00:00 | 1430.70 | 2026-04-29 10:20:00 | 1435.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-04 10:30:00 | 1414.10 | 2026-05-04 10:40:00 | 1418.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-06 11:05:00 | 1419.40 | 2026-05-06 11:50:00 | 1422.69 | STOP_HIT | 1.00 | -0.23% |
