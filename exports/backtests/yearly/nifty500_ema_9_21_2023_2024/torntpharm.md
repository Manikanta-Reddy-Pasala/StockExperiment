# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4385.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 243 |
| ALERT1 | 170 |
| ALERT2 | 168 |
| ALERT2_SKIP | 119 |
| ALERT3 | 397 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 156 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 155 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 159 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 35 / 124
- **Target hits / Stop hits / Partials:** 1 / 155 / 3
- **Avg / median % per leg:** -0.39% / -0.86%
- **Sum % (uncompounded):** -62.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 90 | 12 | 13.3% | 1 | 89 | 0 | -0.53% | -48.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.62% | -1.2% |
| BUY @ 3rd Alert (retest2) | 88 | 12 | 13.6% | 1 | 87 | 0 | -0.53% | -46.8% |
| SELL (all) | 69 | 23 | 33.3% | 0 | 66 | 3 | -0.21% | -14.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 69 | 23 | 33.3% | 0 | 66 | 3 | -0.21% | -14.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.62% | -1.2% |
| retest2 (combined) | 157 | 35 | 22.3% | 1 | 153 | 3 | -0.39% | -61.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 1667.75 | 1660.20 | 1660.02 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 10:15:00 | 1624.60 | 1654.30 | 1658.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 11:15:00 | 1614.90 | 1646.42 | 1654.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 1633.30 | 1632.13 | 1642.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 11:15:00 | 1646.05 | 1634.90 | 1642.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 1646.05 | 1634.90 | 1642.26 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 15:15:00 | 1653.00 | 1647.25 | 1646.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 12:15:00 | 1657.75 | 1652.10 | 1649.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 14:15:00 | 1690.00 | 1693.12 | 1683.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 15:15:00 | 1685.05 | 1691.50 | 1684.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 1685.05 | 1691.50 | 1684.03 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 1841.80 | 1858.73 | 1859.56 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 1860.30 | 1859.10 | 1858.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 1880.10 | 1867.46 | 1863.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 15:15:00 | 1873.30 | 1874.96 | 1869.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 14:15:00 | 1904.85 | 1910.92 | 1900.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 1904.85 | 1910.92 | 1900.80 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 15:15:00 | 1891.00 | 1898.54 | 1899.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 11:15:00 | 1875.15 | 1890.72 | 1895.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 1897.00 | 1886.22 | 1890.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 1897.00 | 1886.22 | 1890.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1897.00 | 1886.22 | 1890.48 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 1915.45 | 1894.97 | 1893.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 09:15:00 | 1944.20 | 1909.04 | 1901.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 11:15:00 | 1897.50 | 1908.44 | 1902.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 11:15:00 | 1897.50 | 1908.44 | 1902.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 1897.50 | 1908.44 | 1902.83 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 1892.85 | 1900.83 | 1900.94 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 11:15:00 | 1904.30 | 1901.53 | 1901.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 12:15:00 | 1910.00 | 1903.22 | 1902.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 09:15:00 | 1930.00 | 1930.12 | 1920.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 12:15:00 | 1929.85 | 1929.32 | 1922.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 1929.85 | 1929.32 | 1922.57 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 1925.90 | 1937.91 | 1938.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 11:15:00 | 1922.55 | 1931.22 | 1934.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 1930.60 | 1927.29 | 1930.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 1930.60 | 1927.29 | 1930.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 1930.60 | 1927.29 | 1930.62 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 1953.50 | 1934.64 | 1932.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 10:15:00 | 1965.00 | 1940.71 | 1935.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 12:15:00 | 1941.85 | 1942.43 | 1937.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 13:15:00 | 1932.00 | 1940.34 | 1937.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 1932.00 | 1940.34 | 1937.10 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 09:15:00 | 1977.60 | 1989.22 | 1990.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 1963.60 | 1979.38 | 1985.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 1984.20 | 1975.41 | 1981.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 1984.20 | 1975.41 | 1981.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 1984.20 | 1975.41 | 1981.49 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 13:15:00 | 2005.20 | 1985.62 | 1984.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 2027.10 | 1997.03 | 1990.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 14:15:00 | 2051.85 | 2052.11 | 2034.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 2038.10 | 2049.35 | 2037.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 2038.10 | 2049.35 | 2037.34 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 2029.95 | 2050.07 | 2050.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 2012.00 | 2037.62 | 2043.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 1981.70 | 1980.90 | 1998.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 15:15:00 | 1988.00 | 1983.97 | 1992.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 1988.00 | 1983.97 | 1992.41 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 1963.60 | 1952.04 | 1951.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 13:15:00 | 1970.20 | 1955.67 | 1952.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 15:15:00 | 1950.05 | 1955.07 | 1953.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 15:15:00 | 1950.05 | 1955.07 | 1953.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 1950.05 | 1955.07 | 1953.13 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 1902.00 | 1946.79 | 1951.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 1879.60 | 1921.61 | 1937.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 1832.50 | 1811.88 | 1850.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 12:15:00 | 1847.65 | 1827.61 | 1848.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 1847.65 | 1827.61 | 1848.77 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 14:15:00 | 1873.95 | 1853.55 | 1851.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 09:15:00 | 1908.80 | 1866.91 | 1857.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 12:15:00 | 1902.00 | 1908.56 | 1893.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 13:15:00 | 1893.80 | 1905.61 | 1893.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 1893.80 | 1905.61 | 1893.46 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 1865.85 | 1886.27 | 1887.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 1860.80 | 1868.90 | 1874.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 09:15:00 | 1860.35 | 1860.01 | 1866.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 1860.35 | 1860.01 | 1866.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1860.35 | 1860.01 | 1866.24 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 14:15:00 | 1869.85 | 1860.48 | 1859.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 09:15:00 | 1876.15 | 1865.09 | 1862.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 1869.95 | 1879.14 | 1875.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 1869.95 | 1879.14 | 1875.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 1869.95 | 1879.14 | 1875.07 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 1838.75 | 1869.50 | 1872.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 1835.30 | 1849.64 | 1858.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 1849.75 | 1849.40 | 1857.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 13:15:00 | 1839.15 | 1842.81 | 1848.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 1839.15 | 1842.81 | 1848.43 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 1854.25 | 1847.10 | 1846.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 13:15:00 | 1867.45 | 1854.49 | 1850.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 11:15:00 | 1904.65 | 1909.78 | 1892.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 13:15:00 | 1890.15 | 1903.48 | 1892.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 1890.15 | 1903.48 | 1892.10 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 1856.30 | 1882.19 | 1884.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 09:15:00 | 1849.45 | 1864.23 | 1873.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 11:15:00 | 1863.85 | 1860.98 | 1870.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 1868.15 | 1858.28 | 1864.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 1868.15 | 1858.28 | 1864.79 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 1885.75 | 1868.57 | 1868.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 1891.45 | 1876.52 | 1872.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 15:15:00 | 1886.30 | 1889.17 | 1882.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1882.95 | 1887.93 | 1882.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1882.95 | 1887.93 | 1882.82 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 13:15:00 | 1871.30 | 1880.66 | 1880.72 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1901.95 | 1881.56 | 1880.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 1878.55 | 1883.56 | 1883.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 1876.60 | 1882.01 | 1883.04 | Break + close below crossover candle low |

### Cycle 27 — BUY (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 09:15:00 | 1902.15 | 1886.04 | 1884.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 11:15:00 | 1911.00 | 1893.86 | 1888.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 12:15:00 | 1893.00 | 1893.68 | 1889.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 1896.30 | 1894.21 | 1889.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 1896.30 | 1894.21 | 1889.77 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 14:15:00 | 1890.45 | 1910.80 | 1913.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 1878.00 | 1893.73 | 1902.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 1913.75 | 1894.86 | 1901.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 1913.75 | 1894.86 | 1901.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 1913.75 | 1894.86 | 1901.34 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 11:15:00 | 1935.95 | 1908.63 | 1906.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 1883.05 | 1907.12 | 1908.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 11:15:00 | 1864.75 | 1898.65 | 1904.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 1894.70 | 1893.27 | 1900.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 1889.95 | 1892.61 | 1899.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 1889.95 | 1892.61 | 1899.89 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 1924.95 | 1906.72 | 1905.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 10:15:00 | 1942.65 | 1931.63 | 1928.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-12 18:15:00 | 2051.10 | 2054.76 | 2040.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 2062.40 | 2060.47 | 2051.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 2062.40 | 2060.47 | 2051.72 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 11:15:00 | 2043.55 | 2050.25 | 2050.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 14:15:00 | 2039.95 | 2046.66 | 2049.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 2063.75 | 2048.58 | 2049.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 2063.75 | 2048.58 | 2049.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 2063.75 | 2048.58 | 2049.40 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 10:15:00 | 2071.45 | 2053.15 | 2051.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 13:15:00 | 2080.00 | 2063.43 | 2056.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 15:15:00 | 2115.60 | 2115.69 | 2101.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 11:15:00 | 2096.75 | 2112.23 | 2103.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 2096.75 | 2112.23 | 2103.19 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 2093.00 | 2100.05 | 2100.51 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 2112.75 | 2102.32 | 2101.08 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 2086.30 | 2097.60 | 2099.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 15:15:00 | 2084.75 | 2093.39 | 2096.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 2100.00 | 2094.72 | 2096.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 2100.00 | 2094.72 | 2096.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 2100.00 | 2094.72 | 2096.99 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 2110.15 | 2096.58 | 2094.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 2128.15 | 2115.00 | 2106.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 09:15:00 | 2138.80 | 2139.85 | 2127.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 10:15:00 | 2135.00 | 2138.88 | 2127.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 2135.00 | 2138.88 | 2127.99 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 14:15:00 | 2099.35 | 2124.94 | 2127.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 11:15:00 | 2090.00 | 2109.03 | 2118.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 12:15:00 | 2089.85 | 2088.55 | 2099.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 13:15:00 | 2100.05 | 2090.85 | 2099.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 13:15:00 | 2100.05 | 2090.85 | 2099.73 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 2094.80 | 2068.49 | 2065.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 10:15:00 | 2104.40 | 2075.67 | 2069.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 09:15:00 | 2083.55 | 2091.27 | 2081.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 2083.55 | 2091.27 | 2081.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 2083.55 | 2091.27 | 2081.38 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 2326.95 | 2342.41 | 2343.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 15:15:00 | 2312.50 | 2330.97 | 2337.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 2346.90 | 2334.16 | 2338.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 2346.90 | 2334.16 | 2338.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 2346.90 | 2334.16 | 2338.13 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 2355.05 | 2341.61 | 2341.03 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 2334.90 | 2340.99 | 2341.26 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 2352.60 | 2343.14 | 2341.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 2468.15 | 2369.56 | 2354.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 09:15:00 | 2452.70 | 2467.95 | 2452.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 2452.70 | 2467.95 | 2452.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 2452.70 | 2467.95 | 2452.05 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 2429.50 | 2449.24 | 2450.31 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 13:15:00 | 2479.30 | 2456.13 | 2453.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 2516.95 | 2475.47 | 2467.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 2500.05 | 2507.34 | 2492.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 10:15:00 | 2500.05 | 2507.34 | 2492.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 2500.05 | 2507.34 | 2492.59 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 2469.00 | 2491.04 | 2492.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 2460.20 | 2484.87 | 2489.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 2472.35 | 2452.54 | 2462.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 2472.35 | 2452.54 | 2462.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 2472.35 | 2452.54 | 2462.88 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 14:15:00 | 2475.15 | 2467.78 | 2467.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 2494.30 | 2472.88 | 2469.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 2499.00 | 2504.60 | 2491.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 10:15:00 | 2502.00 | 2504.08 | 2492.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 2502.00 | 2504.08 | 2492.21 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 2620.95 | 2640.66 | 2641.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 09:15:00 | 2586.55 | 2615.79 | 2624.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 14:15:00 | 2604.10 | 2601.04 | 2612.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 15:15:00 | 2607.25 | 2602.28 | 2612.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 2607.25 | 2602.28 | 2612.09 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 13:15:00 | 2629.05 | 2607.12 | 2604.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 14:15:00 | 2634.00 | 2612.50 | 2607.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 12:15:00 | 2621.90 | 2622.09 | 2614.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 13:15:00 | 2614.20 | 2620.51 | 2614.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 2614.20 | 2620.51 | 2614.86 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 2598.90 | 2626.37 | 2627.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 2589.45 | 2618.98 | 2624.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 2611.05 | 2605.02 | 2613.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 15:15:00 | 2611.05 | 2605.02 | 2613.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 2611.05 | 2605.02 | 2613.17 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 13:15:00 | 2623.50 | 2617.87 | 2617.36 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 2609.00 | 2620.26 | 2621.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 2596.00 | 2614.80 | 2618.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 2616.15 | 2612.53 | 2616.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 2616.15 | 2612.53 | 2616.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 2616.15 | 2612.53 | 2616.19 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 2644.80 | 2622.58 | 2620.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 2668.55 | 2631.78 | 2624.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 14:15:00 | 2661.65 | 2677.07 | 2657.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 2687.95 | 2679.97 | 2663.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 2687.95 | 2679.97 | 2663.32 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 2658.15 | 2679.07 | 2679.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 2652.60 | 2668.85 | 2673.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 2683.90 | 2671.15 | 2674.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 2683.90 | 2671.15 | 2674.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 2683.90 | 2671.15 | 2674.08 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 15:15:00 | 2694.95 | 2678.66 | 2677.15 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 09:15:00 | 2658.30 | 2674.59 | 2675.44 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 2723.05 | 2678.49 | 2675.51 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 2669.75 | 2682.22 | 2683.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 15:15:00 | 2656.65 | 2675.26 | 2679.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 13:15:00 | 2667.20 | 2665.66 | 2672.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 13:15:00 | 2667.20 | 2665.66 | 2672.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 13:15:00 | 2667.20 | 2665.66 | 2672.73 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 2719.15 | 2679.51 | 2676.70 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 2615.00 | 2672.60 | 2676.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 2612.20 | 2660.52 | 2670.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 11:15:00 | 2524.05 | 2516.83 | 2545.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 10:15:00 | 2546.15 | 2526.04 | 2537.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 2546.15 | 2526.04 | 2537.78 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 2548.00 | 2541.09 | 2540.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 14:15:00 | 2562.00 | 2545.27 | 2542.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 2566.95 | 2568.34 | 2557.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 2559.05 | 2565.47 | 2558.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 2559.05 | 2565.47 | 2558.17 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 09:15:00 | 2551.00 | 2560.62 | 2560.64 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 2576.00 | 2563.02 | 2561.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 2596.05 | 2569.63 | 2564.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 11:15:00 | 2656.20 | 2657.74 | 2630.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 2634.30 | 2656.56 | 2640.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 2634.30 | 2656.56 | 2640.88 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 13:15:00 | 2598.60 | 2627.01 | 2630.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 14:15:00 | 2590.50 | 2619.71 | 2626.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 2584.00 | 2562.22 | 2583.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 2584.00 | 2562.22 | 2583.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 2584.00 | 2562.22 | 2583.76 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 2611.05 | 2587.92 | 2586.97 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 2584.45 | 2591.09 | 2591.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 2565.40 | 2584.30 | 2588.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 15:15:00 | 2585.90 | 2583.56 | 2587.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 2581.45 | 2583.14 | 2586.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 2581.45 | 2583.14 | 2586.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 2580.35 | 2581.55 | 2583.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 2590.60 | 2583.36 | 2583.91 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 11:15:00 | 2591.55 | 2585.00 | 2584.60 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 2574.80 | 2582.96 | 2583.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 2563.25 | 2577.87 | 2581.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 12:15:00 | 2540.40 | 2537.88 | 2552.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 2534.85 | 2537.14 | 2547.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2534.85 | 2537.14 | 2547.25 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 2575.00 | 2554.59 | 2553.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 15:15:00 | 2582.00 | 2559.76 | 2556.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 2538.85 | 2555.58 | 2554.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 2538.85 | 2555.58 | 2554.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 2538.85 | 2555.58 | 2554.65 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 2530.90 | 2550.65 | 2552.49 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 2582.00 | 2553.05 | 2551.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 10:15:00 | 2609.50 | 2564.34 | 2557.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 2698.65 | 2702.41 | 2681.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 14:00:00 | 2698.65 | 2702.41 | 2681.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 2680.45 | 2695.81 | 2683.59 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 2641.40 | 2676.84 | 2680.91 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 2766.60 | 2691.03 | 2683.58 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 2663.85 | 2702.66 | 2704.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 2625.60 | 2687.25 | 2697.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 10:15:00 | 2588.75 | 2561.96 | 2588.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 10:15:00 | 2588.75 | 2561.96 | 2588.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 2588.75 | 2561.96 | 2588.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 2588.75 | 2561.96 | 2588.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 2587.60 | 2567.08 | 2588.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:45:00 | 2586.20 | 2567.08 | 2588.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 2579.60 | 2569.59 | 2587.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:30:00 | 2585.25 | 2569.59 | 2587.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 2588.40 | 2573.35 | 2587.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 2588.40 | 2573.35 | 2587.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 2587.00 | 2576.08 | 2587.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 2587.00 | 2576.08 | 2587.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 2609.80 | 2582.82 | 2589.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 2607.65 | 2582.82 | 2589.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 2583.05 | 2582.87 | 2589.25 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 2611.45 | 2593.88 | 2593.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 2625.00 | 2608.52 | 2601.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 2676.35 | 2694.42 | 2682.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 2676.35 | 2694.42 | 2682.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 2676.35 | 2694.42 | 2682.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 2676.35 | 2694.42 | 2682.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 2662.10 | 2687.96 | 2681.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 2662.10 | 2687.96 | 2681.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 2680.45 | 2678.74 | 2677.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 2680.45 | 2678.74 | 2677.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 2689.95 | 2680.98 | 2678.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 2698.70 | 2680.98 | 2678.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 2675.25 | 2679.76 | 2678.73 | SL hit (close<static) qty=1.00 sl=2676.35 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 2670.00 | 2676.90 | 2677.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 2653.15 | 2672.15 | 2675.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 15:15:00 | 2682.40 | 2672.97 | 2675.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 15:15:00 | 2682.40 | 2672.97 | 2675.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 2682.40 | 2672.97 | 2675.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 2673.00 | 2672.97 | 2675.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 2675.00 | 2673.37 | 2675.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 2658.65 | 2674.54 | 2675.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 2688.90 | 2677.41 | 2676.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 2688.90 | 2677.41 | 2676.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 2693.70 | 2682.03 | 2679.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 2655.15 | 2678.28 | 2677.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 2655.15 | 2678.28 | 2677.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 2655.15 | 2678.28 | 2677.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 2655.15 | 2678.28 | 2677.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 2656.55 | 2673.94 | 2676.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 12:15:00 | 2631.85 | 2661.33 | 2669.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 2700.00 | 2650.46 | 2659.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 2700.00 | 2650.46 | 2659.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2700.00 | 2650.46 | 2659.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:30:00 | 2689.00 | 2657.61 | 2662.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 2686.20 | 2657.61 | 2662.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 2686.60 | 2663.23 | 2661.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 2686.60 | 2663.23 | 2661.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 2722.45 | 2675.07 | 2667.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 2673.20 | 2693.01 | 2680.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 2673.20 | 2693.01 | 2680.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 2673.20 | 2693.01 | 2680.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 2673.20 | 2693.01 | 2680.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 2667.95 | 2688.00 | 2679.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 2667.95 | 2688.00 | 2679.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 2683.05 | 2687.01 | 2679.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:45:00 | 2690.00 | 2686.01 | 2679.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 13:30:00 | 2688.75 | 2684.81 | 2679.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 2659.65 | 2679.78 | 2677.87 | SL hit (close<static) qty=1.00 sl=2667.05 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 2659.55 | 2675.73 | 2676.21 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 2690.85 | 2676.64 | 2676.40 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 2670.00 | 2675.31 | 2675.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 12:15:00 | 2663.60 | 2672.97 | 2674.71 | Break + close below crossover candle low |

### Cycle 83 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 2691.75 | 2676.03 | 2675.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 2697.90 | 2680.41 | 2677.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 13:15:00 | 2682.85 | 2690.36 | 2684.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 13:15:00 | 2682.85 | 2690.36 | 2684.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 2682.85 | 2690.36 | 2684.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:00:00 | 2682.85 | 2690.36 | 2684.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 2675.05 | 2687.30 | 2684.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:30:00 | 2675.15 | 2687.30 | 2684.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 2683.00 | 2686.44 | 2683.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 2660.10 | 2686.44 | 2683.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2714.00 | 2691.95 | 2686.66 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2616.45 | 2671.14 | 2677.76 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 2705.00 | 2682.79 | 2681.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 2752.95 | 2699.57 | 2689.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 2763.05 | 2773.32 | 2753.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 09:45:00 | 2765.95 | 2773.32 | 2753.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2849.85 | 2870.71 | 2859.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 2849.85 | 2870.71 | 2859.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 2869.75 | 2870.52 | 2860.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 2891.00 | 2870.90 | 2864.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 14:00:00 | 2886.00 | 2888.25 | 2877.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:30:00 | 2881.55 | 2881.24 | 2876.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:45:00 | 2881.65 | 2877.52 | 2875.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 2885.90 | 2879.19 | 2876.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:45:00 | 2881.15 | 2879.19 | 2876.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 2871.25 | 2885.21 | 2881.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 2871.25 | 2885.21 | 2881.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 2893.00 | 2886.77 | 2882.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 2860.65 | 2878.69 | 2879.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 2860.65 | 2878.69 | 2879.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 10:15:00 | 2850.00 | 2872.95 | 2876.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 2893.00 | 2866.08 | 2870.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 2893.00 | 2866.08 | 2870.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2893.00 | 2866.08 | 2870.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 2893.00 | 2866.08 | 2870.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2889.60 | 2870.79 | 2872.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 2866.60 | 2869.85 | 2871.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:45:00 | 2860.70 | 2867.14 | 2870.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:45:00 | 2854.65 | 2854.87 | 2860.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 2818.10 | 2799.02 | 2796.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 12:15:00 | 2818.10 | 2799.02 | 2796.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 13:15:00 | 2830.90 | 2805.39 | 2799.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 2811.25 | 2824.55 | 2816.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 2811.25 | 2824.55 | 2816.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 2811.25 | 2824.55 | 2816.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 2811.25 | 2824.55 | 2816.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 2833.00 | 2826.24 | 2817.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 2833.00 | 2826.24 | 2817.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 2870.00 | 2879.88 | 2866.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 2870.00 | 2879.88 | 2866.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2860.15 | 2875.93 | 2866.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:00:00 | 2860.15 | 2875.93 | 2866.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 2851.50 | 2871.05 | 2864.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 2851.50 | 2871.05 | 2864.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 2880.15 | 2871.75 | 2866.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 2903.50 | 2876.85 | 2869.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:30:00 | 2901.10 | 2888.33 | 2876.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 2949.90 | 2976.27 | 2979.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 2949.90 | 2976.27 | 2979.10 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 3032.75 | 2989.83 | 2984.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 3063.15 | 3022.94 | 3005.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 3049.80 | 3073.61 | 3042.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 3049.80 | 3073.61 | 3042.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 3049.80 | 3073.61 | 3042.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 3179.55 | 3094.99 | 3063.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 3177.45 | 3133.69 | 3110.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:45:00 | 3171.70 | 3149.38 | 3124.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 3117.00 | 3146.52 | 3146.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 3117.00 | 3146.52 | 3146.86 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 3158.35 | 3148.88 | 3147.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 10:15:00 | 3192.95 | 3157.70 | 3152.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 11:15:00 | 3200.00 | 3200.97 | 3186.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 12:30:00 | 3209.50 | 3208.01 | 3191.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 3223.75 | 3221.76 | 3204.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 3199.80 | 3217.37 | 3204.07 | SL hit (close<ema400) qty=1.00 sl=3204.07 alert=retest1 |

### Cycle 92 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 3166.00 | 3195.19 | 3196.74 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 3220.90 | 3197.60 | 3197.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 10:15:00 | 3223.95 | 3202.87 | 3199.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 3213.70 | 3222.76 | 3211.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 3213.70 | 3222.76 | 3211.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 3213.70 | 3222.76 | 3211.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 3213.70 | 3222.76 | 3211.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 3220.00 | 3222.21 | 3212.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 3261.15 | 3222.21 | 3212.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 11:15:00 | 3294.25 | 3328.97 | 3331.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 3294.25 | 3328.97 | 3331.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 12:15:00 | 3283.40 | 3319.85 | 3326.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 3343.55 | 3321.68 | 3326.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 3343.55 | 3321.68 | 3326.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 3343.55 | 3321.68 | 3326.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:45:00 | 3347.90 | 3321.68 | 3326.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 3341.00 | 3325.54 | 3327.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 3333.25 | 3325.54 | 3327.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 3348.80 | 3321.12 | 3323.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 3348.80 | 3321.12 | 3323.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 3343.50 | 3325.60 | 3324.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 3355.00 | 3331.48 | 3327.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 3348.00 | 3353.78 | 3347.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 10:15:00 | 3348.00 | 3353.78 | 3347.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 3348.00 | 3353.78 | 3347.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 3348.00 | 3353.78 | 3347.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 3354.10 | 3353.84 | 3348.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:30:00 | 3344.95 | 3353.84 | 3348.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 3356.50 | 3354.37 | 3348.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:45:00 | 3349.60 | 3354.37 | 3348.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 3353.60 | 3354.22 | 3349.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:15:00 | 3345.95 | 3354.22 | 3349.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 3350.00 | 3353.37 | 3349.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 3348.15 | 3353.37 | 3349.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 3350.00 | 3352.70 | 3349.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:15:00 | 3345.50 | 3352.70 | 3349.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 3338.45 | 3349.85 | 3348.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 3331.85 | 3349.85 | 3348.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 3344.80 | 3348.84 | 3348.12 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 11:15:00 | 3342.15 | 3347.50 | 3347.57 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 12:15:00 | 3356.75 | 3349.35 | 3348.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 14:15:00 | 3365.50 | 3353.22 | 3350.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 12:15:00 | 3353.80 | 3358.43 | 3354.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 12:15:00 | 3353.80 | 3358.43 | 3354.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 3353.80 | 3358.43 | 3354.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 3346.50 | 3358.43 | 3354.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 3356.90 | 3358.12 | 3354.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 3361.30 | 3358.06 | 3355.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 3361.90 | 3358.17 | 3355.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 12:15:00 | 3335.40 | 3351.91 | 3353.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 3335.40 | 3351.91 | 3353.13 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 15:15:00 | 3365.00 | 3352.06 | 3350.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 3435.35 | 3368.71 | 3358.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 3480.60 | 3482.51 | 3449.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 15:00:00 | 3480.60 | 3482.51 | 3449.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 3474.35 | 3482.02 | 3457.69 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 3430.30 | 3449.42 | 3449.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 3421.75 | 3440.86 | 3445.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 3463.75 | 3434.77 | 3437.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 14:15:00 | 3463.75 | 3434.77 | 3437.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 3463.75 | 3434.77 | 3437.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 3463.75 | 3434.77 | 3437.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 15:15:00 | 3464.00 | 3440.62 | 3440.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 3475.00 | 3447.49 | 3443.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 3443.00 | 3452.85 | 3448.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 3443.00 | 3452.85 | 3448.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 3443.00 | 3452.85 | 3448.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 3443.00 | 3452.85 | 3448.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 3434.00 | 3449.08 | 3447.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 3455.15 | 3449.08 | 3447.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 3423.05 | 3443.87 | 3444.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 3423.05 | 3443.87 | 3444.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 3397.20 | 3417.75 | 3427.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 3431.15 | 3418.77 | 3426.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 3431.15 | 3418.77 | 3426.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 3431.15 | 3418.77 | 3426.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 3431.15 | 3418.77 | 3426.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 3438.75 | 3422.77 | 3427.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 3474.00 | 3422.77 | 3427.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 3431.55 | 3424.52 | 3427.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 3484.55 | 3424.52 | 3427.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 3421.80 | 3423.98 | 3427.23 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 3456.00 | 3430.92 | 3428.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 3466.65 | 3440.55 | 3433.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 3431.05 | 3443.70 | 3437.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 3431.05 | 3443.70 | 3437.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 3431.05 | 3443.70 | 3437.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 3431.05 | 3443.70 | 3437.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 3449.00 | 3444.76 | 3438.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 3488.95 | 3444.76 | 3438.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:30:00 | 3456.30 | 3453.21 | 3451.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:45:00 | 3460.40 | 3456.56 | 3453.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:45:00 | 3451.50 | 3459.93 | 3457.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 3474.95 | 3462.93 | 3458.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-17 15:15:00 | 3448.10 | 3456.66 | 3456.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 3448.10 | 3456.66 | 3456.98 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 3463.20 | 3457.52 | 3457.28 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 3435.20 | 3453.05 | 3455.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 3397.55 | 3441.95 | 3450.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 3369.90 | 3361.10 | 3386.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 3369.90 | 3361.10 | 3386.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 3369.90 | 3361.10 | 3386.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 3391.60 | 3361.10 | 3386.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 3414.65 | 3371.81 | 3388.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 3414.65 | 3371.81 | 3388.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 3409.90 | 3379.43 | 3390.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 3413.40 | 3379.43 | 3390.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 3450.00 | 3403.37 | 3400.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 3456.85 | 3414.07 | 3405.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 3429.35 | 3432.24 | 3418.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 3429.35 | 3432.24 | 3418.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 3473.65 | 3471.79 | 3455.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 3449.95 | 3471.79 | 3455.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 3434.50 | 3464.34 | 3453.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 3439.95 | 3464.34 | 3453.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 3426.25 | 3456.72 | 3451.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 3426.25 | 3456.72 | 3451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 3419.15 | 3443.32 | 3445.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 10:15:00 | 3406.20 | 3430.32 | 3438.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 3411.50 | 3405.31 | 3422.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 3411.50 | 3405.31 | 3422.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 3427.00 | 3409.65 | 3422.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 3408.00 | 3409.65 | 3422.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:00:00 | 3402.05 | 3412.09 | 3420.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 13:15:00 | 3409.10 | 3413.39 | 3420.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 13:15:00 | 3440.05 | 3418.72 | 3422.35 | SL hit (close>static) qty=1.00 sl=3430.25 alert=retest2 |

### Cycle 109 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 3491.50 | 3433.28 | 3428.63 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 3386.35 | 3426.30 | 3426.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 3356.05 | 3404.87 | 3416.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 10:15:00 | 3398.60 | 3374.52 | 3388.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 3398.60 | 3374.52 | 3388.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 3398.60 | 3374.52 | 3388.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 3398.60 | 3374.52 | 3388.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 3391.80 | 3377.98 | 3388.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:45:00 | 3375.95 | 3376.58 | 3386.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 3419.05 | 3388.56 | 3389.75 | SL hit (close>static) qty=1.00 sl=3402.05 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 3440.15 | 3398.88 | 3394.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 3460.20 | 3411.14 | 3400.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 12:15:00 | 3421.00 | 3448.60 | 3432.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 12:15:00 | 3421.00 | 3448.60 | 3432.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 3421.00 | 3448.60 | 3432.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 3421.00 | 3448.60 | 3432.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 3406.15 | 3440.11 | 3430.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:30:00 | 3407.70 | 3440.11 | 3430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 3446.30 | 3432.49 | 3428.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 3445.40 | 3432.49 | 3428.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 3450.35 | 3436.86 | 3431.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:30:00 | 3425.05 | 3436.86 | 3431.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 3521.20 | 3528.86 | 3502.20 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 3444.00 | 3482.34 | 3486.76 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3517.00 | 3483.74 | 3483.72 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 3460.80 | 3483.33 | 3483.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 3453.20 | 3473.31 | 3478.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 3482.20 | 3475.08 | 3479.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 3482.20 | 3475.08 | 3479.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 3482.20 | 3475.08 | 3479.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 3482.20 | 3475.08 | 3479.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 3490.00 | 3478.07 | 3480.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 3490.00 | 3478.07 | 3480.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 3497.65 | 3481.98 | 3481.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 15:15:00 | 3510.00 | 3494.24 | 3488.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 3502.05 | 3502.53 | 3494.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 3502.05 | 3502.53 | 3494.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 3502.05 | 3502.53 | 3494.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 3507.95 | 3502.53 | 3494.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 3513.95 | 3508.98 | 3500.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 3475.75 | 3508.98 | 3500.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 3448.25 | 3496.84 | 3495.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 3448.25 | 3496.84 | 3495.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 3443.70 | 3486.21 | 3491.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 3399.00 | 3451.13 | 3471.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 3448.60 | 3444.52 | 3464.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:00:00 | 3448.60 | 3444.52 | 3464.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 3464.70 | 3448.56 | 3464.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 3464.70 | 3448.56 | 3464.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 3460.65 | 3450.98 | 3464.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 3460.65 | 3450.98 | 3464.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 3430.85 | 3446.95 | 3461.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 3421.40 | 3440.27 | 3453.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:00:00 | 3418.60 | 3435.94 | 3450.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 3422.85 | 3431.54 | 3447.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 3413.95 | 3431.54 | 3447.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 3339.50 | 3346.13 | 3367.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 3318.15 | 3334.54 | 3358.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:45:00 | 3318.65 | 3328.14 | 3349.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 10:15:00 | 3377.80 | 3343.71 | 3351.37 | SL hit (close>static) qty=1.00 sl=3368.25 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 11:15:00 | 3414.10 | 3357.79 | 3357.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 14:15:00 | 3436.10 | 3389.05 | 3372.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 09:15:00 | 3387.10 | 3395.71 | 3379.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 3387.10 | 3395.71 | 3379.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 3387.10 | 3395.71 | 3379.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 3387.10 | 3395.71 | 3379.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 3367.30 | 3390.03 | 3378.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 3367.30 | 3390.03 | 3378.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 3334.85 | 3378.99 | 3374.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:00:00 | 3334.85 | 3378.99 | 3374.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 3320.25 | 3361.26 | 3366.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 3314.00 | 3351.81 | 3361.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 3186.50 | 3166.69 | 3217.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:15:00 | 3200.00 | 3166.69 | 3217.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 3218.35 | 3183.07 | 3216.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 3214.00 | 3183.07 | 3216.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 3223.95 | 3191.25 | 3217.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 3223.95 | 3191.25 | 3217.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 3220.85 | 3197.17 | 3217.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 3220.85 | 3197.17 | 3217.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 3201.75 | 3198.08 | 3216.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 15:15:00 | 3195.00 | 3198.08 | 3216.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 3192.65 | 3201.32 | 3213.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 3195.50 | 3184.21 | 3196.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3225.00 | 3184.20 | 3187.77 | SL hit (close>static) qty=1.00 sl=3223.55 alert=retest2 |

### Cycle 119 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 3203.20 | 3192.02 | 3190.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 3220.60 | 3201.09 | 3195.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 3168.85 | 3198.86 | 3196.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 3168.85 | 3198.86 | 3196.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 3168.85 | 3198.86 | 3196.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 3168.85 | 3198.86 | 3196.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 3166.05 | 3192.30 | 3193.45 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 13:15:00 | 3211.90 | 3185.99 | 3184.69 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 3173.50 | 3182.61 | 3183.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 3156.80 | 3175.86 | 3180.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 3191.90 | 3171.11 | 3176.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 3191.90 | 3171.11 | 3176.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 3191.90 | 3171.11 | 3176.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 3191.90 | 3171.11 | 3176.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 3185.00 | 3173.89 | 3177.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:00:00 | 3159.35 | 3172.30 | 3175.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 3162.80 | 3169.28 | 3174.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:00:00 | 3157.20 | 3169.28 | 3174.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 3154.90 | 3119.58 | 3115.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 3154.90 | 3119.58 | 3115.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 3182.00 | 3132.06 | 3121.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 3225.50 | 3236.82 | 3208.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 3225.50 | 3236.82 | 3208.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 3215.95 | 3230.27 | 3210.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 3212.05 | 3230.27 | 3210.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 3193.20 | 3222.86 | 3208.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 3193.20 | 3222.86 | 3208.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 3172.90 | 3212.87 | 3205.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 3172.90 | 3212.87 | 3205.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 3176.70 | 3200.12 | 3200.60 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 3227.20 | 3205.54 | 3203.02 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 3189.90 | 3202.27 | 3202.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 3181.50 | 3198.11 | 3200.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 14:15:00 | 3202.00 | 3198.89 | 3200.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 14:15:00 | 3202.00 | 3198.89 | 3200.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 3202.00 | 3198.89 | 3200.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 3202.00 | 3198.89 | 3200.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 3205.00 | 3200.11 | 3201.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 3227.05 | 3200.11 | 3201.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 3245.30 | 3209.15 | 3205.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 3300.00 | 3227.32 | 3213.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 3316.95 | 3352.85 | 3332.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 3316.95 | 3352.85 | 3332.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 3316.95 | 3352.85 | 3332.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 3316.95 | 3352.85 | 3332.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 3318.00 | 3345.88 | 3330.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 3296.05 | 3345.88 | 3330.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 3335.00 | 3348.04 | 3339.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 3335.00 | 3348.04 | 3339.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 3339.35 | 3346.30 | 3339.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 3323.15 | 3346.30 | 3339.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 3348.90 | 3346.82 | 3340.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:30:00 | 3350.45 | 3347.55 | 3341.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:00:00 | 3350.45 | 3347.55 | 3341.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 3364.95 | 3351.03 | 3343.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:30:00 | 3359.05 | 3351.62 | 3345.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 3340.50 | 3349.40 | 3345.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:45:00 | 3340.70 | 3349.40 | 3345.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 3341.00 | 3347.72 | 3344.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:00:00 | 3341.00 | 3347.72 | 3344.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-06 13:15:00 | 3335.35 | 3345.24 | 3344.02 | SL hit (close<static) qty=1.00 sl=3335.70 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 3330.15 | 3342.22 | 3342.76 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 3349.00 | 3342.93 | 3342.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 12:15:00 | 3361.50 | 3346.64 | 3344.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 14:15:00 | 3335.45 | 3345.09 | 3344.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 14:15:00 | 3335.45 | 3345.09 | 3344.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 3335.45 | 3345.09 | 3344.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 3335.45 | 3345.09 | 3344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 3344.00 | 3344.88 | 3344.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 3346.95 | 3344.88 | 3344.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 12:45:00 | 3346.50 | 3345.25 | 3344.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 13:15:00 | 3347.85 | 3345.25 | 3344.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 3348.70 | 3345.29 | 3344.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 3350.00 | 3346.23 | 3345.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:45:00 | 3345.55 | 3346.23 | 3345.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 3354.95 | 3347.98 | 3346.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 3352.45 | 3347.98 | 3346.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 3366.00 | 3351.58 | 3348.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 3378.10 | 3356.45 | 3350.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 13:00:00 | 3374.25 | 3364.54 | 3355.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 10:15:00 | 3370.95 | 3368.14 | 3360.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 3327.85 | 3353.16 | 3355.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 3327.85 | 3353.16 | 3355.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 3321.00 | 3346.73 | 3352.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 3342.40 | 3337.57 | 3345.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 3342.40 | 3337.57 | 3345.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 3342.40 | 3337.57 | 3345.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 3342.40 | 3337.57 | 3345.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 3340.90 | 3338.24 | 3344.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 3347.75 | 3338.24 | 3344.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 3345.35 | 3339.66 | 3344.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 3345.35 | 3339.66 | 3344.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 3339.90 | 3339.71 | 3344.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 3344.90 | 3339.71 | 3344.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 3346.00 | 3340.97 | 3344.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 3322.40 | 3340.97 | 3344.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 3324.70 | 3337.71 | 3342.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 3360.00 | 3345.23 | 3345.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 3360.00 | 3345.23 | 3345.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 3377.25 | 3355.60 | 3350.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 3354.15 | 3355.31 | 3350.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 11:00:00 | 3354.15 | 3355.31 | 3350.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 3348.30 | 3353.91 | 3350.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:45:00 | 3343.50 | 3353.91 | 3350.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 3341.50 | 3351.43 | 3349.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 3341.50 | 3351.43 | 3349.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 3343.30 | 3349.80 | 3349.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 15:00:00 | 3379.85 | 3355.81 | 3351.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 3398.85 | 3428.37 | 3430.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 13:15:00 | 3398.85 | 3428.37 | 3430.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 11:15:00 | 3389.05 | 3408.40 | 3419.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 14:15:00 | 3404.00 | 3402.39 | 3413.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 15:00:00 | 3404.00 | 3402.39 | 3413.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 3393.15 | 3400.23 | 3410.32 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 3418.70 | 3412.23 | 3411.35 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 3387.10 | 3406.85 | 3409.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 3371.25 | 3394.22 | 3402.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 15:15:00 | 3405.10 | 3391.90 | 3399.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 15:15:00 | 3405.10 | 3391.90 | 3399.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 3405.10 | 3391.90 | 3399.38 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 3400.00 | 3387.79 | 3387.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 3419.50 | 3396.32 | 3391.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 3369.05 | 3395.51 | 3392.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 3369.05 | 3395.51 | 3392.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 3369.05 | 3395.51 | 3392.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 3369.05 | 3395.51 | 3392.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 3397.55 | 3395.92 | 3392.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 3404.85 | 3399.31 | 3395.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:15:00 | 3405.80 | 3406.98 | 3400.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 13:15:00 | 3406.45 | 3398.80 | 3398.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 3370.50 | 3394.94 | 3397.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 3370.50 | 3394.94 | 3397.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 3366.05 | 3384.20 | 3391.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 15:15:00 | 3380.00 | 3378.57 | 3387.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:15:00 | 3390.40 | 3378.57 | 3387.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 3403.45 | 3383.55 | 3388.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 3392.20 | 3383.55 | 3388.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 3425.20 | 3391.88 | 3392.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 3427.90 | 3391.88 | 3392.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 3419.35 | 3397.37 | 3394.49 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 3356.10 | 3391.87 | 3394.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 3336.80 | 3376.15 | 3386.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 3332.90 | 3327.51 | 3346.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 09:15:00 | 3340.00 | 3327.51 | 3346.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 3327.00 | 3327.41 | 3344.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:00:00 | 3303.90 | 3321.82 | 3338.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:15:00 | 3138.70 | 3171.23 | 3189.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 3182.00 | 3166.02 | 3180.11 | SL hit (close>ema200) qty=0.50 sl=3166.02 alert=retest2 |

### Cycle 139 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 3208.50 | 3172.41 | 3169.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 3261.15 | 3190.16 | 3177.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 3231.35 | 3234.95 | 3208.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 3231.35 | 3234.95 | 3208.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 3248.90 | 3240.86 | 3222.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 3328.30 | 3240.86 | 3222.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 3277.00 | 3332.44 | 3333.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 10:15:00 | 3277.00 | 3332.44 | 3333.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 12:15:00 | 3272.15 | 3311.03 | 3322.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 3301.00 | 3291.28 | 3307.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 10:00:00 | 3301.00 | 3291.28 | 3307.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 3121.30 | 3170.76 | 3211.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 3103.65 | 3170.76 | 3211.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 12:30:00 | 3110.30 | 3120.85 | 3150.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 14:45:00 | 3114.75 | 3119.67 | 3144.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 3185.80 | 3154.98 | 3154.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 3185.80 | 3154.98 | 3154.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 3221.00 | 3171.05 | 3162.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 10:15:00 | 3218.00 | 3236.03 | 3210.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 10:15:00 | 3218.00 | 3236.03 | 3210.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 3218.00 | 3236.03 | 3210.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 3218.00 | 3236.03 | 3210.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 3187.85 | 3226.39 | 3208.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 3187.85 | 3226.39 | 3208.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 3189.60 | 3219.04 | 3206.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:45:00 | 3202.20 | 3208.57 | 3203.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 3140.85 | 3191.31 | 3196.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 3140.85 | 3191.31 | 3196.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 3120.10 | 3167.93 | 3184.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 3081.75 | 3075.82 | 3107.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 3081.75 | 3075.82 | 3107.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 3081.75 | 3075.82 | 3107.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 3094.85 | 3075.82 | 3107.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 2996.80 | 3057.06 | 3083.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 2974.55 | 3042.05 | 3073.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 09:15:00 | 3055.10 | 3051.97 | 3051.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 3055.10 | 3051.97 | 3051.58 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 11:15:00 | 3031.90 | 3049.40 | 3050.57 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 3075.80 | 3051.28 | 3050.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 3079.80 | 3056.99 | 3053.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 10:15:00 | 3084.05 | 3094.27 | 3078.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 10:15:00 | 3084.05 | 3094.27 | 3078.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 3084.05 | 3094.27 | 3078.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 3084.05 | 3094.27 | 3078.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 3087.05 | 3092.82 | 3079.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:30:00 | 3080.00 | 3092.82 | 3079.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 3070.15 | 3088.29 | 3078.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 3070.15 | 3088.29 | 3078.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 3086.70 | 3087.97 | 3079.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:15:00 | 3095.40 | 3087.97 | 3079.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 3048.40 | 3080.48 | 3078.32 | SL hit (close<static) qty=1.00 sl=3065.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 3039.40 | 3072.26 | 3074.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 3024.55 | 3038.31 | 3050.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 3032.50 | 3031.82 | 3044.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:30:00 | 3033.40 | 3031.82 | 3044.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 3047.55 | 3034.61 | 3043.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 3047.55 | 3034.61 | 3043.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 3050.50 | 3037.79 | 3043.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:45:00 | 3034.95 | 3036.15 | 3042.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 12:15:00 | 2975.45 | 2958.42 | 2957.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 2975.45 | 2958.42 | 2957.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 2985.00 | 2964.69 | 2960.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 3070.00 | 3080.65 | 3058.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:30:00 | 3071.00 | 3080.65 | 3058.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 3053.50 | 3075.22 | 3058.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 3041.60 | 3075.22 | 3058.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 3080.60 | 3076.30 | 3060.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 3050.40 | 3076.30 | 3060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 3068.85 | 3075.40 | 3062.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:30:00 | 3066.45 | 3075.40 | 3062.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 3071.20 | 3074.56 | 3063.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:30:00 | 3075.05 | 3074.56 | 3063.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 3059.20 | 3070.76 | 3063.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 3059.20 | 3070.76 | 3063.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 3053.25 | 3067.26 | 3062.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 3051.00 | 3067.26 | 3062.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 3086.30 | 3072.74 | 3066.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 3074.25 | 3072.74 | 3066.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 3070.00 | 3074.14 | 3068.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 3070.00 | 3074.14 | 3068.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 3099.10 | 3079.13 | 3071.30 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 3049.60 | 3067.24 | 3069.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 3047.55 | 3060.74 | 3065.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 3067.60 | 3061.91 | 3065.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 3067.60 | 3061.91 | 3065.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 3067.60 | 3061.91 | 3065.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 3067.60 | 3061.91 | 3065.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 3068.20 | 3063.17 | 3065.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 3054.15 | 3063.17 | 3065.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 3081.00 | 3066.74 | 3066.94 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 3075.00 | 3068.39 | 3067.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 3089.15 | 3072.54 | 3069.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 3229.95 | 3235.05 | 3201.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:00:00 | 3229.95 | 3235.05 | 3201.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 3223.25 | 3232.69 | 3203.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:45:00 | 3206.50 | 3232.69 | 3203.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 3226.25 | 3230.97 | 3207.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 3214.80 | 3230.97 | 3207.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 3222.70 | 3229.32 | 3209.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:30:00 | 3219.00 | 3229.32 | 3209.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 3218.15 | 3242.92 | 3227.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 3218.15 | 3242.92 | 3227.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 3203.50 | 3235.03 | 3225.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 3198.25 | 3235.03 | 3225.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 3247.80 | 3232.39 | 3225.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:15:00 | 3221.50 | 3232.39 | 3225.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 3221.50 | 3230.21 | 3225.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 3217.90 | 3228.49 | 3225.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 3207.20 | 3224.23 | 3223.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 3207.20 | 3224.23 | 3223.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 3211.40 | 3221.67 | 3222.44 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 13:15:00 | 3226.95 | 3223.50 | 3223.19 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 3219.35 | 3222.67 | 3222.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 3177.55 | 3212.42 | 3218.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 3225.25 | 3207.54 | 3213.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 3225.25 | 3207.54 | 3213.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 3225.25 | 3207.54 | 3213.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 3230.50 | 3207.54 | 3213.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 3228.40 | 3211.71 | 3215.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 3228.40 | 3211.71 | 3215.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 3223.85 | 3217.21 | 3217.07 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 3202.05 | 3217.08 | 3218.10 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 3247.00 | 3208.55 | 3205.63 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 09:15:00 | 3209.60 | 3236.57 | 3239.43 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 3256.30 | 3238.09 | 3237.72 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 3201.85 | 3230.84 | 3234.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 11:15:00 | 3191.40 | 3221.66 | 3229.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 3193.40 | 3155.77 | 3174.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 3193.40 | 3155.77 | 3174.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 3193.40 | 3155.77 | 3174.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 3193.40 | 3155.77 | 3174.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 3164.60 | 3157.54 | 3173.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:30:00 | 3182.10 | 3157.54 | 3173.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 3185.50 | 3163.13 | 3174.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:00:00 | 3185.50 | 3163.13 | 3174.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 12:15:00 | 3209.90 | 3172.49 | 3177.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:45:00 | 3212.00 | 3172.49 | 3177.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 13:15:00 | 3218.50 | 3181.69 | 3181.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 3221.70 | 3189.69 | 3185.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 3214.40 | 3219.07 | 3208.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 3214.40 | 3219.07 | 3208.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3214.40 | 3219.07 | 3208.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 3217.50 | 3219.07 | 3208.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 3213.70 | 3217.99 | 3208.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 3213.70 | 3217.99 | 3208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 3220.00 | 3218.39 | 3209.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 3220.00 | 3218.39 | 3209.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 3211.40 | 3217.00 | 3209.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 3211.40 | 3217.00 | 3209.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 3241.40 | 3221.88 | 3212.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:45:00 | 3249.90 | 3229.02 | 3216.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:00:00 | 3243.00 | 3248.97 | 3238.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:45:00 | 3247.30 | 3247.33 | 3238.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:30:00 | 3245.00 | 3249.85 | 3240.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 3259.70 | 3252.50 | 3243.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:30:00 | 3241.90 | 3252.50 | 3243.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 3244.00 | 3250.80 | 3243.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 3248.90 | 3250.80 | 3243.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 3250.00 | 3250.64 | 3243.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 3256.80 | 3250.64 | 3243.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 3274.60 | 3255.43 | 3246.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 3291.80 | 3255.43 | 3246.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:45:00 | 3289.10 | 3269.42 | 3254.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 3194.50 | 3284.20 | 3286.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 3194.50 | 3284.20 | 3286.27 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 3319.00 | 3284.31 | 3279.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 3340.70 | 3295.59 | 3285.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 3320.60 | 3326.23 | 3310.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 13:15:00 | 3320.60 | 3326.23 | 3310.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 3320.60 | 3326.23 | 3310.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:45:00 | 3315.80 | 3326.23 | 3310.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 3328.40 | 3329.72 | 3318.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 3323.90 | 3329.72 | 3318.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 3324.70 | 3328.71 | 3318.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 3327.30 | 3328.71 | 3318.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 3326.30 | 3329.80 | 3321.99 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 3268.30 | 3315.61 | 3316.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 3254.90 | 3303.47 | 3311.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 3274.70 | 3271.64 | 3289.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 3274.70 | 3271.64 | 3289.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 3274.70 | 3271.64 | 3289.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 3259.00 | 3271.64 | 3289.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 3275.70 | 3272.45 | 3288.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:15:00 | 3271.00 | 3272.45 | 3288.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 3264.70 | 3271.42 | 3281.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:15:00 | 3268.70 | 3274.59 | 3280.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:00:00 | 3260.30 | 3271.73 | 3278.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 3299.80 | 3276.94 | 3279.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:45:00 | 3256.80 | 3273.83 | 3278.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 12:15:00 | 3294.10 | 3282.60 | 3281.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 3294.10 | 3282.60 | 3281.51 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 09:15:00 | 3247.10 | 3279.67 | 3280.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 3230.90 | 3261.12 | 3271.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 3177.30 | 3159.36 | 3192.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 10:00:00 | 3177.30 | 3159.36 | 3192.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 11:15:00 | 3195.40 | 3168.67 | 3191.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 12:00:00 | 3195.40 | 3168.67 | 3191.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 3205.50 | 3176.04 | 3192.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 13:00:00 | 3205.50 | 3176.04 | 3192.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 13:15:00 | 3216.80 | 3184.19 | 3194.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 14:00:00 | 3216.80 | 3184.19 | 3194.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 3228.70 | 3202.15 | 3201.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 3253.00 | 3212.32 | 3206.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 3225.00 | 3225.65 | 3217.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 3245.00 | 3225.65 | 3217.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 3231.60 | 3226.84 | 3218.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 3261.60 | 3234.25 | 3228.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:00:00 | 3264.10 | 3240.22 | 3231.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 3265.00 | 3249.94 | 3237.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 3267.80 | 3257.33 | 3246.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 3296.20 | 3265.60 | 3252.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:30:00 | 3304.00 | 3271.68 | 3256.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 3309.50 | 3276.48 | 3259.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 3304.40 | 3282.07 | 3263.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:45:00 | 3308.90 | 3286.45 | 3267.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3243.30 | 3282.25 | 3270.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 3243.30 | 3282.25 | 3270.70 | SL hit (close<static) qty=1.00 sl=3250.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 3244.60 | 3265.76 | 3266.12 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3354.00 | 3278.18 | 3271.38 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 3233.00 | 3267.39 | 3271.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 14:15:00 | 3211.30 | 3242.07 | 3256.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 13:15:00 | 3179.00 | 3172.74 | 3196.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:00:00 | 3179.00 | 3172.74 | 3196.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 3172.10 | 3168.43 | 3180.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 3155.90 | 3168.43 | 3180.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 3165.40 | 3160.23 | 3170.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:00:00 | 3163.00 | 3160.79 | 3170.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 3162.40 | 3160.47 | 3165.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 3173.30 | 3163.03 | 3166.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 3173.30 | 3163.03 | 3166.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 3158.50 | 3162.13 | 3165.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:15:00 | 3150.10 | 3162.13 | 3165.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 3150.10 | 3159.72 | 3164.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 3129.20 | 3162.87 | 3163.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 3140.20 | 3133.44 | 3143.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 3147.40 | 3131.79 | 3133.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 3147.40 | 3134.91 | 3134.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 3147.40 | 3134.91 | 3134.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 3155.70 | 3145.65 | 3141.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 3203.80 | 3208.98 | 3188.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 3196.20 | 3208.98 | 3188.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3194.90 | 3205.42 | 3190.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:45:00 | 3193.30 | 3205.42 | 3190.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3203.40 | 3202.69 | 3191.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:30:00 | 3210.00 | 3204.13 | 3193.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 3209.90 | 3204.13 | 3193.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 3224.30 | 3244.03 | 3246.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 3224.30 | 3244.03 | 3246.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 3216.90 | 3238.60 | 3243.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 3186.20 | 3185.72 | 3206.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 3186.20 | 3185.72 | 3206.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3183.20 | 3168.56 | 3176.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 3183.20 | 3168.56 | 3176.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3175.00 | 3169.85 | 3176.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 3168.30 | 3169.85 | 3176.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 3182.60 | 3172.40 | 3177.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:30:00 | 3179.00 | 3172.40 | 3177.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 3182.70 | 3174.46 | 3177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 3182.70 | 3174.46 | 3177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 3182.40 | 3176.05 | 3177.98 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 3196.10 | 3180.20 | 3179.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 3214.70 | 3187.10 | 3182.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 3181.10 | 3205.65 | 3196.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3189.20 | 3202.36 | 3195.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 3185.60 | 3202.36 | 3195.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 3188.20 | 3199.53 | 3195.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 3188.20 | 3199.53 | 3195.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 3196.50 | 3198.92 | 3195.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 3178.60 | 3198.92 | 3195.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 3208.00 | 3200.74 | 3196.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 3212.50 | 3200.74 | 3196.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 3322.00 | 3358.34 | 3360.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 3322.00 | 3358.34 | 3360.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 3314.80 | 3349.63 | 3355.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 3342.60 | 3342.49 | 3350.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:00:00 | 3342.60 | 3342.49 | 3350.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 3356.00 | 3342.95 | 3348.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:15:00 | 3374.80 | 3342.95 | 3348.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 3374.20 | 3349.20 | 3350.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 3384.20 | 3349.20 | 3350.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 3368.30 | 3353.02 | 3352.46 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 3317.00 | 3349.23 | 3353.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 3302.30 | 3339.85 | 3348.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 3345.40 | 3330.65 | 3338.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 3327.80 | 3330.08 | 3337.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 3341.70 | 3330.08 | 3337.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 3340.50 | 3333.05 | 3337.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 3347.20 | 3333.05 | 3337.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 3357.60 | 3337.96 | 3339.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 3357.60 | 3337.96 | 3339.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 3317.50 | 3334.48 | 3337.39 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 3344.00 | 3338.89 | 3338.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 3348.70 | 3340.86 | 3339.55 | Break + close above crossover candle high |

### Cycle 176 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 3329.40 | 3338.56 | 3338.62 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 3339.10 | 3338.67 | 3338.67 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 3322.00 | 3335.34 | 3337.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 3313.00 | 3329.10 | 3333.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 3351.10 | 3330.64 | 3333.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 3367.30 | 3337.97 | 3336.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 3387.10 | 3347.80 | 3341.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 3428.70 | 3434.04 | 3410.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 3428.70 | 3434.04 | 3410.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3476.00 | 3512.30 | 3505.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3476.00 | 3512.30 | 3505.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3475.10 | 3504.86 | 3502.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 3475.10 | 3504.86 | 3502.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 3491.90 | 3500.62 | 3500.83 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 3504.60 | 3501.42 | 3501.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 15:15:00 | 3510.90 | 3503.79 | 3502.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 3542.80 | 3542.97 | 3528.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 3546.80 | 3542.97 | 3528.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 3542.00 | 3542.77 | 3529.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 3573.40 | 3532.29 | 3528.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 3649.10 | 3700.13 | 3701.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 3649.10 | 3700.13 | 3701.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 3630.40 | 3671.56 | 3686.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 10:15:00 | 3552.00 | 3548.77 | 3578.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:30:00 | 3551.40 | 3548.77 | 3578.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3595.80 | 3564.90 | 3576.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 3595.80 | 3564.90 | 3576.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3595.00 | 3570.92 | 3578.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 3582.40 | 3570.92 | 3578.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 3599.60 | 3584.93 | 3583.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 3599.60 | 3584.93 | 3583.61 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 3567.50 | 3583.09 | 3583.11 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 3596.10 | 3584.43 | 3583.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 3600.00 | 3587.55 | 3584.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 3600.30 | 3600.79 | 3593.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 3600.30 | 3600.79 | 3593.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 3597.00 | 3600.03 | 3593.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 3594.60 | 3600.03 | 3593.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 3625.00 | 3605.03 | 3596.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 3642.00 | 3624.38 | 3616.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 3645.10 | 3623.24 | 3619.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:00:00 | 3642.00 | 3636.44 | 3627.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 3646.40 | 3637.18 | 3630.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 3665.50 | 3659.41 | 3648.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 3643.60 | 3659.41 | 3648.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 3660.10 | 3659.55 | 3649.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 3632.20 | 3659.55 | 3649.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 3668.50 | 3665.95 | 3656.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 3660.20 | 3665.95 | 3656.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 3648.60 | 3662.48 | 3655.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 3648.60 | 3662.48 | 3655.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 3651.70 | 3660.33 | 3655.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 3664.00 | 3660.33 | 3655.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 3650.50 | 3659.41 | 3655.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 3650.50 | 3659.41 | 3655.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 3656.80 | 3658.89 | 3655.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:45:00 | 3668.40 | 3660.63 | 3657.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 3678.00 | 3671.66 | 3665.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3633.00 | 3664.72 | 3664.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 3633.00 | 3664.72 | 3664.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 3596.00 | 3628.88 | 3645.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 3606.40 | 3592.02 | 3610.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 3606.40 | 3592.02 | 3610.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3586.90 | 3577.99 | 3592.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 3586.90 | 3577.99 | 3592.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3599.60 | 3581.79 | 3591.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3599.60 | 3581.79 | 3591.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 3597.50 | 3584.93 | 3592.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 3594.40 | 3584.93 | 3592.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 3584.10 | 3584.76 | 3591.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 3599.90 | 3584.76 | 3591.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 3599.90 | 3587.79 | 3592.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 3547.50 | 3587.79 | 3592.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3568.00 | 3583.83 | 3590.18 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 3606.50 | 3589.39 | 3589.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 13:15:00 | 3609.90 | 3600.93 | 3596.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 3594.10 | 3599.57 | 3596.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 3596.20 | 3598.89 | 3596.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 3614.80 | 3603.53 | 3598.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3577.60 | 3598.12 | 3597.26 | SL hit (close<static) qty=1.00 sl=3591.30 alert=retest2 |

### Cycle 188 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 3573.10 | 3593.11 | 3595.06 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 3606.20 | 3595.64 | 3594.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 3627.50 | 3604.71 | 3599.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 3595.50 | 3603.51 | 3600.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 3615.00 | 3605.80 | 3601.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 3607.50 | 3605.80 | 3601.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3592.00 | 3621.09 | 3614.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 3592.00 | 3621.09 | 3614.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3574.50 | 3611.77 | 3610.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 3574.50 | 3611.77 | 3610.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 3581.90 | 3605.80 | 3608.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 3560.00 | 3577.47 | 3586.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 3540.90 | 3539.79 | 3554.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:30:00 | 3540.10 | 3539.79 | 3554.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 3556.30 | 3545.15 | 3554.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 3551.50 | 3545.15 | 3554.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 3536.00 | 3543.32 | 3552.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3521.70 | 3536.76 | 3548.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 3522.80 | 3533.25 | 3545.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 3519.20 | 3533.25 | 3545.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 3556.80 | 3546.27 | 3545.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 3556.80 | 3546.27 | 3545.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 3565.30 | 3550.07 | 3547.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 3650.70 | 3661.04 | 3638.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 3650.70 | 3661.04 | 3638.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3624.50 | 3651.34 | 3638.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 3624.50 | 3651.34 | 3638.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3612.00 | 3643.47 | 3635.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:30:00 | 3610.30 | 3643.47 | 3635.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 3612.60 | 3630.14 | 3631.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 3593.60 | 3621.16 | 3626.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 12:15:00 | 3558.30 | 3552.49 | 3578.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:30:00 | 3553.90 | 3552.49 | 3578.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 3564.40 | 3554.87 | 3576.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 3578.80 | 3554.87 | 3576.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 3587.90 | 3563.10 | 3576.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3594.00 | 3571.02 | 3579.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3593.20 | 3575.46 | 3580.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 3580.90 | 3576.92 | 3580.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 3589.90 | 3583.70 | 3583.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 3589.90 | 3583.70 | 3583.18 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 3558.00 | 3580.37 | 3581.87 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 3608.40 | 3581.49 | 3581.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 3624.10 | 3594.54 | 3587.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 3585.00 | 3592.80 | 3588.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 3575.10 | 3589.26 | 3586.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 3575.10 | 3589.26 | 3586.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 3564.30 | 3582.69 | 3584.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 3544.00 | 3572.19 | 3579.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 3521.90 | 3513.73 | 3531.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 3521.90 | 3513.73 | 3531.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3492.00 | 3511.01 | 3527.58 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 3548.60 | 3529.11 | 3526.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 13:15:00 | 3560.60 | 3535.41 | 3529.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 3538.60 | 3545.08 | 3538.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 3548.70 | 3545.81 | 3539.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 3545.70 | 3545.81 | 3539.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3532.00 | 3543.41 | 3540.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 3532.00 | 3543.41 | 3540.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3537.50 | 3542.23 | 3539.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 3547.50 | 3542.23 | 3539.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 3545.70 | 3546.72 | 3543.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:30:00 | 3542.40 | 3544.03 | 3543.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 3536.80 | 3541.49 | 3542.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3536.80 | 3541.49 | 3542.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 3490.80 | 3524.63 | 3533.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 3526.50 | 3516.50 | 3525.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3518.50 | 3516.90 | 3524.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3520.70 | 3516.90 | 3524.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3540.60 | 3521.64 | 3525.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 3540.60 | 3521.64 | 3525.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 3545.10 | 3526.33 | 3527.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 3545.10 | 3526.33 | 3527.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 3554.00 | 3531.87 | 3530.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 3570.00 | 3540.89 | 3536.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 3568.60 | 3578.18 | 3566.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 3568.60 | 3578.18 | 3566.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 3569.50 | 3576.45 | 3566.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3565.20 | 3574.96 | 3566.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3568.00 | 3573.57 | 3567.00 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 3564.80 | 3586.24 | 3588.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 3557.90 | 3580.57 | 3585.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 3598.00 | 3586.33 | 3585.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 3605.80 | 3590.22 | 3587.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 3560.30 | 3587.78 | 3586.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3596.90 | 3589.60 | 3587.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 3598.20 | 3591.32 | 3588.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 3598.20 | 3591.32 | 3588.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 3599.90 | 3594.38 | 3590.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 3604.00 | 3594.38 | 3590.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 3585.20 | 3594.86 | 3591.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 3585.20 | 3594.86 | 3591.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 3590.00 | 3593.89 | 3591.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 3582.80 | 3593.89 | 3591.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 3590.00 | 3593.11 | 3591.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 3562.60 | 3586.37 | 3588.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 3562.60 | 3586.37 | 3588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 3545.50 | 3578.19 | 3584.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 3595.30 | 3576.66 | 3582.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3591.80 | 3579.69 | 3583.43 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 3618.20 | 3589.82 | 3587.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 3626.50 | 3597.16 | 3591.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 3604.40 | 3611.31 | 3601.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 3603.40 | 3609.73 | 3602.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 3600.10 | 3609.73 | 3602.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 3581.60 | 3604.10 | 3600.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 3581.60 | 3604.10 | 3600.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 3574.60 | 3598.20 | 3597.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 3574.60 | 3598.20 | 3597.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 3573.00 | 3593.16 | 3595.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 3550.50 | 3584.63 | 3591.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3565.00 | 3563.00 | 3573.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 3565.00 | 3563.00 | 3573.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3580.90 | 3566.58 | 3574.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3585.40 | 3566.58 | 3574.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3581.10 | 3569.49 | 3574.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 3581.10 | 3569.49 | 3574.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3581.00 | 3573.82 | 3575.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3727.00 | 3573.82 | 3575.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3790.80 | 3617.22 | 3595.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 3813.00 | 3683.04 | 3630.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 3817.80 | 3821.07 | 3768.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 3822.40 | 3837.95 | 3822.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 3822.40 | 3837.95 | 3822.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 3830.20 | 3837.95 | 3822.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3818.30 | 3834.02 | 3822.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 3817.10 | 3834.02 | 3822.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3827.10 | 3836.35 | 3826.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 3827.10 | 3836.35 | 3826.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3817.20 | 3832.52 | 3825.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 3813.40 | 3832.52 | 3825.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3819.30 | 3829.87 | 3825.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 3819.00 | 3829.87 | 3825.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 3814.10 | 3822.30 | 3822.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 3791.30 | 3816.10 | 3819.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 3715.00 | 3708.69 | 3728.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 3715.00 | 3708.69 | 3728.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 3725.20 | 3711.99 | 3727.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 3707.00 | 3720.38 | 3727.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 3742.00 | 3728.02 | 3729.21 | SL hit (close>static) qty=1.00 sl=3740.90 alert=retest2 |

### Cycle 207 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 3748.30 | 3733.58 | 3731.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 3754.60 | 3740.72 | 3735.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 3737.60 | 3748.60 | 3743.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 3729.60 | 3744.80 | 3742.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 3729.60 | 3744.80 | 3742.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3728.50 | 3741.54 | 3740.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 3725.40 | 3741.54 | 3740.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 3725.40 | 3738.31 | 3739.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 3724.10 | 3732.64 | 3736.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 3721.00 | 3720.69 | 3726.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:00:00 | 3721.00 | 3720.69 | 3726.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 3732.10 | 3722.97 | 3726.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 3732.10 | 3722.97 | 3726.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 3733.30 | 3725.04 | 3727.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 3734.60 | 3725.04 | 3727.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 3739.30 | 3727.89 | 3728.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 3741.00 | 3727.89 | 3728.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 3732.00 | 3728.71 | 3728.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 3726.60 | 3728.71 | 3728.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 3711.40 | 3725.25 | 3727.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 3704.30 | 3721.14 | 3725.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3745.20 | 3725.30 | 3724.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 3745.20 | 3725.30 | 3724.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 3771.30 | 3742.36 | 3734.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 3759.50 | 3772.98 | 3755.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 3759.50 | 3772.98 | 3755.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3788.90 | 3776.17 | 3758.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:30:00 | 3792.20 | 3779.39 | 3761.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 3793.00 | 3782.44 | 3766.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 3789.80 | 3783.91 | 3768.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 3757.30 | 3768.78 | 3766.64 | SL hit (close<static) qty=1.00 sl=3758.00 alert=retest2 |

### Cycle 210 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 3753.00 | 3764.08 | 3764.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 3744.90 | 3760.24 | 3762.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 3758.30 | 3756.76 | 3760.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 3774.10 | 3760.23 | 3761.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 3774.10 | 3760.23 | 3761.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 3763.90 | 3760.96 | 3762.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 3771.00 | 3760.96 | 3762.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3769.30 | 3762.63 | 3762.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3769.30 | 3762.63 | 3762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 3764.00 | 3762.90 | 3762.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 3788.10 | 3768.12 | 3765.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 3800.00 | 3785.20 | 3778.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 3776.60 | 3781.73 | 3782.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 3776.60 | 3781.73 | 3782.10 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 3789.90 | 3783.08 | 3782.64 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 3775.40 | 3781.55 | 3781.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 3762.30 | 3777.19 | 3779.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 3745.10 | 3770.65 | 3775.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 3747.00 | 3760.55 | 3770.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 3740.50 | 3756.49 | 3762.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 3747.60 | 3752.15 | 3754.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 3752.00 | 3752.12 | 3754.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 3752.00 | 3752.12 | 3754.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 3761.30 | 3753.95 | 3754.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 3761.30 | 3753.95 | 3754.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 3798.90 | 3762.94 | 3758.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 3798.90 | 3762.94 | 3758.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 3810.10 | 3782.40 | 3769.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 3811.10 | 3815.17 | 3801.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 3805.70 | 3813.28 | 3802.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3805.70 | 3813.28 | 3802.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 3809.50 | 3813.28 | 3802.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 3806.10 | 3822.81 | 3813.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 3806.10 | 3822.81 | 3813.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 3812.40 | 3820.73 | 3813.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 3821.50 | 3816.90 | 3813.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:15:00 | 3835.00 | 3816.24 | 3813.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 3802.20 | 3813.96 | 3813.22 | SL hit (close<static) qty=1.00 sl=3804.30 alert=retest2 |

### Cycle 216 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3799.70 | 3811.10 | 3811.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 3779.90 | 3803.23 | 3808.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 3859.40 | 3802.85 | 3800.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 3871.70 | 3846.01 | 3832.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 3862.00 | 3871.76 | 3859.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 3862.00 | 3871.76 | 3859.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 3873.00 | 3872.01 | 3860.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 3873.40 | 3872.01 | 3860.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 3872.90 | 3872.19 | 3861.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 3881.30 | 3872.19 | 3861.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 3884.50 | 3878.35 | 3867.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 15:15:00 | 3955.00 | 3985.98 | 3990.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 15:15:00 | 3955.00 | 3985.98 | 3990.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 3923.40 | 3973.46 | 3984.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 3953.10 | 3945.25 | 3963.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 3953.10 | 3945.25 | 3963.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3940.00 | 3944.20 | 3961.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 3926.30 | 3944.20 | 3961.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 3933.20 | 3939.01 | 3955.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 3925.00 | 3931.09 | 3947.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 3933.10 | 3933.58 | 3947.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3932.50 | 3933.28 | 3944.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:15:00 | 3945.20 | 3933.28 | 3944.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 3953.60 | 3937.35 | 3945.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 3953.60 | 3937.35 | 3945.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 3958.30 | 3941.54 | 3946.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 3983.90 | 3941.54 | 3946.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 3966.60 | 3946.55 | 3948.49 | SL hit (close>static) qty=1.00 sl=3961.90 alert=retest2 |

### Cycle 219 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 3987.40 | 3954.72 | 3952.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 4003.30 | 3964.44 | 3956.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4038.30 | 4055.71 | 4028.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 4039.30 | 4055.71 | 4028.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4030.50 | 4050.66 | 4029.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:30:00 | 4043.70 | 4048.87 | 4030.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 4021.30 | 4043.36 | 4029.36 | SL hit (close<static) qty=1.00 sl=4023.80 alert=retest2 |

### Cycle 220 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 3975.10 | 4013.28 | 4017.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 3948.20 | 3994.46 | 4008.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 3994.60 | 3990.54 | 4003.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 3950.10 | 3974.94 | 3992.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 4020.50 | 3984.35 | 3995.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 4010.90 | 3989.66 | 3996.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 4010.90 | 3989.66 | 3996.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 4000.00 | 3992.10 | 3996.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 4004.50 | 3992.10 | 3996.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 4010.70 | 3995.82 | 3997.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 4014.10 | 3995.82 | 3997.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 4020.90 | 4000.84 | 3999.91 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 3952.70 | 3993.86 | 3997.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 3922.60 | 3979.60 | 3990.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 3973.30 | 3965.24 | 3977.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 13:00:00 | 3973.30 | 3965.24 | 3977.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 3966.80 | 3965.56 | 3976.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 3950.10 | 3969.54 | 3976.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 3752.59 | 3932.64 | 3944.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 3943.70 | 3934.85 | 3944.58 | SL hit (close>ema200) qty=0.50 sl=3934.85 alert=retest2 |

### Cycle 223 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 3988.10 | 3953.10 | 3951.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 4001.90 | 3962.54 | 3956.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 3972.60 | 3979.72 | 3968.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 3959.20 | 3979.72 | 3968.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3943.30 | 3972.44 | 3966.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 3940.60 | 3972.44 | 3966.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3913.90 | 3960.73 | 3961.44 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3984.90 | 3960.25 | 3960.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 3993.00 | 3966.80 | 3963.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 4006.80 | 4019.99 | 4001.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 4017.90 | 4019.57 | 4003.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 4025.00 | 4019.57 | 4003.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 4025.20 | 4018.78 | 4004.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 3969.90 | 4011.38 | 4006.12 | SL hit (close<static) qty=1.00 sl=3986.70 alert=retest2 |

### Cycle 226 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 3982.80 | 4001.64 | 4002.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 3977.10 | 3996.73 | 4000.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3974.00 | 3952.28 | 3966.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4011.00 | 3964.02 | 3970.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 4011.00 | 3964.02 | 3970.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 3995.00 | 3978.72 | 3976.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 4005.00 | 3986.76 | 3980.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 4053.60 | 4056.59 | 4033.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:45:00 | 4054.60 | 4056.59 | 4033.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 4038.10 | 4052.05 | 4038.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 4032.20 | 4052.05 | 4038.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 4050.40 | 4051.72 | 4039.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:45:00 | 4063.10 | 4053.47 | 4041.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 4065.00 | 4056.18 | 4044.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-26 09:15:00 | 4469.41 | 4421.14 | 4378.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 4356.60 | 4380.22 | 4382.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 4331.90 | 4370.55 | 4378.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 4378.90 | 4356.54 | 4365.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 4366.70 | 4358.57 | 4365.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 4395.80 | 4358.57 | 4365.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 4365.00 | 4359.86 | 4365.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 4311.30 | 4359.86 | 4365.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 4356.10 | 4339.91 | 4346.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 4363.00 | 4351.96 | 4350.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 4363.00 | 4351.96 | 4350.86 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 09:15:00 | 4331.00 | 4348.33 | 4349.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 4299.90 | 4338.64 | 4345.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 4334.70 | 4338.13 | 4343.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 4344.90 | 4320.12 | 4329.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 4344.90 | 4320.12 | 4329.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 4342.80 | 4324.65 | 4330.35 | EMA400 retest candle locked (from downside) |

### Cycle 231 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 4382.00 | 4336.12 | 4335.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 4424.50 | 4359.22 | 4346.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 4400.20 | 4422.64 | 4406.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4419.60 | 4422.03 | 4408.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 4440.00 | 4424.13 | 4411.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 4436.80 | 4432.70 | 4419.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 4387.80 | 4420.31 | 4415.66 | SL hit (close<static) qty=1.00 sl=4400.00 alert=retest2 |

### Cycle 232 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 4370.40 | 4406.90 | 4410.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 4282.00 | 4373.89 | 4392.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 4316.10 | 4295.69 | 4327.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 4316.10 | 4295.69 | 4327.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 4341.60 | 4304.87 | 4328.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 4341.60 | 4304.87 | 4328.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 4308.30 | 4305.56 | 4326.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 4302.90 | 4305.56 | 4326.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:30:00 | 4302.30 | 4301.20 | 4320.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 4299.30 | 4246.20 | 4239.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 4299.30 | 4246.20 | 4239.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 4316.30 | 4269.95 | 4252.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA400 retest candle locked (from upside) |

### Cycle 234 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4225.00 | 4264.68 | 4266.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4201.30 | 4252.01 | 4260.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 4243.40 | 4243.28 | 4253.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 4221.10 | 4235.18 | 4248.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 4327.20 | 4235.18 | 4248.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4262.80 | 4240.71 | 4249.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 4234.10 | 4240.71 | 4249.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 4022.39 | 4109.98 | 4175.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 3999.80 | 3992.69 | 4053.57 | SL hit (close>ema200) qty=0.50 sl=3992.69 alert=retest2 |

### Cycle 235 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 4090.20 | 4048.28 | 4042.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 4100.50 | 4058.72 | 4048.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 15:15:00 | 4158.00 | 4171.70 | 4140.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:15:00 | 4199.60 | 4171.70 | 4140.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 4160.00 | 4180.69 | 4161.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 4160.00 | 4180.69 | 4161.24 | SL hit (close<ema400) qty=1.00 sl=4161.24 alert=retest1 |

### Cycle 236 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 4117.50 | 4151.28 | 4152.27 | EMA200 below EMA400 |

### Cycle 237 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 4166.80 | 4154.11 | 4152.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 4172.50 | 4159.09 | 4155.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 11:15:00 | 4165.10 | 4169.55 | 4162.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 12:00:00 | 4165.10 | 4169.55 | 4162.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 4180.10 | 4176.16 | 4169.18 | EMA400 retest candle locked (from upside) |

### Cycle 238 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 4135.80 | 4161.94 | 4164.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 15:15:00 | 4121.10 | 4153.77 | 4160.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 4156.40 | 4114.33 | 4130.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 4156.20 | 4122.70 | 4132.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 4140.70 | 4124.80 | 4132.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 4146.20 | 4137.71 | 4137.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 4146.20 | 4137.71 | 4137.43 | EMA200 above EMA400 |

### Cycle 240 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 4115.10 | 4133.73 | 4135.70 | EMA200 below EMA400 |

### Cycle 241 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 4176.00 | 4133.15 | 4132.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 4210.00 | 4148.52 | 4139.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 4204.20 | 4214.41 | 4189.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:30:00 | 4199.20 | 4214.41 | 4189.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 4183.00 | 4207.84 | 4191.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 4183.00 | 4207.84 | 4191.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 4186.10 | 4203.49 | 4190.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 4246.40 | 4203.49 | 4190.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 4163.80 | 4221.08 | 4213.39 | SL hit (close<static) qty=1.00 sl=4175.10 alert=retest2 |

### Cycle 242 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 4154.70 | 4207.80 | 4208.05 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 4248.30 | 4204.48 | 4203.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 4262.90 | 4240.60 | 4225.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 4352.00 | 4360.17 | 4332.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 4372.70 | 4360.17 | 4332.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4386.80 | 4365.50 | 4337.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 4394.50 | 4365.50 | 4337.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 4395.00 | 4371.90 | 4351.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:15:00 | 2698.70 | 2024-05-22 10:15:00 | 2675.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-05-23 10:30:00 | 2658.65 | 2024-05-23 11:15:00 | 2688.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-27 10:30:00 | 2689.00 | 2024-05-29 11:15:00 | 2686.60 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-05-27 11:00:00 | 2686.20 | 2024-05-29 11:15:00 | 2686.60 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-05-30 12:45:00 | 2690.00 | 2024-05-30 14:15:00 | 2659.65 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-30 13:30:00 | 2688.75 | 2024-05-30 14:15:00 | 2659.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-06-14 09:15:00 | 2891.00 | 2024-06-20 09:15:00 | 2860.65 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-06-14 14:00:00 | 2886.00 | 2024-06-20 09:15:00 | 2860.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-06-18 09:30:00 | 2881.55 | 2024-06-20 09:15:00 | 2860.65 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-06-18 13:45:00 | 2881.65 | 2024-06-20 09:15:00 | 2860.65 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-06-21 12:30:00 | 2866.60 | 2024-07-02 12:15:00 | 2818.10 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2024-06-21 13:45:00 | 2860.70 | 2024-07-02 12:15:00 | 2818.10 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2024-06-24 12:45:00 | 2854.65 | 2024-07-02 12:15:00 | 2818.10 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2024-07-09 10:15:00 | 2903.50 | 2024-07-19 14:15:00 | 2949.90 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2024-07-09 12:30:00 | 2901.10 | 2024-07-19 14:15:00 | 2949.90 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-07-24 14:30:00 | 3179.55 | 2024-07-30 15:15:00 | 3117.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-07-29 09:15:00 | 3177.45 | 2024-07-30 15:15:00 | 3117.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-07-29 11:45:00 | 3171.70 | 2024-07-30 15:15:00 | 3117.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2024-08-02 12:30:00 | 3209.50 | 2024-08-05 10:15:00 | 3199.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-08-07 09:15:00 | 3261.15 | 2024-08-14 11:15:00 | 3294.25 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2024-08-26 09:30:00 | 3361.30 | 2024-08-26 12:15:00 | 3335.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-08-26 11:15:00 | 3361.90 | 2024-08-26 12:15:00 | 3335.40 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-09-06 09:15:00 | 3455.15 | 2024-09-06 09:15:00 | 3423.05 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-12 09:15:00 | 3488.95 | 2024-09-17 15:15:00 | 3448.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-09-13 12:30:00 | 3456.30 | 2024-09-17 15:15:00 | 3448.10 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-09-16 09:45:00 | 3460.40 | 2024-09-17 15:15:00 | 3448.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-09-16 14:45:00 | 3451.50 | 2024-09-17 15:15:00 | 3448.10 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-09-27 09:15:00 | 3408.00 | 2024-09-27 13:15:00 | 3440.05 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-27 12:00:00 | 3402.05 | 2024-09-27 13:15:00 | 3440.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-09-27 13:15:00 | 3409.10 | 2024-09-27 13:15:00 | 3440.05 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-10-03 13:45:00 | 3375.95 | 2024-10-04 09:15:00 | 3419.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-21 09:30:00 | 3421.40 | 2024-10-25 10:15:00 | 3377.80 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2024-10-21 11:00:00 | 3418.60 | 2024-10-25 10:15:00 | 3377.80 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-10-21 11:30:00 | 3422.85 | 2024-10-25 11:15:00 | 3414.10 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-10-21 12:00:00 | 3413.95 | 2024-10-25 11:15:00 | 3414.10 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-10-24 11:30:00 | 3318.15 | 2024-10-25 11:15:00 | 3414.10 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-10-24 14:45:00 | 3318.65 | 2024-10-25 11:15:00 | 3414.10 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-10-31 15:15:00 | 3195.00 | 2024-11-06 09:15:00 | 3225.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-11-04 09:15:00 | 3192.65 | 2024-11-06 09:15:00 | 3225.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-05 09:30:00 | 3195.50 | 2024-11-06 09:15:00 | 3225.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-11-12 13:00:00 | 3159.35 | 2024-11-22 10:15:00 | 3154.90 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-11-12 13:30:00 | 3162.80 | 2024-11-22 10:15:00 | 3154.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-11-12 14:00:00 | 3157.20 | 2024-11-22 10:15:00 | 3154.90 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-12-05 13:30:00 | 3350.45 | 2024-12-06 13:15:00 | 3335.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-12-05 14:00:00 | 3350.45 | 2024-12-06 13:15:00 | 3335.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-12-05 15:00:00 | 3364.95 | 2024-12-06 13:15:00 | 3335.35 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-12-06 10:30:00 | 3359.05 | 2024-12-06 13:15:00 | 3335.35 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-12-10 09:15:00 | 3346.95 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-10 12:45:00 | 3346.50 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-12-10 13:15:00 | 3347.85 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-12-10 14:15:00 | 3348.70 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-12-11 10:30:00 | 3378.10 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-11 13:00:00 | 3374.25 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-12 10:15:00 | 3370.95 | 2024-12-12 14:15:00 | 3327.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-12-16 10:15:00 | 3322.40 | 2024-12-16 13:15:00 | 3360.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-16 11:00:00 | 3324.70 | 2024-12-16 13:15:00 | 3360.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-17 15:00:00 | 3379.85 | 2024-12-23 13:15:00 | 3398.85 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-01-02 13:30:00 | 3404.85 | 2025-01-06 10:15:00 | 3370.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-01-03 10:15:00 | 3405.80 | 2025-01-06 10:15:00 | 3370.50 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-03 13:15:00 | 3406.45 | 2025-01-06 10:15:00 | 3370.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-10 12:00:00 | 3303.90 | 2025-01-20 09:15:00 | 3138.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:00:00 | 3303.90 | 2025-01-20 13:15:00 | 3182.00 | STOP_HIT | 0.50 | 3.69% |
| BUY | retest2 | 2025-01-27 09:15:00 | 3328.30 | 2025-01-31 10:15:00 | 3277.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-02-04 10:15:00 | 3103.65 | 2025-02-06 11:15:00 | 3185.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-02-05 12:30:00 | 3110.30 | 2025-02-06 11:15:00 | 3185.80 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-02-05 14:45:00 | 3114.75 | 2025-02-06 11:15:00 | 3185.80 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-02-10 14:45:00 | 3202.20 | 2025-02-11 09:15:00 | 3140.85 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-02-14 10:30:00 | 2974.55 | 2025-02-18 09:15:00 | 3055.10 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-02-20 14:15:00 | 3095.40 | 2025-02-21 09:15:00 | 3048.40 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-02-25 14:45:00 | 3034.95 | 2025-03-05 12:15:00 | 2975.45 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-04-17 14:45:00 | 3249.90 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-04-22 10:00:00 | 3243.00 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-04-22 10:45:00 | 3247.30 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-04-22 11:30:00 | 3245.00 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-04-23 10:15:00 | 3291.80 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-04-23 11:45:00 | 3289.10 | 2025-04-25 10:15:00 | 3194.50 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-05 11:15:00 | 3271.00 | 2025-05-07 12:15:00 | 3294.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-05-06 09:30:00 | 3264.70 | 2025-05-07 12:15:00 | 3294.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-05-06 13:15:00 | 3268.70 | 2025-05-07 12:15:00 | 3294.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-06 14:00:00 | 3260.30 | 2025-05-07 12:15:00 | 3294.10 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-05-07 09:45:00 | 3256.80 | 2025-05-07 12:15:00 | 3294.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-15 14:00:00 | 3261.60 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-15 15:00:00 | 3264.10 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-16 10:00:00 | 3265.00 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-16 15:00:00 | 3267.80 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-05-19 10:30:00 | 3304.00 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-19 12:15:00 | 3309.50 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-19 13:00:00 | 3304.40 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-05-19 13:45:00 | 3308.90 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-05-20 12:15:00 | 3272.70 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-20 12:45:00 | 3271.40 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-28 09:15:00 | 3155.90 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-05-28 14:15:00 | 3165.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-05-28 15:00:00 | 3163.00 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-05-29 13:15:00 | 3162.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-06-02 09:15:00 | 3129.20 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-03 09:45:00 | 3140.20 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-06-05 10:00:00 | 3147.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-06-11 14:30:00 | 3210.00 | 2025-06-18 11:15:00 | 3224.30 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-06-11 15:00:00 | 3209.90 | 2025-06-18 11:15:00 | 3224.30 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-06-26 14:15:00 | 3212.50 | 2025-07-03 09:15:00 | 3322.00 | STOP_HIT | 1.00 | 3.41% |
| BUY | retest2 | 2025-07-25 09:15:00 | 3573.40 | 2025-08-01 11:15:00 | 3649.10 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-08-08 09:15:00 | 3582.40 | 2025-08-08 12:15:00 | 3599.60 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-14 11:45:00 | 3642.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-18 14:15:00 | 3645.10 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-08-19 10:00:00 | 3642.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-19 13:15:00 | 3646.40 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-22 13:45:00 | 3668.40 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-25 11:45:00 | 3678.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-05 09:30:00 | 3614.80 | 2025-09-05 11:15:00 | 3577.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3521.70 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 3522.80 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-17 11:00:00 | 3519.20 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-29 11:30:00 | 3580.90 | 2025-09-29 14:15:00 | 3589.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-10 11:15:00 | 3547.50 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-10-10 15:15:00 | 3545.70 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-13 09:30:00 | 3542.40 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-10-30 11:30:00 | 3598.20 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-30 12:00:00 | 3598.20 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-30 14:30:00 | 3599.90 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-30 15:00:00 | 3604.00 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-25 09:15:00 | 3707.00 | 2025-11-25 11:15:00 | 3742.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-02 10:30:00 | 3704.30 | 2025-12-03 09:15:00 | 3745.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-05 11:30:00 | 3792.20 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-05 13:45:00 | 3793.00 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-05 15:00:00 | 3789.80 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-11 15:00:00 | 3800.00 | 2025-12-15 13:15:00 | 3776.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-16 15:15:00 | 3745.10 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-17 09:45:00 | 3747.00 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-18 09:15:00 | 3740.50 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-19 11:30:00 | 3747.60 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-26 13:45:00 | 3821.50 | 2025-12-29 10:15:00 | 3802.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-26 15:15:00 | 3835.00 | 2025-12-29 10:15:00 | 3802.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 09:15:00 | 3881.30 | 2026-01-09 15:15:00 | 3955.00 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2026-01-06 12:45:00 | 3884.50 | 2026-01-09 15:15:00 | 3955.00 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3926.30 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-13 11:15:00 | 3933.20 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-01-13 13:45:00 | 3925.00 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3933.10 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-20 11:30:00 | 4043.70 | 2026-01-20 12:15:00 | 4021.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-28 10:00:00 | 3950.10 | 2026-01-30 09:15:00 | 3752.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 10:00:00 | 3950.10 | 2026-01-30 10:15:00 | 3943.70 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest2 | 2026-01-30 11:00:00 | 3943.70 | 2026-01-30 13:15:00 | 3988.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-30 11:45:00 | 3956.60 | 2026-01-30 13:15:00 | 3988.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-04 12:15:00 | 4025.00 | 2026-02-05 09:15:00 | 3969.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-04 13:15:00 | 4025.20 | 2026-02-05 09:15:00 | 3969.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-12 11:45:00 | 4063.10 | 2026-02-26 09:15:00 | 4469.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 12:45:00 | 4065.00 | 2026-02-27 13:15:00 | 4356.60 | STOP_HIT | 1.00 | 7.17% |
| SELL | retest2 | 2026-03-04 09:15:00 | 4311.30 | 2026-03-05 12:15:00 | 4363.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-05 10:15:00 | 4356.10 | 2026-03-05 12:15:00 | 4363.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-03-12 13:00:00 | 4440.00 | 2026-03-13 10:15:00 | 4387.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-03-13 09:15:00 | 4436.80 | 2026-03-13 10:15:00 | 4387.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-03-17 15:15:00 | 4302.90 | 2026-03-25 09:15:00 | 4299.30 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-03-18 09:30:00 | 4302.30 | 2026-03-25 09:15:00 | 4299.30 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-04-01 10:15:00 | 4234.10 | 2026-04-02 09:15:00 | 4022.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 4234.10 | 2026-04-06 12:15:00 | 3999.80 | STOP_HIT | 0.50 | 5.53% |
| BUY | retest1 | 2026-04-15 09:15:00 | 4199.60 | 2026-04-15 14:15:00 | 4160.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-23 11:30:00 | 4140.70 | 2026-04-23 14:15:00 | 4146.20 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-29 09:15:00 | 4246.40 | 2026-04-30 09:15:00 | 4163.80 | STOP_HIT | 1.00 | -1.95% |
