# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2214.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 244 |
| ALERT1 | 149 |
| ALERT2 | 144 |
| ALERT2_SKIP | 103 |
| ALERT3 | 276 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 106 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 42 / 71
- **Target hits / Stop hits / Partials:** 4 / 99 / 10
- **Avg / median % per leg:** 0.28% / -0.93%
- **Sum % (uncompounded):** 31.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 13 | 30.2% | 3 | 40 | 0 | 0.31% | 13.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| BUY @ 3rd Alert (retest2) | 42 | 13 | 31.0% | 3 | 39 | 0 | 0.34% | 14.2% |
| SELL (all) | 70 | 29 | 41.4% | 1 | 59 | 10 | 0.26% | 18.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 70 | 29 | 41.4% | 1 | 59 | 10 | 0.26% | 18.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| retest2 (combined) | 112 | 42 | 37.5% | 4 | 98 | 10 | 0.29% | 32.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 1877.00 | 1865.31 | 1864.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 10:15:00 | 1890.00 | 1870.25 | 1866.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 15:15:00 | 1876.00 | 1879.03 | 1873.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 1865.50 | 1876.32 | 1872.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 1865.50 | 1876.32 | 1872.56 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 1850.70 | 1866.35 | 1868.37 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 1876.00 | 1867.48 | 1867.40 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 13:15:00 | 1865.00 | 1866.91 | 1867.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 15:15:00 | 1850.10 | 1863.24 | 1865.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 09:15:00 | 1873.05 | 1865.20 | 1866.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 1873.05 | 1865.20 | 1866.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 1873.05 | 1865.20 | 1866.11 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 1885.00 | 1869.16 | 1867.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 1909.60 | 1877.25 | 1871.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 1945.00 | 1955.61 | 1934.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 15:15:00 | 1942.00 | 1952.89 | 1935.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 1942.00 | 1952.89 | 1935.54 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 1926.15 | 1934.81 | 1935.22 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1969.35 | 1941.91 | 1938.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 1994.00 | 1973.23 | 1958.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 11:15:00 | 1974.15 | 1975.72 | 1962.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 1965.95 | 1972.62 | 1964.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 1965.95 | 1972.62 | 1964.41 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 1938.00 | 1963.86 | 1966.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 2012.30 | 1972.93 | 1970.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 15:15:00 | 2025.00 | 2005.61 | 1993.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 10:15:00 | 2004.10 | 2005.37 | 1995.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 10:15:00 | 2004.10 | 2005.37 | 1995.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 2004.10 | 2005.37 | 1995.70 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 1918.00 | 1981.30 | 1988.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 1900.60 | 1965.16 | 1980.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 1875.60 | 1861.80 | 1879.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 10:15:00 | 1882.70 | 1865.98 | 1879.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 1882.70 | 1865.98 | 1879.38 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 1899.75 | 1885.64 | 1884.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 10:15:00 | 1905.65 | 1889.64 | 1886.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 15:15:00 | 1898.00 | 1898.68 | 1893.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 15:15:00 | 1898.00 | 1898.68 | 1893.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 1898.00 | 1898.68 | 1893.12 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 12:15:00 | 1879.45 | 1889.49 | 1889.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 15:15:00 | 1875.00 | 1880.08 | 1883.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 1883.05 | 1880.68 | 1883.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 1883.05 | 1880.68 | 1883.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 1883.05 | 1880.68 | 1883.71 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 13:15:00 | 1883.55 | 1875.02 | 1874.63 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 1862.25 | 1875.98 | 1876.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1827.20 | 1854.33 | 1864.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 10:15:00 | 1830.55 | 1824.43 | 1834.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 11:15:00 | 1833.00 | 1826.15 | 1834.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 1833.00 | 1826.15 | 1834.39 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 1847.50 | 1838.42 | 1837.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 14:15:00 | 1849.15 | 1842.66 | 1839.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 11:15:00 | 1900.60 | 1901.01 | 1888.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 1868.75 | 1899.66 | 1893.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1868.75 | 1899.66 | 1893.47 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 12:15:00 | 1875.00 | 1887.68 | 1888.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 1868.30 | 1880.40 | 1883.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 13:15:00 | 1876.05 | 1875.97 | 1880.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 14:15:00 | 1878.20 | 1876.41 | 1879.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 1878.20 | 1876.41 | 1879.96 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 15:15:00 | 1885.90 | 1880.46 | 1879.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 1897.00 | 1883.77 | 1881.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 09:15:00 | 1894.60 | 1898.72 | 1891.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 10:15:00 | 1897.50 | 1898.48 | 1892.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 1897.50 | 1898.48 | 1892.27 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 2224.50 | 2262.66 | 2267.73 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 2282.95 | 2265.04 | 2264.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 2296.85 | 2277.02 | 2270.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 14:15:00 | 2294.45 | 2297.02 | 2286.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 2248.05 | 2287.70 | 2283.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 2248.05 | 2287.70 | 2283.75 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 2215.05 | 2273.17 | 2277.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 2185.75 | 2239.25 | 2259.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 2239.55 | 2224.32 | 2237.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 14:15:00 | 2239.55 | 2224.32 | 2237.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 2239.55 | 2224.32 | 2237.70 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 2263.40 | 2247.16 | 2245.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 2285.40 | 2261.18 | 2252.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 2308.00 | 2326.98 | 2308.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 2308.00 | 2326.98 | 2308.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 2308.00 | 2326.98 | 2308.15 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 2327.60 | 2335.28 | 2336.30 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 13:15:00 | 2350.35 | 2337.16 | 2336.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 14:15:00 | 2368.90 | 2343.51 | 2339.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 13:15:00 | 2353.00 | 2353.89 | 2347.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 14:15:00 | 2311.00 | 2345.31 | 2344.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 2311.00 | 2345.31 | 2344.24 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 15:15:00 | 2319.00 | 2340.05 | 2341.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 2276.75 | 2323.60 | 2331.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 2318.45 | 2299.12 | 2310.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 2318.45 | 2299.12 | 2310.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 2318.45 | 2299.12 | 2310.86 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 2358.00 | 2322.25 | 2319.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 2366.45 | 2347.12 | 2334.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 11:15:00 | 2344.00 | 2353.01 | 2343.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 11:15:00 | 2344.00 | 2353.01 | 2343.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 2344.00 | 2353.01 | 2343.27 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 10:15:00 | 2376.00 | 2396.45 | 2397.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 13:15:00 | 2351.55 | 2379.55 | 2388.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 2329.70 | 2326.72 | 2346.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 2329.70 | 2326.72 | 2346.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 2329.70 | 2326.72 | 2346.86 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 15:15:00 | 2366.00 | 2354.09 | 2353.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 2411.00 | 2365.47 | 2359.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 2426.70 | 2426.84 | 2410.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 10:15:00 | 2429.95 | 2427.46 | 2412.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 2429.95 | 2427.46 | 2412.43 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 13:15:00 | 2470.00 | 2479.57 | 2479.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 09:15:00 | 2463.60 | 2473.21 | 2476.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 13:15:00 | 2475.25 | 2466.76 | 2471.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 13:15:00 | 2475.25 | 2466.76 | 2471.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 2475.25 | 2466.76 | 2471.57 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 15:15:00 | 2476.00 | 2472.77 | 2472.52 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 2448.35 | 2467.88 | 2470.32 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 2468.00 | 2461.55 | 2461.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 2491.15 | 2467.47 | 2463.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 2475.60 | 2485.30 | 2476.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 2475.60 | 2485.30 | 2476.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 2475.60 | 2485.30 | 2476.98 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 2468.05 | 2480.46 | 2480.47 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 10:15:00 | 2480.60 | 2480.49 | 2480.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 14:15:00 | 2489.10 | 2482.92 | 2481.67 | Break + close above crossover candle high |

### Cycle 34 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 2458.10 | 2479.89 | 2480.63 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 2485.00 | 2481.17 | 2480.97 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 13:15:00 | 2476.00 | 2480.14 | 2480.52 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 14:15:00 | 2485.00 | 2481.11 | 2480.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 10:15:00 | 2491.85 | 2484.73 | 2482.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 13:15:00 | 2484.95 | 2488.56 | 2485.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 13:15:00 | 2484.95 | 2488.56 | 2485.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 2484.95 | 2488.56 | 2485.28 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 2449.55 | 2477.82 | 2481.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 2434.70 | 2464.62 | 2474.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 11:15:00 | 2457.40 | 2450.21 | 2461.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 12:15:00 | 2463.40 | 2452.85 | 2461.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 2463.40 | 2452.85 | 2461.27 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 13:15:00 | 2419.20 | 2402.48 | 2400.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 15:15:00 | 2433.20 | 2412.05 | 2405.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 2472.90 | 2478.41 | 2460.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 2459.30 | 2474.59 | 2460.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 2459.30 | 2474.59 | 2460.45 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 2463.35 | 2476.37 | 2476.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 2446.40 | 2467.72 | 2472.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 09:15:00 | 2295.95 | 2275.58 | 2304.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 10:15:00 | 2264.50 | 2273.36 | 2300.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 2264.50 | 2273.36 | 2300.64 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 2145.05 | 2129.82 | 2128.73 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 2126.50 | 2130.39 | 2130.63 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 2141.00 | 2131.17 | 2130.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 10:15:00 | 2159.50 | 2136.84 | 2133.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 2141.60 | 2143.33 | 2138.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 15:15:00 | 2148.55 | 2144.37 | 2139.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 2148.55 | 2144.37 | 2139.20 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 15:15:00 | 2183.90 | 2192.85 | 2193.20 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 13:15:00 | 2197.00 | 2193.72 | 2193.45 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 2189.00 | 2193.11 | 2193.27 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 14:15:00 | 2202.45 | 2193.14 | 2192.94 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 2178.50 | 2192.97 | 2193.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 11:15:00 | 2170.00 | 2188.38 | 2191.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 2260.60 | 2187.55 | 2187.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 2260.60 | 2187.55 | 2187.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 2260.60 | 2187.55 | 2187.68 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 2254.00 | 2200.84 | 2193.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 15:15:00 | 2271.50 | 2240.97 | 2218.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 12:15:00 | 2330.35 | 2333.01 | 2296.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 12:15:00 | 2333.20 | 2343.57 | 2331.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 2333.20 | 2343.57 | 2331.44 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 2332.95 | 2342.12 | 2342.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 2324.75 | 2338.65 | 2341.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 15:15:00 | 2310.00 | 2305.00 | 2316.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 2360.85 | 2316.17 | 2320.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 2360.85 | 2316.17 | 2320.64 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 2341.05 | 2325.90 | 2324.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 12:15:00 | 2353.45 | 2331.41 | 2327.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 11:15:00 | 2344.40 | 2344.76 | 2336.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 12:15:00 | 2345.00 | 2348.54 | 2343.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 2345.00 | 2348.54 | 2343.59 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 2336.40 | 2357.41 | 2358.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 2326.35 | 2351.19 | 2355.51 | Break + close below crossover candle low |

### Cycle 53 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 2391.00 | 2357.34 | 2357.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 2437.00 | 2397.77 | 2383.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 15:15:00 | 2425.00 | 2432.56 | 2418.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 2432.85 | 2443.33 | 2431.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 2432.85 | 2443.33 | 2431.94 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 2400.00 | 2425.54 | 2426.00 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 2569.00 | 2449.37 | 2435.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 2596.60 | 2500.16 | 2461.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 2636.60 | 2666.05 | 2630.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 2636.60 | 2666.05 | 2630.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 2636.60 | 2666.05 | 2630.10 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 2638.15 | 2649.04 | 2649.61 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 2666.70 | 2648.82 | 2648.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 2725.00 | 2671.94 | 2660.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 2716.35 | 2721.64 | 2701.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 2710.05 | 2719.32 | 2702.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 2710.05 | 2719.32 | 2702.42 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 2697.35 | 2730.21 | 2730.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 2686.50 | 2721.46 | 2726.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 14:15:00 | 2601.50 | 2596.65 | 2622.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 2673.00 | 2614.85 | 2626.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 2673.00 | 2614.85 | 2626.87 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 14:15:00 | 2643.65 | 2633.34 | 2632.86 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 2581.40 | 2622.90 | 2628.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 2566.50 | 2611.62 | 2622.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 2605.80 | 2591.25 | 2604.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 2605.80 | 2591.25 | 2604.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 2605.80 | 2591.25 | 2604.68 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 2710.00 | 2599.19 | 2585.11 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 2604.60 | 2628.45 | 2629.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 2597.25 | 2622.21 | 2626.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 10:15:00 | 2621.80 | 2613.96 | 2620.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 10:15:00 | 2621.80 | 2613.96 | 2620.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 2621.80 | 2613.96 | 2620.11 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 2601.35 | 2568.07 | 2567.03 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 2541.10 | 2564.42 | 2566.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 2531.10 | 2557.76 | 2563.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 2538.00 | 2537.04 | 2547.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 2550.00 | 2539.64 | 2547.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 2550.00 | 2539.64 | 2547.67 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 2581.90 | 2543.46 | 2541.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 2584.15 | 2551.59 | 2544.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 2587.45 | 2590.35 | 2575.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 2587.45 | 2590.35 | 2575.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 2587.45 | 2590.35 | 2575.76 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 10:15:00 | 2551.10 | 2582.50 | 2585.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 2528.65 | 2559.68 | 2571.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 2553.00 | 2526.77 | 2544.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 2553.00 | 2526.77 | 2544.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 2553.00 | 2526.77 | 2544.39 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 2577.35 | 2556.46 | 2553.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 13:15:00 | 2585.25 | 2567.78 | 2561.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 13:15:00 | 2606.30 | 2609.29 | 2589.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 15:15:00 | 2604.25 | 2605.81 | 2591.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 2604.25 | 2605.81 | 2591.09 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 2529.85 | 2587.37 | 2594.63 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 2617.00 | 2586.05 | 2585.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 2640.00 | 2596.84 | 2590.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 2780.20 | 2783.16 | 2734.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 13:15:00 | 2753.95 | 2771.18 | 2749.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 2753.95 | 2771.18 | 2749.49 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 2660.90 | 2731.24 | 2740.42 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 2745.80 | 2724.59 | 2722.09 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 2659.00 | 2714.51 | 2719.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 2634.00 | 2662.47 | 2675.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 12:15:00 | 2626.55 | 2626.13 | 2641.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 2638.80 | 2625.81 | 2636.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 2638.80 | 2625.81 | 2636.42 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 2487.55 | 2433.00 | 2432.72 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 2441.20 | 2461.07 | 2461.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 12:15:00 | 2417.75 | 2448.26 | 2455.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 2450.70 | 2402.72 | 2415.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 2450.70 | 2402.72 | 2415.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 2450.70 | 2402.72 | 2415.51 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 2435.45 | 2422.56 | 2421.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 15:15:00 | 2446.20 | 2427.29 | 2424.03 | Break + close above crossover candle high |

### Cycle 76 — SELL (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 09:15:00 | 2380.10 | 2417.85 | 2420.03 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 10:15:00 | 2445.25 | 2400.76 | 2399.25 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 2390.00 | 2401.58 | 2403.05 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 2428.55 | 2405.44 | 2404.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 2444.55 | 2415.90 | 2409.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 2480.85 | 2488.81 | 2468.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 12:15:00 | 2495.85 | 2502.45 | 2491.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 2495.85 | 2502.45 | 2491.82 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 11:15:00 | 2457.70 | 2484.75 | 2487.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 2440.00 | 2471.05 | 2480.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 2498.00 | 2472.55 | 2478.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 2498.00 | 2472.55 | 2478.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 2498.00 | 2472.55 | 2478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:30:00 | 2433.80 | 2449.60 | 2455.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 13:15:00 | 2312.11 | 2369.91 | 2402.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 2353.80 | 2349.46 | 2383.55 | SL hit (close>ema200) qty=0.50 sl=2349.46 alert=retest2 |

### Cycle 81 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 2356.00 | 2257.96 | 2254.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 10:15:00 | 2385.00 | 2324.15 | 2296.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 12:15:00 | 2347.75 | 2348.32 | 2327.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 13:00:00 | 2347.75 | 2348.32 | 2327.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 2314.65 | 2340.28 | 2327.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 2314.65 | 2340.28 | 2327.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 2322.00 | 2336.62 | 2327.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 2309.30 | 2336.62 | 2327.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 2318.35 | 2332.97 | 2326.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 10:15:00 | 2330.00 | 2332.97 | 2326.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:45:00 | 2329.25 | 2328.31 | 2325.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 2330.45 | 2325.88 | 2325.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 2298.35 | 2321.14 | 2323.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 2298.35 | 2321.14 | 2323.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 15:15:00 | 2292.00 | 2305.31 | 2314.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 2322.25 | 2308.70 | 2314.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 2322.25 | 2308.70 | 2314.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 2322.25 | 2308.70 | 2314.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:45:00 | 2338.75 | 2308.70 | 2314.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 2307.00 | 2308.36 | 2314.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 15:00:00 | 2296.40 | 2307.28 | 2312.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 2277.80 | 2248.09 | 2247.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2277.80 | 2248.09 | 2247.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 2285.90 | 2255.66 | 2250.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 2284.50 | 2286.64 | 2274.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 2284.50 | 2286.64 | 2274.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 2280.25 | 2285.37 | 2274.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:00:00 | 2280.25 | 2285.37 | 2274.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 2275.50 | 2283.39 | 2274.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 2275.50 | 2283.39 | 2274.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 2275.80 | 2281.87 | 2274.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 2343.50 | 2281.87 | 2274.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 2408.30 | 2412.93 | 2413.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 2408.30 | 2412.93 | 2413.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 2367.10 | 2397.55 | 2405.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 15:15:00 | 2293.75 | 2293.51 | 2311.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 2271.65 | 2293.51 | 2311.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 2282.55 | 2277.20 | 2297.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 2282.55 | 2277.20 | 2297.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2341.75 | 2291.99 | 2297.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 2341.75 | 2291.99 | 2297.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 2406.85 | 2314.96 | 2307.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 2408.95 | 2333.76 | 2316.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 14:15:00 | 2350.65 | 2352.58 | 2331.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 2350.65 | 2352.58 | 2331.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 2360.60 | 2454.46 | 2433.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 2360.60 | 2454.46 | 2433.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 2388.95 | 2441.36 | 2429.87 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 12:15:00 | 2392.50 | 2422.50 | 2422.73 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 2439.25 | 2409.67 | 2408.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 10:15:00 | 2444.25 | 2416.59 | 2411.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 09:15:00 | 2414.95 | 2431.90 | 2423.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 2414.95 | 2431.90 | 2423.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 2414.95 | 2431.90 | 2423.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 2418.45 | 2431.90 | 2423.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 2410.45 | 2427.61 | 2422.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 2408.55 | 2427.61 | 2422.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 2406.00 | 2417.45 | 2418.44 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 2427.75 | 2418.25 | 2418.02 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 2398.15 | 2414.33 | 2416.44 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 2453.25 | 2419.22 | 2415.15 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 2405.45 | 2420.10 | 2420.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 2393.30 | 2410.07 | 2415.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 2400.60 | 2397.18 | 2405.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:45:00 | 2395.00 | 2397.18 | 2405.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 2411.30 | 2400.50 | 2405.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 2411.30 | 2400.50 | 2405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 2410.00 | 2402.40 | 2406.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 2419.00 | 2402.40 | 2406.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 2434.20 | 2411.30 | 2409.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 2438.75 | 2423.35 | 2417.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 14:15:00 | 2492.65 | 2497.20 | 2471.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 2492.65 | 2497.20 | 2471.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2592.60 | 2607.20 | 2592.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 2594.65 | 2607.20 | 2592.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 2605.15 | 2606.79 | 2593.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:30:00 | 2601.70 | 2606.79 | 2593.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 2591.00 | 2605.78 | 2598.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 2591.00 | 2605.78 | 2598.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 2592.00 | 2603.02 | 2598.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 2592.00 | 2603.02 | 2598.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 2600.35 | 2602.49 | 2598.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:30:00 | 2593.05 | 2602.49 | 2598.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 2606.00 | 2602.74 | 2599.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 2541.00 | 2602.74 | 2599.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2520.90 | 2586.37 | 2592.00 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 2659.40 | 2577.76 | 2572.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 2735.60 | 2628.53 | 2597.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 15:15:00 | 2837.00 | 2858.08 | 2821.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:15:00 | 2875.00 | 2858.08 | 2821.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2878.75 | 2862.22 | 2826.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 2890.00 | 2862.22 | 2826.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 2910.00 | 2908.62 | 2879.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 2846.40 | 2879.76 | 2879.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 2846.40 | 2879.76 | 2879.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 2817.65 | 2864.28 | 2872.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 15:15:00 | 2847.90 | 2847.86 | 2858.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 2932.15 | 2847.86 | 2858.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 97 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 2992.30 | 2876.75 | 2870.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 3048.75 | 2911.15 | 2887.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 2942.25 | 2983.32 | 2943.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 2942.25 | 2983.32 | 2943.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 2942.25 | 2983.32 | 2943.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:00:00 | 2942.25 | 2983.32 | 2943.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 2904.55 | 2967.56 | 2940.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:00:00 | 2904.55 | 2967.56 | 2940.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 2903.10 | 2954.67 | 2936.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 2903.10 | 2954.67 | 2936.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 2931.30 | 2938.71 | 2933.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 2910.75 | 2938.71 | 2933.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2903.45 | 2931.66 | 2930.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 2903.45 | 2931.66 | 2930.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 10:15:00 | 2916.60 | 2928.65 | 2929.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 12:15:00 | 2888.00 | 2911.59 | 2918.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 2932.25 | 2909.37 | 2914.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 2932.25 | 2909.37 | 2914.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 2932.25 | 2909.37 | 2914.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 2932.25 | 2909.37 | 2914.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2922.60 | 2912.01 | 2915.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:30:00 | 2927.85 | 2912.01 | 2915.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 2891.80 | 2907.97 | 2913.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:15:00 | 2888.05 | 2907.97 | 2913.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 2743.65 | 2781.51 | 2831.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-05 11:15:00 | 2599.25 | 2723.34 | 2794.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 2718.55 | 2703.17 | 2701.89 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 2676.65 | 2698.70 | 2701.53 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 2725.20 | 2706.59 | 2704.24 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 2696.35 | 2703.90 | 2703.91 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 11:15:00 | 2724.25 | 2704.32 | 2703.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 12:15:00 | 2727.65 | 2708.99 | 2705.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 2996.00 | 2999.41 | 2960.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:00:00 | 2996.00 | 2999.41 | 2960.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2993.00 | 3051.48 | 3027.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 2986.20 | 3051.48 | 3027.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 3008.20 | 3042.82 | 3025.85 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 2997.20 | 3015.52 | 3016.58 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 3075.10 | 3024.15 | 3020.12 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 3013.55 | 3022.51 | 3022.80 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 13:15:00 | 3035.70 | 3025.08 | 3023.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 3059.00 | 3035.86 | 3029.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 15:15:00 | 3075.10 | 3079.25 | 3058.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:15:00 | 3070.80 | 3079.25 | 3058.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 3088.10 | 3081.02 | 3061.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 3093.65 | 3081.02 | 3061.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 11:00:00 | 3098.95 | 3084.61 | 3065.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 13:45:00 | 3096.30 | 3087.35 | 3071.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 3097.60 | 3089.70 | 3073.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 3090.00 | 3097.56 | 3085.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:30:00 | 3087.50 | 3097.56 | 3085.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 3102.95 | 3098.64 | 3087.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 3088.15 | 3098.64 | 3087.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 3105.00 | 3112.66 | 3101.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 3100.15 | 3112.66 | 3101.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 3107.40 | 3111.61 | 3102.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 3085.55 | 3111.61 | 3102.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 3099.00 | 3109.09 | 3101.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 3096.30 | 3109.09 | 3101.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 3105.45 | 3108.36 | 3102.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 3113.90 | 3118.69 | 3107.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 3040.35 | 3111.08 | 3109.67 | SL hit (close<static) qty=1.00 sl=3052.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 3017.60 | 3092.38 | 3101.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 2996.65 | 3033.67 | 3054.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 3022.70 | 3016.87 | 3038.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 3022.70 | 3016.87 | 3038.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 3022.70 | 3016.87 | 3038.31 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 3111.60 | 3052.42 | 3049.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 3130.00 | 3097.23 | 3084.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 3122.75 | 3140.94 | 3126.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 14:15:00 | 3122.75 | 3140.94 | 3126.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 3122.75 | 3140.94 | 3126.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 3122.75 | 3140.94 | 3126.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 3130.00 | 3138.75 | 3127.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 3120.05 | 3138.75 | 3127.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 3138.00 | 3138.60 | 3128.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:15:00 | 3161.35 | 3140.63 | 3130.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 3003.90 | 3130.87 | 3133.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 3003.90 | 3130.87 | 3133.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2980.40 | 3052.08 | 3090.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 3002.00 | 2986.67 | 3024.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 2981.00 | 2986.67 | 3024.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 3004.30 | 2990.20 | 3022.47 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 3039.25 | 3028.91 | 3028.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 3044.00 | 3031.93 | 3029.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 3031.50 | 3057.49 | 3050.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 3031.50 | 3057.49 | 3050.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 3031.50 | 3057.49 | 3050.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 3031.50 | 3057.49 | 3050.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 3011.80 | 3048.35 | 3046.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 3011.80 | 3048.35 | 3046.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 3011.00 | 3040.88 | 3043.56 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 3055.25 | 3044.12 | 3042.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 3088.00 | 3053.84 | 3047.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 3062.35 | 3064.11 | 3054.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 12:15:00 | 3062.35 | 3064.11 | 3054.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 3062.35 | 3064.11 | 3054.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:45:00 | 3058.00 | 3064.11 | 3054.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 3053.50 | 3061.98 | 3054.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 3053.50 | 3061.98 | 3054.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 3084.05 | 3066.40 | 3057.20 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 3014.45 | 3047.98 | 3051.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 3009.75 | 3034.54 | 3044.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 3036.05 | 3031.24 | 3041.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 3036.05 | 3031.24 | 3041.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3036.05 | 3031.24 | 3041.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 3015.15 | 3026.59 | 3038.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:45:00 | 3016.50 | 3024.71 | 3036.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 15:00:00 | 3012.20 | 3027.57 | 3035.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 2996.60 | 3026.04 | 3033.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 2864.39 | 2919.80 | 2956.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 2865.67 | 2919.80 | 2956.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 2861.59 | 2919.80 | 2956.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 2846.77 | 2919.80 | 2956.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 2910.95 | 2898.57 | 2935.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 2910.95 | 2898.57 | 2935.86 | SL hit (close>ema200) qty=0.50 sl=2898.57 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 2917.35 | 2908.91 | 2908.90 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 2898.60 | 2906.85 | 2907.96 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 2921.00 | 2908.86 | 2908.63 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 2891.00 | 2905.49 | 2907.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 12:15:00 | 2885.65 | 2901.52 | 2905.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 2889.00 | 2883.45 | 2894.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 2889.00 | 2883.45 | 2894.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2889.00 | 2883.45 | 2894.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 2886.95 | 2883.45 | 2894.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 2878.00 | 2877.93 | 2886.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 2915.40 | 2877.93 | 2886.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 2901.75 | 2882.69 | 2887.69 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 2928.50 | 2891.85 | 2891.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 2950.75 | 2903.63 | 2896.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 11:15:00 | 2915.85 | 2948.12 | 2928.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 11:15:00 | 2915.85 | 2948.12 | 2928.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 2915.85 | 2948.12 | 2928.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 2915.85 | 2948.12 | 2928.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 2909.60 | 2940.42 | 2926.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 2909.60 | 2940.42 | 2926.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 2931.50 | 2938.63 | 2926.82 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 2891.10 | 2919.43 | 2921.53 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 09:15:00 | 2970.55 | 2928.97 | 2924.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-17 11:15:00 | 3053.50 | 2971.18 | 2945.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 3034.05 | 3037.82 | 2994.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:00:00 | 3034.05 | 3037.82 | 2994.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 3027.55 | 3061.65 | 3038.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 3027.55 | 3061.65 | 3038.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 2992.70 | 3047.86 | 3034.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 2992.70 | 3047.86 | 3034.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2986.25 | 3035.54 | 3030.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 2986.25 | 3035.54 | 3030.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 2979.00 | 3024.23 | 3025.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 2967.15 | 3007.00 | 3016.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 3072.45 | 3015.33 | 3018.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 3072.45 | 3015.33 | 3018.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 3072.45 | 3015.33 | 3018.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 3072.45 | 3015.33 | 3018.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 3128.05 | 3037.88 | 3028.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 12:15:00 | 3141.40 | 3072.09 | 3046.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 09:15:00 | 3089.00 | 3095.37 | 3067.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 3089.00 | 3095.37 | 3067.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 3089.00 | 3095.37 | 3067.69 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 3023.65 | 3068.79 | 3072.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 10:15:00 | 2996.15 | 3039.84 | 3056.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 15:15:00 | 3028.50 | 3025.63 | 3041.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 09:15:00 | 3035.00 | 3025.63 | 3041.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 3001.20 | 3020.74 | 3038.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:15:00 | 2984.00 | 3020.74 | 3038.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 3048.10 | 3016.29 | 3024.00 | SL hit (close>static) qty=1.00 sl=3040.95 alert=retest2 |

### Cycle 125 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 3040.45 | 3028.66 | 3028.39 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 3003.70 | 3023.67 | 3026.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 2904.75 | 2994.71 | 3011.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 2910.45 | 2866.27 | 2883.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 2910.45 | 2866.27 | 2883.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 2910.45 | 2866.27 | 2883.80 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 2897.30 | 2891.82 | 2891.66 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2826.75 | 2879.79 | 2886.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2816.05 | 2867.04 | 2879.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 2852.50 | 2850.77 | 2864.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 2852.50 | 2850.77 | 2864.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2852.50 | 2850.77 | 2864.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 2869.65 | 2850.77 | 2864.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 2859.80 | 2852.58 | 2863.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 2865.00 | 2852.58 | 2863.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 2859.45 | 2853.92 | 2862.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 2859.45 | 2853.92 | 2862.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 2857.55 | 2854.64 | 2862.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:45:00 | 2857.65 | 2854.64 | 2862.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 2845.15 | 2852.97 | 2860.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 2842.00 | 2852.97 | 2860.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 2872.50 | 2856.88 | 2861.22 | SL hit (close>static) qty=1.00 sl=2864.50 alert=retest2 |

### Cycle 129 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 2882.15 | 2865.63 | 2864.68 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 2860.35 | 2863.94 | 2864.14 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 15:15:00 | 2868.05 | 2864.76 | 2864.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 2887.95 | 2869.40 | 2866.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 2866.50 | 2872.48 | 2869.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 2866.50 | 2872.48 | 2869.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2866.50 | 2872.48 | 2869.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:00:00 | 2866.50 | 2872.48 | 2869.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 2865.75 | 2871.14 | 2868.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 2856.40 | 2871.14 | 2868.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 2851.90 | 2867.29 | 2867.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 2851.90 | 2867.29 | 2867.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 2850.00 | 2863.83 | 2865.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 2829.70 | 2857.00 | 2862.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 2857.35 | 2854.47 | 2859.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 12:15:00 | 2857.35 | 2854.47 | 2859.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 2857.35 | 2854.47 | 2859.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:30:00 | 2856.70 | 2854.47 | 2859.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 2868.30 | 2857.24 | 2860.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 14:00:00 | 2868.30 | 2857.24 | 2860.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 2843.60 | 2854.51 | 2858.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 2766.10 | 2846.56 | 2851.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:30:00 | 2829.40 | 2813.56 | 2818.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 2818.45 | 2814.54 | 2818.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 2878.45 | 2802.69 | 2803.87 | SL hit (close>static) qty=1.00 sl=2868.75 alert=retest2 |

### Cycle 133 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 2850.00 | 2812.15 | 2808.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 2885.10 | 2845.11 | 2826.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 14:15:00 | 3013.80 | 3014.76 | 2978.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:00:00 | 3013.80 | 3014.76 | 2978.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 2977.10 | 3008.87 | 2982.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 2977.10 | 3008.87 | 2982.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 2962.45 | 2999.58 | 2980.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 2962.45 | 2999.58 | 2980.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 2954.70 | 2990.61 | 2978.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:45:00 | 2954.85 | 2990.61 | 2978.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 2963.00 | 2970.56 | 2971.34 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 2975.00 | 2972.17 | 2971.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 3000.00 | 2979.62 | 2976.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 3006.45 | 3026.73 | 3010.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 12:15:00 | 3006.45 | 3026.73 | 3010.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 3006.45 | 3026.73 | 3010.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:45:00 | 3001.45 | 3026.73 | 3010.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 3012.25 | 3023.83 | 3010.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 3044.65 | 3021.99 | 3012.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 3027.25 | 3027.28 | 3017.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 3053.50 | 3021.95 | 3017.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 3044.05 | 3032.11 | 3024.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 3053.80 | 3042.74 | 3031.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 3035.85 | 3042.74 | 3031.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 3168.35 | 3187.30 | 3171.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:15:00 | 3147.80 | 3187.30 | 3171.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 3167.75 | 3183.39 | 3171.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 3150.00 | 3183.39 | 3171.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 3163.90 | 3179.49 | 3170.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 3163.90 | 3179.49 | 3170.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 3182.35 | 3180.06 | 3171.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:15:00 | 3188.45 | 3180.06 | 3171.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:15:00 | 3189.40 | 3180.47 | 3172.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 3205.00 | 3180.57 | 3173.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:30:00 | 3190.55 | 3191.72 | 3182.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 3192.65 | 3191.90 | 3183.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 3184.95 | 3191.90 | 3183.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 3216.40 | 3197.93 | 3188.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 3158.75 | 3180.07 | 3182.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 3158.75 | 3180.07 | 3182.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 3092.05 | 3161.24 | 3171.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 2999.95 | 2985.50 | 3022.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 2999.95 | 2985.50 | 3022.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2906.35 | 2915.06 | 2943.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:30:00 | 2897.30 | 2912.85 | 2939.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 2895.35 | 2912.85 | 2939.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 13:00:00 | 2895.75 | 2905.27 | 2931.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:15:00 | 2894.60 | 2904.23 | 2928.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 2903.00 | 2897.93 | 2915.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:30:00 | 2907.70 | 2897.93 | 2915.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2944.60 | 2905.09 | 2914.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 2944.60 | 2905.09 | 2914.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 2909.50 | 2905.97 | 2913.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2876.00 | 2905.97 | 2913.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 11:30:00 | 2890.40 | 2866.85 | 2869.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 2886.80 | 2872.97 | 2872.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 2886.80 | 2872.97 | 2872.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 2898.95 | 2878.16 | 2874.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 2859.70 | 2876.36 | 2874.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 2859.70 | 2876.36 | 2874.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 2859.70 | 2876.36 | 2874.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 2859.70 | 2876.36 | 2874.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 2839.00 | 2868.89 | 2871.30 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 09:15:00 | 2906.70 | 2876.72 | 2873.78 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 2872.85 | 2897.53 | 2900.47 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 2921.90 | 2904.54 | 2902.49 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 2859.05 | 2902.57 | 2903.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 2795.00 | 2850.05 | 2875.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 2791.90 | 2787.21 | 2815.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 2799.70 | 2789.71 | 2814.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 2799.70 | 2789.71 | 2814.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 2804.95 | 2789.71 | 2814.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 2794.15 | 2794.16 | 2812.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 2778.75 | 2794.16 | 2812.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 2830.50 | 2796.11 | 2806.71 | SL hit (close>static) qty=1.00 sl=2812.70 alert=retest2 |

### Cycle 143 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 2834.65 | 2816.65 | 2814.63 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 2811.95 | 2814.62 | 2814.94 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 2821.25 | 2815.95 | 2815.52 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 2755.40 | 2804.58 | 2810.50 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 2832.90 | 2809.51 | 2809.26 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 2785.35 | 2805.94 | 2807.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 2775.00 | 2799.75 | 2804.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2813.95 | 2789.86 | 2795.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 2813.95 | 2789.86 | 2795.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 2813.95 | 2789.86 | 2795.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 2813.95 | 2789.86 | 2795.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 2803.00 | 2792.49 | 2796.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 2841.45 | 2792.49 | 2796.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 2903.00 | 2814.59 | 2805.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 2916.85 | 2835.04 | 2816.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2790.90 | 2873.58 | 2851.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2790.90 | 2873.58 | 2851.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2790.90 | 2873.58 | 2851.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2790.90 | 2873.58 | 2851.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2850.35 | 2868.94 | 2851.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 2910.00 | 2868.94 | 2851.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 2883.70 | 2931.40 | 2932.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 2883.70 | 2931.40 | 2932.53 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 2948.50 | 2935.08 | 2933.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 2968.95 | 2941.85 | 2936.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 2932.85 | 2965.45 | 2954.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 10:15:00 | 2932.85 | 2965.45 | 2954.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 2932.85 | 2965.45 | 2954.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 2932.85 | 2965.45 | 2954.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 2887.20 | 2949.80 | 2948.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 2887.20 | 2949.80 | 2948.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 2886.85 | 2937.21 | 2942.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 10:15:00 | 2863.80 | 2891.32 | 2914.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 2825.70 | 2820.88 | 2850.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 11:00:00 | 2825.70 | 2820.88 | 2850.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2838.90 | 2820.91 | 2836.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 2812.75 | 2820.31 | 2834.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:45:00 | 2814.15 | 2804.36 | 2817.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:30:00 | 2817.20 | 2814.50 | 2820.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 2840.30 | 2826.54 | 2825.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 2840.30 | 2826.54 | 2825.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 10:15:00 | 2874.85 | 2842.47 | 2835.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 2839.30 | 2855.57 | 2847.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 2839.30 | 2855.57 | 2847.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2839.30 | 2855.57 | 2847.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 2839.30 | 2855.57 | 2847.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2828.65 | 2850.19 | 2845.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 2826.85 | 2850.19 | 2845.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 2800.10 | 2840.17 | 2841.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 2790.30 | 2830.20 | 2836.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 2561.45 | 2554.49 | 2594.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 2561.45 | 2554.49 | 2594.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2591.90 | 2565.40 | 2592.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 2591.90 | 2565.40 | 2592.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 2591.70 | 2570.66 | 2592.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 2588.05 | 2570.66 | 2592.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2590.25 | 2574.58 | 2592.10 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 2618.05 | 2593.60 | 2590.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 2620.50 | 2605.12 | 2597.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2585.70 | 2615.43 | 2607.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2585.70 | 2615.43 | 2607.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2585.70 | 2615.43 | 2607.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2585.70 | 2615.43 | 2607.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2583.15 | 2608.97 | 2605.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 2583.15 | 2608.97 | 2605.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 2588.65 | 2602.35 | 2602.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 2567.05 | 2595.29 | 2599.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 2267.50 | 2265.22 | 2308.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:30:00 | 2272.65 | 2265.22 | 2308.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2299.80 | 2258.08 | 2277.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 2326.25 | 2258.08 | 2277.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 2301.35 | 2266.74 | 2279.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 2301.35 | 2266.74 | 2279.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 2307.95 | 2286.09 | 2286.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 2361.45 | 2307.36 | 2296.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 2306.30 | 2330.72 | 2318.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 2306.30 | 2330.72 | 2318.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2306.30 | 2330.72 | 2318.00 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 2286.80 | 2311.37 | 2311.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 2261.00 | 2290.54 | 2300.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 2277.95 | 2260.93 | 2275.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 2277.95 | 2260.93 | 2275.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2277.95 | 2260.93 | 2275.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 2277.95 | 2260.93 | 2275.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2270.25 | 2262.79 | 2275.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 2243.00 | 2262.79 | 2275.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 2254.50 | 2235.12 | 2245.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:45:00 | 2246.55 | 2218.24 | 2222.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 2273.50 | 2229.30 | 2226.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 2273.50 | 2229.30 | 2226.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 2305.65 | 2244.57 | 2233.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 11:15:00 | 2279.20 | 2279.49 | 2261.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 11:45:00 | 2281.20 | 2279.49 | 2261.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2370.50 | 2318.97 | 2298.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 2391.40 | 2318.97 | 2298.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 2453.75 | 2496.47 | 2502.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 2453.75 | 2496.47 | 2502.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 2396.05 | 2456.62 | 2469.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2140.00 | 2131.50 | 2210.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 12:15:00 | 2209.70 | 2159.86 | 2204.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 2209.70 | 2159.86 | 2204.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 2215.00 | 2159.86 | 2204.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 2179.85 | 2163.85 | 2202.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 2065.05 | 2165.19 | 2196.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:45:00 | 2157.55 | 2158.42 | 2162.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 15:00:00 | 2160.45 | 2158.82 | 2162.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 2238.30 | 2175.80 | 2169.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 2238.30 | 2175.80 | 2169.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 2248.40 | 2214.03 | 2191.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2227.90 | 2253.55 | 2234.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 2227.90 | 2253.55 | 2234.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2227.90 | 2253.55 | 2234.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 2257.60 | 2254.36 | 2236.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 2260.90 | 2254.15 | 2237.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 2483.36 | 2412.43 | 2369.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 2468.30 | 2497.59 | 2498.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 2455.80 | 2489.23 | 2494.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 2518.00 | 2494.98 | 2496.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 2518.00 | 2494.98 | 2496.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2518.00 | 2494.98 | 2496.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 2518.00 | 2494.98 | 2496.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2495.80 | 2495.15 | 2496.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 2473.00 | 2487.94 | 2493.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:15:00 | 2471.20 | 2476.00 | 2483.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 2349.35 | 2384.76 | 2419.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 2347.64 | 2384.76 | 2419.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 2386.90 | 2385.19 | 2416.89 | SL hit (close>ema200) qty=0.50 sl=2385.19 alert=retest2 |

### Cycle 163 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2494.00 | 2406.55 | 2401.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 2543.10 | 2460.24 | 2429.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 2496.70 | 2503.03 | 2475.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 2496.70 | 2503.03 | 2475.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2521.40 | 2503.55 | 2481.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 2538.70 | 2508.48 | 2486.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 2537.20 | 2514.22 | 2490.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 2545.00 | 2520.38 | 2495.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 2539.90 | 2568.99 | 2569.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2539.90 | 2568.99 | 2569.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 2535.70 | 2562.34 | 2566.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 2578.80 | 2559.07 | 2563.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2585.10 | 2564.28 | 2565.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 2585.10 | 2564.28 | 2565.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2574.50 | 2566.32 | 2566.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 12:45:00 | 2553.10 | 2564.54 | 2565.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:30:00 | 2561.40 | 2548.14 | 2555.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 2583.50 | 2549.82 | 2549.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 2583.50 | 2549.82 | 2549.48 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 2538.00 | 2549.05 | 2550.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 2530.90 | 2545.16 | 2548.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 2546.20 | 2545.36 | 2547.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2540.00 | 2544.29 | 2546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 2522.80 | 2544.29 | 2546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2525.00 | 2540.43 | 2544.97 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 2544.80 | 2542.05 | 2541.82 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 2529.70 | 2540.09 | 2541.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 2528.50 | 2537.77 | 2539.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2562.20 | 2544.37 | 2542.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 2570.00 | 2557.36 | 2550.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 2546.90 | 2555.27 | 2550.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 2553.00 | 2554.81 | 2550.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 2554.70 | 2554.81 | 2550.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 2465.10 | 2538.15 | 2544.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 2465.10 | 2538.15 | 2544.97 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 2544.80 | 2520.53 | 2518.02 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 2494.30 | 2514.56 | 2516.89 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 2531.40 | 2517.60 | 2517.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 2548.60 | 2526.91 | 2522.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 2651.90 | 2671.95 | 2648.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2654.00 | 2670.87 | 2658.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2646.70 | 2670.87 | 2658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2670.00 | 2670.69 | 2659.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 2673.90 | 2670.69 | 2659.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 2646.40 | 2656.11 | 2656.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 14:15:00 | 2646.40 | 2656.11 | 2656.12 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 2670.20 | 2656.78 | 2656.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 2699.60 | 2668.00 | 2661.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 2694.70 | 2700.01 | 2687.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 2694.70 | 2700.01 | 2687.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2679.10 | 2695.83 | 2686.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 2703.00 | 2695.83 | 2686.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 2697.00 | 2687.20 | 2685.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 2642.50 | 2679.83 | 2682.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 2642.50 | 2679.83 | 2682.27 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 2694.40 | 2674.59 | 2672.94 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2658.80 | 2672.15 | 2672.16 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 2682.80 | 2673.74 | 2672.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2713.00 | 2680.96 | 2676.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 2692.10 | 2694.61 | 2685.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 2698.80 | 2694.61 | 2685.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2681.40 | 2691.96 | 2684.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:45:00 | 2679.00 | 2691.96 | 2684.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2678.00 | 2689.17 | 2684.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 2704.40 | 2689.17 | 2684.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 2872.10 | 2891.51 | 2892.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 2872.10 | 2891.51 | 2892.44 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2930.10 | 2896.71 | 2893.73 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 2853.20 | 2892.86 | 2893.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 2829.00 | 2858.15 | 2873.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 2860.10 | 2854.03 | 2866.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 2860.10 | 2854.03 | 2866.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 2871.00 | 2858.34 | 2866.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 2871.80 | 2858.34 | 2866.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 2850.20 | 2856.72 | 2865.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2797.90 | 2838.47 | 2850.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 2819.00 | 2835.31 | 2846.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 2815.60 | 2825.84 | 2838.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 2876.50 | 2843.02 | 2842.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 2876.50 | 2843.02 | 2842.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 2891.00 | 2870.43 | 2857.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 2864.90 | 2872.84 | 2866.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2876.00 | 2873.47 | 2867.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 2856.90 | 2873.47 | 2867.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2858.00 | 2870.38 | 2866.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 2857.20 | 2870.38 | 2866.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 2832.10 | 2862.72 | 2863.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 2818.70 | 2845.48 | 2854.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 14:30:00 | 2617.30 | 2651.11 | 2683.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 2705.50 | 2695.14 | 2693.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 14:15:00 | 2705.50 | 2695.14 | 2693.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 2724.90 | 2705.33 | 2699.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 2753.70 | 2787.99 | 2763.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 2782.00 | 2786.80 | 2764.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 2749.00 | 2786.80 | 2764.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 2774.90 | 2786.34 | 2775.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 2771.20 | 2786.34 | 2775.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 2775.80 | 2784.23 | 2775.15 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 2753.50 | 2769.86 | 2770.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 2735.70 | 2763.03 | 2767.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2743.30 | 2741.56 | 2753.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 2743.30 | 2741.56 | 2753.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 2758.60 | 2744.97 | 2754.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 2758.60 | 2744.97 | 2754.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 2746.00 | 2745.17 | 2753.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 2762.50 | 2745.17 | 2753.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 2667.40 | 2671.88 | 2688.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 2684.40 | 2671.88 | 2688.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2700.80 | 2678.72 | 2688.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2700.80 | 2678.72 | 2688.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2706.00 | 2684.18 | 2690.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2684.30 | 2684.18 | 2690.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 2691.50 | 2684.81 | 2687.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 2718.60 | 2684.58 | 2684.67 | SL hit (close>static) qty=1.00 sl=2707.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 2714.70 | 2690.60 | 2687.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 2727.80 | 2703.47 | 2694.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 2707.40 | 2710.86 | 2701.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 2707.40 | 2710.86 | 2701.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2698.00 | 2708.29 | 2701.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2698.00 | 2708.29 | 2701.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2715.00 | 2709.63 | 2702.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 2727.20 | 2710.92 | 2705.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 2727.90 | 2711.73 | 2706.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 2726.30 | 2716.38 | 2710.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2682.60 | 2708.26 | 2707.71 | SL hit (close<static) qty=1.00 sl=2696.40 alert=retest2 |

### Cycle 188 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 2701.50 | 2706.91 | 2707.14 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 2731.90 | 2710.78 | 2708.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 2732.20 | 2715.06 | 2710.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 2848.80 | 2851.37 | 2814.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 2836.70 | 2851.37 | 2814.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 2884.60 | 2901.72 | 2887.53 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 2850.00 | 2876.20 | 2878.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 2835.40 | 2858.45 | 2868.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 2862.60 | 2819.26 | 2837.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2892.00 | 2833.81 | 2842.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 2892.00 | 2833.81 | 2842.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 2921.70 | 2851.38 | 2849.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 2925.80 | 2886.47 | 2868.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 2907.10 | 2914.01 | 2892.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 2907.10 | 2914.01 | 2892.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2861.90 | 2899.81 | 2891.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 2861.90 | 2899.81 | 2891.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2861.30 | 2892.11 | 2888.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 2861.30 | 2892.11 | 2888.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 2855.50 | 2884.78 | 2885.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2781.20 | 2850.57 | 2866.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 2796.00 | 2793.12 | 2816.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:30:00 | 2789.30 | 2793.12 | 2816.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2818.00 | 2796.60 | 2810.55 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 2843.10 | 2819.74 | 2818.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2961.40 | 2859.57 | 2838.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 2930.00 | 2937.50 | 2898.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:30:00 | 2941.20 | 2937.50 | 2898.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 2906.90 | 2926.83 | 2902.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 2910.70 | 2926.83 | 2902.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2900.10 | 2921.49 | 2902.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2900.10 | 2921.49 | 2902.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2923.70 | 2921.93 | 2904.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 2946.00 | 2922.74 | 2906.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 2897.30 | 2914.37 | 2906.47 | SL hit (close<static) qty=1.00 sl=2900.10 alert=retest2 |

### Cycle 194 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 2881.70 | 2901.67 | 2902.75 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 2913.10 | 2904.56 | 2903.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 2917.50 | 2907.14 | 2905.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 2986.00 | 2991.73 | 2972.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:00:00 | 2986.00 | 2991.73 | 2972.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 2978.90 | 2991.51 | 2981.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 2979.70 | 2991.51 | 2981.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2977.10 | 2988.63 | 2981.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 2975.00 | 2988.63 | 2981.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 2873.60 | 2967.77 | 2973.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 2779.30 | 2865.50 | 2910.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 2749.60 | 2745.42 | 2784.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 2749.60 | 2745.42 | 2784.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2687.30 | 2673.01 | 2689.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 2658.30 | 2672.81 | 2688.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 2661.20 | 2661.40 | 2676.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 2649.00 | 2655.27 | 2671.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 2660.30 | 2663.47 | 2672.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 2678.30 | 2666.44 | 2672.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2690.00 | 2677.61 | 2676.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2690.00 | 2677.61 | 2676.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2706.10 | 2683.31 | 2679.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 2818.70 | 2821.16 | 2795.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 2818.70 | 2821.16 | 2795.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 2794.90 | 2815.91 | 2795.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 2794.90 | 2815.91 | 2795.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 2798.80 | 2812.49 | 2795.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 2811.80 | 2812.49 | 2795.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 2815.50 | 2811.71 | 2796.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 2810.00 | 2806.15 | 2799.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 2787.20 | 2802.98 | 2799.26 | SL hit (close<static) qty=1.00 sl=2793.50 alert=retest2 |

### Cycle 198 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 2782.00 | 2796.08 | 2796.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2757.90 | 2782.47 | 2789.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 2779.20 | 2758.19 | 2770.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2791.40 | 2757.37 | 2762.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2792.30 | 2757.37 | 2762.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2784.20 | 2762.73 | 2764.50 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 2787.10 | 2767.61 | 2766.56 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 2721.90 | 2767.54 | 2770.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 2693.20 | 2730.83 | 2749.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 2717.60 | 2714.63 | 2734.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:00:00 | 2717.60 | 2714.63 | 2734.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 2743.70 | 2724.53 | 2734.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 2746.30 | 2724.53 | 2734.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 2750.30 | 2729.69 | 2736.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 2750.30 | 2729.69 | 2736.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 2748.00 | 2733.35 | 2737.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 2760.90 | 2735.68 | 2737.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2750.47 | 2744.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2797.46 | 2775.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 15:15:00 | 2812.10 | 2812.70 | 2794.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:15:00 | 2851.30 | 2812.70 | 2794.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2848.00 | 2866.53 | 2842.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 2848.00 | 2866.53 | 2842.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 2824.60 | 2858.14 | 2841.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 2824.60 | 2858.14 | 2841.21 | SL hit (close<ema400) qty=1.00 sl=2841.21 alert=retest1 |

### Cycle 202 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 2801.60 | 2834.82 | 2835.24 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 2885.20 | 2843.48 | 2838.86 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 2793.60 | 2853.46 | 2854.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 2770.80 | 2836.92 | 2846.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2786.10 | 2770.25 | 2793.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 2786.10 | 2770.25 | 2793.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2775.20 | 2774.29 | 2791.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 2763.10 | 2772.05 | 2789.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 2772.80 | 2766.56 | 2774.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 2771.60 | 2767.57 | 2774.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 2777.40 | 2763.97 | 2763.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2777.40 | 2763.97 | 2763.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 2789.30 | 2769.04 | 2765.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2767.30 | 2770.64 | 2767.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2775.20 | 2771.55 | 2768.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 2787.90 | 2774.35 | 2770.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 2781.20 | 2805.04 | 2806.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 2781.20 | 2805.04 | 2806.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2749.10 | 2791.34 | 2799.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 2709.50 | 2681.91 | 2707.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2704.00 | 2686.33 | 2707.19 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2746.40 | 2714.30 | 2714.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 2755.00 | 2729.73 | 2721.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 2738.00 | 2748.87 | 2738.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 2733.70 | 2745.84 | 2738.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 2795.00 | 2745.84 | 2738.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 2885.90 | 2903.83 | 2904.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 2885.90 | 2903.83 | 2904.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2851.00 | 2893.27 | 2899.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2914.80 | 2895.08 | 2898.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2887.20 | 2893.51 | 2897.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2871.90 | 2891.47 | 2896.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2863.90 | 2884.15 | 2891.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 2858.40 | 2868.23 | 2879.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2893.90 | 2884.36 | 2884.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 2893.90 | 2884.36 | 2884.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 2901.30 | 2887.75 | 2885.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 2879.50 | 2899.60 | 2894.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2879.10 | 2895.50 | 2893.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 2873.50 | 2895.50 | 2893.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2899.60 | 2896.32 | 2893.85 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 2849.40 | 2886.26 | 2890.18 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 2900.00 | 2877.40 | 2875.05 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 10:15:00 | 2866.20 | 2877.73 | 2878.41 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 2891.00 | 2879.77 | 2879.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 2909.90 | 2885.83 | 2881.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 2907.90 | 2920.59 | 2907.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 10:15:00 | 2907.90 | 2920.59 | 2907.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2907.90 | 2920.59 | 2907.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 2907.90 | 2920.59 | 2907.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 2887.00 | 2913.87 | 2905.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 2887.00 | 2913.87 | 2905.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 2902.60 | 2911.62 | 2904.97 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 2873.50 | 2897.48 | 2899.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 2866.10 | 2891.20 | 2896.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 2800.00 | 2799.53 | 2822.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 2787.00 | 2799.53 | 2822.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2809.50 | 2796.90 | 2807.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 2812.90 | 2796.90 | 2807.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 2816.10 | 2800.74 | 2808.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 2814.30 | 2800.74 | 2808.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 2814.90 | 2803.57 | 2808.79 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 2829.20 | 2813.64 | 2812.52 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 2807.50 | 2811.54 | 2811.91 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 2815.00 | 2812.23 | 2812.19 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 2808.30 | 2811.44 | 2811.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 2788.00 | 2806.28 | 2809.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 2805.60 | 2804.74 | 2808.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:00:00 | 2805.60 | 2804.74 | 2808.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 2808.10 | 2805.41 | 2808.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 2810.60 | 2805.41 | 2808.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2798.60 | 2804.05 | 2807.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:30:00 | 2800.00 | 2804.05 | 2807.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2802.00 | 2803.64 | 2806.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 2802.00 | 2803.64 | 2806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2807.60 | 2803.07 | 2805.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 2807.60 | 2803.07 | 2805.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2823.10 | 2807.07 | 2807.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2823.10 | 2807.07 | 2807.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 2831.30 | 2811.92 | 2809.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 2838.40 | 2820.52 | 2814.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2842.90 | 2852.39 | 2837.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 2842.90 | 2852.39 | 2837.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2831.90 | 2848.29 | 2836.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 2821.20 | 2848.29 | 2836.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2828.30 | 2844.30 | 2835.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2828.30 | 2844.30 | 2835.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2825.00 | 2840.44 | 2834.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 2822.60 | 2840.44 | 2834.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 2817.40 | 2830.32 | 2830.93 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 2835.70 | 2831.32 | 2830.89 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 2805.00 | 2826.47 | 2829.28 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2866.00 | 2832.58 | 2829.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 2875.20 | 2848.17 | 2838.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2829.30 | 2848.91 | 2841.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2829.30 | 2848.91 | 2841.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2829.30 | 2848.91 | 2841.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 2829.30 | 2848.91 | 2841.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 2782.70 | 2835.67 | 2835.76 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2919.30 | 2845.54 | 2836.22 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 2825.40 | 2860.89 | 2864.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 2807.20 | 2850.15 | 2859.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 2804.50 | 2789.60 | 2816.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 2804.50 | 2789.60 | 2816.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 2768.00 | 2785.28 | 2811.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 2854.50 | 2785.28 | 2811.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2872.20 | 2802.66 | 2817.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2884.00 | 2802.66 | 2817.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2842.10 | 2810.55 | 2819.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 2837.40 | 2814.34 | 2820.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 2825.00 | 2794.06 | 2793.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2825.00 | 2794.06 | 2793.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 2830.90 | 2801.43 | 2796.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 13:15:00 | 2819.00 | 2821.48 | 2809.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:45:00 | 2820.80 | 2821.48 | 2809.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2780.00 | 2817.41 | 2810.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 2785.50 | 2817.41 | 2810.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2781.10 | 2810.15 | 2808.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 2779.40 | 2810.15 | 2808.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 2763.60 | 2800.84 | 2804.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 2740.30 | 2776.02 | 2789.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 2815.10 | 2763.93 | 2769.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 2815.10 | 2763.93 | 2769.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2815.10 | 2763.93 | 2769.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 2810.00 | 2763.93 | 2769.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2768.50 | 2764.84 | 2769.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 2755.20 | 2764.84 | 2769.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 2754.00 | 2761.13 | 2766.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2846.00 | 2769.65 | 2764.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2846.00 | 2769.65 | 2764.69 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 2621.00 | 2748.60 | 2764.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2600.30 | 2645.55 | 2680.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2612.80 | 2612.43 | 2642.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2612.80 | 2612.43 | 2642.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2612.80 | 2612.43 | 2642.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 2600.30 | 2621.16 | 2628.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 2470.28 | 2565.48 | 2591.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 2488.90 | 2469.32 | 2513.91 | SL hit (close>ema200) qty=0.50 sl=2469.32 alert=retest2 |

### Cycle 231 — BUY (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 11:15:00 | 2295.40 | 2281.37 | 2279.60 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 2257.50 | 2279.81 | 2281.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 09:15:00 | 2244.20 | 2266.08 | 2271.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 2257.40 | 2240.24 | 2251.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 2257.40 | 2240.24 | 2251.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2257.40 | 2240.24 | 2251.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 2224.30 | 2238.77 | 2249.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 2113.09 | 2150.36 | 2170.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 2084.00 | 2083.93 | 2108.58 | SL hit (close>ema200) qty=0.50 sl=2083.93 alert=retest2 |

### Cycle 233 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 2157.70 | 2119.62 | 2114.60 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2066.90 | 2108.40 | 2113.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 2050.90 | 2090.36 | 2103.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2079.90 | 2073.62 | 2090.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2079.90 | 2073.62 | 2090.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2079.90 | 2073.62 | 2090.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2062.50 | 2084.49 | 2089.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 10:00:00 | 2061.90 | 2079.97 | 2086.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 12:45:00 | 2052.00 | 2073.85 | 2082.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 14:45:00 | 2061.50 | 2070.76 | 2079.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2060.20 | 2068.03 | 2076.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 2125.80 | 2080.89 | 2080.94 | SL hit (close>static) qty=1.00 sl=2109.70 alert=retest2 |

### Cycle 235 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 2117.00 | 2088.11 | 2084.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2134.40 | 2109.29 | 2096.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2116.10 | 2127.24 | 2114.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2116.10 | 2127.24 | 2114.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2116.10 | 2127.24 | 2114.60 | EMA400 retest candle locked (from upside) |

### Cycle 236 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2060.50 | 2104.94 | 2108.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 2053.20 | 2081.20 | 2094.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2136.60 | 2086.77 | 2093.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2136.60 | 2086.77 | 2093.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2136.60 | 2086.77 | 2093.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 2136.60 | 2086.77 | 2093.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2153.20 | 2100.06 | 2098.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 2159.40 | 2111.93 | 2104.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2098.30 | 2121.98 | 2114.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2098.30 | 2121.98 | 2114.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2098.30 | 2121.98 | 2114.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 10:30:00 | 2120.10 | 2121.99 | 2114.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 11:15:00 | 2332.11 | 2259.18 | 2213.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 238 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 2317.60 | 2325.72 | 2325.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 2315.30 | 2323.63 | 2324.98 | Break + close below crossover candle low |

### Cycle 239 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2373.00 | 2331.64 | 2328.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 2393.70 | 2351.23 | 2338.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.84 | EMA400 retest candle locked (from upside) |

### Cycle 240 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 2416.40 | 2423.68 | 2423.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2388.50 | 2415.47 | 2420.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2231.30 | 2202.31 | 2253.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 2232.60 | 2202.31 | 2253.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2244.00 | 2210.65 | 2252.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 2248.50 | 2210.65 | 2252.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 2258.50 | 2220.22 | 2253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 2258.50 | 2220.22 | 2253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2249.90 | 2226.15 | 2252.85 | EMA400 retest candle locked (from downside) |

### Cycle 241 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 2267.20 | 2261.46 | 2261.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 2335.40 | 2277.61 | 2268.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 2256.00 | 2278.31 | 2270.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2253.40 | 2273.33 | 2269.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 2253.00 | 2273.33 | 2269.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 242 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 2254.90 | 2265.06 | 2265.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 2241.90 | 2260.43 | 2263.78 | Break + close below crossover candle low |

### Cycle 243 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 2304.90 | 2269.32 | 2267.52 | EMA200 above EMA400 |

### Cycle 244 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 2256.50 | 2274.57 | 2276.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 2223.50 | 2261.59 | 2269.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 2221.60 | 2226.19 | 2243.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 2218.30 | 2222.09 | 2240.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 2220.00 | 2216.44 | 2232.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 2217.70 | 2207.77 | 2212.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 2214.90 | 2209.19 | 2213.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 2225.00 | 2209.19 | 2213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2214.50 | 2210.25 | 2213.24 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 09:30:00 | 2433.80 | 2024-04-16 13:15:00 | 2312.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-15 09:30:00 | 2433.80 | 2024-04-18 09:15:00 | 2353.80 | STOP_HIT | 0.50 | 3.29% |
| BUY | retest2 | 2024-05-02 10:15:00 | 2330.00 | 2024-05-03 11:15:00 | 2298.35 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-02 13:45:00 | 2329.25 | 2024-05-03 11:15:00 | 2298.35 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-03 09:15:00 | 2330.45 | 2024-05-03 11:15:00 | 2298.35 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-06 15:00:00 | 2296.40 | 2024-05-14 10:15:00 | 2277.80 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-05-16 09:15:00 | 2343.50 | 2024-05-29 09:15:00 | 2408.30 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2024-07-22 10:15:00 | 2890.00 | 2024-07-24 12:15:00 | 2846.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-23 12:30:00 | 2910.00 | 2024-07-24 12:15:00 | 2846.40 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-08-01 12:15:00 | 2888.05 | 2024-08-05 09:15:00 | 2743.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:15:00 | 2888.05 | 2024-08-05 11:15:00 | 2599.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-29 10:15:00 | 3093.65 | 2024-09-04 09:15:00 | 3040.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-29 11:00:00 | 3098.95 | 2024-09-04 09:15:00 | 3040.35 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-08-29 13:45:00 | 3096.30 | 2024-09-04 09:15:00 | 3040.35 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-29 14:45:00 | 3097.60 | 2024-09-04 09:15:00 | 3040.35 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-09-03 11:30:00 | 3113.90 | 2024-09-04 09:15:00 | 3040.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-09-17 11:15:00 | 3161.35 | 2024-09-18 09:15:00 | 3003.90 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2024-10-01 10:45:00 | 3015.15 | 2024-10-04 13:15:00 | 2864.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:45:00 | 3016.50 | 2024-10-04 13:15:00 | 2865.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 15:00:00 | 3012.20 | 2024-10-04 13:15:00 | 2861.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2996.60 | 2024-10-04 13:15:00 | 2846.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:45:00 | 3015.15 | 2024-10-07 09:15:00 | 2910.95 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2024-10-01 11:45:00 | 3016.50 | 2024-10-07 09:15:00 | 2910.95 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2024-10-01 15:00:00 | 3012.20 | 2024-10-07 09:15:00 | 2910.95 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2996.60 | 2024-10-07 09:15:00 | 2910.95 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2024-10-09 10:45:00 | 2911.25 | 2024-10-09 13:15:00 | 2917.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-10-09 12:15:00 | 2918.65 | 2024-10-09 13:15:00 | 2917.35 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-10-09 12:45:00 | 2918.00 | 2024-10-09 13:15:00 | 2917.35 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-10-29 10:15:00 | 2984.00 | 2024-10-30 09:15:00 | 3048.10 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-11-11 09:15:00 | 2842.00 | 2024-11-11 09:15:00 | 2872.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-18 09:15:00 | 2766.10 | 2024-11-22 09:15:00 | 2878.45 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2024-11-19 12:30:00 | 2829.40 | 2024-11-22 09:15:00 | 2878.45 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-19 14:00:00 | 2818.45 | 2024-11-22 09:15:00 | 2878.45 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-04 09:15:00 | 3044.65 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2024-12-04 12:15:00 | 3027.25 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2024-12-05 09:15:00 | 3053.50 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | 3.45% |
| BUY | retest2 | 2024-12-05 12:00:00 | 3044.05 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2024-12-13 13:15:00 | 3188.45 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-12-13 14:15:00 | 3189.40 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-12-13 15:15:00 | 3205.00 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-12-16 12:30:00 | 3190.55 | 2024-12-17 13:15:00 | 3158.75 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-27 10:30:00 | 2897.30 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2024-12-27 11:15:00 | 2895.35 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-12-27 13:00:00 | 2895.75 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-12-27 14:15:00 | 2894.60 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-31 09:15:00 | 2876.00 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-01-02 11:30:00 | 2890.40 | 2025-01-02 13:15:00 | 2886.80 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-01-15 13:15:00 | 2778.75 | 2025-01-16 09:15:00 | 2830.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-01-24 11:15:00 | 2910.00 | 2025-01-28 14:15:00 | 2883.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-02-04 10:45:00 | 2812.75 | 2025-02-05 13:15:00 | 2840.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-05 09:45:00 | 2814.15 | 2025-02-05 13:15:00 | 2840.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-02-05 11:30:00 | 2817.20 | 2025-02-05 13:15:00 | 2840.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-03-12 09:15:00 | 2243.00 | 2025-03-18 10:15:00 | 2273.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-13 11:15:00 | 2254.50 | 2025-03-18 10:15:00 | 2273.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-03-18 09:45:00 | 2246.55 | 2025-03-18 10:15:00 | 2273.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-03-21 10:15:00 | 2391.40 | 2025-04-01 09:15:00 | 2453.75 | STOP_HIT | 1.00 | 2.61% |
| SELL | retest2 | 2025-04-09 09:15:00 | 2065.05 | 2025-04-15 09:15:00 | 2238.30 | STOP_HIT | 1.00 | -8.39% |
| SELL | retest2 | 2025-04-11 13:45:00 | 2157.55 | 2025-04-15 09:15:00 | 2238.30 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-04-11 15:00:00 | 2160.45 | 2025-04-15 09:15:00 | 2238.30 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-04-17 11:00:00 | 2257.60 | 2025-04-23 09:15:00 | 2483.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:15:00 | 2260.90 | 2025-04-23 09:15:00 | 2486.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-02 11:30:00 | 2473.00 | 2025-05-07 09:15:00 | 2349.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 10:15:00 | 2471.20 | 2025-05-07 09:15:00 | 2347.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:30:00 | 2473.00 | 2025-05-07 10:15:00 | 2386.90 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-05-05 10:15:00 | 2471.20 | 2025-05-07 10:15:00 | 2386.90 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-05-14 11:15:00 | 2538.70 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-05-14 12:00:00 | 2537.20 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-05-14 13:00:00 | 2545.00 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-05-20 12:45:00 | 2553.10 | 2025-05-23 09:15:00 | 2583.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-05-21 11:30:00 | 2561.40 | 2025-05-23 09:15:00 | 2583.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-30 11:15:00 | 2554.70 | 2025-06-02 09:15:00 | 2465.10 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-06-13 10:15:00 | 2673.90 | 2025-06-13 14:15:00 | 2646.40 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-18 09:15:00 | 2703.00 | 2025-06-19 09:15:00 | 2642.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-18 15:15:00 | 2697.00 | 2025-06-19 09:15:00 | 2642.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-06-25 09:15:00 | 2704.40 | 2025-07-07 14:15:00 | 2872.10 | STOP_HIT | 1.00 | 6.20% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2797.90 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-14 11:45:00 | 2819.00 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-14 15:15:00 | 2815.60 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-07-25 14:30:00 | 2617.30 | 2025-07-28 14:15:00 | 2705.50 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2684.30 | 2025-08-12 09:15:00 | 2718.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-08 15:00:00 | 2691.50 | 2025-08-12 09:15:00 | 2718.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-14 10:30:00 | 2727.20 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-14 14:15:00 | 2727.90 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-08-18 09:30:00 | 2726.30 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-12 09:15:00 | 2946.00 | 2025-09-12 11:15:00 | 2897.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-30 11:15:00 | 2658.30 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-30 14:45:00 | 2661.20 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-01 09:45:00 | 2649.00 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-10-01 11:30:00 | 2660.30 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-09 09:15:00 | 2811.80 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-09 10:15:00 | 2815.50 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-09 15:15:00 | 2810.00 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-10-27 09:15:00 | 2851.30 | 2025-10-28 11:15:00 | 2824.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-04 11:00:00 | 2763.10 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-06 14:15:00 | 2772.80 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-11-06 15:00:00 | 2771.60 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-11 15:15:00 | 2787.90 | 2025-11-14 10:15:00 | 2781.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-11-24 09:15:00 | 2795.00 | 2025-12-08 15:15:00 | 2885.90 | STOP_HIT | 1.00 | 3.25% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2871.90 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2863.90 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-11 09:30:00 | 2858.40 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-22 11:30:00 | 2837.40 | 2026-01-27 15:15:00 | 2825.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2026-02-01 15:15:00 | 2755.20 | 2026-02-03 09:15:00 | 2846.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-02-02 10:00:00 | 2754.00 | 2026-02-03 09:15:00 | 2846.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-02-11 09:15:00 | 2600.30 | 2026-02-12 09:15:00 | 2470.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 2600.30 | 2026-02-13 11:15:00 | 2488.90 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2026-03-06 12:00:00 | 2224.30 | 2026-03-13 12:15:00 | 2113.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 2224.30 | 2026-03-17 11:15:00 | 2084.00 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2062.50 | 2026-03-24 11:15:00 | 2125.80 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-23 10:00:00 | 2061.90 | 2026-03-24 11:15:00 | 2125.80 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2026-03-23 12:45:00 | 2052.00 | 2026-03-24 11:15:00 | 2125.80 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-03-23 14:45:00 | 2061.50 | 2026-03-24 11:15:00 | 2125.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-04-02 10:30:00 | 2120.10 | 2026-04-07 11:15:00 | 2332.11 | TARGET_HIT | 1.00 | 10.00% |
