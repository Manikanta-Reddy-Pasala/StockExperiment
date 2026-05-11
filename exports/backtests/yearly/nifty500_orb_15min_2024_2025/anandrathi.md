# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 3602.30
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
| ENTRY1 | 55 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 48
- **Target hits / Stop hits / Partials:** 7 / 48 / 20
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 6.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 13 | 40.6% | 4 | 19 | 9 | 0.18% | 5.6% |
| BUY @ 2nd Alert (retest1) | 32 | 13 | 40.6% | 4 | 19 | 9 | 0.18% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 14 | 32.6% | 3 | 29 | 11 | 0.01% | 0.6% |
| SELL @ 2nd Alert (retest1) | 43 | 14 | 32.6% | 3 | 29 | 11 | 0.01% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 27 | 36.0% | 7 | 48 | 20 | 0.08% | 6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:45:00 | 1955.48 | 1968.05 | 0.00 | ORB-short ORB[1969.95,1979.53] vol=4.7x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-05-14 09:50:00 | 1961.63 | 1967.91 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 1983.75 | 1978.81 | 0.00 | ORB-long ORB[1972.50,1981.50] vol=1.5x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-05-15 10:00:00 | 1979.42 | 1979.96 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-18 09:55:00 | 1974.50 | 1974.95 | 0.00 | ORB-short ORB[1980.50,1987.48] vol=2.2x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-18 11:30:00 | 1965.96 | 1974.61 | 0.00 | T1 1.5R @ 1965.96 |
| Stop hit — per-position SL triggered | 2024-05-18 11:40:00 | 1974.50 | 1974.53 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 11:05:00 | 1957.23 | 1967.46 | 0.00 | ORB-short ORB[1962.50,1984.50] vol=2.7x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 11:35:00 | 1950.70 | 1964.91 | 0.00 | T1 1.5R @ 1950.70 |
| Stop hit — per-position SL triggered | 2024-05-21 12:25:00 | 1957.23 | 1961.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 09:45:00 | 1990.10 | 2005.90 | 0.00 | ORB-short ORB[2002.23,2022.33] vol=2.6x ATR=7.57 |
| Stop hit — per-position SL triggered | 2024-06-07 09:55:00 | 1997.67 | 2002.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 1935.00 | 1949.69 | 0.00 | ORB-short ORB[1942.50,1967.48] vol=2.2x ATR=10.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:55:00 | 1918.75 | 1935.70 | 0.00 | T1 1.5R @ 1918.75 |
| Target hit | 2024-06-10 15:00:00 | 1930.00 | 1928.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 1910.33 | 1919.84 | 0.00 | ORB-short ORB[1911.73,1937.08] vol=1.5x ATR=7.67 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 1918.00 | 1919.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:35:00 | 1959.33 | 1974.23 | 0.00 | ORB-short ORB[1964.15,1988.48] vol=2.2x ATR=8.54 |
| Stop hit — per-position SL triggered | 2024-06-12 12:35:00 | 1967.87 | 1964.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 1969.00 | 1971.00 | 0.00 | ORB-short ORB[1970.00,1990.00] vol=1.9x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 12:20:00 | 1965.32 | 1970.54 | 0.00 | T1 1.5R @ 1965.32 |
| Stop hit — per-position SL triggered | 2024-06-13 12:25:00 | 1969.00 | 1970.52 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:00:00 | 1989.40 | 2002.66 | 0.00 | ORB-short ORB[1993.08,2022.05] vol=1.6x ATR=5.67 |
| Stop hit — per-position SL triggered | 2024-06-18 10:05:00 | 1995.07 | 2001.89 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:35:00 | 2014.98 | 2007.23 | 0.00 | ORB-long ORB[1990.05,2010.98] vol=1.8x ATR=6.51 |
| Stop hit — per-position SL triggered | 2024-06-20 09:45:00 | 2008.47 | 2007.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:50:00 | 1940.48 | 1954.28 | 0.00 | ORB-short ORB[1945.38,1969.45] vol=4.3x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-06-25 11:35:00 | 1945.28 | 1949.26 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 1937.30 | 1945.06 | 0.00 | ORB-short ORB[1940.28,1958.23] vol=1.9x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-07-05 09:55:00 | 1941.12 | 1944.67 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 2045.00 | 2058.17 | 0.00 | ORB-short ORB[2057.48,2087.50] vol=1.5x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:25:00 | 2031.91 | 2053.62 | 0.00 | T1 1.5R @ 2031.91 |
| Stop hit — per-position SL triggered | 2024-07-10 10:50:00 | 2045.00 | 2043.77 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:30:00 | 1895.73 | 1903.95 | 0.00 | ORB-short ORB[1900.00,1927.50] vol=4.2x ATR=5.96 |
| Stop hit — per-position SL triggered | 2024-07-19 10:40:00 | 1901.69 | 1903.63 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 1829.50 | 1821.76 | 0.00 | ORB-long ORB[1799.98,1825.00] vol=1.7x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-08-09 10:00:00 | 1824.60 | 1822.18 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1796.98 | 1801.49 | 0.00 | ORB-short ORB[1800.00,1818.00] vol=1.6x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:40:00 | 1790.54 | 1799.54 | 0.00 | T1 1.5R @ 1790.54 |
| Stop hit — per-position SL triggered | 2024-08-16 10:00:00 | 1796.98 | 1797.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:40:00 | 1880.03 | 1871.70 | 0.00 | ORB-long ORB[1850.00,1877.50] vol=1.5x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:50:00 | 1887.75 | 1874.81 | 0.00 | T1 1.5R @ 1887.75 |
| Stop hit — per-position SL triggered | 2024-08-29 09:55:00 | 1880.03 | 1875.25 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:45:00 | 1882.85 | 1877.15 | 0.00 | ORB-long ORB[1852.53,1879.98] vol=3.8x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-08-30 10:55:00 | 1877.28 | 1877.76 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:45:00 | 1971.48 | 1964.74 | 0.00 | ORB-long ORB[1940.85,1966.40] vol=4.1x ATR=6.33 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 1965.15 | 1965.00 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:55:00 | 1955.83 | 1951.14 | 0.00 | ORB-long ORB[1942.35,1955.00] vol=3.6x ATR=4.62 |
| Stop hit — per-position SL triggered | 2024-09-13 10:10:00 | 1951.21 | 1951.52 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:35:00 | 1977.83 | 1987.75 | 0.00 | ORB-short ORB[1980.03,2000.00] vol=1.6x ATR=4.74 |
| Stop hit — per-position SL triggered | 2024-09-17 11:10:00 | 1982.57 | 1981.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 11:05:00 | 1910.10 | 1920.15 | 0.00 | ORB-short ORB[1916.68,1944.98] vol=7.5x ATR=6.42 |
| Stop hit — per-position SL triggered | 2024-09-20 11:25:00 | 1916.52 | 1918.89 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:40:00 | 1931.18 | 1941.32 | 0.00 | ORB-short ORB[1934.03,1949.93] vol=4.2x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 13:15:00 | 1920.22 | 1929.16 | 0.00 | T1 1.5R @ 1920.22 |
| Target hit | 2024-09-24 14:10:00 | 1920.80 | 1913.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2024-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:50:00 | 1931.90 | 1943.75 | 0.00 | ORB-short ORB[1945.75,1963.55] vol=3.9x ATR=7.32 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 1939.22 | 1940.79 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2075.98 | 2046.39 | 0.00 | ORB-long ORB[2021.30,2046.00] vol=4.9x ATR=10.01 |
| Stop hit — per-position SL triggered | 2024-10-15 09:35:00 | 2065.97 | 2048.06 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-11-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:35:00 | 2020.30 | 2007.55 | 0.00 | ORB-long ORB[1992.50,2018.35] vol=1.7x ATR=8.83 |
| Stop hit — per-position SL triggered | 2024-11-05 09:40:00 | 2011.47 | 2011.49 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-11-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 10:45:00 | 1953.55 | 1941.25 | 0.00 | ORB-long ORB[1925.00,1946.98] vol=2.7x ATR=8.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 11:05:00 | 1965.85 | 1945.47 | 0.00 | T1 1.5R @ 1965.85 |
| Target hit | 2024-11-18 15:20:00 | 1990.35 | 1978.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:45:00 | 2001.40 | 1987.46 | 0.00 | ORB-long ORB[1971.28,1998.40] vol=2.1x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:00:00 | 2012.69 | 1990.47 | 0.00 | T1 1.5R @ 2012.69 |
| Stop hit — per-position SL triggered | 2024-11-19 12:05:00 | 2001.40 | 1995.50 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 11:10:00 | 2045.40 | 2050.88 | 0.00 | ORB-short ORB[2046.03,2072.50] vol=1.8x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-11-27 12:05:00 | 2050.30 | 2050.40 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 11:05:00 | 2083.75 | 2091.80 | 0.00 | ORB-short ORB[2084.57,2103.50] vol=2.0x ATR=4.25 |
| Stop hit — per-position SL triggered | 2024-12-03 12:10:00 | 2088.00 | 2089.37 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:35:00 | 2116.73 | 2107.43 | 0.00 | ORB-long ORB[2092.07,2105.65] vol=3.8x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:45:00 | 2129.33 | 2110.04 | 0.00 | T1 1.5R @ 2129.33 |
| Stop hit — per-position SL triggered | 2024-12-05 09:50:00 | 2116.73 | 2110.36 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 2191.32 | 2174.70 | 0.00 | ORB-long ORB[2154.13,2184.98] vol=1.7x ATR=10.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:35:00 | 2207.66 | 2194.90 | 0.00 | T1 1.5R @ 2207.66 |
| Target hit | 2024-12-09 10:05:00 | 2213.28 | 2215.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2024-12-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:10:00 | 2152.50 | 2159.42 | 0.00 | ORB-short ORB[2155.70,2180.55] vol=1.6x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:30:00 | 2140.73 | 2154.64 | 0.00 | T1 1.5R @ 2140.73 |
| Target hit | 2024-12-13 15:20:00 | 2131.25 | 2144.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-12-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:45:00 | 2054.78 | 2064.93 | 0.00 | ORB-short ORB[2060.00,2090.18] vol=3.5x ATR=6.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:50:00 | 2044.59 | 2060.82 | 0.00 | T1 1.5R @ 2044.59 |
| Stop hit — per-position SL triggered | 2024-12-20 09:55:00 | 2054.78 | 2060.69 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 1992.80 | 2000.56 | 0.00 | ORB-short ORB[1997.50,2015.73] vol=3.4x ATR=6.24 |
| Stop hit — per-position SL triggered | 2024-12-26 11:30:00 | 1999.04 | 2000.39 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:15:00 | 1992.88 | 1982.15 | 0.00 | ORB-long ORB[1967.53,1985.00] vol=5.9x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-01-01 11:20:00 | 1988.20 | 1982.33 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1968.15 | 1969.81 | 0.00 | ORB-short ORB[1972.83,1989.00] vol=2.7x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-01-02 15:15:00 | 1972.88 | 1968.39 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:55:00 | 1974.40 | 1984.98 | 0.00 | ORB-short ORB[1980.00,1999.60] vol=1.8x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-01-06 10:00:00 | 1980.77 | 1984.26 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 1890.50 | 1895.55 | 0.00 | ORB-short ORB[1892.55,1904.85] vol=2.2x ATR=4.86 |
| Stop hit — per-position SL triggered | 2025-01-24 10:40:00 | 1895.36 | 1891.11 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:05:00 | 1792.88 | 1780.74 | 0.00 | ORB-long ORB[1765.43,1788.93] vol=1.8x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:10:00 | 1802.29 | 1782.81 | 0.00 | T1 1.5R @ 1802.29 |
| Target hit | 2025-01-30 12:45:00 | 1822.63 | 1828.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:15:00 | 1828.75 | 1814.73 | 0.00 | ORB-long ORB[1803.05,1821.40] vol=3.3x ATR=6.89 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 1821.86 | 1815.56 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-02-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 10:35:00 | 1834.63 | 1819.52 | 0.00 | ORB-long ORB[1805.50,1831.25] vol=2.0x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-02-04 10:40:00 | 1827.52 | 1820.22 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:55:00 | 1885.00 | 1895.84 | 0.00 | ORB-short ORB[1888.90,1914.65] vol=2.5x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-02-06 13:20:00 | 1889.55 | 1890.09 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-02-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:20:00 | 1863.53 | 1873.53 | 0.00 | ORB-short ORB[1868.50,1885.00] vol=2.6x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 10:25:00 | 1853.72 | 1872.07 | 0.00 | T1 1.5R @ 1853.72 |
| Stop hit — per-position SL triggered | 2025-02-07 11:10:00 | 1863.53 | 1868.18 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-02-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:10:00 | 1800.00 | 1808.34 | 0.00 | ORB-short ORB[1812.93,1832.40] vol=6.9x ATR=7.69 |
| Stop hit — per-position SL triggered | 2025-02-14 10:20:00 | 1807.69 | 1807.58 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 10:50:00 | 1821.48 | 1809.72 | 0.00 | ORB-long ORB[1794.23,1810.03] vol=2.0x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:20:00 | 1832.68 | 1814.23 | 0.00 | T1 1.5R @ 1832.68 |
| Stop hit — per-position SL triggered | 2025-02-18 12:50:00 | 1821.48 | 1820.12 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:45:00 | 1867.15 | 1861.55 | 0.00 | ORB-long ORB[1840.85,1867.08] vol=2.8x ATR=10.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:45:00 | 1883.13 | 1867.65 | 0.00 | T1 1.5R @ 1883.13 |
| Target hit | 2025-02-19 14:25:00 | 1880.53 | 1885.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-03-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-03 10:25:00 | 2012.63 | 2002.45 | 0.00 | ORB-long ORB[1990.00,2012.35] vol=1.8x ATR=7.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:40:00 | 2024.28 | 2004.38 | 0.00 | T1 1.5R @ 2024.28 |
| Stop hit — per-position SL triggered | 2025-03-03 11:25:00 | 2012.63 | 2007.63 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-03-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:50:00 | 1744.25 | 1743.06 | 0.00 | ORB-long ORB[1722.90,1743.00] vol=6.0x ATR=13.41 |
| Stop hit — per-position SL triggered | 2025-03-21 10:00:00 | 1730.84 | 1742.94 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 1839.25 | 1822.71 | 0.00 | ORB-long ORB[1813.50,1832.75] vol=1.7x ATR=8.03 |
| Stop hit — per-position SL triggered | 2025-03-27 10:55:00 | 1831.22 | 1823.48 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:40:00 | 1769.25 | 1769.78 | 0.00 | ORB-short ORB[1770.00,1785.95] vol=3.9x ATR=5.96 |
| Stop hit — per-position SL triggered | 2025-04-09 10:45:00 | 1775.21 | 1770.45 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:30:00 | 1781.10 | 1773.69 | 0.00 | ORB-long ORB[1761.00,1779.00] vol=1.6x ATR=6.70 |
| Stop hit — per-position SL triggered | 2025-04-23 10:40:00 | 1774.40 | 1774.00 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:05:00 | 1719.00 | 1728.88 | 0.00 | ORB-short ORB[1730.00,1753.20] vol=1.9x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:20:00 | 1713.09 | 1727.35 | 0.00 | T1 1.5R @ 1713.09 |
| Stop hit — per-position SL triggered | 2025-05-06 11:35:00 | 1719.00 | 1726.80 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-05-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 09:30:00 | 1652.10 | 1657.76 | 0.00 | ORB-short ORB[1652.50,1670.00] vol=1.6x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-05-09 09:35:00 | 1658.08 | 1657.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:45:00 | 1955.48 | 2024-05-14 09:50:00 | 1961.63 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-15 09:30:00 | 1983.75 | 2024-05-15 10:00:00 | 1979.42 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-18 09:55:00 | 1974.50 | 2024-05-18 11:30:00 | 1965.96 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-18 09:55:00 | 1974.50 | 2024-05-18 11:40:00 | 1974.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-21 11:05:00 | 1957.23 | 2024-05-21 11:35:00 | 1950.70 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-21 11:05:00 | 1957.23 | 2024-05-21 12:25:00 | 1957.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-07 09:45:00 | 1990.10 | 2024-06-07 09:55:00 | 1997.67 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-10 09:30:00 | 1935.00 | 2024-06-10 10:55:00 | 1918.75 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-06-10 09:30:00 | 1935.00 | 2024-06-10 15:00:00 | 1930.00 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-11 09:35:00 | 1910.33 | 2024-06-11 09:40:00 | 1918.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-12 09:35:00 | 1959.33 | 2024-06-12 12:35:00 | 1967.87 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-13 11:15:00 | 1969.00 | 2024-06-13 12:20:00 | 1965.32 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2024-06-13 11:15:00 | 1969.00 | 2024-06-13 12:25:00 | 1969.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 10:00:00 | 1989.40 | 2024-06-18 10:05:00 | 1995.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-20 09:35:00 | 2014.98 | 2024-06-20 09:45:00 | 2008.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-25 10:50:00 | 1940.48 | 2024-06-25 11:35:00 | 1945.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-05 09:50:00 | 1937.30 | 2024-07-05 09:55:00 | 1941.12 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-10 10:05:00 | 2045.00 | 2024-07-10 10:25:00 | 2031.91 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-07-10 10:05:00 | 2045.00 | 2024-07-10 10:50:00 | 2045.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:30:00 | 1895.73 | 2024-07-19 10:40:00 | 1901.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-09 09:50:00 | 1829.50 | 2024-08-09 10:00:00 | 1824.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-16 09:30:00 | 1796.98 | 2024-08-16 09:40:00 | 1790.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-16 09:30:00 | 1796.98 | 2024-08-16 10:00:00 | 1796.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:40:00 | 1880.03 | 2024-08-29 09:50:00 | 1887.75 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-29 09:40:00 | 1880.03 | 2024-08-29 09:55:00 | 1880.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:45:00 | 1882.85 | 2024-08-30 10:55:00 | 1877.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-11 09:45:00 | 1971.48 | 2024-09-11 10:00:00 | 1965.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-13 09:55:00 | 1955.83 | 2024-09-13 10:10:00 | 1951.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-17 10:35:00 | 1977.83 | 2024-09-17 11:10:00 | 1982.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-20 11:05:00 | 1910.10 | 2024-09-20 11:25:00 | 1916.52 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-24 09:40:00 | 1931.18 | 2024-09-24 13:15:00 | 1920.22 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-24 09:40:00 | 1931.18 | 2024-09-24 14:10:00 | 1920.80 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-01 10:50:00 | 1931.90 | 2024-10-01 10:55:00 | 1939.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-15 09:30:00 | 2075.98 | 2024-10-15 09:35:00 | 2065.97 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-11-05 09:35:00 | 2020.30 | 2024-11-05 09:40:00 | 2011.47 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-18 10:45:00 | 1953.55 | 2024-11-18 11:05:00 | 1965.85 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-11-18 10:45:00 | 1953.55 | 2024-11-18 15:20:00 | 1990.35 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2024-11-19 10:45:00 | 2001.40 | 2024-11-19 11:00:00 | 2012.69 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-19 10:45:00 | 2001.40 | 2024-11-19 12:05:00 | 2001.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 11:10:00 | 2045.40 | 2024-11-27 12:05:00 | 2050.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-03 11:05:00 | 2083.75 | 2024-12-03 12:10:00 | 2088.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-05 09:35:00 | 2116.73 | 2024-12-05 09:45:00 | 2129.33 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-12-05 09:35:00 | 2116.73 | 2024-12-05 09:50:00 | 2116.73 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 09:30:00 | 2191.32 | 2024-12-09 09:35:00 | 2207.66 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-12-09 09:30:00 | 2191.32 | 2024-12-09 10:05:00 | 2213.28 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2024-12-13 10:10:00 | 2152.50 | 2024-12-13 10:30:00 | 2140.73 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-13 10:10:00 | 2152.50 | 2024-12-13 15:20:00 | 2131.25 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2024-12-20 09:45:00 | 2054.78 | 2024-12-20 09:50:00 | 2044.59 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-20 09:45:00 | 2054.78 | 2024-12-20 09:55:00 | 2054.78 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 11:05:00 | 1992.80 | 2024-12-26 11:30:00 | 1999.04 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-01 11:15:00 | 1992.88 | 2025-01-01 11:20:00 | 1988.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-02 11:15:00 | 1968.15 | 2025-01-02 15:15:00 | 1972.88 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-06 09:55:00 | 1974.40 | 2025-01-06 10:00:00 | 1980.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-24 09:35:00 | 1890.50 | 2025-01-24 10:40:00 | 1895.36 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-30 11:05:00 | 1792.88 | 2025-01-30 11:10:00 | 1802.29 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-30 11:05:00 | 1792.88 | 2025-01-30 12:45:00 | 1822.63 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2025-01-31 11:15:00 | 1828.75 | 2025-01-31 11:30:00 | 1821.86 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-04 10:35:00 | 1834.63 | 2025-02-04 10:40:00 | 1827.52 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-06 10:55:00 | 1885.00 | 2025-02-06 13:20:00 | 1889.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-02-07 10:20:00 | 1863.53 | 2025-02-07 10:25:00 | 1853.72 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-02-07 10:20:00 | 1863.53 | 2025-02-07 11:10:00 | 1863.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:10:00 | 1800.00 | 2025-02-14 10:20:00 | 1807.69 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-18 10:50:00 | 1821.48 | 2025-02-18 11:20:00 | 1832.68 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-02-18 10:50:00 | 1821.48 | 2025-02-18 12:50:00 | 1821.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-19 09:45:00 | 1867.15 | 2025-02-19 10:45:00 | 1883.13 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-02-19 09:45:00 | 1867.15 | 2025-02-19 14:25:00 | 1880.53 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2025-03-03 10:25:00 | 2012.63 | 2025-03-03 10:40:00 | 2024.28 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-03 10:25:00 | 2012.63 | 2025-03-03 11:25:00 | 2012.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:50:00 | 1744.25 | 2025-03-21 10:00:00 | 1730.84 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest1 | 2025-03-27 10:45:00 | 1839.25 | 2025-03-27 10:55:00 | 1831.22 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-04-09 10:40:00 | 1769.25 | 2025-04-09 10:45:00 | 1775.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-23 10:30:00 | 1781.10 | 2025-04-23 10:40:00 | 1774.40 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-06 11:05:00 | 1719.00 | 2025-05-06 11:20:00 | 1713.09 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-06 11:05:00 | 1719.00 | 2025-05-06 11:35:00 | 1719.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-09 09:30:00 | 1652.10 | 2025-05-09 09:35:00 | 1658.08 | STOP_HIT | 1.00 | -0.36% |
