# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 902.50
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
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 43
- **Target hits / Stop hits / Partials:** 9 / 43 / 23
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 13.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 17 | 43.6% | 5 | 22 | 12 | 0.16% | 6.3% |
| BUY @ 2nd Alert (retest1) | 39 | 17 | 43.6% | 5 | 22 | 12 | 0.16% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 15 | 41.7% | 4 | 21 | 11 | 0.21% | 7.5% |
| SELL @ 2nd Alert (retest1) | 36 | 15 | 41.7% | 4 | 21 | 11 | 0.21% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 32 | 42.7% | 9 | 43 | 23 | 0.18% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 1711.90 | 1708.01 | 0.00 | ORB-long ORB[1695.15,1711.80] vol=2.2x ATR=4.19 |
| Stop hit — per-position SL triggered | 2024-05-15 10:05:00 | 1707.71 | 1710.31 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 1739.50 | 1757.06 | 0.00 | ORB-short ORB[1754.05,1770.95] vol=2.4x ATR=6.14 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 1745.64 | 1754.06 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1780.10 | 1774.75 | 0.00 | ORB-long ORB[1756.30,1779.70] vol=1.7x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:40:00 | 1791.20 | 1779.83 | 0.00 | T1 1.5R @ 1791.20 |
| Target hit | 2024-05-27 12:55:00 | 1800.10 | 1800.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2024-05-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:55:00 | 1796.70 | 1798.91 | 0.00 | ORB-short ORB[1797.05,1815.45] vol=6.6x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:05:00 | 1787.52 | 1798.75 | 0.00 | T1 1.5R @ 1787.52 |
| Stop hit — per-position SL triggered | 2024-05-28 13:50:00 | 1796.70 | 1793.69 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 09:55:00 | 1754.00 | 1772.22 | 0.00 | ORB-short ORB[1772.00,1787.00] vol=1.6x ATR=8.52 |
| Stop hit — per-position SL triggered | 2024-05-29 10:00:00 | 1762.52 | 1769.99 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 10:50:00 | 1883.20 | 1891.51 | 0.00 | ORB-short ORB[1891.70,1903.95] vol=2.6x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-06-20 11:00:00 | 1887.59 | 1891.16 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:20:00 | 1893.05 | 1873.88 | 0.00 | ORB-long ORB[1858.40,1870.00] vol=1.8x ATR=8.07 |
| Stop hit — per-position SL triggered | 2024-06-24 10:45:00 | 1884.98 | 1875.91 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:10:00 | 1853.30 | 1865.21 | 0.00 | ORB-short ORB[1861.10,1884.35] vol=2.8x ATR=3.81 |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 1857.11 | 1864.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 1826.40 | 1832.19 | 0.00 | ORB-short ORB[1828.00,1850.55] vol=2.6x ATR=6.46 |
| Stop hit — per-position SL triggered | 2024-06-27 09:35:00 | 1832.86 | 1831.18 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:00:00 | 1870.35 | 1864.23 | 0.00 | ORB-long ORB[1845.00,1866.80] vol=1.7x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-06-28 11:35:00 | 1863.91 | 1865.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 1814.80 | 1826.16 | 0.00 | ORB-short ORB[1825.00,1849.80] vol=2.0x ATR=6.35 |
| Stop hit — per-position SL triggered | 2024-07-04 10:00:00 | 1821.15 | 1825.39 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:55:00 | 1804.50 | 1809.72 | 0.00 | ORB-short ORB[1811.30,1829.20] vol=5.7x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 12:45:00 | 1798.30 | 1807.41 | 0.00 | T1 1.5R @ 1798.30 |
| Target hit | 2024-07-05 15:20:00 | 1790.90 | 1800.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:45:00 | 1793.45 | 1796.62 | 0.00 | ORB-short ORB[1796.80,1807.95] vol=1.6x ATR=6.35 |
| Stop hit — per-position SL triggered | 2024-07-10 10:50:00 | 1799.80 | 1796.63 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:05:00 | 1789.95 | 1793.53 | 0.00 | ORB-short ORB[1791.10,1801.35] vol=3.3x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:40:00 | 1782.15 | 1791.31 | 0.00 | T1 1.5R @ 1782.15 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 1789.95 | 1790.77 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1808.00 | 1824.28 | 0.00 | ORB-short ORB[1823.20,1842.00] vol=2.7x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1813.38 | 1820.04 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:35:00 | 1877.75 | 1867.16 | 0.00 | ORB-long ORB[1845.55,1869.85] vol=2.1x ATR=9.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:05:00 | 1892.24 | 1876.07 | 0.00 | T1 1.5R @ 1892.24 |
| Stop hit — per-position SL triggered | 2024-07-24 12:10:00 | 1877.75 | 1879.72 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 1897.85 | 1883.69 | 0.00 | ORB-long ORB[1861.25,1886.60] vol=2.8x ATR=6.59 |
| Stop hit — per-position SL triggered | 2024-07-25 15:10:00 | 1891.26 | 1893.45 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 11:10:00 | 1805.25 | 1798.07 | 0.00 | ORB-long ORB[1780.05,1802.80] vol=1.6x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 12:25:00 | 1810.80 | 1800.37 | 0.00 | T1 1.5R @ 1810.80 |
| Target hit | 2024-08-19 15:20:00 | 1820.45 | 1808.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-09-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:25:00 | 1994.45 | 1984.51 | 0.00 | ORB-long ORB[1973.90,1989.85] vol=1.8x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:45:00 | 2002.62 | 1986.87 | 0.00 | T1 1.5R @ 2002.62 |
| Stop hit — per-position SL triggered | 2024-09-10 11:30:00 | 1994.45 | 1988.81 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 2098.85 | 2117.99 | 0.00 | ORB-short ORB[2108.40,2139.90] vol=1.8x ATR=8.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:40:00 | 2086.71 | 2114.91 | 0.00 | T1 1.5R @ 2086.71 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 2098.85 | 2108.72 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 2082.85 | 2073.20 | 0.00 | ORB-long ORB[2054.75,2082.30] vol=2.6x ATR=12.32 |
| Stop hit — per-position SL triggered | 2024-09-19 09:35:00 | 2070.53 | 2073.48 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:05:00 | 2038.55 | 2044.62 | 0.00 | ORB-short ORB[2045.05,2069.75] vol=2.8x ATR=7.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:15:00 | 2026.84 | 2043.06 | 0.00 | T1 1.5R @ 2026.84 |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 2038.55 | 2035.73 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 2014.95 | 2038.64 | 0.00 | ORB-short ORB[2035.00,2060.15] vol=5.3x ATR=8.11 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 2023.06 | 2028.63 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 11:05:00 | 2008.45 | 1997.21 | 0.00 | ORB-long ORB[1984.50,1998.65] vol=1.8x ATR=6.03 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 2002.42 | 1997.46 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 1927.95 | 1946.46 | 0.00 | ORB-short ORB[1965.55,1992.60] vol=5.9x ATR=7.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:20:00 | 1916.03 | 1941.76 | 0.00 | T1 1.5R @ 1916.03 |
| Target hit | 2024-09-27 15:20:00 | 1895.35 | 1928.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-10-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:10:00 | 1859.90 | 1869.45 | 0.00 | ORB-short ORB[1875.00,1896.70] vol=2.4x ATR=8.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:25:00 | 1847.86 | 1866.41 | 0.00 | T1 1.5R @ 1847.86 |
| Target hit | 2024-10-07 11:45:00 | 1852.95 | 1850.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2024-10-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:55:00 | 1889.05 | 1883.43 | 0.00 | ORB-long ORB[1870.05,1883.10] vol=2.0x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-10-10 11:10:00 | 1885.02 | 1883.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 1880.80 | 1871.70 | 0.00 | ORB-long ORB[1848.00,1874.60] vol=3.6x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 1874.30 | 1876.00 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 1858.90 | 1865.65 | 0.00 | ORB-short ORB[1860.60,1881.10] vol=1.8x ATR=5.35 |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 1864.25 | 1861.48 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:35:00 | 1810.55 | 1821.19 | 0.00 | ORB-short ORB[1820.05,1844.95] vol=5.7x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:30:00 | 1802.13 | 1810.70 | 0.00 | T1 1.5R @ 1802.13 |
| Stop hit — per-position SL triggered | 2024-10-21 11:35:00 | 1810.55 | 1810.70 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:05:00 | 1851.75 | 1835.46 | 0.00 | ORB-long ORB[1814.25,1838.00] vol=2.9x ATR=8.53 |
| Stop hit — per-position SL triggered | 2024-10-31 10:20:00 | 1843.22 | 1838.10 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 11:05:00 | 1880.05 | 1865.91 | 0.00 | ORB-long ORB[1855.00,1876.50] vol=3.3x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:15:00 | 1891.16 | 1870.43 | 0.00 | T1 1.5R @ 1891.16 |
| Target hit | 2024-11-06 15:20:00 | 1910.40 | 1904.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 1864.50 | 1860.88 | 0.00 | ORB-long ORB[1848.70,1862.00] vol=3.0x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:55:00 | 1871.80 | 1863.86 | 0.00 | T1 1.5R @ 1871.80 |
| Stop hit — per-position SL triggered | 2024-12-02 10:10:00 | 1864.50 | 1864.37 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:55:00 | 2003.95 | 2012.41 | 0.00 | ORB-short ORB[2013.55,2042.25] vol=1.7x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:50:00 | 1994.53 | 2010.17 | 0.00 | T1 1.5R @ 1994.53 |
| Target hit | 2024-12-20 15:20:00 | 1920.80 | 1966.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-12-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:45:00 | 1924.90 | 1910.47 | 0.00 | ORB-long ORB[1887.65,1911.00] vol=2.9x ATR=7.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:55:00 | 1935.90 | 1918.49 | 0.00 | T1 1.5R @ 1935.90 |
| Target hit | 2024-12-24 10:35:00 | 1935.35 | 1937.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 1829.35 | 1817.16 | 0.00 | ORB-long ORB[1804.05,1818.95] vol=2.1x ATR=6.69 |
| Stop hit — per-position SL triggered | 2025-01-02 10:05:00 | 1822.66 | 1821.87 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:35:00 | 1806.85 | 1813.80 | 0.00 | ORB-short ORB[1810.05,1825.35] vol=1.9x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:15:00 | 1796.83 | 1811.07 | 0.00 | T1 1.5R @ 1796.83 |
| Stop hit — per-position SL triggered | 2025-01-08 11:30:00 | 1806.85 | 1810.70 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:00:00 | 1717.60 | 1732.65 | 0.00 | ORB-short ORB[1725.50,1748.60] vol=1.8x ATR=9.93 |
| Stop hit — per-position SL triggered | 2025-01-13 10:10:00 | 1727.53 | 1731.79 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:35:00 | 1708.85 | 1714.96 | 0.00 | ORB-short ORB[1712.25,1730.65] vol=2.1x ATR=7.80 |
| Stop hit — per-position SL triggered | 2025-01-15 10:05:00 | 1716.65 | 1714.13 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:20:00 | 1770.90 | 1754.54 | 0.00 | ORB-long ORB[1740.00,1764.90] vol=2.1x ATR=7.92 |
| Stop hit — per-position SL triggered | 2025-01-20 10:55:00 | 1762.98 | 1762.10 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:40:00 | 1739.05 | 1725.69 | 0.00 | ORB-long ORB[1694.00,1717.50] vol=2.0x ATR=8.49 |
| Stop hit — per-position SL triggered | 2025-01-23 10:10:00 | 1730.56 | 1731.11 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 1371.75 | 1354.12 | 0.00 | ORB-long ORB[1345.55,1360.00] vol=1.9x ATR=7.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:30:00 | 1382.80 | 1363.43 | 0.00 | T1 1.5R @ 1382.80 |
| Stop hit — per-position SL triggered | 2025-01-30 11:40:00 | 1371.75 | 1369.48 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-02-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:55:00 | 1504.80 | 1495.96 | 0.00 | ORB-long ORB[1483.90,1503.85] vol=2.8x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:30:00 | 1513.59 | 1500.71 | 0.00 | T1 1.5R @ 1513.59 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1504.80 | 1503.86 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:00:00 | 1468.75 | 1482.71 | 0.00 | ORB-short ORB[1478.75,1496.00] vol=2.9x ATR=6.82 |
| Stop hit — per-position SL triggered | 2025-02-11 10:30:00 | 1475.57 | 1477.08 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:35:00 | 1456.25 | 1450.83 | 0.00 | ORB-long ORB[1433.00,1454.85] vol=1.6x ATR=8.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:40:00 | 1468.65 | 1454.74 | 0.00 | T1 1.5R @ 1468.65 |
| Target hit | 2025-02-13 13:00:00 | 1467.00 | 1467.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-19 09:30:00 | 1446.95 | 1456.04 | 0.00 | ORB-short ORB[1450.00,1469.90] vol=2.1x ATR=6.80 |
| Stop hit — per-position SL triggered | 2025-02-19 09:45:00 | 1453.75 | 1455.69 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:55:00 | 1325.00 | 1331.74 | 0.00 | ORB-short ORB[1330.25,1341.55] vol=1.5x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 10:00:00 | 1316.96 | 1330.30 | 0.00 | T1 1.5R @ 1316.96 |
| Stop hit — per-position SL triggered | 2025-02-27 10:05:00 | 1325.00 | 1328.10 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:20:00 | 1297.80 | 1295.41 | 0.00 | ORB-long ORB[1274.05,1288.75] vol=16.1x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 10:25:00 | 1306.03 | 1295.52 | 0.00 | T1 1.5R @ 1306.03 |
| Stop hit — per-position SL triggered | 2025-03-06 10:50:00 | 1297.80 | 1295.67 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:00:00 | 1213.45 | 1205.31 | 0.00 | ORB-long ORB[1191.55,1207.60] vol=1.6x ATR=5.09 |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 1208.36 | 1205.99 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 1210.80 | 1201.16 | 0.00 | ORB-long ORB[1190.30,1207.50] vol=1.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 1206.34 | 1201.99 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:35:00 | 1213.50 | 1203.77 | 0.00 | ORB-long ORB[1190.00,1205.50] vol=2.6x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:45:00 | 1221.06 | 1209.26 | 0.00 | T1 1.5R @ 1221.06 |
| Stop hit — per-position SL triggered | 2025-05-05 10:05:00 | 1213.50 | 1212.71 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 1195.80 | 1190.37 | 0.00 | ORB-long ORB[1180.70,1194.70] vol=1.8x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-05-08 09:40:00 | 1191.63 | 1190.57 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:35:00 | 1711.90 | 2024-05-15 10:05:00 | 1707.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-22 09:40:00 | 1739.50 | 2024-05-22 09:50:00 | 1745.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1780.10 | 2024-05-27 09:40:00 | 1791.20 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1780.10 | 2024-05-27 12:55:00 | 1800.10 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2024-05-28 10:55:00 | 1796.70 | 2024-05-28 11:05:00 | 1787.52 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-28 10:55:00 | 1796.70 | 2024-05-28 13:50:00 | 1796.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-29 09:55:00 | 1754.00 | 2024-05-29 10:00:00 | 1762.52 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-06-20 10:50:00 | 1883.20 | 2024-06-20 11:00:00 | 1887.59 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-24 10:20:00 | 1893.05 | 2024-06-24 10:45:00 | 1884.98 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-06-26 11:10:00 | 1853.30 | 2024-06-26 11:15:00 | 1857.11 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-27 09:30:00 | 1826.40 | 2024-06-27 09:35:00 | 1832.86 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-28 11:00:00 | 1870.35 | 2024-06-28 11:35:00 | 1863.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-04 09:55:00 | 1814.80 | 2024-07-04 10:00:00 | 1821.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-05 10:55:00 | 1804.50 | 2024-07-05 12:45:00 | 1798.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-07-05 10:55:00 | 1804.50 | 2024-07-05 15:20:00 | 1790.90 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-07-10 10:45:00 | 1793.45 | 2024-07-10 10:50:00 | 1799.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-11 10:05:00 | 1789.95 | 2024-07-11 10:40:00 | 1782.15 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-11 10:05:00 | 1789.95 | 2024-07-11 11:40:00 | 1789.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1808.00 | 2024-07-18 09:40:00 | 1813.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-24 09:35:00 | 1877.75 | 2024-07-24 10:05:00 | 1892.24 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-07-24 09:35:00 | 1877.75 | 2024-07-24 12:10:00 | 1877.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 09:40:00 | 1897.85 | 2024-07-25 15:10:00 | 1891.26 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-19 11:10:00 | 1805.25 | 2024-08-19 12:25:00 | 1810.80 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-19 11:10:00 | 1805.25 | 2024-08-19 15:20:00 | 1820.45 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-09-10 10:25:00 | 1994.45 | 2024-09-10 10:45:00 | 2002.62 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-10 10:25:00 | 1994.45 | 2024-09-10 11:30:00 | 1994.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-13 09:35:00 | 2098.85 | 2024-09-13 09:40:00 | 2086.71 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-13 09:35:00 | 2098.85 | 2024-09-13 09:50:00 | 2098.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:30:00 | 2082.85 | 2024-09-19 09:35:00 | 2070.53 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-09-23 10:05:00 | 2038.55 | 2024-09-23 10:15:00 | 2026.84 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-23 10:05:00 | 2038.55 | 2024-09-23 11:15:00 | 2038.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 10:55:00 | 2014.95 | 2024-09-24 11:05:00 | 2023.06 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-25 11:05:00 | 2008.45 | 2024-09-25 11:15:00 | 2002.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-27 10:50:00 | 1927.95 | 2024-09-27 11:20:00 | 1916.03 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-09-27 10:50:00 | 1927.95 | 2024-09-27 15:20:00 | 1895.35 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2024-10-07 10:10:00 | 1859.90 | 2024-10-07 10:25:00 | 1847.86 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-07 10:10:00 | 1859.90 | 2024-10-07 11:45:00 | 1852.95 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-10-10 10:55:00 | 1889.05 | 2024-10-10 11:10:00 | 1885.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-10-15 09:35:00 | 1880.80 | 2024-10-15 10:00:00 | 1874.30 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-16 09:30:00 | 1858.90 | 2024-10-16 10:15:00 | 1864.25 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-21 10:35:00 | 1810.55 | 2024-10-21 11:30:00 | 1802.13 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-21 10:35:00 | 1810.55 | 2024-10-21 11:35:00 | 1810.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 10:05:00 | 1851.75 | 2024-10-31 10:20:00 | 1843.22 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-11-06 11:05:00 | 1880.05 | 2024-11-06 11:15:00 | 1891.16 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-11-06 11:05:00 | 1880.05 | 2024-11-06 15:20:00 | 1910.40 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1864.50 | 2024-12-02 09:55:00 | 1871.80 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1864.50 | 2024-12-02 10:10:00 | 1864.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:55:00 | 2003.95 | 2024-12-20 11:50:00 | 1994.53 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-20 10:55:00 | 2003.95 | 2024-12-20 15:20:00 | 1920.80 | TARGET_HIT | 0.50 | 4.15% |
| BUY | retest1 | 2024-12-24 09:45:00 | 1924.90 | 2024-12-24 09:55:00 | 1935.90 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-12-24 09:45:00 | 1924.90 | 2024-12-24 10:35:00 | 1935.35 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1829.35 | 2025-01-02 10:05:00 | 1822.66 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-08 10:35:00 | 1806.85 | 2025-01-08 11:15:00 | 1796.83 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-08 10:35:00 | 1806.85 | 2025-01-08 11:30:00 | 1806.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-13 10:00:00 | 1717.60 | 2025-01-13 10:10:00 | 1727.53 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-01-15 09:35:00 | 1708.85 | 2025-01-15 10:05:00 | 1716.65 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-20 10:20:00 | 1770.90 | 2025-01-20 10:55:00 | 1762.98 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-23 09:40:00 | 1739.05 | 2025-01-23 10:10:00 | 1730.56 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1371.75 | 2025-01-30 10:30:00 | 1382.80 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1371.75 | 2025-01-30 11:40:00 | 1371.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 10:55:00 | 1504.80 | 2025-02-07 11:30:00 | 1513.59 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-02-07 10:55:00 | 1504.80 | 2025-02-07 12:15:00 | 1504.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 10:00:00 | 1468.75 | 2025-02-11 10:30:00 | 1475.57 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-02-13 09:35:00 | 1456.25 | 2025-02-13 09:40:00 | 1468.65 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2025-02-13 09:35:00 | 1456.25 | 2025-02-13 13:00:00 | 1467.00 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2025-02-19 09:30:00 | 1446.95 | 2025-02-19 09:45:00 | 1453.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-02-27 09:55:00 | 1325.00 | 2025-02-27 10:00:00 | 1316.96 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-02-27 09:55:00 | 1325.00 | 2025-02-27 10:05:00 | 1325.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-06 10:20:00 | 1297.80 | 2025-03-06 10:25:00 | 1306.03 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-06 10:20:00 | 1297.80 | 2025-03-06 10:50:00 | 1297.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-17 10:00:00 | 1213.45 | 2025-03-17 10:15:00 | 1208.36 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-21 09:35:00 | 1210.80 | 2025-04-21 09:45:00 | 1206.34 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-05-05 09:35:00 | 1213.50 | 2025-05-05 09:45:00 | 1221.06 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-05-05 09:35:00 | 1213.50 | 2025-05-05 10:05:00 | 1213.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:35:00 | 1195.80 | 2025-05-08 09:40:00 | 1191.63 | STOP_HIT | 1.00 | -0.35% |
