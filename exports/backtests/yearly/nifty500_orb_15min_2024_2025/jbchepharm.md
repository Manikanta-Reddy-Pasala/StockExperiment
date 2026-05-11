# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-03-30 15:25:00 (34996 bars)
- **Last close:** 2065.00
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 8 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 59
- **Target hits / Stop hits / Partials:** 8 / 59 / 29
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 6.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 21 | 38.9% | 4 | 33 | 17 | 0.07% | 3.5% |
| BUY @ 2nd Alert (retest1) | 54 | 21 | 38.9% | 4 | 33 | 17 | 0.07% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 16 | 38.1% | 4 | 26 | 12 | 0.08% | 3.3% |
| SELL @ 2nd Alert (retest1) | 42 | 16 | 38.1% | 4 | 26 | 12 | 0.08% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 37 | 38.5% | 8 | 59 | 29 | 0.07% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 1810.65 | 1803.31 | 0.00 | ORB-long ORB[1792.10,1804.70] vol=3.1x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-05-15 09:45:00 | 1804.59 | 1804.10 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1712.05 | 1690.45 | 0.00 | ORB-long ORB[1670.05,1683.45] vol=5.9x ATR=7.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:40:00 | 1722.76 | 1697.57 | 0.00 | T1 1.5R @ 1722.76 |
| Stop hit — per-position SL triggered | 2024-05-27 09:45:00 | 1712.05 | 1700.07 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 1760.10 | 1761.23 | 0.00 | ORB-short ORB[1761.10,1773.05] vol=16.1x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:25:00 | 1753.31 | 1761.21 | 0.00 | T1 1.5R @ 1753.31 |
| Target hit | 2024-05-30 14:05:00 | 1739.05 | 1735.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2024-06-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:40:00 | 1796.00 | 1791.64 | 0.00 | ORB-long ORB[1774.80,1795.35] vol=2.2x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:45:00 | 1807.34 | 1795.98 | 0.00 | T1 1.5R @ 1807.34 |
| Stop hit — per-position SL triggered | 2024-06-07 10:05:00 | 1796.00 | 1799.24 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 11:00:00 | 1875.00 | 1884.09 | 0.00 | ORB-short ORB[1886.70,1903.35] vol=3.9x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:25:00 | 1866.46 | 1881.05 | 0.00 | T1 1.5R @ 1866.46 |
| Stop hit — per-position SL triggered | 2024-06-14 12:15:00 | 1875.00 | 1877.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 11:00:00 | 1799.95 | 1803.30 | 0.00 | ORB-short ORB[1805.00,1824.50] vol=3.8x ATR=5.91 |
| Stop hit — per-position SL triggered | 2024-06-19 11:40:00 | 1805.86 | 1803.51 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:15:00 | 1739.35 | 1740.72 | 0.00 | ORB-short ORB[1741.00,1760.50] vol=13.6x ATR=6.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:10:00 | 1729.11 | 1740.08 | 0.00 | T1 1.5R @ 1729.11 |
| Target hit | 2024-06-26 15:20:00 | 1724.00 | 1731.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 1744.05 | 1744.99 | 0.00 | ORB-short ORB[1744.20,1765.00] vol=1.7x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-07-08 12:00:00 | 1748.95 | 1745.41 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:00:00 | 1722.10 | 1732.51 | 0.00 | ORB-short ORB[1735.00,1749.00] vol=2.8x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-07-09 11:05:00 | 1726.00 | 1732.32 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 1702.95 | 1715.39 | 0.00 | ORB-short ORB[1729.05,1740.00] vol=5.6x ATR=6.01 |
| Stop hit — per-position SL triggered | 2024-07-10 11:00:00 | 1708.96 | 1712.36 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:15:00 | 1759.25 | 1745.77 | 0.00 | ORB-long ORB[1730.00,1751.85] vol=4.5x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:25:00 | 1767.08 | 1751.23 | 0.00 | T1 1.5R @ 1767.08 |
| Stop hit — per-position SL triggered | 2024-07-12 11:50:00 | 1759.25 | 1754.71 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 1729.20 | 1738.47 | 0.00 | ORB-short ORB[1737.80,1749.60] vol=2.2x ATR=6.86 |
| Stop hit — per-position SL triggered | 2024-07-15 09:40:00 | 1736.06 | 1737.27 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1790.05 | 1797.81 | 0.00 | ORB-short ORB[1792.50,1807.15] vol=1.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:35:00 | 1783.80 | 1790.96 | 0.00 | T1 1.5R @ 1783.80 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1790.05 | 1790.86 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 09:40:00 | 1807.95 | 1799.91 | 0.00 | ORB-long ORB[1785.05,1802.10] vol=2.0x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:10:00 | 1817.81 | 1805.20 | 0.00 | T1 1.5R @ 1817.81 |
| Stop hit — per-position SL triggered | 2024-07-22 10:20:00 | 1807.95 | 1806.29 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 1885.50 | 1879.38 | 0.00 | ORB-long ORB[1862.75,1879.75] vol=4.0x ATR=4.97 |
| Stop hit — per-position SL triggered | 2024-07-26 11:10:00 | 1880.53 | 1879.41 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:00:00 | 1902.65 | 1896.97 | 0.00 | ORB-long ORB[1880.05,1901.55] vol=8.2x ATR=6.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 10:20:00 | 1912.88 | 1901.70 | 0.00 | T1 1.5R @ 1912.88 |
| Target hit | 2024-07-29 11:05:00 | 1922.95 | 1924.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2024-08-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:40:00 | 1903.05 | 1909.27 | 0.00 | ORB-short ORB[1904.10,1924.60] vol=3.7x ATR=5.56 |
| Stop hit — per-position SL triggered | 2024-08-01 11:00:00 | 1908.61 | 1908.97 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:45:00 | 1900.35 | 1909.35 | 0.00 | ORB-short ORB[1904.70,1931.10] vol=3.0x ATR=8.95 |
| Stop hit — per-position SL triggered | 2024-08-06 11:00:00 | 1909.30 | 1909.05 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:50:00 | 1970.00 | 1952.50 | 0.00 | ORB-long ORB[1933.95,1960.00] vol=4.1x ATR=9.14 |
| Stop hit — per-position SL triggered | 2024-08-13 09:55:00 | 1960.86 | 1954.75 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 1964.50 | 1952.25 | 0.00 | ORB-long ORB[1928.85,1950.00] vol=4.1x ATR=7.45 |
| Stop hit — per-position SL triggered | 2024-08-21 09:40:00 | 1957.05 | 1953.81 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 1942.20 | 1946.30 | 0.00 | ORB-short ORB[1950.00,1961.15] vol=3.2x ATR=6.78 |
| Stop hit — per-position SL triggered | 2024-08-30 09:45:00 | 1948.98 | 1946.17 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 1916.15 | 1921.06 | 0.00 | ORB-short ORB[1921.65,1934.95] vol=2.7x ATR=6.60 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 1922.75 | 1922.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1949.90 | 1938.32 | 0.00 | ORB-long ORB[1921.10,1945.95] vol=1.7x ATR=8.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:35:00 | 1963.05 | 1946.44 | 0.00 | T1 1.5R @ 1963.05 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 1949.90 | 1952.14 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 1918.85 | 1903.60 | 0.00 | ORB-long ORB[1892.85,1914.10] vol=1.9x ATR=5.91 |
| Stop hit — per-position SL triggered | 2024-09-13 10:50:00 | 1912.94 | 1906.48 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1877.00 | 1870.89 | 0.00 | ORB-long ORB[1856.05,1874.45] vol=2.0x ATR=8.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:40:00 | 1889.89 | 1877.16 | 0.00 | T1 1.5R @ 1889.89 |
| Stop hit — per-position SL triggered | 2024-09-19 11:00:00 | 1877.00 | 1880.88 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:10:00 | 1912.05 | 1911.21 | 0.00 | ORB-long ORB[1885.00,1909.95] vol=1.9x ATR=8.12 |
| Stop hit — per-position SL triggered | 2024-09-26 13:10:00 | 1903.93 | 1913.42 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:45:00 | 1876.50 | 1867.80 | 0.00 | ORB-long ORB[1849.90,1875.00] vol=1.6x ATR=10.14 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 1866.36 | 1870.81 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:35:00 | 1836.65 | 1853.71 | 0.00 | ORB-short ORB[1872.00,1894.90] vol=1.7x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-10-01 10:50:00 | 1841.83 | 1852.18 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:15:00 | 1702.00 | 1713.83 | 0.00 | ORB-short ORB[1707.10,1725.10] vol=1.8x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:55:00 | 1688.06 | 1708.98 | 0.00 | T1 1.5R @ 1688.06 |
| Target hit | 2024-10-07 14:50:00 | 1692.15 | 1685.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 1730.30 | 1707.03 | 0.00 | ORB-long ORB[1682.10,1705.00] vol=2.7x ATR=6.46 |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 1723.84 | 1708.12 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:05:00 | 1786.10 | 1775.32 | 0.00 | ORB-long ORB[1742.85,1768.55] vol=3.1x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-10-09 10:10:00 | 1778.32 | 1776.82 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:15:00 | 1803.20 | 1806.81 | 0.00 | ORB-short ORB[1816.75,1835.35] vol=6.7x ATR=6.28 |
| Stop hit — per-position SL triggered | 2024-10-11 10:20:00 | 1809.48 | 1807.38 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:30:00 | 1798.50 | 1813.74 | 0.00 | ORB-short ORB[1803.20,1823.00] vol=2.2x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-10-14 10:40:00 | 1804.65 | 1812.13 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:55:00 | 1906.95 | 1912.83 | 0.00 | ORB-short ORB[1909.50,1923.45] vol=6.0x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:10:00 | 1897.79 | 1911.72 | 0.00 | T1 1.5R @ 1897.79 |
| Stop hit — per-position SL triggered | 2024-10-22 14:45:00 | 1906.95 | 1894.25 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 1893.20 | 1904.60 | 0.00 | ORB-short ORB[1897.20,1917.75] vol=1.7x ATR=7.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1882.36 | 1894.95 | 0.00 | T1 1.5R @ 1882.36 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 1893.20 | 1894.80 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:45:00 | 1910.00 | 1888.34 | 0.00 | ORB-long ORB[1865.05,1892.80] vol=6.6x ATR=8.36 |
| Stop hit — per-position SL triggered | 2024-10-28 11:20:00 | 1901.64 | 1892.34 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:10:00 | 1857.05 | 1862.59 | 0.00 | ORB-short ORB[1859.20,1886.45] vol=3.8x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 12:20:00 | 1848.75 | 1859.52 | 0.00 | T1 1.5R @ 1848.75 |
| Stop hit — per-position SL triggered | 2024-10-29 14:10:00 | 1857.05 | 1852.67 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 1749.30 | 1758.95 | 0.00 | ORB-short ORB[1753.60,1775.00] vol=1.7x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:50:00 | 1739.45 | 1750.50 | 0.00 | T1 1.5R @ 1739.45 |
| Stop hit — per-position SL triggered | 2024-11-13 10:30:00 | 1749.30 | 1747.76 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 10:30:00 | 1667.10 | 1680.96 | 0.00 | ORB-short ORB[1668.05,1692.50] vol=1.7x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-11-18 10:40:00 | 1674.83 | 1680.60 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:55:00 | 1698.20 | 1684.73 | 0.00 | ORB-long ORB[1670.60,1685.00] vol=1.7x ATR=6.26 |
| Stop hit — per-position SL triggered | 2024-11-19 10:05:00 | 1691.94 | 1686.16 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:00:00 | 1689.05 | 1674.71 | 0.00 | ORB-long ORB[1658.05,1677.85] vol=1.9x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 12:00:00 | 1698.55 | 1679.27 | 0.00 | T1 1.5R @ 1698.55 |
| Target hit | 2024-11-21 15:20:00 | 1705.50 | 1694.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-11-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:45:00 | 1813.20 | 1791.89 | 0.00 | ORB-long ORB[1768.45,1793.15] vol=1.6x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-11-25 09:50:00 | 1804.58 | 1796.76 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:05:00 | 1771.00 | 1756.53 | 0.00 | ORB-long ORB[1740.95,1754.00] vol=4.3x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-11-26 10:20:00 | 1764.42 | 1759.51 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 1729.95 | 1729.16 | 0.00 | ORB-long ORB[1716.55,1726.45] vol=1.8x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:20:00 | 1736.48 | 1730.22 | 0.00 | T1 1.5R @ 1736.48 |
| Stop hit — per-position SL triggered | 2024-11-29 13:05:00 | 1729.95 | 1733.60 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:40:00 | 1750.00 | 1748.94 | 0.00 | ORB-long ORB[1734.20,1747.85] vol=1.8x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 10:05:00 | 1758.16 | 1751.20 | 0.00 | T1 1.5R @ 1758.16 |
| Stop hit — per-position SL triggered | 2024-12-02 10:40:00 | 1750.00 | 1752.41 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 1815.50 | 1807.29 | 0.00 | ORB-long ORB[1780.00,1806.00] vol=2.5x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:50:00 | 1823.22 | 1810.23 | 0.00 | T1 1.5R @ 1823.22 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 1815.50 | 1810.04 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1791.20 | 1816.91 | 0.00 | ORB-short ORB[1812.00,1836.45] vol=2.2x ATR=6.83 |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 1798.03 | 1811.86 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:55:00 | 1851.75 | 1845.52 | 0.00 | ORB-long ORB[1829.20,1850.00] vol=1.6x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:15:00 | 1858.02 | 1847.05 | 0.00 | T1 1.5R @ 1858.02 |
| Stop hit — per-position SL triggered | 2024-12-18 11:20:00 | 1851.75 | 1847.04 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:00:00 | 1857.85 | 1852.54 | 0.00 | ORB-long ORB[1838.00,1847.60] vol=12.9x ATR=7.32 |
| Stop hit — per-position SL triggered | 2025-01-01 12:25:00 | 1850.53 | 1854.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 1885.00 | 1878.76 | 0.00 | ORB-long ORB[1852.15,1877.95] vol=15.8x ATR=7.38 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 1877.62 | 1878.95 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:40:00 | 1849.70 | 1847.09 | 0.00 | ORB-long ORB[1829.90,1844.95] vol=18.3x ATR=7.31 |
| Stop hit — per-position SL triggered | 2025-01-07 10:45:00 | 1842.39 | 1847.05 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:55:00 | 1876.50 | 1887.24 | 0.00 | ORB-short ORB[1884.65,1908.75] vol=1.6x ATR=7.74 |
| Stop hit — per-position SL triggered | 2025-01-08 10:00:00 | 1884.24 | 1886.50 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:40:00 | 1844.75 | 1855.91 | 0.00 | ORB-short ORB[1853.35,1871.65] vol=2.2x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:45:00 | 1835.41 | 1848.21 | 0.00 | T1 1.5R @ 1835.41 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 1844.75 | 1844.18 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:35:00 | 1768.65 | 1786.73 | 0.00 | ORB-short ORB[1770.05,1788.50] vol=4.9x ATR=6.20 |
| Stop hit — per-position SL triggered | 2025-01-17 11:30:00 | 1774.85 | 1786.35 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:35:00 | 1759.75 | 1755.67 | 0.00 | ORB-long ORB[1739.05,1759.15] vol=1.6x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:30:00 | 1768.61 | 1760.32 | 0.00 | T1 1.5R @ 1768.61 |
| Target hit | 2025-01-20 11:40:00 | 1760.65 | 1760.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 1833.40 | 1821.33 | 0.00 | ORB-long ORB[1795.15,1811.90] vol=3.3x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:05:00 | 1843.84 | 1832.59 | 0.00 | T1 1.5R @ 1843.84 |
| Stop hit — per-position SL triggered | 2025-01-21 10:10:00 | 1833.40 | 1837.17 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 1779.05 | 1787.17 | 0.00 | ORB-short ORB[1782.00,1806.95] vol=1.8x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-01-24 09:55:00 | 1785.41 | 1784.08 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 1785.80 | 1777.53 | 0.00 | ORB-long ORB[1765.00,1775.30] vol=1.7x ATR=8.54 |
| Stop hit — per-position SL triggered | 2025-02-01 09:35:00 | 1777.26 | 1775.72 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 1618.65 | 1627.73 | 0.00 | ORB-short ORB[1620.95,1639.95] vol=1.8x ATR=7.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 13:10:00 | 1607.05 | 1612.10 | 0.00 | T1 1.5R @ 1607.05 |
| Target hit | 2025-02-21 15:05:00 | 1609.80 | 1607.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2025-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 09:45:00 | 1514.15 | 1518.98 | 0.00 | ORB-short ORB[1515.20,1532.00] vol=6.4x ATR=6.31 |
| Stop hit — per-position SL triggered | 2025-03-18 09:50:00 | 1520.46 | 1519.07 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:40:00 | 1623.60 | 1618.16 | 0.00 | ORB-long ORB[1604.55,1615.30] vol=3.8x ATR=6.76 |
| Stop hit — per-position SL triggered | 2025-03-20 09:50:00 | 1616.84 | 1618.08 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 1644.40 | 1636.05 | 0.00 | ORB-long ORB[1620.05,1640.00] vol=1.9x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:55:00 | 1652.65 | 1642.19 | 0.00 | T1 1.5R @ 1652.65 |
| Stop hit — per-position SL triggered | 2025-03-21 10:05:00 | 1644.40 | 1643.03 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 11:05:00 | 1527.80 | 1518.23 | 0.00 | ORB-long ORB[1510.00,1525.65] vol=2.0x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 12:15:00 | 1532.98 | 1520.52 | 0.00 | T1 1.5R @ 1532.98 |
| Target hit | 2025-04-09 15:20:00 | 1539.95 | 1529.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:00:00 | 1611.00 | 1598.96 | 0.00 | ORB-long ORB[1580.80,1601.00] vol=1.6x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:15:00 | 1619.09 | 1603.52 | 0.00 | T1 1.5R @ 1619.09 |
| Stop hit — per-position SL triggered | 2025-04-16 10:20:00 | 1611.00 | 1603.70 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:00:00 | 1606.00 | 1585.30 | 0.00 | ORB-long ORB[1573.40,1597.20] vol=3.7x ATR=6.43 |
| Stop hit — per-position SL triggered | 2025-04-24 11:05:00 | 1599.57 | 1585.69 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:20:00 | 1590.70 | 1593.06 | 0.00 | ORB-short ORB[1604.00,1622.50] vol=3.6x ATR=6.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:25:00 | 1581.17 | 1592.63 | 0.00 | T1 1.5R @ 1581.17 |
| Stop hit — per-position SL triggered | 2025-04-25 11:10:00 | 1590.70 | 1588.15 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 1557.60 | 1549.52 | 0.00 | ORB-long ORB[1537.40,1555.00] vol=1.8x ATR=7.03 |
| Stop hit — per-position SL triggered | 2025-05-08 10:55:00 | 1550.57 | 1552.97 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:35:00 | 1810.65 | 2024-05-15 09:45:00 | 1804.59 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1712.05 | 2024-05-27 09:40:00 | 1722.76 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1712.05 | 2024-05-27 09:45:00 | 1712.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1760.10 | 2024-05-30 10:25:00 | 1753.31 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1760.10 | 2024-05-30 14:05:00 | 1739.05 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2024-06-07 09:40:00 | 1796.00 | 2024-06-07 09:45:00 | 1807.34 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-07 09:40:00 | 1796.00 | 2024-06-07 10:05:00 | 1796.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 11:00:00 | 1875.00 | 2024-06-14 11:25:00 | 1866.46 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-06-14 11:00:00 | 1875.00 | 2024-06-14 12:15:00 | 1875.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 11:00:00 | 1799.95 | 2024-06-19 11:40:00 | 1805.86 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-26 10:15:00 | 1739.35 | 2024-06-26 11:10:00 | 1729.11 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-06-26 10:15:00 | 1739.35 | 2024-06-26 15:20:00 | 1724.00 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2024-07-08 10:50:00 | 1744.05 | 2024-07-08 12:00:00 | 1748.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-09 11:00:00 | 1722.10 | 2024-07-09 11:05:00 | 1726.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-10 10:20:00 | 1702.95 | 2024-07-10 11:00:00 | 1708.96 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-12 11:15:00 | 1759.25 | 2024-07-12 11:25:00 | 1767.08 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-12 11:15:00 | 1759.25 | 2024-07-12 11:50:00 | 1759.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 09:35:00 | 1729.20 | 2024-07-15 09:40:00 | 1736.06 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1790.05 | 2024-07-18 09:35:00 | 1783.80 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1790.05 | 2024-07-18 09:40:00 | 1790.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-22 09:40:00 | 1807.95 | 2024-07-22 10:10:00 | 1817.81 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-07-22 09:40:00 | 1807.95 | 2024-07-22 10:20:00 | 1807.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 11:05:00 | 1885.50 | 2024-07-26 11:10:00 | 1880.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-29 10:00:00 | 1902.65 | 2024-07-29 10:20:00 | 1912.88 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-29 10:00:00 | 1902.65 | 2024-07-29 11:05:00 | 1922.95 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2024-08-01 10:40:00 | 1903.05 | 2024-08-01 11:00:00 | 1908.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-06 10:45:00 | 1900.35 | 2024-08-06 11:00:00 | 1909.30 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-13 09:50:00 | 1970.00 | 2024-08-13 09:55:00 | 1960.86 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-08-21 09:30:00 | 1964.50 | 2024-08-21 09:40:00 | 1957.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-30 09:35:00 | 1942.20 | 2024-08-30 09:45:00 | 1948.98 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-09 09:30:00 | 1916.15 | 2024-09-09 09:35:00 | 1922.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1949.90 | 2024-09-11 09:35:00 | 1963.05 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1949.90 | 2024-09-11 09:55:00 | 1949.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:45:00 | 1918.85 | 2024-09-13 10:50:00 | 1912.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-19 10:15:00 | 1877.00 | 2024-09-19 10:40:00 | 1889.89 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-09-19 10:15:00 | 1877.00 | 2024-09-19 11:00:00 | 1877.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:10:00 | 1912.05 | 2024-09-26 13:10:00 | 1903.93 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-30 09:45:00 | 1876.50 | 2024-09-30 09:55:00 | 1866.36 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-10-01 10:35:00 | 1836.65 | 2024-10-01 10:50:00 | 1841.83 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-07 10:15:00 | 1702.00 | 2024-10-07 10:55:00 | 1688.06 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-10-07 10:15:00 | 1702.00 | 2024-10-07 14:50:00 | 1692.15 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2024-10-08 11:05:00 | 1730.30 | 2024-10-08 11:15:00 | 1723.84 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-09 10:05:00 | 1786.10 | 2024-10-09 10:10:00 | 1778.32 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-11 10:15:00 | 1803.20 | 2024-10-11 10:20:00 | 1809.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-14 10:30:00 | 1798.50 | 2024-10-14 10:40:00 | 1804.65 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-22 09:55:00 | 1906.95 | 2024-10-22 10:10:00 | 1897.79 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-22 09:55:00 | 1906.95 | 2024-10-22 14:45:00 | 1906.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 09:30:00 | 1893.20 | 2024-10-25 10:15:00 | 1882.36 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-10-25 09:30:00 | 1893.20 | 2024-10-25 10:20:00 | 1893.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-28 10:45:00 | 1910.00 | 2024-10-28 11:20:00 | 1901.64 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-29 11:10:00 | 1857.05 | 2024-10-29 12:20:00 | 1848.75 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-29 11:10:00 | 1857.05 | 2024-10-29 14:10:00 | 1857.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1749.30 | 2024-11-13 09:50:00 | 1739.45 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1749.30 | 2024-11-13 10:30:00 | 1749.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-18 10:30:00 | 1667.10 | 2024-11-18 10:40:00 | 1674.83 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-11-19 09:55:00 | 1698.20 | 2024-11-19 10:05:00 | 1691.94 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-21 11:00:00 | 1689.05 | 2024-11-21 12:00:00 | 1698.55 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-21 11:00:00 | 1689.05 | 2024-11-21 15:20:00 | 1705.50 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-11-25 09:45:00 | 1813.20 | 2024-11-25 09:50:00 | 1804.58 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-11-26 10:05:00 | 1771.00 | 2024-11-26 10:20:00 | 1764.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-29 10:55:00 | 1729.95 | 2024-11-29 11:20:00 | 1736.48 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-11-29 10:55:00 | 1729.95 | 2024-11-29 13:05:00 | 1729.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 09:40:00 | 1750.00 | 2024-12-02 10:05:00 | 1758.16 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-02 09:40:00 | 1750.00 | 2024-12-02 10:40:00 | 1750.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 10:55:00 | 1815.50 | 2024-12-04 11:50:00 | 1823.22 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-04 10:55:00 | 1815.50 | 2024-12-04 11:55:00 | 1815.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1791.20 | 2024-12-05 12:15:00 | 1798.03 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-18 10:55:00 | 1851.75 | 2024-12-18 11:15:00 | 1858.02 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-18 10:55:00 | 1851.75 | 2024-12-18 11:20:00 | 1851.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:00:00 | 1857.85 | 2025-01-01 12:25:00 | 1850.53 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-03 09:55:00 | 1885.00 | 2025-01-03 10:15:00 | 1877.62 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-07 10:40:00 | 1849.70 | 2025-01-07 10:45:00 | 1842.39 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-08 09:55:00 | 1876.50 | 2025-01-08 10:00:00 | 1884.24 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-10 09:40:00 | 1844.75 | 2025-01-10 09:45:00 | 1835.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-10 09:40:00 | 1844.75 | 2025-01-10 10:05:00 | 1844.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 10:35:00 | 1768.65 | 2025-01-17 11:30:00 | 1774.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-20 09:35:00 | 1759.75 | 2025-01-20 10:30:00 | 1768.61 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-20 09:35:00 | 1759.75 | 2025-01-20 11:40:00 | 1760.65 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-01-21 09:40:00 | 1833.40 | 2025-01-21 10:05:00 | 1843.84 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-01-21 09:40:00 | 1833.40 | 2025-01-21 10:10:00 | 1833.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:35:00 | 1779.05 | 2025-01-24 09:55:00 | 1785.41 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-02-01 09:30:00 | 1785.80 | 2025-02-01 09:35:00 | 1777.26 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-21 09:40:00 | 1618.65 | 2025-02-21 13:10:00 | 1607.05 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-02-21 09:40:00 | 1618.65 | 2025-02-21 15:05:00 | 1609.80 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-03-18 09:45:00 | 1514.15 | 2025-03-18 09:50:00 | 1520.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-20 09:40:00 | 1623.60 | 2025-03-20 09:50:00 | 1616.84 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-21 09:40:00 | 1644.40 | 2025-03-21 09:55:00 | 1652.65 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-21 09:40:00 | 1644.40 | 2025-03-21 10:05:00 | 1644.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-09 11:05:00 | 1527.80 | 2025-04-09 12:15:00 | 1532.98 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-04-09 11:05:00 | 1527.80 | 2025-04-09 15:20:00 | 1539.95 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2025-04-16 10:00:00 | 1611.00 | 2025-04-16 10:15:00 | 1619.09 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-16 10:00:00 | 1611.00 | 2025-04-16 10:20:00 | 1611.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 11:00:00 | 1606.00 | 2025-04-24 11:05:00 | 1599.57 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-25 10:20:00 | 1590.70 | 2025-04-25 10:25:00 | 1581.17 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-04-25 10:20:00 | 1590.70 | 2025-04-25 11:10:00 | 1590.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:35:00 | 1557.60 | 2025-05-08 10:55:00 | 1550.57 | STOP_HIT | 1.00 | -0.45% |
