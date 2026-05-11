# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 5391.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 19 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 70
- **Target hits / Stop hits / Partials:** 19 / 70 / 41
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 18.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 33 | 47.1% | 10 | 37 | 23 | 0.15% | 10.5% |
| BUY @ 2nd Alert (retest1) | 70 | 33 | 47.1% | 10 | 37 | 23 | 0.15% | 10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 27 | 45.0% | 9 | 33 | 18 | 0.13% | 7.7% |
| SELL @ 2nd Alert (retest1) | 60 | 27 | 45.0% | 9 | 33 | 18 | 0.13% | 7.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 60 | 46.2% | 19 | 70 | 41 | 0.14% | 18.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:30:00 | 1637.05 | 1631.31 | 0.00 | ORB-long ORB[1620.90,1634.65] vol=1.6x ATR=4.69 |
| Stop hit — per-position SL triggered | 2023-05-12 10:55:00 | 1632.36 | 1631.84 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 10:35:00 | 1647.70 | 1650.63 | 0.00 | ORB-short ORB[1649.00,1666.00] vol=1.8x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 11:10:00 | 1642.53 | 1648.42 | 0.00 | T1 1.5R @ 1642.53 |
| Target hit | 2023-05-16 12:20:00 | 1642.80 | 1641.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2023-05-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:40:00 | 1647.10 | 1650.60 | 0.00 | ORB-short ORB[1648.55,1657.85] vol=2.1x ATR=4.00 |
| Stop hit — per-position SL triggered | 2023-05-17 10:45:00 | 1651.10 | 1648.61 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:30:00 | 1696.00 | 1685.13 | 0.00 | ORB-long ORB[1675.40,1688.00] vol=2.2x ATR=4.57 |
| Stop hit — per-position SL triggered | 2023-05-24 09:40:00 | 1691.43 | 1689.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 11:10:00 | 1744.30 | 1752.89 | 0.00 | ORB-short ORB[1751.85,1761.90] vol=2.5x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:40:00 | 1738.70 | 1751.44 | 0.00 | T1 1.5R @ 1738.70 |
| Stop hit — per-position SL triggered | 2023-05-31 12:15:00 | 1744.30 | 1750.31 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 10:15:00 | 1761.80 | 1755.39 | 0.00 | ORB-long ORB[1747.45,1758.75] vol=1.9x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-06-02 10:20:00 | 1758.06 | 1755.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:10:00 | 1783.55 | 1774.97 | 0.00 | ORB-long ORB[1754.95,1779.40] vol=2.0x ATR=5.61 |
| Stop hit — per-position SL triggered | 2023-06-05 10:15:00 | 1777.94 | 1775.41 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 10:00:00 | 1775.50 | 1784.17 | 0.00 | ORB-short ORB[1786.75,1797.95] vol=2.2x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-06-08 10:10:00 | 1779.10 | 1783.34 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:35:00 | 1801.55 | 1797.54 | 0.00 | ORB-long ORB[1786.00,1798.20] vol=1.8x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 09:50:00 | 1808.77 | 1803.17 | 0.00 | T1 1.5R @ 1808.77 |
| Target hit | 2023-06-13 10:25:00 | 1811.15 | 1813.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2023-06-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 09:40:00 | 1841.65 | 1846.20 | 0.00 | ORB-short ORB[1845.15,1855.00] vol=2.4x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-06-15 10:05:00 | 1845.45 | 1844.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:50:00 | 1896.70 | 1889.79 | 0.00 | ORB-long ORB[1879.50,1894.00] vol=2.1x ATR=4.87 |
| Stop hit — per-position SL triggered | 2023-06-20 10:05:00 | 1891.83 | 1890.10 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:30:00 | 1884.20 | 1890.55 | 0.00 | ORB-short ORB[1890.65,1908.00] vol=1.6x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:45:00 | 1878.81 | 1888.55 | 0.00 | T1 1.5R @ 1878.81 |
| Target hit | 2023-06-21 15:20:00 | 1858.40 | 1868.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 11:15:00 | 1888.20 | 1877.17 | 0.00 | ORB-long ORB[1856.35,1877.00] vol=3.7x ATR=3.19 |
| Stop hit — per-position SL triggered | 2023-06-27 11:25:00 | 1885.01 | 1877.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:55:00 | 1877.55 | 1881.16 | 0.00 | ORB-short ORB[1877.85,1893.45] vol=1.5x ATR=4.52 |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 1882.07 | 1879.98 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 1920.65 | 1913.83 | 0.00 | ORB-long ORB[1894.50,1917.35] vol=2.7x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 09:55:00 | 1929.56 | 1917.54 | 0.00 | T1 1.5R @ 1929.56 |
| Stop hit — per-position SL triggered | 2023-07-04 10:00:00 | 1920.65 | 1917.85 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:45:00 | 1898.50 | 1909.97 | 0.00 | ORB-short ORB[1900.20,1915.95] vol=3.1x ATR=4.99 |
| Stop hit — per-position SL triggered | 2023-07-07 11:10:00 | 1903.49 | 1907.97 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:55:00 | 1905.85 | 1902.16 | 0.00 | ORB-long ORB[1885.05,1903.10] vol=2.8x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:05:00 | 1913.16 | 1903.46 | 0.00 | T1 1.5R @ 1913.16 |
| Target hit | 2023-07-11 12:00:00 | 1912.40 | 1912.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2023-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:45:00 | 1924.00 | 1913.09 | 0.00 | ORB-long ORB[1895.00,1922.00] vol=1.8x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:10:00 | 1933.69 | 1919.00 | 0.00 | T1 1.5R @ 1933.69 |
| Stop hit — per-position SL triggered | 2023-07-12 11:10:00 | 1924.00 | 1924.92 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 11:05:00 | 1898.75 | 1909.04 | 0.00 | ORB-short ORB[1900.00,1915.00] vol=1.8x ATR=4.30 |
| Stop hit — per-position SL triggered | 2023-07-14 12:35:00 | 1903.05 | 1904.26 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 09:45:00 | 1911.65 | 1920.37 | 0.00 | ORB-short ORB[1918.35,1933.75] vol=1.5x ATR=4.90 |
| Stop hit — per-position SL triggered | 2023-07-17 09:50:00 | 1916.55 | 1920.07 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 10:20:00 | 1928.75 | 1917.52 | 0.00 | ORB-long ORB[1901.25,1915.75] vol=2.3x ATR=4.83 |
| Stop hit — per-position SL triggered | 2023-07-18 10:30:00 | 1923.92 | 1918.96 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:00:00 | 1911.60 | 1918.64 | 0.00 | ORB-short ORB[1916.20,1929.00] vol=1.7x ATR=4.04 |
| Stop hit — per-position SL triggered | 2023-07-19 11:25:00 | 1915.64 | 1917.58 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 10:45:00 | 1905.35 | 1922.64 | 0.00 | ORB-short ORB[1925.55,1949.85] vol=7.5x ATR=5.92 |
| Stop hit — per-position SL triggered | 2023-07-20 10:50:00 | 1911.27 | 1920.42 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 09:40:00 | 1891.10 | 1898.16 | 0.00 | ORB-short ORB[1896.80,1920.00] vol=1.7x ATR=4.68 |
| Stop hit — per-position SL triggered | 2023-07-21 09:50:00 | 1895.78 | 1896.93 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:25:00 | 1903.50 | 1908.14 | 0.00 | ORB-short ORB[1903.70,1922.00] vol=1.5x ATR=3.89 |
| Stop hit — per-position SL triggered | 2023-07-25 10:35:00 | 1907.39 | 1907.99 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:35:00 | 1947.00 | 1934.72 | 0.00 | ORB-long ORB[1909.05,1932.55] vol=2.3x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:45:00 | 1955.17 | 1939.35 | 0.00 | T1 1.5R @ 1955.17 |
| Stop hit — per-position SL triggered | 2023-07-28 10:55:00 | 1947.00 | 1942.01 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 11:10:00 | 1978.90 | 1968.25 | 0.00 | ORB-long ORB[1960.70,1974.80] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2023-08-01 11:15:00 | 1975.24 | 1971.18 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 11:05:00 | 1905.00 | 1900.80 | 0.00 | ORB-long ORB[1886.05,1903.80] vol=2.3x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 11:25:00 | 1912.31 | 1902.32 | 0.00 | T1 1.5R @ 1912.31 |
| Stop hit — per-position SL triggered | 2023-08-03 11:50:00 | 1905.00 | 1903.38 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 11:15:00 | 1762.55 | 1750.10 | 0.00 | ORB-long ORB[1742.30,1760.75] vol=1.8x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 11:55:00 | 1770.53 | 1754.13 | 0.00 | T1 1.5R @ 1770.53 |
| Stop hit — per-position SL triggered | 2023-08-10 12:00:00 | 1762.55 | 1754.58 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:20:00 | 1763.15 | 1747.45 | 0.00 | ORB-long ORB[1731.60,1751.45] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-08-17 10:25:00 | 1758.37 | 1748.71 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:15:00 | 1748.25 | 1751.78 | 0.00 | ORB-short ORB[1749.15,1760.00] vol=1.6x ATR=3.93 |
| Stop hit — per-position SL triggered | 2023-08-22 12:45:00 | 1752.18 | 1751.39 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 10:40:00 | 1735.05 | 1727.77 | 0.00 | ORB-long ORB[1712.35,1735.00] vol=1.9x ATR=4.62 |
| Stop hit — per-position SL triggered | 2023-08-28 11:10:00 | 1730.43 | 1729.83 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:05:00 | 1685.30 | 1692.55 | 0.00 | ORB-short ORB[1688.90,1700.60] vol=1.8x ATR=5.40 |
| Stop hit — per-position SL triggered | 2023-09-01 11:15:00 | 1690.70 | 1691.89 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:30:00 | 1716.55 | 1709.61 | 0.00 | ORB-long ORB[1705.40,1714.65] vol=1.6x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 11:20:00 | 1722.06 | 1711.76 | 0.00 | T1 1.5R @ 1722.06 |
| Target hit | 2023-09-04 14:50:00 | 1721.50 | 1724.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2023-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:45:00 | 1740.60 | 1735.98 | 0.00 | ORB-long ORB[1726.30,1735.00] vol=1.7x ATR=4.58 |
| Stop hit — per-position SL triggered | 2023-09-05 10:00:00 | 1736.02 | 1737.33 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:55:00 | 1731.10 | 1729.71 | 0.00 | ORB-long ORB[1722.10,1730.00] vol=2.9x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 14:15:00 | 1736.53 | 1730.57 | 0.00 | T1 1.5R @ 1736.53 |
| Target hit | 2023-09-06 15:20:00 | 1749.95 | 1734.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2023-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 10:45:00 | 1729.25 | 1735.54 | 0.00 | ORB-short ORB[1761.00,1776.25] vol=2.6x ATR=6.18 |
| Stop hit — per-position SL triggered | 2023-09-12 11:00:00 | 1735.43 | 1735.04 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 10:35:00 | 1722.50 | 1729.89 | 0.00 | ORB-short ORB[1731.05,1741.30] vol=1.9x ATR=3.53 |
| Stop hit — per-position SL triggered | 2023-09-15 11:35:00 | 1726.03 | 1727.06 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 10:35:00 | 1739.70 | 1735.30 | 0.00 | ORB-long ORB[1719.20,1737.10] vol=1.5x ATR=4.17 |
| Stop hit — per-position SL triggered | 2023-09-21 10:45:00 | 1735.53 | 1735.48 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 11:15:00 | 1725.55 | 1730.30 | 0.00 | ORB-short ORB[1735.00,1756.85] vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2023-09-22 12:05:00 | 1728.90 | 1727.89 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:05:00 | 1691.00 | 1705.89 | 0.00 | ORB-short ORB[1709.00,1729.90] vol=2.1x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 10:10:00 | 1684.16 | 1699.24 | 0.00 | T1 1.5R @ 1684.16 |
| Stop hit — per-position SL triggered | 2023-09-25 10:50:00 | 1691.00 | 1691.30 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:30:00 | 1730.75 | 1732.36 | 0.00 | ORB-short ORB[1731.25,1740.00] vol=5.3x ATR=5.47 |
| Stop hit — per-position SL triggered | 2023-09-26 10:50:00 | 1736.22 | 1732.16 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 11:00:00 | 1707.85 | 1715.21 | 0.00 | ORB-short ORB[1708.75,1730.00] vol=1.7x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 11:10:00 | 1702.98 | 1714.27 | 0.00 | T1 1.5R @ 1702.98 |
| Target hit | 2023-09-28 15:20:00 | 1675.65 | 1687.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2023-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:40:00 | 1699.95 | 1686.66 | 0.00 | ORB-long ORB[1672.55,1687.50] vol=1.6x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 10:45:00 | 1708.23 | 1696.54 | 0.00 | T1 1.5R @ 1708.23 |
| Stop hit — per-position SL triggered | 2023-10-04 10:55:00 | 1699.95 | 1697.14 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:25:00 | 1690.05 | 1696.08 | 0.00 | ORB-short ORB[1692.75,1703.00] vol=2.4x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:40:00 | 1684.95 | 1695.47 | 0.00 | T1 1.5R @ 1684.95 |
| Stop hit — per-position SL triggered | 2023-10-05 10:50:00 | 1690.05 | 1695.25 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 11:00:00 | 1700.00 | 1699.06 | 0.00 | ORB-long ORB[1688.85,1698.00] vol=13.9x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 11:10:00 | 1703.83 | 1699.14 | 0.00 | T1 1.5R @ 1703.83 |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 1700.00 | 1699.27 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 10:50:00 | 1705.30 | 1700.17 | 0.00 | ORB-long ORB[1692.50,1702.35] vol=1.9x ATR=3.43 |
| Stop hit — per-position SL triggered | 2023-10-10 10:55:00 | 1701.87 | 1700.58 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:40:00 | 1747.80 | 1734.07 | 0.00 | ORB-long ORB[1713.50,1736.25] vol=2.4x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:05:00 | 1754.68 | 1742.24 | 0.00 | T1 1.5R @ 1754.68 |
| Stop hit — per-position SL triggered | 2023-10-11 10:25:00 | 1747.80 | 1743.36 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:40:00 | 1731.20 | 1738.12 | 0.00 | ORB-short ORB[1741.45,1758.90] vol=1.7x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 11:10:00 | 1725.85 | 1736.41 | 0.00 | T1 1.5R @ 1725.85 |
| Target hit | 2023-10-12 14:50:00 | 1729.00 | 1728.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — SELL (started 2023-10-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:35:00 | 1708.65 | 1715.87 | 0.00 | ORB-short ORB[1713.05,1726.40] vol=1.7x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 10:45:00 | 1703.77 | 1713.56 | 0.00 | T1 1.5R @ 1703.77 |
| Stop hit — per-position SL triggered | 2023-10-13 10:55:00 | 1708.65 | 1709.20 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 10:50:00 | 1709.90 | 1715.84 | 0.00 | ORB-short ORB[1713.00,1730.00] vol=4.4x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 11:30:00 | 1705.19 | 1712.55 | 0.00 | T1 1.5R @ 1705.19 |
| Stop hit — per-position SL triggered | 2023-10-17 11:55:00 | 1709.90 | 1711.63 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-10-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:35:00 | 1723.85 | 1718.03 | 0.00 | ORB-long ORB[1702.80,1718.65] vol=1.6x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:40:00 | 1728.75 | 1718.49 | 0.00 | T1 1.5R @ 1728.75 |
| Stop hit — per-position SL triggered | 2023-10-18 10:45:00 | 1723.85 | 1718.68 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 10:15:00 | 1717.10 | 1702.91 | 0.00 | ORB-long ORB[1690.50,1710.95] vol=2.0x ATR=6.21 |
| Stop hit — per-position SL triggered | 2023-10-25 10:40:00 | 1710.89 | 1705.95 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-10-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:55:00 | 1670.80 | 1674.94 | 0.00 | ORB-short ORB[1678.00,1692.75] vol=1.9x ATR=4.62 |
| Stop hit — per-position SL triggered | 2023-10-26 10:40:00 | 1675.42 | 1673.54 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 11:15:00 | 1693.75 | 1684.10 | 0.00 | ORB-long ORB[1672.80,1686.00] vol=1.7x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:25:00 | 1698.57 | 1686.07 | 0.00 | T1 1.5R @ 1698.57 |
| Target hit | 2023-11-02 15:20:00 | 1705.55 | 1696.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2023-11-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:10:00 | 1725.00 | 1723.14 | 0.00 | ORB-long ORB[1711.15,1724.70] vol=5.3x ATR=3.25 |
| Stop hit — per-position SL triggered | 2023-11-07 11:25:00 | 1721.75 | 1723.09 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 11:10:00 | 1750.00 | 1769.47 | 0.00 | ORB-short ORB[1768.85,1784.25] vol=2.0x ATR=5.19 |
| Stop hit — per-position SL triggered | 2023-11-09 15:20:00 | 1750.40 | 1759.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 1753.05 | 1761.44 | 0.00 | ORB-short ORB[1757.80,1775.00] vol=1.5x ATR=6.09 |
| Stop hit — per-position SL triggered | 2023-11-13 10:30:00 | 1759.14 | 1761.19 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:55:00 | 1808.70 | 1795.46 | 0.00 | ORB-long ORB[1780.00,1790.00] vol=2.6x ATR=5.12 |
| Stop hit — per-position SL triggered | 2023-11-15 10:00:00 | 1803.58 | 1798.58 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:00:00 | 1841.55 | 1834.63 | 0.00 | ORB-long ORB[1822.35,1834.95] vol=2.1x ATR=4.75 |
| Stop hit — per-position SL triggered | 2023-11-16 10:55:00 | 1836.80 | 1837.09 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-11-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:45:00 | 1819.60 | 1826.17 | 0.00 | ORB-short ORB[1825.05,1835.00] vol=2.4x ATR=4.66 |
| Stop hit — per-position SL triggered | 2023-11-20 09:50:00 | 1824.26 | 1825.99 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-11-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:45:00 | 1864.00 | 1859.97 | 0.00 | ORB-long ORB[1847.40,1858.10] vol=2.0x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 10:00:00 | 1869.89 | 1861.70 | 0.00 | T1 1.5R @ 1869.89 |
| Target hit | 2023-11-23 15:20:00 | 1875.15 | 1868.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2023-11-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:45:00 | 1858.80 | 1863.52 | 0.00 | ORB-short ORB[1864.80,1875.00] vol=2.0x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 10:20:00 | 1852.56 | 1860.72 | 0.00 | T1 1.5R @ 1852.56 |
| Stop hit — per-position SL triggered | 2023-11-24 10:35:00 | 1858.80 | 1860.49 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-11-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:25:00 | 1892.55 | 1884.08 | 0.00 | ORB-long ORB[1875.10,1889.25] vol=1.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:30:00 | 1898.88 | 1887.27 | 0.00 | T1 1.5R @ 1898.88 |
| Stop hit — per-position SL triggered | 2023-11-29 11:00:00 | 1892.55 | 1890.27 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 10:50:00 | 1954.50 | 1938.71 | 0.00 | ORB-long ORB[1925.80,1946.30] vol=2.3x ATR=5.78 |
| Stop hit — per-position SL triggered | 2023-12-04 11:00:00 | 1948.72 | 1940.04 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:45:00 | 1980.50 | 1969.09 | 0.00 | ORB-long ORB[1953.90,1968.80] vol=1.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2023-12-05 10:05:00 | 1974.17 | 1972.86 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-07 10:50:00 | 1932.05 | 1937.49 | 0.00 | ORB-short ORB[1936.05,1959.00] vol=1.6x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 11:25:00 | 1926.22 | 1935.28 | 0.00 | T1 1.5R @ 1926.22 |
| Stop hit — per-position SL triggered | 2023-12-07 11:50:00 | 1932.05 | 1933.81 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:50:00 | 1946.80 | 1957.01 | 0.00 | ORB-short ORB[1955.10,1967.20] vol=3.6x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:05:00 | 1940.85 | 1955.29 | 0.00 | T1 1.5R @ 1940.85 |
| Stop hit — per-position SL triggered | 2023-12-08 12:45:00 | 1946.80 | 1944.70 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:05:00 | 1950.45 | 1941.33 | 0.00 | ORB-long ORB[1931.00,1945.30] vol=1.7x ATR=5.04 |
| Stop hit — per-position SL triggered | 2023-12-18 11:30:00 | 1945.41 | 1941.83 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:50:00 | 2002.45 | 1991.65 | 0.00 | ORB-long ORB[1973.80,1995.00] vol=2.5x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 11:05:00 | 2010.97 | 1998.59 | 0.00 | T1 1.5R @ 2010.97 |
| Stop hit — per-position SL triggered | 2023-12-20 11:20:00 | 2002.45 | 1999.96 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:50:00 | 1959.50 | 1944.70 | 0.00 | ORB-long ORB[1927.50,1949.00] vol=1.8x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:25:00 | 1970.23 | 1953.50 | 0.00 | T1 1.5R @ 1970.23 |
| Target hit | 2023-12-22 15:20:00 | 2007.95 | 1993.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2023-12-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:35:00 | 1987.85 | 1998.40 | 0.00 | ORB-short ORB[1990.00,2007.60] vol=1.9x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 11:35:00 | 1979.76 | 1992.61 | 0.00 | T1 1.5R @ 1979.76 |
| Target hit | 2023-12-28 15:20:00 | 1940.25 | 1951.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:15:00 | 1953.15 | 1962.67 | 0.00 | ORB-short ORB[1959.05,1968.00] vol=1.8x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-01-01 12:40:00 | 1958.65 | 1954.70 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 1946.00 | 1947.14 | 0.00 | ORB-short ORB[1946.25,1958.95] vol=4.1x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:00:00 | 1939.44 | 1946.07 | 0.00 | T1 1.5R @ 1939.44 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 1946.00 | 1946.13 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:20:00 | 1994.55 | 1996.37 | 0.00 | ORB-short ORB[1996.10,2011.95] vol=1.6x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 11:05:00 | 1984.63 | 1993.64 | 0.00 | T1 1.5R @ 1984.63 |
| Target hit | 2024-01-08 11:45:00 | 1988.85 | 1988.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — BUY (started 2024-01-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:20:00 | 2040.50 | 2027.78 | 0.00 | ORB-long ORB[2008.00,2031.80] vol=2.6x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:50:00 | 2051.86 | 2034.15 | 0.00 | T1 1.5R @ 2051.86 |
| Target hit | 2024-01-09 12:10:00 | 2045.50 | 2045.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:15:00 | 2080.00 | 2073.01 | 0.00 | ORB-long ORB[2054.65,2079.00] vol=4.0x ATR=7.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 11:00:00 | 2091.26 | 2076.49 | 0.00 | T1 1.5R @ 2091.26 |
| Stop hit — per-position SL triggered | 2024-01-15 11:45:00 | 2080.00 | 2079.93 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 10:20:00 | 2378.50 | 2394.76 | 0.00 | ORB-short ORB[2389.95,2421.85] vol=2.8x ATR=11.46 |
| Stop hit — per-position SL triggered | 2024-02-07 10:45:00 | 2389.96 | 2391.45 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:50:00 | 2638.15 | 2658.34 | 0.00 | ORB-short ORB[2651.10,2679.50] vol=3.7x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 14:35:00 | 2625.47 | 2647.25 | 0.00 | T1 1.5R @ 2625.47 |
| Target hit | 2024-02-20 15:20:00 | 2630.10 | 2644.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:15:00 | 2638.50 | 2627.33 | 0.00 | ORB-long ORB[2610.50,2628.45] vol=2.1x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-02-21 10:25:00 | 2632.29 | 2629.18 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 10:50:00 | 2811.50 | 2786.33 | 0.00 | ORB-long ORB[2752.10,2783.95] vol=1.9x ATR=11.46 |
| Stop hit — per-position SL triggered | 2024-02-23 11:25:00 | 2800.04 | 2797.55 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:00:00 | 2721.95 | 2745.34 | 0.00 | ORB-short ORB[2745.00,2784.00] vol=1.8x ATR=8.89 |
| Stop hit — per-position SL triggered | 2024-03-06 10:05:00 | 2730.84 | 2743.32 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:35:00 | 2813.25 | 2811.12 | 0.00 | ORB-long ORB[2790.05,2811.40] vol=1.9x ATR=8.05 |
| Stop hit — per-position SL triggered | 2024-03-07 10:40:00 | 2805.20 | 2810.62 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:35:00 | 2710.60 | 2717.32 | 0.00 | ORB-short ORB[2713.00,2733.20] vol=2.3x ATR=11.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 09:50:00 | 2693.18 | 2711.95 | 0.00 | T1 1.5R @ 2693.18 |
| Target hit | 2024-03-19 12:30:00 | 2702.00 | 2698.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 85 — BUY (started 2024-03-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 09:45:00 | 2997.90 | 2983.94 | 0.00 | ORB-long ORB[2968.00,2989.45] vol=2.1x ATR=9.56 |
| Stop hit — per-position SL triggered | 2024-03-27 11:35:00 | 2988.34 | 2993.19 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:20:00 | 2999.95 | 3013.27 | 0.00 | ORB-short ORB[3010.60,3044.85] vol=1.9x ATR=8.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 10:30:00 | 2987.18 | 3010.99 | 0.00 | T1 1.5R @ 2987.18 |
| Target hit | 2024-04-08 12:20:00 | 2987.50 | 2981.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 87 — BUY (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 3087.90 | 3055.27 | 0.00 | ORB-long ORB[3013.50,3042.60] vol=1.8x ATR=7.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 11:30:00 | 3099.82 | 3065.62 | 0.00 | T1 1.5R @ 3099.82 |
| Target hit | 2024-04-16 15:20:00 | 3124.70 | 3102.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2024-04-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 11:05:00 | 3131.65 | 3112.02 | 0.00 | ORB-long ORB[3105.55,3122.75] vol=3.1x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 11:35:00 | 3144.98 | 3117.38 | 0.00 | T1 1.5R @ 3144.98 |
| Stop hit — per-position SL triggered | 2024-04-18 11:50:00 | 3131.65 | 3119.64 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:05:00 | 3159.00 | 3145.94 | 0.00 | ORB-long ORB[3118.00,3154.95] vol=2.0x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:10:00 | 3175.47 | 3150.78 | 0.00 | T1 1.5R @ 3175.47 |
| Target hit | 2024-04-23 15:00:00 | 3162.35 | 3164.97 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:30:00 | 1637.05 | 2023-05-12 10:55:00 | 1632.36 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-05-16 10:35:00 | 1647.70 | 2023-05-16 11:10:00 | 1642.53 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-05-16 10:35:00 | 1647.70 | 2023-05-16 12:20:00 | 1642.80 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2023-05-17 10:40:00 | 1647.10 | 2023-05-17 10:45:00 | 1651.10 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-24 09:30:00 | 1696.00 | 2023-05-24 09:40:00 | 1691.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-31 11:10:00 | 1744.30 | 2023-05-31 11:40:00 | 1738.70 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-05-31 11:10:00 | 1744.30 | 2023-05-31 12:15:00 | 1744.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-02 10:15:00 | 1761.80 | 2023-06-02 10:20:00 | 1758.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-05 10:10:00 | 1783.55 | 2023-06-05 10:15:00 | 1777.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-06-08 10:00:00 | 1775.50 | 2023-06-08 10:10:00 | 1779.10 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-13 09:35:00 | 1801.55 | 2023-06-13 09:50:00 | 1808.77 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-06-13 09:35:00 | 1801.55 | 2023-06-13 10:25:00 | 1811.15 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2023-06-15 09:40:00 | 1841.65 | 2023-06-15 10:05:00 | 1845.45 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-20 09:50:00 | 1896.70 | 2023-06-20 10:05:00 | 1891.83 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-06-21 10:30:00 | 1884.20 | 2023-06-21 10:45:00 | 1878.81 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-06-21 10:30:00 | 1884.20 | 2023-06-21 15:20:00 | 1858.40 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2023-06-27 11:15:00 | 1888.20 | 2023-06-27 11:25:00 | 1885.01 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-28 09:55:00 | 1877.55 | 2023-06-28 11:15:00 | 1882.07 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-04 09:40:00 | 1920.65 | 2023-07-04 09:55:00 | 1929.56 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-07-04 09:40:00 | 1920.65 | 2023-07-04 10:00:00 | 1920.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-07 10:45:00 | 1898.50 | 2023-07-07 11:10:00 | 1903.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-11 09:55:00 | 1905.85 | 2023-07-11 10:05:00 | 1913.16 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-11 09:55:00 | 1905.85 | 2023-07-11 12:00:00 | 1912.40 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-12 09:45:00 | 1924.00 | 2023-07-12 10:10:00 | 1933.69 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-07-12 09:45:00 | 1924.00 | 2023-07-12 11:10:00 | 1924.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-14 11:05:00 | 1898.75 | 2023-07-14 12:35:00 | 1903.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-17 09:45:00 | 1911.65 | 2023-07-17 09:50:00 | 1916.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-18 10:20:00 | 1928.75 | 2023-07-18 10:30:00 | 1923.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-19 11:00:00 | 1911.60 | 2023-07-19 11:25:00 | 1915.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-07-20 10:45:00 | 1905.35 | 2023-07-20 10:50:00 | 1911.27 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-07-21 09:40:00 | 1891.10 | 2023-07-21 09:50:00 | 1895.78 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-25 10:25:00 | 1903.50 | 2023-07-25 10:35:00 | 1907.39 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-28 10:35:00 | 1947.00 | 2023-07-28 10:45:00 | 1955.17 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-07-28 10:35:00 | 1947.00 | 2023-07-28 10:55:00 | 1947.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-01 11:10:00 | 1978.90 | 2023-08-01 11:15:00 | 1975.24 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-08-03 11:05:00 | 1905.00 | 2023-08-03 11:25:00 | 1912.31 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-03 11:05:00 | 1905.00 | 2023-08-03 11:50:00 | 1905.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 11:15:00 | 1762.55 | 2023-08-10 11:55:00 | 1770.53 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-08-10 11:15:00 | 1762.55 | 2023-08-10 12:00:00 | 1762.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 10:20:00 | 1763.15 | 2023-08-17 10:25:00 | 1758.37 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-22 11:15:00 | 1748.25 | 2023-08-22 12:45:00 | 1752.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-28 10:40:00 | 1735.05 | 2023-08-28 11:10:00 | 1730.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-01 11:05:00 | 1685.30 | 2023-09-01 11:15:00 | 1690.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-04 10:30:00 | 1716.55 | 2023-09-04 11:20:00 | 1722.06 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-04 10:30:00 | 1716.55 | 2023-09-04 14:50:00 | 1721.50 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-05 09:45:00 | 1740.60 | 2023-09-05 10:00:00 | 1736.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-06 09:55:00 | 1731.10 | 2023-09-06 14:15:00 | 1736.53 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-09-06 09:55:00 | 1731.10 | 2023-09-06 15:20:00 | 1749.95 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2023-09-12 10:45:00 | 1729.25 | 2023-09-12 11:00:00 | 1735.43 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-09-15 10:35:00 | 1722.50 | 2023-09-15 11:35:00 | 1726.03 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-21 10:35:00 | 1739.70 | 2023-09-21 10:45:00 | 1735.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-09-22 11:15:00 | 1725.55 | 2023-09-22 12:05:00 | 1728.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-09-25 10:05:00 | 1691.00 | 2023-09-25 10:10:00 | 1684.16 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-09-25 10:05:00 | 1691.00 | 2023-09-25 10:50:00 | 1691.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-26 10:30:00 | 1730.75 | 2023-09-26 10:50:00 | 1736.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-28 11:00:00 | 1707.85 | 2023-09-28 11:10:00 | 1702.98 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-09-28 11:00:00 | 1707.85 | 2023-09-28 15:20:00 | 1675.65 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2023-10-04 09:40:00 | 1699.95 | 2023-10-04 10:45:00 | 1708.23 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-10-04 09:40:00 | 1699.95 | 2023-10-04 10:55:00 | 1699.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 10:25:00 | 1690.05 | 2023-10-05 10:40:00 | 1684.95 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-10-05 10:25:00 | 1690.05 | 2023-10-05 10:50:00 | 1690.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-06 11:00:00 | 1700.00 | 2023-10-06 11:10:00 | 1703.83 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-10-06 11:00:00 | 1700.00 | 2023-10-06 11:15:00 | 1700.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-10 10:50:00 | 1705.30 | 2023-10-10 10:55:00 | 1701.87 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-11 09:40:00 | 1747.80 | 2023-10-11 10:05:00 | 1754.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-10-11 09:40:00 | 1747.80 | 2023-10-11 10:25:00 | 1747.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 10:40:00 | 1731.20 | 2023-10-12 11:10:00 | 1725.85 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-10-12 10:40:00 | 1731.20 | 2023-10-12 14:50:00 | 1729.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2023-10-13 10:35:00 | 1708.65 | 2023-10-13 10:45:00 | 1703.77 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-10-13 10:35:00 | 1708.65 | 2023-10-13 10:55:00 | 1708.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-17 10:50:00 | 1709.90 | 2023-10-17 11:30:00 | 1705.19 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-10-17 10:50:00 | 1709.90 | 2023-10-17 11:55:00 | 1709.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 10:35:00 | 1723.85 | 2023-10-18 10:40:00 | 1728.75 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-10-18 10:35:00 | 1723.85 | 2023-10-18 10:45:00 | 1723.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-25 10:15:00 | 1717.10 | 2023-10-25 10:40:00 | 1710.89 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-10-26 09:55:00 | 1670.80 | 2023-10-26 10:40:00 | 1675.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-02 11:15:00 | 1693.75 | 2023-11-02 11:25:00 | 1698.57 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-02 11:15:00 | 1693.75 | 2023-11-02 15:20:00 | 1705.55 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2023-11-07 11:10:00 | 1725.00 | 2023-11-07 11:25:00 | 1721.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-09 11:10:00 | 1750.00 | 2023-11-09 15:20:00 | 1750.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest1 | 2023-11-13 10:20:00 | 1753.05 | 2023-11-13 10:30:00 | 1759.14 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-15 09:55:00 | 1808.70 | 2023-11-15 10:00:00 | 1803.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-16 10:00:00 | 1841.55 | 2023-11-16 10:55:00 | 1836.80 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-20 09:45:00 | 1819.60 | 2023-11-20 09:50:00 | 1824.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-23 09:45:00 | 1864.00 | 2023-11-23 10:00:00 | 1869.89 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-11-23 09:45:00 | 1864.00 | 2023-11-23 15:20:00 | 1875.15 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2023-11-24 09:45:00 | 1858.80 | 2023-11-24 10:20:00 | 1852.56 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-11-24 09:45:00 | 1858.80 | 2023-11-24 10:35:00 | 1858.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 10:25:00 | 1892.55 | 2023-11-29 10:30:00 | 1898.88 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-29 10:25:00 | 1892.55 | 2023-11-29 11:00:00 | 1892.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-04 10:50:00 | 1954.50 | 2023-12-04 11:00:00 | 1948.72 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-05 09:45:00 | 1980.50 | 2023-12-05 10:05:00 | 1974.17 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-12-07 10:50:00 | 1932.05 | 2023-12-07 11:25:00 | 1926.22 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-07 10:50:00 | 1932.05 | 2023-12-07 11:50:00 | 1932.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 10:50:00 | 1946.80 | 2023-12-08 11:05:00 | 1940.85 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-12-08 10:50:00 | 1946.80 | 2023-12-08 12:45:00 | 1946.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-18 11:05:00 | 1950.45 | 2023-12-18 11:30:00 | 1945.41 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-20 10:50:00 | 2002.45 | 2023-12-20 11:05:00 | 2010.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-12-20 10:50:00 | 2002.45 | 2023-12-20 11:20:00 | 2002.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 09:50:00 | 1959.50 | 2023-12-22 10:25:00 | 1970.23 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-12-22 09:50:00 | 1959.50 | 2023-12-22 15:20:00 | 2007.95 | TARGET_HIT | 0.50 | 2.47% |
| SELL | retest1 | 2023-12-28 10:35:00 | 1987.85 | 2023-12-28 11:35:00 | 1979.76 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-12-28 10:35:00 | 1987.85 | 2023-12-28 15:20:00 | 1940.25 | TARGET_HIT | 0.50 | 2.39% |
| SELL | retest1 | 2024-01-01 10:15:00 | 1953.15 | 2024-01-01 12:40:00 | 1958.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-02 09:55:00 | 1946.00 | 2024-01-02 10:00:00 | 1939.44 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-01-02 09:55:00 | 1946.00 | 2024-01-02 10:05:00 | 1946.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 10:20:00 | 1994.55 | 2024-01-08 11:05:00 | 1984.63 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-08 10:20:00 | 1994.55 | 2024-01-08 11:45:00 | 1988.85 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-01-09 10:20:00 | 2040.50 | 2024-01-09 10:50:00 | 2051.86 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-09 10:20:00 | 2040.50 | 2024-01-09 12:10:00 | 2045.50 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2024-01-15 10:15:00 | 2080.00 | 2024-01-15 11:00:00 | 2091.26 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-01-15 10:15:00 | 2080.00 | 2024-01-15 11:45:00 | 2080.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-07 10:20:00 | 2378.50 | 2024-02-07 10:45:00 | 2389.96 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-02-20 10:50:00 | 2638.15 | 2024-02-20 14:35:00 | 2625.47 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-20 10:50:00 | 2638.15 | 2024-02-20 15:20:00 | 2630.10 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-02-21 10:15:00 | 2638.50 | 2024-02-21 10:25:00 | 2632.29 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-23 10:50:00 | 2811.50 | 2024-02-23 11:25:00 | 2800.04 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-06 10:00:00 | 2721.95 | 2024-03-06 10:05:00 | 2730.84 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-07 10:35:00 | 2813.25 | 2024-03-07 10:40:00 | 2805.20 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-19 09:35:00 | 2710.60 | 2024-03-19 09:50:00 | 2693.18 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-03-19 09:35:00 | 2710.60 | 2024-03-19 12:30:00 | 2702.00 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-27 09:45:00 | 2997.90 | 2024-03-27 11:35:00 | 2988.34 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-04-08 10:20:00 | 2999.95 | 2024-04-08 10:30:00 | 2987.18 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-04-08 10:20:00 | 2999.95 | 2024-04-08 12:20:00 | 2987.50 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-16 11:05:00 | 3087.90 | 2024-04-16 11:30:00 | 3099.82 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-04-16 11:05:00 | 3087.90 | 2024-04-16 15:20:00 | 3124.70 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2024-04-18 11:05:00 | 3131.65 | 2024-04-18 11:35:00 | 3144.98 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-04-18 11:05:00 | 3131.65 | 2024-04-18 11:50:00 | 3131.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 10:05:00 | 3159.00 | 2024-04-23 10:10:00 | 3175.47 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-04-23 10:05:00 | 3159.00 | 2024-04-23 15:00:00 | 3162.35 | TARGET_HIT | 0.50 | 0.11% |
