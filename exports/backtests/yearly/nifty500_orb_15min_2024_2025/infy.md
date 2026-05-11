# Infosys Ltd. (INFY)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1179.50
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 52
- **Target hits / Stop hits / Partials:** 11 / 52 / 21
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 8.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 19 | 40.4% | 7 | 28 | 12 | 0.15% | 7.2% |
| BUY @ 2nd Alert (retest1) | 47 | 19 | 40.4% | 7 | 28 | 12 | 0.15% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 13 | 35.1% | 4 | 24 | 9 | 0.03% | 1.0% |
| SELL @ 2nd Alert (retest1) | 37 | 13 | 35.1% | 4 | 24 | 9 | 0.03% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 84 | 32 | 38.1% | 11 | 52 | 21 | 0.10% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:45:00 | 1419.30 | 1422.04 | 0.00 | ORB-short ORB[1422.50,1428.70] vol=2.4x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-05-14 11:05:00 | 1421.12 | 1421.69 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 1464.70 | 1457.75 | 0.00 | ORB-long ORB[1450.00,1458.80] vol=1.9x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 1461.94 | 1459.96 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:05:00 | 1436.80 | 1438.86 | 0.00 | ORB-short ORB[1438.60,1449.85] vol=2.0x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-05-30 12:05:00 | 1439.06 | 1438.13 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:05:00 | 1420.10 | 1410.22 | 0.00 | ORB-long ORB[1405.10,1415.45] vol=1.8x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 12:10:00 | 1426.07 | 1414.60 | 0.00 | T1 1.5R @ 1426.07 |
| Stop hit — per-position SL triggered | 2024-05-31 15:00:00 | 1420.10 | 1420.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 11:15:00 | 1454.45 | 1447.10 | 0.00 | ORB-long ORB[1439.95,1453.00] vol=1.8x ATR=3.47 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 1450.98 | 1447.56 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 1505.05 | 1494.85 | 0.00 | ORB-long ORB[1477.25,1498.40] vol=1.8x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:45:00 | 1512.56 | 1500.02 | 0.00 | T1 1.5R @ 1512.56 |
| Target hit | 2024-06-07 15:20:00 | 1534.20 | 1523.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 1495.00 | 1500.95 | 0.00 | ORB-short ORB[1501.95,1508.75] vol=3.7x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-06-12 12:25:00 | 1497.54 | 1499.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 11:10:00 | 1501.75 | 1502.64 | 0.00 | ORB-short ORB[1502.30,1507.00] vol=2.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-06-19 11:40:00 | 1503.82 | 1502.60 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:40:00 | 1534.05 | 1527.30 | 0.00 | ORB-long ORB[1520.35,1529.00] vol=1.7x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 12:35:00 | 1537.50 | 1532.04 | 0.00 | T1 1.5R @ 1537.50 |
| Stop hit — per-position SL triggered | 2024-06-25 13:20:00 | 1534.05 | 1532.92 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:50:00 | 1549.40 | 1541.47 | 0.00 | ORB-long ORB[1533.25,1541.80] vol=2.5x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-06-27 10:55:00 | 1546.24 | 1542.14 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:45:00 | 1582.50 | 1573.43 | 0.00 | ORB-long ORB[1559.50,1579.30] vol=1.7x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:10:00 | 1587.98 | 1576.96 | 0.00 | T1 1.5R @ 1587.98 |
| Target hit | 2024-07-01 15:20:00 | 1589.00 | 1588.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 1655.00 | 1645.27 | 0.00 | ORB-long ORB[1628.00,1650.80] vol=2.1x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-07-04 09:35:00 | 1650.69 | 1646.07 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 11:05:00 | 1661.45 | 1652.75 | 0.00 | ORB-long ORB[1640.00,1657.80] vol=1.6x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 1657.49 | 1653.03 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 1693.45 | 1681.08 | 0.00 | ORB-long ORB[1666.65,1681.00] vol=1.7x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:00:00 | 1700.88 | 1684.83 | 0.00 | T1 1.5R @ 1700.88 |
| Target hit | 2024-07-12 15:20:00 | 1710.10 | 1701.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:50:00 | 1731.15 | 1715.59 | 0.00 | ORB-long ORB[1700.00,1720.00] vol=1.5x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-07-16 10:10:00 | 1725.87 | 1720.91 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:45:00 | 1818.55 | 1806.39 | 0.00 | ORB-long ORB[1792.95,1812.15] vol=2.8x ATR=5.04 |
| Stop hit — per-position SL triggered | 2024-07-22 12:25:00 | 1813.51 | 1811.12 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 11:00:00 | 1872.00 | 1876.96 | 0.00 | ORB-short ORB[1872.15,1885.15] vol=1.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:10:00 | 1867.26 | 1876.37 | 0.00 | T1 1.5R @ 1867.26 |
| Target hit | 2024-07-31 15:20:00 | 1868.85 | 1870.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-08-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 10:40:00 | 1755.65 | 1779.94 | 0.00 | ORB-short ORB[1783.00,1796.40] vol=1.6x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:50:00 | 1746.74 | 1774.12 | 0.00 | T1 1.5R @ 1746.74 |
| Target hit | 2024-08-05 13:25:00 | 1747.20 | 1746.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2024-08-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:10:00 | 1791.05 | 1776.30 | 0.00 | ORB-long ORB[1769.00,1782.00] vol=1.6x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:20:00 | 1796.94 | 1778.31 | 0.00 | T1 1.5R @ 1796.94 |
| Stop hit — per-position SL triggered | 2024-08-12 15:00:00 | 1791.05 | 1791.60 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:55:00 | 1896.30 | 1886.88 | 0.00 | ORB-long ORB[1873.90,1887.90] vol=2.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-08-27 10:35:00 | 1892.64 | 1891.14 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:35:00 | 1928.30 | 1911.68 | 0.00 | ORB-long ORB[1902.20,1911.00] vol=2.0x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:50:00 | 1934.69 | 1915.67 | 0.00 | T1 1.5R @ 1934.69 |
| Target hit | 2024-08-28 15:20:00 | 1939.05 | 1935.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 1947.00 | 1940.17 | 0.00 | ORB-long ORB[1925.00,1937.05] vol=1.6x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-08-29 11:20:00 | 1943.65 | 1940.38 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 1910.60 | 1919.05 | 0.00 | ORB-short ORB[1921.05,1933.30] vol=2.1x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-09-05 11:20:00 | 1914.18 | 1918.31 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:05:00 | 1917.60 | 1907.59 | 0.00 | ORB-long ORB[1889.00,1911.50] vol=1.5x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-09-09 12:55:00 | 1912.97 | 1910.24 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:55:00 | 1924.50 | 1911.28 | 0.00 | ORB-long ORB[1900.00,1917.00] vol=2.8x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-09-10 11:05:00 | 1920.17 | 1912.78 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 1951.30 | 1948.16 | 0.00 | ORB-long ORB[1942.15,1949.90] vol=1.8x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-09-17 09:45:00 | 1947.94 | 1948.08 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 1904.00 | 1917.12 | 0.00 | ORB-short ORB[1914.00,1938.75] vol=3.7x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:30:00 | 1895.66 | 1906.65 | 0.00 | T1 1.5R @ 1895.66 |
| Stop hit — per-position SL triggered | 2024-09-18 12:45:00 | 1904.00 | 1903.25 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:00:00 | 1900.00 | 1912.36 | 0.00 | ORB-short ORB[1905.30,1924.90] vol=1.8x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-09-19 10:05:00 | 1906.21 | 1911.89 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:55:00 | 1890.25 | 1898.56 | 0.00 | ORB-short ORB[1898.30,1918.90] vol=2.3x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:20:00 | 1883.78 | 1896.30 | 0.00 | T1 1.5R @ 1883.78 |
| Stop hit — per-position SL triggered | 2024-09-23 11:55:00 | 1890.25 | 1892.17 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 10:10:00 | 1904.45 | 1901.51 | 0.00 | ORB-long ORB[1883.35,1903.60] vol=1.6x ATR=5.14 |
| Stop hit — per-position SL triggered | 2024-10-03 10:25:00 | 1899.31 | 1901.49 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:50:00 | 1931.25 | 1908.98 | 0.00 | ORB-long ORB[1886.00,1903.00] vol=2.0x ATR=6.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 11:15:00 | 1940.78 | 1916.44 | 0.00 | T1 1.5R @ 1940.78 |
| Stop hit — per-position SL triggered | 2024-10-04 12:40:00 | 1931.25 | 1929.42 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1918.65 | 1929.92 | 0.00 | ORB-short ORB[1921.65,1937.00] vol=1.5x ATR=5.71 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1924.36 | 1929.43 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:15:00 | 1971.85 | 1960.02 | 0.00 | ORB-long ORB[1949.00,1959.85] vol=3.2x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-10-09 11:35:00 | 1967.31 | 1961.39 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 1954.70 | 1942.25 | 0.00 | ORB-long ORB[1930.15,1945.15] vol=2.3x ATR=7.34 |
| Stop hit — per-position SL triggered | 2024-10-17 09:45:00 | 1947.36 | 1943.01 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:35:00 | 1870.45 | 1858.09 | 0.00 | ORB-long ORB[1844.00,1864.00] vol=1.7x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:00:00 | 1878.32 | 1861.46 | 0.00 | T1 1.5R @ 1878.32 |
| Stop hit — per-position SL triggered | 2024-10-23 14:35:00 | 1870.45 | 1871.70 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:55:00 | 1832.30 | 1842.38 | 0.00 | ORB-short ORB[1834.60,1851.60] vol=2.9x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 1837.24 | 1842.02 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 1793.90 | 1777.26 | 0.00 | ORB-long ORB[1762.65,1779.60] vol=2.6x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:50:00 | 1803.09 | 1784.57 | 0.00 | T1 1.5R @ 1803.09 |
| Target hit | 2024-11-06 15:20:00 | 1824.95 | 1809.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2024-11-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:10:00 | 1793.95 | 1804.73 | 0.00 | ORB-short ORB[1809.00,1825.70] vol=5.1x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-11-07 11:55:00 | 1798.06 | 1803.15 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 1880.75 | 1899.06 | 0.00 | ORB-short ORB[1895.20,1914.90] vol=2.1x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 1871.76 | 1895.50 | 0.00 | T1 1.5R @ 1871.76 |
| Stop hit — per-position SL triggered | 2024-11-28 11:10:00 | 1880.75 | 1891.60 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 1869.60 | 1861.74 | 0.00 | ORB-long ORB[1845.00,1859.45] vol=2.2x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-11-29 11:05:00 | 1863.79 | 1862.01 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:15:00 | 1869.35 | 1855.03 | 0.00 | ORB-long ORB[1835.20,1861.00] vol=1.9x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-12-02 12:00:00 | 1865.34 | 1858.11 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1891.80 | 1902.59 | 0.00 | ORB-short ORB[1896.25,1912.55] vol=1.6x ATR=4.55 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 1896.35 | 1902.46 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:15:00 | 1965.75 | 1963.25 | 0.00 | ORB-long ORB[1954.00,1965.00] vol=3.1x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 15:00:00 | 1971.05 | 1966.21 | 0.00 | T1 1.5R @ 1971.05 |
| Target hit | 2024-12-11 15:20:00 | 1973.70 | 1968.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-12-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:05:00 | 1956.35 | 1965.85 | 0.00 | ORB-short ORB[1965.00,1975.40] vol=2.7x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 1960.59 | 1965.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1946.00 | 1958.09 | 0.00 | ORB-short ORB[1951.00,1979.95] vol=1.9x ATR=6.49 |
| Stop hit — per-position SL triggered | 2024-12-20 09:35:00 | 1952.49 | 1957.21 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:15:00 | 1906.55 | 1909.91 | 0.00 | ORB-short ORB[1907.25,1919.75] vol=2.3x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-12-26 11:40:00 | 1910.39 | 1909.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:55:00 | 1895.00 | 1897.76 | 0.00 | ORB-short ORB[1902.00,1916.00] vol=1.6x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-12-30 11:25:00 | 1899.49 | 1897.00 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:45:00 | 1914.30 | 1902.61 | 0.00 | ORB-long ORB[1885.30,1904.00] vol=1.6x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:50:00 | 1921.52 | 1912.23 | 0.00 | T1 1.5R @ 1921.52 |
| Target hit | 2025-01-02 15:20:00 | 1954.95 | 1944.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-01-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:40:00 | 1932.60 | 1938.98 | 0.00 | ORB-short ORB[1940.80,1952.95] vol=1.6x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:50:00 | 1926.61 | 1937.30 | 0.00 | T1 1.5R @ 1926.61 |
| Stop hit — per-position SL triggered | 2025-01-03 11:20:00 | 1932.60 | 1933.87 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 11:05:00 | 1972.70 | 1964.73 | 0.00 | ORB-long ORB[1949.00,1966.25] vol=1.7x ATR=5.94 |
| Stop hit — per-position SL triggered | 2025-01-13 11:30:00 | 1966.76 | 1965.74 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:45:00 | 1956.60 | 1948.86 | 0.00 | ORB-long ORB[1937.10,1951.80] vol=3.2x ATR=4.18 |
| Stop hit — per-position SL triggered | 2025-01-15 11:00:00 | 1952.42 | 1949.37 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:30:00 | 1827.30 | 1841.77 | 0.00 | ORB-short ORB[1831.50,1858.00] vol=2.0x ATR=12.06 |
| Target hit | 2025-01-17 15:20:00 | 1816.20 | 1824.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-27 09:35:00 | 1861.80 | 1855.61 | 0.00 | ORB-long ORB[1841.00,1857.15] vol=4.4x ATR=5.39 |
| Stop hit — per-position SL triggered | 2025-01-27 09:40:00 | 1856.41 | 1855.76 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 11:00:00 | 1859.85 | 1868.45 | 0.00 | ORB-short ORB[1863.35,1887.20] vol=1.5x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-01-30 11:45:00 | 1864.77 | 1865.32 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 1907.60 | 1911.04 | 0.00 | ORB-short ORB[1910.00,1924.00] vol=1.5x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 13:00:00 | 1901.67 | 1909.53 | 0.00 | T1 1.5R @ 1901.67 |
| Stop hit — per-position SL triggered | 2025-02-06 14:10:00 | 1907.60 | 1908.43 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:45:00 | 1861.55 | 1875.77 | 0.00 | ORB-short ORB[1880.05,1894.00] vol=4.7x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 11:25:00 | 1853.69 | 1873.52 | 0.00 | T1 1.5R @ 1853.69 |
| Stop hit — per-position SL triggered | 2025-02-12 11:55:00 | 1861.55 | 1871.49 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 11:00:00 | 1851.00 | 1857.48 | 0.00 | ORB-short ORB[1853.00,1873.50] vol=1.5x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 1855.00 | 1856.92 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 10:35:00 | 1826.05 | 1822.81 | 0.00 | ORB-long ORB[1810.60,1825.85] vol=9.9x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-02-21 10:55:00 | 1822.04 | 1822.98 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:40:00 | 1775.20 | 1785.31 | 0.00 | ORB-short ORB[1789.10,1808.00] vol=2.2x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:45:00 | 1768.56 | 1780.04 | 0.00 | T1 1.5R @ 1768.56 |
| Target hit | 2025-02-24 15:20:00 | 1764.85 | 1766.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:10:00 | 1614.60 | 1609.78 | 0.00 | ORB-long ORB[1595.55,1609.80] vol=1.9x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1610.87 | 1609.93 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-04-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:25:00 | 1521.40 | 1536.51 | 0.00 | ORB-short ORB[1526.05,1547.40] vol=1.7x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-04-01 11:20:00 | 1526.76 | 1533.48 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 09:30:00 | 1386.00 | 1390.75 | 0.00 | ORB-short ORB[1386.10,1400.00] vol=1.9x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-04-17 09:40:00 | 1390.31 | 1390.46 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:10:00 | 1478.30 | 1471.38 | 0.00 | ORB-long ORB[1461.50,1477.40] vol=3.8x ATR=3.27 |
| Stop hit — per-position SL triggered | 2025-04-24 12:00:00 | 1475.03 | 1472.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:45:00 | 1419.30 | 2024-05-14 11:05:00 | 1421.12 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2024-05-23 09:35:00 | 1464.70 | 2024-05-23 09:50:00 | 1461.94 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-30 11:05:00 | 1436.80 | 2024-05-30 12:05:00 | 1439.06 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-05-31 11:05:00 | 1420.10 | 2024-05-31 12:10:00 | 1426.07 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-05-31 11:05:00 | 1420.10 | 2024-05-31 15:00:00 | 1420.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 11:15:00 | 1454.45 | 2024-06-06 11:30:00 | 1450.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-07 09:35:00 | 1505.05 | 2024-06-07 09:45:00 | 1512.56 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-07 09:35:00 | 1505.05 | 2024-06-07 15:20:00 | 1534.20 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2024-06-12 11:00:00 | 1495.00 | 2024-06-12 12:25:00 | 1497.54 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-06-19 11:10:00 | 1501.75 | 2024-06-19 11:40:00 | 1503.82 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-06-25 10:40:00 | 1534.05 | 2024-06-25 12:35:00 | 1537.50 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-06-25 10:40:00 | 1534.05 | 2024-06-25 13:20:00 | 1534.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:50:00 | 1549.40 | 2024-06-27 10:55:00 | 1546.24 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-01 10:45:00 | 1582.50 | 2024-07-01 11:10:00 | 1587.98 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-01 10:45:00 | 1582.50 | 2024-07-01 15:20:00 | 1589.00 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-04 09:30:00 | 1655.00 | 2024-07-04 09:35:00 | 1650.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-08 11:05:00 | 1661.45 | 2024-07-08 11:10:00 | 1657.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1693.45 | 2024-07-12 11:00:00 | 1700.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1693.45 | 2024-07-12 15:20:00 | 1710.10 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-07-16 09:50:00 | 1731.15 | 2024-07-16 10:10:00 | 1725.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-22 10:45:00 | 1818.55 | 2024-07-22 12:25:00 | 1813.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-31 11:00:00 | 1872.00 | 2024-07-31 11:10:00 | 1867.26 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-07-31 11:00:00 | 1872.00 | 2024-07-31 15:20:00 | 1868.85 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2024-08-05 10:40:00 | 1755.65 | 2024-08-05 10:50:00 | 1746.74 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-08-05 10:40:00 | 1755.65 | 2024-08-05 13:25:00 | 1747.20 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-12 11:10:00 | 1791.05 | 2024-08-12 11:20:00 | 1796.94 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-12 11:10:00 | 1791.05 | 2024-08-12 15:00:00 | 1791.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 09:55:00 | 1896.30 | 2024-08-27 10:35:00 | 1892.64 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-28 10:35:00 | 1928.30 | 2024-08-28 10:50:00 | 1934.69 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-28 10:35:00 | 1928.30 | 2024-08-28 15:20:00 | 1939.05 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-29 11:10:00 | 1947.00 | 2024-08-29 11:20:00 | 1943.65 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-05 11:00:00 | 1910.60 | 2024-09-05 11:20:00 | 1914.18 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-09 11:05:00 | 1917.60 | 2024-09-09 12:55:00 | 1912.97 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-10 10:55:00 | 1924.50 | 2024-09-10 11:05:00 | 1920.17 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-17 09:40:00 | 1951.30 | 2024-09-17 09:45:00 | 1947.94 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-18 09:30:00 | 1904.00 | 2024-09-18 10:30:00 | 1895.66 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-18 09:30:00 | 1904.00 | 2024-09-18 12:45:00 | 1904.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:00:00 | 1900.00 | 2024-09-19 10:05:00 | 1906.21 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-23 10:55:00 | 1890.25 | 2024-09-23 11:20:00 | 1883.78 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-23 10:55:00 | 1890.25 | 2024-09-23 11:55:00 | 1890.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-03 10:10:00 | 1904.45 | 2024-10-03 10:25:00 | 1899.31 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-04 10:50:00 | 1931.25 | 2024-10-04 11:15:00 | 1940.78 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-04 10:50:00 | 1931.25 | 2024-10-04 12:40:00 | 1931.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1918.65 | 2024-10-07 11:15:00 | 1924.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-09 11:15:00 | 1971.85 | 2024-10-09 11:35:00 | 1967.31 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-17 09:40:00 | 1954.70 | 2024-10-17 09:45:00 | 1947.36 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-23 10:35:00 | 1870.45 | 2024-10-23 11:00:00 | 1878.32 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-23 10:35:00 | 1870.45 | 2024-10-23 14:35:00 | 1870.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:55:00 | 1832.30 | 2024-10-29 11:15:00 | 1837.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1793.90 | 2024-11-06 09:50:00 | 1803.09 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1793.90 | 2024-11-06 15:20:00 | 1824.95 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2024-11-07 11:10:00 | 1793.95 | 2024-11-07 11:55:00 | 1798.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1880.75 | 2024-11-28 10:45:00 | 1871.76 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1880.75 | 2024-11-28 11:10:00 | 1880.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:55:00 | 1869.60 | 2024-11-29 11:05:00 | 1863.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-02 11:15:00 | 1869.35 | 2024-12-02 12:00:00 | 1865.34 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1891.80 | 2024-12-05 11:00:00 | 1896.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-11 11:15:00 | 1965.75 | 2024-12-11 15:00:00 | 1971.05 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-12-11 11:15:00 | 1965.75 | 2024-12-11 15:20:00 | 1973.70 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-13 11:05:00 | 1956.35 | 2024-12-13 11:10:00 | 1960.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-20 09:30:00 | 1946.00 | 2024-12-20 09:35:00 | 1952.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-26 11:15:00 | 1906.55 | 2024-12-26 11:40:00 | 1910.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-30 10:55:00 | 1895.00 | 2024-12-30 11:25:00 | 1899.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-02 09:45:00 | 1914.30 | 2025-01-02 10:50:00 | 1921.52 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-02 09:45:00 | 1914.30 | 2025-01-02 15:20:00 | 1954.95 | TARGET_HIT | 0.50 | 2.12% |
| SELL | retest1 | 2025-01-03 10:40:00 | 1932.60 | 2025-01-03 10:50:00 | 1926.61 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-03 10:40:00 | 1932.60 | 2025-01-03 11:20:00 | 1932.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-13 11:05:00 | 1972.70 | 2025-01-13 11:30:00 | 1966.76 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-15 10:45:00 | 1956.60 | 2025-01-15 11:00:00 | 1952.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-17 09:30:00 | 1827.30 | 2025-01-17 15:20:00 | 1816.20 | TARGET_HIT | 1.00 | 0.61% |
| BUY | retest1 | 2025-01-27 09:35:00 | 1861.80 | 2025-01-27 09:40:00 | 1856.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-30 11:00:00 | 1859.85 | 2025-01-30 11:45:00 | 1864.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1907.60 | 2025-02-06 13:00:00 | 1901.67 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1907.60 | 2025-02-06 14:10:00 | 1907.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 10:45:00 | 1861.55 | 2025-02-12 11:25:00 | 1853.69 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-12 10:45:00 | 1861.55 | 2025-02-12 11:55:00 | 1861.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-13 11:00:00 | 1851.00 | 2025-02-13 11:15:00 | 1855.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-02-21 10:35:00 | 1826.05 | 2025-02-21 10:55:00 | 1822.04 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-02-24 10:40:00 | 1775.20 | 2025-02-24 11:45:00 | 1768.56 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-02-24 10:40:00 | 1775.20 | 2025-02-24 15:20:00 | 1764.85 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-27 11:10:00 | 1614.60 | 2025-03-27 11:15:00 | 1610.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-01 10:25:00 | 1521.40 | 2025-04-01 11:20:00 | 1526.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-17 09:30:00 | 1386.00 | 2025-04-17 09:40:00 | 1390.31 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-24 11:10:00 | 1478.30 | 2025-04-24 12:00:00 | 1475.03 | STOP_HIT | 1.00 | -0.22% |
