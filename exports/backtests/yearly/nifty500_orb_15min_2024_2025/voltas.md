# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1323.00
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
| PARTIAL | 24 |
| TARGET_HIT | 14 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 53
- **Target hits / Stop hits / Partials:** 14 / 53 / 24
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 18.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 17 | 38.6% | 6 | 27 | 11 | 0.12% | 5.3% |
| BUY @ 2nd Alert (retest1) | 44 | 17 | 38.6% | 6 | 27 | 11 | 0.12% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 21 | 44.7% | 8 | 26 | 13 | 0.27% | 12.8% |
| SELL @ 2nd Alert (retest1) | 47 | 21 | 44.7% | 8 | 26 | 13 | 0.27% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 91 | 38 | 41.8% | 14 | 53 | 24 | 0.20% | 18.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 1277.00 | 1282.28 | 0.00 | ORB-short ORB[1279.30,1296.85] vol=3.5x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 12:05:00 | 1267.85 | 1277.70 | 0.00 | T1 1.5R @ 1267.85 |
| Target hit | 2024-05-13 12:30:00 | 1274.85 | 1274.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:55:00 | 1321.55 | 1315.36 | 0.00 | ORB-long ORB[1304.05,1315.95] vol=1.9x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-05-16 11:05:00 | 1317.70 | 1315.61 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 1311.80 | 1304.24 | 0.00 | ORB-long ORB[1289.65,1309.00] vol=3.1x ATR=4.08 |
| Stop hit — per-position SL triggered | 2024-05-23 10:40:00 | 1307.72 | 1304.81 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:00:00 | 1363.55 | 1347.89 | 0.00 | ORB-long ORB[1335.50,1353.00] vol=1.6x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:05:00 | 1370.75 | 1352.91 | 0.00 | T1 1.5R @ 1370.75 |
| Target hit | 2024-05-24 15:20:00 | 1371.45 | 1368.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 1378.00 | 1369.89 | 0.00 | ORB-long ORB[1359.85,1374.75] vol=1.8x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-05-27 09:50:00 | 1372.51 | 1372.59 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:45:00 | 1470.85 | 1465.29 | 0.00 | ORB-long ORB[1448.05,1464.45] vol=1.8x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-06-11 11:20:00 | 1466.43 | 1466.45 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:55:00 | 1448.10 | 1458.28 | 0.00 | ORB-short ORB[1455.00,1469.85] vol=3.7x ATR=4.26 |
| Stop hit — per-position SL triggered | 2024-06-12 11:40:00 | 1452.36 | 1456.33 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 11:15:00 | 1494.70 | 1490.54 | 0.00 | ORB-long ORB[1475.15,1494.00] vol=2.2x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 11:50:00 | 1500.52 | 1492.44 | 0.00 | T1 1.5R @ 1500.52 |
| Stop hit — per-position SL triggered | 2024-06-21 11:55:00 | 1494.70 | 1492.52 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:40:00 | 1505.00 | 1504.44 | 0.00 | ORB-long ORB[1492.65,1501.50] vol=2.0x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-06-27 10:45:00 | 1501.71 | 1504.16 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:50:00 | 1484.55 | 1491.82 | 0.00 | ORB-short ORB[1489.15,1507.45] vol=2.1x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:15:00 | 1478.89 | 1490.33 | 0.00 | T1 1.5R @ 1478.89 |
| Stop hit — per-position SL triggered | 2024-06-28 11:35:00 | 1484.55 | 1489.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:40:00 | 1472.95 | 1463.66 | 0.00 | ORB-long ORB[1453.65,1464.45] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-07-03 10:45:00 | 1469.22 | 1464.11 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 1444.05 | 1447.81 | 0.00 | ORB-short ORB[1445.05,1457.20] vol=3.3x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-07-08 09:50:00 | 1448.10 | 1447.46 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:40:00 | 1461.65 | 1463.22 | 0.00 | ORB-short ORB[1470.00,1479.00] vol=5.0x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 1466.96 | 1463.39 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:15:00 | 1511.30 | 1513.59 | 0.00 | ORB-short ORB[1512.35,1522.95] vol=2.0x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:30:00 | 1505.57 | 1513.10 | 0.00 | T1 1.5R @ 1505.57 |
| Stop hit — per-position SL triggered | 2024-07-12 12:35:00 | 1511.30 | 1511.57 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:55:00 | 1498.90 | 1507.06 | 0.00 | ORB-short ORB[1506.55,1525.00] vol=1.9x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:25:00 | 1492.80 | 1504.10 | 0.00 | T1 1.5R @ 1492.80 |
| Stop hit — per-position SL triggered | 2024-07-18 12:20:00 | 1498.90 | 1502.19 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1476.80 | 1480.58 | 0.00 | ORB-short ORB[1478.05,1495.90] vol=2.9x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-07-23 11:40:00 | 1482.07 | 1480.32 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 11:15:00 | 1479.30 | 1480.53 | 0.00 | ORB-short ORB[1483.30,1494.95] vol=1.6x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-07-25 11:20:00 | 1483.43 | 1480.57 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:00:00 | 1523.50 | 1533.86 | 0.00 | ORB-short ORB[1537.60,1547.05] vol=9.5x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-08-01 11:05:00 | 1527.97 | 1533.61 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:15:00 | 1623.20 | 1608.88 | 0.00 | ORB-long ORB[1604.75,1622.00] vol=3.4x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:25:00 | 1632.20 | 1614.90 | 0.00 | T1 1.5R @ 1632.20 |
| Target hit | 2024-08-20 15:20:00 | 1659.75 | 1640.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:05:00 | 1686.60 | 1677.42 | 0.00 | ORB-long ORB[1674.80,1683.65] vol=2.1x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 1682.45 | 1677.88 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 11:15:00 | 1706.25 | 1696.65 | 0.00 | ORB-long ORB[1684.25,1700.40] vol=5.2x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:20:00 | 1712.56 | 1699.12 | 0.00 | T1 1.5R @ 1712.56 |
| Target hit | 2024-08-26 15:20:00 | 1720.95 | 1709.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2024-09-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:40:00 | 1766.20 | 1771.97 | 0.00 | ORB-short ORB[1776.00,1788.75] vol=2.7x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-09-06 10:45:00 | 1770.52 | 1771.95 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 11:10:00 | 1904.80 | 1907.35 | 0.00 | ORB-short ORB[1905.00,1924.90] vol=2.5x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-09-17 11:25:00 | 1908.43 | 1907.27 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 1916.50 | 1907.77 | 0.00 | ORB-long ORB[1890.00,1907.95] vol=2.4x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-09-19 09:45:00 | 1911.24 | 1909.03 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:05:00 | 1891.00 | 1896.55 | 0.00 | ORB-short ORB[1898.20,1910.85] vol=3.6x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:40:00 | 1884.45 | 1893.75 | 0.00 | T1 1.5R @ 1884.45 |
| Target hit | 2024-09-26 15:20:00 | 1855.45 | 1862.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 1843.40 | 1851.80 | 0.00 | ORB-short ORB[1847.20,1866.85] vol=1.7x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-09-27 11:45:00 | 1847.67 | 1851.16 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:30:00 | 1845.90 | 1837.58 | 0.00 | ORB-long ORB[1825.15,1841.75] vol=1.7x ATR=6.53 |
| Stop hit — per-position SL triggered | 2024-10-03 09:40:00 | 1839.37 | 1839.05 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:50:00 | 1765.50 | 1792.57 | 0.00 | ORB-short ORB[1800.40,1818.50] vol=1.8x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 1772.75 | 1789.49 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:50:00 | 1789.15 | 1795.42 | 0.00 | ORB-short ORB[1797.25,1814.60] vol=1.5x ATR=5.82 |
| Target hit | 2024-10-09 15:20:00 | 1786.50 | 1789.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-10-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:00:00 | 1760.00 | 1767.86 | 0.00 | ORB-short ORB[1765.55,1785.40] vol=2.8x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-10-11 10:05:00 | 1765.57 | 1767.43 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:25:00 | 1859.20 | 1835.23 | 0.00 | ORB-long ORB[1810.00,1833.65] vol=1.6x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-10-18 13:00:00 | 1851.69 | 1852.38 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 11:15:00 | 1837.45 | 1827.60 | 0.00 | ORB-long ORB[1807.80,1829.15] vol=3.4x ATR=7.02 |
| Stop hit — per-position SL triggered | 2024-10-22 11:20:00 | 1830.43 | 1827.80 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:50:00 | 1799.05 | 1788.65 | 0.00 | ORB-long ORB[1775.35,1794.00] vol=2.7x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:00:00 | 1810.02 | 1790.13 | 0.00 | T1 1.5R @ 1810.02 |
| Target hit | 2024-10-23 14:30:00 | 1801.20 | 1802.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:10:00 | 1816.50 | 1806.90 | 0.00 | ORB-long ORB[1787.15,1804.00] vol=4.5x ATR=7.90 |
| Stop hit — per-position SL triggered | 2024-10-24 10:35:00 | 1808.60 | 1809.16 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:50:00 | 1741.95 | 1772.31 | 0.00 | ORB-short ORB[1781.60,1803.00] vol=1.6x ATR=7.83 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 1749.78 | 1771.28 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:35:00 | 1712.80 | 1705.55 | 0.00 | ORB-long ORB[1689.05,1708.35] vol=2.2x ATR=5.82 |
| Stop hit — per-position SL triggered | 2024-11-06 11:10:00 | 1706.98 | 1706.86 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 11:00:00 | 1688.25 | 1703.86 | 0.00 | ORB-short ORB[1709.05,1725.65] vol=1.9x ATR=4.81 |
| Stop hit — per-position SL triggered | 2024-12-03 12:45:00 | 1693.06 | 1699.50 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:50:00 | 1670.05 | 1672.16 | 0.00 | ORB-short ORB[1680.00,1692.35] vol=3.9x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 10:55:00 | 1664.73 | 1671.88 | 0.00 | T1 1.5R @ 1664.73 |
| Stop hit — per-position SL triggered | 2024-12-05 12:10:00 | 1670.05 | 1669.26 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 1781.45 | 1767.78 | 0.00 | ORB-long ORB[1757.80,1769.95] vol=3.1x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-12-11 11:10:00 | 1777.47 | 1768.31 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 1786.75 | 1790.38 | 0.00 | ORB-short ORB[1797.60,1811.00] vol=3.6x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-12-17 11:25:00 | 1790.74 | 1789.21 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:50:00 | 1727.75 | 1735.36 | 0.00 | ORB-short ORB[1739.20,1755.00] vol=3.4x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:05:00 | 1720.51 | 1734.55 | 0.00 | T1 1.5R @ 1720.51 |
| Target hit | 2024-12-20 15:20:00 | 1679.20 | 1712.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:05:00 | 1713.65 | 1693.98 | 0.00 | ORB-long ORB[1680.85,1696.80] vol=1.8x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-12-30 11:15:00 | 1707.15 | 1707.20 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 1847.85 | 1831.84 | 0.00 | ORB-long ORB[1815.20,1832.00] vol=1.8x ATR=6.94 |
| Stop hit — per-position SL triggered | 2025-01-02 10:35:00 | 1840.91 | 1836.38 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 1851.00 | 1837.83 | 0.00 | ORB-long ORB[1822.85,1842.50] vol=1.5x ATR=5.54 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 1845.46 | 1838.76 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:00:00 | 1811.80 | 1816.50 | 0.00 | ORB-short ORB[1821.10,1833.10] vol=1.9x ATR=6.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:20:00 | 1801.61 | 1812.09 | 0.00 | T1 1.5R @ 1801.61 |
| Target hit | 2025-01-06 15:20:00 | 1774.30 | 1791.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:40:00 | 1712.20 | 1716.10 | 0.00 | ORB-short ORB[1712.60,1731.35] vol=2.2x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:45:00 | 1703.68 | 1715.26 | 0.00 | T1 1.5R @ 1703.68 |
| Target hit | 2025-01-10 15:20:00 | 1669.45 | 1681.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 1619.85 | 1637.42 | 0.00 | ORB-short ORB[1635.75,1654.00] vol=2.2x ATR=5.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:50:00 | 1611.17 | 1634.13 | 0.00 | T1 1.5R @ 1611.17 |
| Target hit | 2025-01-13 15:20:00 | 1596.40 | 1605.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-01-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:30:00 | 1608.10 | 1597.75 | 0.00 | ORB-long ORB[1578.60,1591.70] vol=2.6x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-01-17 10:40:00 | 1602.75 | 1598.33 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 1523.30 | 1537.75 | 0.00 | ORB-short ORB[1536.95,1558.90] vol=1.5x ATR=5.76 |
| Stop hit — per-position SL triggered | 2025-01-21 10:30:00 | 1529.06 | 1537.28 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:40:00 | 1488.05 | 1497.87 | 0.00 | ORB-short ORB[1495.00,1511.90] vol=1.5x ATR=5.88 |
| Stop hit — per-position SL triggered | 2025-01-22 09:55:00 | 1493.93 | 1496.46 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:35:00 | 1508.30 | 1500.22 | 0.00 | ORB-long ORB[1464.50,1484.30] vol=3.5x ATR=7.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:55:00 | 1518.99 | 1502.04 | 0.00 | T1 1.5R @ 1518.99 |
| Target hit | 2025-01-23 12:45:00 | 1508.60 | 1509.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2025-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:55:00 | 1444.05 | 1435.46 | 0.00 | ORB-long ORB[1418.55,1435.00] vol=2.8x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:05:00 | 1453.83 | 1440.23 | 0.00 | T1 1.5R @ 1453.83 |
| Target hit | 2025-01-29 15:20:00 | 1482.25 | 1459.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:15:00 | 1276.80 | 1271.26 | 0.00 | ORB-long ORB[1260.25,1273.95] vol=2.3x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:35:00 | 1283.39 | 1273.03 | 0.00 | T1 1.5R @ 1283.39 |
| Stop hit — per-position SL triggered | 2025-02-01 10:45:00 | 1276.80 | 1273.51 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 10:45:00 | 1270.70 | 1257.67 | 0.00 | ORB-long ORB[1243.75,1260.00] vol=5.6x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:20:00 | 1279.51 | 1260.22 | 0.00 | T1 1.5R @ 1279.51 |
| Stop hit — per-position SL triggered | 2025-02-24 13:35:00 | 1270.70 | 1271.10 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:40:00 | 1295.40 | 1284.19 | 0.00 | ORB-long ORB[1271.05,1284.90] vol=1.7x ATR=5.49 |
| Stop hit — per-position SL triggered | 2025-02-25 10:10:00 | 1289.91 | 1289.96 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 10:55:00 | 1393.00 | 1405.31 | 0.00 | ORB-short ORB[1393.55,1414.00] vol=1.6x ATR=6.21 |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 1399.21 | 1403.69 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:05:00 | 1415.45 | 1404.05 | 0.00 | ORB-long ORB[1390.40,1410.00] vol=2.0x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 1409.45 | 1407.04 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 11:10:00 | 1401.30 | 1409.94 | 0.00 | ORB-short ORB[1402.20,1416.65] vol=1.5x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:45:00 | 1395.33 | 1406.57 | 0.00 | T1 1.5R @ 1395.33 |
| Stop hit — per-position SL triggered | 2025-03-07 12:55:00 | 1401.30 | 1400.66 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:40:00 | 1424.95 | 1416.84 | 0.00 | ORB-long ORB[1398.00,1417.85] vol=2.7x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:45:00 | 1434.30 | 1419.42 | 0.00 | T1 1.5R @ 1434.30 |
| Stop hit — per-position SL triggered | 2025-03-10 09:55:00 | 1424.95 | 1421.24 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 1427.90 | 1416.56 | 0.00 | ORB-long ORB[1402.95,1421.00] vol=1.5x ATR=5.16 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 1422.74 | 1419.00 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:30:00 | 1477.05 | 1464.80 | 0.00 | ORB-long ORB[1450.75,1472.55] vol=1.5x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-03-18 09:40:00 | 1471.69 | 1467.35 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-03-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:00:00 | 1419.55 | 1425.76 | 0.00 | ORB-short ORB[1422.10,1433.95] vol=1.9x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-03-26 11:05:00 | 1423.42 | 1425.70 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-04-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:45:00 | 1324.95 | 1334.53 | 0.00 | ORB-short ORB[1333.25,1351.00] vol=1.9x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:50:00 | 1317.75 | 1331.93 | 0.00 | T1 1.5R @ 1317.75 |
| Target hit | 2025-04-04 15:20:00 | 1297.65 | 1309.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1286.95 | 1296.42 | 0.00 | ORB-short ORB[1297.05,1313.50] vol=2.5x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 11:00:00 | 1280.14 | 1290.54 | 0.00 | T1 1.5R @ 1280.14 |
| Stop hit — per-position SL triggered | 2025-04-09 11:40:00 | 1286.95 | 1285.88 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 1289.90 | 1296.80 | 0.00 | ORB-short ORB[1295.00,1310.30] vol=4.5x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-04-15 09:35:00 | 1294.69 | 1295.90 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 1277.60 | 1284.11 | 0.00 | ORB-short ORB[1283.20,1297.10] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-04-16 12:25:00 | 1280.80 | 1282.32 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 1305.00 | 1293.04 | 0.00 | ORB-long ORB[1283.80,1296.00] vol=1.6x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:40:00 | 1311.20 | 1298.58 | 0.00 | T1 1.5R @ 1311.20 |
| Stop hit — per-position SL triggered | 2025-04-21 11:25:00 | 1305.00 | 1306.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 1277.00 | 2024-05-13 12:05:00 | 1267.85 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-05-13 11:05:00 | 1277.00 | 2024-05-13 12:30:00 | 1274.85 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-05-16 10:55:00 | 1321.55 | 2024-05-16 11:05:00 | 1317.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-23 10:35:00 | 1311.80 | 2024-05-23 10:40:00 | 1307.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-24 11:00:00 | 1363.55 | 2024-05-24 11:05:00 | 1370.75 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-05-24 11:00:00 | 1363.55 | 2024-05-24 15:20:00 | 1371.45 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2024-05-27 09:35:00 | 1378.00 | 2024-05-27 09:50:00 | 1372.51 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-11 10:45:00 | 1470.85 | 2024-06-11 11:20:00 | 1466.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-12 10:55:00 | 1448.10 | 2024-06-12 11:40:00 | 1452.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-21 11:15:00 | 1494.70 | 2024-06-21 11:50:00 | 1500.52 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-21 11:15:00 | 1494.70 | 2024-06-21 11:55:00 | 1494.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:40:00 | 1505.00 | 2024-06-27 10:45:00 | 1501.71 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-06-28 10:50:00 | 1484.55 | 2024-06-28 11:15:00 | 1478.89 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-06-28 10:50:00 | 1484.55 | 2024-06-28 11:35:00 | 1484.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 10:40:00 | 1472.95 | 2024-07-03 10:45:00 | 1469.22 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-08 09:40:00 | 1444.05 | 2024-07-08 09:50:00 | 1448.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-10 10:40:00 | 1461.65 | 2024-07-10 10:55:00 | 1466.96 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-12 11:15:00 | 1511.30 | 2024-07-12 11:30:00 | 1505.57 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-12 11:15:00 | 1511.30 | 2024-07-12 12:35:00 | 1511.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 10:55:00 | 1498.90 | 2024-07-18 11:25:00 | 1492.80 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-18 10:55:00 | 1498.90 | 2024-07-18 12:20:00 | 1498.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1476.80 | 2024-07-23 11:40:00 | 1482.07 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-25 11:15:00 | 1479.30 | 2024-07-25 11:20:00 | 1483.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-01 11:00:00 | 1523.50 | 2024-08-01 11:05:00 | 1527.97 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-20 11:15:00 | 1623.20 | 2024-08-20 11:25:00 | 1632.20 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-20 11:15:00 | 1623.20 | 2024-08-20 15:20:00 | 1659.75 | TARGET_HIT | 0.50 | 2.25% |
| BUY | retest1 | 2024-08-22 11:05:00 | 1686.60 | 2024-08-22 11:15:00 | 1682.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-26 11:15:00 | 1706.25 | 2024-08-26 11:20:00 | 1712.56 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-26 11:15:00 | 1706.25 | 2024-08-26 15:20:00 | 1720.95 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2024-09-06 10:40:00 | 1766.20 | 2024-09-06 10:45:00 | 1770.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-17 11:10:00 | 1904.80 | 2024-09-17 11:25:00 | 1908.43 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-19 09:40:00 | 1916.50 | 2024-09-19 09:45:00 | 1911.24 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-26 10:05:00 | 1891.00 | 2024-09-26 10:40:00 | 1884.45 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-26 10:05:00 | 1891.00 | 2024-09-26 15:20:00 | 1855.45 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2024-09-27 11:00:00 | 1843.40 | 2024-09-27 11:45:00 | 1847.67 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-03 09:30:00 | 1845.90 | 2024-10-03 09:40:00 | 1839.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-07 10:50:00 | 1765.50 | 2024-10-07 11:05:00 | 1772.75 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-09 10:50:00 | 1789.15 | 2024-10-09 15:20:00 | 1786.50 | TARGET_HIT | 1.00 | 0.15% |
| SELL | retest1 | 2024-10-11 10:00:00 | 1760.00 | 2024-10-11 10:05:00 | 1765.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-18 10:25:00 | 1859.20 | 2024-10-18 13:00:00 | 1851.69 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-22 11:15:00 | 1837.45 | 2024-10-22 11:20:00 | 1830.43 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-23 10:50:00 | 1799.05 | 2024-10-23 11:00:00 | 1810.02 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-23 10:50:00 | 1799.05 | 2024-10-23 14:30:00 | 1801.20 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-10-24 10:10:00 | 1816.50 | 2024-10-24 10:35:00 | 1808.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-25 10:50:00 | 1741.95 | 2024-10-25 10:55:00 | 1749.78 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-11-06 10:35:00 | 1712.80 | 2024-11-06 11:10:00 | 1706.98 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-03 11:00:00 | 1688.25 | 2024-12-03 12:45:00 | 1693.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-05 10:50:00 | 1670.05 | 2024-12-05 10:55:00 | 1664.73 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-05 10:50:00 | 1670.05 | 2024-12-05 12:10:00 | 1670.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:05:00 | 1781.45 | 2024-12-11 11:10:00 | 1777.47 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-17 10:45:00 | 1786.75 | 2024-12-17 11:25:00 | 1790.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-20 10:50:00 | 1727.75 | 2024-12-20 11:05:00 | 1720.51 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-20 10:50:00 | 1727.75 | 2024-12-20 15:20:00 | 1679.20 | TARGET_HIT | 0.50 | 2.81% |
| BUY | retest1 | 2024-12-30 10:05:00 | 1713.65 | 2024-12-30 11:15:00 | 1707.15 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-02 10:05:00 | 1847.85 | 2025-01-02 10:35:00 | 1840.91 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-03 09:35:00 | 1851.00 | 2025-01-03 09:40:00 | 1845.46 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-06 10:00:00 | 1811.80 | 2025-01-06 10:20:00 | 1801.61 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-06 10:00:00 | 1811.80 | 2025-01-06 15:20:00 | 1774.30 | TARGET_HIT | 0.50 | 2.07% |
| SELL | retest1 | 2025-01-10 09:40:00 | 1712.20 | 2025-01-10 09:45:00 | 1703.68 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-10 09:40:00 | 1712.20 | 2025-01-10 15:20:00 | 1669.45 | TARGET_HIT | 0.50 | 2.50% |
| SELL | retest1 | 2025-01-13 11:10:00 | 1619.85 | 2025-01-13 11:50:00 | 1611.17 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-13 11:10:00 | 1619.85 | 2025-01-13 15:20:00 | 1596.40 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-01-17 10:30:00 | 1608.10 | 2025-01-17 10:40:00 | 1602.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1523.30 | 2025-01-21 10:30:00 | 1529.06 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-22 09:40:00 | 1488.05 | 2025-01-22 09:55:00 | 1493.93 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-23 10:35:00 | 1508.30 | 2025-01-23 10:55:00 | 1518.99 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-01-23 10:35:00 | 1508.30 | 2025-01-23 12:45:00 | 1508.60 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2025-01-29 09:55:00 | 1444.05 | 2025-01-29 10:05:00 | 1453.83 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-29 09:55:00 | 1444.05 | 2025-01-29 15:20:00 | 1482.25 | TARGET_HIT | 0.50 | 2.65% |
| BUY | retest1 | 2025-02-01 10:15:00 | 1276.80 | 2025-02-01 10:35:00 | 1283.39 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-02-01 10:15:00 | 1276.80 | 2025-02-01 10:45:00 | 1276.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-24 10:45:00 | 1270.70 | 2025-02-24 11:20:00 | 1279.51 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-02-24 10:45:00 | 1270.70 | 2025-02-24 13:35:00 | 1270.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:40:00 | 1295.40 | 2025-02-25 10:10:00 | 1289.91 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-05 10:55:00 | 1393.00 | 2025-03-05 11:15:00 | 1399.21 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-06 10:05:00 | 1415.45 | 2025-03-06 11:15:00 | 1409.45 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-07 11:10:00 | 1401.30 | 2025-03-07 11:45:00 | 1395.33 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-07 11:10:00 | 1401.30 | 2025-03-07 12:55:00 | 1401.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 09:40:00 | 1424.95 | 2025-03-10 09:45:00 | 1434.30 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-03-10 09:40:00 | 1424.95 | 2025-03-10 09:55:00 | 1424.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-17 09:30:00 | 1427.90 | 2025-03-17 09:35:00 | 1422.74 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-18 09:30:00 | 1477.05 | 2025-03-18 09:40:00 | 1471.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-03-26 11:00:00 | 1419.55 | 2025-03-26 11:05:00 | 1423.42 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-04 09:45:00 | 1324.95 | 2025-04-04 09:50:00 | 1317.75 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-04 09:45:00 | 1324.95 | 2025-04-04 15:20:00 | 1297.65 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-04-09 10:00:00 | 1286.95 | 2025-04-09 11:00:00 | 1280.14 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-09 10:00:00 | 1286.95 | 2025-04-09 11:40:00 | 1286.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-15 09:30:00 | 1289.90 | 2025-04-15 09:35:00 | 1294.69 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-16 11:15:00 | 1277.60 | 2025-04-16 12:25:00 | 1280.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-04-21 09:35:00 | 1305.00 | 2025-04-21 09:40:00 | 1311.20 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-21 09:35:00 | 1305.00 | 2025-04-21 11:25:00 | 1305.00 | STOP_HIT | 0.50 | 0.00% |
