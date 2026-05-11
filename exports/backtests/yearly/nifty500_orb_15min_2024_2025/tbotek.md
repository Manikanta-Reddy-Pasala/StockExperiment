# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2024-05-15 09:40:00 → 2026-05-08 15:25:00 (36716 bars)
- **Last close:** 1227.20
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 35
- **Target hits / Stop hits / Partials:** 7 / 35 / 20
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 13.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 19 | 50.0% | 5 | 19 | 14 | 0.35% | 13.4% |
| BUY @ 2nd Alert (retest1) | 38 | 19 | 50.0% | 5 | 19 | 14 | 0.35% | 13.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 8 | 33.3% | 2 | 16 | 6 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 24 | 8 | 33.3% | 2 | 16 | 6 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 62 | 27 | 43.5% | 7 | 35 | 20 | 0.22% | 13.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:50:00 | 1467.00 | 1445.82 | 0.00 | ORB-long ORB[1432.90,1450.45] vol=4.0x ATR=10.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:55:00 | 1482.34 | 1455.70 | 0.00 | T1 1.5R @ 1482.34 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 1467.00 | 1460.66 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:10:00 | 1437.60 | 1447.45 | 0.00 | ORB-short ORB[1445.30,1459.95] vol=2.5x ATR=8.25 |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 1445.85 | 1447.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:20:00 | 1399.60 | 1411.14 | 0.00 | ORB-short ORB[1413.00,1432.00] vol=2.2x ATR=6.86 |
| Stop hit — per-position SL triggered | 2024-05-22 10:25:00 | 1406.46 | 1409.37 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:10:00 | 1410.60 | 1400.52 | 0.00 | ORB-long ORB[1387.05,1403.00] vol=2.5x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:20:00 | 1418.54 | 1408.00 | 0.00 | T1 1.5R @ 1418.54 |
| Target hit | 2024-05-27 13:50:00 | 1438.05 | 1441.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2024-05-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:05:00 | 1422.00 | 1432.32 | 0.00 | ORB-short ORB[1435.05,1454.00] vol=2.9x ATR=4.95 |
| Stop hit — per-position SL triggered | 2024-05-28 11:10:00 | 1426.95 | 1431.76 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:50:00 | 1423.05 | 1421.78 | 0.00 | ORB-long ORB[1409.00,1422.90] vol=11.6x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:10:00 | 1428.50 | 1421.95 | 0.00 | T1 1.5R @ 1428.50 |
| Stop hit — per-position SL triggered | 2024-05-29 10:30:00 | 1423.05 | 1422.03 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 1512.70 | 1519.79 | 0.00 | ORB-short ORB[1513.05,1532.75] vol=1.6x ATR=8.01 |
| Stop hit — per-position SL triggered | 2024-06-11 09:45:00 | 1520.71 | 1519.62 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:50:00 | 1580.00 | 1592.58 | 0.00 | ORB-short ORB[1590.30,1614.00] vol=1.6x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-06-21 14:55:00 | 1585.99 | 1586.05 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:05:00 | 1887.20 | 1897.96 | 0.00 | ORB-short ORB[1887.60,1914.90] vol=3.9x ATR=8.23 |
| Stop hit — per-position SL triggered | 2024-06-28 10:15:00 | 1895.43 | 1897.72 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 1835.75 | 1829.78 | 0.00 | ORB-long ORB[1821.30,1832.45] vol=5.0x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:35:00 | 1843.72 | 1831.60 | 0.00 | T1 1.5R @ 1843.72 |
| Stop hit — per-position SL triggered | 2024-07-09 09:50:00 | 1835.75 | 1832.79 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:40:00 | 1817.00 | 1804.15 | 0.00 | ORB-long ORB[1790.05,1811.55] vol=1.7x ATR=6.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:45:00 | 1827.12 | 1807.46 | 0.00 | T1 1.5R @ 1827.12 |
| Stop hit — per-position SL triggered | 2024-07-18 09:55:00 | 1817.00 | 1808.67 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:35:00 | 1746.75 | 1742.88 | 0.00 | ORB-long ORB[1717.40,1730.90] vol=9.0x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-07-26 10:40:00 | 1742.34 | 1742.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:45:00 | 1760.00 | 1750.96 | 0.00 | ORB-long ORB[1715.10,1735.00] vol=3.1x ATR=8.42 |
| Stop hit — per-position SL triggered | 2024-07-30 09:50:00 | 1751.58 | 1751.36 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:40:00 | 1747.35 | 1741.15 | 0.00 | ORB-long ORB[1722.70,1743.65] vol=12.2x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:45:00 | 1756.51 | 1741.72 | 0.00 | T1 1.5R @ 1756.51 |
| Stop hit — per-position SL triggered | 2024-07-31 11:30:00 | 1747.35 | 1746.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:35:00 | 1737.40 | 1724.99 | 0.00 | ORB-long ORB[1712.70,1732.90] vol=1.7x ATR=5.19 |
| Stop hit — per-position SL triggered | 2024-08-12 09:45:00 | 1732.21 | 1725.63 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 1713.05 | 1684.30 | 0.00 | ORB-long ORB[1656.75,1679.95] vol=2.1x ATR=9.81 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 1703.24 | 1695.04 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 1741.95 | 1736.66 | 0.00 | ORB-long ORB[1718.05,1738.95] vol=1.9x ATR=7.61 |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 1734.34 | 1737.45 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 1691.95 | 1683.47 | 0.00 | ORB-long ORB[1673.40,1690.25] vol=1.7x ATR=7.90 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 1684.05 | 1682.53 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:55:00 | 1828.30 | 1818.03 | 0.00 | ORB-long ORB[1801.00,1825.00] vol=2.4x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 14:35:00 | 1843.52 | 1827.53 | 0.00 | T1 1.5R @ 1843.52 |
| Target hit | 2024-09-03 15:20:00 | 1850.50 | 1833.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:15:00 | 1770.35 | 1781.38 | 0.00 | ORB-short ORB[1776.00,1795.45] vol=1.7x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-09-10 11:40:00 | 1774.76 | 1780.66 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:35:00 | 1776.00 | 1791.28 | 0.00 | ORB-short ORB[1800.45,1814.70] vol=1.6x ATR=8.47 |
| Stop hit — per-position SL triggered | 2024-09-16 10:45:00 | 1784.47 | 1789.43 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 09:55:00 | 1758.60 | 1774.83 | 0.00 | ORB-short ORB[1763.55,1787.55] vol=2.0x ATR=8.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:25:00 | 1746.42 | 1766.73 | 0.00 | T1 1.5R @ 1746.42 |
| Target hit | 2024-09-23 15:20:00 | 1752.00 | 1756.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-09-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:00:00 | 1768.00 | 1759.32 | 0.00 | ORB-long ORB[1749.15,1766.10] vol=1.7x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:05:00 | 1773.31 | 1765.26 | 0.00 | T1 1.5R @ 1773.31 |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 1768.00 | 1766.84 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 11:05:00 | 1708.85 | 1700.86 | 0.00 | ORB-long ORB[1683.30,1707.30] vol=1.7x ATR=7.43 |
| Stop hit — per-position SL triggered | 2024-10-18 11:55:00 | 1701.42 | 1701.43 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-11-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:50:00 | 1592.85 | 1593.96 | 0.00 | ORB-short ORB[1595.10,1615.00] vol=3.0x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:30:00 | 1584.52 | 1593.07 | 0.00 | T1 1.5R @ 1584.52 |
| Target hit | 2024-11-05 15:20:00 | 1570.35 | 1579.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-11-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:10:00 | 1530.00 | 1522.37 | 0.00 | ORB-long ORB[1503.00,1524.60] vol=1.7x ATR=8.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:05:00 | 1542.80 | 1527.61 | 0.00 | T1 1.5R @ 1542.80 |
| Target hit | 2024-11-21 12:00:00 | 1540.80 | 1542.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2024-12-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:00:00 | 1547.70 | 1536.25 | 0.00 | ORB-long ORB[1515.05,1528.10] vol=2.2x ATR=6.79 |
| Stop hit — per-position SL triggered | 2024-12-10 10:25:00 | 1540.91 | 1539.74 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:10:00 | 1582.70 | 1571.74 | 0.00 | ORB-long ORB[1563.10,1581.70] vol=2.2x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:20:00 | 1590.87 | 1580.93 | 0.00 | T1 1.5R @ 1590.87 |
| Target hit | 2024-12-11 15:20:00 | 1650.00 | 1627.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-12-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:25:00 | 1606.50 | 1613.79 | 0.00 | ORB-short ORB[1608.20,1629.95] vol=2.5x ATR=7.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:30:00 | 1595.52 | 1612.60 | 0.00 | T1 1.5R @ 1595.52 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 1606.50 | 1610.51 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1709.95 | 1703.15 | 0.00 | ORB-long ORB[1689.05,1708.00] vol=1.9x ATR=7.56 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 1702.39 | 1703.46 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:15:00 | 1707.25 | 1697.21 | 0.00 | ORB-long ORB[1681.00,1699.30] vol=4.7x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:30:00 | 1718.62 | 1701.27 | 0.00 | T1 1.5R @ 1718.62 |
| Stop hit — per-position SL triggered | 2024-12-27 10:50:00 | 1707.25 | 1711.04 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:55:00 | 1797.00 | 1786.58 | 0.00 | ORB-long ORB[1762.55,1788.35] vol=5.0x ATR=8.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:25:00 | 1810.38 | 1793.50 | 0.00 | T1 1.5R @ 1810.38 |
| Stop hit — per-position SL triggered | 2025-01-02 13:10:00 | 1797.00 | 1797.06 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 09:35:00 | 1774.40 | 1760.63 | 0.00 | ORB-long ORB[1743.00,1769.15] vol=2.1x ATR=8.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:50:00 | 1786.42 | 1766.82 | 0.00 | T1 1.5R @ 1786.42 |
| Target hit | 2025-01-08 10:20:00 | 1788.25 | 1794.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-01-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:05:00 | 1752.40 | 1758.10 | 0.00 | ORB-short ORB[1765.00,1784.85] vol=6.0x ATR=6.03 |
| Stop hit — per-position SL triggered | 2025-01-09 10:25:00 | 1758.43 | 1756.88 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:25:00 | 1623.55 | 1633.83 | 0.00 | ORB-short ORB[1640.00,1655.00] vol=3.1x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:45:00 | 1615.59 | 1630.31 | 0.00 | T1 1.5R @ 1615.59 |
| Stop hit — per-position SL triggered | 2025-01-21 10:50:00 | 1623.55 | 1630.09 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:05:00 | 1615.55 | 1610.70 | 0.00 | ORB-long ORB[1598.60,1608.50] vol=2.8x ATR=5.20 |
| Stop hit — per-position SL triggered | 2025-01-30 10:55:00 | 1610.35 | 1610.63 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-02-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:35:00 | 1660.65 | 1657.60 | 0.00 | ORB-long ORB[1641.60,1655.90] vol=15.6x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 09:45:00 | 1670.50 | 1658.04 | 0.00 | T1 1.5R @ 1670.50 |
| Stop hit — per-position SL triggered | 2025-02-06 10:25:00 | 1660.65 | 1659.01 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-02-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:40:00 | 1631.40 | 1639.13 | 0.00 | ORB-short ORB[1635.00,1655.00] vol=1.5x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:50:00 | 1623.07 | 1637.29 | 0.00 | T1 1.5R @ 1623.07 |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 1631.40 | 1634.77 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:35:00 | 1617.80 | 1628.91 | 0.00 | ORB-short ORB[1624.60,1644.50] vol=2.9x ATR=8.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:00:00 | 1604.42 | 1624.21 | 0.00 | T1 1.5R @ 1604.42 |
| Stop hit — per-position SL triggered | 2025-02-11 15:00:00 | 1617.80 | 1613.42 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:15:00 | 1438.30 | 1474.19 | 0.00 | ORB-short ORB[1495.00,1510.45] vol=2.3x ATR=9.96 |
| Stop hit — per-position SL triggered | 2025-02-25 10:30:00 | 1448.26 | 1462.01 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:45:00 | 1251.25 | 1263.76 | 0.00 | ORB-short ORB[1261.00,1277.00] vol=2.5x ATR=8.21 |
| Stop hit — per-position SL triggered | 2025-03-20 10:00:00 | 1259.46 | 1262.19 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:00:00 | 1060.40 | 1067.73 | 0.00 | ORB-short ORB[1066.40,1081.10] vol=2.1x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 1064.94 | 1067.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 09:50:00 | 1467.00 | 2024-05-17 09:55:00 | 1482.34 | PARTIAL | 0.50 | 1.05% |
| BUY | retest1 | 2024-05-17 09:50:00 | 1467.00 | 2024-05-17 11:20:00 | 1467.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-21 10:10:00 | 1437.60 | 2024-05-21 10:15:00 | 1445.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-05-22 10:20:00 | 1399.60 | 2024-05-22 10:25:00 | 1406.46 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-05-27 10:10:00 | 1410.60 | 2024-05-27 10:20:00 | 1418.54 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-27 10:10:00 | 1410.60 | 2024-05-27 13:50:00 | 1438.05 | TARGET_HIT | 0.50 | 1.95% |
| SELL | retest1 | 2024-05-28 11:05:00 | 1422.00 | 2024-05-28 11:10:00 | 1426.95 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-29 09:50:00 | 1423.05 | 2024-05-29 10:10:00 | 1428.50 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-05-29 09:50:00 | 1423.05 | 2024-05-29 10:30:00 | 1423.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-11 09:35:00 | 1512.70 | 2024-06-11 09:45:00 | 1520.71 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-06-21 10:50:00 | 1580.00 | 2024-06-21 14:55:00 | 1585.99 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-28 10:05:00 | 1887.20 | 2024-06-28 10:15:00 | 1895.43 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-09 09:30:00 | 1835.75 | 2024-07-09 09:35:00 | 1843.72 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-09 09:30:00 | 1835.75 | 2024-07-09 09:50:00 | 1835.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-18 09:40:00 | 1817.00 | 2024-07-18 09:45:00 | 1827.12 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-07-18 09:40:00 | 1817.00 | 2024-07-18 09:55:00 | 1817.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:35:00 | 1746.75 | 2024-07-26 10:40:00 | 1742.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-30 09:45:00 | 1760.00 | 2024-07-30 09:50:00 | 1751.58 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-31 10:40:00 | 1747.35 | 2024-07-31 10:45:00 | 1756.51 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-31 10:40:00 | 1747.35 | 2024-07-31 11:30:00 | 1747.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 09:35:00 | 1737.40 | 2024-08-12 09:45:00 | 1732.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-23 09:50:00 | 1713.05 | 2024-08-23 09:55:00 | 1703.24 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-08-26 10:50:00 | 1741.95 | 2024-08-26 11:15:00 | 1734.34 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-30 09:30:00 | 1691.95 | 2024-08-30 09:40:00 | 1684.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-09-03 09:55:00 | 1828.30 | 2024-09-03 14:35:00 | 1843.52 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-09-03 09:55:00 | 1828.30 | 2024-09-03 15:20:00 | 1850.50 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2024-09-10 11:15:00 | 1770.35 | 2024-09-10 11:40:00 | 1774.76 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-16 10:35:00 | 1776.00 | 2024-09-16 10:45:00 | 1784.47 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-09-23 09:55:00 | 1758.60 | 2024-09-23 11:25:00 | 1746.42 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-09-23 09:55:00 | 1758.60 | 2024-09-23 15:20:00 | 1752.00 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-27 10:00:00 | 1768.00 | 2024-09-27 10:05:00 | 1773.31 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-09-27 10:00:00 | 1768.00 | 2024-09-27 10:15:00 | 1768.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 11:05:00 | 1708.85 | 2024-10-18 11:55:00 | 1701.42 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-05 10:50:00 | 1592.85 | 2024-11-05 11:30:00 | 1584.52 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-05 10:50:00 | 1592.85 | 2024-11-05 15:20:00 | 1570.35 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-11-21 10:10:00 | 1530.00 | 2024-11-21 11:05:00 | 1542.80 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2024-11-21 10:10:00 | 1530.00 | 2024-11-21 12:00:00 | 1540.80 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2024-12-10 10:00:00 | 1547.70 | 2024-12-10 10:25:00 | 1540.91 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-11 11:10:00 | 1582.70 | 2024-12-11 11:20:00 | 1590.87 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-11 11:10:00 | 1582.70 | 2024-12-11 15:20:00 | 1650.00 | TARGET_HIT | 0.50 | 4.25% |
| SELL | retest1 | 2024-12-13 10:25:00 | 1606.50 | 2024-12-13 10:30:00 | 1595.52 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-12-13 10:25:00 | 1606.50 | 2024-12-13 10:50:00 | 1606.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 09:30:00 | 1709.95 | 2024-12-20 09:45:00 | 1702.39 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-27 10:15:00 | 1707.25 | 2024-12-27 10:30:00 | 1718.62 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-12-27 10:15:00 | 1707.25 | 2024-12-27 10:50:00 | 1707.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1797.00 | 2025-01-02 11:25:00 | 1810.38 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1797.00 | 2025-01-02 13:10:00 | 1797.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-08 09:35:00 | 1774.40 | 2025-01-08 09:50:00 | 1786.42 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-08 09:35:00 | 1774.40 | 2025-01-08 10:20:00 | 1788.25 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-01-09 10:05:00 | 1752.40 | 2025-01-09 10:25:00 | 1758.43 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-21 10:25:00 | 1623.55 | 2025-01-21 10:45:00 | 1615.59 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-21 10:25:00 | 1623.55 | 2025-01-21 10:50:00 | 1623.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 10:05:00 | 1615.55 | 2025-01-30 10:55:00 | 1610.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-06 09:35:00 | 1660.65 | 2025-02-06 09:45:00 | 1670.50 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-02-06 09:35:00 | 1660.65 | 2025-02-06 10:25:00 | 1660.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 10:40:00 | 1631.40 | 2025-02-10 10:50:00 | 1623.07 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-02-10 10:40:00 | 1631.40 | 2025-02-10 11:15:00 | 1631.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 10:35:00 | 1617.80 | 2025-02-11 11:00:00 | 1604.42 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2025-02-11 10:35:00 | 1617.80 | 2025-02-11 15:00:00 | 1617.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-25 10:15:00 | 1438.30 | 2025-02-25 10:30:00 | 1448.26 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-03-20 09:45:00 | 1251.25 | 2025-03-20 10:00:00 | 1259.46 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-04-23 11:00:00 | 1060.40 | 2025-04-23 11:15:00 | 1064.94 | STOP_HIT | 1.00 | -0.43% |
