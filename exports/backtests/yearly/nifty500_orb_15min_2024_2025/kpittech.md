# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 725.00
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 8 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 40
- **Target hits / Stop hits / Partials:** 8 / 40 / 21
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 18.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 15 | 40.5% | 3 | 22 | 12 | 0.26% | 9.5% |
| BUY @ 2nd Alert (retest1) | 37 | 15 | 40.5% | 3 | 22 | 12 | 0.26% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.27% | 8.6% |
| SELL @ 2nd Alert (retest1) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.27% | 8.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 69 | 29 | 42.0% | 8 | 40 | 21 | 0.26% | 18.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 11:05:00 | 1515.60 | 1518.51 | 0.00 | ORB-short ORB[1518.05,1539.95] vol=1.9x ATR=3.95 |
| Stop hit — per-position SL triggered | 2024-05-17 14:00:00 | 1519.55 | 1516.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:25:00 | 1458.55 | 1462.71 | 0.00 | ORB-short ORB[1463.00,1482.85] vol=2.3x ATR=4.97 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 1463.52 | 1462.71 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 1513.00 | 1502.36 | 0.00 | ORB-long ORB[1490.40,1507.95] vol=4.3x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-06-13 10:05:00 | 1506.94 | 1506.47 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:15:00 | 1504.95 | 1490.62 | 0.00 | ORB-long ORB[1473.40,1492.80] vol=2.7x ATR=4.83 |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 1500.12 | 1497.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:05:00 | 1609.20 | 1595.49 | 0.00 | ORB-long ORB[1584.95,1602.00] vol=1.7x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:20:00 | 1618.45 | 1601.86 | 0.00 | T1 1.5R @ 1618.45 |
| Stop hit — per-position SL triggered | 2024-06-26 10:40:00 | 1609.20 | 1603.34 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:40:00 | 1721.15 | 1712.97 | 0.00 | ORB-long ORB[1696.00,1716.45] vol=4.9x ATR=7.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 09:45:00 | 1732.24 | 1715.55 | 0.00 | T1 1.5R @ 1732.24 |
| Stop hit — per-position SL triggered | 2024-07-11 09:50:00 | 1721.15 | 1716.06 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:40:00 | 1736.75 | 1718.17 | 0.00 | ORB-long ORB[1711.00,1732.25] vol=3.3x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:45:00 | 1745.76 | 1724.16 | 0.00 | T1 1.5R @ 1745.76 |
| Target hit | 2024-07-12 15:20:00 | 1861.85 | 1836.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:55:00 | 1787.00 | 1803.53 | 0.00 | ORB-short ORB[1797.25,1821.50] vol=1.6x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:25:00 | 1776.05 | 1800.03 | 0.00 | T1 1.5R @ 1776.05 |
| Target hit | 2024-08-13 15:20:00 | 1749.00 | 1777.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-08-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:50:00 | 1839.90 | 1819.09 | 0.00 | ORB-long ORB[1815.50,1836.80] vol=1.6x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-08-19 11:05:00 | 1832.78 | 1823.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:55:00 | 1866.50 | 1843.81 | 0.00 | ORB-long ORB[1832.00,1848.70] vol=5.3x ATR=6.59 |
| Stop hit — per-position SL triggered | 2024-08-22 11:00:00 | 1859.91 | 1845.99 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:40:00 | 1911.50 | 1888.51 | 0.00 | ORB-long ORB[1846.40,1875.00] vol=4.9x ATR=11.06 |
| Stop hit — per-position SL triggered | 2024-08-28 10:50:00 | 1900.44 | 1891.30 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:15:00 | 1865.90 | 1853.28 | 0.00 | ORB-long ORB[1839.05,1863.50] vol=1.7x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:25:00 | 1874.32 | 1857.39 | 0.00 | T1 1.5R @ 1874.32 |
| Stop hit — per-position SL triggered | 2024-08-29 10:30:00 | 1865.90 | 1857.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1749.70 | 1759.43 | 0.00 | ORB-short ORB[1750.00,1774.70] vol=1.6x ATR=4.61 |
| Stop hit — per-position SL triggered | 2024-09-04 10:35:00 | 1754.31 | 1758.77 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1733.60 | 1742.98 | 0.00 | ORB-short ORB[1740.40,1757.40] vol=1.8x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-09-05 09:35:00 | 1737.47 | 1741.73 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:35:00 | 1727.00 | 1731.64 | 0.00 | ORB-short ORB[1727.30,1743.35] vol=2.8x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 13:00:00 | 1717.21 | 1729.56 | 0.00 | T1 1.5R @ 1717.21 |
| Stop hit — per-position SL triggered | 2024-09-06 13:30:00 | 1727.00 | 1729.22 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:10:00 | 1744.30 | 1755.88 | 0.00 | ORB-short ORB[1752.50,1774.95] vol=1.8x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:25:00 | 1736.20 | 1754.47 | 0.00 | T1 1.5R @ 1736.20 |
| Stop hit — per-position SL triggered | 2024-09-10 14:00:00 | 1744.30 | 1743.41 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 1780.50 | 1761.79 | 0.00 | ORB-long ORB[1740.65,1755.25] vol=5.6x ATR=5.77 |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 1774.73 | 1765.23 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:20:00 | 1838.65 | 1825.59 | 0.00 | ORB-long ORB[1802.50,1829.00] vol=2.4x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:45:00 | 1849.72 | 1830.51 | 0.00 | T1 1.5R @ 1849.72 |
| Stop hit — per-position SL triggered | 2024-09-12 10:55:00 | 1838.65 | 1831.33 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:10:00 | 1745.35 | 1735.46 | 0.00 | ORB-long ORB[1721.10,1741.15] vol=1.5x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:20:00 | 1753.57 | 1737.78 | 0.00 | T1 1.5R @ 1753.57 |
| Stop hit — per-position SL triggered | 2024-09-18 10:45:00 | 1745.35 | 1740.50 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 1634.85 | 1640.21 | 0.00 | ORB-short ORB[1635.10,1654.60] vol=1.8x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 1639.36 | 1640.16 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:25:00 | 1672.30 | 1692.02 | 0.00 | ORB-short ORB[1692.00,1705.00] vol=1.7x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:35:00 | 1661.85 | 1689.14 | 0.00 | T1 1.5R @ 1661.85 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1672.30 | 1678.05 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:10:00 | 1737.15 | 1729.32 | 0.00 | ORB-long ORB[1714.05,1732.85] vol=2.2x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 12:05:00 | 1745.03 | 1733.28 | 0.00 | T1 1.5R @ 1745.03 |
| Target hit | 2024-10-09 14:40:00 | 1740.25 | 1744.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:40:00 | 1769.70 | 1757.28 | 0.00 | ORB-long ORB[1740.60,1765.05] vol=1.7x ATR=6.78 |
| Stop hit — per-position SL triggered | 2024-10-10 10:00:00 | 1762.92 | 1760.12 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 1749.65 | 1743.64 | 0.00 | ORB-long ORB[1721.05,1746.00] vol=4.0x ATR=7.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:25:00 | 1760.23 | 1748.60 | 0.00 | T1 1.5R @ 1760.23 |
| Stop hit — per-position SL triggered | 2024-10-11 11:20:00 | 1749.65 | 1750.33 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:10:00 | 1699.00 | 1708.95 | 0.00 | ORB-short ORB[1725.55,1744.95] vol=8.0x ATR=7.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:25:00 | 1687.79 | 1707.28 | 0.00 | T1 1.5R @ 1687.79 |
| Stop hit — per-position SL triggered | 2024-10-22 12:25:00 | 1699.00 | 1702.74 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:15:00 | 1314.00 | 1296.47 | 0.00 | ORB-long ORB[1285.05,1300.00] vol=1.7x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-11-22 10:20:00 | 1307.85 | 1298.17 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:05:00 | 1388.15 | 1394.95 | 0.00 | ORB-short ORB[1390.00,1409.00] vol=1.7x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:10:00 | 1381.15 | 1390.34 | 0.00 | T1 1.5R @ 1381.15 |
| Target hit | 2024-11-29 15:20:00 | 1368.55 | 1372.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-12-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:45:00 | 1511.95 | 1497.71 | 0.00 | ORB-long ORB[1480.55,1495.90] vol=4.0x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:05:00 | 1520.49 | 1504.55 | 0.00 | T1 1.5R @ 1520.49 |
| Target hit | 2024-12-09 14:20:00 | 1525.55 | 1530.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2024-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:00:00 | 1530.25 | 1535.20 | 0.00 | ORB-short ORB[1533.00,1545.40] vol=4.0x ATR=3.81 |
| Stop hit — per-position SL triggered | 2024-12-11 11:20:00 | 1534.06 | 1534.69 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 1508.65 | 1523.79 | 0.00 | ORB-short ORB[1520.00,1540.80] vol=2.1x ATR=5.43 |
| Stop hit — per-position SL triggered | 2024-12-13 10:35:00 | 1514.08 | 1522.41 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:30:00 | 1544.40 | 1531.89 | 0.00 | ORB-long ORB[1520.20,1539.70] vol=2.4x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:40:00 | 1551.91 | 1536.91 | 0.00 | T1 1.5R @ 1551.91 |
| Stop hit — per-position SL triggered | 2024-12-16 11:10:00 | 1544.40 | 1539.94 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:45:00 | 1502.80 | 1520.55 | 0.00 | ORB-short ORB[1516.00,1535.90] vol=1.6x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:05:00 | 1493.16 | 1516.63 | 0.00 | T1 1.5R @ 1493.16 |
| Target hit | 2024-12-20 15:20:00 | 1451.55 | 1480.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 1433.20 | 1425.14 | 0.00 | ORB-long ORB[1415.00,1429.95] vol=2.1x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:55:00 | 1441.94 | 1429.75 | 0.00 | T1 1.5R @ 1441.94 |
| Stop hit — per-position SL triggered | 2024-12-24 12:10:00 | 1433.20 | 1434.05 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:10:00 | 1469.45 | 1458.48 | 0.00 | ORB-long ORB[1450.00,1466.95] vol=3.1x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-12-27 11:35:00 | 1464.94 | 1459.81 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:45:00 | 1481.65 | 1486.21 | 0.00 | ORB-short ORB[1482.20,1493.95] vol=1.6x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:55:00 | 1476.08 | 1484.60 | 0.00 | T1 1.5R @ 1476.08 |
| Target hit | 2025-01-03 11:20:00 | 1475.05 | 1472.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1446.80 | 1452.63 | 0.00 | ORB-short ORB[1448.50,1458.80] vol=2.3x ATR=4.43 |
| Stop hit — per-position SL triggered | 2025-01-06 11:25:00 | 1451.23 | 1452.58 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:15:00 | 1303.95 | 1313.89 | 0.00 | ORB-short ORB[1314.80,1327.60] vol=2.0x ATR=4.62 |
| Stop hit — per-position SL triggered | 2025-01-17 10:30:00 | 1308.57 | 1307.74 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:35:00 | 1311.40 | 1322.74 | 0.00 | ORB-short ORB[1321.30,1339.00] vol=1.7x ATR=4.23 |
| Stop hit — per-position SL triggered | 2025-01-21 10:45:00 | 1315.63 | 1321.97 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-27 10:15:00 | 1303.00 | 1295.70 | 0.00 | ORB-long ORB[1285.80,1302.90] vol=3.9x ATR=7.39 |
| Stop hit — per-position SL triggered | 2025-01-27 11:00:00 | 1295.61 | 1296.80 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:20:00 | 1420.00 | 1432.77 | 0.00 | ORB-short ORB[1437.00,1454.55] vol=1.7x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-02-10 10:25:00 | 1424.60 | 1431.81 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-02-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:05:00 | 1317.40 | 1335.00 | 0.00 | ORB-short ORB[1336.20,1353.25] vol=2.4x ATR=5.56 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 1322.96 | 1334.54 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:30:00 | 1342.10 | 1335.96 | 0.00 | ORB-long ORB[1328.05,1339.75] vol=1.6x ATR=4.85 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 1337.25 | 1338.16 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:00:00 | 1276.50 | 1256.59 | 0.00 | ORB-long ORB[1240.05,1257.45] vol=1.7x ATR=4.70 |
| Stop hit — per-position SL triggered | 2025-03-19 11:05:00 | 1271.80 | 1256.99 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:25:00 | 1325.40 | 1334.73 | 0.00 | ORB-short ORB[1334.00,1349.00] vol=1.6x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-03-28 10:30:00 | 1329.93 | 1334.40 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-04-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:10:00 | 1293.00 | 1300.52 | 0.00 | ORB-short ORB[1294.70,1312.95] vol=1.6x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-04-01 10:20:00 | 1298.05 | 1300.21 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:30:00 | 1157.70 | 1151.15 | 0.00 | ORB-long ORB[1143.30,1155.00] vol=2.6x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:35:00 | 1164.77 | 1154.54 | 0.00 | T1 1.5R @ 1164.77 |
| Stop hit — per-position SL triggered | 2025-04-16 09:45:00 | 1157.70 | 1155.65 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:45:00 | 1211.00 | 1199.82 | 0.00 | ORB-long ORB[1179.90,1198.00] vol=2.0x ATR=6.59 |
| Stop hit — per-position SL triggered | 2025-04-23 10:10:00 | 1204.41 | 1203.94 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 1212.10 | 1221.36 | 0.00 | ORB-short ORB[1216.00,1233.80] vol=2.6x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 1203.35 | 1219.20 | 0.00 | T1 1.5R @ 1203.35 |
| Target hit | 2025-04-25 13:10:00 | 1206.50 | 1198.17 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 11:05:00 | 1515.60 | 2024-05-17 14:00:00 | 1519.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-30 10:25:00 | 1458.55 | 2024-05-30 10:35:00 | 1463.52 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-13 09:35:00 | 1513.00 | 2024-06-13 10:05:00 | 1506.94 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-18 10:15:00 | 1504.95 | 2024-06-18 11:15:00 | 1500.12 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-26 10:05:00 | 1609.20 | 2024-06-26 10:20:00 | 1618.45 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-26 10:05:00 | 1609.20 | 2024-06-26 10:40:00 | 1609.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:40:00 | 1721.15 | 2024-07-11 09:45:00 | 1732.24 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-11 09:40:00 | 1721.15 | 2024-07-11 09:50:00 | 1721.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 10:40:00 | 1736.75 | 2024-07-12 10:45:00 | 1745.76 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-12 10:40:00 | 1736.75 | 2024-07-12 15:20:00 | 1861.85 | TARGET_HIT | 0.50 | 7.20% |
| SELL | retest1 | 2024-08-13 10:55:00 | 1787.00 | 2024-08-13 11:25:00 | 1776.05 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-08-13 10:55:00 | 1787.00 | 2024-08-13 15:20:00 | 1749.00 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2024-08-19 10:50:00 | 1839.90 | 2024-08-19 11:05:00 | 1832.78 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-22 10:55:00 | 1866.50 | 2024-08-22 11:00:00 | 1859.91 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-28 10:40:00 | 1911.50 | 2024-08-28 10:50:00 | 1900.44 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-08-29 10:15:00 | 1865.90 | 2024-08-29 10:25:00 | 1874.32 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-29 10:15:00 | 1865.90 | 2024-08-29 10:30:00 | 1865.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-04 10:30:00 | 1749.70 | 2024-09-04 10:35:00 | 1754.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-05 09:30:00 | 1733.60 | 2024-09-05 09:35:00 | 1737.47 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 10:35:00 | 1727.00 | 2024-09-06 13:00:00 | 1717.21 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-06 10:35:00 | 1727.00 | 2024-09-06 13:30:00 | 1727.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 11:10:00 | 1744.30 | 2024-09-10 11:25:00 | 1736.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-10 11:10:00 | 1744.30 | 2024-09-10 14:00:00 | 1744.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:55:00 | 1780.50 | 2024-09-11 11:15:00 | 1774.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-12 10:20:00 | 1838.65 | 2024-09-12 10:45:00 | 1849.72 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-09-12 10:20:00 | 1838.65 | 2024-09-12 10:55:00 | 1838.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 10:10:00 | 1745.35 | 2024-09-18 10:20:00 | 1753.57 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-18 10:10:00 | 1745.35 | 2024-09-18 10:45:00 | 1745.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 11:10:00 | 1634.85 | 2024-10-01 11:15:00 | 1639.36 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-07 10:25:00 | 1672.30 | 2024-10-07 10:35:00 | 1661.85 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-07 10:25:00 | 1672.30 | 2024-10-07 11:15:00 | 1672.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 11:10:00 | 1737.15 | 2024-10-09 12:05:00 | 1745.03 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-09 11:10:00 | 1737.15 | 2024-10-09 14:40:00 | 1740.25 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2024-10-10 09:40:00 | 1769.70 | 2024-10-10 10:00:00 | 1762.92 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-11 09:35:00 | 1749.65 | 2024-10-11 10:25:00 | 1760.23 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-10-11 09:35:00 | 1749.65 | 2024-10-11 11:20:00 | 1749.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:10:00 | 1699.00 | 2024-10-22 10:25:00 | 1687.79 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-22 10:10:00 | 1699.00 | 2024-10-22 12:25:00 | 1699.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:15:00 | 1314.00 | 2024-11-22 10:20:00 | 1307.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-11-29 10:05:00 | 1388.15 | 2024-11-29 10:10:00 | 1381.15 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-29 10:05:00 | 1388.15 | 2024-11-29 15:20:00 | 1368.55 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-12-09 09:45:00 | 1511.95 | 2024-12-09 10:05:00 | 1520.49 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-09 09:45:00 | 1511.95 | 2024-12-09 14:20:00 | 1525.55 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-12-11 11:00:00 | 1530.25 | 2024-12-11 11:20:00 | 1534.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-13 10:30:00 | 1508.65 | 2024-12-13 10:35:00 | 1514.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-16 10:30:00 | 1544.40 | 2024-12-16 10:40:00 | 1551.91 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-16 10:30:00 | 1544.40 | 2024-12-16 11:10:00 | 1544.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:45:00 | 1502.80 | 2024-12-20 10:05:00 | 1493.16 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-12-20 09:45:00 | 1502.80 | 2024-12-20 15:20:00 | 1451.55 | TARGET_HIT | 0.50 | 3.41% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1433.20 | 2024-12-24 09:55:00 | 1441.94 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1433.20 | 2024-12-24 12:10:00 | 1433.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 11:10:00 | 1469.45 | 2024-12-27 11:35:00 | 1464.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-03 09:45:00 | 1481.65 | 2025-01-03 09:55:00 | 1476.08 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-03 09:45:00 | 1481.65 | 2025-01-03 11:20:00 | 1475.05 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-06 11:15:00 | 1446.80 | 2025-01-06 11:25:00 | 1451.23 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-17 10:15:00 | 1303.95 | 2025-01-17 10:30:00 | 1308.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-21 10:35:00 | 1311.40 | 2025-01-21 10:45:00 | 1315.63 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-27 10:15:00 | 1303.00 | 2025-01-27 11:00:00 | 1295.61 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-02-10 10:20:00 | 1420.00 | 2025-02-10 10:25:00 | 1424.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-14 11:05:00 | 1317.40 | 2025-02-14 11:10:00 | 1322.96 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-20 09:30:00 | 1342.10 | 2025-02-20 09:45:00 | 1337.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-19 11:00:00 | 1276.50 | 2025-03-19 11:05:00 | 1271.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-28 10:25:00 | 1325.40 | 2025-03-28 10:30:00 | 1329.93 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-01 10:10:00 | 1293.00 | 2025-04-01 10:20:00 | 1298.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-04-16 09:30:00 | 1157.70 | 2025-04-16 09:35:00 | 1164.77 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-04-16 09:30:00 | 1157.70 | 2025-04-16 09:45:00 | 1157.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-23 09:45:00 | 1211.00 | 2025-04-23 10:10:00 | 1204.41 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1212.10 | 2025-04-25 09:35:00 | 1203.35 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1212.10 | 2025-04-25 13:10:00 | 1206.50 | TARGET_HIT | 0.50 | 0.46% |
