# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (15088 bars)
- **Last close:** 1255.00
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 21 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 63
- **Target hits / Stop hits / Partials:** 21 / 63 / 40
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 35.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 30 | 45.5% | 10 | 36 | 20 | 0.25% | 16.5% |
| BUY @ 2nd Alert (retest1) | 66 | 30 | 45.5% | 10 | 36 | 20 | 0.25% | 16.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 31 | 53.4% | 11 | 27 | 20 | 0.33% | 19.2% |
| SELL @ 2nd Alert (retest1) | 58 | 31 | 53.4% | 11 | 27 | 20 | 0.33% | 19.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 124 | 61 | 49.2% | 21 | 63 | 40 | 0.29% | 35.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 11:15:00 | 1309.00 | 1294.66 | 0.00 | ORB-long ORB[1275.40,1290.90] vol=1.9x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:25:00 | 1315.72 | 1295.89 | 0.00 | T1 1.5R @ 1315.72 |
| Stop hit — per-position SL triggered | 2025-05-13 13:00:00 | 1309.00 | 1302.44 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:35:00 | 1357.90 | 1350.98 | 0.00 | ORB-long ORB[1337.10,1356.60] vol=1.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2025-05-16 09:55:00 | 1354.05 | 1353.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 1454.00 | 1462.52 | 0.00 | ORB-short ORB[1456.00,1473.30] vol=1.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-05-27 09:35:00 | 1458.15 | 1462.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:40:00 | 1553.70 | 1545.22 | 0.00 | ORB-long ORB[1531.10,1551.00] vol=2.1x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:50:00 | 1562.02 | 1549.18 | 0.00 | T1 1.5R @ 1562.02 |
| Target hit | 2025-06-02 15:20:00 | 1682.60 | 1642.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1706.70 | 1716.16 | 0.00 | ORB-short ORB[1707.10,1726.40] vol=1.6x ATR=6.52 |
| Stop hit — per-position SL triggered | 2025-06-05 09:40:00 | 1713.22 | 1715.61 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1802.40 | 1789.52 | 0.00 | ORB-long ORB[1773.00,1799.80] vol=2.5x ATR=8.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:40:00 | 1815.17 | 1795.80 | 0.00 | T1 1.5R @ 1815.17 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 1802.40 | 1796.37 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:40:00 | 1823.90 | 1812.17 | 0.00 | ORB-long ORB[1799.00,1819.90] vol=1.8x ATR=6.15 |
| Stop hit — per-position SL triggered | 2025-06-10 09:50:00 | 1817.75 | 1814.96 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 1706.40 | 1693.99 | 0.00 | ORB-long ORB[1680.40,1697.00] vol=2.8x ATR=6.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:40:00 | 1716.89 | 1698.05 | 0.00 | T1 1.5R @ 1716.89 |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 1706.40 | 1705.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 1691.00 | 1683.02 | 0.00 | ORB-long ORB[1672.30,1689.60] vol=2.6x ATR=6.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:55:00 | 1701.39 | 1687.88 | 0.00 | T1 1.5R @ 1701.39 |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 1691.00 | 1688.40 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 1674.90 | 1680.50 | 0.00 | ORB-short ORB[1675.00,1685.90] vol=1.6x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:30:00 | 1667.93 | 1676.69 | 0.00 | T1 1.5R @ 1667.93 |
| Target hit | 2025-06-19 15:20:00 | 1635.50 | 1656.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1728.20 | 1740.85 | 0.00 | ORB-short ORB[1733.00,1755.00] vol=2.5x ATR=5.81 |
| Stop hit — per-position SL triggered | 2025-06-26 11:30:00 | 1734.01 | 1740.19 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1776.20 | 1772.54 | 0.00 | ORB-long ORB[1764.00,1775.80] vol=2.1x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 09:40:00 | 1783.71 | 1776.40 | 0.00 | T1 1.5R @ 1783.71 |
| Stop hit — per-position SL triggered | 2025-06-30 09:50:00 | 1776.20 | 1776.89 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:55:00 | 1807.00 | 1788.64 | 0.00 | ORB-long ORB[1764.70,1787.90] vol=4.5x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-07-03 11:25:00 | 1801.86 | 1792.90 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:30:00 | 1775.30 | 1786.32 | 0.00 | ORB-short ORB[1780.00,1794.90] vol=2.3x ATR=5.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:35:00 | 1767.66 | 1783.08 | 0.00 | T1 1.5R @ 1767.66 |
| Target hit | 2025-07-08 13:10:00 | 1753.60 | 1752.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 1768.30 | 1759.20 | 0.00 | ORB-long ORB[1750.10,1766.40] vol=1.8x ATR=5.40 |
| Stop hit — per-position SL triggered | 2025-07-09 09:35:00 | 1762.90 | 1759.74 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:55:00 | 1724.00 | 1737.48 | 0.00 | ORB-short ORB[1734.00,1749.00] vol=2.0x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:00:00 | 1716.47 | 1734.77 | 0.00 | T1 1.5R @ 1716.47 |
| Target hit | 2025-07-11 15:20:00 | 1687.10 | 1708.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-07-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:35:00 | 1720.30 | 1711.97 | 0.00 | ORB-long ORB[1695.00,1716.90] vol=1.5x ATR=4.51 |
| Stop hit — per-position SL triggered | 2025-07-16 10:40:00 | 1715.79 | 1712.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1718.30 | 1725.61 | 0.00 | ORB-short ORB[1723.50,1731.30] vol=1.8x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-07-17 09:35:00 | 1722.24 | 1724.52 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1702.40 | 1711.67 | 0.00 | ORB-short ORB[1708.00,1716.00] vol=2.1x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:10:00 | 1695.80 | 1709.64 | 0.00 | T1 1.5R @ 1695.80 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 1702.40 | 1703.43 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:35:00 | 1714.50 | 1705.41 | 0.00 | ORB-long ORB[1690.30,1712.30] vol=2.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2025-07-21 09:40:00 | 1709.32 | 1705.86 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:35:00 | 1740.70 | 1745.71 | 0.00 | ORB-short ORB[1741.60,1753.90] vol=1.7x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:40:00 | 1733.33 | 1744.88 | 0.00 | T1 1.5R @ 1733.33 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 1740.70 | 1744.45 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:35:00 | 1683.80 | 1700.60 | 0.00 | ORB-short ORB[1703.10,1724.00] vol=4.1x ATR=5.84 |
| Stop hit — per-position SL triggered | 2025-07-23 09:45:00 | 1689.64 | 1696.66 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:05:00 | 1641.70 | 1655.38 | 0.00 | ORB-short ORB[1648.70,1668.80] vol=2.7x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:35:00 | 1632.29 | 1650.45 | 0.00 | T1 1.5R @ 1632.29 |
| Stop hit — per-position SL triggered | 2025-07-25 11:05:00 | 1641.70 | 1648.66 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:45:00 | 1471.90 | 1481.35 | 0.00 | ORB-short ORB[1475.00,1486.00] vol=2.3x ATR=5.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 09:55:00 | 1463.29 | 1478.77 | 0.00 | T1 1.5R @ 1463.29 |
| Stop hit — per-position SL triggered | 2025-08-01 10:00:00 | 1471.90 | 1478.34 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:55:00 | 1551.60 | 1566.32 | 0.00 | ORB-short ORB[1560.00,1577.40] vol=2.6x ATR=6.32 |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 1557.92 | 1564.04 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 1544.00 | 1552.54 | 0.00 | ORB-short ORB[1549.00,1565.50] vol=1.9x ATR=5.17 |
| Stop hit — per-position SL triggered | 2025-08-06 09:40:00 | 1549.17 | 1551.57 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 11:15:00 | 1575.10 | 1565.80 | 0.00 | ORB-long ORB[1550.10,1570.50] vol=3.1x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 11:45:00 | 1585.74 | 1568.73 | 0.00 | T1 1.5R @ 1585.74 |
| Target hit | 2025-08-08 14:25:00 | 1578.20 | 1578.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-08-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:05:00 | 1566.30 | 1558.02 | 0.00 | ORB-long ORB[1543.90,1562.00] vol=1.9x ATR=4.69 |
| Stop hit — per-position SL triggered | 2025-08-13 10:35:00 | 1561.61 | 1559.85 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 1569.90 | 1574.81 | 0.00 | ORB-short ORB[1570.00,1583.40] vol=1.6x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:35:00 | 1563.30 | 1573.26 | 0.00 | T1 1.5R @ 1563.30 |
| Stop hit — per-position SL triggered | 2025-08-18 09:40:00 | 1569.90 | 1573.16 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:15:00 | 1598.80 | 1589.82 | 0.00 | ORB-long ORB[1581.10,1593.80] vol=4.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:30:00 | 1603.54 | 1592.58 | 0.00 | T1 1.5R @ 1603.54 |
| Stop hit — per-position SL triggered | 2025-08-21 11:50:00 | 1598.80 | 1595.96 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 1580.50 | 1575.22 | 0.00 | ORB-long ORB[1565.00,1579.00] vol=1.6x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:50:00 | 1587.30 | 1578.12 | 0.00 | T1 1.5R @ 1587.30 |
| Stop hit — per-position SL triggered | 2025-08-22 10:35:00 | 1580.50 | 1580.68 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:40:00 | 1559.80 | 1565.67 | 0.00 | ORB-short ORB[1563.00,1579.30] vol=2.0x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 13:05:00 | 1553.68 | 1563.71 | 0.00 | T1 1.5R @ 1553.68 |
| Target hit | 2025-08-25 15:20:00 | 1541.60 | 1556.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 1521.50 | 1528.12 | 0.00 | ORB-short ORB[1523.00,1539.90] vol=1.8x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 1515.22 | 1526.39 | 0.00 | T1 1.5R @ 1515.22 |
| Stop hit — per-position SL triggered | 2025-08-26 10:00:00 | 1521.50 | 1521.80 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:50:00 | 1469.60 | 1453.83 | 0.00 | ORB-long ORB[1431.30,1453.20] vol=1.7x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-09-01 10:20:00 | 1464.37 | 1458.14 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:00:00 | 1492.40 | 1478.58 | 0.00 | ORB-long ORB[1463.30,1484.00] vol=1.5x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:10:00 | 1498.78 | 1482.64 | 0.00 | T1 1.5R @ 1498.78 |
| Target hit | 2025-09-02 13:15:00 | 1511.50 | 1512.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 1520.30 | 1516.77 | 0.00 | ORB-long ORB[1507.10,1519.60] vol=2.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-09-03 09:40:00 | 1515.57 | 1516.88 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 1538.00 | 1531.15 | 0.00 | ORB-long ORB[1523.40,1534.70] vol=1.8x ATR=3.89 |
| Stop hit — per-position SL triggered | 2025-09-08 09:55:00 | 1534.11 | 1534.45 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 1542.80 | 1548.88 | 0.00 | ORB-short ORB[1545.40,1559.70] vol=1.8x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-09-09 09:35:00 | 1546.56 | 1548.59 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:30:00 | 1565.10 | 1558.86 | 0.00 | ORB-long ORB[1551.00,1559.70] vol=4.7x ATR=3.43 |
| Stop hit — per-position SL triggered | 2025-09-10 09:40:00 | 1561.67 | 1560.34 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:40:00 | 1556.50 | 1552.65 | 0.00 | ORB-long ORB[1543.00,1554.90] vol=2.5x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-09-11 09:55:00 | 1552.05 | 1552.95 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1551.50 | 1543.15 | 0.00 | ORB-long ORB[1534.70,1545.00] vol=1.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-09-12 10:45:00 | 1547.04 | 1545.36 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 1568.00 | 1558.38 | 0.00 | ORB-long ORB[1551.10,1559.90] vol=8.4x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:15:00 | 1572.49 | 1561.08 | 0.00 | T1 1.5R @ 1572.49 |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1568.00 | 1566.84 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 1587.00 | 1579.84 | 0.00 | ORB-long ORB[1570.00,1579.90] vol=5.1x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 09:35:00 | 1592.02 | 1584.12 | 0.00 | T1 1.5R @ 1592.02 |
| Target hit | 2025-09-18 10:15:00 | 1605.10 | 1606.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 1535.70 | 1547.27 | 0.00 | ORB-short ORB[1543.70,1559.20] vol=2.1x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-09-23 09:35:00 | 1540.38 | 1545.62 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 1481.60 | 1493.74 | 0.00 | ORB-short ORB[1487.80,1502.60] vol=2.3x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 1486.64 | 1490.86 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 1497.90 | 1486.92 | 0.00 | ORB-long ORB[1473.00,1486.80] vol=1.6x ATR=4.35 |
| Stop hit — per-position SL triggered | 2025-09-29 11:05:00 | 1493.55 | 1487.70 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:10:00 | 1489.60 | 1481.51 | 0.00 | ORB-long ORB[1470.00,1485.00] vol=2.0x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1485.13 | 1481.76 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:55:00 | 1502.10 | 1493.98 | 0.00 | ORB-long ORB[1485.10,1498.80] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:05:00 | 1508.97 | 1496.93 | 0.00 | T1 1.5R @ 1508.97 |
| Target hit | 2025-10-06 11:25:00 | 1505.10 | 1506.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 1536.60 | 1529.34 | 0.00 | ORB-long ORB[1520.40,1533.20] vol=1.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:35:00 | 1542.86 | 1534.74 | 0.00 | T1 1.5R @ 1542.86 |
| Target hit | 2025-10-07 11:15:00 | 1569.20 | 1570.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:40:00 | 1532.60 | 1541.49 | 0.00 | ORB-short ORB[1537.10,1547.80] vol=1.7x ATR=5.76 |
| Stop hit — per-position SL triggered | 2025-10-09 09:55:00 | 1538.36 | 1539.94 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:55:00 | 1617.60 | 1610.95 | 0.00 | ORB-long ORB[1603.00,1615.00] vol=1.8x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:20:00 | 1624.06 | 1614.32 | 0.00 | T1 1.5R @ 1624.06 |
| Target hit | 2025-10-15 13:55:00 | 1622.00 | 1622.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 1636.70 | 1631.95 | 0.00 | ORB-long ORB[1623.90,1635.00] vol=2.8x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 1632.91 | 1632.44 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-10-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:05:00 | 1611.60 | 1617.27 | 0.00 | ORB-short ORB[1616.90,1629.70] vol=1.7x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:15:00 | 1606.62 | 1616.61 | 0.00 | T1 1.5R @ 1606.62 |
| Target hit | 2025-10-20 15:20:00 | 1601.50 | 1608.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 1613.30 | 1601.99 | 0.00 | ORB-long ORB[1589.00,1613.00] vol=2.4x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:10:00 | 1620.89 | 1609.49 | 0.00 | T1 1.5R @ 1620.89 |
| Stop hit — per-position SL triggered | 2025-10-23 13:20:00 | 1613.30 | 1614.00 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 1608.30 | 1602.94 | 0.00 | ORB-long ORB[1594.20,1607.50] vol=2.0x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:50:00 | 1615.06 | 1607.57 | 0.00 | T1 1.5R @ 1615.06 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 1608.30 | 1608.60 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1620.60 | 1611.78 | 0.00 | ORB-long ORB[1592.00,1615.00] vol=3.0x ATR=4.80 |
| Stop hit — per-position SL triggered | 2025-10-27 10:35:00 | 1615.80 | 1613.72 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 1603.30 | 1608.36 | 0.00 | ORB-short ORB[1605.50,1619.00] vol=2.4x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:10:00 | 1598.46 | 1605.81 | 0.00 | T1 1.5R @ 1598.46 |
| Target hit | 2025-10-31 11:15:00 | 1595.20 | 1591.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 1600.50 | 1590.06 | 0.00 | ORB-long ORB[1577.00,1597.70] vol=2.5x ATR=6.63 |
| Stop hit — per-position SL triggered | 2025-11-03 09:45:00 | 1593.87 | 1591.83 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 09:30:00 | 1561.20 | 1554.72 | 0.00 | ORB-long ORB[1542.30,1559.00] vol=3.5x ATR=6.47 |
| Stop hit — per-position SL triggered | 2025-12-08 09:50:00 | 1554.73 | 1556.19 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:35:00 | 1502.60 | 1511.79 | 0.00 | ORB-short ORB[1509.00,1523.10] vol=1.7x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:50:00 | 1495.55 | 1509.93 | 0.00 | T1 1.5R @ 1495.55 |
| Target hit | 2025-12-10 15:20:00 | 1477.90 | 1494.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:20:00 | 1524.60 | 1531.03 | 0.00 | ORB-short ORB[1528.00,1535.00] vol=1.6x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 10:50:00 | 1518.58 | 1529.38 | 0.00 | T1 1.5R @ 1518.58 |
| Stop hit — per-position SL triggered | 2025-12-12 11:00:00 | 1524.60 | 1528.99 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:10:00 | 1512.90 | 1509.91 | 0.00 | ORB-long ORB[1503.30,1511.50] vol=1.6x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:55:00 | 1517.04 | 1510.93 | 0.00 | T1 1.5R @ 1517.04 |
| Target hit | 2025-12-22 15:20:00 | 1516.80 | 1514.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 1455.60 | 1451.04 | 0.00 | ORB-long ORB[1440.90,1453.90] vol=2.3x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-01-02 09:35:00 | 1452.68 | 1451.23 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1460.00 | 1470.08 | 0.00 | ORB-short ORB[1466.00,1477.40] vol=2.2x ATR=3.52 |
| Stop hit — per-position SL triggered | 2026-01-06 11:35:00 | 1463.52 | 1469.35 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 1459.50 | 1469.00 | 0.00 | ORB-short ORB[1465.10,1480.30] vol=2.1x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:00:00 | 1453.65 | 1467.98 | 0.00 | T1 1.5R @ 1453.65 |
| Target hit | 2026-01-08 15:20:00 | 1437.10 | 1448.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 1418.50 | 1426.99 | 0.00 | ORB-short ORB[1425.20,1437.10] vol=1.7x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 1423.22 | 1426.08 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:10:00 | 1422.80 | 1410.41 | 0.00 | ORB-long ORB[1395.10,1414.80] vol=1.7x ATR=7.36 |
| Stop hit — per-position SL triggered | 2026-01-12 10:40:00 | 1415.44 | 1414.15 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:45:00 | 1441.00 | 1431.17 | 0.00 | ORB-long ORB[1417.00,1438.00] vol=1.8x ATR=4.59 |
| Stop hit — per-position SL triggered | 2026-01-16 10:50:00 | 1436.41 | 1431.43 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 1394.80 | 1402.61 | 0.00 | ORB-short ORB[1395.50,1415.20] vol=2.8x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:40:00 | 1388.31 | 1398.25 | 0.00 | T1 1.5R @ 1388.31 |
| Target hit | 2026-01-20 10:40:00 | 1382.80 | 1381.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — SELL (started 2026-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:35:00 | 1338.90 | 1349.89 | 0.00 | ORB-short ORB[1346.70,1359.60] vol=1.6x ATR=4.26 |
| Stop hit — per-position SL triggered | 2026-01-29 09:40:00 | 1343.16 | 1349.00 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 1242.60 | 1251.76 | 0.00 | ORB-short ORB[1246.10,1261.70] vol=2.5x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:45:00 | 1236.29 | 1247.16 | 0.00 | T1 1.5R @ 1236.29 |
| Target hit | 2026-03-11 15:20:00 | 1218.50 | 1237.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 1197.60 | 1203.94 | 0.00 | ORB-short ORB[1198.20,1213.40] vol=2.0x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 1202.16 | 1202.59 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 1233.00 | 1221.22 | 0.00 | ORB-long ORB[1205.30,1223.70] vol=1.9x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 1229.22 | 1221.61 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-03-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:50:00 | 1147.00 | 1160.55 | 0.00 | ORB-short ORB[1161.10,1176.90] vol=1.7x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:00:00 | 1140.24 | 1159.23 | 0.00 | T1 1.5R @ 1140.24 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1147.00 | 1158.83 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 11:10:00 | 1284.50 | 1264.32 | 0.00 | ORB-long ORB[1249.60,1268.30] vol=2.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-04-08 12:30:00 | 1279.49 | 1269.69 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1335.10 | 1326.10 | 0.00 | ORB-long ORB[1318.10,1331.00] vol=3.1x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:30:00 | 1344.26 | 1332.17 | 0.00 | T1 1.5R @ 1344.26 |
| Target hit | 2026-04-15 15:20:00 | 1339.20 | 1336.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1357.20 | 1369.64 | 0.00 | ORB-short ORB[1365.10,1373.50] vol=2.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 1360.46 | 1369.01 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 1347.60 | 1358.68 | 0.00 | ORB-short ORB[1354.10,1367.20] vol=1.6x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 1341.99 | 1354.43 | 0.00 | T1 1.5R @ 1341.99 |
| Target hit | 2026-04-22 15:20:00 | 1321.40 | 1331.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1319.60 | 1328.92 | 0.00 | ORB-short ORB[1323.60,1335.90] vol=2.2x ATR=3.81 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 1323.41 | 1327.93 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 1329.80 | 1322.38 | 0.00 | ORB-long ORB[1314.50,1325.70] vol=2.3x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 1325.92 | 1322.98 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 1319.10 | 1323.35 | 0.00 | ORB-short ORB[1320.20,1332.60] vol=2.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 1322.51 | 1322.88 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 1343.50 | 1333.56 | 0.00 | ORB-long ORB[1326.50,1338.60] vol=4.9x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 1340.26 | 1334.47 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 1279.90 | 1290.23 | 0.00 | ORB-short ORB[1282.00,1298.00] vol=1.7x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:30:00 | 1272.88 | 1285.67 | 0.00 | T1 1.5R @ 1272.88 |
| Stop hit — per-position SL triggered | 2026-04-30 13:50:00 | 1279.90 | 1280.51 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1243.00 | 1230.23 | 0.00 | ORB-long ORB[1222.10,1232.80] vol=3.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:25:00 | 1249.25 | 1232.70 | 0.00 | T1 1.5R @ 1249.25 |
| Target hit | 2026-05-05 15:20:00 | 1254.50 | 1243.11 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 11:15:00 | 1309.00 | 2025-05-13 11:25:00 | 1315.72 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-05-13 11:15:00 | 1309.00 | 2025-05-13 13:00:00 | 1309.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-16 09:35:00 | 1357.90 | 2025-05-16 09:55:00 | 1354.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-27 09:30:00 | 1454.00 | 2025-05-27 09:35:00 | 1458.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-02 09:40:00 | 1553.70 | 2025-06-02 09:50:00 | 1562.02 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-02 09:40:00 | 1553.70 | 2025-06-02 15:20:00 | 1682.60 | TARGET_HIT | 0.50 | 8.30% |
| SELL | retest1 | 2025-06-05 09:30:00 | 1706.70 | 2025-06-05 09:40:00 | 1713.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1802.40 | 2025-06-09 09:40:00 | 1815.17 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1802.40 | 2025-06-09 09:45:00 | 1802.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 09:40:00 | 1823.90 | 2025-06-10 09:50:00 | 1817.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-17 09:35:00 | 1706.40 | 2025-06-17 09:40:00 | 1716.89 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-17 09:35:00 | 1706.40 | 2025-06-17 10:15:00 | 1706.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 09:30:00 | 1691.00 | 2025-06-18 09:55:00 | 1701.39 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-18 09:30:00 | 1691.00 | 2025-06-18 10:15:00 | 1691.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 09:50:00 | 1674.90 | 2025-06-19 10:30:00 | 1667.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-19 09:50:00 | 1674.90 | 2025-06-19 15:20:00 | 1635.50 | TARGET_HIT | 0.50 | 2.35% |
| SELL | retest1 | 2025-06-26 11:15:00 | 1728.20 | 2025-06-26 11:30:00 | 1734.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-30 09:30:00 | 1776.20 | 2025-06-30 09:40:00 | 1783.71 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-30 09:30:00 | 1776.20 | 2025-06-30 09:50:00 | 1776.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 10:55:00 | 1807.00 | 2025-07-03 11:25:00 | 1801.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-08 09:30:00 | 1775.30 | 2025-07-08 09:35:00 | 1767.66 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-08 09:30:00 | 1775.30 | 2025-07-08 13:10:00 | 1753.60 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-07-09 09:30:00 | 1768.30 | 2025-07-09 09:35:00 | 1762.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-11 09:55:00 | 1724.00 | 2025-07-11 10:00:00 | 1716.47 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-11 09:55:00 | 1724.00 | 2025-07-11 15:20:00 | 1687.10 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2025-07-16 10:35:00 | 1720.30 | 2025-07-16 10:40:00 | 1715.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-17 09:30:00 | 1718.30 | 2025-07-17 09:35:00 | 1722.24 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-18 10:00:00 | 1702.40 | 2025-07-18 10:10:00 | 1695.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-18 10:00:00 | 1702.40 | 2025-07-18 11:15:00 | 1702.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:35:00 | 1714.50 | 2025-07-21 09:40:00 | 1709.32 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-22 09:35:00 | 1740.70 | 2025-07-22 09:40:00 | 1733.33 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-22 09:35:00 | 1740.70 | 2025-07-22 09:45:00 | 1740.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:35:00 | 1683.80 | 2025-07-23 09:45:00 | 1689.64 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-25 10:05:00 | 1641.70 | 2025-07-25 10:35:00 | 1632.29 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-07-25 10:05:00 | 1641.70 | 2025-07-25 11:05:00 | 1641.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-01 09:45:00 | 1471.90 | 2025-08-01 09:55:00 | 1463.29 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-08-01 09:45:00 | 1471.90 | 2025-08-01 10:00:00 | 1471.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-05 10:55:00 | 1551.60 | 2025-08-05 11:15:00 | 1557.92 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-06 09:35:00 | 1544.00 | 2025-08-06 09:40:00 | 1549.17 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-08 11:15:00 | 1575.10 | 2025-08-08 11:45:00 | 1585.74 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-08-08 11:15:00 | 1575.10 | 2025-08-08 14:25:00 | 1578.20 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-08-13 10:05:00 | 1566.30 | 2025-08-13 10:35:00 | 1561.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-18 09:30:00 | 1569.90 | 2025-08-18 09:35:00 | 1563.30 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-18 09:30:00 | 1569.90 | 2025-08-18 09:40:00 | 1569.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 11:15:00 | 1598.80 | 2025-08-21 11:30:00 | 1603.54 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-08-21 11:15:00 | 1598.80 | 2025-08-21 11:50:00 | 1598.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 09:35:00 | 1580.50 | 2025-08-22 09:50:00 | 1587.30 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-22 09:35:00 | 1580.50 | 2025-08-22 10:35:00 | 1580.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-25 10:40:00 | 1559.80 | 2025-08-25 13:05:00 | 1553.68 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-25 10:40:00 | 1559.80 | 2025-08-25 15:20:00 | 1541.60 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2025-08-26 09:30:00 | 1521.50 | 2025-08-26 09:35:00 | 1515.22 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-26 09:30:00 | 1521.50 | 2025-08-26 10:00:00 | 1521.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 09:50:00 | 1469.60 | 2025-09-01 10:20:00 | 1464.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-02 10:00:00 | 1492.40 | 2025-09-02 10:10:00 | 1498.78 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-02 10:00:00 | 1492.40 | 2025-09-02 13:15:00 | 1511.50 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-09-03 09:30:00 | 1520.30 | 2025-09-03 09:40:00 | 1515.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-08 09:30:00 | 1538.00 | 2025-09-08 09:55:00 | 1534.11 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-09 09:30:00 | 1542.80 | 2025-09-09 09:35:00 | 1546.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-10 09:30:00 | 1565.10 | 2025-09-10 09:40:00 | 1561.67 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-11 09:40:00 | 1556.50 | 2025-09-11 09:55:00 | 1552.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-12 10:00:00 | 1551.50 | 2025-09-12 10:45:00 | 1547.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-17 11:10:00 | 1568.00 | 2025-09-17 11:15:00 | 1572.49 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-17 11:10:00 | 1568.00 | 2025-09-17 12:15:00 | 1568.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 09:30:00 | 1587.00 | 2025-09-18 09:35:00 | 1592.02 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-18 09:30:00 | 1587.00 | 2025-09-18 10:15:00 | 1605.10 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-09-23 09:30:00 | 1535.70 | 2025-09-23 09:35:00 | 1540.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-26 09:30:00 | 1481.60 | 2025-09-26 09:40:00 | 1486.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1497.90 | 2025-09-29 11:05:00 | 1493.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-03 10:10:00 | 1489.60 | 2025-10-03 10:15:00 | 1485.13 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-06 09:55:00 | 1502.10 | 2025-10-06 10:05:00 | 1508.97 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-06 09:55:00 | 1502.10 | 2025-10-06 11:25:00 | 1505.10 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-10-07 09:30:00 | 1536.60 | 2025-10-07 09:35:00 | 1542.86 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-10-07 09:30:00 | 1536.60 | 2025-10-07 11:15:00 | 1569.20 | TARGET_HIT | 0.50 | 2.12% |
| SELL | retest1 | 2025-10-09 09:40:00 | 1532.60 | 2025-10-09 09:55:00 | 1538.36 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-15 09:55:00 | 1617.60 | 2025-10-15 10:20:00 | 1624.06 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-15 09:55:00 | 1617.60 | 2025-10-15 13:55:00 | 1622.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-10-17 09:30:00 | 1636.70 | 2025-10-17 09:40:00 | 1632.91 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-20 11:05:00 | 1611.60 | 2025-10-20 11:15:00 | 1606.62 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-20 11:05:00 | 1611.60 | 2025-10-20 15:20:00 | 1601.50 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-10-23 09:30:00 | 1613.30 | 2025-10-23 10:10:00 | 1620.89 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-23 09:30:00 | 1613.30 | 2025-10-23 13:20:00 | 1613.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:30:00 | 1608.30 | 2025-10-24 09:50:00 | 1615.06 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-24 09:30:00 | 1608.30 | 2025-10-24 10:15:00 | 1608.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 10:15:00 | 1620.60 | 2025-10-27 10:35:00 | 1615.80 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-31 09:50:00 | 1603.30 | 2025-10-31 10:10:00 | 1598.46 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-31 09:50:00 | 1603.30 | 2025-10-31 11:15:00 | 1595.20 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-11-03 09:30:00 | 1600.50 | 2025-11-03 09:45:00 | 1593.87 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-12-08 09:30:00 | 1561.20 | 2025-12-08 09:50:00 | 1554.73 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-10 10:35:00 | 1502.60 | 2025-12-10 10:50:00 | 1495.55 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-10 10:35:00 | 1502.60 | 2025-12-10 15:20:00 | 1477.90 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2025-12-12 10:20:00 | 1524.60 | 2025-12-12 10:50:00 | 1518.58 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-12 10:20:00 | 1524.60 | 2025-12-12 11:00:00 | 1524.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 11:10:00 | 1512.90 | 2025-12-22 11:55:00 | 1517.04 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-12-22 11:10:00 | 1512.90 | 2025-12-22 15:20:00 | 1516.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-01-02 09:30:00 | 1455.60 | 2026-01-02 09:35:00 | 1452.68 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-06 11:15:00 | 1460.00 | 2026-01-06 11:35:00 | 1463.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-08 10:50:00 | 1459.50 | 2026-01-08 11:00:00 | 1453.65 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-08 10:50:00 | 1459.50 | 2026-01-08 15:20:00 | 1437.10 | TARGET_HIT | 0.50 | 1.53% |
| SELL | retest1 | 2026-01-09 09:35:00 | 1418.50 | 2026-01-09 09:45:00 | 1423.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-12 10:10:00 | 1422.80 | 2026-01-12 10:40:00 | 1415.44 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-01-16 10:45:00 | 1441.00 | 2026-01-16 10:50:00 | 1436.41 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-20 09:30:00 | 1394.80 | 2026-01-20 09:40:00 | 1388.31 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-01-20 09:30:00 | 1394.80 | 2026-01-20 10:40:00 | 1382.80 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2026-01-29 09:35:00 | 1338.90 | 2026-01-29 09:40:00 | 1343.16 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-11 10:10:00 | 1242.60 | 2026-03-11 10:45:00 | 1236.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-11 10:10:00 | 1242.60 | 2026-03-11 15:20:00 | 1218.50 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2026-03-12 10:10:00 | 1197.60 | 2026-03-12 10:35:00 | 1202.16 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 11:05:00 | 1233.00 | 2026-03-18 11:10:00 | 1229.22 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1147.00 | 2026-03-23 11:00:00 | 1140.24 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1147.00 | 2026-03-23 11:05:00 | 1147.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 11:10:00 | 1284.50 | 2026-04-08 12:30:00 | 1279.49 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1335.10 | 2026-04-15 11:30:00 | 1344.26 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1335.10 | 2026-04-15 15:20:00 | 1339.20 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1357.20 | 2026-04-21 11:00:00 | 1360.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-22 09:55:00 | 1347.60 | 2026-04-22 10:15:00 | 1341.99 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-22 09:55:00 | 1347.60 | 2026-04-22 15:20:00 | 1321.40 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1319.60 | 2026-04-24 09:35:00 | 1323.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 09:55:00 | 1329.80 | 2026-04-27 10:05:00 | 1325.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-28 09:35:00 | 1319.10 | 2026-04-28 09:50:00 | 1322.51 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1343.50 | 2026-04-29 11:10:00 | 1340.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1279.90 | 2026-04-30 10:30:00 | 1272.88 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1279.90 | 2026-04-30 13:50:00 | 1279.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 11:15:00 | 1243.00 | 2026-05-05 11:25:00 | 1249.25 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-05 11:15:00 | 1243.00 | 2026-05-05 15:20:00 | 1254.50 | TARGET_HIT | 0.50 | 0.93% |
