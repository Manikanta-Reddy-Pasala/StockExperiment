# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1611.50
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 16 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 58
- **Target hits / Stop hits / Partials:** 16 / 58 / 31
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 21.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 20 | 40.0% | 6 | 30 | 14 | 0.16% | 8.0% |
| BUY @ 2nd Alert (retest1) | 50 | 20 | 40.0% | 6 | 30 | 14 | 0.16% | 8.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 27 | 49.1% | 10 | 28 | 17 | 0.24% | 13.1% |
| SELL @ 2nd Alert (retest1) | 55 | 27 | 49.1% | 10 | 28 | 17 | 0.24% | 13.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 47 | 44.8% | 16 | 58 | 31 | 0.20% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 11:15:00 | 1333.80 | 1348.70 | 0.00 | ORB-short ORB[1348.10,1365.00] vol=2.0x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 12:40:00 | 1326.78 | 1345.12 | 0.00 | T1 1.5R @ 1326.78 |
| Stop hit — per-position SL triggered | 2025-05-26 15:00:00 | 1333.80 | 1339.76 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:55:00 | 1321.50 | 1331.38 | 0.00 | ORB-short ORB[1328.50,1343.90] vol=3.7x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-05-27 10:00:00 | 1326.38 | 1330.89 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 1328.00 | 1339.72 | 0.00 | ORB-short ORB[1336.00,1354.00] vol=2.4x ATR=5.68 |
| Stop hit — per-position SL triggered | 2025-05-29 11:50:00 | 1333.68 | 1336.59 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 11:00:00 | 1485.40 | 1465.66 | 0.00 | ORB-long ORB[1457.10,1475.00] vol=2.5x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 1478.29 | 1470.28 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:40:00 | 1534.10 | 1522.49 | 0.00 | ORB-long ORB[1509.00,1523.90] vol=5.0x ATR=6.15 |
| Stop hit — per-position SL triggered | 2025-06-10 09:45:00 | 1527.95 | 1522.83 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 09:55:00 | 1549.20 | 1560.14 | 0.00 | ORB-short ORB[1551.20,1571.80] vol=1.5x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-06-11 10:00:00 | 1555.84 | 1559.79 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 1663.50 | 1647.96 | 0.00 | ORB-long ORB[1641.20,1658.00] vol=2.1x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 11:25:00 | 1674.87 | 1655.69 | 0.00 | T1 1.5R @ 1674.87 |
| Target hit | 2025-06-30 15:20:00 | 1709.00 | 1700.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:45:00 | 1662.90 | 1674.92 | 0.00 | ORB-short ORB[1671.20,1691.00] vol=1.7x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:10:00 | 1652.52 | 1668.74 | 0.00 | T1 1.5R @ 1652.52 |
| Stop hit — per-position SL triggered | 2025-07-02 10:25:00 | 1662.90 | 1667.78 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:10:00 | 1705.00 | 1692.86 | 0.00 | ORB-long ORB[1680.50,1698.50] vol=1.7x ATR=9.06 |
| Stop hit — per-position SL triggered | 2025-07-03 10:25:00 | 1695.94 | 1694.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:00:00 | 1686.80 | 1708.25 | 0.00 | ORB-short ORB[1703.10,1728.20] vol=2.6x ATR=9.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:00:00 | 1672.68 | 1701.94 | 0.00 | T1 1.5R @ 1672.68 |
| Target hit | 2025-07-08 15:20:00 | 1654.00 | 1681.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1817.20 | 1829.27 | 0.00 | ORB-short ORB[1817.50,1841.90] vol=1.5x ATR=9.35 |
| Stop hit — per-position SL triggered | 2025-07-15 09:50:00 | 1826.55 | 1827.88 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1854.10 | 1834.98 | 0.00 | ORB-long ORB[1815.80,1838.80] vol=5.8x ATR=10.00 |
| Stop hit — per-position SL triggered | 2025-07-17 09:50:00 | 1844.10 | 1836.26 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 1784.50 | 1809.49 | 0.00 | ORB-short ORB[1812.00,1826.50] vol=3.0x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1774.65 | 1803.29 | 0.00 | T1 1.5R @ 1774.65 |
| Stop hit — per-position SL triggered | 2025-07-18 10:20:00 | 1784.50 | 1802.36 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1694.70 | 1704.56 | 0.00 | ORB-short ORB[1698.50,1719.60] vol=2.0x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:35:00 | 1685.20 | 1701.39 | 0.00 | T1 1.5R @ 1685.20 |
| Stop hit — per-position SL triggered | 2025-07-23 09:40:00 | 1694.70 | 1700.38 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:55:00 | 1732.80 | 1718.99 | 0.00 | ORB-long ORB[1706.00,1728.00] vol=2.3x ATR=6.51 |
| Stop hit — per-position SL triggered | 2025-07-24 11:00:00 | 1726.29 | 1719.28 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:15:00 | 1669.10 | 1686.61 | 0.00 | ORB-short ORB[1682.30,1703.80] vol=1.6x ATR=6.54 |
| Stop hit — per-position SL triggered | 2025-07-25 11:00:00 | 1675.64 | 1681.82 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:15:00 | 1694.00 | 1696.88 | 0.00 | ORB-short ORB[1696.50,1717.00] vol=1.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2025-07-30 11:30:00 | 1698.09 | 1697.64 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:30:00 | 1656.00 | 1661.98 | 0.00 | ORB-short ORB[1657.10,1676.70] vol=3.1x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 09:40:00 | 1646.37 | 1657.93 | 0.00 | T1 1.5R @ 1646.37 |
| Target hit | 2025-08-01 11:00:00 | 1646.00 | 1645.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 1574.40 | 1583.20 | 0.00 | ORB-short ORB[1575.10,1597.20] vol=1.6x ATR=7.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 09:35:00 | 1562.80 | 1579.31 | 0.00 | T1 1.5R @ 1562.80 |
| Target hit | 2025-08-05 15:20:00 | 1547.70 | 1560.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:50:00 | 1478.50 | 1469.36 | 0.00 | ORB-long ORB[1455.70,1477.00] vol=1.6x ATR=7.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:00:00 | 1489.03 | 1474.25 | 0.00 | T1 1.5R @ 1489.03 |
| Target hit | 2025-08-12 11:00:00 | 1481.00 | 1483.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2025-08-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:10:00 | 1518.80 | 1508.01 | 0.00 | ORB-long ORB[1496.10,1514.40] vol=2.5x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:15:00 | 1527.50 | 1512.11 | 0.00 | T1 1.5R @ 1527.50 |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 1518.80 | 1519.45 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1499.40 | 1504.52 | 0.00 | ORB-short ORB[1501.10,1512.80] vol=2.0x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:40:00 | 1494.35 | 1502.73 | 0.00 | T1 1.5R @ 1494.35 |
| Target hit | 2025-08-19 15:20:00 | 1493.30 | 1497.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 1479.70 | 1486.96 | 0.00 | ORB-short ORB[1480.10,1498.90] vol=1.7x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-08-20 09:40:00 | 1484.37 | 1484.69 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 1496.60 | 1492.15 | 0.00 | ORB-long ORB[1483.90,1494.90] vol=1.7x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:50:00 | 1504.25 | 1494.90 | 0.00 | T1 1.5R @ 1504.25 |
| Stop hit — per-position SL triggered | 2025-08-21 10:10:00 | 1496.60 | 1495.69 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:20:00 | 1421.10 | 1428.67 | 0.00 | ORB-short ORB[1427.10,1436.30] vol=1.6x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-09-01 10:25:00 | 1424.64 | 1428.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1442.50 | 1430.18 | 0.00 | ORB-long ORB[1417.10,1431.20] vol=4.4x ATR=5.93 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 1436.57 | 1432.27 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:40:00 | 1492.20 | 1479.03 | 0.00 | ORB-long ORB[1460.60,1482.00] vol=1.8x ATR=6.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:50:00 | 1502.37 | 1491.51 | 0.00 | T1 1.5R @ 1502.37 |
| Target hit | 2025-09-09 10:55:00 | 1504.30 | 1506.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:45:00 | 1531.60 | 1526.30 | 0.00 | ORB-long ORB[1519.20,1530.90] vol=2.0x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-09-19 09:55:00 | 1527.11 | 1526.21 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:15:00 | 1499.90 | 1492.27 | 0.00 | ORB-long ORB[1481.00,1496.80] vol=2.6x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-09-25 10:20:00 | 1494.03 | 1492.52 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 1401.50 | 1413.39 | 0.00 | ORB-short ORB[1418.80,1428.80] vol=2.3x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 1405.05 | 1412.45 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:20:00 | 1425.90 | 1413.79 | 0.00 | ORB-long ORB[1398.20,1417.70] vol=3.7x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-10-09 10:25:00 | 1419.03 | 1414.50 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1378.60 | 1392.04 | 0.00 | ORB-short ORB[1389.30,1399.50] vol=2.0x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:25:00 | 1373.52 | 1390.88 | 0.00 | T1 1.5R @ 1373.52 |
| Target hit | 2025-10-14 15:20:00 | 1367.60 | 1379.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1380.50 | 1374.84 | 0.00 | ORB-long ORB[1367.20,1377.50] vol=2.8x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-10-15 10:25:00 | 1375.95 | 1374.96 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:50:00 | 1337.90 | 1345.45 | 0.00 | ORB-short ORB[1345.90,1358.00] vol=2.0x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-10-17 09:55:00 | 1342.43 | 1345.08 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 1367.00 | 1360.67 | 0.00 | ORB-long ORB[1349.80,1364.20] vol=1.8x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:35:00 | 1374.44 | 1364.82 | 0.00 | T1 1.5R @ 1374.44 |
| Stop hit — per-position SL triggered | 2025-10-20 09:40:00 | 1367.00 | 1363.31 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:45:00 | 1340.80 | 1348.47 | 0.00 | ORB-short ORB[1346.00,1360.10] vol=2.7x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:30:00 | 1335.90 | 1346.68 | 0.00 | T1 1.5R @ 1335.90 |
| Target hit | 2025-10-24 15:20:00 | 1333.00 | 1336.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1318.30 | 1326.66 | 0.00 | ORB-short ORB[1321.20,1338.90] vol=2.1x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-10-27 09:35:00 | 1323.09 | 1325.75 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:50:00 | 1310.60 | 1299.54 | 0.00 | ORB-long ORB[1289.90,1304.00] vol=2.1x ATR=4.41 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 1306.19 | 1301.15 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:40:00 | 1348.70 | 1357.03 | 0.00 | ORB-short ORB[1354.00,1369.90] vol=1.5x ATR=5.10 |
| Stop hit — per-position SL triggered | 2025-11-06 11:30:00 | 1353.80 | 1355.25 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:10:00 | 1335.20 | 1347.77 | 0.00 | ORB-short ORB[1347.20,1361.30] vol=1.7x ATR=5.88 |
| Stop hit — per-position SL triggered | 2025-11-11 10:55:00 | 1341.08 | 1343.86 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:00:00 | 1343.00 | 1354.59 | 0.00 | ORB-short ORB[1357.00,1367.80] vol=1.8x ATR=4.44 |
| Stop hit — per-position SL triggered | 2025-11-18 10:20:00 | 1347.44 | 1353.36 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:30:00 | 1289.70 | 1296.03 | 0.00 | ORB-short ORB[1295.30,1313.20] vol=1.7x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-11-24 10:50:00 | 1293.24 | 1295.58 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 10:55:00 | 1261.80 | 1271.40 | 0.00 | ORB-short ORB[1266.10,1284.10] vol=2.2x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-11-26 11:20:00 | 1265.56 | 1269.97 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 1261.40 | 1268.69 | 0.00 | ORB-short ORB[1268.00,1281.50] vol=3.1x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-11-28 10:05:00 | 1266.07 | 1266.56 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:55:00 | 1362.20 | 1336.82 | 0.00 | ORB-long ORB[1312.20,1332.20] vol=5.2x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:15:00 | 1375.16 | 1353.34 | 0.00 | T1 1.5R @ 1375.16 |
| Stop hit — per-position SL triggered | 2025-12-10 10:30:00 | 1362.20 | 1354.99 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:55:00 | 1342.60 | 1338.07 | 0.00 | ORB-long ORB[1329.30,1341.00] vol=1.6x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:40:00 | 1347.89 | 1340.29 | 0.00 | T1 1.5R @ 1347.89 |
| Stop hit — per-position SL triggered | 2025-12-16 13:05:00 | 1342.60 | 1341.15 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 1369.80 | 1363.58 | 0.00 | ORB-long ORB[1348.30,1368.60] vol=1.8x ATR=5.26 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 1364.54 | 1363.86 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:40:00 | 1370.60 | 1348.67 | 0.00 | ORB-long ORB[1340.40,1358.90] vol=1.9x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 13:45:00 | 1379.25 | 1361.67 | 0.00 | T1 1.5R @ 1379.25 |
| Target hit | 2025-12-18 15:20:00 | 1379.30 | 1365.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2025-12-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:10:00 | 1432.00 | 1421.95 | 0.00 | ORB-long ORB[1412.00,1428.00] vol=1.8x ATR=6.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:20:00 | 1441.32 | 1424.87 | 0.00 | T1 1.5R @ 1441.32 |
| Stop hit — per-position SL triggered | 2025-12-22 11:20:00 | 1432.00 | 1431.02 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:25:00 | 1421.90 | 1411.66 | 0.00 | ORB-long ORB[1402.40,1418.10] vol=1.5x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:35:00 | 1429.31 | 1418.62 | 0.00 | T1 1.5R @ 1429.31 |
| Target hit | 2026-01-05 11:50:00 | 1458.60 | 1462.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — BUY (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 1462.90 | 1458.55 | 0.00 | ORB-long ORB[1443.00,1457.50] vol=2.4x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1457.61 | 1458.54 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 1484.70 | 1498.67 | 0.00 | ORB-short ORB[1490.00,1511.90] vol=1.7x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 1475.53 | 1497.59 | 0.00 | T1 1.5R @ 1475.53 |
| Target hit | 2026-01-08 15:20:00 | 1456.00 | 1482.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 1386.60 | 1395.83 | 0.00 | ORB-short ORB[1391.80,1408.00] vol=2.0x ATR=6.23 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 1392.83 | 1394.11 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:55:00 | 1414.50 | 1403.68 | 0.00 | ORB-long ORB[1393.90,1411.90] vol=1.7x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:05:00 | 1422.99 | 1408.38 | 0.00 | T1 1.5R @ 1422.99 |
| Target hit | 2026-01-16 15:00:00 | 1432.20 | 1433.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — SELL (started 2026-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:40:00 | 1397.20 | 1404.33 | 0.00 | ORB-short ORB[1401.00,1419.20] vol=1.9x ATR=5.89 |
| Stop hit — per-position SL triggered | 2026-01-19 09:50:00 | 1403.09 | 1403.80 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:15:00 | 1394.50 | 1407.16 | 0.00 | ORB-short ORB[1404.00,1417.70] vol=3.7x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-01-20 11:25:00 | 1399.47 | 1406.53 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:10:00 | 1346.20 | 1355.12 | 0.00 | ORB-short ORB[1347.00,1367.00] vol=1.9x ATR=3.95 |
| Stop hit — per-position SL triggered | 2026-01-23 12:00:00 | 1350.15 | 1353.03 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:15:00 | 1349.00 | 1352.72 | 0.00 | ORB-short ORB[1351.00,1364.00] vol=1.9x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:50:00 | 1344.18 | 1351.99 | 0.00 | T1 1.5R @ 1344.18 |
| Stop hit — per-position SL triggered | 2026-01-28 12:00:00 | 1349.00 | 1351.78 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:50:00 | 1386.00 | 1369.57 | 0.00 | ORB-long ORB[1350.00,1365.00] vol=1.7x ATR=6.49 |
| Stop hit — per-position SL triggered | 2026-01-30 10:10:00 | 1379.51 | 1371.92 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:10:00 | 1393.00 | 1374.90 | 0.00 | ORB-long ORB[1363.00,1372.90] vol=2.1x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 1388.32 | 1375.25 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 1402.00 | 1396.72 | 0.00 | ORB-long ORB[1381.00,1400.00] vol=1.9x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1398.45 | 1397.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1415.60 | 1426.33 | 0.00 | ORB-short ORB[1422.10,1436.90] vol=2.1x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 1408.54 | 1424.83 | 0.00 | T1 1.5R @ 1408.54 |
| Target hit | 2026-02-19 15:20:00 | 1389.10 | 1410.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 1371.50 | 1378.80 | 0.00 | ORB-short ORB[1374.00,1390.50] vol=2.2x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:45:00 | 1366.12 | 1376.74 | 0.00 | T1 1.5R @ 1366.12 |
| Target hit | 2026-02-23 15:20:00 | 1365.10 | 1366.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 1371.00 | 1365.82 | 0.00 | ORB-long ORB[1351.20,1370.30] vol=2.5x ATR=3.93 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 1367.07 | 1365.85 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 1401.60 | 1389.30 | 0.00 | ORB-long ORB[1369.50,1389.80] vol=2.8x ATR=5.90 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 1395.70 | 1390.07 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 1313.00 | 1301.19 | 0.00 | ORB-long ORB[1288.90,1306.60] vol=2.3x ATR=6.95 |
| Stop hit — per-position SL triggered | 2026-03-05 10:20:00 | 1306.05 | 1303.36 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 1215.30 | 1205.29 | 0.00 | ORB-long ORB[1192.00,1210.00] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2026-03-18 09:40:00 | 1210.12 | 1207.15 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1214.10 | 1205.90 | 0.00 | ORB-long ORB[1193.20,1211.00] vol=2.1x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 1209.09 | 1206.70 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 1365.00 | 1372.27 | 0.00 | ORB-short ORB[1366.10,1382.40] vol=3.2x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 13:15:00 | 1357.40 | 1368.70 | 0.00 | T1 1.5R @ 1357.40 |
| Target hit | 2026-04-10 15:20:00 | 1344.90 | 1354.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-04-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:10:00 | 1330.60 | 1320.53 | 0.00 | ORB-long ORB[1307.50,1326.20] vol=1.8x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-04-13 13:00:00 | 1326.10 | 1322.58 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1381.00 | 1370.97 | 0.00 | ORB-long ORB[1360.00,1372.00] vol=2.6x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1387.97 | 1373.63 | 0.00 | T1 1.5R @ 1387.97 |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1381.00 | 1374.19 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 1428.60 | 1418.26 | 0.00 | ORB-long ORB[1405.00,1424.00] vol=1.7x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:00:00 | 1438.40 | 1420.34 | 0.00 | T1 1.5R @ 1438.40 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 1428.60 | 1424.36 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 1419.80 | 1425.36 | 0.00 | ORB-short ORB[1423.80,1437.40] vol=1.6x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:10:00 | 1413.13 | 1420.60 | 0.00 | T1 1.5R @ 1413.13 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 1419.80 | 1420.32 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 1411.00 | 1420.29 | 0.00 | ORB-short ORB[1417.90,1437.00] vol=1.9x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 1405.08 | 1418.55 | 0.00 | T1 1.5R @ 1405.08 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1411.00 | 1418.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-26 11:15:00 | 1333.80 | 2025-05-26 12:40:00 | 1326.78 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-05-26 11:15:00 | 1333.80 | 2025-05-26 15:00:00 | 1333.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 09:55:00 | 1321.50 | 2025-05-27 10:00:00 | 1326.38 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-05-29 11:00:00 | 1328.00 | 2025-05-29 11:50:00 | 1333.68 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-03 11:00:00 | 1485.40 | 2025-06-03 11:15:00 | 1478.29 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-06-10 09:40:00 | 1534.10 | 2025-06-10 09:45:00 | 1527.95 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-11 09:55:00 | 1549.20 | 2025-06-11 10:00:00 | 1555.84 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-30 11:15:00 | 1663.50 | 2025-06-30 11:25:00 | 1674.87 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-06-30 11:15:00 | 1663.50 | 2025-06-30 15:20:00 | 1709.00 | TARGET_HIT | 0.50 | 2.74% |
| SELL | retest1 | 2025-07-02 09:45:00 | 1662.90 | 2025-07-02 10:10:00 | 1652.52 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-07-02 09:45:00 | 1662.90 | 2025-07-02 10:25:00 | 1662.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 10:10:00 | 1705.00 | 2025-07-03 10:25:00 | 1695.94 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-07-08 10:00:00 | 1686.80 | 2025-07-08 11:00:00 | 1672.68 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2025-07-08 10:00:00 | 1686.80 | 2025-07-08 15:20:00 | 1654.00 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2025-07-15 09:30:00 | 1817.20 | 2025-07-15 09:50:00 | 1826.55 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-07-17 09:45:00 | 1854.10 | 2025-07-17 09:50:00 | 1844.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-07-18 10:05:00 | 1784.50 | 2025-07-18 10:15:00 | 1774.65 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-07-18 10:05:00 | 1784.50 | 2025-07-18 10:20:00 | 1784.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1694.70 | 2025-07-23 09:35:00 | 1685.20 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1694.70 | 2025-07-23 09:40:00 | 1694.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 10:55:00 | 1732.80 | 2025-07-24 11:00:00 | 1726.29 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-25 10:15:00 | 1669.10 | 2025-07-25 11:00:00 | 1675.64 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-07-30 11:15:00 | 1694.00 | 2025-07-30 11:30:00 | 1698.09 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-01 09:30:00 | 1656.00 | 2025-08-01 09:40:00 | 1646.37 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-08-01 09:30:00 | 1656.00 | 2025-08-01 11:00:00 | 1646.00 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-08-05 09:30:00 | 1574.40 | 2025-08-05 09:35:00 | 1562.80 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-08-05 09:30:00 | 1574.40 | 2025-08-05 15:20:00 | 1547.70 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-08-12 09:50:00 | 1478.50 | 2025-08-12 10:00:00 | 1489.03 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-08-12 09:50:00 | 1478.50 | 2025-08-12 11:00:00 | 1481.00 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-08-14 10:10:00 | 1518.80 | 2025-08-14 10:15:00 | 1527.50 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-08-14 10:10:00 | 1518.80 | 2025-08-14 11:15:00 | 1518.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-19 11:15:00 | 1499.40 | 2025-08-19 11:40:00 | 1494.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-19 11:15:00 | 1499.40 | 2025-08-19 15:20:00 | 1493.30 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-20 09:30:00 | 1479.70 | 2025-08-20 09:40:00 | 1484.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-21 09:40:00 | 1496.60 | 2025-08-21 09:50:00 | 1504.25 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-21 09:40:00 | 1496.60 | 2025-08-21 10:10:00 | 1496.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-01 10:20:00 | 1421.10 | 2025-09-01 10:25:00 | 1424.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-03 09:45:00 | 1442.50 | 2025-09-03 09:55:00 | 1436.57 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-09-09 09:40:00 | 1492.20 | 2025-09-09 09:50:00 | 1502.37 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-09-09 09:40:00 | 1492.20 | 2025-09-09 10:55:00 | 1504.30 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-09-19 09:45:00 | 1531.60 | 2025-09-19 09:55:00 | 1527.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-25 10:15:00 | 1499.90 | 2025-09-25 10:20:00 | 1494.03 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-10-08 11:15:00 | 1401.50 | 2025-10-08 11:25:00 | 1405.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-09 10:20:00 | 1425.90 | 2025-10-09 10:25:00 | 1419.03 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1378.60 | 2025-10-14 11:25:00 | 1373.52 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1378.60 | 2025-10-14 15:20:00 | 1367.60 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2025-10-15 10:15:00 | 1380.50 | 2025-10-15 10:25:00 | 1375.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-17 09:50:00 | 1337.90 | 2025-10-17 09:55:00 | 1342.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-20 09:30:00 | 1367.00 | 2025-10-20 09:35:00 | 1374.44 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-20 09:30:00 | 1367.00 | 2025-10-20 09:40:00 | 1367.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:45:00 | 1340.80 | 2025-10-24 11:30:00 | 1335.90 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-24 10:45:00 | 1340.80 | 2025-10-24 15:20:00 | 1333.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-10-27 09:30:00 | 1318.30 | 2025-10-27 09:35:00 | 1323.09 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-29 10:50:00 | 1310.60 | 2025-10-29 11:10:00 | 1306.19 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-06 10:40:00 | 1348.70 | 2025-11-06 11:30:00 | 1353.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-11-11 10:10:00 | 1335.20 | 2025-11-11 10:55:00 | 1341.08 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-11-18 10:00:00 | 1343.00 | 2025-11-18 10:20:00 | 1347.44 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-24 10:30:00 | 1289.70 | 2025-11-24 10:50:00 | 1293.24 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-26 10:55:00 | 1261.80 | 2025-11-26 11:20:00 | 1265.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-28 09:35:00 | 1261.40 | 2025-11-28 10:05:00 | 1266.07 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-10 09:55:00 | 1362.20 | 2025-12-10 10:15:00 | 1375.16 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2025-12-10 09:55:00 | 1362.20 | 2025-12-10 10:30:00 | 1362.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-16 09:55:00 | 1342.60 | 2025-12-16 11:40:00 | 1347.89 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-16 09:55:00 | 1342.60 | 2025-12-16 13:05:00 | 1342.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 09:30:00 | 1369.80 | 2025-12-17 09:40:00 | 1364.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-18 10:40:00 | 1370.60 | 2025-12-18 13:45:00 | 1379.25 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-12-18 10:40:00 | 1370.60 | 2025-12-18 15:20:00 | 1379.30 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-12-22 10:10:00 | 1432.00 | 2025-12-22 10:20:00 | 1441.32 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-12-22 10:10:00 | 1432.00 | 2025-12-22 11:20:00 | 1432.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1421.90 | 2026-01-05 10:35:00 | 1429.31 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1421.90 | 2026-01-05 11:50:00 | 1458.60 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2026-01-06 11:05:00 | 1462.90 | 2026-01-06 11:15:00 | 1457.61 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1484.70 | 2026-01-08 11:10:00 | 1475.53 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1484.70 | 2026-01-08 15:20:00 | 1456.00 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2026-01-13 09:35:00 | 1386.60 | 2026-01-13 09:45:00 | 1392.83 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-01-16 09:55:00 | 1414.50 | 2026-01-16 10:05:00 | 1422.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-01-16 09:55:00 | 1414.50 | 2026-01-16 15:00:00 | 1432.20 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2026-01-19 09:40:00 | 1397.20 | 2026-01-19 09:50:00 | 1403.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-01-20 11:15:00 | 1394.50 | 2026-01-20 11:25:00 | 1399.47 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-23 11:10:00 | 1346.20 | 2026-01-23 12:00:00 | 1350.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-28 11:15:00 | 1349.00 | 2026-01-28 11:50:00 | 1344.18 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-28 11:15:00 | 1349.00 | 2026-01-28 12:00:00 | 1349.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 09:50:00 | 1386.00 | 2026-01-30 10:10:00 | 1379.51 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-13 11:10:00 | 1393.00 | 2026-02-13 11:15:00 | 1388.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1402.00 | 2026-02-17 10:15:00 | 1398.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1415.60 | 2026-02-19 11:20:00 | 1408.54 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1415.60 | 2026-02-19 15:20:00 | 1389.10 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-02-23 10:35:00 | 1371.50 | 2026-02-23 10:45:00 | 1366.12 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-23 10:35:00 | 1371.50 | 2026-02-23 15:20:00 | 1365.10 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-25 10:45:00 | 1371.00 | 2026-02-25 10:50:00 | 1367.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1401.60 | 2026-02-26 09:55:00 | 1395.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-05 10:05:00 | 1313.00 | 2026-03-05 10:20:00 | 1306.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-03-18 09:35:00 | 1215.30 | 2026-03-18 09:40:00 | 1210.12 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1214.10 | 2026-03-20 09:35:00 | 1209.09 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-10 10:45:00 | 1365.00 | 2026-04-10 13:15:00 | 1357.40 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-10 10:45:00 | 1365.00 | 2026-04-10 15:20:00 | 1344.90 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2026-04-13 11:10:00 | 1330.60 | 2026-04-13 13:00:00 | 1326.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1381.00 | 2026-04-17 11:10:00 | 1387.97 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1381.00 | 2026-04-17 11:15:00 | 1381.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:55:00 | 1428.60 | 2026-04-22 10:00:00 | 1438.40 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-22 09:55:00 | 1428.60 | 2026-04-22 10:55:00 | 1428.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 1419.80 | 2026-04-28 10:10:00 | 1413.13 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-28 09:30:00 | 1419.80 | 2026-04-28 10:20:00 | 1419.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:35:00 | 1411.00 | 2026-04-29 10:45:00 | 1405.08 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-29 10:35:00 | 1411.00 | 2026-04-29 10:50:00 | 1411.00 | STOP_HIT | 0.50 | 0.00% |
