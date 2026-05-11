# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-01-05 15:25:00 (12238 bars)
- **Last close:** 2240.00
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 22
- **Target hits / Stop hits / Partials:** 10 / 22 / 19
- **Avg / median % per leg:** 0.35% / 0.34%
- **Sum % (uncompounded):** 17.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 12 | 52.2% | 4 | 11 | 8 | 0.23% | 5.4% |
| BUY @ 2nd Alert (retest1) | 23 | 12 | 52.2% | 4 | 11 | 8 | 0.23% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 17 | 60.7% | 6 | 11 | 11 | 0.44% | 12.3% |
| SELL @ 2nd Alert (retest1) | 28 | 17 | 60.7% | 6 | 11 | 11 | 0.44% | 12.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 51 | 29 | 56.9% | 10 | 22 | 19 | 0.35% | 17.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:35:00 | 1278.05 | 1284.43 | 0.00 | ORB-short ORB[1283.40,1293.80] vol=2.7x ATR=4.36 |
| Stop hit — per-position SL triggered | 2023-05-30 09:45:00 | 1282.41 | 1283.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-06-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 11:00:00 | 1367.20 | 1357.08 | 0.00 | ORB-long ORB[1345.50,1364.65] vol=4.0x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:05:00 | 1372.62 | 1358.44 | 0.00 | T1 1.5R @ 1372.62 |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 1367.20 | 1359.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:30:00 | 1470.40 | 1477.99 | 0.00 | ORB-short ORB[1472.00,1488.90] vol=1.8x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 10:00:00 | 1462.89 | 1472.90 | 0.00 | T1 1.5R @ 1462.89 |
| Target hit | 2023-06-28 12:15:00 | 1467.55 | 1467.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2023-07-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 09:45:00 | 1474.75 | 1484.45 | 0.00 | ORB-short ORB[1481.20,1498.50] vol=1.7x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:20:00 | 1468.46 | 1480.61 | 0.00 | T1 1.5R @ 1468.46 |
| Stop hit — per-position SL triggered | 2023-07-07 11:20:00 | 1474.75 | 1476.74 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 09:30:00 | 1474.00 | 1485.07 | 0.00 | ORB-short ORB[1481.00,1496.05] vol=1.5x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 10:10:00 | 1464.12 | 1475.86 | 0.00 | T1 1.5R @ 1464.12 |
| Target hit | 2023-07-10 15:20:00 | 1400.00 | 1433.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2023-07-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 11:10:00 | 1415.10 | 1427.96 | 0.00 | ORB-short ORB[1429.00,1448.15] vol=1.7x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 11:50:00 | 1408.13 | 1424.53 | 0.00 | T1 1.5R @ 1408.13 |
| Stop hit — per-position SL triggered | 2023-07-12 12:10:00 | 1415.10 | 1423.63 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-07-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:35:00 | 1439.75 | 1434.02 | 0.00 | ORB-long ORB[1419.30,1439.45] vol=1.6x ATR=4.99 |
| Stop hit — per-position SL triggered | 2023-07-13 09:45:00 | 1434.76 | 1435.10 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 1456.00 | 1465.94 | 0.00 | ORB-short ORB[1461.00,1479.00] vol=1.9x ATR=6.08 |
| Stop hit — per-position SL triggered | 2023-07-18 09:55:00 | 1462.08 | 1463.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-07-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:35:00 | 1480.00 | 1473.71 | 0.00 | ORB-long ORB[1458.00,1479.00] vol=5.3x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:40:00 | 1489.82 | 1479.00 | 0.00 | T1 1.5R @ 1489.82 |
| Target hit | 2023-07-19 15:20:00 | 1485.00 | 1484.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2023-07-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:45:00 | 1460.90 | 1464.01 | 0.00 | ORB-short ORB[1461.65,1476.30] vol=6.4x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:10:00 | 1456.64 | 1463.33 | 0.00 | T1 1.5R @ 1456.64 |
| Target hit | 2023-07-27 15:20:00 | 1435.60 | 1450.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-08-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 10:40:00 | 1471.60 | 1464.40 | 0.00 | ORB-long ORB[1456.30,1469.05] vol=1.5x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:45:00 | 1477.84 | 1466.54 | 0.00 | T1 1.5R @ 1477.84 |
| Stop hit — per-position SL triggered | 2023-08-02 10:55:00 | 1471.60 | 1472.14 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-08-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 11:00:00 | 1500.00 | 1500.38 | 0.00 | ORB-short ORB[1500.50,1519.60] vol=1.6x ATR=4.19 |
| Stop hit — per-position SL triggered | 2023-08-03 15:20:00 | 1505.05 | 1499.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-08-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:50:00 | 1519.95 | 1509.30 | 0.00 | ORB-long ORB[1499.10,1511.40] vol=4.2x ATR=5.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 09:55:00 | 1528.82 | 1523.92 | 0.00 | T1 1.5R @ 1528.82 |
| Target hit | 2023-08-04 15:20:00 | 1560.00 | 1555.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:40:00 | 1526.30 | 1537.83 | 0.00 | ORB-short ORB[1534.05,1550.90] vol=1.9x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 10:15:00 | 1517.86 | 1531.04 | 0.00 | T1 1.5R @ 1517.86 |
| Stop hit — per-position SL triggered | 2023-08-11 10:20:00 | 1526.30 | 1531.51 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-08-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:50:00 | 1548.95 | 1541.68 | 0.00 | ORB-long ORB[1530.10,1545.95] vol=2.1x ATR=5.04 |
| Stop hit — per-position SL triggered | 2023-08-17 11:20:00 | 1543.91 | 1545.68 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-08-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:55:00 | 1565.00 | 1546.87 | 0.00 | ORB-long ORB[1541.00,1559.70] vol=3.5x ATR=4.85 |
| Stop hit — per-position SL triggered | 2023-08-23 11:00:00 | 1560.15 | 1548.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-08-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:50:00 | 1571.50 | 1561.45 | 0.00 | ORB-long ORB[1552.00,1564.05] vol=3.7x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 10:05:00 | 1579.86 | 1569.99 | 0.00 | T1 1.5R @ 1579.86 |
| Target hit | 2023-08-24 11:10:00 | 1580.00 | 1581.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2023-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:45:00 | 1594.00 | 1581.14 | 0.00 | ORB-long ORB[1571.30,1588.65] vol=3.6x ATR=5.24 |
| Stop hit — per-position SL triggered | 2023-08-29 11:00:00 | 1588.76 | 1586.24 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:15:00 | 1626.80 | 1614.56 | 0.00 | ORB-long ORB[1596.05,1620.00] vol=1.6x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:40:00 | 1635.04 | 1623.73 | 0.00 | T1 1.5R @ 1635.04 |
| Target hit | 2023-08-31 13:00:00 | 1628.80 | 1629.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2023-09-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:35:00 | 1812.00 | 1826.19 | 0.00 | ORB-short ORB[1816.85,1844.00] vol=1.8x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:40:00 | 1803.83 | 1825.04 | 0.00 | T1 1.5R @ 1803.83 |
| Target hit | 2023-09-08 15:20:00 | 1786.00 | 1804.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-10-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 10:45:00 | 1683.05 | 1690.41 | 0.00 | ORB-short ORB[1686.85,1704.00] vol=2.5x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:25:00 | 1674.84 | 1686.81 | 0.00 | T1 1.5R @ 1674.84 |
| Stop hit — per-position SL triggered | 2023-10-04 12:10:00 | 1683.05 | 1679.71 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-10-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 11:10:00 | 1721.00 | 1701.85 | 0.00 | ORB-long ORB[1686.25,1705.00] vol=6.3x ATR=4.66 |
| Stop hit — per-position SL triggered | 2023-10-11 11:15:00 | 1716.34 | 1704.47 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 09:45:00 | 1598.25 | 1618.88 | 0.00 | ORB-short ORB[1619.00,1643.00] vol=1.6x ATR=8.69 |
| Stop hit — per-position SL triggered | 2023-10-30 09:50:00 | 1606.94 | 1617.47 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-11-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:45:00 | 1666.15 | 1655.17 | 0.00 | ORB-long ORB[1642.45,1663.90] vol=2.7x ATR=7.87 |
| Stop hit — per-position SL triggered | 2023-11-02 09:55:00 | 1658.28 | 1655.96 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 09:45:00 | 1653.25 | 1668.78 | 0.00 | ORB-short ORB[1665.00,1680.00] vol=1.6x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 09:50:00 | 1645.21 | 1665.57 | 0.00 | T1 1.5R @ 1645.21 |
| Stop hit — per-position SL triggered | 2023-11-08 10:15:00 | 1653.25 | 1661.84 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-11-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 10:05:00 | 1648.50 | 1652.40 | 0.00 | ORB-short ORB[1649.00,1660.00] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 1652.67 | 1652.16 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-11-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:40:00 | 1701.25 | 1690.78 | 0.00 | ORB-long ORB[1682.05,1691.85] vol=3.5x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 10:45:00 | 1707.30 | 1692.90 | 0.00 | T1 1.5R @ 1707.30 |
| Stop hit — per-position SL triggered | 2023-11-16 10:50:00 | 1701.25 | 1693.18 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-11-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 10:10:00 | 1824.30 | 1811.14 | 0.00 | ORB-long ORB[1798.65,1819.10] vol=1.7x ATR=6.27 |
| Stop hit — per-position SL triggered | 2023-11-22 10:45:00 | 1818.03 | 1815.84 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1962.65 | 1982.69 | 0.00 | ORB-short ORB[1981.20,2000.00] vol=2.9x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 12:35:00 | 1953.95 | 1975.83 | 0.00 | T1 1.5R @ 1953.95 |
| Target hit | 2023-12-08 15:20:00 | 1955.05 | 1960.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-12-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:30:00 | 2036.00 | 2025.35 | 0.00 | ORB-long ORB[2005.00,2034.00] vol=1.7x ATR=9.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 09:45:00 | 2050.20 | 2033.38 | 0.00 | T1 1.5R @ 2050.20 |
| Stop hit — per-position SL triggered | 2023-12-14 09:55:00 | 2036.00 | 2035.25 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:30:00 | 2344.00 | 2353.40 | 0.00 | ORB-short ORB[2350.15,2381.00] vol=2.7x ATR=10.24 |
| Stop hit — per-position SL triggered | 2023-12-26 10:40:00 | 2354.24 | 2353.31 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 2301.85 | 2321.16 | 0.00 | ORB-short ORB[2310.25,2334.10] vol=2.9x ATR=11.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 09:35:00 | 2285.21 | 2303.87 | 0.00 | T1 1.5R @ 2285.21 |
| Target hit | 2024-01-03 10:10:00 | 2293.25 | 2289.64 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-30 09:35:00 | 1278.05 | 2023-05-30 09:45:00 | 1282.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-06-08 11:00:00 | 1367.20 | 2023-06-08 11:05:00 | 1372.62 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-06-08 11:00:00 | 1367.20 | 2023-06-08 11:15:00 | 1367.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-28 09:30:00 | 1470.40 | 2023-06-28 10:00:00 | 1462.89 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-06-28 09:30:00 | 1470.40 | 2023-06-28 12:15:00 | 1467.55 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-07-07 09:45:00 | 1474.75 | 2023-07-07 10:20:00 | 1468.46 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-07-07 09:45:00 | 1474.75 | 2023-07-07 11:20:00 | 1474.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-10 09:30:00 | 1474.00 | 2023-07-10 10:10:00 | 1464.12 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2023-07-10 09:30:00 | 1474.00 | 2023-07-10 15:20:00 | 1400.00 | TARGET_HIT | 0.50 | 5.02% |
| SELL | retest1 | 2023-07-12 11:10:00 | 1415.10 | 2023-07-12 11:50:00 | 1408.13 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-07-12 11:10:00 | 1415.10 | 2023-07-12 12:10:00 | 1415.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-13 09:35:00 | 1439.75 | 2023-07-13 09:45:00 | 1434.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-18 09:35:00 | 1456.00 | 2023-07-18 09:55:00 | 1462.08 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-07-19 09:35:00 | 1480.00 | 2023-07-19 10:40:00 | 1489.82 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-07-19 09:35:00 | 1480.00 | 2023-07-19 15:20:00 | 1485.00 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2023-07-27 10:45:00 | 1460.90 | 2023-07-27 11:10:00 | 1456.64 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-07-27 10:45:00 | 1460.90 | 2023-07-27 15:20:00 | 1435.60 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2023-08-02 10:40:00 | 1471.60 | 2023-08-02 10:45:00 | 1477.84 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-08-02 10:40:00 | 1471.60 | 2023-08-02 10:55:00 | 1471.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-03 11:00:00 | 1500.00 | 2023-08-03 15:20:00 | 1505.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-04 09:50:00 | 1519.95 | 2023-08-04 09:55:00 | 1528.82 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-08-04 09:50:00 | 1519.95 | 2023-08-04 15:20:00 | 1560.00 | TARGET_HIT | 0.50 | 2.63% |
| SELL | retest1 | 2023-08-11 09:40:00 | 1526.30 | 2023-08-11 10:15:00 | 1517.86 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-08-11 09:40:00 | 1526.30 | 2023-08-11 10:20:00 | 1526.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 09:50:00 | 1548.95 | 2023-08-17 11:20:00 | 1543.91 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-08-23 10:55:00 | 1565.00 | 2023-08-23 11:00:00 | 1560.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-24 09:50:00 | 1571.50 | 2023-08-24 10:05:00 | 1579.86 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-08-24 09:50:00 | 1571.50 | 2023-08-24 11:10:00 | 1580.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2023-08-29 10:45:00 | 1594.00 | 2023-08-29 11:00:00 | 1588.76 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-08-31 10:15:00 | 1626.80 | 2023-08-31 11:40:00 | 1635.04 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-31 10:15:00 | 1626.80 | 2023-08-31 13:00:00 | 1628.80 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2023-09-08 10:35:00 | 1812.00 | 2023-09-08 10:40:00 | 1803.83 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-09-08 10:35:00 | 1812.00 | 2023-09-08 15:20:00 | 1786.00 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2023-10-04 10:45:00 | 1683.05 | 2023-10-04 11:25:00 | 1674.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-04 10:45:00 | 1683.05 | 2023-10-04 12:10:00 | 1683.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 11:10:00 | 1721.00 | 2023-10-11 11:15:00 | 1716.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-30 09:45:00 | 1598.25 | 2023-10-30 09:50:00 | 1606.94 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2023-11-02 09:45:00 | 1666.15 | 2023-11-02 09:55:00 | 1658.28 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-11-08 09:45:00 | 1653.25 | 2023-11-08 09:50:00 | 1645.21 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-11-08 09:45:00 | 1653.25 | 2023-11-08 10:15:00 | 1653.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-10 10:05:00 | 1648.50 | 2023-11-10 10:15:00 | 1652.67 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-16 10:40:00 | 1701.25 | 2023-11-16 10:45:00 | 1707.30 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-16 10:40:00 | 1701.25 | 2023-11-16 10:50:00 | 1701.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 10:10:00 | 1824.30 | 2023-11-22 10:45:00 | 1818.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1962.65 | 2023-12-08 12:35:00 | 1953.95 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1962.65 | 2023-12-08 15:20:00 | 1955.05 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-12-14 09:30:00 | 2036.00 | 2023-12-14 09:45:00 | 2050.20 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2023-12-14 09:30:00 | 2036.00 | 2023-12-14 09:55:00 | 2036.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-26 10:30:00 | 2344.00 | 2023-12-26 10:40:00 | 2354.24 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-01-03 09:30:00 | 2301.85 | 2024-01-03 09:35:00 | 2285.21 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-01-03 09:30:00 | 2301.85 | 2024-01-03 10:10:00 | 2293.25 | TARGET_HIT | 0.50 | 0.37% |
