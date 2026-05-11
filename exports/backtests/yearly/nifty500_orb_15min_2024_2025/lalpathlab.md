# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1655.00
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
| ENTRY1 | 76 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 10 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 66
- **Target hits / Stop hits / Partials:** 10 / 66 / 31
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 12.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.16% | 9.8% |
| BUY @ 2nd Alert (retest1) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.16% | 9.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 16 | 34.0% | 2 | 31 | 14 | 0.07% | 3.2% |
| SELL @ 2nd Alert (retest1) | 47 | 16 | 34.0% | 2 | 31 | 14 | 0.07% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 107 | 41 | 38.3% | 10 | 66 | 31 | 0.12% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 11:10:00 | 1224.50 | 1234.32 | 0.00 | ORB-short ORB[1233.38,1248.97] vol=2.7x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 1228.86 | 1233.60 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:25:00 | 1265.25 | 1261.01 | 0.00 | ORB-long ORB[1251.15,1265.00] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:55:00 | 1272.11 | 1263.89 | 0.00 | T1 1.5R @ 1272.11 |
| Target hit | 2024-05-17 15:20:00 | 1275.47 | 1271.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 1274.22 | 1282.68 | 0.00 | ORB-short ORB[1275.63,1292.50] vol=2.1x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 1267.08 | 1281.08 | 0.00 | T1 1.5R @ 1267.08 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 1274.22 | 1279.31 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 1267.63 | 1276.84 | 0.00 | ORB-short ORB[1275.03,1291.00] vol=1.6x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-05-23 10:45:00 | 1271.72 | 1276.45 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 1297.70 | 1286.72 | 0.00 | ORB-long ORB[1277.47,1285.00] vol=2.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-05-24 10:50:00 | 1294.20 | 1287.53 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:50:00 | 1303.25 | 1296.05 | 0.00 | ORB-long ORB[1285.80,1302.38] vol=1.8x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 12:35:00 | 1310.06 | 1300.70 | 0.00 | T1 1.5R @ 1310.06 |
| Target hit | 2024-05-29 15:20:00 | 1327.50 | 1315.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:55:00 | 1384.53 | 1392.21 | 0.00 | ORB-short ORB[1392.28,1410.98] vol=6.1x ATR=5.85 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 1390.38 | 1391.87 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:35:00 | 1379.18 | 1384.37 | 0.00 | ORB-short ORB[1381.65,1399.95] vol=2.1x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:45:00 | 1372.86 | 1382.40 | 0.00 | T1 1.5R @ 1372.86 |
| Stop hit — per-position SL triggered | 2024-06-14 09:50:00 | 1379.18 | 1382.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:10:00 | 1349.13 | 1357.39 | 0.00 | ORB-short ORB[1356.15,1368.10] vol=2.1x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 1352.84 | 1357.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 1340.73 | 1335.65 | 0.00 | ORB-long ORB[1325.15,1339.90] vol=1.5x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:45:00 | 1347.03 | 1338.08 | 0.00 | T1 1.5R @ 1347.03 |
| Target hit | 2024-06-20 13:50:00 | 1348.95 | 1348.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 1339.10 | 1350.74 | 0.00 | ORB-short ORB[1346.00,1358.15] vol=1.9x ATR=3.97 |
| Stop hit — per-position SL triggered | 2024-06-21 10:50:00 | 1343.07 | 1350.66 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 1370.08 | 1362.63 | 0.00 | ORB-long ORB[1352.05,1364.10] vol=2.2x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-06-26 09:55:00 | 1366.18 | 1364.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 1386.23 | 1382.39 | 0.00 | ORB-long ORB[1375.98,1384.93] vol=3.2x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:45:00 | 1392.02 | 1384.13 | 0.00 | T1 1.5R @ 1392.02 |
| Stop hit — per-position SL triggered | 2024-06-27 10:25:00 | 1386.23 | 1387.10 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 1415.50 | 1403.99 | 0.00 | ORB-long ORB[1391.08,1407.50] vol=2.0x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:00:00 | 1424.05 | 1414.20 | 0.00 | T1 1.5R @ 1424.05 |
| Stop hit — per-position SL triggered | 2024-07-01 11:25:00 | 1415.50 | 1414.99 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 1423.65 | 1417.59 | 0.00 | ORB-long ORB[1412.60,1422.18] vol=2.0x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-07-02 10:30:00 | 1419.78 | 1417.81 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1416.50 | 1407.75 | 0.00 | ORB-long ORB[1391.95,1412.43] vol=2.7x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:40:00 | 1423.18 | 1412.98 | 0.00 | T1 1.5R @ 1423.18 |
| Stop hit — per-position SL triggered | 2024-07-03 10:35:00 | 1416.50 | 1416.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:10:00 | 1432.48 | 1420.23 | 0.00 | ORB-long ORB[1413.00,1426.93] vol=4.1x ATR=4.14 |
| Stop hit — per-position SL triggered | 2024-07-04 11:15:00 | 1428.34 | 1420.97 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 1437.58 | 1442.73 | 0.00 | ORB-short ORB[1438.03,1454.58] vol=1.9x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-07-08 09:35:00 | 1442.30 | 1442.01 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 1447.50 | 1450.32 | 0.00 | ORB-short ORB[1450.95,1466.75] vol=1.8x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 1440.61 | 1448.51 | 0.00 | T1 1.5R @ 1440.61 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 1447.50 | 1448.22 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 1499.10 | 1491.94 | 0.00 | ORB-long ORB[1480.03,1495.50] vol=5.2x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:55:00 | 1505.18 | 1492.90 | 0.00 | T1 1.5R @ 1505.18 |
| Stop hit — per-position SL triggered | 2024-07-19 10:05:00 | 1499.10 | 1493.72 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1527.35 | 1534.49 | 0.00 | ORB-short ORB[1528.28,1547.50] vol=3.8x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:35:00 | 1520.32 | 1533.58 | 0.00 | T1 1.5R @ 1520.32 |
| Stop hit — per-position SL triggered | 2024-07-23 11:45:00 | 1527.35 | 1532.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:40:00 | 1559.08 | 1545.15 | 0.00 | ORB-long ORB[1530.73,1553.10] vol=1.8x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-08-02 11:30:00 | 1554.20 | 1548.41 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 09:40:00 | 1546.58 | 1538.06 | 0.00 | ORB-long ORB[1525.08,1544.90] vol=1.6x ATR=5.65 |
| Stop hit — per-position SL triggered | 2024-08-05 09:45:00 | 1540.93 | 1538.90 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:40:00 | 1689.00 | 1678.52 | 0.00 | ORB-long ORB[1661.50,1670.00] vol=5.1x ATR=5.82 |
| Stop hit — per-position SL triggered | 2024-08-21 09:50:00 | 1683.18 | 1682.65 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:15:00 | 1658.63 | 1656.72 | 0.00 | ORB-long ORB[1645.08,1653.95] vol=2.3x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-08-22 11:20:00 | 1655.29 | 1656.59 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 1664.60 | 1661.25 | 0.00 | ORB-long ORB[1646.03,1662.23] vol=2.0x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-08-26 09:55:00 | 1659.85 | 1662.24 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:55:00 | 1668.70 | 1665.95 | 0.00 | ORB-long ORB[1655.48,1667.50] vol=2.0x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:05:00 | 1674.87 | 1668.20 | 0.00 | T1 1.5R @ 1674.87 |
| Stop hit — per-position SL triggered | 2024-08-27 10:30:00 | 1668.70 | 1670.66 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1665.00 | 1674.48 | 0.00 | ORB-short ORB[1673.15,1682.45] vol=2.0x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 1669.59 | 1673.35 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 11:10:00 | 1702.00 | 1691.38 | 0.00 | ORB-long ORB[1674.75,1685.95] vol=4.8x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:25:00 | 1709.03 | 1694.60 | 0.00 | T1 1.5R @ 1709.03 |
| Stop hit — per-position SL triggered | 2024-08-30 12:00:00 | 1702.00 | 1697.58 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 1700.00 | 1705.35 | 0.00 | ORB-short ORB[1707.50,1726.45] vol=2.3x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:20:00 | 1695.30 | 1703.61 | 0.00 | T1 1.5R @ 1695.30 |
| Stop hit — per-position SL triggered | 2024-09-02 12:10:00 | 1700.00 | 1701.33 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:25:00 | 1712.63 | 1705.76 | 0.00 | ORB-long ORB[1695.00,1703.48] vol=1.8x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-09-03 10:35:00 | 1708.84 | 1707.42 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:10:00 | 1627.05 | 1633.74 | 0.00 | ORB-short ORB[1634.53,1642.95] vol=2.6x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:20:00 | 1621.11 | 1632.11 | 0.00 | T1 1.5R @ 1621.11 |
| Stop hit — per-position SL triggered | 2024-09-18 11:45:00 | 1627.05 | 1625.89 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:00:00 | 1636.48 | 1644.49 | 0.00 | ORB-short ORB[1640.08,1657.98] vol=1.6x ATR=5.00 |
| Stop hit — per-position SL triggered | 2024-09-20 11:05:00 | 1641.48 | 1640.83 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:15:00 | 1666.95 | 1657.83 | 0.00 | ORB-long ORB[1640.85,1657.53] vol=4.2x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:20:00 | 1672.70 | 1663.63 | 0.00 | T1 1.5R @ 1672.70 |
| Target hit | 2024-09-24 11:35:00 | 1670.50 | 1671.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2024-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:55:00 | 1695.93 | 1681.11 | 0.00 | ORB-long ORB[1671.00,1685.00] vol=1.8x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-09-25 10:00:00 | 1690.62 | 1682.79 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:10:00 | 1631.40 | 1647.46 | 0.00 | ORB-short ORB[1650.53,1669.85] vol=2.2x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-09-26 10:20:00 | 1636.41 | 1645.79 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:40:00 | 1744.65 | 1728.02 | 0.00 | ORB-long ORB[1712.50,1729.98] vol=2.1x ATR=7.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:50:00 | 1755.90 | 1739.52 | 0.00 | T1 1.5R @ 1755.90 |
| Target hit | 2024-10-08 11:35:00 | 1764.33 | 1767.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 1768.70 | 1764.00 | 0.00 | ORB-long ORB[1754.23,1767.50] vol=1.6x ATR=4.95 |
| Stop hit — per-position SL triggered | 2024-10-11 09:55:00 | 1763.75 | 1764.10 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 1696.50 | 1702.60 | 0.00 | ORB-short ORB[1701.03,1716.43] vol=3.4x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:10:00 | 1686.58 | 1699.19 | 0.00 | T1 1.5R @ 1686.58 |
| Stop hit — per-position SL triggered | 2024-10-15 11:10:00 | 1696.50 | 1696.37 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:40:00 | 1669.80 | 1678.19 | 0.00 | ORB-short ORB[1671.05,1695.40] vol=2.2x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:50:00 | 1660.62 | 1674.03 | 0.00 | T1 1.5R @ 1660.62 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 1669.80 | 1664.09 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1684.10 | 1675.07 | 0.00 | ORB-long ORB[1656.05,1679.50] vol=1.9x ATR=6.89 |
| Stop hit — per-position SL triggered | 2024-10-18 09:50:00 | 1677.21 | 1677.60 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:05:00 | 1540.23 | 1543.37 | 0.00 | ORB-short ORB[1544.00,1557.00] vol=2.3x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 1533.65 | 1542.65 | 0.00 | T1 1.5R @ 1533.65 |
| Stop hit — per-position SL triggered | 2024-10-29 10:20:00 | 1540.23 | 1542.65 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:45:00 | 1531.93 | 1527.93 | 0.00 | ORB-long ORB[1518.00,1531.18] vol=2.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-10-30 13:40:00 | 1527.63 | 1530.59 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 10:30:00 | 1518.18 | 1526.37 | 0.00 | ORB-short ORB[1526.05,1535.00] vol=1.6x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:25:00 | 1511.69 | 1521.70 | 0.00 | T1 1.5R @ 1511.69 |
| Stop hit — per-position SL triggered | 2024-10-31 11:40:00 | 1518.18 | 1519.18 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:10:00 | 1479.50 | 1491.98 | 0.00 | ORB-short ORB[1496.00,1513.98] vol=1.6x ATR=6.07 |
| Stop hit — per-position SL triggered | 2024-11-13 10:20:00 | 1485.57 | 1490.15 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:35:00 | 1462.08 | 1472.08 | 0.00 | ORB-short ORB[1467.80,1488.38] vol=2.5x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 1467.46 | 1468.31 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:35:00 | 1515.25 | 1507.74 | 0.00 | ORB-long ORB[1485.23,1502.50] vol=1.6x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:00:00 | 1521.06 | 1509.23 | 0.00 | T1 1.5R @ 1521.06 |
| Stop hit — per-position SL triggered | 2024-11-19 11:20:00 | 1515.25 | 1510.14 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 1499.53 | 1508.98 | 0.00 | ORB-short ORB[1505.78,1521.48] vol=2.5x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 1494.56 | 1507.18 | 0.00 | T1 1.5R @ 1494.56 |
| Stop hit — per-position SL triggered | 2024-11-28 11:40:00 | 1499.53 | 1503.23 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:50:00 | 1492.85 | 1484.41 | 0.00 | ORB-long ORB[1480.50,1492.70] vol=1.6x ATR=4.08 |
| Stop hit — per-position SL triggered | 2024-12-05 10:55:00 | 1488.77 | 1484.74 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 11:15:00 | 1600.48 | 1585.23 | 0.00 | ORB-long ORB[1570.50,1590.80] vol=3.1x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-12-10 11:40:00 | 1595.76 | 1586.98 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:20:00 | 1552.95 | 1561.87 | 0.00 | ORB-short ORB[1560.55,1583.25] vol=2.8x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:30:00 | 1547.09 | 1560.16 | 0.00 | T1 1.5R @ 1547.09 |
| Stop hit — per-position SL triggered | 2024-12-12 10:50:00 | 1552.95 | 1557.79 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:40:00 | 1504.48 | 1512.61 | 0.00 | ORB-short ORB[1515.00,1526.23] vol=4.8x ATR=5.42 |
| Stop hit — per-position SL triggered | 2024-12-13 11:05:00 | 1509.90 | 1511.54 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:50:00 | 1512.58 | 1505.10 | 0.00 | ORB-long ORB[1487.38,1507.35] vol=1.6x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:10:00 | 1519.07 | 1508.00 | 0.00 | T1 1.5R @ 1519.07 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 1512.58 | 1508.68 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:25:00 | 1523.50 | 1520.75 | 0.00 | ORB-long ORB[1509.33,1521.58] vol=4.3x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:35:00 | 1529.33 | 1522.37 | 0.00 | T1 1.5R @ 1529.33 |
| Stop hit — per-position SL triggered | 2024-12-18 11:30:00 | 1523.50 | 1524.31 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1465.45 | 1461.09 | 0.00 | ORB-long ORB[1454.20,1464.68] vol=1.7x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-12-26 09:40:00 | 1462.33 | 1461.74 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 1481.40 | 1473.58 | 0.00 | ORB-long ORB[1465.53,1476.88] vol=1.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2024-12-30 10:40:00 | 1477.83 | 1474.25 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:40:00 | 1512.50 | 1508.22 | 0.00 | ORB-long ORB[1490.93,1506.40] vol=3.5x ATR=3.25 |
| Stop hit — per-position SL triggered | 2025-01-01 10:50:00 | 1509.25 | 1508.65 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:50:00 | 1476.98 | 1490.89 | 0.00 | ORB-short ORB[1485.00,1507.00] vol=3.1x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-01-06 09:55:00 | 1482.85 | 1489.96 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:50:00 | 1521.03 | 1526.78 | 0.00 | ORB-short ORB[1537.53,1555.50] vol=5.3x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:00:00 | 1512.29 | 1525.33 | 0.00 | T1 1.5R @ 1512.29 |
| Target hit | 2025-01-08 15:20:00 | 1497.45 | 1512.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 1486.93 | 1493.15 | 0.00 | ORB-short ORB[1492.03,1506.25] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-01-09 11:05:00 | 1491.14 | 1492.64 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:30:00 | 1490.43 | 1495.77 | 0.00 | ORB-short ORB[1491.50,1501.73] vol=2.0x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:40:00 | 1484.78 | 1493.68 | 0.00 | T1 1.5R @ 1484.78 |
| Target hit | 2025-01-10 15:20:00 | 1462.33 | 1471.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-01-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:00:00 | 1434.60 | 1443.77 | 0.00 | ORB-short ORB[1440.53,1453.53] vol=2.7x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-01-13 11:20:00 | 1439.66 | 1443.02 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:55:00 | 1406.50 | 1416.35 | 0.00 | ORB-short ORB[1407.88,1425.33] vol=2.0x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-01-21 10:00:00 | 1411.24 | 1415.57 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:55:00 | 1416.78 | 1409.65 | 0.00 | ORB-long ORB[1397.15,1409.68] vol=3.8x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-01-22 11:30:00 | 1412.70 | 1411.41 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:20:00 | 1412.45 | 1421.41 | 0.00 | ORB-short ORB[1416.10,1434.30] vol=1.7x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-01-24 11:35:00 | 1417.35 | 1411.52 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:35:00 | 1356.40 | 1367.38 | 0.00 | ORB-short ORB[1360.75,1380.98] vol=2.5x ATR=4.98 |
| Stop hit — per-position SL triggered | 2025-01-28 09:45:00 | 1361.38 | 1366.49 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:35:00 | 1401.50 | 1398.12 | 0.00 | ORB-long ORB[1379.50,1395.98] vol=15.3x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-01-30 09:45:00 | 1396.27 | 1398.13 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-03 11:15:00 | 1167.50 | 1163.82 | 0.00 | ORB-long ORB[1146.78,1164.13] vol=1.7x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-03-03 12:10:00 | 1163.88 | 1165.58 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:55:00 | 1271.97 | 1259.13 | 0.00 | ORB-long ORB[1249.80,1264.72] vol=2.3x ATR=6.16 |
| Stop hit — per-position SL triggered | 2025-03-12 11:35:00 | 1265.81 | 1263.28 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:25:00 | 1260.97 | 1255.09 | 0.00 | ORB-long ORB[1246.88,1260.93] vol=1.7x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-03-13 10:45:00 | 1256.73 | 1255.66 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 1229.93 | 1245.99 | 0.00 | ORB-short ORB[1244.43,1258.72] vol=1.8x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-03-26 09:45:00 | 1234.48 | 1243.98 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 09:30:00 | 1275.00 | 1265.20 | 0.00 | ORB-long ORB[1253.00,1270.38] vol=1.9x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:05:00 | 1283.28 | 1275.08 | 0.00 | T1 1.5R @ 1283.28 |
| Target hit | 2025-04-09 15:20:00 | 1330.28 | 1321.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:25:00 | 1385.30 | 1379.55 | 0.00 | ORB-long ORB[1364.50,1384.00] vol=2.3x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-04-16 13:00:00 | 1380.51 | 1381.75 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:15:00 | 1376.75 | 1371.50 | 0.00 | ORB-long ORB[1361.80,1375.25] vol=1.7x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:45:00 | 1383.14 | 1372.59 | 0.00 | T1 1.5R @ 1383.14 |
| Target hit | 2025-04-22 15:20:00 | 1382.00 | 1379.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:15:00 | 1412.75 | 1395.61 | 0.00 | ORB-long ORB[1378.55,1397.70] vol=3.1x ATR=5.95 |
| Stop hit — per-position SL triggered | 2025-04-24 10:25:00 | 1406.80 | 1397.67 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:40:00 | 1398.65 | 1381.00 | 0.00 | ORB-long ORB[1366.75,1385.05] vol=2.2x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 13:10:00 | 1406.91 | 1397.28 | 0.00 | T1 1.5R @ 1406.91 |
| Target hit | 2025-05-06 15:20:00 | 1402.30 | 1399.23 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 11:10:00 | 1224.50 | 2024-05-14 11:15:00 | 1228.86 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-17 10:25:00 | 1265.25 | 2024-05-17 10:55:00 | 1272.11 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-05-17 10:25:00 | 1265.25 | 2024-05-17 15:20:00 | 1275.47 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2024-05-22 09:35:00 | 1274.22 | 2024-05-22 09:40:00 | 1267.08 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-05-22 09:35:00 | 1274.22 | 2024-05-22 09:55:00 | 1274.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:35:00 | 1267.63 | 2024-05-23 10:45:00 | 1271.72 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-24 10:40:00 | 1297.70 | 2024-05-24 10:50:00 | 1294.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-29 09:50:00 | 1303.25 | 2024-05-29 12:35:00 | 1310.06 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-29 09:50:00 | 1303.25 | 2024-05-29 15:20:00 | 1327.50 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2024-06-12 09:55:00 | 1384.53 | 2024-06-12 10:05:00 | 1390.38 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-14 09:35:00 | 1379.18 | 2024-06-14 09:45:00 | 1372.86 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-06-14 09:35:00 | 1379.18 | 2024-06-14 09:50:00 | 1379.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 10:10:00 | 1349.13 | 2024-06-19 10:15:00 | 1352.84 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-20 09:30:00 | 1340.73 | 2024-06-20 09:45:00 | 1347.03 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-20 09:30:00 | 1340.73 | 2024-06-20 13:50:00 | 1348.95 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-21 10:45:00 | 1339.10 | 2024-06-21 10:50:00 | 1343.07 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-26 09:40:00 | 1370.08 | 2024-06-26 09:55:00 | 1366.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-27 09:40:00 | 1386.23 | 2024-06-27 09:45:00 | 1392.02 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-27 09:40:00 | 1386.23 | 2024-06-27 10:25:00 | 1386.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:35:00 | 1415.50 | 2024-07-01 11:00:00 | 1424.05 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-07-01 09:35:00 | 1415.50 | 2024-07-01 11:25:00 | 1415.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 10:20:00 | 1423.65 | 2024-07-02 10:30:00 | 1419.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1416.50 | 2024-07-03 09:40:00 | 1423.18 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1416.50 | 2024-07-03 10:35:00 | 1416.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 11:10:00 | 1432.48 | 2024-07-04 11:15:00 | 1428.34 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-08 09:30:00 | 1437.58 | 2024-07-08 09:35:00 | 1442.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-10 10:25:00 | 1447.50 | 2024-07-10 10:35:00 | 1440.61 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-10 10:25:00 | 1447.50 | 2024-07-10 10:40:00 | 1447.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-19 09:50:00 | 1499.10 | 2024-07-19 09:55:00 | 1505.18 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-19 09:50:00 | 1499.10 | 2024-07-19 10:05:00 | 1499.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1527.35 | 2024-07-23 11:35:00 | 1520.32 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1527.35 | 2024-07-23 11:45:00 | 1527.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-02 10:40:00 | 1559.08 | 2024-08-02 11:30:00 | 1554.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-05 09:40:00 | 1546.58 | 2024-08-05 09:45:00 | 1540.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-21 09:40:00 | 1689.00 | 2024-08-21 09:50:00 | 1683.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-22 11:15:00 | 1658.63 | 2024-08-22 11:20:00 | 1655.29 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-26 09:45:00 | 1664.60 | 2024-08-26 09:55:00 | 1659.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-27 09:55:00 | 1668.70 | 2024-08-27 10:05:00 | 1674.87 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-27 09:55:00 | 1668.70 | 2024-08-27 10:30:00 | 1668.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1665.00 | 2024-08-28 09:35:00 | 1669.59 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-30 11:10:00 | 1702.00 | 2024-08-30 11:25:00 | 1709.03 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-30 11:10:00 | 1702.00 | 2024-08-30 12:00:00 | 1702.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1700.00 | 2024-09-02 11:20:00 | 1695.30 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1700.00 | 2024-09-02 12:10:00 | 1700.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 10:25:00 | 1712.63 | 2024-09-03 10:35:00 | 1708.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-18 10:10:00 | 1627.05 | 2024-09-18 10:20:00 | 1621.11 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-18 10:10:00 | 1627.05 | 2024-09-18 11:45:00 | 1627.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-20 10:00:00 | 1636.48 | 2024-09-20 11:05:00 | 1641.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-24 10:15:00 | 1666.95 | 2024-09-24 10:20:00 | 1672.70 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-24 10:15:00 | 1666.95 | 2024-09-24 11:35:00 | 1670.50 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-09-25 09:55:00 | 1695.93 | 2024-09-25 10:00:00 | 1690.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-26 10:10:00 | 1631.40 | 2024-09-26 10:20:00 | 1636.41 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-08 09:40:00 | 1744.65 | 2024-10-08 09:50:00 | 1755.90 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-08 09:40:00 | 1744.65 | 2024-10-08 11:35:00 | 1764.33 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2024-10-11 09:50:00 | 1768.70 | 2024-10-11 09:55:00 | 1763.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-15 09:50:00 | 1696.50 | 2024-10-15 10:10:00 | 1686.58 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-15 09:50:00 | 1696.50 | 2024-10-15 11:10:00 | 1696.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 09:40:00 | 1669.80 | 2024-10-16 09:50:00 | 1660.62 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-16 09:40:00 | 1669.80 | 2024-10-16 10:35:00 | 1669.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1684.10 | 2024-10-18 09:50:00 | 1677.21 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-29 10:05:00 | 1540.23 | 2024-10-29 10:15:00 | 1533.65 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-29 10:05:00 | 1540.23 | 2024-10-29 10:20:00 | 1540.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 10:45:00 | 1531.93 | 2024-10-30 13:40:00 | 1527.63 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-31 10:30:00 | 1518.18 | 2024-10-31 11:25:00 | 1511.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-31 10:30:00 | 1518.18 | 2024-10-31 11:40:00 | 1518.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 10:10:00 | 1479.50 | 2024-11-13 10:20:00 | 1485.57 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-11-18 09:35:00 | 1462.08 | 2024-11-18 09:45:00 | 1467.46 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-19 10:35:00 | 1515.25 | 2024-11-19 11:00:00 | 1521.06 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-11-19 10:35:00 | 1515.25 | 2024-11-19 11:20:00 | 1515.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1499.53 | 2024-11-28 10:45:00 | 1494.56 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1499.53 | 2024-11-28 11:40:00 | 1499.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-05 10:50:00 | 1492.85 | 2024-12-05 10:55:00 | 1488.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-10 11:15:00 | 1600.48 | 2024-12-10 11:40:00 | 1595.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-12 10:20:00 | 1552.95 | 2024-12-12 10:30:00 | 1547.09 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-12 10:20:00 | 1552.95 | 2024-12-12 10:50:00 | 1552.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:40:00 | 1504.48 | 2024-12-13 11:05:00 | 1509.90 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-17 09:50:00 | 1512.58 | 2024-12-17 10:10:00 | 1519.07 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-17 09:50:00 | 1512.58 | 2024-12-17 10:20:00 | 1512.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 10:25:00 | 1523.50 | 2024-12-18 10:35:00 | 1529.33 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-18 10:25:00 | 1523.50 | 2024-12-18 11:30:00 | 1523.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 09:30:00 | 1465.45 | 2024-12-26 09:40:00 | 1462.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-30 10:35:00 | 1481.40 | 2024-12-30 10:40:00 | 1477.83 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-01 10:40:00 | 1512.50 | 2025-01-01 10:50:00 | 1509.25 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-06 09:50:00 | 1476.98 | 2025-01-06 09:55:00 | 1482.85 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-08 10:50:00 | 1521.03 | 2025-01-08 11:00:00 | 1512.29 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-08 10:50:00 | 1521.03 | 2025-01-08 15:20:00 | 1497.45 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2025-01-09 10:50:00 | 1486.93 | 2025-01-09 11:05:00 | 1491.14 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-10 09:30:00 | 1490.43 | 2025-01-10 09:40:00 | 1484.78 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-10 09:30:00 | 1490.43 | 2025-01-10 15:20:00 | 1462.33 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2025-01-13 11:00:00 | 1434.60 | 2025-01-13 11:20:00 | 1439.66 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-21 09:55:00 | 1406.50 | 2025-01-21 10:00:00 | 1411.24 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-22 10:55:00 | 1416.78 | 2025-01-22 11:30:00 | 1412.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-24 10:20:00 | 1412.45 | 2025-01-24 11:35:00 | 1417.35 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-28 09:35:00 | 1356.40 | 2025-01-28 09:45:00 | 1361.38 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-30 09:35:00 | 1401.50 | 2025-01-30 09:45:00 | 1396.27 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-03 11:15:00 | 1167.50 | 2025-03-03 12:10:00 | 1163.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-12 09:55:00 | 1271.97 | 2025-03-12 11:35:00 | 1265.81 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-13 10:25:00 | 1260.97 | 2025-03-13 10:45:00 | 1256.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-03-26 09:40:00 | 1229.93 | 2025-03-26 09:45:00 | 1234.48 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-09 09:30:00 | 1275.00 | 2025-04-09 10:05:00 | 1283.28 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-04-09 09:30:00 | 1275.00 | 2025-04-09 15:20:00 | 1330.28 | TARGET_HIT | 0.50 | 4.34% |
| BUY | retest1 | 2025-04-16 10:25:00 | 1385.30 | 2025-04-16 13:00:00 | 1380.51 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 10:15:00 | 1376.75 | 2025-04-22 10:45:00 | 1383.14 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-22 10:15:00 | 1376.75 | 2025-04-22 15:20:00 | 1382.00 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-24 10:15:00 | 1412.75 | 2025-04-24 10:25:00 | 1406.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-06 09:40:00 | 1398.65 | 2025-05-06 13:10:00 | 1406.91 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-05-06 09:40:00 | 1398.65 | 2025-05-06 15:20:00 | 1402.30 | TARGET_HIT | 0.50 | 0.26% |
