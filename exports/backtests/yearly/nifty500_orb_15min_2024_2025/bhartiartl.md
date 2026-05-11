# Bharti Airtel Ltd. (BHARTIARTL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1834.70
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
| PARTIAL | 31 |
| TARGET_HIT | 15 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 52
- **Target hits / Stop hits / Partials:** 15 / 52 / 31
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 16.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 28 | 48.3% | 10 | 30 | 18 | 0.22% | 12.9% |
| BUY @ 2nd Alert (retest1) | 58 | 28 | 48.3% | 10 | 30 | 18 | 0.22% | 12.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 18 | 45.0% | 5 | 22 | 13 | 0.10% | 3.9% |
| SELL @ 2nd Alert (retest1) | 40 | 18 | 45.0% | 5 | 22 | 13 | 0.10% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 46 | 46.9% | 15 | 52 | 31 | 0.17% | 16.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:35:00 | 1284.25 | 1292.14 | 0.00 | ORB-short ORB[1290.20,1302.90] vol=2.1x ATR=6.01 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 1290.26 | 1290.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 1344.75 | 1333.47 | 0.00 | ORB-long ORB[1321.30,1339.45] vol=2.0x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:45:00 | 1353.80 | 1337.57 | 0.00 | T1 1.5R @ 1353.80 |
| Stop hit — per-position SL triggered | 2024-05-16 10:05:00 | 1344.75 | 1339.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 1346.35 | 1353.37 | 0.00 | ORB-short ORB[1348.55,1362.75] vol=1.8x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-05-21 10:20:00 | 1350.34 | 1352.57 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:45:00 | 1356.75 | 1353.79 | 0.00 | ORB-long ORB[1344.30,1356.10] vol=1.6x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:50:00 | 1361.53 | 1354.81 | 0.00 | T1 1.5R @ 1361.53 |
| Target hit | 2024-05-23 11:05:00 | 1360.40 | 1362.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-06-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:55:00 | 1432.50 | 1430.73 | 0.00 | ORB-long ORB[1421.05,1431.45] vol=3.0x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-06-10 12:10:00 | 1428.26 | 1431.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:25:00 | 1440.05 | 1443.48 | 0.00 | ORB-short ORB[1443.00,1455.95] vol=4.3x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:50:00 | 1435.20 | 1442.53 | 0.00 | T1 1.5R @ 1435.20 |
| Target hit | 2024-06-13 15:20:00 | 1429.55 | 1434.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:55:00 | 1425.40 | 1433.22 | 0.00 | ORB-short ORB[1430.50,1444.00] vol=1.6x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:25:00 | 1420.76 | 1431.51 | 0.00 | T1 1.5R @ 1420.76 |
| Stop hit — per-position SL triggered | 2024-06-18 12:55:00 | 1425.40 | 1427.61 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:30:00 | 1414.60 | 1419.30 | 0.00 | ORB-short ORB[1422.00,1434.80] vol=2.0x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:45:00 | 1409.90 | 1417.56 | 0.00 | T1 1.5R @ 1409.90 |
| Stop hit — per-position SL triggered | 2024-06-19 11:35:00 | 1414.60 | 1415.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 1384.50 | 1396.06 | 0.00 | ORB-short ORB[1394.00,1406.95] vol=2.0x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 13:55:00 | 1379.08 | 1389.19 | 0.00 | T1 1.5R @ 1379.08 |
| Target hit | 2024-06-20 15:20:00 | 1380.25 | 1386.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:10:00 | 1411.30 | 1422.56 | 0.00 | ORB-short ORB[1421.65,1434.00] vol=2.0x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-06-25 11:15:00 | 1414.60 | 1421.97 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 1424.35 | 1416.31 | 0.00 | ORB-long ORB[1408.95,1421.95] vol=2.0x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:20:00 | 1429.58 | 1418.21 | 0.00 | T1 1.5R @ 1429.58 |
| Stop hit — per-position SL triggered | 2024-06-26 11:40:00 | 1424.35 | 1419.68 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 11:05:00 | 1435.50 | 1431.70 | 0.00 | ORB-long ORB[1423.15,1435.40] vol=2.2x ATR=2.89 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 1432.61 | 1431.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 1445.50 | 1439.77 | 0.00 | ORB-long ORB[1435.05,1441.00] vol=2.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-07-10 09:35:00 | 1442.54 | 1440.39 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:10:00 | 1428.00 | 1438.26 | 0.00 | ORB-short ORB[1444.20,1450.85] vol=2.0x ATR=3.05 |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 1431.05 | 1437.78 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:00:00 | 1444.80 | 1433.10 | 0.00 | ORB-long ORB[1434.25,1442.00] vol=2.9x ATR=4.25 |
| Stop hit — per-position SL triggered | 2024-07-12 11:05:00 | 1440.55 | 1433.55 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:40:00 | 1465.85 | 1457.90 | 0.00 | ORB-long ORB[1445.00,1461.55] vol=3.4x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-07-16 13:55:00 | 1460.87 | 1463.13 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:15:00 | 1505.75 | 1497.03 | 0.00 | ORB-long ORB[1485.00,1499.80] vol=1.9x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:25:00 | 1510.23 | 1501.26 | 0.00 | T1 1.5R @ 1510.23 |
| Stop hit — per-position SL triggered | 2024-08-01 12:45:00 | 1505.75 | 1502.25 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:55:00 | 1472.35 | 1463.83 | 0.00 | ORB-long ORB[1459.70,1469.90] vol=1.7x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:20:00 | 1477.23 | 1466.72 | 0.00 | T1 1.5R @ 1477.23 |
| Stop hit — per-position SL triggered | 2024-08-12 11:25:00 | 1472.35 | 1466.91 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:40:00 | 1451.00 | 1456.04 | 0.00 | ORB-short ORB[1452.70,1463.40] vol=3.0x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-08-21 09:50:00 | 1453.91 | 1455.63 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:15:00 | 1495.10 | 1480.23 | 0.00 | ORB-long ORB[1465.00,1474.40] vol=2.4x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 11:25:00 | 1499.87 | 1481.75 | 0.00 | T1 1.5R @ 1499.87 |
| Stop hit — per-position SL triggered | 2024-08-22 11:30:00 | 1495.10 | 1482.13 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 11:10:00 | 1604.85 | 1585.30 | 0.00 | ORB-long ORB[1562.15,1576.00] vol=1.7x ATR=5.48 |
| Stop hit — per-position SL triggered | 2024-08-30 11:20:00 | 1599.37 | 1586.91 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:55:00 | 1564.20 | 1569.03 | 0.00 | ORB-short ORB[1569.05,1579.00] vol=4.4x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-09-03 10:05:00 | 1567.49 | 1568.73 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:15:00 | 1539.70 | 1534.47 | 0.00 | ORB-long ORB[1523.25,1538.55] vol=2.4x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-09-09 11:25:00 | 1536.40 | 1535.09 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 11:15:00 | 1650.15 | 1645.06 | 0.00 | ORB-long ORB[1638.50,1650.00] vol=1.9x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:35:00 | 1653.98 | 1646.21 | 0.00 | T1 1.5R @ 1653.98 |
| Target hit | 2024-09-17 15:20:00 | 1662.00 | 1656.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 1676.85 | 1665.26 | 0.00 | ORB-long ORB[1647.70,1672.75] vol=2.6x ATR=4.67 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 1672.18 | 1665.76 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1680.35 | 1672.87 | 0.00 | ORB-long ORB[1667.55,1678.55] vol=1.8x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:40:00 | 1688.90 | 1673.95 | 0.00 | T1 1.5R @ 1688.90 |
| Target hit | 2024-09-20 13:40:00 | 1690.00 | 1691.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 1761.30 | 1754.20 | 0.00 | ORB-long ORB[1746.10,1758.65] vol=1.8x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-09-24 11:00:00 | 1756.62 | 1758.08 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:55:00 | 1773.20 | 1769.52 | 0.00 | ORB-long ORB[1759.40,1771.00] vol=1.8x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-09-26 10:55:00 | 1768.31 | 1772.27 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 1671.30 | 1677.05 | 0.00 | ORB-short ORB[1677.00,1698.95] vol=2.0x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-10-10 12:40:00 | 1675.09 | 1674.62 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:05:00 | 1710.40 | 1723.86 | 0.00 | ORB-short ORB[1726.80,1737.95] vol=2.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-10-17 10:10:00 | 1714.88 | 1723.08 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:05:00 | 1645.65 | 1630.24 | 0.00 | ORB-long ORB[1626.00,1643.25] vol=1.8x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-10-30 11:50:00 | 1640.64 | 1634.08 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 10:50:00 | 1606.00 | 1624.35 | 0.00 | ORB-short ORB[1625.00,1638.80] vol=3.5x ATR=4.50 |
| Stop hit — per-position SL triggered | 2024-10-31 10:55:00 | 1610.50 | 1622.73 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:10:00 | 1586.55 | 1602.51 | 0.00 | ORB-short ORB[1602.00,1623.85] vol=2.2x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:55:00 | 1576.96 | 1598.34 | 0.00 | T1 1.5R @ 1576.96 |
| Stop hit — per-position SL triggered | 2024-11-04 12:00:00 | 1586.55 | 1594.63 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:25:00 | 1576.90 | 1567.67 | 0.00 | ORB-long ORB[1551.80,1569.20] vol=2.0x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-11-11 10:55:00 | 1572.94 | 1570.56 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:35:00 | 1586.60 | 1582.85 | 0.00 | ORB-long ORB[1567.20,1579.70] vol=1.6x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:00:00 | 1593.20 | 1584.37 | 0.00 | T1 1.5R @ 1593.20 |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 1586.60 | 1584.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 11:15:00 | 1537.40 | 1531.17 | 0.00 | ORB-long ORB[1521.85,1537.25] vol=1.7x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 12:00:00 | 1543.59 | 1533.06 | 0.00 | T1 1.5R @ 1543.59 |
| Target hit | 2024-11-22 15:20:00 | 1570.55 | 1551.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:00:00 | 1581.45 | 1584.81 | 0.00 | ORB-short ORB[1585.00,1604.85] vol=2.2x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:50:00 | 1574.01 | 1583.62 | 0.00 | T1 1.5R @ 1574.01 |
| Target hit | 2024-11-26 14:35:00 | 1579.25 | 1579.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 1568.30 | 1574.76 | 0.00 | ORB-short ORB[1568.55,1579.40] vol=1.9x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 12:00:00 | 1559.66 | 1571.24 | 0.00 | T1 1.5R @ 1559.66 |
| Stop hit — per-position SL triggered | 2024-11-28 13:30:00 | 1568.30 | 1568.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 1583.95 | 1592.38 | 0.00 | ORB-short ORB[1590.05,1611.85] vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-12-10 10:05:00 | 1587.30 | 1591.47 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:45:00 | 1621.80 | 1618.28 | 0.00 | ORB-long ORB[1606.80,1619.90] vol=2.0x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:00:00 | 1629.09 | 1619.92 | 0.00 | T1 1.5R @ 1629.09 |
| Target hit | 2024-12-13 15:20:00 | 1683.35 | 1654.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:05:00 | 1605.95 | 1615.57 | 0.00 | ORB-short ORB[1612.00,1623.00] vol=1.7x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:30:00 | 1601.11 | 1613.95 | 0.00 | T1 1.5R @ 1601.11 |
| Stop hit — per-position SL triggered | 2024-12-18 14:00:00 | 1605.95 | 1608.14 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1603.75 | 1597.44 | 0.00 | ORB-long ORB[1582.10,1601.45] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-12-23 11:55:00 | 1599.73 | 1598.78 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 1627.80 | 1615.89 | 0.00 | ORB-long ORB[1599.50,1615.90] vol=1.5x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-12-27 10:00:00 | 1623.09 | 1616.62 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:40:00 | 1595.55 | 1591.48 | 0.00 | ORB-long ORB[1584.05,1595.00] vol=1.8x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:55:00 | 1601.45 | 1592.80 | 0.00 | T1 1.5R @ 1601.45 |
| Stop hit — per-position SL triggered | 2025-01-01 11:35:00 | 1595.55 | 1595.33 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 1600.30 | 1595.50 | 0.00 | ORB-long ORB[1588.40,1599.90] vol=4.3x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:50:00 | 1606.22 | 1599.10 | 0.00 | T1 1.5R @ 1606.22 |
| Target hit | 2025-01-02 11:45:00 | 1601.55 | 1601.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 1587.45 | 1599.78 | 0.00 | ORB-short ORB[1596.00,1609.85] vol=2.0x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:35:00 | 1580.70 | 1597.36 | 0.00 | T1 1.5R @ 1580.70 |
| Stop hit — per-position SL triggered | 2025-01-06 14:55:00 | 1587.45 | 1589.29 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 11:05:00 | 1585.85 | 1591.97 | 0.00 | ORB-short ORB[1588.50,1603.00] vol=1.7x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 1589.45 | 1591.76 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 11:00:00 | 1604.90 | 1599.72 | 0.00 | ORB-long ORB[1592.40,1604.00] vol=2.2x ATR=3.82 |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 1601.08 | 1600.40 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 1619.55 | 1614.87 | 0.00 | ORB-long ORB[1600.55,1619.05] vol=1.8x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:45:00 | 1624.59 | 1616.42 | 0.00 | T1 1.5R @ 1624.59 |
| Target hit | 2025-01-16 15:20:00 | 1630.50 | 1624.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 1629.60 | 1635.31 | 0.00 | ORB-short ORB[1630.05,1654.00] vol=4.6x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:25:00 | 1623.14 | 1634.35 | 0.00 | T1 1.5R @ 1623.14 |
| Stop hit — per-position SL triggered | 2025-01-21 10:40:00 | 1629.60 | 1633.15 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 11:00:00 | 1637.45 | 1635.54 | 0.00 | ORB-long ORB[1622.55,1637.00] vol=2.2x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-01-22 11:05:00 | 1633.77 | 1635.50 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 11:00:00 | 1612.00 | 1605.49 | 0.00 | ORB-long ORB[1597.10,1609.85] vol=3.5x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-01-28 11:35:00 | 1608.80 | 1607.22 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-02-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:45:00 | 1656.15 | 1663.99 | 0.00 | ORB-short ORB[1661.25,1675.20] vol=1.8x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:10:00 | 1649.70 | 1662.70 | 0.00 | T1 1.5R @ 1649.70 |
| Stop hit — per-position SL triggered | 2025-02-05 11:25:00 | 1656.15 | 1661.91 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-17 10:15:00 | 1689.90 | 1700.57 | 0.00 | ORB-short ORB[1703.85,1716.90] vol=1.7x ATR=3.86 |
| Stop hit — per-position SL triggered | 2025-02-17 10:20:00 | 1693.76 | 1699.93 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:50:00 | 1642.35 | 1650.88 | 0.00 | ORB-short ORB[1642.95,1656.90] vol=2.6x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 13:50:00 | 1636.87 | 1646.78 | 0.00 | T1 1.5R @ 1636.87 |
| Target hit | 2025-02-21 15:20:00 | 1638.05 | 1642.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:25:00 | 1612.60 | 1623.53 | 0.00 | ORB-short ORB[1623.25,1637.55] vol=1.8x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 12:00:00 | 1605.23 | 1618.69 | 0.00 | T1 1.5R @ 1605.23 |
| Target hit | 2025-02-24 15:20:00 | 1601.40 | 1609.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:55:00 | 1629.95 | 1620.38 | 0.00 | ORB-long ORB[1597.00,1610.85] vol=1.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 11:35:00 | 1633.86 | 1622.54 | 0.00 | T1 1.5R @ 1633.86 |
| Target hit | 2025-02-25 15:20:00 | 1640.70 | 1633.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 09:45:00 | 1569.95 | 1574.91 | 0.00 | ORB-short ORB[1575.20,1590.50] vol=3.5x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-03-04 09:55:00 | 1574.55 | 1574.15 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:55:00 | 1600.80 | 1592.99 | 0.00 | ORB-long ORB[1577.10,1590.80] vol=1.5x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:15:00 | 1605.80 | 1594.34 | 0.00 | T1 1.5R @ 1605.80 |
| Target hit | 2025-03-05 15:20:00 | 1618.95 | 1612.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 1762.15 | 1751.38 | 0.00 | ORB-long ORB[1733.10,1759.00] vol=1.9x ATR=6.04 |
| Stop hit — per-position SL triggered | 2025-03-26 09:35:00 | 1756.11 | 1751.75 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:50:00 | 1751.55 | 1738.28 | 0.00 | ORB-long ORB[1716.95,1741.45] vol=1.6x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 10:15:00 | 1758.83 | 1741.78 | 0.00 | T1 1.5R @ 1758.83 |
| Stop hit — per-position SL triggered | 2025-04-02 10:35:00 | 1751.55 | 1743.03 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:35:00 | 1758.40 | 1752.54 | 0.00 | ORB-long ORB[1743.90,1756.20] vol=1.6x ATR=3.99 |
| Stop hit — per-position SL triggered | 2025-04-04 09:50:00 | 1754.41 | 1753.79 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1845.70 | 1836.92 | 0.00 | ORB-long ORB[1825.70,1844.00] vol=1.5x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:30:00 | 1851.47 | 1838.93 | 0.00 | T1 1.5R @ 1851.47 |
| Target hit | 2025-04-17 15:20:00 | 1881.20 | 1866.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 1842.80 | 1836.73 | 0.00 | ORB-long ORB[1820.00,1840.70] vol=1.7x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 1838.59 | 1837.60 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:40:00 | 1871.00 | 1862.13 | 0.00 | ORB-long ORB[1851.90,1868.00] vol=2.2x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-05-05 11:05:00 | 1866.48 | 1863.59 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:00:00 | 1911.20 | 1903.69 | 0.00 | ORB-long ORB[1891.20,1910.00] vol=1.8x ATR=4.33 |
| Stop hit — per-position SL triggered | 2025-05-07 11:45:00 | 1906.87 | 1904.89 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-05-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 10:50:00 | 1848.50 | 1855.91 | 0.00 | ORB-short ORB[1855.00,1871.10] vol=2.1x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-05-09 11:30:00 | 1852.64 | 1853.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:35:00 | 1284.25 | 2024-05-13 11:05:00 | 1290.26 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-05-16 09:35:00 | 1344.75 | 2024-05-16 09:45:00 | 1353.80 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-05-16 09:35:00 | 1344.75 | 2024-05-16 10:05:00 | 1344.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-21 09:55:00 | 1346.35 | 2024-05-21 10:20:00 | 1350.34 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-23 09:45:00 | 1356.75 | 2024-05-23 09:50:00 | 1361.53 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-05-23 09:45:00 | 1356.75 | 2024-05-23 11:05:00 | 1360.40 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2024-06-10 10:55:00 | 1432.50 | 2024-06-10 12:10:00 | 1428.26 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-13 10:25:00 | 1440.05 | 2024-06-13 10:50:00 | 1435.20 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-13 10:25:00 | 1440.05 | 2024-06-13 15:20:00 | 1429.55 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-06-18 10:55:00 | 1425.40 | 2024-06-18 11:25:00 | 1420.76 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-18 10:55:00 | 1425.40 | 2024-06-18 12:55:00 | 1425.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 10:30:00 | 1414.60 | 2024-06-19 10:45:00 | 1409.90 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-19 10:30:00 | 1414.60 | 2024-06-19 11:35:00 | 1414.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-20 11:00:00 | 1384.50 | 2024-06-20 13:55:00 | 1379.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-20 11:00:00 | 1384.50 | 2024-06-20 15:20:00 | 1380.25 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-06-25 11:10:00 | 1411.30 | 2024-06-25 11:15:00 | 1414.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1424.35 | 2024-06-26 11:20:00 | 1429.58 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1424.35 | 2024-06-26 11:40:00 | 1424.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 11:05:00 | 1435.50 | 2024-07-08 11:10:00 | 1432.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-10 09:30:00 | 1445.50 | 2024-07-10 09:35:00 | 1442.54 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-11 11:10:00 | 1428.00 | 2024-07-11 11:15:00 | 1431.05 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-12 11:00:00 | 1444.80 | 2024-07-12 11:05:00 | 1440.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-16 09:40:00 | 1465.85 | 2024-07-16 13:55:00 | 1460.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-01 11:15:00 | 1505.75 | 2024-08-01 12:25:00 | 1510.23 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-08-01 11:15:00 | 1505.75 | 2024-08-01 12:45:00 | 1505.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 10:55:00 | 1472.35 | 2024-08-12 11:20:00 | 1477.23 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-12 10:55:00 | 1472.35 | 2024-08-12 11:25:00 | 1472.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 09:40:00 | 1451.00 | 2024-08-21 09:50:00 | 1453.91 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-22 11:15:00 | 1495.10 | 2024-08-22 11:25:00 | 1499.87 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-22 11:15:00 | 1495.10 | 2024-08-22 11:30:00 | 1495.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 11:10:00 | 1604.85 | 2024-08-30 11:20:00 | 1599.37 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-03 09:55:00 | 1564.20 | 2024-09-03 10:05:00 | 1567.49 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-09 11:15:00 | 1539.70 | 2024-09-09 11:25:00 | 1536.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-17 11:15:00 | 1650.15 | 2024-09-17 11:35:00 | 1653.98 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2024-09-17 11:15:00 | 1650.15 | 2024-09-17 15:20:00 | 1662.00 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-09-19 09:45:00 | 1676.85 | 2024-09-19 09:50:00 | 1672.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1680.35 | 2024-09-20 10:40:00 | 1688.90 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1680.35 | 2024-09-20 13:40:00 | 1690.00 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-24 09:35:00 | 1761.30 | 2024-09-24 11:00:00 | 1756.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-26 09:55:00 | 1773.20 | 2024-09-26 10:55:00 | 1768.31 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-10 11:10:00 | 1671.30 | 2024-10-10 12:40:00 | 1675.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-17 10:05:00 | 1710.40 | 2024-10-17 10:10:00 | 1714.88 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-30 11:05:00 | 1645.65 | 2024-10-30 11:50:00 | 1640.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-31 10:50:00 | 1606.00 | 2024-10-31 10:55:00 | 1610.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-04 10:10:00 | 1586.55 | 2024-11-04 10:55:00 | 1576.96 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-04 10:10:00 | 1586.55 | 2024-11-04 12:00:00 | 1586.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:25:00 | 1576.90 | 2024-11-11 10:55:00 | 1572.94 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-12 10:35:00 | 1586.60 | 2024-11-12 11:00:00 | 1593.20 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-12 10:35:00 | 1586.60 | 2024-11-12 11:15:00 | 1586.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 11:15:00 | 1537.40 | 2024-11-22 12:00:00 | 1543.59 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-22 11:15:00 | 1537.40 | 2024-11-22 15:20:00 | 1570.55 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2024-11-26 11:00:00 | 1581.45 | 2024-11-26 11:50:00 | 1574.01 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-11-26 11:00:00 | 1581.45 | 2024-11-26 14:35:00 | 1579.25 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1568.30 | 2024-11-28 12:00:00 | 1559.66 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1568.30 | 2024-11-28 13:30:00 | 1568.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 09:55:00 | 1583.95 | 2024-12-10 10:05:00 | 1587.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-13 09:45:00 | 1621.80 | 2024-12-13 10:00:00 | 1629.09 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-13 09:45:00 | 1621.80 | 2024-12-13 15:20:00 | 1683.35 | TARGET_HIT | 0.50 | 3.80% |
| SELL | retest1 | 2024-12-18 11:05:00 | 1605.95 | 2024-12-18 11:30:00 | 1601.11 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-18 11:05:00 | 1605.95 | 2024-12-18 14:00:00 | 1605.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-23 11:15:00 | 1603.75 | 2024-12-23 11:55:00 | 1599.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-27 09:55:00 | 1627.80 | 2024-12-27 10:00:00 | 1623.09 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-01 10:40:00 | 1595.55 | 2025-01-01 10:55:00 | 1601.45 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-01 10:40:00 | 1595.55 | 2025-01-01 11:35:00 | 1595.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1600.30 | 2025-01-02 10:50:00 | 1606.22 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1600.30 | 2025-01-02 11:45:00 | 1601.55 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-01-06 11:10:00 | 1587.45 | 2025-01-06 11:35:00 | 1580.70 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-06 11:10:00 | 1587.45 | 2025-01-06 14:55:00 | 1587.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 11:05:00 | 1585.85 | 2025-01-07 11:15:00 | 1589.45 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-09 11:00:00 | 1604.90 | 2025-01-09 11:15:00 | 1601.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-16 11:05:00 | 1619.55 | 2025-01-16 11:45:00 | 1624.59 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-01-16 11:05:00 | 1619.55 | 2025-01-16 15:20:00 | 1630.50 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1629.60 | 2025-01-21 10:25:00 | 1623.14 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1629.60 | 2025-01-21 10:40:00 | 1629.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-22 11:00:00 | 1637.45 | 2025-01-22 11:05:00 | 1633.77 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-01-28 11:00:00 | 1612.00 | 2025-01-28 11:35:00 | 1608.80 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-02-05 10:45:00 | 1656.15 | 2025-02-05 11:10:00 | 1649.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-02-05 10:45:00 | 1656.15 | 2025-02-05 11:25:00 | 1656.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-17 10:15:00 | 1689.90 | 2025-02-17 10:20:00 | 1693.76 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-21 10:50:00 | 1642.35 | 2025-02-21 13:50:00 | 1636.87 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-02-21 10:50:00 | 1642.35 | 2025-02-21 15:20:00 | 1638.05 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-02-24 10:25:00 | 1612.60 | 2025-02-24 12:00:00 | 1605.23 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-24 10:25:00 | 1612.60 | 2025-02-24 15:20:00 | 1601.40 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-02-25 10:55:00 | 1629.95 | 2025-02-25 11:35:00 | 1633.86 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-02-25 10:55:00 | 1629.95 | 2025-02-25 15:20:00 | 1640.70 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2025-03-04 09:45:00 | 1569.95 | 2025-03-04 09:55:00 | 1574.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-05 10:55:00 | 1600.80 | 2025-03-05 11:15:00 | 1605.80 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-03-05 10:55:00 | 1600.80 | 2025-03-05 15:20:00 | 1618.95 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2025-03-26 09:30:00 | 1762.15 | 2025-03-26 09:35:00 | 1756.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-02 09:50:00 | 1751.55 | 2025-04-02 10:15:00 | 1758.83 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-04-02 09:50:00 | 1751.55 | 2025-04-02 10:35:00 | 1751.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-04 09:35:00 | 1758.40 | 2025-04-04 09:50:00 | 1754.41 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1845.70 | 2025-04-17 11:30:00 | 1851.47 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1845.70 | 2025-04-17 15:20:00 | 1881.20 | TARGET_HIT | 0.50 | 1.92% |
| BUY | retest1 | 2025-04-29 09:35:00 | 1842.80 | 2025-04-29 09:40:00 | 1838.59 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-05 10:40:00 | 1871.00 | 2025-05-05 11:05:00 | 1866.48 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-07 11:00:00 | 1911.20 | 2025-05-07 11:45:00 | 1906.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-09 10:50:00 | 1848.50 | 2025-05-09 11:30:00 | 1852.64 | STOP_HIT | 1.00 | -0.22% |
