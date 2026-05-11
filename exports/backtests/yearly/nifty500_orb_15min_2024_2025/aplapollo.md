# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 1950.00
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 10 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 55
- **Target hits / Stop hits / Partials:** 10 / 55 / 25
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 10.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 16 | 34.0% | 5 | 31 | 11 | 0.08% | 4.0% |
| BUY @ 2nd Alert (retest1) | 47 | 16 | 34.0% | 5 | 31 | 11 | 0.08% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 19 | 44.2% | 5 | 24 | 14 | 0.16% | 6.7% |
| SELL @ 2nd Alert (retest1) | 43 | 19 | 44.2% | 5 | 24 | 14 | 0.16% | 6.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 35 | 38.9% | 10 | 55 | 25 | 0.12% | 10.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:50:00 | 1606.30 | 1592.59 | 0.00 | ORB-long ORB[1582.15,1593.45] vol=2.3x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:55:00 | 1614.06 | 1601.35 | 0.00 | T1 1.5R @ 1614.06 |
| Target hit | 2024-05-16 13:20:00 | 1627.05 | 1628.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:30:00 | 1654.00 | 1644.83 | 0.00 | ORB-long ORB[1631.00,1650.95] vol=2.1x ATR=6.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:50:00 | 1664.31 | 1652.95 | 0.00 | T1 1.5R @ 1664.31 |
| Target hit | 2024-05-17 15:20:00 | 1700.10 | 1679.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:30:00 | 1719.00 | 1708.90 | 0.00 | ORB-long ORB[1690.75,1716.00] vol=1.9x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-05-24 11:10:00 | 1713.78 | 1718.88 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:40:00 | 1644.40 | 1636.34 | 0.00 | ORB-long ORB[1620.20,1643.95] vol=4.2x ATR=6.22 |
| Stop hit — per-position SL triggered | 2024-06-11 10:55:00 | 1638.18 | 1637.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:40:00 | 1556.35 | 1547.85 | 0.00 | ORB-long ORB[1533.25,1549.00] vol=1.9x ATR=5.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:20:00 | 1564.47 | 1550.59 | 0.00 | T1 1.5R @ 1564.47 |
| Stop hit — per-position SL triggered | 2024-06-14 11:55:00 | 1556.35 | 1552.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:20:00 | 1572.95 | 1564.03 | 0.00 | ORB-long ORB[1556.70,1568.20] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-06-18 10:25:00 | 1568.93 | 1564.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1578.90 | 1568.13 | 0.00 | ORB-long ORB[1563.65,1573.00] vol=2.0x ATR=4.46 |
| Stop hit — per-position SL triggered | 2024-06-20 10:20:00 | 1574.44 | 1568.83 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:05:00 | 1625.00 | 1631.20 | 0.00 | ORB-short ORB[1627.00,1647.20] vol=1.6x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:25:00 | 1615.71 | 1627.92 | 0.00 | T1 1.5R @ 1615.71 |
| Target hit | 2024-06-25 15:00:00 | 1622.75 | 1622.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2024-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:45:00 | 1593.20 | 1596.48 | 0.00 | ORB-short ORB[1594.65,1602.65] vol=1.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-06-27 10:55:00 | 1597.29 | 1595.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:55:00 | 1590.00 | 1597.70 | 0.00 | ORB-short ORB[1595.00,1609.65] vol=2.0x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:20:00 | 1582.79 | 1595.44 | 0.00 | T1 1.5R @ 1582.79 |
| Stop hit — per-position SL triggered | 2024-07-03 11:40:00 | 1590.00 | 1594.91 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 1549.55 | 1553.27 | 0.00 | ORB-short ORB[1551.95,1565.75] vol=8.6x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:30:00 | 1542.41 | 1551.97 | 0.00 | T1 1.5R @ 1542.41 |
| Target hit | 2024-07-09 14:00:00 | 1545.45 | 1543.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2024-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:45:00 | 1554.20 | 1567.65 | 0.00 | ORB-short ORB[1573.25,1584.00] vol=1.7x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 12:05:00 | 1547.18 | 1560.56 | 0.00 | T1 1.5R @ 1547.18 |
| Stop hit — per-position SL triggered | 2024-07-11 15:05:00 | 1554.20 | 1552.21 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:15:00 | 1475.65 | 1478.84 | 0.00 | ORB-short ORB[1482.35,1496.00] vol=2.1x ATR=6.27 |
| Stop hit — per-position SL triggered | 2024-07-23 10:50:00 | 1481.92 | 1478.36 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:30:00 | 1494.85 | 1475.42 | 0.00 | ORB-long ORB[1455.40,1465.00] vol=2.0x ATR=6.13 |
| Stop hit — per-position SL triggered | 2024-07-26 10:35:00 | 1488.72 | 1475.83 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:55:00 | 1495.50 | 1489.58 | 0.00 | ORB-long ORB[1478.45,1494.75] vol=1.8x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:05:00 | 1502.30 | 1490.86 | 0.00 | T1 1.5R @ 1502.30 |
| Stop hit — per-position SL triggered | 2024-07-29 11:20:00 | 1495.50 | 1491.42 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:40:00 | 1478.65 | 1488.82 | 0.00 | ORB-short ORB[1486.10,1506.80] vol=3.8x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:00:00 | 1471.93 | 1486.54 | 0.00 | T1 1.5R @ 1471.93 |
| Stop hit — per-position SL triggered | 2024-08-01 14:40:00 | 1478.65 | 1483.11 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 1408.50 | 1416.74 | 0.00 | ORB-short ORB[1410.05,1427.80] vol=1.5x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-08-08 11:05:00 | 1412.72 | 1416.29 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:45:00 | 1431.90 | 1419.48 | 0.00 | ORB-long ORB[1412.75,1424.85] vol=2.2x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-08-16 10:55:00 | 1426.64 | 1420.36 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 1350.50 | 1355.91 | 0.00 | ORB-short ORB[1353.00,1368.00] vol=1.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-08-22 11:05:00 | 1354.73 | 1352.15 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 1420.20 | 1424.04 | 0.00 | ORB-short ORB[1421.30,1435.00] vol=3.2x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 1423.73 | 1422.14 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:25:00 | 1455.00 | 1443.97 | 0.00 | ORB-long ORB[1430.30,1448.75] vol=2.7x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:40:00 | 1463.75 | 1448.73 | 0.00 | T1 1.5R @ 1463.75 |
| Target hit | 2024-08-28 15:20:00 | 1473.80 | 1463.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:15:00 | 1434.65 | 1438.56 | 0.00 | ORB-short ORB[1435.05,1448.80] vol=2.5x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:30:00 | 1428.63 | 1437.21 | 0.00 | T1 1.5R @ 1428.63 |
| Target hit | 2024-09-03 13:20:00 | 1433.10 | 1433.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 1470.00 | 1453.90 | 0.00 | ORB-long ORB[1448.25,1464.00] vol=3.9x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-09-05 11:45:00 | 1465.11 | 1459.76 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 1437.95 | 1419.16 | 0.00 | ORB-long ORB[1409.00,1422.70] vol=2.4x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:10:00 | 1444.25 | 1422.33 | 0.00 | T1 1.5R @ 1444.25 |
| Stop hit — per-position SL triggered | 2024-09-11 11:35:00 | 1437.95 | 1426.68 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:30:00 | 1450.85 | 1438.85 | 0.00 | ORB-long ORB[1423.95,1435.00] vol=1.6x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 12:20:00 | 1459.34 | 1447.81 | 0.00 | T1 1.5R @ 1459.34 |
| Target hit | 2024-09-13 15:00:00 | 1461.80 | 1465.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2024-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:00:00 | 1456.20 | 1460.61 | 0.00 | ORB-short ORB[1461.00,1478.75] vol=2.6x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 12:00:00 | 1446.38 | 1457.89 | 0.00 | T1 1.5R @ 1446.38 |
| Stop hit — per-position SL triggered | 2024-09-16 13:25:00 | 1456.20 | 1456.45 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:45:00 | 1437.60 | 1443.71 | 0.00 | ORB-short ORB[1440.00,1460.00] vol=2.1x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:05:00 | 1432.48 | 1442.67 | 0.00 | T1 1.5R @ 1432.48 |
| Stop hit — per-position SL triggered | 2024-09-18 14:00:00 | 1437.60 | 1438.15 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 1445.05 | 1436.75 | 0.00 | ORB-long ORB[1427.65,1442.55] vol=2.2x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-09-19 10:00:00 | 1440.74 | 1438.91 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:35:00 | 1552.00 | 1543.92 | 0.00 | ORB-long ORB[1521.90,1540.10] vol=2.5x ATR=6.13 |
| Stop hit — per-position SL triggered | 2024-09-30 09:40:00 | 1545.87 | 1544.30 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 11:00:00 | 1535.30 | 1542.56 | 0.00 | ORB-short ORB[1540.15,1561.65] vol=1.9x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-10-15 11:35:00 | 1538.81 | 1541.59 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:50:00 | 1553.60 | 1563.03 | 0.00 | ORB-short ORB[1567.50,1589.90] vol=3.3x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:50:00 | 1542.48 | 1554.47 | 0.00 | T1 1.5R @ 1542.48 |
| Target hit | 2024-10-21 15:20:00 | 1540.90 | 1544.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-10-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:45:00 | 1523.20 | 1514.94 | 0.00 | ORB-long ORB[1482.15,1499.65] vol=4.9x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:55:00 | 1532.60 | 1515.82 | 0.00 | T1 1.5R @ 1532.60 |
| Stop hit — per-position SL triggered | 2024-10-31 13:25:00 | 1523.20 | 1525.82 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 11:00:00 | 1511.35 | 1524.01 | 0.00 | ORB-short ORB[1524.10,1545.10] vol=7.3x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 12:15:00 | 1502.61 | 1520.59 | 0.00 | T1 1.5R @ 1502.61 |
| Stop hit — per-position SL triggered | 2024-11-04 14:25:00 | 1511.35 | 1515.32 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-11-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:40:00 | 1532.55 | 1540.15 | 0.00 | ORB-short ORB[1534.00,1556.70] vol=7.2x ATR=6.08 |
| Stop hit — per-position SL triggered | 2024-11-07 10:45:00 | 1538.63 | 1540.05 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:00:00 | 1460.95 | 1459.35 | 0.00 | ORB-long ORB[1435.05,1454.35] vol=9.3x ATR=6.80 |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 1454.15 | 1459.11 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:15:00 | 1488.90 | 1476.46 | 0.00 | ORB-long ORB[1466.55,1484.55] vol=4.6x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-11-27 11:20:00 | 1483.64 | 1476.63 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:40:00 | 1582.10 | 1572.18 | 0.00 | ORB-long ORB[1559.05,1580.50] vol=1.5x ATR=5.43 |
| Stop hit — per-position SL triggered | 2024-12-04 11:05:00 | 1576.67 | 1573.97 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:10:00 | 1556.55 | 1569.80 | 0.00 | ORB-short ORB[1570.00,1587.95] vol=2.0x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-12-05 10:20:00 | 1561.28 | 1569.03 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:50:00 | 1589.80 | 1579.33 | 0.00 | ORB-long ORB[1563.10,1582.65] vol=1.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-12-09 10:55:00 | 1584.52 | 1586.36 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:15:00 | 1638.95 | 1630.57 | 0.00 | ORB-long ORB[1613.50,1625.55] vol=2.2x ATR=6.71 |
| Stop hit — per-position SL triggered | 2024-12-10 10:25:00 | 1632.24 | 1631.26 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:50:00 | 1614.85 | 1604.33 | 0.00 | ORB-long ORB[1597.05,1613.45] vol=2.3x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-12-12 11:05:00 | 1610.61 | 1605.53 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 1581.00 | 1585.73 | 0.00 | ORB-short ORB[1591.90,1611.90] vol=9.8x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-12-13 11:55:00 | 1585.36 | 1584.39 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1610.00 | 1605.90 | 0.00 | ORB-long ORB[1590.05,1609.80] vol=4.6x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-12-17 09:40:00 | 1605.01 | 1605.04 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:10:00 | 1569.00 | 1576.54 | 0.00 | ORB-short ORB[1575.05,1592.00] vol=1.9x ATR=4.84 |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 1573.84 | 1576.39 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 10:40:00 | 1528.15 | 1532.86 | 0.00 | ORB-short ORB[1530.00,1541.40] vol=1.7x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-12-24 10:50:00 | 1531.95 | 1531.92 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:10:00 | 1493.50 | 1503.22 | 0.00 | ORB-short ORB[1505.75,1521.65] vol=1.6x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:50:00 | 1486.79 | 1499.43 | 0.00 | T1 1.5R @ 1486.79 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 1493.50 | 1495.27 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 1514.75 | 1511.01 | 0.00 | ORB-long ORB[1500.15,1513.50] vol=2.1x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:15:00 | 1521.89 | 1512.77 | 0.00 | T1 1.5R @ 1521.89 |
| Target hit | 2024-12-27 12:45:00 | 1517.10 | 1517.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2024-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:25:00 | 1528.95 | 1520.18 | 0.00 | ORB-long ORB[1502.50,1524.45] vol=2.4x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-12-30 10:35:00 | 1523.73 | 1521.40 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 1614.10 | 1604.55 | 0.00 | ORB-long ORB[1590.00,1612.70] vol=2.0x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-01-03 10:25:00 | 1608.12 | 1606.17 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1602.35 | 1615.09 | 0.00 | ORB-short ORB[1612.55,1635.30] vol=2.4x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 1593.58 | 1611.64 | 0.00 | T1 1.5R @ 1593.58 |
| Stop hit — per-position SL triggered | 2025-01-06 13:35:00 | 1602.35 | 1604.12 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:35:00 | 1603.30 | 1595.25 | 0.00 | ORB-long ORB[1577.55,1600.55] vol=1.7x ATR=7.38 |
| Stop hit — per-position SL triggered | 2025-01-07 09:40:00 | 1595.92 | 1596.33 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:40:00 | 1542.50 | 1534.11 | 0.00 | ORB-long ORB[1524.10,1539.00] vol=3.0x ATR=4.97 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 1537.53 | 1534.34 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:55:00 | 1470.70 | 1482.33 | 0.00 | ORB-short ORB[1480.00,1502.00] vol=2.2x ATR=6.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:00:00 | 1461.08 | 1481.37 | 0.00 | T1 1.5R @ 1461.08 |
| Target hit | 2025-01-13 15:20:00 | 1425.15 | 1438.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-01-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:25:00 | 1499.05 | 1517.50 | 0.00 | ORB-short ORB[1524.45,1541.35] vol=1.8x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-01-27 10:35:00 | 1505.42 | 1515.99 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:35:00 | 1517.25 | 1507.39 | 0.00 | ORB-long ORB[1485.70,1499.95] vol=1.7x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:45:00 | 1527.38 | 1513.31 | 0.00 | T1 1.5R @ 1527.38 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 1517.25 | 1515.67 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:35:00 | 1519.95 | 1525.59 | 0.00 | ORB-short ORB[1522.00,1535.50] vol=1.6x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:50:00 | 1513.04 | 1523.42 | 0.00 | T1 1.5R @ 1513.04 |
| Stop hit — per-position SL triggered | 2025-01-30 10:55:00 | 1519.95 | 1522.94 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:20:00 | 1480.40 | 1474.18 | 0.00 | ORB-long ORB[1457.15,1473.10] vol=1.5x ATR=7.05 |
| Stop hit — per-position SL triggered | 2025-03-07 10:40:00 | 1473.35 | 1474.57 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:05:00 | 1405.80 | 1400.95 | 0.00 | ORB-long ORB[1379.50,1400.00] vol=6.8x ATR=6.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 10:45:00 | 1415.13 | 1403.26 | 0.00 | T1 1.5R @ 1415.13 |
| Stop hit — per-position SL triggered | 2025-03-17 11:10:00 | 1405.80 | 1407.28 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 1540.55 | 1531.65 | 0.00 | ORB-long ORB[1509.50,1529.00] vol=2.0x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-03-24 13:25:00 | 1536.78 | 1535.57 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-04-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:20:00 | 1504.20 | 1520.15 | 0.00 | ORB-short ORB[1515.60,1532.95] vol=3.7x ATR=7.02 |
| Stop hit — per-position SL triggered | 2025-04-01 12:30:00 | 1511.22 | 1513.09 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:30:00 | 1567.80 | 1560.67 | 0.00 | ORB-long ORB[1548.20,1566.80] vol=2.4x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-04-03 15:20:00 | 1566.75 | 1564.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:45:00 | 1576.50 | 1568.94 | 0.00 | ORB-long ORB[1563.10,1574.30] vol=1.9x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-04-17 10:10:00 | 1571.67 | 1571.31 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 10:50:00 | 1622.30 | 1630.28 | 0.00 | ORB-short ORB[1625.60,1640.00] vol=3.0x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-04-24 11:10:00 | 1627.21 | 1629.94 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:05:00 | 1627.80 | 1616.09 | 0.00 | ORB-long ORB[1603.60,1623.30] vol=5.4x ATR=5.52 |
| Stop hit — per-position SL triggered | 2025-04-28 11:40:00 | 1622.28 | 1618.42 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:50:00 | 1601.50 | 1610.57 | 0.00 | ORB-short ORB[1603.80,1626.90] vol=3.3x ATR=7.25 |
| Stop hit — per-position SL triggered | 2025-05-06 10:00:00 | 1608.75 | 1610.98 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:50:00 | 1606.30 | 2024-05-16 09:55:00 | 1614.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-16 09:50:00 | 1606.30 | 2024-05-16 13:20:00 | 1627.05 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2024-05-17 09:30:00 | 1654.00 | 2024-05-17 09:50:00 | 1664.31 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-05-17 09:30:00 | 1654.00 | 2024-05-17 15:20:00 | 1700.10 | TARGET_HIT | 0.50 | 2.79% |
| BUY | retest1 | 2024-05-24 10:30:00 | 1719.00 | 2024-05-24 11:10:00 | 1713.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-11 10:40:00 | 1644.40 | 2024-06-11 10:55:00 | 1638.18 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-14 10:40:00 | 1556.35 | 2024-06-14 11:20:00 | 1564.47 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-14 10:40:00 | 1556.35 | 2024-06-14 11:55:00 | 1556.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 10:20:00 | 1572.95 | 2024-06-18 10:25:00 | 1568.93 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-20 10:15:00 | 1578.90 | 2024-06-20 10:20:00 | 1574.44 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-25 11:05:00 | 1625.00 | 2024-06-25 11:25:00 | 1615.71 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-25 11:05:00 | 1625.00 | 2024-06-25 15:00:00 | 1622.75 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-06-27 10:45:00 | 1593.20 | 2024-06-27 10:55:00 | 1597.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-03 10:55:00 | 1590.00 | 2024-07-03 11:20:00 | 1582.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-03 10:55:00 | 1590.00 | 2024-07-03 11:40:00 | 1590.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 10:15:00 | 1549.55 | 2024-07-09 10:30:00 | 1542.41 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-09 10:15:00 | 1549.55 | 2024-07-09 14:00:00 | 1545.45 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-07-11 10:45:00 | 1554.20 | 2024-07-11 12:05:00 | 1547.18 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-11 10:45:00 | 1554.20 | 2024-07-11 15:05:00 | 1554.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 10:15:00 | 1475.65 | 2024-07-23 10:50:00 | 1481.92 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-26 10:30:00 | 1494.85 | 2024-07-26 10:35:00 | 1488.72 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-29 10:55:00 | 1495.50 | 2024-07-29 11:05:00 | 1502.30 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-29 10:55:00 | 1495.50 | 2024-07-29 11:20:00 | 1495.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-01 10:40:00 | 1478.65 | 2024-08-01 12:00:00 | 1471.93 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-08-01 10:40:00 | 1478.65 | 2024-08-01 14:40:00 | 1478.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:55:00 | 1408.50 | 2024-08-08 11:05:00 | 1412.72 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-16 10:45:00 | 1431.90 | 2024-08-16 10:55:00 | 1426.64 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-22 09:30:00 | 1350.50 | 2024-08-22 11:05:00 | 1354.73 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-27 10:50:00 | 1420.20 | 2024-08-27 13:15:00 | 1423.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-28 10:25:00 | 1455.00 | 2024-08-28 10:40:00 | 1463.75 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-28 10:25:00 | 1455.00 | 2024-08-28 15:20:00 | 1473.80 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2024-09-03 10:15:00 | 1434.65 | 2024-09-03 10:30:00 | 1428.63 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-03 10:15:00 | 1434.65 | 2024-09-03 13:20:00 | 1433.10 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-09-05 11:00:00 | 1470.00 | 2024-09-05 11:45:00 | 1465.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-11 10:55:00 | 1437.95 | 2024-09-11 11:10:00 | 1444.25 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-11 10:55:00 | 1437.95 | 2024-09-11 11:35:00 | 1437.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:30:00 | 1450.85 | 2024-09-13 12:20:00 | 1459.34 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-13 10:30:00 | 1450.85 | 2024-09-13 15:00:00 | 1461.80 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-09-16 10:00:00 | 1456.20 | 2024-09-16 12:00:00 | 1446.38 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-09-16 10:00:00 | 1456.20 | 2024-09-16 13:25:00 | 1456.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:45:00 | 1437.60 | 2024-09-18 11:05:00 | 1432.48 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-18 10:45:00 | 1437.60 | 2024-09-18 14:00:00 | 1437.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:50:00 | 1445.05 | 2024-09-19 10:00:00 | 1440.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-30 09:35:00 | 1552.00 | 2024-09-30 09:40:00 | 1545.87 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-15 11:00:00 | 1535.30 | 2024-10-15 11:35:00 | 1538.81 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-21 09:50:00 | 1553.60 | 2024-10-21 11:50:00 | 1542.48 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-10-21 09:50:00 | 1553.60 | 2024-10-21 15:20:00 | 1540.90 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-10-31 10:45:00 | 1523.20 | 2024-10-31 10:55:00 | 1532.60 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-31 10:45:00 | 1523.20 | 2024-10-31 13:25:00 | 1523.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 11:00:00 | 1511.35 | 2024-11-04 12:15:00 | 1502.61 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-04 11:00:00 | 1511.35 | 2024-11-04 14:25:00 | 1511.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:40:00 | 1532.55 | 2024-11-07 10:45:00 | 1538.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-25 10:00:00 | 1460.95 | 2024-11-25 10:15:00 | 1454.15 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-11-27 11:15:00 | 1488.90 | 2024-11-27 11:20:00 | 1483.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-04 10:40:00 | 1582.10 | 2024-12-04 11:05:00 | 1576.67 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-05 10:10:00 | 1556.55 | 2024-12-05 10:20:00 | 1561.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-09 09:50:00 | 1589.80 | 2024-12-09 10:55:00 | 1584.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-10 10:15:00 | 1638.95 | 2024-12-10 10:25:00 | 1632.24 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-12 10:50:00 | 1614.85 | 2024-12-12 11:05:00 | 1610.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-13 10:30:00 | 1581.00 | 2024-12-13 11:55:00 | 1585.36 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-17 09:30:00 | 1610.00 | 2024-12-17 09:40:00 | 1605.01 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-18 10:10:00 | 1569.00 | 2024-12-18 10:15:00 | 1573.84 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-24 10:40:00 | 1528.15 | 2024-12-24 10:50:00 | 1531.95 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-26 10:10:00 | 1493.50 | 2024-12-26 10:50:00 | 1486.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-26 10:10:00 | 1493.50 | 2024-12-26 11:10:00 | 1493.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 11:05:00 | 1514.75 | 2024-12-27 11:15:00 | 1521.89 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-27 11:05:00 | 1514.75 | 2024-12-27 12:45:00 | 1517.10 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2024-12-30 10:25:00 | 1528.95 | 2024-12-30 10:35:00 | 1523.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-03 10:00:00 | 1614.10 | 2025-01-03 10:25:00 | 1608.12 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1602.35 | 2025-01-06 11:10:00 | 1593.58 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1602.35 | 2025-01-06 13:35:00 | 1602.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-07 09:35:00 | 1603.30 | 2025-01-07 09:40:00 | 1595.92 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-09 10:40:00 | 1542.50 | 2025-01-09 10:45:00 | 1537.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-13 10:55:00 | 1470.70 | 2025-01-13 11:00:00 | 1461.08 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-01-13 10:55:00 | 1470.70 | 2025-01-13 15:20:00 | 1425.15 | TARGET_HIT | 0.50 | 3.10% |
| SELL | retest1 | 2025-01-27 10:25:00 | 1499.05 | 2025-01-27 10:35:00 | 1505.42 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-29 09:35:00 | 1517.25 | 2025-01-29 10:45:00 | 1527.38 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-01-29 09:35:00 | 1517.25 | 2025-01-29 11:20:00 | 1517.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 10:35:00 | 1519.95 | 2025-01-30 10:50:00 | 1513.04 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-30 10:35:00 | 1519.95 | 2025-01-30 10:55:00 | 1519.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 10:20:00 | 1480.40 | 2025-03-07 10:40:00 | 1473.35 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-17 10:05:00 | 1405.80 | 2025-03-17 10:45:00 | 1415.13 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-03-17 10:05:00 | 1405.80 | 2025-03-17 11:10:00 | 1405.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 11:10:00 | 1540.55 | 2025-03-24 13:25:00 | 1536.78 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-04-01 10:20:00 | 1504.20 | 2025-04-01 12:30:00 | 1511.22 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-03 10:30:00 | 1567.80 | 2025-04-03 15:20:00 | 1566.75 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2025-04-17 09:45:00 | 1576.50 | 2025-04-17 10:10:00 | 1571.67 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-24 10:50:00 | 1622.30 | 2025-04-24 11:10:00 | 1627.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-28 11:05:00 | 1627.80 | 2025-04-28 11:40:00 | 1622.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-05-06 09:50:00 | 1601.50 | 2025-05-06 10:00:00 | 1608.75 | STOP_HIT | 1.00 | -0.45% |
