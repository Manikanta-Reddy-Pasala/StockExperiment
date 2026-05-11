# TECHM (TECHM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 1460.90
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
| ENTRY1 | 66 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 13 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 53
- **Target hits / Stop hits / Partials:** 13 / 53 / 25
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 10.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 22 | 40.0% | 7 | 33 | 15 | 0.12% | 6.4% |
| BUY @ 2nd Alert (retest1) | 55 | 22 | 40.0% | 7 | 33 | 15 | 0.12% | 6.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 16 | 44.4% | 6 | 20 | 10 | 0.11% | 3.9% |
| SELL @ 2nd Alert (retest1) | 36 | 16 | 44.4% | 6 | 20 | 10 | 0.11% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 91 | 38 | 41.8% | 13 | 53 | 25 | 0.11% | 10.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:00:00 | 1254.95 | 1261.75 | 0.00 | ORB-short ORB[1260.20,1270.00] vol=2.0x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-05-14 10:45:00 | 1258.63 | 1259.94 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 1270.55 | 1277.38 | 0.00 | ORB-short ORB[1275.00,1282.80] vol=1.6x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-05-15 11:00:00 | 1274.29 | 1274.24 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:00:00 | 1339.30 | 1331.85 | 0.00 | ORB-long ORB[1327.00,1336.00] vol=2.1x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-05-23 13:10:00 | 1335.75 | 1335.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:10:00 | 1341.55 | 1337.69 | 0.00 | ORB-long ORB[1329.05,1339.00] vol=4.9x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-05-24 11:30:00 | 1338.54 | 1337.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 11:05:00 | 1332.25 | 1327.97 | 0.00 | ORB-long ORB[1318.25,1327.20] vol=2.2x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 11:20:00 | 1337.19 | 1328.79 | 0.00 | T1 1.5R @ 1337.19 |
| Stop hit — per-position SL triggered | 2024-05-27 11:35:00 | 1332.25 | 1328.96 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 1320.10 | 1323.36 | 0.00 | ORB-short ORB[1321.75,1330.35] vol=1.8x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-05-28 09:40:00 | 1323.04 | 1322.91 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 11:05:00 | 1299.05 | 1303.52 | 0.00 | ORB-short ORB[1300.00,1314.00] vol=4.2x ATR=2.95 |
| Stop hit — per-position SL triggered | 2024-05-29 11:25:00 | 1302.00 | 1303.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 11:15:00 | 1355.10 | 1350.54 | 0.00 | ORB-long ORB[1341.15,1354.00] vol=2.3x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-06-11 11:25:00 | 1352.19 | 1350.64 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:55:00 | 1395.60 | 1388.41 | 0.00 | ORB-long ORB[1375.35,1392.65] vol=1.7x ATR=3.47 |
| Stop hit — per-position SL triggered | 2024-06-13 11:20:00 | 1392.13 | 1389.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 1407.25 | 1403.29 | 0.00 | ORB-long ORB[1394.75,1406.30] vol=2.1x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-06-25 11:05:00 | 1403.66 | 1403.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 1442.00 | 1436.64 | 0.00 | ORB-long ORB[1424.15,1440.60] vol=1.7x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:55:00 | 1448.22 | 1440.93 | 0.00 | T1 1.5R @ 1448.22 |
| Target hit | 2024-07-01 15:20:00 | 1474.50 | 1464.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 1452.75 | 1462.80 | 0.00 | ORB-short ORB[1465.50,1472.80] vol=1.5x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-07-10 10:25:00 | 1455.77 | 1462.08 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 1449.70 | 1459.76 | 0.00 | ORB-short ORB[1456.00,1476.00] vol=1.7x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 12:00:00 | 1443.87 | 1456.27 | 0.00 | T1 1.5R @ 1443.87 |
| Stop hit — per-position SL triggered | 2024-07-11 12:10:00 | 1449.70 | 1455.27 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:50:00 | 1511.50 | 1501.70 | 0.00 | ORB-long ORB[1489.10,1507.00] vol=1.6x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 09:55:00 | 1517.83 | 1504.24 | 0.00 | T1 1.5R @ 1517.83 |
| Stop hit — per-position SL triggered | 2024-07-16 10:10:00 | 1511.50 | 1506.26 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:40:00 | 1528.15 | 1517.11 | 0.00 | ORB-long ORB[1505.00,1525.00] vol=1.7x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-07-18 11:10:00 | 1523.38 | 1519.86 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:50:00 | 1487.55 | 1489.43 | 0.00 | ORB-short ORB[1488.00,1497.55] vol=1.6x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-07-23 11:05:00 | 1491.24 | 1488.70 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:05:00 | 1556.20 | 1545.28 | 0.00 | ORB-long ORB[1538.00,1548.95] vol=1.7x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-07-31 11:50:00 | 1551.31 | 1552.07 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:45:00 | 1521.70 | 1514.90 | 0.00 | ORB-long ORB[1504.70,1515.95] vol=1.8x ATR=3.77 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 1517.93 | 1515.80 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:30:00 | 1525.40 | 1518.23 | 0.00 | ORB-long ORB[1505.00,1521.55] vol=1.6x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-08-14 11:40:00 | 1520.88 | 1521.04 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:30:00 | 1593.00 | 1588.43 | 0.00 | ORB-long ORB[1575.85,1590.00] vol=1.8x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 14:05:00 | 1599.55 | 1592.31 | 0.00 | T1 1.5R @ 1599.55 |
| Stop hit — per-position SL triggered | 2024-08-19 15:05:00 | 1593.00 | 1593.14 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:25:00 | 1610.95 | 1602.78 | 0.00 | ORB-long ORB[1595.25,1604.70] vol=1.6x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 1607.55 | 1605.58 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1595.00 | 1602.42 | 0.00 | ORB-short ORB[1602.00,1616.95] vol=1.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-08-23 09:50:00 | 1599.28 | 1602.14 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 11:05:00 | 1630.00 | 1637.40 | 0.00 | ORB-short ORB[1634.95,1645.00] vol=3.3x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 14:45:00 | 1624.19 | 1633.05 | 0.00 | T1 1.5R @ 1624.19 |
| Target hit | 2024-08-27 15:20:00 | 1625.00 | 1631.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:35:00 | 1645.35 | 1630.43 | 0.00 | ORB-long ORB[1619.20,1632.40] vol=3.8x ATR=4.62 |
| Stop hit — per-position SL triggered | 2024-08-28 10:45:00 | 1640.73 | 1632.95 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 11:10:00 | 1658.25 | 1652.64 | 0.00 | ORB-long ORB[1635.55,1650.50] vol=5.0x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-09-02 11:25:00 | 1654.12 | 1652.76 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:00:00 | 1649.35 | 1643.72 | 0.00 | ORB-long ORB[1632.10,1648.95] vol=3.1x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 11:55:00 | 1654.91 | 1646.34 | 0.00 | T1 1.5R @ 1654.91 |
| Stop hit — per-position SL triggered | 2024-09-03 14:30:00 | 1649.35 | 1649.60 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 1629.90 | 1635.16 | 0.00 | ORB-short ORB[1639.45,1650.00] vol=2.5x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-09-06 11:15:00 | 1634.01 | 1634.06 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 1616.65 | 1609.25 | 0.00 | ORB-long ORB[1602.70,1614.75] vol=3.9x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:45:00 | 1622.58 | 1612.96 | 0.00 | T1 1.5R @ 1622.58 |
| Target hit | 2024-09-11 13:45:00 | 1617.80 | 1617.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2024-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 11:10:00 | 1612.25 | 1619.08 | 0.00 | ORB-short ORB[1612.30,1628.45] vol=1.7x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:25:00 | 1606.05 | 1617.33 | 0.00 | T1 1.5R @ 1606.05 |
| Stop hit — per-position SL triggered | 2024-09-12 12:25:00 | 1612.25 | 1614.97 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:50:00 | 1651.55 | 1645.16 | 0.00 | ORB-long ORB[1639.25,1650.80] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-09-13 10:55:00 | 1647.31 | 1645.38 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:45:00 | 1666.00 | 1662.34 | 0.00 | ORB-long ORB[1651.00,1665.00] vol=2.2x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-09-16 11:05:00 | 1662.48 | 1662.54 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1654.40 | 1649.85 | 0.00 | ORB-long ORB[1641.10,1650.00] vol=1.7x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:25:00 | 1659.80 | 1651.32 | 0.00 | T1 1.5R @ 1659.80 |
| Target hit | 2024-09-17 13:05:00 | 1659.20 | 1659.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2024-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:05:00 | 1610.05 | 1618.80 | 0.00 | ORB-short ORB[1621.15,1631.00] vol=1.8x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:30:00 | 1603.68 | 1612.76 | 0.00 | T1 1.5R @ 1603.68 |
| Target hit | 2024-09-23 15:00:00 | 1605.50 | 1604.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2024-09-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:25:00 | 1610.70 | 1619.60 | 0.00 | ORB-short ORB[1623.10,1643.10] vol=1.5x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:25:00 | 1602.07 | 1613.70 | 0.00 | T1 1.5R @ 1602.07 |
| Target hit | 2024-09-25 15:05:00 | 1604.75 | 1603.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1606.25 | 1614.75 | 0.00 | ORB-short ORB[1618.30,1632.05] vol=1.6x ATR=5.25 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1611.50 | 1614.30 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 1676.45 | 1666.66 | 0.00 | ORB-long ORB[1648.00,1668.75] vol=3.8x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:30:00 | 1683.00 | 1670.30 | 0.00 | T1 1.5R @ 1683.00 |
| Target hit | 2024-10-14 15:20:00 | 1693.00 | 1683.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 1685.70 | 1670.88 | 0.00 | ORB-long ORB[1656.00,1672.00] vol=2.8x ATR=6.26 |
| Stop hit — per-position SL triggered | 2024-10-17 10:00:00 | 1679.44 | 1672.06 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-18 10:25:00 | 1657.20 | 1673.40 | 0.00 | ORB-short ORB[1676.60,1696.95] vol=1.8x ATR=8.47 |
| Stop hit — per-position SL triggered | 2024-10-18 10:30:00 | 1665.67 | 1673.05 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 11:00:00 | 1728.05 | 1735.03 | 0.00 | ORB-short ORB[1730.00,1742.40] vol=1.7x ATR=5.02 |
| Stop hit — per-position SL triggered | 2024-10-25 11:10:00 | 1733.07 | 1734.81 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 1663.30 | 1650.38 | 0.00 | ORB-long ORB[1635.00,1654.00] vol=1.7x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:55:00 | 1671.93 | 1659.01 | 0.00 | T1 1.5R @ 1671.93 |
| Target hit | 2024-11-06 15:20:00 | 1697.00 | 1681.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-11-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:35:00 | 1672.65 | 1664.55 | 0.00 | ORB-long ORB[1650.05,1668.80] vol=1.5x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:45:00 | 1678.66 | 1668.05 | 0.00 | T1 1.5R @ 1678.66 |
| Target hit | 2024-11-08 15:20:00 | 1680.15 | 1679.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-11-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:05:00 | 1713.75 | 1694.56 | 0.00 | ORB-long ORB[1666.05,1683.50] vol=2.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-11-11 11:10:00 | 1709.54 | 1696.77 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:40:00 | 1705.70 | 1697.02 | 0.00 | ORB-long ORB[1664.10,1684.95] vol=2.0x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-11-19 10:45:00 | 1701.70 | 1697.45 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:55:00 | 1733.50 | 1719.31 | 0.00 | ORB-long ORB[1702.25,1716.00] vol=1.6x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:30:00 | 1740.87 | 1725.86 | 0.00 | T1 1.5R @ 1740.87 |
| Stop hit — per-position SL triggered | 2024-11-22 10:40:00 | 1733.50 | 1726.79 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 10:55:00 | 1750.00 | 1753.89 | 0.00 | ORB-short ORB[1752.00,1767.80] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-11-25 11:00:00 | 1753.86 | 1753.82 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 1719.25 | 1715.16 | 0.00 | ORB-long ORB[1700.50,1714.60] vol=1.7x ATR=5.72 |
| Stop hit — per-position SL triggered | 2024-11-29 13:40:00 | 1713.53 | 1716.94 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 1774.00 | 1764.07 | 0.00 | ORB-long ORB[1740.60,1763.55] vol=2.0x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-12-04 09:35:00 | 1769.71 | 1765.70 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 1766.90 | 1776.63 | 0.00 | ORB-short ORB[1777.40,1789.80] vol=1.5x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-12-13 11:25:00 | 1771.75 | 1775.88 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:10:00 | 1716.95 | 1699.65 | 0.00 | ORB-long ORB[1685.85,1707.65] vol=1.8x ATR=4.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:30:00 | 1724.34 | 1702.09 | 0.00 | T1 1.5R @ 1724.34 |
| Stop hit — per-position SL triggered | 2024-12-23 12:00:00 | 1716.95 | 1704.72 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:15:00 | 1697.15 | 1699.64 | 0.00 | ORB-short ORB[1701.25,1714.65] vol=2.4x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-12-26 10:30:00 | 1701.95 | 1699.55 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1720.45 | 1712.79 | 0.00 | ORB-long ORB[1696.00,1712.55] vol=5.6x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-01-02 11:35:00 | 1716.30 | 1713.77 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:30:00 | 1697.10 | 1702.83 | 0.00 | ORB-short ORB[1712.20,1729.85] vol=3.0x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:45:00 | 1689.95 | 1700.31 | 0.00 | T1 1.5R @ 1689.95 |
| Stop hit — per-position SL triggered | 2025-01-03 11:20:00 | 1697.10 | 1696.52 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 1685.70 | 1699.17 | 0.00 | ORB-short ORB[1689.30,1707.50] vol=2.8x ATR=4.35 |
| Stop hit — per-position SL triggered | 2025-01-06 11:25:00 | 1690.05 | 1697.74 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:45:00 | 1666.00 | 1651.95 | 0.00 | ORB-long ORB[1635.25,1653.90] vol=1.6x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:00:00 | 1673.21 | 1658.37 | 0.00 | T1 1.5R @ 1673.21 |
| Stop hit — per-position SL triggered | 2025-01-22 13:30:00 | 1666.00 | 1665.60 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:55:00 | 1679.60 | 1671.97 | 0.00 | ORB-long ORB[1654.05,1670.45] vol=2.6x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-01-29 11:10:00 | 1673.90 | 1672.18 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:10:00 | 1699.30 | 1685.37 | 0.00 | ORB-long ORB[1671.00,1690.90] vol=1.6x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-01-30 11:00:00 | 1693.80 | 1689.57 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:10:00 | 1649.40 | 1658.33 | 0.00 | ORB-short ORB[1656.05,1676.95] vol=1.8x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-02-04 11:25:00 | 1653.79 | 1657.28 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 10:50:00 | 1672.55 | 1684.33 | 0.00 | ORB-short ORB[1679.60,1693.65] vol=1.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 12:30:00 | 1666.18 | 1678.63 | 0.00 | T1 1.5R @ 1666.18 |
| Target hit | 2025-02-20 15:20:00 | 1657.60 | 1668.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 1524.55 | 1532.40 | 0.00 | ORB-short ORB[1530.30,1550.80] vol=3.7x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:20:00 | 1518.71 | 1531.56 | 0.00 | T1 1.5R @ 1518.71 |
| Stop hit — per-position SL triggered | 2025-03-06 11:50:00 | 1524.55 | 1529.76 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:10:00 | 1448.80 | 1446.32 | 0.00 | ORB-long ORB[1435.10,1448.45] vol=4.3x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 1445.30 | 1447.19 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:50:00 | 1482.50 | 1469.18 | 0.00 | ORB-long ORB[1454.10,1474.00] vol=2.6x ATR=5.43 |
| Stop hit — per-position SL triggered | 2025-03-25 10:00:00 | 1477.07 | 1470.83 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-03-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:20:00 | 1439.40 | 1446.08 | 0.00 | ORB-short ORB[1446.70,1464.00] vol=1.6x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:10:00 | 1432.11 | 1443.64 | 0.00 | T1 1.5R @ 1432.11 |
| Target hit | 2025-03-26 15:20:00 | 1416.45 | 1427.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1424.00 | 1412.12 | 0.00 | ORB-long ORB[1397.15,1417.40] vol=1.6x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 1418.97 | 1413.42 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-04-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 11:00:00 | 1380.00 | 1387.06 | 0.00 | ORB-short ORB[1390.00,1400.00] vol=2.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:15:00 | 1373.75 | 1385.70 | 0.00 | T1 1.5R @ 1373.75 |
| Target hit | 2025-04-03 15:20:00 | 1371.25 | 1377.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1384.80 | 1373.80 | 0.00 | ORB-long ORB[1361.20,1379.30] vol=1.6x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:45:00 | 1393.64 | 1378.18 | 0.00 | T1 1.5R @ 1393.64 |
| Stop hit — per-position SL triggered | 2025-04-22 09:55:00 | 1384.80 | 1379.71 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:40:00 | 1481.60 | 1472.22 | 0.00 | ORB-long ORB[1465.00,1475.90] vol=1.7x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:55:00 | 1487.72 | 1474.21 | 0.00 | T1 1.5R @ 1487.72 |
| Target hit | 2025-04-29 15:20:00 | 1494.90 | 1489.87 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:00:00 | 1254.95 | 2024-05-14 10:45:00 | 1258.63 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-15 10:05:00 | 1270.55 | 2024-05-15 11:00:00 | 1274.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-23 11:00:00 | 1339.30 | 2024-05-23 13:10:00 | 1335.75 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-24 11:10:00 | 1341.55 | 2024-05-24 11:30:00 | 1338.54 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-27 11:05:00 | 1332.25 | 2024-05-27 11:20:00 | 1337.19 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-27 11:05:00 | 1332.25 | 2024-05-27 11:35:00 | 1332.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 09:30:00 | 1320.10 | 2024-05-28 09:40:00 | 1323.04 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-29 11:05:00 | 1299.05 | 2024-05-29 11:25:00 | 1302.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-11 11:15:00 | 1355.10 | 2024-06-11 11:25:00 | 1352.19 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-13 10:55:00 | 1395.60 | 2024-06-13 11:20:00 | 1392.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-25 10:45:00 | 1407.25 | 2024-06-25 11:05:00 | 1403.66 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-01 09:30:00 | 1442.00 | 2024-07-01 09:55:00 | 1448.22 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-01 09:30:00 | 1442.00 | 2024-07-01 15:20:00 | 1474.50 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2024-07-10 10:20:00 | 1452.75 | 2024-07-10 10:25:00 | 1455.77 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-11 11:05:00 | 1449.70 | 2024-07-11 12:00:00 | 1443.87 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-11 11:05:00 | 1449.70 | 2024-07-11 12:10:00 | 1449.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:50:00 | 1511.50 | 2024-07-16 09:55:00 | 1517.83 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-16 09:50:00 | 1511.50 | 2024-07-16 10:10:00 | 1511.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-18 10:40:00 | 1528.15 | 2024-07-18 11:10:00 | 1523.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-23 09:50:00 | 1487.55 | 2024-07-23 11:05:00 | 1491.24 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-31 10:05:00 | 1556.20 | 2024-07-31 11:50:00 | 1551.31 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-13 10:45:00 | 1521.70 | 2024-08-13 11:10:00 | 1517.93 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-14 10:30:00 | 1525.40 | 2024-08-14 11:40:00 | 1520.88 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-19 10:30:00 | 1593.00 | 2024-08-19 14:05:00 | 1599.55 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-19 10:30:00 | 1593.00 | 2024-08-19 15:05:00 | 1593.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:25:00 | 1610.95 | 2024-08-20 11:15:00 | 1607.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-23 09:45:00 | 1595.00 | 2024-08-23 09:50:00 | 1599.28 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-27 11:05:00 | 1630.00 | 2024-08-27 14:45:00 | 1624.19 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-27 11:05:00 | 1630.00 | 2024-08-27 15:20:00 | 1625.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-28 10:35:00 | 1645.35 | 2024-08-28 10:45:00 | 1640.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-02 11:10:00 | 1658.25 | 2024-09-02 11:25:00 | 1654.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-03 11:00:00 | 1649.35 | 2024-09-03 11:55:00 | 1654.91 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-03 11:00:00 | 1649.35 | 2024-09-03 14:30:00 | 1649.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 11:05:00 | 1629.90 | 2024-09-06 11:15:00 | 1634.01 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-11 10:55:00 | 1616.65 | 2024-09-11 11:45:00 | 1622.58 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-11 10:55:00 | 1616.65 | 2024-09-11 13:45:00 | 1617.80 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-09-12 11:10:00 | 1612.25 | 2024-09-12 11:25:00 | 1606.05 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-12 11:10:00 | 1612.25 | 2024-09-12 12:25:00 | 1612.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:50:00 | 1651.55 | 2024-09-13 10:55:00 | 1647.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-16 10:45:00 | 1666.00 | 2024-09-16 11:05:00 | 1662.48 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-17 10:15:00 | 1654.40 | 2024-09-17 10:25:00 | 1659.80 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-09-17 10:15:00 | 1654.40 | 2024-09-17 13:05:00 | 1659.20 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-23 11:05:00 | 1610.05 | 2024-09-23 11:30:00 | 1603.68 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-23 11:05:00 | 1610.05 | 2024-09-23 15:00:00 | 1605.50 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-25 10:25:00 | 1610.70 | 2024-09-25 11:25:00 | 1602.07 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-25 10:25:00 | 1610.70 | 2024-09-25 15:05:00 | 1604.75 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1606.25 | 2024-10-07 11:15:00 | 1611.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-14 11:10:00 | 1676.45 | 2024-10-14 11:30:00 | 1683.00 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-10-14 11:10:00 | 1676.45 | 2024-10-14 15:20:00 | 1693.00 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2024-10-17 09:55:00 | 1685.70 | 2024-10-17 10:00:00 | 1679.44 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-18 10:25:00 | 1657.20 | 2024-10-18 10:30:00 | 1665.67 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-10-25 11:00:00 | 1728.05 | 2024-10-25 11:10:00 | 1733.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1663.30 | 2024-11-06 09:55:00 | 1671.93 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1663.30 | 2024-11-06 15:20:00 | 1697.00 | TARGET_HIT | 0.50 | 2.03% |
| BUY | retest1 | 2024-11-08 09:35:00 | 1672.65 | 2024-11-08 09:45:00 | 1678.66 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-11-08 09:35:00 | 1672.65 | 2024-11-08 15:20:00 | 1680.15 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-11 11:05:00 | 1713.75 | 2024-11-11 11:10:00 | 1709.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-19 10:40:00 | 1705.70 | 2024-11-19 10:45:00 | 1701.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-11-22 09:55:00 | 1733.50 | 2024-11-22 10:30:00 | 1740.87 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-22 09:55:00 | 1733.50 | 2024-11-22 10:40:00 | 1733.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-25 10:55:00 | 1750.00 | 2024-11-25 11:00:00 | 1753.86 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-11-29 10:50:00 | 1719.25 | 2024-11-29 13:40:00 | 1713.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-04 09:30:00 | 1774.00 | 2024-12-04 09:35:00 | 1769.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-13 11:10:00 | 1766.90 | 2024-12-13 11:25:00 | 1771.75 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-23 11:10:00 | 1716.95 | 2024-12-23 11:30:00 | 1724.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-23 11:10:00 | 1716.95 | 2024-12-23 12:00:00 | 1716.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:15:00 | 1697.15 | 2024-12-26 10:30:00 | 1701.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-02 11:15:00 | 1720.45 | 2025-01-02 11:35:00 | 1716.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-03 10:30:00 | 1697.10 | 2025-01-03 10:45:00 | 1689.95 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-03 10:30:00 | 1697.10 | 2025-01-03 11:20:00 | 1697.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:10:00 | 1685.70 | 2025-01-06 11:25:00 | 1690.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-22 10:45:00 | 1666.00 | 2025-01-22 11:00:00 | 1673.21 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-22 10:45:00 | 1666.00 | 2025-01-22 13:30:00 | 1666.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 10:55:00 | 1679.60 | 2025-01-29 11:10:00 | 1673.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 10:10:00 | 1699.30 | 2025-01-30 11:00:00 | 1693.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-04 11:10:00 | 1649.40 | 2025-02-04 11:25:00 | 1653.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-20 10:50:00 | 1672.55 | 2025-02-20 12:30:00 | 1666.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-02-20 10:50:00 | 1672.55 | 2025-02-20 15:20:00 | 1657.60 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2025-03-06 11:05:00 | 1524.55 | 2025-03-06 11:20:00 | 1518.71 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-03-06 11:05:00 | 1524.55 | 2025-03-06 11:50:00 | 1524.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:10:00 | 1448.80 | 2025-03-18 10:55:00 | 1445.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-25 09:50:00 | 1482.50 | 2025-03-25 10:00:00 | 1477.07 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-26 10:20:00 | 1439.40 | 2025-03-26 11:10:00 | 1432.11 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-03-26 10:20:00 | 1439.40 | 2025-03-26 15:20:00 | 1416.45 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-04-02 09:30:00 | 1424.00 | 2025-04-02 09:35:00 | 1418.97 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-03 11:00:00 | 1380.00 | 2025-04-03 11:15:00 | 1373.75 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-03 11:00:00 | 1380.00 | 2025-04-03 15:20:00 | 1371.25 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1384.80 | 2025-04-22 09:45:00 | 1393.64 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1384.80 | 2025-04-22 09:55:00 | 1384.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-29 10:40:00 | 1481.60 | 2025-04-29 10:55:00 | 1487.72 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-29 10:40:00 | 1481.60 | 2025-04-29 15:20:00 | 1494.90 | TARGET_HIT | 0.50 | 0.90% |
