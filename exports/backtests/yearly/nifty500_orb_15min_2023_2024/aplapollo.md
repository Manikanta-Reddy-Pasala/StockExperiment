# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-02-28 15:25:00 (33593 bars)
- **Last close:** 1430.45
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 16 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 55
- **Target hits / Stop hits / Partials:** 16 / 55 / 29
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 21.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 29 | 50.9% | 12 | 28 | 17 | 0.27% | 15.4% |
| BUY @ 2nd Alert (retest1) | 57 | 29 | 50.9% | 12 | 28 | 17 | 0.27% | 15.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 16 | 37.2% | 4 | 27 | 12 | 0.14% | 5.9% |
| SELL @ 2nd Alert (retest1) | 43 | 16 | 37.2% | 4 | 27 | 12 | 0.14% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 100 | 45 | 45.0% | 16 | 55 | 29 | 0.21% | 21.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-12 10:50:00 | 1179.35 | 1188.15 | 0.00 | ORB-short ORB[1184.00,1195.00] vol=3.4x ATR=4.28 |
| Stop hit — per-position SL triggered | 2023-05-12 11:00:00 | 1183.63 | 1187.39 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:05:00 | 1152.55 | 1156.48 | 0.00 | ORB-short ORB[1155.65,1163.35] vol=5.1x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-05-18 10:10:00 | 1154.84 | 1156.44 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 10:00:00 | 1108.25 | 1115.09 | 0.00 | ORB-short ORB[1110.50,1125.00] vol=1.8x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-05-31 10:05:00 | 1111.99 | 1114.78 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:45:00 | 1126.60 | 1130.40 | 0.00 | ORB-short ORB[1127.80,1143.00] vol=1.6x ATR=2.95 |
| Stop hit — per-position SL triggered | 2023-06-02 11:00:00 | 1129.55 | 1130.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:40:00 | 1151.90 | 1146.93 | 0.00 | ORB-long ORB[1140.95,1151.75] vol=2.8x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 10:20:00 | 1157.77 | 1149.78 | 0.00 | T1 1.5R @ 1157.77 |
| Target hit | 2023-06-05 15:20:00 | 1171.00 | 1158.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 1190.00 | 1186.77 | 0.00 | ORB-long ORB[1176.55,1188.85] vol=2.1x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:00:00 | 1196.05 | 1188.94 | 0.00 | T1 1.5R @ 1196.05 |
| Target hit | 2023-06-08 15:20:00 | 1200.95 | 1200.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2023-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:30:00 | 1247.00 | 1239.85 | 0.00 | ORB-long ORB[1228.10,1245.00] vol=1.8x ATR=3.98 |
| Stop hit — per-position SL triggered | 2023-06-12 09:40:00 | 1243.02 | 1240.46 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:45:00 | 1286.95 | 1278.05 | 0.00 | ORB-long ORB[1267.95,1283.00] vol=2.1x ATR=4.60 |
| Stop hit — per-position SL triggered | 2023-06-13 09:50:00 | 1282.35 | 1278.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 09:35:00 | 1296.65 | 1304.85 | 0.00 | ORB-short ORB[1304.15,1322.25] vol=1.6x ATR=5.43 |
| Stop hit — per-position SL triggered | 2023-06-16 11:25:00 | 1302.08 | 1298.89 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 1304.20 | 1309.82 | 0.00 | ORB-short ORB[1305.50,1324.00] vol=1.8x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 1308.98 | 1309.03 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 09:55:00 | 1345.05 | 1354.32 | 0.00 | ORB-short ORB[1351.25,1369.90] vol=1.7x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:30:00 | 1335.58 | 1352.63 | 0.00 | T1 1.5R @ 1335.58 |
| Stop hit — per-position SL triggered | 2023-06-21 13:10:00 | 1345.05 | 1344.28 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 1358.75 | 1349.86 | 0.00 | ORB-long ORB[1330.30,1350.50] vol=1.5x ATR=6.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 10:00:00 | 1368.36 | 1354.07 | 0.00 | T1 1.5R @ 1368.36 |
| Target hit | 2023-06-22 15:20:00 | 1390.55 | 1379.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2023-06-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:50:00 | 1337.10 | 1342.92 | 0.00 | ORB-short ORB[1338.55,1352.00] vol=1.7x ATR=4.57 |
| Stop hit — per-position SL triggered | 2023-06-28 10:15:00 | 1341.67 | 1342.06 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 11:15:00 | 1309.40 | 1314.26 | 0.00 | ORB-short ORB[1311.00,1329.75] vol=2.2x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-06-30 11:35:00 | 1312.82 | 1313.55 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:55:00 | 1326.35 | 1332.49 | 0.00 | ORB-short ORB[1332.10,1340.45] vol=3.5x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 1330.48 | 1332.17 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:50:00 | 1329.95 | 1324.81 | 0.00 | ORB-long ORB[1313.95,1329.00] vol=1.6x ATR=3.57 |
| Stop hit — per-position SL triggered | 2023-07-05 11:00:00 | 1326.38 | 1324.98 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:30:00 | 1325.00 | 1321.65 | 0.00 | ORB-long ORB[1313.05,1324.20] vol=1.6x ATR=2.77 |
| Stop hit — per-position SL triggered | 2023-07-13 09:50:00 | 1322.23 | 1323.56 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:30:00 | 1351.20 | 1343.70 | 0.00 | ORB-long ORB[1332.50,1344.90] vol=4.7x ATR=5.00 |
| Stop hit — per-position SL triggered | 2023-07-17 09:35:00 | 1346.20 | 1344.37 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:55:00 | 1412.00 | 1403.49 | 0.00 | ORB-long ORB[1392.60,1409.40] vol=1.9x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:00:00 | 1418.98 | 1405.21 | 0.00 | T1 1.5R @ 1418.98 |
| Stop hit — per-position SL triggered | 2023-07-20 10:15:00 | 1412.00 | 1406.52 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 11:15:00 | 1486.30 | 1482.18 | 0.00 | ORB-long ORB[1470.10,1483.90] vol=1.7x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 11:55:00 | 1491.53 | 1483.83 | 0.00 | T1 1.5R @ 1491.53 |
| Target hit | 2023-07-26 15:20:00 | 1494.25 | 1487.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2023-07-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:05:00 | 1516.80 | 1514.45 | 0.00 | ORB-long ORB[1497.65,1514.00] vol=8.4x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 11:25:00 | 1522.80 | 1516.36 | 0.00 | T1 1.5R @ 1522.80 |
| Stop hit — per-position SL triggered | 2023-07-28 12:05:00 | 1516.80 | 1516.65 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 10:25:00 | 1573.00 | 1565.27 | 0.00 | ORB-long ORB[1543.70,1563.75] vol=6.5x ATR=4.72 |
| Stop hit — per-position SL triggered | 2023-08-14 10:45:00 | 1568.28 | 1565.68 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:30:00 | 1595.75 | 1586.16 | 0.00 | ORB-long ORB[1571.55,1588.20] vol=1.5x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 09:35:00 | 1603.80 | 1593.04 | 0.00 | T1 1.5R @ 1603.80 |
| Stop hit — per-position SL triggered | 2023-08-17 09:40:00 | 1595.75 | 1595.88 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:30:00 | 1583.15 | 1578.47 | 0.00 | ORB-long ORB[1565.00,1580.00] vol=1.8x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 10:00:00 | 1591.24 | 1581.24 | 0.00 | T1 1.5R @ 1591.24 |
| Target hit | 2023-08-23 12:50:00 | 1592.00 | 1592.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 11:15:00 | 1627.90 | 1617.65 | 0.00 | ORB-long ORB[1607.25,1622.65] vol=1.6x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 14:35:00 | 1633.79 | 1623.10 | 0.00 | T1 1.5R @ 1633.79 |
| Target hit | 2023-08-30 15:20:00 | 1642.90 | 1626.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2023-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:00:00 | 1778.10 | 1762.75 | 0.00 | ORB-long ORB[1745.80,1769.70] vol=1.5x ATR=8.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 12:05:00 | 1791.46 | 1773.88 | 0.00 | T1 1.5R @ 1791.46 |
| Target hit | 2023-09-05 15:20:00 | 1796.30 | 1784.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2023-09-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 11:05:00 | 1722.35 | 1729.61 | 0.00 | ORB-short ORB[1724.10,1744.95] vol=2.3x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 11:20:00 | 1715.14 | 1728.33 | 0.00 | T1 1.5R @ 1715.14 |
| Stop hit — per-position SL triggered | 2023-09-08 11:30:00 | 1722.35 | 1728.03 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 09:35:00 | 1621.30 | 1629.01 | 0.00 | ORB-short ORB[1625.00,1648.85] vol=1.6x ATR=11.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 09:55:00 | 1603.52 | 1622.32 | 0.00 | T1 1.5R @ 1603.52 |
| Target hit | 2023-09-13 11:35:00 | 1616.90 | 1610.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2023-09-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 11:05:00 | 1638.00 | 1643.35 | 0.00 | ORB-short ORB[1640.05,1657.70] vol=1.9x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:55:00 | 1630.47 | 1641.84 | 0.00 | T1 1.5R @ 1630.47 |
| Target hit | 2023-09-15 15:20:00 | 1617.35 | 1631.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-09-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 11:00:00 | 1600.55 | 1590.67 | 0.00 | ORB-long ORB[1579.00,1599.50] vol=1.8x ATR=5.95 |
| Target hit | 2023-09-21 15:20:00 | 1603.60 | 1595.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2023-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:30:00 | 1569.45 | 1561.39 | 0.00 | ORB-long ORB[1547.50,1568.00] vol=1.6x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 09:35:00 | 1579.03 | 1565.77 | 0.00 | T1 1.5R @ 1579.03 |
| Target hit | 2023-09-27 15:20:00 | 1628.50 | 1613.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 1648.50 | 1637.28 | 0.00 | ORB-long ORB[1619.15,1641.70] vol=2.0x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 09:55:00 | 1658.04 | 1643.23 | 0.00 | T1 1.5R @ 1658.04 |
| Target hit | 2023-10-11 14:35:00 | 1661.45 | 1661.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — BUY (started 2023-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:40:00 | 1737.00 | 1726.80 | 0.00 | ORB-long ORB[1717.15,1734.95] vol=1.8x ATR=5.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 09:55:00 | 1745.81 | 1733.40 | 0.00 | T1 1.5R @ 1745.81 |
| Stop hit — per-position SL triggered | 2023-10-17 10:00:00 | 1737.00 | 1733.85 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 1781.35 | 1779.57 | 0.00 | ORB-long ORB[1755.85,1776.40] vol=5.1x ATR=5.94 |
| Stop hit — per-position SL triggered | 2023-10-18 09:50:00 | 1775.41 | 1779.00 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-11-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:10:00 | 1545.40 | 1561.49 | 0.00 | ORB-short ORB[1560.50,1575.30] vol=1.8x ATR=7.33 |
| Stop hit — per-position SL triggered | 2023-11-01 10:15:00 | 1552.73 | 1560.16 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-11-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:55:00 | 1593.40 | 1582.43 | 0.00 | ORB-long ORB[1566.05,1588.00] vol=1.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 10:05:00 | 1600.48 | 1588.95 | 0.00 | T1 1.5R @ 1600.48 |
| Target hit | 2023-11-08 11:10:00 | 1597.05 | 1597.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2023-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:40:00 | 1686.45 | 1678.23 | 0.00 | ORB-long ORB[1666.10,1681.95] vol=2.2x ATR=7.24 |
| Stop hit — per-position SL triggered | 2023-11-13 10:05:00 | 1679.21 | 1682.50 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-11-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 10:55:00 | 1679.65 | 1670.35 | 0.00 | ORB-long ORB[1657.35,1677.00] vol=1.9x ATR=5.16 |
| Stop hit — per-position SL triggered | 2023-11-23 12:10:00 | 1674.49 | 1673.07 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 10:55:00 | 1705.85 | 1701.36 | 0.00 | ORB-long ORB[1685.05,1700.00] vol=3.0x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 11:00:00 | 1713.39 | 1703.01 | 0.00 | T1 1.5R @ 1713.39 |
| Stop hit — per-position SL triggered | 2023-11-24 12:10:00 | 1705.85 | 1706.56 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 11:15:00 | 1614.00 | 1627.57 | 0.00 | ORB-short ORB[1622.10,1642.90] vol=1.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-12-05 11:20:00 | 1617.94 | 1627.21 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-12-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:55:00 | 1607.00 | 1596.02 | 0.00 | ORB-long ORB[1583.10,1600.50] vol=1.9x ATR=4.57 |
| Stop hit — per-position SL triggered | 2023-12-08 10:20:00 | 1602.43 | 1599.30 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-12-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:55:00 | 1599.90 | 1605.66 | 0.00 | ORB-short ORB[1604.25,1626.00] vol=2.5x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-12-13 12:10:00 | 1603.50 | 1604.03 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 10:45:00 | 1621.40 | 1614.66 | 0.00 | ORB-long ORB[1600.25,1615.85] vol=4.0x ATR=4.39 |
| Stop hit — per-position SL triggered | 2023-12-18 11:10:00 | 1617.01 | 1615.98 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 10:45:00 | 1628.65 | 1624.16 | 0.00 | ORB-long ORB[1608.00,1623.30] vol=10.4x ATR=3.31 |
| Stop hit — per-position SL triggered | 2023-12-19 10:50:00 | 1625.34 | 1624.20 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 11:15:00 | 1597.00 | 1605.48 | 0.00 | ORB-short ORB[1600.05,1622.60] vol=2.7x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:05:00 | 1589.87 | 1601.35 | 0.00 | T1 1.5R @ 1589.87 |
| Target hit | 2023-12-20 15:20:00 | 1575.00 | 1585.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2023-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:30:00 | 1602.00 | 1596.10 | 0.00 | ORB-long ORB[1586.50,1599.95] vol=5.4x ATR=7.07 |
| Stop hit — per-position SL triggered | 2023-12-22 09:35:00 | 1594.93 | 1596.24 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 1597.95 | 1592.81 | 0.00 | ORB-long ORB[1586.50,1595.70] vol=2.1x ATR=4.43 |
| Stop hit — per-position SL triggered | 2023-12-27 12:15:00 | 1593.52 | 1598.50 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-01-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:05:00 | 1486.70 | 1481.52 | 0.00 | ORB-long ORB[1472.10,1485.00] vol=1.6x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-01-04 10:15:00 | 1483.06 | 1482.11 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:15:00 | 1551.65 | 1560.07 | 0.00 | ORB-short ORB[1555.95,1577.00] vol=1.8x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-01-11 10:20:00 | 1555.14 | 1560.14 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 10:40:00 | 1519.95 | 1528.27 | 0.00 | ORB-short ORB[1523.00,1542.85] vol=1.6x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:45:00 | 1513.27 | 1527.66 | 0.00 | T1 1.5R @ 1513.27 |
| Stop hit — per-position SL triggered | 2024-01-23 10:50:00 | 1519.95 | 1527.52 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-02-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 11:00:00 | 1471.95 | 1472.61 | 0.00 | ORB-short ORB[1476.05,1494.00] vol=1.8x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:20:00 | 1464.41 | 1472.16 | 0.00 | T1 1.5R @ 1464.41 |
| Target hit | 2024-02-05 15:20:00 | 1434.45 | 1452.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-02-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:50:00 | 1351.80 | 1366.15 | 0.00 | ORB-short ORB[1369.00,1389.00] vol=1.9x ATR=6.74 |
| Stop hit — per-position SL triggered | 2024-02-09 09:55:00 | 1358.54 | 1365.43 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-02-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:55:00 | 1440.75 | 1436.01 | 0.00 | ORB-long ORB[1413.95,1432.25] vol=9.4x ATR=4.62 |
| Stop hit — per-position SL triggered | 2024-02-16 11:30:00 | 1436.13 | 1438.56 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:35:00 | 1423.15 | 1431.41 | 0.00 | ORB-short ORB[1425.10,1444.00] vol=1.8x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:45:00 | 1415.73 | 1428.06 | 0.00 | T1 1.5R @ 1415.73 |
| Stop hit — per-position SL triggered | 2024-02-20 10:50:00 | 1423.15 | 1425.44 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:55:00 | 1439.90 | 1433.94 | 0.00 | ORB-long ORB[1422.25,1433.00] vol=1.6x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-02-23 10:00:00 | 1434.89 | 1434.18 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:55:00 | 1471.20 | 1462.17 | 0.00 | ORB-long ORB[1450.80,1467.65] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-02-27 12:00:00 | 1466.96 | 1465.49 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 1464.15 | 1468.59 | 0.00 | ORB-short ORB[1468.10,1485.95] vol=2.3x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 12:10:00 | 1459.21 | 1466.55 | 0.00 | T1 1.5R @ 1459.21 |
| Stop hit — per-position SL triggered | 2024-02-28 14:05:00 | 1464.15 | 1462.31 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-03-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:45:00 | 1571.95 | 1563.72 | 0.00 | ORB-long ORB[1545.05,1563.80] vol=3.0x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 12:30:00 | 1584.12 | 1572.95 | 0.00 | T1 1.5R @ 1584.12 |
| Target hit | 2024-03-07 15:20:00 | 1579.10 | 1579.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2024-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 10:50:00 | 1503.00 | 1491.90 | 0.00 | ORB-long ORB[1463.70,1480.00] vol=3.0x ATR=7.64 |
| Stop hit — per-position SL triggered | 2024-03-18 11:10:00 | 1495.36 | 1492.51 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:05:00 | 1546.00 | 1553.00 | 0.00 | ORB-short ORB[1548.00,1570.85] vol=1.6x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:05:00 | 1535.27 | 1546.42 | 0.00 | T1 1.5R @ 1535.27 |
| Stop hit — per-position SL triggered | 2024-03-19 11:20:00 | 1546.00 | 1546.24 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-03-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 09:40:00 | 1473.30 | 1480.62 | 0.00 | ORB-short ORB[1483.15,1494.75] vol=2.0x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-03-27 10:40:00 | 1478.15 | 1477.05 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 11:00:00 | 1490.70 | 1502.36 | 0.00 | ORB-short ORB[1497.20,1509.95] vol=2.3x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-04-01 11:20:00 | 1496.08 | 1501.06 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 1589.30 | 1595.37 | 0.00 | ORB-short ORB[1590.10,1610.00] vol=1.6x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-04-04 09:45:00 | 1594.57 | 1593.38 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:55:00 | 1565.95 | 1585.70 | 0.00 | ORB-short ORB[1586.30,1607.00] vol=3.7x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 11:10:00 | 1557.67 | 1581.74 | 0.00 | T1 1.5R @ 1557.67 |
| Stop hit — per-position SL triggered | 2024-04-10 11:15:00 | 1565.95 | 1580.97 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-04-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:45:00 | 1586.85 | 1574.18 | 0.00 | ORB-long ORB[1561.55,1582.00] vol=2.4x ATR=5.15 |
| Stop hit — per-position SL triggered | 2024-04-12 10:50:00 | 1581.70 | 1574.46 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:45:00 | 1545.00 | 1555.52 | 0.00 | ORB-short ORB[1553.10,1575.00] vol=1.6x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-04-22 12:45:00 | 1549.65 | 1549.97 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:35:00 | 1560.05 | 1556.34 | 0.00 | ORB-long ORB[1550.00,1558.85] vol=1.7x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-04-23 09:45:00 | 1555.42 | 1556.44 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:45:00 | 1590.00 | 1582.83 | 0.00 | ORB-long ORB[1567.30,1585.00] vol=2.1x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-04-24 09:55:00 | 1584.79 | 1583.24 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-04-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 11:00:00 | 1564.25 | 1573.10 | 0.00 | ORB-short ORB[1575.00,1595.95] vol=1.5x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:20:00 | 1558.78 | 1571.76 | 0.00 | T1 1.5R @ 1558.78 |
| Stop hit — per-position SL triggered | 2024-04-25 13:45:00 | 1564.25 | 1566.31 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 11:10:00 | 1546.95 | 1551.38 | 0.00 | ORB-short ORB[1550.00,1563.70] vol=2.9x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-04-29 11:20:00 | 1549.80 | 1551.30 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:35:00 | 1572.00 | 1563.51 | 0.00 | ORB-long ORB[1555.85,1568.95] vol=1.5x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:25:00 | 1581.54 | 1572.70 | 0.00 | T1 1.5R @ 1581.54 |
| Stop hit — per-position SL triggered | 2024-04-30 11:55:00 | 1572.00 | 1574.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-12 10:50:00 | 1179.35 | 2023-05-12 11:00:00 | 1183.63 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-05-18 10:05:00 | 1152.55 | 2023-05-18 10:10:00 | 1154.84 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-05-31 10:00:00 | 1108.25 | 2023-05-31 10:05:00 | 1111.99 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-02 10:45:00 | 1126.60 | 2023-06-02 11:00:00 | 1129.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-05 09:40:00 | 1151.90 | 2023-06-05 10:20:00 | 1157.77 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-06-05 09:40:00 | 1151.90 | 2023-06-05 15:20:00 | 1171.00 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2023-06-08 09:35:00 | 1190.00 | 2023-06-08 10:00:00 | 1196.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-06-08 09:35:00 | 1190.00 | 2023-06-08 15:20:00 | 1200.95 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2023-06-12 09:30:00 | 1247.00 | 2023-06-12 09:40:00 | 1243.02 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-06-13 09:45:00 | 1286.95 | 2023-06-13 09:50:00 | 1282.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-06-16 09:35:00 | 1296.65 | 2023-06-16 11:25:00 | 1302.08 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-06-20 09:30:00 | 1304.20 | 2023-06-20 09:35:00 | 1308.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-06-21 09:55:00 | 1345.05 | 2023-06-21 10:30:00 | 1335.58 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2023-06-21 09:55:00 | 1345.05 | 2023-06-21 13:10:00 | 1345.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-22 09:40:00 | 1358.75 | 2023-06-22 10:00:00 | 1368.36 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2023-06-22 09:40:00 | 1358.75 | 2023-06-22 15:20:00 | 1390.55 | TARGET_HIT | 0.50 | 2.34% |
| SELL | retest1 | 2023-06-28 09:50:00 | 1337.10 | 2023-06-28 10:15:00 | 1341.67 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-30 11:15:00 | 1309.40 | 2023-06-30 11:35:00 | 1312.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-04 09:55:00 | 1326.35 | 2023-07-04 10:05:00 | 1330.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-05 10:50:00 | 1329.95 | 2023-07-05 11:00:00 | 1326.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-13 09:30:00 | 1325.00 | 2023-07-13 09:50:00 | 1322.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-17 09:30:00 | 1351.20 | 2023-07-17 09:35:00 | 1346.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-20 09:55:00 | 1412.00 | 2023-07-20 10:00:00 | 1418.98 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-07-20 09:55:00 | 1412.00 | 2023-07-20 10:15:00 | 1412.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-26 11:15:00 | 1486.30 | 2023-07-26 11:55:00 | 1491.53 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-26 11:15:00 | 1486.30 | 2023-07-26 15:20:00 | 1494.25 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-28 10:05:00 | 1516.80 | 2023-07-28 11:25:00 | 1522.80 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-28 10:05:00 | 1516.80 | 2023-07-28 12:05:00 | 1516.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-14 10:25:00 | 1573.00 | 2023-08-14 10:45:00 | 1568.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-08-17 09:30:00 | 1595.75 | 2023-08-17 09:35:00 | 1603.80 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-08-17 09:30:00 | 1595.75 | 2023-08-17 09:40:00 | 1595.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-23 09:30:00 | 1583.15 | 2023-08-23 10:00:00 | 1591.24 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-23 09:30:00 | 1583.15 | 2023-08-23 12:50:00 | 1592.00 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2023-08-30 11:15:00 | 1627.90 | 2023-08-30 14:35:00 | 1633.79 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-30 11:15:00 | 1627.90 | 2023-08-30 15:20:00 | 1642.90 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2023-09-05 10:00:00 | 1778.10 | 2023-09-05 12:05:00 | 1791.46 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2023-09-05 10:00:00 | 1778.10 | 2023-09-05 15:20:00 | 1796.30 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2023-09-08 11:05:00 | 1722.35 | 2023-09-08 11:20:00 | 1715.14 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-09-08 11:05:00 | 1722.35 | 2023-09-08 11:30:00 | 1722.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-13 09:35:00 | 1621.30 | 2023-09-13 09:55:00 | 1603.52 | PARTIAL | 0.50 | 1.10% |
| SELL | retest1 | 2023-09-13 09:35:00 | 1621.30 | 2023-09-13 11:35:00 | 1616.90 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2023-09-15 11:05:00 | 1638.00 | 2023-09-15 11:55:00 | 1630.47 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-15 11:05:00 | 1638.00 | 2023-09-15 15:20:00 | 1617.35 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2023-09-21 11:00:00 | 1600.55 | 2023-09-21 15:20:00 | 1603.60 | TARGET_HIT | 1.00 | 0.19% |
| BUY | retest1 | 2023-09-27 09:30:00 | 1569.45 | 2023-09-27 09:35:00 | 1579.03 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-09-27 09:30:00 | 1569.45 | 2023-09-27 15:20:00 | 1628.50 | TARGET_HIT | 0.50 | 3.76% |
| BUY | retest1 | 2023-10-11 09:35:00 | 1648.50 | 2023-10-11 09:55:00 | 1658.04 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-10-11 09:35:00 | 1648.50 | 2023-10-11 14:35:00 | 1661.45 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2023-10-17 09:40:00 | 1737.00 | 2023-10-17 09:55:00 | 1745.81 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-10-17 09:40:00 | 1737.00 | 2023-10-17 10:00:00 | 1737.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 09:45:00 | 1781.35 | 2023-10-18 09:50:00 | 1775.41 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-11-01 10:10:00 | 1545.40 | 2023-11-01 10:15:00 | 1552.73 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-11-08 09:55:00 | 1593.40 | 2023-11-08 10:05:00 | 1600.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-08 09:55:00 | 1593.40 | 2023-11-08 11:10:00 | 1597.05 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2023-11-13 09:40:00 | 1686.45 | 2023-11-13 10:05:00 | 1679.21 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-11-23 10:55:00 | 1679.65 | 2023-11-23 12:10:00 | 1674.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-24 10:55:00 | 1705.85 | 2023-11-24 11:00:00 | 1713.39 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-24 10:55:00 | 1705.85 | 2023-11-24 12:10:00 | 1705.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-05 11:15:00 | 1614.00 | 2023-12-05 11:20:00 | 1617.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-08 09:55:00 | 1607.00 | 2023-12-08 10:20:00 | 1602.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-13 10:55:00 | 1599.90 | 2023-12-13 12:10:00 | 1603.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-18 10:45:00 | 1621.40 | 2023-12-18 11:10:00 | 1617.01 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-19 10:45:00 | 1628.65 | 2023-12-19 10:50:00 | 1625.34 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-20 11:15:00 | 1597.00 | 2023-12-20 13:05:00 | 1589.87 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-12-20 11:15:00 | 1597.00 | 2023-12-20 15:20:00 | 1575.00 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2023-12-22 09:30:00 | 1602.00 | 2023-12-22 09:35:00 | 1594.93 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-12-27 09:40:00 | 1597.95 | 2023-12-27 12:15:00 | 1593.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-01-04 10:05:00 | 1486.70 | 2024-01-04 10:15:00 | 1483.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-11 10:15:00 | 1551.65 | 2024-01-11 10:20:00 | 1555.14 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-23 10:40:00 | 1519.95 | 2024-01-23 10:45:00 | 1513.27 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-01-23 10:40:00 | 1519.95 | 2024-01-23 10:50:00 | 1519.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-05 11:00:00 | 1471.95 | 2024-02-05 11:20:00 | 1464.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-02-05 11:00:00 | 1471.95 | 2024-02-05 15:20:00 | 1434.45 | TARGET_HIT | 0.50 | 2.55% |
| SELL | retest1 | 2024-02-09 09:50:00 | 1351.80 | 2024-02-09 09:55:00 | 1358.54 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-02-16 09:55:00 | 1440.75 | 2024-02-16 11:30:00 | 1436.13 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-20 10:35:00 | 1423.15 | 2024-02-20 10:45:00 | 1415.73 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-20 10:35:00 | 1423.15 | 2024-02-20 10:50:00 | 1423.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-23 09:55:00 | 1439.90 | 2024-02-23 10:00:00 | 1434.89 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-02-27 10:55:00 | 1471.20 | 2024-02-27 12:00:00 | 1466.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-28 10:50:00 | 1464.15 | 2024-02-28 12:10:00 | 1459.21 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-02-28 10:50:00 | 1464.15 | 2024-02-28 14:05:00 | 1464.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-07 09:45:00 | 1571.95 | 2024-03-07 12:30:00 | 1584.12 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-03-07 09:45:00 | 1571.95 | 2024-03-07 15:20:00 | 1579.10 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-03-18 10:50:00 | 1503.00 | 2024-03-18 11:10:00 | 1495.36 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-03-19 10:05:00 | 1546.00 | 2024-03-19 11:05:00 | 1535.27 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-03-19 10:05:00 | 1546.00 | 2024-03-19 11:20:00 | 1546.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-27 09:40:00 | 1473.30 | 2024-03-27 10:40:00 | 1478.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-01 11:00:00 | 1490.70 | 2024-04-01 11:20:00 | 1496.08 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-04 09:30:00 | 1589.30 | 2024-04-04 09:45:00 | 1594.57 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-10 10:55:00 | 1565.95 | 2024-04-10 11:10:00 | 1557.67 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-04-10 10:55:00 | 1565.95 | 2024-04-10 11:15:00 | 1565.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-12 10:45:00 | 1586.85 | 2024-04-12 10:50:00 | 1581.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-04-22 10:45:00 | 1545.00 | 2024-04-22 12:45:00 | 1549.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-23 09:35:00 | 1560.05 | 2024-04-23 09:45:00 | 1555.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-24 09:45:00 | 1590.00 | 2024-04-24 09:55:00 | 1584.79 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-25 11:00:00 | 1564.25 | 2024-04-25 11:20:00 | 1558.78 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-04-25 11:00:00 | 1564.25 | 2024-04-25 13:45:00 | 1564.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-29 11:10:00 | 1546.95 | 2024-04-29 11:20:00 | 1549.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-30 09:35:00 | 1572.00 | 2024-04-30 10:25:00 | 1581.54 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-04-30 09:35:00 | 1572.00 | 2024-04-30 11:55:00 | 1572.00 | STOP_HIT | 0.50 | 0.00% |
