# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 1671.00
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
| TARGET_HIT | 15 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 61
- **Target hits / Stop hits / Partials:** 15 / 61 / 31
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 23.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 28 | 47.5% | 9 | 31 | 19 | 0.35% | 20.8% |
| BUY @ 2nd Alert (retest1) | 59 | 28 | 47.5% | 9 | 31 | 19 | 0.35% | 20.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 18 | 37.5% | 6 | 30 | 12 | 0.06% | 3.0% |
| SELL @ 2nd Alert (retest1) | 48 | 18 | 37.5% | 6 | 30 | 12 | 0.06% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 107 | 46 | 43.0% | 15 | 61 | 31 | 0.22% | 23.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:45:00 | 1234.75 | 1242.49 | 0.00 | ORB-short ORB[1237.05,1250.00] vol=1.6x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-05-16 09:50:00 | 1239.08 | 1242.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 1267.45 | 1258.47 | 0.00 | ORB-long ORB[1256.45,1265.95] vol=2.7x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:30:00 | 1272.97 | 1261.16 | 0.00 | T1 1.5R @ 1272.97 |
| Target hit | 2024-05-17 15:20:00 | 1286.50 | 1276.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:15:00 | 1281.65 | 1290.85 | 0.00 | ORB-short ORB[1291.35,1304.95] vol=2.3x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:40:00 | 1276.15 | 1289.54 | 0.00 | T1 1.5R @ 1276.15 |
| Target hit | 2024-05-22 15:20:00 | 1273.00 | 1279.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 1247.50 | 1256.38 | 0.00 | ORB-short ORB[1252.75,1269.70] vol=2.4x ATR=4.43 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 1251.93 | 1255.87 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 1260.00 | 1265.09 | 0.00 | ORB-short ORB[1261.95,1272.90] vol=1.5x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-05-27 10:40:00 | 1264.36 | 1262.44 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:10:00 | 1227.00 | 1230.96 | 0.00 | ORB-short ORB[1231.20,1246.00] vol=2.3x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:15:00 | 1221.19 | 1228.78 | 0.00 | T1 1.5R @ 1221.19 |
| Target hit | 2024-05-30 15:20:00 | 1216.40 | 1219.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:30:00 | 1308.00 | 1298.29 | 0.00 | ORB-long ORB[1283.85,1302.95] vol=1.6x ATR=6.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 1318.14 | 1307.51 | 0.00 | T1 1.5R @ 1318.14 |
| Target hit | 2024-06-06 15:20:00 | 1333.70 | 1327.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 1394.00 | 1382.16 | 0.00 | ORB-long ORB[1365.80,1385.00] vol=4.1x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:50:00 | 1403.17 | 1392.03 | 0.00 | T1 1.5R @ 1403.17 |
| Target hit | 2024-06-13 15:20:00 | 1437.35 | 1416.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:30:00 | 1455.95 | 1447.09 | 0.00 | ORB-long ORB[1435.30,1454.80] vol=1.7x ATR=4.76 |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 1451.19 | 1449.30 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 1442.00 | 1449.65 | 0.00 | ORB-short ORB[1447.20,1463.95] vol=1.5x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-06-19 10:40:00 | 1446.96 | 1449.55 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:00:00 | 1431.85 | 1437.65 | 0.00 | ORB-short ORB[1435.40,1448.00] vol=1.5x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:25:00 | 1425.48 | 1436.29 | 0.00 | T1 1.5R @ 1425.48 |
| Target hit | 2024-06-21 15:20:00 | 1398.95 | 1411.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:05:00 | 1447.15 | 1435.52 | 0.00 | ORB-long ORB[1420.90,1438.00] vol=1.8x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-06-26 11:35:00 | 1442.30 | 1436.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:20:00 | 1439.80 | 1431.51 | 0.00 | ORB-long ORB[1418.00,1436.00] vol=1.6x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-06-27 10:30:00 | 1434.47 | 1432.11 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 1409.00 | 1417.06 | 0.00 | ORB-short ORB[1419.60,1427.35] vol=1.9x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-07-05 11:25:00 | 1411.60 | 1416.52 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:40:00 | 1418.00 | 1420.24 | 0.00 | ORB-short ORB[1421.30,1433.05] vol=10.4x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:05:00 | 1412.26 | 1419.92 | 0.00 | T1 1.5R @ 1412.26 |
| Target hit | 2024-07-08 15:20:00 | 1406.75 | 1409.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 1404.15 | 1411.45 | 0.00 | ORB-short ORB[1409.00,1425.00] vol=1.5x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1409.14 | 1408.84 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:05:00 | 1427.90 | 1421.22 | 0.00 | ORB-long ORB[1414.20,1425.90] vol=4.3x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 1423.84 | 1422.14 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:15:00 | 1394.35 | 1388.43 | 0.00 | ORB-long ORB[1375.20,1388.40] vol=1.6x ATR=5.30 |
| Stop hit — per-position SL triggered | 2024-07-15 10:25:00 | 1389.05 | 1388.84 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:15:00 | 1415.10 | 1408.73 | 0.00 | ORB-long ORB[1398.95,1408.55] vol=1.5x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:25:00 | 1422.66 | 1413.46 | 0.00 | T1 1.5R @ 1422.66 |
| Target hit | 2024-07-18 15:20:00 | 1451.15 | 1435.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 1359.00 | 1367.53 | 0.00 | ORB-short ORB[1366.70,1385.00] vol=1.6x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:10:00 | 1350.48 | 1360.62 | 0.00 | T1 1.5R @ 1350.48 |
| Target hit | 2024-07-25 12:00:00 | 1354.55 | 1352.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:15:00 | 1404.60 | 1388.03 | 0.00 | ORB-long ORB[1368.75,1387.00] vol=3.3x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-08-02 12:00:00 | 1400.29 | 1393.70 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 1375.70 | 1367.53 | 0.00 | ORB-long ORB[1348.30,1363.45] vol=2.3x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:05:00 | 1384.21 | 1371.84 | 0.00 | T1 1.5R @ 1384.21 |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 1375.70 | 1373.71 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:55:00 | 1349.35 | 1345.35 | 0.00 | ORB-long ORB[1335.00,1346.45] vol=5.1x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-08-12 15:00:00 | 1345.03 | 1348.57 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:15:00 | 1332.10 | 1326.05 | 0.00 | ORB-long ORB[1309.10,1323.00] vol=6.6x ATR=6.13 |
| Stop hit — per-position SL triggered | 2024-08-14 11:05:00 | 1325.97 | 1326.73 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:00:00 | 1351.95 | 1354.79 | 0.00 | ORB-short ORB[1356.05,1369.85] vol=2.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2024-08-23 11:10:00 | 1355.00 | 1354.45 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:55:00 | 1373.60 | 1365.73 | 0.00 | ORB-long ORB[1350.00,1359.90] vol=1.7x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:05:00 | 1379.62 | 1373.89 | 0.00 | T1 1.5R @ 1379.62 |
| Target hit | 2024-08-26 10:40:00 | 1375.00 | 1375.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 1473.90 | 1467.68 | 0.00 | ORB-long ORB[1456.65,1465.80] vol=6.4x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:10:00 | 1482.55 | 1469.21 | 0.00 | T1 1.5R @ 1482.55 |
| Stop hit — per-position SL triggered | 2024-09-02 10:25:00 | 1473.90 | 1470.58 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:10:00 | 1590.05 | 1579.71 | 0.00 | ORB-long ORB[1570.00,1586.95] vol=2.2x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-09-13 10:45:00 | 1585.09 | 1582.79 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 1572.00 | 1565.49 | 0.00 | ORB-long ORB[1557.40,1571.35] vol=1.5x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:00:00 | 1579.77 | 1571.98 | 0.00 | T1 1.5R @ 1579.77 |
| Stop hit — per-position SL triggered | 2024-09-17 10:10:00 | 1572.00 | 1572.53 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 1610.00 | 1605.73 | 0.00 | ORB-long ORB[1595.15,1609.00] vol=2.4x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 1604.79 | 1606.61 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:30:00 | 1592.70 | 1596.73 | 0.00 | ORB-short ORB[1592.85,1608.00] vol=6.6x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:50:00 | 1586.02 | 1595.80 | 0.00 | T1 1.5R @ 1586.02 |
| Stop hit — per-position SL triggered | 2024-09-20 12:05:00 | 1592.70 | 1593.96 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 1602.25 | 1612.19 | 0.00 | ORB-short ORB[1603.80,1624.30] vol=1.5x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:50:00 | 1595.29 | 1610.14 | 0.00 | T1 1.5R @ 1595.29 |
| Stop hit — per-position SL triggered | 2024-09-25 12:45:00 | 1602.25 | 1604.36 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:30:00 | 1629.25 | 1619.22 | 0.00 | ORB-long ORB[1612.00,1625.15] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-09-26 10:35:00 | 1625.52 | 1620.34 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:15:00 | 1608.75 | 1623.34 | 0.00 | ORB-short ORB[1621.05,1638.00] vol=1.5x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:30:00 | 1603.19 | 1621.45 | 0.00 | T1 1.5R @ 1603.19 |
| Stop hit — per-position SL triggered | 2024-09-27 15:00:00 | 1608.75 | 1609.26 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:50:00 | 1602.50 | 1608.02 | 0.00 | ORB-short ORB[1605.00,1623.00] vol=1.8x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 1607.21 | 1607.89 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-04 10:45:00 | 1509.85 | 1515.12 | 0.00 | ORB-short ORB[1510.15,1524.90] vol=1.6x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 1515.66 | 1514.90 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 1474.00 | 1488.05 | 0.00 | ORB-short ORB[1499.30,1514.35] vol=1.8x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 1479.49 | 1484.77 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:05:00 | 1579.95 | 1560.67 | 0.00 | ORB-long ORB[1538.00,1551.10] vol=2.4x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-10-09 11:55:00 | 1573.08 | 1571.86 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:35:00 | 1483.65 | 1497.75 | 0.00 | ORB-short ORB[1501.80,1517.10] vol=1.6x ATR=4.70 |
| Stop hit — per-position SL triggered | 2024-10-11 10:40:00 | 1488.35 | 1497.03 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:15:00 | 1498.30 | 1503.66 | 0.00 | ORB-short ORB[1503.55,1524.70] vol=8.1x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-10-15 10:40:00 | 1503.41 | 1503.14 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:35:00 | 1448.65 | 1439.95 | 0.00 | ORB-long ORB[1425.90,1444.80] vol=1.6x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:55:00 | 1456.96 | 1444.61 | 0.00 | T1 1.5R @ 1456.96 |
| Stop hit — per-position SL triggered | 2024-10-22 10:05:00 | 1448.65 | 1445.43 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 1238.90 | 1245.60 | 0.00 | ORB-short ORB[1239.00,1254.50] vol=1.5x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-11-05 10:50:00 | 1243.84 | 1245.07 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 1270.50 | 1262.10 | 0.00 | ORB-long ORB[1244.85,1263.00] vol=3.6x ATR=3.72 |
| Stop hit — per-position SL triggered | 2024-11-11 11:05:00 | 1266.78 | 1263.04 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 11:15:00 | 1258.70 | 1262.05 | 0.00 | ORB-short ORB[1262.50,1278.35] vol=1.7x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 1262.93 | 1262.06 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:55:00 | 1252.80 | 1242.41 | 0.00 | ORB-long ORB[1232.85,1243.70] vol=2.3x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:30:00 | 1258.70 | 1247.72 | 0.00 | T1 1.5R @ 1258.70 |
| Stop hit — per-position SL triggered | 2024-11-26 12:05:00 | 1252.80 | 1249.05 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:00:00 | 1280.85 | 1276.31 | 0.00 | ORB-long ORB[1266.10,1278.10] vol=1.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-11-28 10:20:00 | 1277.06 | 1276.88 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1281.85 | 1288.57 | 0.00 | ORB-short ORB[1284.45,1302.65] vol=3.1x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-12-05 11:20:00 | 1286.14 | 1288.15 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 1278.70 | 1285.80 | 0.00 | ORB-short ORB[1282.55,1298.65] vol=1.5x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:15:00 | 1271.94 | 1282.11 | 0.00 | T1 1.5R @ 1271.94 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 1278.70 | 1281.64 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 1338.50 | 1343.99 | 0.00 | ORB-short ORB[1344.75,1353.00] vol=2.4x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 1342.40 | 1343.87 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 1220.85 | 1215.37 | 0.00 | ORB-long ORB[1212.00,1219.45] vol=1.8x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:40:00 | 1226.47 | 1218.53 | 0.00 | T1 1.5R @ 1226.47 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 1220.85 | 1219.15 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:10:00 | 1182.95 | 1176.66 | 0.00 | ORB-long ORB[1168.00,1179.75] vol=3.7x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:45:00 | 1188.15 | 1180.11 | 0.00 | T1 1.5R @ 1188.15 |
| Stop hit — per-position SL triggered | 2024-12-24 10:55:00 | 1182.95 | 1180.53 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 1191.90 | 1197.61 | 0.00 | ORB-short ORB[1193.90,1208.45] vol=1.7x ATR=4.26 |
| Stop hit — per-position SL triggered | 2024-12-27 09:55:00 | 1196.16 | 1196.91 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:10:00 | 1208.75 | 1216.24 | 0.00 | ORB-short ORB[1210.00,1223.65] vol=1.8x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-12-30 11:15:00 | 1213.50 | 1214.84 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 1181.20 | 1182.30 | 0.00 | ORB-short ORB[1181.65,1194.90] vol=3.4x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1184.56 | 1182.17 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:30:00 | 1283.90 | 1285.85 | 0.00 | ORB-short ORB[1287.05,1300.25] vol=4.7x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-01-07 10:55:00 | 1288.45 | 1285.56 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1268.40 | 1273.58 | 0.00 | ORB-short ORB[1268.55,1286.30] vol=1.6x ATR=4.44 |
| Stop hit — per-position SL triggered | 2025-01-09 11:35:00 | 1272.84 | 1272.50 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:45:00 | 1241.50 | 1249.49 | 0.00 | ORB-short ORB[1243.80,1256.70] vol=2.4x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:10:00 | 1233.94 | 1244.87 | 0.00 | T1 1.5R @ 1233.94 |
| Stop hit — per-position SL triggered | 2025-01-22 10:35:00 | 1241.50 | 1243.73 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 10:45:00 | 1237.10 | 1244.32 | 0.00 | ORB-short ORB[1239.85,1258.35] vol=2.7x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:55:00 | 1230.13 | 1241.93 | 0.00 | T1 1.5R @ 1230.13 |
| Stop hit — per-position SL triggered | 2025-01-23 11:55:00 | 1237.10 | 1234.70 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:35:00 | 1231.80 | 1236.35 | 0.00 | ORB-short ORB[1245.60,1252.35] vol=2.0x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-01-24 11:40:00 | 1235.94 | 1235.30 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:15:00 | 1280.50 | 1263.67 | 0.00 | ORB-long ORB[1230.05,1240.95] vol=3.8x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:45:00 | 1288.80 | 1269.81 | 0.00 | T1 1.5R @ 1288.80 |
| Target hit | 2025-01-29 15:20:00 | 1295.55 | 1289.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 1287.95 | 1278.45 | 0.00 | ORB-long ORB[1268.95,1284.00] vol=1.6x ATR=5.34 |
| Stop hit — per-position SL triggered | 2025-01-31 09:45:00 | 1282.61 | 1279.21 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-02-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 10:25:00 | 1370.20 | 1360.45 | 0.00 | ORB-long ORB[1342.05,1359.70] vol=1.5x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 11:20:00 | 1378.69 | 1364.04 | 0.00 | T1 1.5R @ 1378.69 |
| Target hit | 2025-02-04 15:20:00 | 1388.80 | 1377.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-02-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:30:00 | 1399.60 | 1391.44 | 0.00 | ORB-long ORB[1381.50,1396.65] vol=1.9x ATR=4.84 |
| Stop hit — per-position SL triggered | 2025-02-06 10:45:00 | 1394.76 | 1392.22 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 1382.25 | 1386.87 | 0.00 | ORB-short ORB[1382.50,1395.00] vol=1.8x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 10:15:00 | 1373.02 | 1385.45 | 0.00 | T1 1.5R @ 1373.02 |
| Target hit | 2025-02-07 10:55:00 | 1379.85 | 1379.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — BUY (started 2025-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:45:00 | 1362.10 | 1354.69 | 0.00 | ORB-long ORB[1343.45,1359.95] vol=3.9x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-02-19 09:50:00 | 1357.65 | 1354.81 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 1380.10 | 1370.92 | 0.00 | ORB-long ORB[1358.40,1373.40] vol=1.7x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:05:00 | 1385.61 | 1371.34 | 0.00 | T1 1.5R @ 1385.61 |
| Stop hit — per-position SL triggered | 2025-02-20 11:45:00 | 1380.10 | 1373.65 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:40:00 | 1393.85 | 1381.85 | 0.00 | ORB-long ORB[1367.55,1384.45] vol=2.0x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:45:00 | 1403.64 | 1386.18 | 0.00 | T1 1.5R @ 1403.64 |
| Stop hit — per-position SL triggered | 2025-02-25 09:55:00 | 1393.85 | 1389.61 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:40:00 | 1474.00 | 1459.84 | 0.00 | ORB-long ORB[1449.65,1464.95] vol=2.7x ATR=4.99 |
| Stop hit — per-position SL triggered | 2025-03-07 10:45:00 | 1469.01 | 1460.25 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:15:00 | 1465.95 | 1458.12 | 0.00 | ORB-long ORB[1447.75,1465.00] vol=1.5x ATR=4.58 |
| Stop hit — per-position SL triggered | 2025-03-13 10:20:00 | 1461.37 | 1458.51 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:10:00 | 1475.40 | 1467.54 | 0.00 | ORB-long ORB[1455.45,1471.00] vol=1.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 1471.06 | 1470.33 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 1565.00 | 1564.27 | 0.00 | ORB-long ORB[1544.00,1559.55] vol=4.2x ATR=6.89 |
| Stop hit — per-position SL triggered | 2025-03-24 10:50:00 | 1558.11 | 1563.77 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:50:00 | 1515.75 | 1522.04 | 0.00 | ORB-short ORB[1529.65,1551.00] vol=6.4x ATR=6.41 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 1522.16 | 1520.94 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-03-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 09:30:00 | 1511.40 | 1518.32 | 0.00 | ORB-short ORB[1513.40,1536.00] vol=1.6x ATR=5.56 |
| Stop hit — per-position SL triggered | 2025-03-28 09:45:00 | 1516.96 | 1517.84 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:40:00 | 1623.20 | 1610.86 | 0.00 | ORB-long ORB[1595.00,1613.80] vol=1.6x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:10:00 | 1632.99 | 1615.74 | 0.00 | T1 1.5R @ 1632.99 |
| Target hit | 2025-04-21 15:20:00 | 1660.00 | 1654.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:40:00 | 1487.20 | 1479.94 | 0.00 | ORB-long ORB[1465.60,1485.00] vol=1.8x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:45:00 | 1495.64 | 1481.67 | 0.00 | T1 1.5R @ 1495.64 |
| Stop hit — per-position SL triggered | 2025-04-30 10:10:00 | 1487.20 | 1485.32 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 1506.20 | 1497.57 | 0.00 | ORB-long ORB[1487.40,1505.00] vol=1.8x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:55:00 | 1514.69 | 1501.83 | 0.00 | T1 1.5R @ 1514.69 |
| Target hit | 2025-05-05 15:20:00 | 1555.60 | 1531.73 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:45:00 | 1234.75 | 2024-05-16 09:50:00 | 1239.08 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-17 11:15:00 | 1267.45 | 2024-05-17 11:30:00 | 1272.97 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-17 11:15:00 | 1267.45 | 2024-05-17 15:20:00 | 1286.50 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-05-22 11:15:00 | 1281.65 | 2024-05-22 11:40:00 | 1276.15 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-22 11:15:00 | 1281.65 | 2024-05-22 15:20:00 | 1273.00 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2024-05-24 09:35:00 | 1247.50 | 2024-05-24 09:40:00 | 1251.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-27 09:45:00 | 1260.00 | 2024-05-27 10:40:00 | 1264.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-30 10:10:00 | 1227.00 | 2024-05-30 11:15:00 | 1221.19 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-30 10:10:00 | 1227.00 | 2024-05-30 15:20:00 | 1216.40 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2024-06-06 09:30:00 | 1308.00 | 2024-06-06 09:45:00 | 1318.14 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-06-06 09:30:00 | 1308.00 | 2024-06-06 15:20:00 | 1333.70 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2024-06-13 09:30:00 | 1394.00 | 2024-06-13 09:50:00 | 1403.17 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-13 09:30:00 | 1394.00 | 2024-06-13 15:20:00 | 1437.35 | TARGET_HIT | 0.50 | 3.11% |
| BUY | retest1 | 2024-06-18 10:30:00 | 1455.95 | 2024-06-18 11:15:00 | 1451.19 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-19 10:35:00 | 1442.00 | 2024-06-19 10:40:00 | 1446.96 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-21 10:00:00 | 1431.85 | 2024-06-21 10:25:00 | 1425.48 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-21 10:00:00 | 1431.85 | 2024-06-21 15:20:00 | 1398.95 | TARGET_HIT | 0.50 | 2.30% |
| BUY | retest1 | 2024-06-26 11:05:00 | 1447.15 | 2024-06-26 11:35:00 | 1442.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-27 10:20:00 | 1439.80 | 2024-06-27 10:30:00 | 1434.47 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-05 11:15:00 | 1409.00 | 2024-07-05 11:25:00 | 1411.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-08 10:40:00 | 1418.00 | 2024-07-08 11:05:00 | 1412.26 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-08 10:40:00 | 1418.00 | 2024-07-08 15:20:00 | 1406.75 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2024-07-09 09:45:00 | 1404.15 | 2024-07-09 10:15:00 | 1409.14 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-11 10:05:00 | 1427.90 | 2024-07-11 10:15:00 | 1423.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-15 10:15:00 | 1394.35 | 2024-07-15 10:25:00 | 1389.05 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-18 10:15:00 | 1415.10 | 2024-07-18 10:25:00 | 1422.66 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-18 10:15:00 | 1415.10 | 2024-07-18 15:20:00 | 1451.15 | TARGET_HIT | 0.50 | 2.55% |
| SELL | retest1 | 2024-07-25 09:40:00 | 1359.00 | 2024-07-25 10:10:00 | 1350.48 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-25 09:40:00 | 1359.00 | 2024-07-25 12:00:00 | 1354.55 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-02 11:15:00 | 1404.60 | 2024-08-02 12:00:00 | 1400.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-07 10:45:00 | 1375.70 | 2024-08-07 11:05:00 | 1384.21 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-08-07 10:45:00 | 1375.70 | 2024-08-07 11:15:00 | 1375.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 10:55:00 | 1349.35 | 2024-08-12 15:00:00 | 1345.03 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-14 10:15:00 | 1332.10 | 2024-08-14 11:05:00 | 1325.97 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-23 11:00:00 | 1351.95 | 2024-08-23 11:10:00 | 1355.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-26 09:55:00 | 1373.60 | 2024-08-26 10:05:00 | 1379.62 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-26 09:55:00 | 1373.60 | 2024-08-26 10:40:00 | 1375.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-09-02 10:00:00 | 1473.90 | 2024-09-02 10:10:00 | 1482.55 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-09-02 10:00:00 | 1473.90 | 2024-09-02 10:25:00 | 1473.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:10:00 | 1590.05 | 2024-09-13 10:45:00 | 1585.09 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-17 09:35:00 | 1572.00 | 2024-09-17 10:00:00 | 1579.77 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-17 09:35:00 | 1572.00 | 2024-09-17 10:10:00 | 1572.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:30:00 | 1610.00 | 2024-09-19 09:50:00 | 1604.79 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-20 10:30:00 | 1592.70 | 2024-09-20 10:50:00 | 1586.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-20 10:30:00 | 1592.70 | 2024-09-20 12:05:00 | 1592.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 11:00:00 | 1602.25 | 2024-09-25 11:50:00 | 1595.29 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-25 11:00:00 | 1602.25 | 2024-09-25 12:45:00 | 1602.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:30:00 | 1629.25 | 2024-09-26 10:35:00 | 1625.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-27 11:15:00 | 1608.75 | 2024-09-27 11:30:00 | 1603.19 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-27 11:15:00 | 1608.75 | 2024-09-27 15:00:00 | 1608.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 10:50:00 | 1602.50 | 2024-10-01 10:55:00 | 1607.21 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-04 10:45:00 | 1509.85 | 2024-10-04 11:15:00 | 1515.66 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-07 10:35:00 | 1474.00 | 2024-10-07 11:05:00 | 1479.49 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-09 10:05:00 | 1579.95 | 2024-10-09 11:55:00 | 1573.08 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-11 10:35:00 | 1483.65 | 2024-10-11 10:40:00 | 1488.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-15 10:15:00 | 1498.30 | 2024-10-15 10:40:00 | 1503.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-22 09:35:00 | 1448.65 | 2024-10-22 09:55:00 | 1456.96 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-22 09:35:00 | 1448.65 | 2024-10-22 10:05:00 | 1448.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 10:35:00 | 1238.90 | 2024-11-05 10:50:00 | 1243.84 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-11 10:55:00 | 1270.50 | 2024-11-11 11:05:00 | 1266.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-12 11:15:00 | 1258.70 | 2024-11-12 11:20:00 | 1262.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-26 10:55:00 | 1252.80 | 2024-11-26 11:30:00 | 1258.70 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-11-26 10:55:00 | 1252.80 | 2024-11-26 12:05:00 | 1252.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 10:00:00 | 1280.85 | 2024-11-28 10:20:00 | 1277.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1281.85 | 2024-12-05 11:20:00 | 1286.14 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-06 10:05:00 | 1278.70 | 2024-12-06 10:15:00 | 1271.94 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-06 10:05:00 | 1278.70 | 2024-12-06 10:20:00 | 1278.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:45:00 | 1338.50 | 2024-12-12 09:50:00 | 1342.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1220.85 | 2024-12-20 09:40:00 | 1226.47 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1220.85 | 2024-12-20 09:45:00 | 1220.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:10:00 | 1182.95 | 2024-12-24 10:45:00 | 1188.15 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-24 10:10:00 | 1182.95 | 2024-12-24 10:55:00 | 1182.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 09:40:00 | 1191.90 | 2024-12-27 09:55:00 | 1196.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-30 10:10:00 | 1208.75 | 2024-12-30 11:15:00 | 1213.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-01 10:50:00 | 1181.20 | 2025-01-01 11:15:00 | 1184.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-07 10:30:00 | 1283.90 | 2025-01-07 10:55:00 | 1288.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1268.40 | 2025-01-09 11:35:00 | 1272.84 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-22 09:45:00 | 1241.50 | 2025-01-22 10:10:00 | 1233.94 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-22 09:45:00 | 1241.50 | 2025-01-22 10:35:00 | 1241.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-23 10:45:00 | 1237.10 | 2025-01-23 10:55:00 | 1230.13 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-23 10:45:00 | 1237.10 | 2025-01-23 11:55:00 | 1237.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 10:35:00 | 1231.80 | 2025-01-24 11:40:00 | 1235.94 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-29 10:15:00 | 1280.50 | 2025-01-29 10:45:00 | 1288.80 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-29 10:15:00 | 1280.50 | 2025-01-29 15:20:00 | 1295.55 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2025-01-31 09:40:00 | 1287.95 | 2025-01-31 09:45:00 | 1282.61 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-04 10:25:00 | 1370.20 | 2025-02-04 11:20:00 | 1378.69 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-02-04 10:25:00 | 1370.20 | 2025-02-04 15:20:00 | 1388.80 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2025-02-06 10:30:00 | 1399.60 | 2025-02-06 10:45:00 | 1394.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-07 10:10:00 | 1382.25 | 2025-02-07 10:15:00 | 1373.02 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-02-07 10:10:00 | 1382.25 | 2025-02-07 10:55:00 | 1379.85 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-02-19 09:45:00 | 1362.10 | 2025-02-19 09:50:00 | 1357.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-20 11:00:00 | 1380.10 | 2025-02-20 11:05:00 | 1385.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-02-20 11:00:00 | 1380.10 | 2025-02-20 11:45:00 | 1380.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:40:00 | 1393.85 | 2025-02-25 09:45:00 | 1403.64 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-02-25 09:40:00 | 1393.85 | 2025-02-25 09:55:00 | 1393.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 10:40:00 | 1474.00 | 2025-03-07 10:45:00 | 1469.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-13 10:15:00 | 1465.95 | 2025-03-13 10:20:00 | 1461.37 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-18 10:10:00 | 1475.40 | 2025-03-18 10:55:00 | 1471.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-24 10:40:00 | 1565.00 | 2025-03-24 10:50:00 | 1558.11 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-03-26 09:50:00 | 1515.75 | 2025-03-26 09:55:00 | 1522.16 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-28 09:30:00 | 1511.40 | 2025-03-28 09:45:00 | 1516.96 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-21 09:40:00 | 1623.20 | 2025-04-21 10:10:00 | 1632.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-21 09:40:00 | 1623.20 | 2025-04-21 15:20:00 | 1660.00 | TARGET_HIT | 0.50 | 2.27% |
| BUY | retest1 | 2025-04-30 09:40:00 | 1487.20 | 2025-04-30 09:45:00 | 1495.64 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-04-30 09:40:00 | 1487.20 | 2025-04-30 10:10:00 | 1487.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:40:00 | 1506.20 | 2025-05-05 09:55:00 | 1514.69 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-05 09:40:00 | 1506.20 | 2025-05-05 15:20:00 | 1555.60 | TARGET_HIT | 0.50 | 3.28% |
