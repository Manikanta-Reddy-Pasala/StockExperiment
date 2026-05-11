# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1173.50
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
| PARTIAL | 26 |
| TARGET_HIT | 15 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 50
- **Target hits / Stop hits / Partials:** 15 / 50 / 26
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 17.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 19 | 41.3% | 7 | 27 | 12 | 0.20% | 9.3% |
| BUY @ 2nd Alert (retest1) | 46 | 19 | 41.3% | 7 | 27 | 12 | 0.20% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 22 | 48.9% | 8 | 23 | 14 | 0.18% | 8.1% |
| SELL @ 2nd Alert (retest1) | 45 | 22 | 48.9% | 8 | 23 | 14 | 0.18% | 8.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 91 | 41 | 45.1% | 15 | 50 | 26 | 0.19% | 17.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 10:50:00 | 1401.00 | 1408.81 | 0.00 | ORB-short ORB[1407.40,1420.00] vol=1.7x ATR=4.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 12:30:00 | 1394.14 | 1405.52 | 0.00 | T1 1.5R @ 1394.14 |
| Target hit | 2025-05-13 15:20:00 | 1389.40 | 1397.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1368.60 | 1375.64 | 0.00 | ORB-short ORB[1371.10,1388.00] vol=1.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 1372.54 | 1375.09 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:45:00 | 1384.90 | 1378.87 | 0.00 | ORB-long ORB[1372.30,1384.40] vol=1.7x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-05-16 10:50:00 | 1380.18 | 1382.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 1360.90 | 1365.37 | 0.00 | ORB-short ORB[1362.30,1378.00] vol=2.4x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-05-22 12:05:00 | 1363.87 | 1364.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 09:55:00 | 1345.10 | 1351.40 | 0.00 | ORB-short ORB[1350.30,1361.80] vol=1.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:10:00 | 1340.65 | 1349.09 | 0.00 | T1 1.5R @ 1340.65 |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 1345.10 | 1348.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1323.80 | 1327.07 | 0.00 | ORB-short ORB[1325.00,1341.60] vol=4.5x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:05:00 | 1318.71 | 1324.16 | 0.00 | T1 1.5R @ 1318.71 |
| Stop hit — per-position SL triggered | 2025-05-27 10:25:00 | 1323.80 | 1323.83 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:05:00 | 1352.70 | 1344.35 | 0.00 | ORB-long ORB[1331.60,1346.30] vol=2.5x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:20:00 | 1358.80 | 1348.56 | 0.00 | T1 1.5R @ 1358.80 |
| Stop hit — per-position SL triggered | 2025-05-28 10:50:00 | 1352.70 | 1353.48 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:40:00 | 1371.60 | 1364.67 | 0.00 | ORB-long ORB[1355.80,1369.30] vol=3.1x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:10:00 | 1378.77 | 1372.36 | 0.00 | T1 1.5R @ 1378.77 |
| Target hit | 2025-05-29 13:50:00 | 1383.00 | 1383.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1299.60 | 1305.76 | 0.00 | ORB-short ORB[1302.00,1319.00] vol=2.5x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-06-04 09:35:00 | 1303.17 | 1305.13 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:50:00 | 1321.60 | 1309.39 | 0.00 | ORB-long ORB[1295.20,1309.90] vol=2.0x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-06-05 11:45:00 | 1316.32 | 1311.02 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 1332.00 | 1325.57 | 0.00 | ORB-long ORB[1311.10,1329.00] vol=5.9x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-06-06 11:20:00 | 1327.81 | 1325.76 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 1412.00 | 1406.40 | 0.00 | ORB-long ORB[1393.20,1411.90] vol=1.8x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:40:00 | 1418.09 | 1411.06 | 0.00 | T1 1.5R @ 1418.09 |
| Target hit | 2025-06-11 09:50:00 | 1413.60 | 1414.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2025-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1400.50 | 1409.17 | 0.00 | ORB-short ORB[1406.20,1420.00] vol=2.3x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:00:00 | 1393.58 | 1403.70 | 0.00 | T1 1.5R @ 1393.58 |
| Stop hit — per-position SL triggered | 2025-06-12 10:30:00 | 1400.50 | 1402.18 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 1459.00 | 1453.19 | 0.00 | ORB-long ORB[1439.70,1454.90] vol=3.9x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:00:00 | 1466.82 | 1457.67 | 0.00 | T1 1.5R @ 1466.82 |
| Target hit | 2025-06-27 14:30:00 | 1501.50 | 1502.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 1524.00 | 1516.32 | 0.00 | ORB-long ORB[1505.00,1521.90] vol=3.2x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-07-04 09:35:00 | 1518.78 | 1518.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 11:10:00 | 1470.50 | 1477.04 | 0.00 | ORB-short ORB[1473.00,1484.90] vol=5.0x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:15:00 | 1465.10 | 1475.56 | 0.00 | T1 1.5R @ 1465.10 |
| Stop hit — per-position SL triggered | 2025-07-14 11:20:00 | 1470.50 | 1474.24 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:10:00 | 1486.60 | 1479.37 | 0.00 | ORB-long ORB[1473.10,1483.00] vol=2.3x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-07-16 10:20:00 | 1483.25 | 1480.16 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 1498.10 | 1510.29 | 0.00 | ORB-short ORB[1506.00,1522.30] vol=1.9x ATR=3.90 |
| Stop hit — per-position SL triggered | 2025-07-22 10:05:00 | 1502.00 | 1504.99 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:55:00 | 1379.80 | 1392.34 | 0.00 | ORB-short ORB[1388.30,1405.90] vol=2.3x ATR=5.71 |
| Stop hit — per-position SL triggered | 2025-07-31 10:25:00 | 1385.51 | 1388.95 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 1318.00 | 1330.20 | 0.00 | ORB-short ORB[1329.40,1344.10] vol=3.3x ATR=5.39 |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1323.39 | 1324.50 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:00:00 | 1275.00 | 1266.35 | 0.00 | ORB-long ORB[1258.10,1274.00] vol=3.4x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-09-02 10:10:00 | 1271.60 | 1266.92 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 1270.60 | 1283.82 | 0.00 | ORB-short ORB[1284.30,1296.70] vol=1.5x ATR=4.40 |
| Stop hit — per-position SL triggered | 2025-09-05 11:55:00 | 1275.00 | 1279.63 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 1283.80 | 1280.22 | 0.00 | ORB-long ORB[1274.20,1282.20] vol=2.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-09-10 10:00:00 | 1280.39 | 1281.27 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:25:00 | 1296.30 | 1300.71 | 0.00 | ORB-short ORB[1296.90,1305.90] vol=1.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-09-12 10:50:00 | 1299.17 | 1300.17 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:45:00 | 1317.30 | 1311.27 | 0.00 | ORB-long ORB[1299.30,1314.00] vol=3.9x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:55:00 | 1322.13 | 1315.06 | 0.00 | T1 1.5R @ 1322.13 |
| Target hit | 2025-09-16 15:20:00 | 1365.00 | 1349.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 1352.90 | 1366.35 | 0.00 | ORB-short ORB[1366.00,1378.00] vol=2.0x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-09-17 11:45:00 | 1356.81 | 1365.58 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:45:00 | 1350.80 | 1359.27 | 0.00 | ORB-short ORB[1359.00,1374.40] vol=3.6x ATR=5.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:30:00 | 1343.17 | 1353.26 | 0.00 | T1 1.5R @ 1343.17 |
| Target hit | 2025-09-18 15:20:00 | 1340.40 | 1347.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-09-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:50:00 | 1288.30 | 1301.90 | 0.00 | ORB-short ORB[1306.50,1312.00] vol=3.3x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 12:50:00 | 1282.70 | 1293.11 | 0.00 | T1 1.5R @ 1282.70 |
| Target hit | 2025-09-24 15:20:00 | 1279.60 | 1287.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 1251.40 | 1251.12 | 0.00 | ORB-long ORB[1236.90,1251.10] vol=2.6x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 1255.90 | 1251.70 | 0.00 | T1 1.5R @ 1255.90 |
| Target hit | 2025-09-29 15:20:00 | 1261.00 | 1255.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:40:00 | 1275.40 | 1270.21 | 0.00 | ORB-long ORB[1254.80,1273.00] vol=2.0x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:00:00 | 1279.81 | 1272.22 | 0.00 | T1 1.5R @ 1279.81 |
| Target hit | 2025-09-30 15:20:00 | 1295.10 | 1284.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 1306.20 | 1297.68 | 0.00 | ORB-long ORB[1291.30,1305.00] vol=1.6x ATR=3.43 |
| Stop hit — per-position SL triggered | 2025-10-01 11:25:00 | 1302.77 | 1298.84 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:35:00 | 1277.20 | 1280.33 | 0.00 | ORB-short ORB[1278.10,1289.40] vol=2.0x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-10-03 14:25:00 | 1280.87 | 1278.56 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:45:00 | 1288.80 | 1294.36 | 0.00 | ORB-short ORB[1292.20,1306.50] vol=2.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-10-17 11:40:00 | 1292.40 | 1291.50 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:35:00 | 1294.60 | 1302.35 | 0.00 | ORB-short ORB[1303.40,1314.00] vol=1.8x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:00:00 | 1290.69 | 1298.23 | 0.00 | T1 1.5R @ 1290.69 |
| Stop hit — per-position SL triggered | 2025-10-27 11:05:00 | 1294.60 | 1298.13 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:40:00 | 1295.20 | 1291.12 | 0.00 | ORB-long ORB[1284.10,1289.90] vol=3.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2025-10-28 10:50:00 | 1292.73 | 1291.18 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:10:00 | 1291.10 | 1293.47 | 0.00 | ORB-short ORB[1291.70,1300.90] vol=2.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-10-29 11:20:00 | 1293.55 | 1293.40 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:30:00 | 1273.10 | 1278.59 | 0.00 | ORB-short ORB[1276.60,1281.50] vol=2.5x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:25:00 | 1269.15 | 1275.18 | 0.00 | T1 1.5R @ 1269.15 |
| Target hit | 2025-11-04 13:45:00 | 1270.10 | 1270.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2025-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:35:00 | 1247.60 | 1254.41 | 0.00 | ORB-short ORB[1251.60,1267.00] vol=2.5x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:50:00 | 1244.04 | 1253.19 | 0.00 | T1 1.5R @ 1244.04 |
| Target hit | 2025-11-06 15:20:00 | 1229.60 | 1239.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-11-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:05:00 | 1239.80 | 1234.01 | 0.00 | ORB-long ORB[1223.20,1235.70] vol=2.2x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-11-12 11:20:00 | 1236.20 | 1234.25 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 1239.10 | 1236.08 | 0.00 | ORB-long ORB[1227.10,1237.60] vol=1.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:55:00 | 1243.63 | 1240.88 | 0.00 | T1 1.5R @ 1243.63 |
| Target hit | 2025-11-14 15:20:00 | 1254.70 | 1249.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-11-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:40:00 | 1234.00 | 1230.79 | 0.00 | ORB-long ORB[1212.70,1229.90] vol=8.2x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-11-24 09:45:00 | 1231.20 | 1230.81 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:15:00 | 1197.60 | 1206.87 | 0.00 | ORB-short ORB[1210.00,1219.90] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-11-25 10:55:00 | 1201.84 | 1203.88 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:30:00 | 1221.60 | 1217.21 | 0.00 | ORB-long ORB[1206.20,1219.20] vol=1.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-11-26 10:40:00 | 1218.79 | 1217.46 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:55:00 | 1216.80 | 1221.62 | 0.00 | ORB-short ORB[1221.10,1228.70] vol=2.4x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-11-27 10:20:00 | 1219.65 | 1220.16 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:30:00 | 1154.90 | 1159.14 | 0.00 | ORB-short ORB[1156.50,1170.00] vol=2.3x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:35:00 | 1150.87 | 1156.96 | 0.00 | T1 1.5R @ 1150.87 |
| Target hit | 2025-12-08 15:20:00 | 1123.60 | 1137.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:40:00 | 1129.00 | 1122.52 | 0.00 | ORB-long ORB[1112.30,1122.60] vol=2.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 1125.74 | 1122.97 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:10:00 | 1110.70 | 1117.40 | 0.00 | ORB-short ORB[1113.30,1127.50] vol=1.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-12-16 10:20:00 | 1113.18 | 1116.66 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:35:00 | 1149.30 | 1155.15 | 0.00 | ORB-short ORB[1153.00,1169.80] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:00:00 | 1145.40 | 1153.64 | 0.00 | T1 1.5R @ 1145.40 |
| Stop hit — per-position SL triggered | 2025-12-23 11:10:00 | 1149.30 | 1153.47 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:20:00 | 1128.00 | 1131.34 | 0.00 | ORB-short ORB[1131.00,1138.70] vol=2.5x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:20:00 | 1124.45 | 1129.83 | 0.00 | T1 1.5R @ 1124.45 |
| Target hit | 2025-12-29 15:20:00 | 1122.10 | 1123.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 1131.70 | 1136.33 | 0.00 | ORB-short ORB[1132.90,1141.80] vol=4.1x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 12:20:00 | 1126.82 | 1133.73 | 0.00 | T1 1.5R @ 1126.82 |
| Target hit | 2026-01-01 15:20:00 | 1128.40 | 1130.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 1141.20 | 1130.30 | 0.00 | ORB-long ORB[1123.00,1133.20] vol=3.2x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:00:00 | 1146.03 | 1132.30 | 0.00 | T1 1.5R @ 1146.03 |
| Stop hit — per-position SL triggered | 2026-01-02 12:25:00 | 1141.20 | 1140.03 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:20:00 | 1061.30 | 1055.28 | 0.00 | ORB-long ORB[1047.70,1057.60] vol=1.6x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-01-30 11:10:00 | 1057.93 | 1056.35 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:15:00 | 1063.00 | 1067.24 | 0.00 | ORB-short ORB[1064.70,1077.50] vol=1.6x ATR=2.17 |
| Stop hit — per-position SL triggered | 2026-02-05 11:25:00 | 1065.17 | 1067.20 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:30:00 | 1120.80 | 1111.27 | 0.00 | ORB-long ORB[1105.00,1119.10] vol=3.2x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-18 10:40:00 | 1117.38 | 1111.91 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 1133.80 | 1129.37 | 0.00 | ORB-long ORB[1119.50,1130.90] vol=5.5x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:50:00 | 1138.44 | 1132.10 | 0.00 | T1 1.5R @ 1138.44 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 1133.80 | 1133.47 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 1201.20 | 1194.08 | 0.00 | ORB-long ORB[1188.00,1198.90] vol=1.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 1197.26 | 1194.54 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1209.80 | 1202.99 | 0.00 | ORB-long ORB[1195.70,1204.40] vol=2.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2026-02-27 09:35:00 | 1207.18 | 1206.05 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:55:00 | 988.90 | 995.45 | 0.00 | ORB-short ORB[992.80,1006.20] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2026-03-20 11:05:00 | 991.76 | 994.97 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 1153.25 | 1147.13 | 0.00 | ORB-long ORB[1132.80,1148.50] vol=1.5x ATR=3.40 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 1149.85 | 1148.72 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 1141.95 | 1131.98 | 0.00 | ORB-long ORB[1126.10,1139.75] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2026-04-22 10:05:00 | 1136.62 | 1137.06 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 1145.00 | 1132.04 | 0.00 | ORB-long ORB[1127.00,1137.45] vol=1.7x ATR=5.33 |
| Stop hit — per-position SL triggered | 2026-04-27 13:30:00 | 1139.67 | 1141.82 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 1146.80 | 1142.34 | 0.00 | ORB-long ORB[1134.20,1146.40] vol=3.0x ATR=4.70 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 1142.10 | 1143.34 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 1137.80 | 1127.53 | 0.00 | ORB-long ORB[1121.10,1132.10] vol=1.7x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 1133.28 | 1129.13 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1152.00 | 1146.15 | 0.00 | ORB-long ORB[1141.00,1149.50] vol=4.3x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 1156.73 | 1148.44 | 0.00 | T1 1.5R @ 1156.73 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1152.00 | 1149.85 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 1177.90 | 1173.70 | 0.00 | ORB-long ORB[1166.00,1177.20] vol=1.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:05:00 | 1182.99 | 1175.07 | 0.00 | T1 1.5R @ 1182.99 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1177.90 | 1175.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-13 10:50:00 | 1401.00 | 2025-05-13 12:30:00 | 1394.14 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-13 10:50:00 | 1401.00 | 2025-05-13 15:20:00 | 1389.40 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-05-15 09:30:00 | 1368.60 | 2025-05-15 09:35:00 | 1372.54 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-16 09:45:00 | 1384.90 | 2025-05-16 10:50:00 | 1380.18 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-05-22 11:15:00 | 1360.90 | 2025-05-22 12:05:00 | 1363.87 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-26 09:55:00 | 1345.10 | 2025-05-26 10:10:00 | 1340.65 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-05-26 09:55:00 | 1345.10 | 2025-05-26 10:15:00 | 1345.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 09:45:00 | 1323.80 | 2025-05-27 10:05:00 | 1318.71 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-27 09:45:00 | 1323.80 | 2025-05-27 10:25:00 | 1323.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 10:05:00 | 1352.70 | 2025-05-28 10:20:00 | 1358.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-28 10:05:00 | 1352.70 | 2025-05-28 10:50:00 | 1352.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-29 09:40:00 | 1371.60 | 2025-05-29 10:10:00 | 1378.77 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-05-29 09:40:00 | 1371.60 | 2025-05-29 13:50:00 | 1383.00 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-06-04 09:30:00 | 1299.60 | 2025-06-04 09:35:00 | 1303.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-05 10:50:00 | 1321.60 | 2025-06-05 11:45:00 | 1316.32 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-06 11:15:00 | 1332.00 | 2025-06-06 11:20:00 | 1327.81 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-11 09:30:00 | 1412.00 | 2025-06-11 09:40:00 | 1418.09 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-06-11 09:30:00 | 1412.00 | 2025-06-11 09:50:00 | 1413.60 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-06-12 09:30:00 | 1400.50 | 2025-06-12 10:00:00 | 1393.58 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-12 09:30:00 | 1400.50 | 2025-06-12 10:30:00 | 1400.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:45:00 | 1459.00 | 2025-06-27 10:00:00 | 1466.82 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-27 09:45:00 | 1459.00 | 2025-06-27 14:30:00 | 1501.50 | TARGET_HIT | 0.50 | 2.91% |
| BUY | retest1 | 2025-07-04 09:30:00 | 1524.00 | 2025-07-04 09:35:00 | 1518.78 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-14 11:10:00 | 1470.50 | 2025-07-14 11:15:00 | 1465.10 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-14 11:10:00 | 1470.50 | 2025-07-14 11:20:00 | 1470.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 10:10:00 | 1486.60 | 2025-07-16 10:20:00 | 1483.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-22 09:30:00 | 1498.10 | 2025-07-22 10:05:00 | 1502.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-31 09:55:00 | 1379.80 | 2025-07-31 10:25:00 | 1385.51 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-14 09:30:00 | 1318.00 | 2025-08-14 10:15:00 | 1323.39 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-09-02 10:00:00 | 1275.00 | 2025-09-02 10:10:00 | 1271.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-05 10:15:00 | 1270.60 | 2025-09-05 11:55:00 | 1275.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-10 09:40:00 | 1283.80 | 2025-09-10 10:00:00 | 1280.39 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-12 10:25:00 | 1296.30 | 2025-09-12 10:50:00 | 1299.17 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-16 09:45:00 | 1317.30 | 2025-09-16 09:55:00 | 1322.13 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-16 09:45:00 | 1317.30 | 2025-09-16 15:20:00 | 1365.00 | TARGET_HIT | 0.50 | 3.62% |
| SELL | retest1 | 2025-09-17 11:10:00 | 1352.90 | 2025-09-17 11:45:00 | 1356.81 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-18 09:45:00 | 1350.80 | 2025-09-18 11:30:00 | 1343.17 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-09-18 09:45:00 | 1350.80 | 2025-09-18 15:20:00 | 1340.40 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-09-24 10:50:00 | 1288.30 | 2025-09-24 12:50:00 | 1282.70 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-24 10:50:00 | 1288.30 | 2025-09-24 15:20:00 | 1279.60 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1251.40 | 2025-09-29 12:15:00 | 1255.90 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1251.40 | 2025-09-29 15:20:00 | 1261.00 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-09-30 10:40:00 | 1275.40 | 2025-09-30 11:00:00 | 1279.81 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-30 10:40:00 | 1275.40 | 2025-09-30 15:20:00 | 1295.10 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2025-10-01 11:15:00 | 1306.20 | 2025-10-01 11:25:00 | 1302.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-03 10:35:00 | 1277.20 | 2025-10-03 14:25:00 | 1280.87 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-17 09:45:00 | 1288.80 | 2025-10-17 11:40:00 | 1292.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-27 10:35:00 | 1294.60 | 2025-10-27 11:00:00 | 1290.69 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-27 10:35:00 | 1294.60 | 2025-10-27 11:05:00 | 1294.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 10:40:00 | 1295.20 | 2025-10-28 10:50:00 | 1292.73 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-29 11:10:00 | 1291.10 | 2025-10-29 11:20:00 | 1293.55 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-04 10:30:00 | 1273.10 | 2025-11-04 11:25:00 | 1269.15 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-04 10:30:00 | 1273.10 | 2025-11-04 13:45:00 | 1270.10 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-06 10:35:00 | 1247.60 | 2025-11-06 10:50:00 | 1244.04 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-06 10:35:00 | 1247.60 | 2025-11-06 15:20:00 | 1229.60 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2025-11-12 11:05:00 | 1239.80 | 2025-11-12 11:20:00 | 1236.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-14 09:30:00 | 1239.10 | 2025-11-14 09:55:00 | 1243.63 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-14 09:30:00 | 1239.10 | 2025-11-14 15:20:00 | 1254.70 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2025-11-24 09:40:00 | 1234.00 | 2025-11-24 09:45:00 | 1231.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-25 10:15:00 | 1197.60 | 2025-11-25 10:55:00 | 1201.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-26 10:30:00 | 1221.60 | 2025-11-26 10:40:00 | 1218.79 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-27 09:55:00 | 1216.80 | 2025-11-27 10:20:00 | 1219.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-08 09:30:00 | 1154.90 | 2025-12-08 09:35:00 | 1150.87 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-08 09:30:00 | 1154.90 | 2025-12-08 15:20:00 | 1123.60 | TARGET_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2025-12-11 10:40:00 | 1129.00 | 2025-12-11 10:55:00 | 1125.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-16 10:10:00 | 1110.70 | 2025-12-16 10:20:00 | 1113.18 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-23 10:35:00 | 1149.30 | 2025-12-23 11:00:00 | 1145.40 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-23 10:35:00 | 1149.30 | 2025-12-23 11:10:00 | 1149.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 10:20:00 | 1128.00 | 2025-12-29 11:20:00 | 1124.45 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-29 10:20:00 | 1128.00 | 2025-12-29 15:20:00 | 1122.10 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-01 11:00:00 | 1131.70 | 2026-01-01 12:20:00 | 1126.82 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-01 11:00:00 | 1131.70 | 2026-01-01 15:20:00 | 1128.40 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-02 10:40:00 | 1141.20 | 2026-01-02 11:00:00 | 1146.03 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-02 10:40:00 | 1141.20 | 2026-01-02 12:25:00 | 1141.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 10:20:00 | 1061.30 | 2026-01-30 11:10:00 | 1057.93 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-05 11:15:00 | 1063.00 | 2026-02-05 11:25:00 | 1065.17 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 10:30:00 | 1120.80 | 2026-02-18 10:40:00 | 1117.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-19 10:15:00 | 1133.80 | 2026-02-19 10:50:00 | 1138.44 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-19 10:15:00 | 1133.80 | 2026-02-19 11:00:00 | 1133.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 1201.20 | 2026-02-25 09:40:00 | 1197.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1209.80 | 2026-02-27 09:35:00 | 1207.18 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-20 10:55:00 | 988.90 | 2026-03-20 11:05:00 | 991.76 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 10:05:00 | 1153.25 | 2026-04-21 10:40:00 | 1149.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 09:40:00 | 1141.95 | 2026-04-22 10:05:00 | 1136.62 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1145.00 | 2026-04-27 13:30:00 | 1139.67 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-05-04 09:40:00 | 1146.80 | 2026-05-04 10:20:00 | 1142.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-05 09:40:00 | 1137.80 | 2026-05-05 09:50:00 | 1133.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1152.00 | 2026-05-06 09:45:00 | 1156.73 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1152.00 | 2026-05-06 10:15:00 | 1152.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1177.90 | 2026-05-07 11:05:00 | 1182.99 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1177.90 | 2026-05-07 11:15:00 | 1177.90 | STOP_HIT | 0.50 | 0.00% |
