# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-01 15:25:00 (25983 bars)
- **Last close:** 1185.70
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 10 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 52
- **Target hits / Stop hits / Partials:** 10 / 52 / 23
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 9.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 4 | 16 | 9 | 0.23% | 6.6% |
| BUY @ 2nd Alert (retest1) | 29 | 13 | 44.8% | 4 | 16 | 9 | 0.23% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 20 | 35.7% | 6 | 36 | 14 | 0.05% | 2.8% |
| SELL @ 2nd Alert (retest1) | 56 | 20 | 35.7% | 6 | 36 | 14 | 0.05% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 85 | 33 | 38.8% | 10 | 52 | 23 | 0.11% | 9.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:55:00 | 1172.00 | 1165.92 | 0.00 | ORB-long ORB[1157.00,1171.40] vol=3.4x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 12:20:00 | 1176.40 | 1168.16 | 0.00 | T1 1.5R @ 1176.40 |
| Target hit | 2024-05-14 15:20:00 | 1191.80 | 1177.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:10:00 | 1221.00 | 1216.68 | 0.00 | ORB-long ORB[1207.00,1219.80] vol=4.1x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:15:00 | 1227.38 | 1220.70 | 0.00 | T1 1.5R @ 1227.38 |
| Stop hit — per-position SL triggered | 2024-05-16 10:25:00 | 1221.00 | 1221.11 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 11:15:00 | 1315.00 | 1306.22 | 0.00 | ORB-long ORB[1286.70,1294.00] vol=2.6x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-05-22 11:25:00 | 1309.67 | 1310.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:00:00 | 1263.65 | 1265.55 | 0.00 | ORB-short ORB[1264.85,1274.65] vol=2.1x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 1267.67 | 1265.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:35:00 | 1305.75 | 1295.53 | 0.00 | ORB-long ORB[1287.05,1304.35] vol=1.9x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 11:55:00 | 1312.99 | 1304.63 | 0.00 | T1 1.5R @ 1312.99 |
| Stop hit — per-position SL triggered | 2024-05-27 12:35:00 | 1305.75 | 1305.03 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:00:00 | 1292.85 | 1298.62 | 0.00 | ORB-short ORB[1295.10,1314.30] vol=4.7x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-05-28 10:05:00 | 1298.03 | 1299.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:35:00 | 1279.30 | 1266.34 | 0.00 | ORB-long ORB[1255.00,1272.45] vol=2.2x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-05-29 09:40:00 | 1273.81 | 1266.58 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:45:00 | 1239.20 | 1242.88 | 0.00 | ORB-short ORB[1243.10,1255.00] vol=1.6x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-06-11 10:50:00 | 1242.89 | 1242.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:10:00 | 1354.90 | 1344.62 | 0.00 | ORB-long ORB[1330.00,1350.00] vol=4.4x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-06-20 11:35:00 | 1349.91 | 1346.86 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:20:00 | 1387.95 | 1375.39 | 0.00 | ORB-long ORB[1361.70,1380.00] vol=3.7x ATR=5.89 |
| Stop hit — per-position SL triggered | 2024-06-24 10:30:00 | 1382.06 | 1378.42 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:50:00 | 1444.00 | 1433.55 | 0.00 | ORB-long ORB[1414.90,1435.70] vol=2.4x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:20:00 | 1453.13 | 1440.32 | 0.00 | T1 1.5R @ 1453.13 |
| Target hit | 2024-06-27 11:40:00 | 1455.85 | 1456.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:40:00 | 1494.45 | 1503.08 | 0.00 | ORB-short ORB[1496.95,1514.00] vol=2.0x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:35:00 | 1484.56 | 1497.83 | 0.00 | T1 1.5R @ 1484.56 |
| Stop hit — per-position SL triggered | 2024-07-05 10:50:00 | 1494.45 | 1497.15 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 1420.50 | 1439.09 | 0.00 | ORB-short ORB[1440.00,1447.80] vol=2.1x ATR=7.23 |
| Stop hit — per-position SL triggered | 2024-07-10 11:50:00 | 1427.73 | 1431.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:10:00 | 1415.30 | 1425.60 | 0.00 | ORB-short ORB[1416.25,1435.95] vol=1.6x ATR=6.22 |
| Stop hit — per-position SL triggered | 2024-07-11 10:30:00 | 1421.52 | 1423.57 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 1401.40 | 1405.34 | 0.00 | ORB-short ORB[1405.00,1415.20] vol=1.8x ATR=5.51 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 1406.91 | 1399.32 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1386.50 | 1380.56 | 0.00 | ORB-long ORB[1369.25,1383.60] vol=1.7x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-07-16 11:30:00 | 1382.62 | 1380.83 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:45:00 | 1450.10 | 1442.03 | 0.00 | ORB-long ORB[1423.15,1444.90] vol=2.2x ATR=8.25 |
| Stop hit — per-position SL triggered | 2024-07-25 14:05:00 | 1441.85 | 1448.95 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:45:00 | 1448.30 | 1453.57 | 0.00 | ORB-short ORB[1450.00,1460.00] vol=2.4x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-07-30 11:30:00 | 1451.65 | 1450.81 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 10:10:00 | 1387.40 | 1394.81 | 0.00 | ORB-short ORB[1393.65,1406.50] vol=2.2x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-08-07 10:25:00 | 1392.36 | 1394.26 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:10:00 | 1470.00 | 1457.17 | 0.00 | ORB-long ORB[1444.00,1463.70] vol=3.6x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:15:00 | 1477.96 | 1460.40 | 0.00 | T1 1.5R @ 1477.96 |
| Stop hit — per-position SL triggered | 2024-08-08 11:30:00 | 1470.00 | 1465.84 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1379.90 | 1400.96 | 0.00 | ORB-short ORB[1397.15,1413.45] vol=1.8x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 1384.07 | 1400.60 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:35:00 | 1359.95 | 1368.36 | 0.00 | ORB-short ORB[1370.30,1384.00] vol=3.4x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:40:00 | 1353.70 | 1362.79 | 0.00 | T1 1.5R @ 1353.70 |
| Target hit | 2024-08-20 14:40:00 | 1358.90 | 1356.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — SELL (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 1361.10 | 1367.05 | 0.00 | ORB-short ORB[1365.75,1382.75] vol=2.5x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-08-21 10:25:00 | 1365.55 | 1366.04 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1345.00 | 1347.87 | 0.00 | ORB-short ORB[1347.00,1362.05] vol=1.8x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-08-23 10:25:00 | 1348.61 | 1347.30 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1347.10 | 1354.24 | 0.00 | ORB-short ORB[1352.55,1365.00] vol=5.0x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-08-28 11:25:00 | 1351.02 | 1350.97 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:00:00 | 1348.75 | 1346.40 | 0.00 | ORB-long ORB[1341.70,1348.60] vol=2.8x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:15:00 | 1352.83 | 1347.08 | 0.00 | T1 1.5R @ 1352.83 |
| Stop hit — per-position SL triggered | 2024-08-29 11:25:00 | 1348.75 | 1347.63 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:30:00 | 1446.10 | 1437.91 | 0.00 | ORB-long ORB[1421.00,1437.45] vol=13.3x ATR=7.29 |
| Stop hit — per-position SL triggered | 2024-09-09 10:50:00 | 1438.81 | 1438.07 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:15:00 | 1448.50 | 1443.97 | 0.00 | ORB-long ORB[1432.00,1445.85] vol=4.1x ATR=5.00 |
| Stop hit — per-position SL triggered | 2024-09-12 11:00:00 | 1443.50 | 1445.27 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:40:00 | 1436.40 | 1449.35 | 0.00 | ORB-short ORB[1449.05,1460.85] vol=2.1x ATR=5.08 |
| Stop hit — per-position SL triggered | 2024-09-13 10:45:00 | 1441.48 | 1448.95 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 1493.55 | 1485.08 | 0.00 | ORB-long ORB[1475.00,1489.90] vol=1.9x ATR=5.94 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 1487.61 | 1486.06 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:55:00 | 1482.00 | 1485.54 | 0.00 | ORB-short ORB[1482.75,1500.00] vol=4.5x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:25:00 | 1475.15 | 1484.25 | 0.00 | T1 1.5R @ 1475.15 |
| Stop hit — per-position SL triggered | 2024-09-20 12:20:00 | 1482.00 | 1481.65 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 09:40:00 | 1405.00 | 1408.98 | 0.00 | ORB-short ORB[1407.25,1421.15] vol=5.3x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 1409.23 | 1406.48 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:10:00 | 1407.00 | 1391.55 | 0.00 | ORB-long ORB[1385.00,1403.25] vol=2.6x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 14:55:00 | 1414.18 | 1397.99 | 0.00 | T1 1.5R @ 1414.18 |
| Stop hit — per-position SL triggered | 2024-10-11 15:15:00 | 1407.00 | 1398.67 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 1393.55 | 1399.78 | 0.00 | ORB-short ORB[1395.05,1410.95] vol=2.0x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:00:00 | 1387.64 | 1398.50 | 0.00 | T1 1.5R @ 1387.64 |
| Stop hit — per-position SL triggered | 2024-10-15 10:20:00 | 1393.55 | 1393.84 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:55:00 | 1397.00 | 1398.04 | 0.00 | ORB-short ORB[1398.05,1412.10] vol=1.8x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 12:40:00 | 1389.62 | 1397.19 | 0.00 | T1 1.5R @ 1389.62 |
| Stop hit — per-position SL triggered | 2024-10-16 15:00:00 | 1397.00 | 1381.39 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:55:00 | 1368.75 | 1373.31 | 0.00 | ORB-short ORB[1375.60,1391.00] vol=3.6x ATR=5.35 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 1374.10 | 1373.60 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:50:00 | 1199.90 | 1208.24 | 0.00 | ORB-short ORB[1205.80,1220.35] vol=3.1x ATR=5.90 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 1205.80 | 1208.14 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:30:00 | 1213.00 | 1205.81 | 0.00 | ORB-long ORB[1200.00,1208.50] vol=1.7x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-10-30 09:35:00 | 1209.41 | 1206.49 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:20:00 | 1224.40 | 1230.01 | 0.00 | ORB-short ORB[1227.05,1243.00] vol=2.7x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:30:00 | 1218.77 | 1229.32 | 0.00 | T1 1.5R @ 1218.77 |
| Stop hit — per-position SL triggered | 2024-11-07 10:50:00 | 1224.40 | 1227.87 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 1157.80 | 1165.22 | 0.00 | ORB-short ORB[1161.05,1176.45] vol=2.6x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:55:00 | 1149.70 | 1159.27 | 0.00 | T1 1.5R @ 1149.70 |
| Target hit | 2024-11-13 15:20:00 | 1142.40 | 1147.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-11-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:55:00 | 1154.55 | 1146.22 | 0.00 | ORB-long ORB[1131.65,1146.00] vol=1.8x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:35:00 | 1160.98 | 1150.65 | 0.00 | T1 1.5R @ 1160.98 |
| Target hit | 2024-11-19 15:20:00 | 1165.45 | 1160.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:00:00 | 1218.95 | 1224.08 | 0.00 | ORB-short ORB[1222.50,1240.65] vol=4.9x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-11-27 10:15:00 | 1223.84 | 1224.27 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1205.00 | 1207.97 | 0.00 | ORB-short ORB[1205.75,1215.10] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-11-29 09:45:00 | 1208.69 | 1207.70 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1199.75 | 1206.90 | 0.00 | ORB-short ORB[1206.15,1219.00] vol=4.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-12-05 11:10:00 | 1202.16 | 1206.33 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 1173.60 | 1180.15 | 0.00 | ORB-short ORB[1178.45,1194.95] vol=1.7x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-12-09 11:05:00 | 1177.56 | 1179.87 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:05:00 | 1204.70 | 1205.32 | 0.00 | ORB-short ORB[1207.15,1217.00] vol=3.7x ATR=5.00 |
| Stop hit — per-position SL triggered | 2024-12-11 11:55:00 | 1209.70 | 1205.35 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 1149.55 | 1153.43 | 0.00 | ORB-short ORB[1151.15,1162.00] vol=2.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 1152.66 | 1151.49 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 1150.40 | 1153.67 | 0.00 | ORB-short ORB[1150.95,1159.60] vol=2.1x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 1154.03 | 1153.60 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 1120.90 | 1125.38 | 0.00 | ORB-short ORB[1122.05,1135.80] vol=2.7x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-12-24 09:45:00 | 1123.77 | 1124.86 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 1118.50 | 1124.42 | 0.00 | ORB-short ORB[1127.25,1138.85] vol=6.1x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:50:00 | 1114.29 | 1121.62 | 0.00 | T1 1.5R @ 1114.29 |
| Target hit | 2024-12-26 15:20:00 | 1110.00 | 1112.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-01-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:10:00 | 1140.00 | 1141.59 | 0.00 | ORB-short ORB[1143.50,1150.45] vol=3.4x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-01-03 11:25:00 | 1142.27 | 1141.68 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:50:00 | 1099.60 | 1111.30 | 0.00 | ORB-short ORB[1110.10,1124.60] vol=2.6x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 12:15:00 | 1093.84 | 1103.86 | 0.00 | T1 1.5R @ 1093.84 |
| Target hit | 2025-01-07 15:20:00 | 1086.85 | 1093.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-01-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:45:00 | 1072.90 | 1081.30 | 0.00 | ORB-short ORB[1078.65,1094.45] vol=3.4x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:05:00 | 1066.34 | 1078.56 | 0.00 | T1 1.5R @ 1066.34 |
| Stop hit — per-position SL triggered | 2025-01-10 10:10:00 | 1072.90 | 1078.25 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:45:00 | 1050.80 | 1053.29 | 0.00 | ORB-short ORB[1052.25,1060.40] vol=2.7x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 12:00:00 | 1047.77 | 1052.18 | 0.00 | T1 1.5R @ 1047.77 |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 1050.80 | 1051.44 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 990.45 | 996.94 | 0.00 | ORB-short ORB[990.55,1003.65] vol=5.2x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 993.57 | 996.56 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 10:40:00 | 873.30 | 871.08 | 0.00 | ORB-long ORB[862.15,871.45] vol=1.8x ATR=3.82 |
| Stop hit — per-position SL triggered | 2025-02-19 10:50:00 | 869.48 | 871.17 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:15:00 | 867.65 | 873.21 | 0.00 | ORB-short ORB[873.05,884.95] vol=1.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:10:00 | 864.68 | 871.55 | 0.00 | T1 1.5R @ 864.68 |
| Stop hit — per-position SL triggered | 2025-02-27 12:30:00 | 867.65 | 871.32 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 901.00 | 905.10 | 0.00 | ORB-short ORB[903.90,915.00] vol=2.7x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 09:50:00 | 896.17 | 903.58 | 0.00 | T1 1.5R @ 896.17 |
| Target hit | 2025-03-20 14:25:00 | 893.75 | 893.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — SELL (started 2025-04-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:20:00 | 799.40 | 805.00 | 0.00 | ORB-short ORB[803.00,814.00] vol=2.1x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-04-16 11:35:00 | 802.25 | 802.53 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 811.00 | 815.41 | 0.00 | ORB-short ORB[812.35,824.00] vol=2.0x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 807.04 | 813.37 | 0.00 | T1 1.5R @ 807.04 |
| Target hit | 2025-04-25 15:20:00 | 800.25 | 801.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-05-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 09:40:00 | 795.85 | 798.11 | 0.00 | ORB-short ORB[796.60,808.55] vol=5.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-05-02 09:55:00 | 799.81 | 797.63 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-05-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 10:35:00 | 831.00 | 824.52 | 0.00 | ORB-long ORB[810.00,821.00] vol=1.7x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 11:15:00 | 837.00 | 826.21 | 0.00 | T1 1.5R @ 837.00 |
| Target hit | 2025-05-09 15:20:00 | 855.00 | 840.15 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:55:00 | 1172.00 | 2024-05-14 12:20:00 | 1176.40 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-05-14 10:55:00 | 1172.00 | 2024-05-14 15:20:00 | 1191.80 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2024-05-16 10:10:00 | 1221.00 | 2024-05-16 10:15:00 | 1227.38 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-16 10:10:00 | 1221.00 | 2024-05-16 10:25:00 | 1221.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-22 11:15:00 | 1315.00 | 2024-05-22 11:25:00 | 1309.67 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-24 10:00:00 | 1263.65 | 2024-05-24 10:15:00 | 1267.67 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-27 10:35:00 | 1305.75 | 2024-05-27 11:55:00 | 1312.99 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-27 10:35:00 | 1305.75 | 2024-05-27 12:35:00 | 1305.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 10:00:00 | 1292.85 | 2024-05-28 10:05:00 | 1298.03 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-29 09:35:00 | 1279.30 | 2024-05-29 09:40:00 | 1273.81 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-06-11 10:45:00 | 1239.20 | 2024-06-11 10:50:00 | 1242.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-20 11:10:00 | 1354.90 | 2024-06-20 11:35:00 | 1349.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-24 10:20:00 | 1387.95 | 2024-06-24 10:30:00 | 1382.06 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-27 09:50:00 | 1444.00 | 2024-06-27 10:20:00 | 1453.13 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-27 09:50:00 | 1444.00 | 2024-06-27 11:40:00 | 1455.85 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2024-07-05 09:40:00 | 1494.45 | 2024-07-05 10:35:00 | 1484.56 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-05 09:40:00 | 1494.45 | 2024-07-05 10:50:00 | 1494.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:35:00 | 1420.50 | 2024-07-10 11:50:00 | 1427.73 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-07-11 10:10:00 | 1415.30 | 2024-07-11 10:30:00 | 1421.52 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-12 09:55:00 | 1401.40 | 2024-07-12 10:50:00 | 1406.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-16 11:15:00 | 1386.50 | 2024-07-16 11:30:00 | 1382.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-25 09:45:00 | 1450.10 | 2024-07-25 14:05:00 | 1441.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-07-30 09:45:00 | 1448.30 | 2024-07-30 11:30:00 | 1451.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-07 10:10:00 | 1387.40 | 2024-08-07 10:25:00 | 1392.36 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-08 11:10:00 | 1470.00 | 2024-08-08 11:15:00 | 1477.96 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-08 11:10:00 | 1470.00 | 2024-08-08 11:30:00 | 1470.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-13 11:00:00 | 1379.90 | 2024-08-13 11:15:00 | 1384.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-20 10:35:00 | 1359.95 | 2024-08-20 11:40:00 | 1353.70 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-20 10:35:00 | 1359.95 | 2024-08-20 14:40:00 | 1358.90 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-08-21 09:55:00 | 1361.10 | 2024-08-21 10:25:00 | 1365.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-23 09:45:00 | 1345.00 | 2024-08-23 10:25:00 | 1348.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1347.10 | 2024-08-28 11:25:00 | 1351.02 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-29 11:00:00 | 1348.75 | 2024-08-29 11:15:00 | 1352.83 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-08-29 11:00:00 | 1348.75 | 2024-08-29 11:25:00 | 1348.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-09 10:30:00 | 1446.10 | 2024-09-09 10:50:00 | 1438.81 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-09-12 10:15:00 | 1448.50 | 2024-09-12 11:00:00 | 1443.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-13 10:40:00 | 1436.40 | 2024-09-13 10:45:00 | 1441.48 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-19 09:45:00 | 1493.55 | 2024-09-19 09:50:00 | 1487.61 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-20 09:55:00 | 1482.00 | 2024-09-20 10:25:00 | 1475.15 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-20 09:55:00 | 1482.00 | 2024-09-20 12:20:00 | 1482.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-09 09:40:00 | 1405.00 | 2024-10-09 13:15:00 | 1409.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-11 10:10:00 | 1407.00 | 2024-10-11 14:55:00 | 1414.18 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-11 10:10:00 | 1407.00 | 2024-10-11 15:15:00 | 1407.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 09:50:00 | 1393.55 | 2024-10-15 10:00:00 | 1387.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-15 09:50:00 | 1393.55 | 2024-10-15 10:20:00 | 1393.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 09:55:00 | 1397.00 | 2024-10-16 12:40:00 | 1389.62 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-16 09:55:00 | 1397.00 | 2024-10-16 15:00:00 | 1397.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:55:00 | 1368.75 | 2024-10-21 10:00:00 | 1374.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-25 10:50:00 | 1199.90 | 2024-10-25 10:55:00 | 1205.80 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-10-30 09:30:00 | 1213.00 | 2024-10-30 09:35:00 | 1209.41 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-07 10:20:00 | 1224.40 | 2024-11-07 10:30:00 | 1218.77 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-07 10:20:00 | 1224.40 | 2024-11-07 10:50:00 | 1224.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1157.80 | 2024-11-13 09:55:00 | 1149.70 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1157.80 | 2024-11-13 15:20:00 | 1142.40 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2024-11-19 09:55:00 | 1154.55 | 2024-11-19 10:35:00 | 1160.98 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-19 09:55:00 | 1154.55 | 2024-11-19 15:20:00 | 1165.45 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-11-27 10:00:00 | 1218.95 | 2024-11-27 10:15:00 | 1223.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-29 09:30:00 | 1205.00 | 2024-11-29 09:45:00 | 1208.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1199.75 | 2024-12-05 11:10:00 | 1202.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-09 11:00:00 | 1173.60 | 2024-12-09 11:05:00 | 1177.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1204.70 | 2024-12-11 11:55:00 | 1209.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-12-17 09:35:00 | 1149.55 | 2024-12-17 10:15:00 | 1152.66 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-20 10:00:00 | 1150.40 | 2024-12-20 10:15:00 | 1154.03 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-24 09:30:00 | 1120.90 | 2024-12-24 09:45:00 | 1123.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-26 10:55:00 | 1118.50 | 2024-12-26 11:50:00 | 1114.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-26 10:55:00 | 1118.50 | 2024-12-26 15:20:00 | 1110.00 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2025-01-03 11:10:00 | 1140.00 | 2025-01-03 11:25:00 | 1142.27 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-07 10:50:00 | 1099.60 | 2025-01-07 12:15:00 | 1093.84 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-07 10:50:00 | 1099.60 | 2025-01-07 15:20:00 | 1086.85 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-01-10 09:45:00 | 1072.90 | 2025-01-10 10:05:00 | 1066.34 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-10 09:45:00 | 1072.90 | 2025-01-10 10:10:00 | 1072.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 10:45:00 | 1050.80 | 2025-01-20 12:00:00 | 1047.77 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-01-20 10:45:00 | 1050.80 | 2025-01-20 13:15:00 | 1050.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 11:00:00 | 990.45 | 2025-02-01 11:10:00 | 993.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-19 10:40:00 | 873.30 | 2025-02-19 10:50:00 | 869.48 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-02-27 11:15:00 | 867.65 | 2025-02-27 12:10:00 | 864.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-02-27 11:15:00 | 867.65 | 2025-02-27 12:30:00 | 867.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-20 09:35:00 | 901.00 | 2025-03-20 09:50:00 | 896.17 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-20 09:35:00 | 901.00 | 2025-03-20 14:25:00 | 893.75 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2025-04-16 10:20:00 | 799.40 | 2025-04-16 11:35:00 | 802.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-25 09:30:00 | 811.00 | 2025-04-25 09:35:00 | 807.04 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-04-25 09:30:00 | 811.00 | 2025-04-25 15:20:00 | 800.25 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2025-05-02 09:40:00 | 795.85 | 2025-05-02 09:55:00 | 799.81 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-05-09 10:35:00 | 831.00 | 2025-05-09 11:15:00 | 837.00 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-05-09 10:35:00 | 831.00 | 2025-05-09 15:20:00 | 855.00 | TARGET_HIT | 0.50 | 2.89% |
