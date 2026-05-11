# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1760.00
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
| ENTRY1 | 97 |
| ENTRY2 | 0 |
| PARTIAL | 44 |
| TARGET_HIT | 18 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 79
- **Target hits / Stop hits / Partials:** 18 / 79 / 44
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 16.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 34 | 45.3% | 9 | 41 | 25 | 0.14% | 10.7% |
| BUY @ 2nd Alert (retest1) | 75 | 34 | 45.3% | 9 | 41 | 25 | 0.14% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 28 | 42.4% | 9 | 38 | 19 | 0.09% | 6.2% |
| SELL @ 2nd Alert (retest1) | 66 | 28 | 42.4% | 9 | 38 | 19 | 0.09% | 6.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 141 | 62 | 44.0% | 18 | 79 | 44 | 0.12% | 16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 11:00:00 | 1374.90 | 1359.52 | 0.00 | ORB-long ORB[1352.60,1364.70] vol=2.5x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:20:00 | 1382.94 | 1364.34 | 0.00 | T1 1.5R @ 1382.94 |
| Stop hit — per-position SL triggered | 2025-05-13 12:50:00 | 1374.90 | 1372.47 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:40:00 | 1391.30 | 1383.02 | 0.00 | ORB-long ORB[1371.90,1387.90] vol=2.0x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-05-15 11:10:00 | 1386.83 | 1384.89 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 1394.00 | 1384.25 | 0.00 | ORB-long ORB[1376.10,1384.00] vol=1.7x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 1390.16 | 1386.88 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 1408.80 | 1404.89 | 0.00 | ORB-long ORB[1395.60,1407.50] vol=1.6x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-05-26 09:35:00 | 1405.33 | 1405.09 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:35:00 | 1418.40 | 1410.16 | 0.00 | ORB-long ORB[1403.00,1409.70] vol=1.8x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:50:00 | 1423.85 | 1413.57 | 0.00 | T1 1.5R @ 1423.85 |
| Stop hit — per-position SL triggered | 2025-05-28 11:10:00 | 1418.40 | 1414.85 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1453.80 | 1446.10 | 0.00 | ORB-long ORB[1438.00,1450.10] vol=2.3x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:40:00 | 1459.27 | 1449.25 | 0.00 | T1 1.5R @ 1459.27 |
| Stop hit — per-position SL triggered | 2025-06-05 09:45:00 | 1453.80 | 1450.00 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1466.60 | 1472.01 | 0.00 | ORB-short ORB[1467.00,1484.00] vol=1.6x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 1470.14 | 1470.81 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:55:00 | 1468.00 | 1468.08 | 0.00 | ORB-short ORB[1469.10,1476.00] vol=1.7x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:45:00 | 1463.95 | 1467.81 | 0.00 | T1 1.5R @ 1463.95 |
| Target hit | 2025-06-11 15:20:00 | 1455.00 | 1461.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:05:00 | 1452.40 | 1456.12 | 0.00 | ORB-short ORB[1453.30,1465.00] vol=2.0x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1456.02 | 1455.45 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1411.40 | 1406.21 | 0.00 | ORB-long ORB[1396.30,1407.70] vol=2.0x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 1408.36 | 1407.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:35:00 | 1352.00 | 1358.33 | 0.00 | ORB-short ORB[1354.60,1372.60] vol=2.3x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:10:00 | 1346.70 | 1354.79 | 0.00 | T1 1.5R @ 1346.70 |
| Target hit | 2025-06-19 15:20:00 | 1337.20 | 1345.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1348.90 | 1340.17 | 0.00 | ORB-long ORB[1335.10,1344.40] vol=1.5x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-06-20 10:25:00 | 1345.42 | 1340.57 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:10:00 | 1390.50 | 1394.74 | 0.00 | ORB-short ORB[1394.10,1404.00] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-06-25 11:25:00 | 1393.37 | 1394.48 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 1442.50 | 1435.26 | 0.00 | ORB-long ORB[1427.00,1434.90] vol=1.5x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:50:00 | 1448.44 | 1437.38 | 0.00 | T1 1.5R @ 1448.44 |
| Stop hit — per-position SL triggered | 2025-06-27 11:55:00 | 1442.50 | 1446.20 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:20:00 | 1441.90 | 1434.43 | 0.00 | ORB-long ORB[1428.30,1437.00] vol=1.6x ATR=3.21 |
| Stop hit — per-position SL triggered | 2025-07-14 10:30:00 | 1438.69 | 1436.17 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:15:00 | 1445.10 | 1436.90 | 0.00 | ORB-long ORB[1433.00,1440.20] vol=3.7x ATR=2.77 |
| Stop hit — per-position SL triggered | 2025-07-15 11:25:00 | 1442.33 | 1437.79 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 1460.00 | 1456.40 | 0.00 | ORB-long ORB[1442.90,1459.60] vol=3.4x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 1456.45 | 1458.11 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:40:00 | 1443.40 | 1444.42 | 0.00 | ORB-short ORB[1444.00,1454.50] vol=1.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-07-22 10:45:00 | 1445.93 | 1444.42 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 1414.70 | 1415.18 | 0.00 | ORB-short ORB[1418.70,1426.00] vol=1.8x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:20:00 | 1411.45 | 1415.07 | 0.00 | T1 1.5R @ 1411.45 |
| Stop hit — per-position SL triggered | 2025-07-24 11:45:00 | 1414.70 | 1414.91 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:10:00 | 1400.80 | 1407.55 | 0.00 | ORB-short ORB[1406.80,1415.60] vol=2.4x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-07-25 11:20:00 | 1403.73 | 1407.29 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:00:00 | 1382.00 | 1385.92 | 0.00 | ORB-short ORB[1382.10,1392.40] vol=4.2x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:35:00 | 1377.45 | 1384.79 | 0.00 | T1 1.5R @ 1377.45 |
| Stop hit — per-position SL triggered | 2025-07-31 10:45:00 | 1382.00 | 1384.48 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:15:00 | 1364.40 | 1369.03 | 0.00 | ORB-short ORB[1365.80,1374.90] vol=2.2x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:35:00 | 1359.08 | 1367.75 | 0.00 | T1 1.5R @ 1359.08 |
| Target hit | 2025-08-01 15:20:00 | 1349.10 | 1357.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-08-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 11:10:00 | 1369.90 | 1361.95 | 0.00 | ORB-long ORB[1345.20,1363.40] vol=2.2x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 1366.30 | 1362.13 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:10:00 | 1373.70 | 1376.04 | 0.00 | ORB-short ORB[1380.30,1394.20] vol=1.9x ATR=3.71 |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 1377.41 | 1377.08 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:55:00 | 1337.60 | 1343.48 | 0.00 | ORB-short ORB[1338.00,1347.20] vol=1.6x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:10:00 | 1332.66 | 1342.55 | 0.00 | T1 1.5R @ 1332.66 |
| Stop hit — per-position SL triggered | 2025-08-12 11:20:00 | 1337.60 | 1341.85 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 1324.90 | 1332.87 | 0.00 | ORB-short ORB[1329.70,1339.30] vol=1.7x ATR=3.02 |
| Stop hit — per-position SL triggered | 2025-08-13 11:00:00 | 1327.92 | 1332.23 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 1305.30 | 1316.42 | 0.00 | ORB-short ORB[1312.60,1329.00] vol=2.7x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-08-14 09:40:00 | 1308.84 | 1314.15 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:35:00 | 1344.80 | 1336.73 | 0.00 | ORB-long ORB[1325.00,1340.00] vol=2.4x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:40:00 | 1350.00 | 1338.31 | 0.00 | T1 1.5R @ 1350.00 |
| Target hit | 2025-08-19 15:20:00 | 1369.10 | 1352.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 1368.70 | 1372.35 | 0.00 | ORB-short ORB[1370.00,1377.80] vol=1.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-08-21 09:45:00 | 1371.16 | 1371.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 1344.10 | 1352.05 | 0.00 | ORB-short ORB[1346.20,1364.00] vol=2.7x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 14:45:00 | 1339.95 | 1348.64 | 0.00 | T1 1.5R @ 1339.95 |
| Target hit | 2025-08-22 15:20:00 | 1342.00 | 1347.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-08-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:50:00 | 1337.40 | 1340.32 | 0.00 | ORB-short ORB[1339.80,1348.90] vol=3.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:10:00 | 1333.61 | 1339.51 | 0.00 | T1 1.5R @ 1333.61 |
| Stop hit — per-position SL triggered | 2025-08-25 11:45:00 | 1337.40 | 1338.83 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 10:40:00 | 1329.00 | 1321.40 | 0.00 | ORB-long ORB[1311.20,1322.90] vol=1.8x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:10:00 | 1334.32 | 1323.66 | 0.00 | T1 1.5R @ 1334.32 |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 1329.00 | 1326.02 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:45:00 | 1304.10 | 1310.85 | 0.00 | ORB-short ORB[1310.00,1320.80] vol=1.5x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-08-29 09:50:00 | 1307.73 | 1310.32 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:30:00 | 1332.00 | 1324.40 | 0.00 | ORB-long ORB[1316.10,1328.60] vol=2.0x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-09-01 10:00:00 | 1328.71 | 1326.14 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 11:10:00 | 1332.90 | 1339.59 | 0.00 | ORB-short ORB[1336.60,1348.00] vol=4.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 11:20:00 | 1328.16 | 1338.36 | 0.00 | T1 1.5R @ 1328.16 |
| Stop hit — per-position SL triggered | 2025-09-04 11:55:00 | 1332.90 | 1336.66 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 1322.00 | 1329.20 | 0.00 | ORB-short ORB[1329.50,1335.70] vol=2.3x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-09-05 10:25:00 | 1324.83 | 1327.57 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:35:00 | 1336.50 | 1332.75 | 0.00 | ORB-long ORB[1321.90,1333.80] vol=1.9x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-09-08 09:45:00 | 1333.62 | 1333.26 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 1361.50 | 1358.08 | 0.00 | ORB-long ORB[1348.30,1361.40] vol=2.2x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:35:00 | 1366.19 | 1359.31 | 0.00 | T1 1.5R @ 1366.19 |
| Target hit | 2025-09-09 10:50:00 | 1369.90 | 1370.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:55:00 | 1420.90 | 1411.31 | 0.00 | ORB-long ORB[1401.00,1416.60] vol=1.7x ATR=4.85 |
| Stop hit — per-position SL triggered | 2025-09-11 10:05:00 | 1416.05 | 1412.23 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:10:00 | 1405.40 | 1398.17 | 0.00 | ORB-long ORB[1391.10,1397.80] vol=2.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-09-12 11:20:00 | 1402.96 | 1398.50 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:40:00 | 1404.50 | 1399.36 | 0.00 | ORB-long ORB[1393.00,1401.20] vol=1.6x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-09-15 10:00:00 | 1402.10 | 1401.11 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:00:00 | 1411.00 | 1406.76 | 0.00 | ORB-long ORB[1396.60,1405.00] vol=2.3x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-09-16 10:20:00 | 1408.40 | 1407.18 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:55:00 | 1445.60 | 1440.64 | 0.00 | ORB-long ORB[1425.00,1445.00] vol=2.4x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:05:00 | 1451.82 | 1442.71 | 0.00 | T1 1.5R @ 1451.82 |
| Stop hit — per-position SL triggered | 2025-09-22 11:45:00 | 1445.60 | 1444.35 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:40:00 | 1432.30 | 1440.61 | 0.00 | ORB-short ORB[1439.70,1447.90] vol=1.8x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-09-24 10:00:00 | 1435.99 | 1439.28 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 10:35:00 | 1391.80 | 1396.22 | 0.00 | ORB-short ORB[1393.60,1401.30] vol=2.5x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:50:00 | 1387.50 | 1393.72 | 0.00 | T1 1.5R @ 1387.50 |
| Target hit | 2025-09-29 13:20:00 | 1389.00 | 1388.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2025-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:55:00 | 1398.10 | 1395.30 | 0.00 | ORB-long ORB[1387.20,1395.00] vol=2.0x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:35:00 | 1402.73 | 1397.35 | 0.00 | T1 1.5R @ 1402.73 |
| Target hit | 2025-09-30 15:20:00 | 1407.10 | 1399.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-10-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:40:00 | 1400.00 | 1405.62 | 0.00 | ORB-short ORB[1408.00,1419.10] vol=4.5x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-10-06 13:55:00 | 1403.34 | 1402.64 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 1413.10 | 1407.51 | 0.00 | ORB-long ORB[1400.80,1409.00] vol=1.7x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-10-07 09:40:00 | 1410.26 | 1407.95 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:10:00 | 1403.70 | 1400.67 | 0.00 | ORB-long ORB[1394.60,1402.40] vol=2.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:15:00 | 1408.68 | 1402.45 | 0.00 | T1 1.5R @ 1408.68 |
| Stop hit — per-position SL triggered | 2025-10-08 10:20:00 | 1403.70 | 1402.74 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 1412.40 | 1408.19 | 0.00 | ORB-long ORB[1395.70,1408.30] vol=2.1x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:15:00 | 1417.26 | 1410.21 | 0.00 | T1 1.5R @ 1417.26 |
| Stop hit — per-position SL triggered | 2025-10-10 10:25:00 | 1412.40 | 1410.57 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:55:00 | 1419.60 | 1409.33 | 0.00 | ORB-long ORB[1396.00,1409.70] vol=1.6x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:00:00 | 1424.40 | 1412.69 | 0.00 | T1 1.5R @ 1424.40 |
| Target hit | 2025-10-13 15:20:00 | 1437.00 | 1427.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:35:00 | 1443.20 | 1438.78 | 0.00 | ORB-long ORB[1430.00,1441.70] vol=4.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:20:00 | 1448.19 | 1440.18 | 0.00 | T1 1.5R @ 1448.19 |
| Target hit | 2025-10-15 15:20:00 | 1450.70 | 1444.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1480.30 | 1472.54 | 0.00 | ORB-long ORB[1453.00,1475.00] vol=2.3x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:30:00 | 1485.99 | 1474.84 | 0.00 | T1 1.5R @ 1485.99 |
| Stop hit — per-position SL triggered | 2025-10-16 10:55:00 | 1480.30 | 1478.09 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 11:05:00 | 1468.00 | 1472.82 | 0.00 | ORB-short ORB[1470.00,1483.50] vol=1.7x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-10-23 11:50:00 | 1470.68 | 1471.79 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 1442.80 | 1444.70 | 0.00 | ORB-short ORB[1443.00,1463.50] vol=4.0x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:15:00 | 1436.99 | 1443.54 | 0.00 | T1 1.5R @ 1436.99 |
| Target hit | 2025-10-24 15:20:00 | 1429.10 | 1434.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-10-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 09:35:00 | 1420.60 | 1423.72 | 0.00 | ORB-short ORB[1421.50,1429.80] vol=2.6x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 09:55:00 | 1415.80 | 1421.67 | 0.00 | T1 1.5R @ 1415.80 |
| Stop hit — per-position SL triggered | 2025-10-27 11:30:00 | 1420.60 | 1417.76 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-10-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:50:00 | 1432.30 | 1427.50 | 0.00 | ORB-long ORB[1421.60,1428.00] vol=1.7x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:00:00 | 1437.37 | 1428.87 | 0.00 | T1 1.5R @ 1437.37 |
| Target hit | 2025-10-29 15:20:00 | 1453.20 | 1451.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:00:00 | 1461.00 | 1455.07 | 0.00 | ORB-long ORB[1449.60,1458.70] vol=1.8x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-10-30 11:50:00 | 1457.35 | 1456.58 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 1443.50 | 1450.99 | 0.00 | ORB-short ORB[1448.00,1457.90] vol=3.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 1445.92 | 1450.68 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:40:00 | 1443.50 | 1432.52 | 0.00 | ORB-long ORB[1425.00,1436.00] vol=1.7x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:10:00 | 1450.15 | 1436.73 | 0.00 | T1 1.5R @ 1450.15 |
| Target hit | 2025-11-07 15:10:00 | 1447.00 | 1447.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2025-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:40:00 | 1490.00 | 1483.78 | 0.00 | ORB-long ORB[1475.50,1486.70] vol=2.1x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:50:00 | 1495.78 | 1488.01 | 0.00 | T1 1.5R @ 1495.78 |
| Target hit | 2025-11-12 10:45:00 | 1498.20 | 1498.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-11-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:45:00 | 1517.00 | 1509.63 | 0.00 | ORB-long ORB[1497.00,1511.00] vol=2.7x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-11-14 09:55:00 | 1512.46 | 1510.90 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-11-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:40:00 | 1507.00 | 1510.66 | 0.00 | ORB-short ORB[1510.20,1519.00] vol=2.0x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-11-17 12:00:00 | 1509.69 | 1508.29 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:15:00 | 1469.40 | 1478.72 | 0.00 | ORB-short ORB[1474.60,1492.30] vol=1.7x ATR=3.37 |
| Stop hit — per-position SL triggered | 2025-11-21 10:40:00 | 1472.77 | 1477.05 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-11-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 11:10:00 | 1477.80 | 1482.17 | 0.00 | ORB-short ORB[1479.90,1492.40] vol=3.1x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-11-25 11:35:00 | 1480.29 | 1481.83 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1514.60 | 1509.74 | 0.00 | ORB-long ORB[1503.10,1510.00] vol=4.0x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:50:00 | 1518.91 | 1512.46 | 0.00 | T1 1.5R @ 1518.91 |
| Stop hit — per-position SL triggered | 2025-11-28 10:20:00 | 1514.60 | 1516.45 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 1542.80 | 1538.29 | 0.00 | ORB-long ORB[1523.20,1541.00] vol=3.6x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-12-01 10:00:00 | 1538.88 | 1540.30 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:55:00 | 1504.20 | 1513.00 | 0.00 | ORB-short ORB[1514.00,1521.00] vol=2.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 1507.51 | 1510.13 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:50:00 | 1504.00 | 1502.25 | 0.00 | ORB-long ORB[1490.20,1501.90] vol=2.5x ATR=3.25 |
| Stop hit — per-position SL triggered | 2025-12-04 11:05:00 | 1500.75 | 1502.29 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-12-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:35:00 | 1493.30 | 1499.11 | 0.00 | ORB-short ORB[1500.10,1506.10] vol=1.9x ATR=3.56 |
| Stop hit — per-position SL triggered | 2025-12-05 10:40:00 | 1496.86 | 1498.85 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-12-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:10:00 | 1494.00 | 1501.57 | 0.00 | ORB-short ORB[1501.50,1510.90] vol=1.9x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-12-08 10:20:00 | 1497.12 | 1501.05 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 1518.00 | 1508.59 | 0.00 | ORB-long ORB[1496.00,1509.80] vol=1.9x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-12-10 10:05:00 | 1513.86 | 1512.70 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:55:00 | 1502.30 | 1505.33 | 0.00 | ORB-short ORB[1505.00,1512.60] vol=4.2x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:15:00 | 1499.13 | 1504.09 | 0.00 | T1 1.5R @ 1499.13 |
| Stop hit — per-position SL triggered | 2025-12-16 12:05:00 | 1502.30 | 1503.43 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-12-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:35:00 | 1512.30 | 1507.69 | 0.00 | ORB-long ORB[1497.50,1509.00] vol=2.0x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-12-22 11:10:00 | 1510.05 | 1509.20 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-12-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:40:00 | 1504.40 | 1508.06 | 0.00 | ORB-short ORB[1505.70,1513.80] vol=4.2x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 1506.47 | 1506.97 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 1488.90 | 1494.52 | 0.00 | ORB-short ORB[1490.30,1501.40] vol=1.5x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-12-26 11:30:00 | 1491.45 | 1494.26 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 1471.50 | 1476.19 | 0.00 | ORB-short ORB[1472.60,1487.10] vol=2.6x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-12-29 09:50:00 | 1474.60 | 1474.69 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 1491.50 | 1487.04 | 0.00 | ORB-long ORB[1478.30,1488.90] vol=2.5x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-01-02 10:00:00 | 1488.95 | 1488.66 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:10:00 | 1469.50 | 1474.17 | 0.00 | ORB-short ORB[1470.30,1478.10] vol=2.6x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:35:00 | 1465.58 | 1472.21 | 0.00 | T1 1.5R @ 1465.58 |
| Target hit | 2026-01-07 14:55:00 | 1466.80 | 1466.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — BUY (started 2026-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:45:00 | 1484.30 | 1477.03 | 0.00 | ORB-long ORB[1465.50,1475.50] vol=1.8x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:05:00 | 1488.85 | 1480.63 | 0.00 | T1 1.5R @ 1488.85 |
| Stop hit — per-position SL triggered | 2026-01-08 10:10:00 | 1484.30 | 1480.74 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 1445.30 | 1450.06 | 0.00 | ORB-short ORB[1445.50,1463.00] vol=1.5x ATR=4.49 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 1449.79 | 1449.01 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:50:00 | 1425.90 | 1414.33 | 0.00 | ORB-long ORB[1402.70,1419.80] vol=1.5x ATR=4.25 |
| Stop hit — per-position SL triggered | 2026-02-01 11:00:00 | 1421.65 | 1415.56 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 1549.40 | 1567.91 | 0.00 | ORB-short ORB[1560.20,1575.50] vol=3.3x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:05:00 | 1542.84 | 1565.15 | 0.00 | T1 1.5R @ 1542.84 |
| Stop hit — per-position SL triggered | 2026-02-06 14:55:00 | 1549.40 | 1559.34 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1543.80 | 1555.20 | 0.00 | ORB-short ORB[1557.00,1567.00] vol=3.3x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 1536.72 | 1549.69 | 0.00 | T1 1.5R @ 1536.72 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 1543.80 | 1548.51 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 1549.90 | 1550.41 | 0.00 | ORB-short ORB[1550.40,1559.40] vol=2.5x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:45:00 | 1545.28 | 1550.09 | 0.00 | T1 1.5R @ 1545.28 |
| Target hit | 2026-02-11 12:50:00 | 1547.60 | 1546.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 1539.30 | 1544.82 | 0.00 | ORB-short ORB[1541.90,1551.00] vol=3.6x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 1535.68 | 1544.16 | 0.00 | T1 1.5R @ 1535.68 |
| Stop hit — per-position SL triggered | 2026-02-12 12:50:00 | 1539.30 | 1541.27 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 1514.70 | 1525.44 | 0.00 | ORB-short ORB[1525.70,1543.00] vol=2.6x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-02-13 12:00:00 | 1519.12 | 1519.22 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 1557.30 | 1545.44 | 0.00 | ORB-long ORB[1535.70,1545.00] vol=1.9x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:45:00 | 1563.45 | 1556.74 | 0.00 | T1 1.5R @ 1563.45 |
| Stop hit — per-position SL triggered | 2026-02-17 10:55:00 | 1557.30 | 1556.93 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 1574.30 | 1567.61 | 0.00 | ORB-long ORB[1563.90,1571.80] vol=1.9x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:05:00 | 1579.98 | 1570.07 | 0.00 | T1 1.5R @ 1579.98 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 1574.30 | 1573.01 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 1530.10 | 1535.92 | 0.00 | ORB-short ORB[1534.10,1545.10] vol=1.6x ATR=3.53 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 1533.63 | 1535.06 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:10:00 | 1356.40 | 1368.68 | 0.00 | ORB-short ORB[1370.00,1390.00] vol=2.0x ATR=4.60 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 1361.00 | 1368.13 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1378.70 | 1374.50 | 0.00 | ORB-long ORB[1362.40,1378.50] vol=1.7x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 1384.52 | 1375.31 | 0.00 | T1 1.5R @ 1384.52 |
| Stop hit — per-position SL triggered | 2026-03-20 11:55:00 | 1378.70 | 1377.10 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 1355.50 | 1361.24 | 0.00 | ORB-short ORB[1356.30,1375.00] vol=1.8x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:10:00 | 1348.00 | 1357.13 | 0.00 | T1 1.5R @ 1348.00 |
| Target hit | 2026-03-27 15:20:00 | 1337.90 | 1343.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1533.40 | 1523.04 | 0.00 | ORB-long ORB[1510.00,1524.90] vol=1.6x ATR=5.10 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 1528.30 | 1524.37 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 1585.60 | 1578.71 | 0.00 | ORB-long ORB[1566.20,1581.20] vol=3.4x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:00:00 | 1591.72 | 1581.25 | 0.00 | T1 1.5R @ 1591.72 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 1585.60 | 1581.69 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 1664.00 | 1652.36 | 0.00 | ORB-long ORB[1644.90,1659.00] vol=3.8x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:50:00 | 1670.83 | 1655.75 | 0.00 | T1 1.5R @ 1670.83 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 1664.00 | 1658.12 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1760.40 | 1750.35 | 0.00 | ORB-long ORB[1727.70,1753.10] vol=5.9x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:05:00 | 1768.72 | 1754.68 | 0.00 | T1 1.5R @ 1768.72 |
| Target hit | 2026-05-08 11:45:00 | 1761.00 | 1762.00 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 11:00:00 | 1374.90 | 2025-05-13 11:20:00 | 1382.94 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-05-13 11:00:00 | 1374.90 | 2025-05-13 12:50:00 | 1374.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:40:00 | 1391.30 | 2025-05-15 11:10:00 | 1386.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-23 10:45:00 | 1394.00 | 2025-05-23 11:45:00 | 1390.16 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-26 09:30:00 | 1408.80 | 2025-05-26 09:35:00 | 1405.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-28 10:35:00 | 1418.40 | 2025-05-28 10:50:00 | 1423.85 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-05-28 10:35:00 | 1418.40 | 2025-05-28 11:10:00 | 1418.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:30:00 | 1453.80 | 2025-06-05 09:40:00 | 1459.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-05 09:30:00 | 1453.80 | 2025-06-05 09:45:00 | 1453.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-09 09:30:00 | 1466.60 | 2025-06-09 09:45:00 | 1470.14 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-11 10:55:00 | 1468.00 | 2025-06-11 11:45:00 | 1463.95 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-11 10:55:00 | 1468.00 | 2025-06-11 15:20:00 | 1455.00 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2025-06-12 10:05:00 | 1452.40 | 2025-06-12 10:15:00 | 1456.02 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1411.40 | 2025-06-17 09:45:00 | 1408.36 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-19 09:35:00 | 1352.00 | 2025-06-19 10:10:00 | 1346.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-19 09:35:00 | 1352.00 | 2025-06-19 15:20:00 | 1337.20 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2025-06-20 10:15:00 | 1348.90 | 2025-06-20 10:25:00 | 1345.42 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-25 11:10:00 | 1390.50 | 2025-06-25 11:25:00 | 1393.37 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-27 09:45:00 | 1442.50 | 2025-06-27 09:50:00 | 1448.44 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-27 09:45:00 | 1442.50 | 2025-06-27 11:55:00 | 1442.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 10:20:00 | 1441.90 | 2025-07-14 10:30:00 | 1438.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-15 11:15:00 | 1445.10 | 2025-07-15 11:25:00 | 1442.33 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-16 09:30:00 | 1460.00 | 2025-07-16 09:40:00 | 1456.45 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-22 10:40:00 | 1443.40 | 2025-07-22 10:45:00 | 1445.93 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-24 11:15:00 | 1414.70 | 2025-07-24 11:20:00 | 1411.45 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-24 11:15:00 | 1414.70 | 2025-07-24 11:45:00 | 1414.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 11:10:00 | 1400.80 | 2025-07-25 11:20:00 | 1403.73 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-31 10:00:00 | 1382.00 | 2025-07-31 10:35:00 | 1377.45 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-31 10:00:00 | 1382.00 | 2025-07-31 10:45:00 | 1382.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-01 11:15:00 | 1364.40 | 2025-08-01 11:35:00 | 1359.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-01 11:15:00 | 1364.40 | 2025-08-01 15:20:00 | 1349.10 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2025-08-04 11:10:00 | 1369.90 | 2025-08-04 11:15:00 | 1366.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-05 11:10:00 | 1373.70 | 2025-08-05 11:15:00 | 1377.41 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-12 10:55:00 | 1337.60 | 2025-08-12 11:10:00 | 1332.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-12 10:55:00 | 1337.60 | 2025-08-12 11:20:00 | 1337.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 10:45:00 | 1324.90 | 2025-08-13 11:00:00 | 1327.92 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-14 09:30:00 | 1305.30 | 2025-08-14 09:40:00 | 1308.84 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-19 10:35:00 | 1344.80 | 2025-08-19 10:40:00 | 1350.00 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-19 10:35:00 | 1344.80 | 2025-08-19 15:20:00 | 1369.10 | TARGET_HIT | 0.50 | 1.81% |
| SELL | retest1 | 2025-08-21 09:40:00 | 1368.70 | 2025-08-21 09:45:00 | 1371.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-22 11:00:00 | 1344.10 | 2025-08-22 14:45:00 | 1339.95 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-22 11:00:00 | 1344.10 | 2025-08-22 15:20:00 | 1342.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-08-25 10:50:00 | 1337.40 | 2025-08-25 11:10:00 | 1333.61 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-25 10:50:00 | 1337.40 | 2025-08-25 11:45:00 | 1337.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-28 10:40:00 | 1329.00 | 2025-08-28 11:10:00 | 1334.32 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-28 10:40:00 | 1329.00 | 2025-08-28 12:15:00 | 1329.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:45:00 | 1304.10 | 2025-08-29 09:50:00 | 1307.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-01 09:30:00 | 1332.00 | 2025-09-01 10:00:00 | 1328.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-04 11:10:00 | 1332.90 | 2025-09-04 11:20:00 | 1328.16 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-04 11:10:00 | 1332.90 | 2025-09-04 11:55:00 | 1332.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:10:00 | 1322.00 | 2025-09-05 10:25:00 | 1324.83 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-08 09:35:00 | 1336.50 | 2025-09-08 09:45:00 | 1333.62 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-09 09:30:00 | 1361.50 | 2025-09-09 09:35:00 | 1366.19 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-09 09:30:00 | 1361.50 | 2025-09-09 10:50:00 | 1369.90 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-11 09:55:00 | 1420.90 | 2025-09-11 10:05:00 | 1416.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-12 11:10:00 | 1405.40 | 2025-09-12 11:20:00 | 1402.96 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-15 09:40:00 | 1404.50 | 2025-09-15 10:00:00 | 1402.10 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-16 10:00:00 | 1411.00 | 2025-09-16 10:20:00 | 1408.40 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-22 10:55:00 | 1445.60 | 2025-09-22 11:05:00 | 1451.82 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-22 10:55:00 | 1445.60 | 2025-09-22 11:45:00 | 1445.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 09:40:00 | 1432.30 | 2025-09-24 10:00:00 | 1435.99 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-29 10:35:00 | 1391.80 | 2025-09-29 10:50:00 | 1387.50 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-29 10:35:00 | 1391.80 | 2025-09-29 13:20:00 | 1389.00 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-09-30 10:55:00 | 1398.10 | 2025-09-30 14:35:00 | 1402.73 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-30 10:55:00 | 1398.10 | 2025-09-30 15:20:00 | 1407.10 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-10-06 10:40:00 | 1400.00 | 2025-10-06 13:55:00 | 1403.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-07 09:35:00 | 1413.10 | 2025-10-07 09:40:00 | 1410.26 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-08 10:10:00 | 1403.70 | 2025-10-08 10:15:00 | 1408.68 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-08 10:10:00 | 1403.70 | 2025-10-08 10:20:00 | 1403.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:40:00 | 1412.40 | 2025-10-10 10:15:00 | 1417.26 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-10 09:40:00 | 1412.40 | 2025-10-10 10:25:00 | 1412.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-13 10:55:00 | 1419.60 | 2025-10-13 11:00:00 | 1424.40 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-13 10:55:00 | 1419.60 | 2025-10-13 15:20:00 | 1437.00 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-10-15 10:35:00 | 1443.20 | 2025-10-15 12:20:00 | 1448.19 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-15 10:35:00 | 1443.20 | 2025-10-15 15:20:00 | 1450.70 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-16 10:15:00 | 1480.30 | 2025-10-16 10:30:00 | 1485.99 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-16 10:15:00 | 1480.30 | 2025-10-16 10:55:00 | 1480.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-23 11:05:00 | 1468.00 | 2025-10-23 11:50:00 | 1470.68 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-24 09:40:00 | 1442.80 | 2025-10-24 10:15:00 | 1436.99 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-24 09:40:00 | 1442.80 | 2025-10-24 15:20:00 | 1429.10 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-10-27 09:35:00 | 1420.60 | 2025-10-27 09:55:00 | 1415.80 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-27 09:35:00 | 1420.60 | 2025-10-27 11:30:00 | 1420.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 09:50:00 | 1432.30 | 2025-10-29 10:00:00 | 1437.37 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-29 09:50:00 | 1432.30 | 2025-10-29 15:20:00 | 1453.20 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-10-30 11:00:00 | 1461.00 | 2025-10-30 11:50:00 | 1457.35 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-31 11:10:00 | 1443.50 | 2025-10-31 11:20:00 | 1445.92 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-07 10:40:00 | 1443.50 | 2025-11-07 11:10:00 | 1450.15 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-07 10:40:00 | 1443.50 | 2025-11-07 15:10:00 | 1447.00 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-12 09:40:00 | 1490.00 | 2025-11-12 09:50:00 | 1495.78 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-12 09:40:00 | 1490.00 | 2025-11-12 10:45:00 | 1498.20 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-14 09:45:00 | 1517.00 | 2025-11-14 09:55:00 | 1512.46 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-17 10:40:00 | 1507.00 | 2025-11-17 12:00:00 | 1509.69 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-21 10:15:00 | 1469.40 | 2025-11-21 10:40:00 | 1472.77 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-25 11:10:00 | 1477.80 | 2025-11-25 11:35:00 | 1480.29 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1514.60 | 2025-11-28 09:50:00 | 1518.91 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1514.60 | 2025-11-28 10:20:00 | 1514.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 09:35:00 | 1542.80 | 2025-12-01 10:00:00 | 1538.88 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-03 10:55:00 | 1504.20 | 2025-12-03 12:15:00 | 1507.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-04 10:50:00 | 1504.00 | 2025-12-04 11:05:00 | 1500.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-05 10:35:00 | 1493.30 | 2025-12-05 10:40:00 | 1496.86 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-08 10:10:00 | 1494.00 | 2025-12-08 10:20:00 | 1497.12 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-10 09:30:00 | 1518.00 | 2025-12-10 10:05:00 | 1513.86 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-16 10:55:00 | 1502.30 | 2025-12-16 11:15:00 | 1499.13 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-12-16 10:55:00 | 1502.30 | 2025-12-16 12:05:00 | 1502.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 10:35:00 | 1512.30 | 2025-12-22 11:10:00 | 1510.05 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-23 10:40:00 | 1504.40 | 2025-12-23 11:15:00 | 1506.47 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-26 11:10:00 | 1488.90 | 2025-12-26 11:30:00 | 1491.45 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-29 09:30:00 | 1471.50 | 2025-12-29 09:50:00 | 1474.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 09:35:00 | 1491.50 | 2026-01-02 10:00:00 | 1488.95 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-01-07 11:10:00 | 1469.50 | 2026-01-07 11:35:00 | 1465.58 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-01-07 11:10:00 | 1469.50 | 2026-01-07 14:55:00 | 1466.80 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-01-08 09:45:00 | 1484.30 | 2026-01-08 10:05:00 | 1488.85 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-08 09:45:00 | 1484.30 | 2026-01-08 10:10:00 | 1484.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-09 09:35:00 | 1445.30 | 2026-01-09 09:45:00 | 1449.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-01 10:50:00 | 1425.90 | 2026-02-01 11:00:00 | 1421.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-06 11:00:00 | 1549.40 | 2026-02-06 12:05:00 | 1542.84 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-06 11:00:00 | 1549.40 | 2026-02-06 14:55:00 | 1549.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1543.80 | 2026-02-10 10:40:00 | 1536.72 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1543.80 | 2026-02-10 10:50:00 | 1543.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:40:00 | 1549.90 | 2026-02-11 10:45:00 | 1545.28 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-11 10:40:00 | 1549.90 | 2026-02-11 12:50:00 | 1547.60 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-02-12 11:10:00 | 1539.30 | 2026-02-12 11:20:00 | 1535.68 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-12 11:10:00 | 1539.30 | 2026-02-12 12:50:00 | 1539.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:45:00 | 1514.70 | 2026-02-13 12:00:00 | 1519.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1557.30 | 2026-02-17 10:45:00 | 1563.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1557.30 | 2026-02-17 10:55:00 | 1557.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1574.30 | 2026-02-25 10:05:00 | 1579.98 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1574.30 | 2026-02-25 10:20:00 | 1574.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:00:00 | 1530.10 | 2026-02-27 10:25:00 | 1533.63 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-13 11:10:00 | 1356.40 | 2026-03-13 11:25:00 | 1361.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-20 10:45:00 | 1378.70 | 2026-03-20 11:15:00 | 1384.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-20 10:45:00 | 1378.70 | 2026-03-20 11:55:00 | 1378.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 09:30:00 | 1355.50 | 2026-03-27 10:10:00 | 1348.00 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-27 09:30:00 | 1355.50 | 2026-03-27 15:20:00 | 1337.90 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2026-04-16 09:45:00 | 1533.40 | 2026-04-16 09:55:00 | 1528.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-23 10:55:00 | 1585.60 | 2026-04-23 11:00:00 | 1591.72 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-23 10:55:00 | 1585.60 | 2026-04-23 11:10:00 | 1585.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:40:00 | 1664.00 | 2026-04-29 10:50:00 | 1670.83 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-29 10:40:00 | 1664.00 | 2026-04-29 11:15:00 | 1664.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1760.40 | 2026-05-08 10:05:00 | 1768.72 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1760.40 | 2026-05-08 11:45:00 | 1761.00 | TARGET_HIT | 0.50 | 0.03% |
