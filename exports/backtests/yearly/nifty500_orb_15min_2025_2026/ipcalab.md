# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2025-06-10 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 1554.00
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
| ENTRY1 | 94 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 12 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 82
- **Target hits / Stop hits / Partials:** 12 / 82 / 39
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 14.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 22 | 37.9% | 6 | 36 | 16 | 0.14% | 8.1% |
| BUY @ 2nd Alert (retest1) | 58 | 22 | 37.9% | 6 | 36 | 16 | 0.14% | 8.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 75 | 29 | 38.7% | 6 | 46 | 23 | 0.09% | 6.9% |
| SELL @ 2nd Alert (retest1) | 75 | 29 | 38.7% | 6 | 46 | 23 | 0.09% | 6.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 133 | 51 | 38.3% | 12 | 82 | 39 | 0.11% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:50:00 | 1368.60 | 1377.31 | 0.00 | ORB-short ORB[1375.10,1389.80] vol=2.0x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-06-11 11:00:00 | 1372.17 | 1376.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:45:00 | 1413.50 | 1408.39 | 0.00 | ORB-long ORB[1392.80,1413.40] vol=1.7x ATR=5.66 |
| Stop hit — per-position SL triggered | 2025-06-12 10:30:00 | 1407.84 | 1410.19 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 10:40:00 | 1352.50 | 1360.95 | 0.00 | ORB-short ORB[1362.00,1370.00] vol=2.1x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-06-16 11:55:00 | 1355.72 | 1356.44 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 10:45:00 | 1333.20 | 1336.67 | 0.00 | ORB-short ORB[1336.00,1348.80] vol=1.6x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 12:45:00 | 1328.74 | 1335.48 | 0.00 | T1 1.5R @ 1328.74 |
| Stop hit — per-position SL triggered | 2025-06-25 12:50:00 | 1333.20 | 1335.42 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:05:00 | 1344.40 | 1353.75 | 0.00 | ORB-short ORB[1357.80,1375.10] vol=2.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 12:50:00 | 1339.66 | 1350.56 | 0.00 | T1 1.5R @ 1339.66 |
| Stop hit — per-position SL triggered | 2025-06-26 13:30:00 | 1344.40 | 1349.91 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:25:00 | 1361.80 | 1349.34 | 0.00 | ORB-long ORB[1342.70,1351.90] vol=3.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-06-27 10:30:00 | 1357.58 | 1350.39 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-07-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:10:00 | 1360.00 | 1369.39 | 0.00 | ORB-short ORB[1369.40,1389.10] vol=2.1x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-07-01 12:10:00 | 1363.73 | 1366.82 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:40:00 | 1395.90 | 1388.32 | 0.00 | ORB-long ORB[1377.70,1392.00] vol=1.6x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:05:00 | 1403.68 | 1395.35 | 0.00 | T1 1.5R @ 1403.68 |
| Target hit | 2025-07-03 11:55:00 | 1411.20 | 1411.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 1439.00 | 1432.16 | 0.00 | ORB-long ORB[1418.40,1431.00] vol=1.7x ATR=6.38 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 1432.62 | 1432.93 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:25:00 | 1431.90 | 1439.40 | 0.00 | ORB-short ORB[1438.20,1458.90] vol=2.7x ATR=4.65 |
| Stop hit — per-position SL triggered | 2025-07-10 10:45:00 | 1436.55 | 1438.73 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:50:00 | 1469.30 | 1460.71 | 0.00 | ORB-long ORB[1443.50,1452.80] vol=2.2x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-07-11 09:55:00 | 1465.08 | 1462.14 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 1464.30 | 1468.03 | 0.00 | ORB-short ORB[1468.10,1475.70] vol=3.4x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:20:00 | 1457.12 | 1466.39 | 0.00 | T1 1.5R @ 1457.12 |
| Stop hit — per-position SL triggered | 2025-07-18 11:35:00 | 1464.30 | 1465.56 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 1445.30 | 1451.97 | 0.00 | ORB-short ORB[1450.40,1470.00] vol=1.8x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-07-22 10:00:00 | 1449.21 | 1449.77 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:55:00 | 1430.00 | 1438.27 | 0.00 | ORB-short ORB[1439.30,1456.50] vol=1.6x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:35:00 | 1424.12 | 1435.10 | 0.00 | T1 1.5R @ 1424.12 |
| Stop hit — per-position SL triggered | 2025-08-05 11:40:00 | 1430.00 | 1434.76 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:35:00 | 1410.00 | 1411.05 | 0.00 | ORB-short ORB[1410.90,1427.80] vol=4.5x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-08-06 12:20:00 | 1415.38 | 1410.39 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1365.20 | 1375.53 | 0.00 | ORB-short ORB[1375.00,1391.80] vol=1.6x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-08-11 11:00:00 | 1369.73 | 1374.05 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:10:00 | 1355.50 | 1377.34 | 0.00 | ORB-short ORB[1381.00,1397.40] vol=3.1x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:20:00 | 1348.07 | 1374.49 | 0.00 | T1 1.5R @ 1348.07 |
| Stop hit — per-position SL triggered | 2025-08-13 12:10:00 | 1355.50 | 1368.32 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:25:00 | 1354.40 | 1364.79 | 0.00 | ORB-short ORB[1360.00,1378.70] vol=2.1x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-08-14 10:55:00 | 1358.55 | 1362.70 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:35:00 | 1391.00 | 1385.01 | 0.00 | ORB-long ORB[1379.30,1389.60] vol=1.5x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-08-20 10:10:00 | 1385.99 | 1388.19 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:40:00 | 1347.00 | 1350.93 | 0.00 | ORB-short ORB[1350.00,1365.50] vol=2.2x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 1350.41 | 1350.87 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 10:15:00 | 1393.70 | 1402.75 | 0.00 | ORB-short ORB[1403.20,1414.80] vol=1.9x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:25:00 | 1387.02 | 1400.06 | 0.00 | T1 1.5R @ 1387.02 |
| Stop hit — per-position SL triggered | 2025-08-29 10:30:00 | 1393.70 | 1396.74 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:40:00 | 1367.50 | 1373.94 | 0.00 | ORB-short ORB[1371.40,1384.40] vol=1.5x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:05:00 | 1361.49 | 1373.01 | 0.00 | T1 1.5R @ 1361.49 |
| Stop hit — per-position SL triggered | 2025-09-01 14:00:00 | 1367.50 | 1366.49 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1371.30 | 1367.58 | 0.00 | ORB-long ORB[1358.40,1371.20] vol=3.1x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 1368.16 | 1368.34 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:55:00 | 1344.60 | 1349.97 | 0.00 | ORB-short ORB[1346.40,1360.40] vol=1.7x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 10:05:00 | 1340.14 | 1345.97 | 0.00 | T1 1.5R @ 1340.14 |
| Stop hit — per-position SL triggered | 2025-09-04 10:10:00 | 1344.60 | 1345.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 1329.50 | 1326.03 | 0.00 | ORB-long ORB[1317.20,1329.00] vol=1.7x ATR=3.27 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 1326.23 | 1326.24 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:05:00 | 1313.10 | 1317.29 | 0.00 | ORB-short ORB[1313.20,1325.50] vol=1.7x ATR=3.71 |
| Stop hit — per-position SL triggered | 2025-09-11 13:40:00 | 1316.81 | 1314.79 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:15:00 | 1326.50 | 1321.13 | 0.00 | ORB-long ORB[1311.00,1326.00] vol=1.5x ATR=3.15 |
| Stop hit — per-position SL triggered | 2025-09-12 11:20:00 | 1323.35 | 1322.04 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 1323.30 | 1317.43 | 0.00 | ORB-long ORB[1308.10,1319.90] vol=2.5x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-09-17 09:45:00 | 1320.27 | 1317.70 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 1319.00 | 1315.80 | 0.00 | ORB-long ORB[1309.00,1318.40] vol=1.5x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 09:55:00 | 1323.56 | 1317.88 | 0.00 | T1 1.5R @ 1323.56 |
| Stop hit — per-position SL triggered | 2025-09-18 10:00:00 | 1319.00 | 1318.01 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:40:00 | 1350.00 | 1343.02 | 0.00 | ORB-long ORB[1333.80,1345.80] vol=2.3x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-09-19 09:55:00 | 1345.76 | 1347.06 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:00:00 | 1369.10 | 1375.11 | 0.00 | ORB-short ORB[1370.50,1388.20] vol=2.3x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:05:00 | 1364.36 | 1373.24 | 0.00 | T1 1.5R @ 1364.36 |
| Target hit | 2025-09-24 15:20:00 | 1344.50 | 1359.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:10:00 | 1317.00 | 1322.21 | 0.00 | ORB-short ORB[1323.00,1339.00] vol=1.7x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:55:00 | 1311.70 | 1318.15 | 0.00 | T1 1.5R @ 1311.70 |
| Stop hit — per-position SL triggered | 2025-10-03 15:00:00 | 1317.00 | 1315.38 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:40:00 | 1309.40 | 1313.87 | 0.00 | ORB-short ORB[1310.40,1323.00] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-10-06 10:45:00 | 1312.66 | 1313.74 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:40:00 | 1334.70 | 1324.10 | 0.00 | ORB-long ORB[1316.10,1325.30] vol=2.4x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:45:00 | 1342.61 | 1329.46 | 0.00 | T1 1.5R @ 1342.61 |
| Target hit | 2025-10-07 13:45:00 | 1372.50 | 1377.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 1347.70 | 1353.37 | 0.00 | ORB-short ORB[1348.70,1361.00] vol=3.9x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 1350.65 | 1353.10 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:40:00 | 1346.90 | 1353.17 | 0.00 | ORB-short ORB[1350.10,1368.20] vol=3.4x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:50:00 | 1339.88 | 1350.84 | 0.00 | T1 1.5R @ 1339.88 |
| Target hit | 2025-10-09 15:20:00 | 1339.00 | 1344.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-10-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1324.00 | 1327.81 | 0.00 | ORB-short ORB[1325.30,1338.30] vol=3.0x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-10-13 11:30:00 | 1327.16 | 1326.39 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:50:00 | 1311.00 | 1319.89 | 0.00 | ORB-short ORB[1320.00,1330.90] vol=4.5x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-10-14 13:25:00 | 1313.93 | 1313.80 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:10:00 | 1283.50 | 1288.78 | 0.00 | ORB-short ORB[1286.00,1297.00] vol=1.7x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-10-17 11:45:00 | 1287.04 | 1287.09 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 1267.50 | 1274.85 | 0.00 | ORB-short ORB[1276.60,1291.50] vol=7.5x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-10-20 09:40:00 | 1272.21 | 1274.58 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1294.70 | 1285.66 | 0.00 | ORB-long ORB[1271.90,1290.40] vol=1.5x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-10-27 10:05:00 | 1290.17 | 1288.31 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 1293.30 | 1287.47 | 0.00 | ORB-long ORB[1281.80,1292.30] vol=4.4x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1290.27 | 1287.78 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:45:00 | 1278.10 | 1282.66 | 0.00 | ORB-short ORB[1281.70,1295.00] vol=3.3x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-10-30 15:00:00 | 1281.68 | 1281.21 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:40:00 | 1275.70 | 1278.80 | 0.00 | ORB-short ORB[1277.00,1296.00] vol=2.0x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-10-31 11:30:00 | 1279.01 | 1278.68 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 1289.70 | 1286.42 | 0.00 | ORB-long ORB[1270.00,1285.50] vol=4.3x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:30:00 | 1294.32 | 1288.81 | 0.00 | T1 1.5R @ 1294.32 |
| Target hit | 2025-11-03 15:20:00 | 1307.00 | 1299.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1311.10 | 1317.13 | 0.00 | ORB-short ORB[1312.00,1323.00] vol=1.5x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:15:00 | 1304.26 | 1313.80 | 0.00 | T1 1.5R @ 1304.26 |
| Target hit | 2025-11-04 12:25:00 | 1307.00 | 1306.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2025-11-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 10:30:00 | 1279.40 | 1285.01 | 0.00 | ORB-short ORB[1288.00,1305.90] vol=5.1x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-11-07 10:45:00 | 1283.08 | 1284.64 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:55:00 | 1327.60 | 1330.42 | 0.00 | ORB-short ORB[1332.60,1352.40] vol=4.7x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-11-11 12:55:00 | 1331.56 | 1329.08 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1343.00 | 1329.48 | 0.00 | ORB-long ORB[1315.10,1329.00] vol=2.0x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:30:00 | 1348.86 | 1335.70 | 0.00 | T1 1.5R @ 1348.86 |
| Stop hit — per-position SL triggered | 2025-11-12 11:35:00 | 1343.00 | 1345.88 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:20:00 | 1353.90 | 1350.13 | 0.00 | ORB-long ORB[1335.90,1352.70] vol=1.8x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:40:00 | 1360.59 | 1352.82 | 0.00 | T1 1.5R @ 1360.59 |
| Stop hit — per-position SL triggered | 2025-11-13 10:50:00 | 1353.90 | 1354.70 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 1440.80 | 1435.67 | 0.00 | ORB-long ORB[1426.90,1439.80] vol=1.6x ATR=5.31 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 1435.49 | 1436.96 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 10:50:00 | 1392.30 | 1399.80 | 0.00 | ORB-short ORB[1400.50,1412.30] vol=3.7x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-11-26 13:35:00 | 1396.49 | 1395.42 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 1416.30 | 1409.31 | 0.00 | ORB-long ORB[1401.00,1412.60] vol=1.9x ATR=4.99 |
| Stop hit — per-position SL triggered | 2025-11-27 09:35:00 | 1411.31 | 1410.05 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 1415.10 | 1409.07 | 0.00 | ORB-long ORB[1396.00,1412.00] vol=2.0x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:40:00 | 1422.53 | 1415.06 | 0.00 | T1 1.5R @ 1422.53 |
| Stop hit — per-position SL triggered | 2025-11-28 10:10:00 | 1415.10 | 1420.73 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:45:00 | 1460.00 | 1451.08 | 0.00 | ORB-long ORB[1433.00,1454.70] vol=3.5x ATR=6.74 |
| Stop hit — per-position SL triggered | 2025-12-04 10:45:00 | 1453.26 | 1457.72 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:35:00 | 1447.80 | 1453.15 | 0.00 | ORB-short ORB[1451.30,1464.70] vol=2.0x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-12-15 11:00:00 | 1451.50 | 1450.82 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 11:05:00 | 1435.80 | 1430.08 | 0.00 | ORB-long ORB[1423.50,1435.00] vol=2.3x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-12-17 11:10:00 | 1432.02 | 1430.60 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:50:00 | 1440.80 | 1430.13 | 0.00 | ORB-long ORB[1421.10,1435.50] vol=3.1x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:30:00 | 1446.57 | 1432.61 | 0.00 | T1 1.5R @ 1446.57 |
| Stop hit — per-position SL triggered | 2025-12-22 12:30:00 | 1440.80 | 1436.86 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 1424.00 | 1429.54 | 0.00 | ORB-short ORB[1427.50,1446.80] vol=3.7x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-12-23 12:10:00 | 1428.19 | 1426.39 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:45:00 | 1422.20 | 1426.91 | 0.00 | ORB-short ORB[1422.50,1434.50] vol=1.6x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:30:00 | 1417.86 | 1424.42 | 0.00 | T1 1.5R @ 1417.86 |
| Target hit | 2025-12-24 15:20:00 | 1414.50 | 1419.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 1405.00 | 1411.88 | 0.00 | ORB-short ORB[1409.70,1420.50] vol=1.7x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 13:00:00 | 1400.36 | 1407.89 | 0.00 | T1 1.5R @ 1400.36 |
| Stop hit — per-position SL triggered | 2025-12-26 13:35:00 | 1405.00 | 1407.44 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:40:00 | 1414.30 | 1422.05 | 0.00 | ORB-short ORB[1419.60,1433.30] vol=1.6x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:55:00 | 1408.50 | 1419.50 | 0.00 | T1 1.5R @ 1408.50 |
| Stop hit — per-position SL triggered | 2025-12-29 12:20:00 | 1414.30 | 1414.85 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 1417.30 | 1403.84 | 0.00 | ORB-long ORB[1386.10,1397.90] vol=2.2x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:10:00 | 1424.75 | 1407.14 | 0.00 | T1 1.5R @ 1424.75 |
| Target hit | 2025-12-31 15:00:00 | 1429.90 | 1433.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 1429.50 | 1425.87 | 0.00 | ORB-long ORB[1416.90,1426.40] vol=3.0x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-01-01 12:10:00 | 1424.66 | 1426.03 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:55:00 | 1412.10 | 1405.31 | 0.00 | ORB-long ORB[1394.00,1412.00] vol=3.5x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:20:00 | 1416.77 | 1407.89 | 0.00 | T1 1.5R @ 1416.77 |
| Stop hit — per-position SL triggered | 2026-01-05 12:35:00 | 1412.10 | 1409.99 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 10:30:00 | 1528.20 | 1516.61 | 0.00 | ORB-long ORB[1501.00,1523.50] vol=1.8x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-01-08 10:50:00 | 1522.54 | 1518.19 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:55:00 | 1493.80 | 1499.42 | 0.00 | ORB-short ORB[1497.10,1516.10] vol=2.5x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:20:00 | 1487.16 | 1497.24 | 0.00 | T1 1.5R @ 1487.16 |
| Stop hit — per-position SL triggered | 2026-01-16 10:35:00 | 1493.80 | 1494.52 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 1481.50 | 1476.27 | 0.00 | ORB-long ORB[1468.00,1481.10] vol=1.6x ATR=4.69 |
| Stop hit — per-position SL triggered | 2026-01-20 09:35:00 | 1476.81 | 1476.14 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:50:00 | 1452.50 | 1468.30 | 0.00 | ORB-short ORB[1462.10,1481.80] vol=1.9x ATR=6.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:55:00 | 1442.04 | 1465.97 | 0.00 | T1 1.5R @ 1442.04 |
| Stop hit — per-position SL triggered | 2026-01-21 13:15:00 | 1452.50 | 1457.02 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 1480.40 | 1484.71 | 0.00 | ORB-short ORB[1483.10,1497.50] vol=1.9x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:10:00 | 1474.48 | 1483.90 | 0.00 | T1 1.5R @ 1474.48 |
| Target hit | 2026-01-23 15:20:00 | 1460.70 | 1472.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-02-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 10:50:00 | 1439.90 | 1419.10 | 0.00 | ORB-long ORB[1397.00,1418.00] vol=2.5x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:00:00 | 1448.63 | 1426.48 | 0.00 | T1 1.5R @ 1448.63 |
| Stop hit — per-position SL triggered | 2026-02-02 11:05:00 | 1439.90 | 1428.06 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:10:00 | 1437.30 | 1446.80 | 0.00 | ORB-short ORB[1455.00,1467.00] vol=3.1x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:10:00 | 1431.40 | 1442.49 | 0.00 | T1 1.5R @ 1431.40 |
| Target hit | 2026-02-10 15:20:00 | 1429.00 | 1434.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1481.70 | 1489.73 | 0.00 | ORB-short ORB[1488.30,1501.80] vol=1.8x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-02-18 12:25:00 | 1486.00 | 1487.14 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:30:00 | 1477.40 | 1488.92 | 0.00 | ORB-short ORB[1485.90,1501.40] vol=1.9x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1481.88 | 1488.00 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 1446.20 | 1453.45 | 0.00 | ORB-short ORB[1447.00,1461.80] vol=2.2x ATR=4.49 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 1450.69 | 1452.14 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 1488.90 | 1478.69 | 0.00 | ORB-long ORB[1465.00,1483.00] vol=1.8x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 1484.15 | 1478.99 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1543.50 | 1532.02 | 0.00 | ORB-long ORB[1519.00,1534.90] vol=2.0x ATR=5.07 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 1538.43 | 1536.16 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:00:00 | 1473.40 | 1485.76 | 0.00 | ORB-short ORB[1486.10,1505.00] vol=5.5x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:20:00 | 1466.83 | 1484.22 | 0.00 | T1 1.5R @ 1466.83 |
| Stop hit — per-position SL triggered | 2026-03-04 13:20:00 | 1473.40 | 1477.57 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 1496.00 | 1497.61 | 0.00 | ORB-short ORB[1501.90,1510.00] vol=3.2x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:25:00 | 1489.82 | 1496.31 | 0.00 | T1 1.5R @ 1489.82 |
| Stop hit — per-position SL triggered | 2026-03-10 13:50:00 | 1496.00 | 1495.11 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 1509.10 | 1497.79 | 0.00 | ORB-long ORB[1487.50,1507.00] vol=2.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:50:00 | 1515.99 | 1501.84 | 0.00 | T1 1.5R @ 1515.99 |
| Target hit | 2026-03-11 15:20:00 | 1529.20 | 1528.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 1536.70 | 1541.41 | 0.00 | ORB-short ORB[1540.70,1562.10] vol=2.2x ATR=4.37 |
| Stop hit — per-position SL triggered | 2026-03-18 12:40:00 | 1541.07 | 1540.54 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1530.00 | 1539.41 | 0.00 | ORB-short ORB[1534.00,1555.00] vol=3.9x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:25:00 | 1520.56 | 1535.11 | 0.00 | T1 1.5R @ 1520.56 |
| Stop hit — per-position SL triggered | 2026-03-20 15:00:00 | 1530.00 | 1529.39 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:35:00 | 1560.00 | 1548.05 | 0.00 | ORB-long ORB[1527.00,1548.80] vol=2.8x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:30:00 | 1572.11 | 1559.00 | 0.00 | T1 1.5R @ 1572.11 |
| Target hit | 2026-03-25 12:35:00 | 1583.00 | 1585.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 84 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1466.00 | 1459.45 | 0.00 | ORB-long ORB[1450.40,1460.00] vol=3.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 1461.85 | 1459.52 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 1450.50 | 1455.35 | 0.00 | ORB-short ORB[1451.80,1471.80] vol=1.9x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 12:30:00 | 1445.26 | 1453.41 | 0.00 | T1 1.5R @ 1445.26 |
| Stop hit — per-position SL triggered | 2026-04-15 13:00:00 | 1450.50 | 1452.91 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1506.80 | 1488.66 | 0.00 | ORB-long ORB[1466.50,1479.90] vol=2.4x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:10:00 | 1513.85 | 1494.73 | 0.00 | T1 1.5R @ 1513.85 |
| Stop hit — per-position SL triggered | 2026-04-22 11:20:00 | 1506.80 | 1498.41 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1536.80 | 1521.74 | 0.00 | ORB-long ORB[1486.30,1506.80] vol=1.6x ATR=8.47 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 1528.33 | 1524.03 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 1540.00 | 1529.48 | 0.00 | ORB-long ORB[1520.00,1536.90] vol=2.0x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:25:00 | 1547.16 | 1531.66 | 0.00 | T1 1.5R @ 1547.16 |
| Stop hit — per-position SL triggered | 2026-04-24 11:45:00 | 1540.00 | 1534.35 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 1525.60 | 1534.12 | 0.00 | ORB-short ORB[1532.50,1548.60] vol=1.6x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 1530.18 | 1533.44 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 1578.50 | 1565.37 | 0.00 | ORB-long ORB[1541.20,1554.70] vol=4.4x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:00:00 | 1588.88 | 1572.70 | 0.00 | T1 1.5R @ 1588.88 |
| Stop hit — per-position SL triggered | 2026-04-30 10:05:00 | 1578.50 | 1573.82 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:40:00 | 1521.00 | 1532.70 | 0.00 | ORB-short ORB[1524.60,1543.50] vol=2.9x ATR=5.48 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 1526.48 | 1528.82 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1563.50 | 1551.54 | 0.00 | ORB-long ORB[1543.20,1559.60] vol=2.2x ATR=5.45 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 1558.05 | 1557.59 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:25:00 | 1587.20 | 1576.75 | 0.00 | ORB-long ORB[1572.10,1581.90] vol=1.7x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-05-07 10:40:00 | 1580.92 | 1577.25 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-05-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:05:00 | 1603.20 | 1589.98 | 0.00 | ORB-long ORB[1560.00,1583.80] vol=5.9x ATR=8.14 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 1595.06 | 1590.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-06-11 10:50:00 | 1368.60 | 2025-06-11 11:00:00 | 1372.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-12 09:45:00 | 1413.50 | 2025-06-12 10:30:00 | 1407.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-16 10:40:00 | 1352.50 | 2025-06-16 11:55:00 | 1355.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-25 10:45:00 | 1333.20 | 2025-06-25 12:45:00 | 1328.74 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-25 10:45:00 | 1333.20 | 2025-06-25 12:50:00 | 1333.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 11:05:00 | 1344.40 | 2025-06-26 12:50:00 | 1339.66 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-06-26 11:05:00 | 1344.40 | 2025-06-26 13:30:00 | 1344.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 10:25:00 | 1361.80 | 2025-06-27 10:30:00 | 1357.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-01 11:10:00 | 1360.00 | 2025-07-01 12:10:00 | 1363.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-03 09:40:00 | 1395.90 | 2025-07-03 10:05:00 | 1403.68 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-07-03 09:40:00 | 1395.90 | 2025-07-03 11:55:00 | 1411.20 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2025-07-07 09:35:00 | 1439.00 | 2025-07-07 09:40:00 | 1432.62 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-07-10 10:25:00 | 1431.90 | 2025-07-10 10:45:00 | 1436.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-11 09:50:00 | 1469.30 | 2025-07-11 09:55:00 | 1465.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-18 10:45:00 | 1464.30 | 2025-07-18 11:20:00 | 1457.12 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-07-18 10:45:00 | 1464.30 | 2025-07-18 11:35:00 | 1464.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 09:30:00 | 1445.30 | 2025-07-22 10:00:00 | 1449.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-05 10:55:00 | 1430.00 | 2025-08-05 11:35:00 | 1424.12 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-05 10:55:00 | 1430.00 | 2025-08-05 11:40:00 | 1430.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:35:00 | 1410.00 | 2025-08-06 12:20:00 | 1415.38 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-08-11 10:30:00 | 1365.20 | 2025-08-11 11:00:00 | 1369.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-13 11:10:00 | 1355.50 | 2025-08-13 11:20:00 | 1348.07 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-08-13 11:10:00 | 1355.50 | 2025-08-13 12:10:00 | 1355.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 10:25:00 | 1354.40 | 2025-08-14 10:55:00 | 1358.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-20 09:35:00 | 1391.00 | 2025-08-20 10:10:00 | 1385.99 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-22 09:40:00 | 1347.00 | 2025-08-22 09:45:00 | 1350.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-29 10:15:00 | 1393.70 | 2025-08-29 10:25:00 | 1387.02 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-29 10:15:00 | 1393.70 | 2025-08-29 10:30:00 | 1393.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-01 10:40:00 | 1367.50 | 2025-09-01 11:05:00 | 1361.49 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-01 10:40:00 | 1367.50 | 2025-09-01 14:00:00 | 1367.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:45:00 | 1371.30 | 2025-09-03 09:55:00 | 1368.16 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-04 09:55:00 | 1344.60 | 2025-09-04 10:05:00 | 1340.14 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-04 09:55:00 | 1344.60 | 2025-09-04 10:10:00 | 1344.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 09:35:00 | 1329.50 | 2025-09-10 09:45:00 | 1326.23 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-11 10:05:00 | 1313.10 | 2025-09-11 13:40:00 | 1316.81 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-12 11:15:00 | 1326.50 | 2025-09-12 11:20:00 | 1323.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-17 09:35:00 | 1323.30 | 2025-09-17 09:45:00 | 1320.27 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-18 09:35:00 | 1319.00 | 2025-09-18 09:55:00 | 1323.56 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-18 09:35:00 | 1319.00 | 2025-09-18 10:00:00 | 1319.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 09:40:00 | 1350.00 | 2025-09-19 09:55:00 | 1345.76 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-24 11:00:00 | 1369.10 | 2025-09-24 11:05:00 | 1364.36 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-24 11:00:00 | 1369.10 | 2025-09-24 15:20:00 | 1344.50 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2025-10-03 10:10:00 | 1317.00 | 2025-10-03 11:55:00 | 1311.70 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-03 10:10:00 | 1317.00 | 2025-10-03 15:00:00 | 1317.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 10:40:00 | 1309.40 | 2025-10-06 10:45:00 | 1312.66 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-07 09:40:00 | 1334.70 | 2025-10-07 09:45:00 | 1342.61 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-10-07 09:40:00 | 1334.70 | 2025-10-07 13:45:00 | 1372.50 | TARGET_HIT | 0.50 | 2.83% |
| SELL | retest1 | 2025-10-08 11:15:00 | 1347.70 | 2025-10-08 11:25:00 | 1350.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-09 10:40:00 | 1346.90 | 2025-10-09 11:50:00 | 1339.88 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-09 10:40:00 | 1346.90 | 2025-10-09 15:20:00 | 1339.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-10-13 10:45:00 | 1324.00 | 2025-10-13 11:30:00 | 1327.16 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-14 10:50:00 | 1311.00 | 2025-10-14 13:25:00 | 1313.93 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-17 10:10:00 | 1283.50 | 2025-10-17 11:45:00 | 1287.04 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-20 09:30:00 | 1267.50 | 2025-10-20 09:40:00 | 1272.21 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-27 09:30:00 | 1294.70 | 2025-10-27 10:05:00 | 1290.17 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-29 10:55:00 | 1293.30 | 2025-10-29 11:15:00 | 1290.27 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-30 10:45:00 | 1278.10 | 2025-10-30 15:00:00 | 1281.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-31 10:40:00 | 1275.70 | 2025-10-31 11:30:00 | 1279.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-03 11:00:00 | 1289.70 | 2025-11-03 11:30:00 | 1294.32 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-03 11:00:00 | 1289.70 | 2025-11-03 15:20:00 | 1307.00 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2025-11-04 09:45:00 | 1311.10 | 2025-11-04 10:15:00 | 1304.26 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-11-04 09:45:00 | 1311.10 | 2025-11-04 12:25:00 | 1307.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-07 10:30:00 | 1279.40 | 2025-11-07 10:45:00 | 1283.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-11 10:55:00 | 1327.60 | 2025-11-11 12:55:00 | 1331.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-12 10:00:00 | 1343.00 | 2025-11-12 10:30:00 | 1348.86 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-12 10:00:00 | 1343.00 | 2025-11-12 11:35:00 | 1343.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 10:20:00 | 1353.90 | 2025-11-13 10:40:00 | 1360.59 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-11-13 10:20:00 | 1353.90 | 2025-11-13 10:50:00 | 1353.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 09:55:00 | 1440.80 | 2025-11-21 10:10:00 | 1435.49 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-26 10:50:00 | 1392.30 | 2025-11-26 13:35:00 | 1396.49 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-27 09:30:00 | 1416.30 | 2025-11-27 09:35:00 | 1411.31 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-28 09:35:00 | 1415.10 | 2025-11-28 09:40:00 | 1422.53 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-28 09:35:00 | 1415.10 | 2025-11-28 10:10:00 | 1415.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 09:45:00 | 1460.00 | 2025-12-04 10:45:00 | 1453.26 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-12-15 10:35:00 | 1447.80 | 2025-12-15 11:00:00 | 1451.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-17 11:05:00 | 1435.80 | 2025-12-17 11:10:00 | 1432.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-22 10:50:00 | 1440.80 | 2025-12-22 11:30:00 | 1446.57 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-22 10:50:00 | 1440.80 | 2025-12-22 12:30:00 | 1440.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-23 10:55:00 | 1424.00 | 2025-12-23 12:10:00 | 1428.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-24 10:45:00 | 1422.20 | 2025-12-24 11:30:00 | 1417.86 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-24 10:45:00 | 1422.20 | 2025-12-24 15:20:00 | 1414.50 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-26 11:05:00 | 1405.00 | 2025-12-26 13:00:00 | 1400.36 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-26 11:05:00 | 1405.00 | 2025-12-26 13:35:00 | 1405.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 10:40:00 | 1414.30 | 2025-12-29 10:55:00 | 1408.50 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-29 10:40:00 | 1414.30 | 2025-12-29 12:20:00 | 1414.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:50:00 | 1417.30 | 2025-12-31 11:10:00 | 1424.75 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-31 10:50:00 | 1417.30 | 2025-12-31 15:00:00 | 1429.90 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2026-01-01 11:15:00 | 1429.50 | 2026-01-01 12:10:00 | 1424.66 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-05 10:55:00 | 1412.10 | 2026-01-05 11:20:00 | 1416.77 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-01-05 10:55:00 | 1412.10 | 2026-01-05 12:35:00 | 1412.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-08 10:30:00 | 1528.20 | 2026-01-08 10:50:00 | 1522.54 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-01-16 09:55:00 | 1493.80 | 2026-01-16 10:20:00 | 1487.16 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-01-16 09:55:00 | 1493.80 | 2026-01-16 10:35:00 | 1493.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-20 09:30:00 | 1481.50 | 2026-01-20 09:35:00 | 1476.81 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-21 10:50:00 | 1452.50 | 2026-01-21 10:55:00 | 1442.04 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-01-21 10:50:00 | 1452.50 | 2026-01-21 13:15:00 | 1452.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1480.40 | 2026-01-23 11:10:00 | 1474.48 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1480.40 | 2026-01-23 15:20:00 | 1460.70 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2026-02-02 10:50:00 | 1439.90 | 2026-02-02 11:00:00 | 1448.63 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-02 10:50:00 | 1439.90 | 2026-02-02 11:05:00 | 1439.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-10 11:10:00 | 1437.30 | 2026-02-10 12:10:00 | 1431.40 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-10 11:10:00 | 1437.30 | 2026-02-10 15:20:00 | 1429.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1481.70 | 2026-02-18 12:25:00 | 1486.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 10:30:00 | 1477.40 | 2026-02-19 10:40:00 | 1481.88 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-20 10:55:00 | 1446.20 | 2026-02-20 11:20:00 | 1450.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-23 11:10:00 | 1488.90 | 2026-02-23 11:20:00 | 1484.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1543.50 | 2026-02-26 10:00:00 | 1538.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1473.40 | 2026-03-04 11:20:00 | 1466.83 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1473.40 | 2026-03-04 13:20:00 | 1473.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 10:40:00 | 1496.00 | 2026-03-10 11:25:00 | 1489.82 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-10 10:40:00 | 1496.00 | 2026-03-10 13:50:00 | 1496.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:25:00 | 1509.10 | 2026-03-11 10:50:00 | 1515.99 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-11 10:25:00 | 1509.10 | 2026-03-11 15:20:00 | 1529.20 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-03-18 11:15:00 | 1536.70 | 2026-03-18 12:40:00 | 1541.07 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-20 10:45:00 | 1530.00 | 2026-03-20 12:25:00 | 1520.56 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-20 10:45:00 | 1530.00 | 2026-03-20 15:00:00 | 1530.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1560.00 | 2026-03-25 10:30:00 | 1572.11 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1560.00 | 2026-03-25 12:35:00 | 1583.00 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2026-04-10 09:30:00 | 1466.00 | 2026-04-10 09:35:00 | 1461.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-15 11:05:00 | 1450.50 | 2026-04-15 12:30:00 | 1445.26 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-15 11:05:00 | 1450.50 | 2026-04-15 13:00:00 | 1450.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1506.80 | 2026-04-22 11:10:00 | 1513.85 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1506.80 | 2026-04-22 11:20:00 | 1506.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:15:00 | 1536.80 | 2026-04-23 10:35:00 | 1528.33 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-04-24 11:10:00 | 1540.00 | 2026-04-24 11:25:00 | 1547.16 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-24 11:10:00 | 1540.00 | 2026-04-24 11:45:00 | 1540.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:20:00 | 1525.60 | 2026-04-29 10:30:00 | 1530.18 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-30 09:50:00 | 1578.50 | 2026-04-30 10:00:00 | 1588.88 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-30 09:50:00 | 1578.50 | 2026-04-30 10:05:00 | 1578.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:40:00 | 1521.00 | 2026-05-05 11:10:00 | 1526.48 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 10:45:00 | 1563.50 | 2026-05-06 11:25:00 | 1558.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-07 10:25:00 | 1587.20 | 2026-05-07 10:40:00 | 1580.92 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-08 10:05:00 | 1603.20 | 2026-05-08 10:10:00 | 1595.06 | STOP_HIT | 1.00 | -0.51% |
