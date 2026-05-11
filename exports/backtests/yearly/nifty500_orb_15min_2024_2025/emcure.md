# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2024-07-10 09:40:00 → 2026-05-08 15:25:00 (33845 bars)
- **Last close:** 1646.00
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 53
- **Target hits / Stop hits / Partials:** 11 / 52 / 32
- **Avg / median % per leg:** 0.31% / 0.00%
- **Sum % (uncompounded):** 29.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 19 | 37.3% | 5 | 32 | 14 | 0.21% | 10.9% |
| BUY @ 2nd Alert (retest1) | 51 | 19 | 37.3% | 5 | 32 | 14 | 0.21% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 23 | 52.3% | 6 | 20 | 18 | 0.41% | 18.1% |
| SELL @ 2nd Alert (retest1) | 44 | 23 | 52.3% | 6 | 20 | 18 | 0.41% | 18.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 42 | 44.2% | 11 | 52 | 32 | 0.31% | 29.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 11:00:00 | 1344.00 | 1349.62 | 0.00 | ORB-short ORB[1345.10,1364.80] vol=3.5x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:10:00 | 1340.25 | 1349.25 | 0.00 | T1 1.5R @ 1340.25 |
| Stop hit — per-position SL triggered | 2024-07-15 11:25:00 | 1344.00 | 1348.49 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 1359.00 | 1360.95 | 0.00 | ORB-short ORB[1360.00,1365.05] vol=1.8x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 1354.18 | 1360.57 | 0.00 | T1 1.5R @ 1354.18 |
| Target hit | 2024-07-19 10:15:00 | 1361.50 | 1360.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 1293.55 | 1299.77 | 0.00 | ORB-short ORB[1301.35,1315.00] vol=3.1x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:50:00 | 1289.33 | 1297.36 | 0.00 | T1 1.5R @ 1289.33 |
| Stop hit — per-position SL triggered | 2024-07-26 12:30:00 | 1293.55 | 1295.84 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:05:00 | 1291.90 | 1293.09 | 0.00 | ORB-short ORB[1292.35,1310.00] vol=2.2x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:25:00 | 1287.56 | 1292.68 | 0.00 | T1 1.5R @ 1287.56 |
| Stop hit — per-position SL triggered | 2024-07-29 12:00:00 | 1291.90 | 1292.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:30:00 | 1306.25 | 1305.15 | 0.00 | ORB-long ORB[1299.95,1305.00] vol=1.6x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-07-31 12:50:00 | 1304.38 | 1306.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:30:00 | 1325.55 | 1317.43 | 0.00 | ORB-long ORB[1302.85,1316.50] vol=4.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-08-01 09:35:00 | 1320.77 | 1318.21 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:00:00 | 1295.00 | 1280.93 | 0.00 | ORB-long ORB[1267.25,1280.00] vol=7.1x ATR=5.61 |
| Stop hit — per-position SL triggered | 2024-08-12 10:10:00 | 1289.39 | 1281.66 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-08-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:35:00 | 1278.50 | 1272.89 | 0.00 | ORB-long ORB[1259.65,1275.00] vol=2.0x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:50:00 | 1285.38 | 1275.91 | 0.00 | T1 1.5R @ 1285.38 |
| Target hit | 2024-08-19 14:40:00 | 1325.00 | 1325.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 1352.50 | 1344.25 | 0.00 | ORB-long ORB[1332.10,1348.10] vol=4.1x ATR=7.44 |
| Stop hit — per-position SL triggered | 2024-08-21 10:05:00 | 1345.06 | 1345.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 1378.00 | 1374.52 | 0.00 | ORB-long ORB[1357.05,1374.00] vol=8.0x ATR=5.19 |
| Stop hit — per-position SL triggered | 2024-08-23 09:45:00 | 1372.81 | 1374.66 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1373.00 | 1382.02 | 0.00 | ORB-short ORB[1380.05,1390.00] vol=2.0x ATR=6.08 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 1379.08 | 1381.45 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1381.60 | 1384.68 | 0.00 | ORB-short ORB[1382.60,1395.00] vol=2.1x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 14:45:00 | 1374.64 | 1381.46 | 0.00 | T1 1.5R @ 1374.64 |
| Target hit | 2024-08-29 15:20:00 | 1373.95 | 1380.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 1395.35 | 1387.12 | 0.00 | ORB-long ORB[1377.40,1392.60] vol=2.3x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:50:00 | 1402.20 | 1390.81 | 0.00 | T1 1.5R @ 1402.20 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 1395.35 | 1391.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:50:00 | 1408.30 | 1396.33 | 0.00 | ORB-long ORB[1385.00,1398.00] vol=2.2x ATR=6.18 |
| Stop hit — per-position SL triggered | 2024-09-04 10:25:00 | 1402.12 | 1398.34 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 1394.85 | 1407.27 | 0.00 | ORB-short ORB[1406.30,1416.00] vol=1.9x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 1385.76 | 1402.50 | 0.00 | T1 1.5R @ 1385.76 |
| Stop hit — per-position SL triggered | 2024-09-06 10:20:00 | 1394.85 | 1399.56 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:15:00 | 1401.30 | 1393.22 | 0.00 | ORB-long ORB[1381.00,1395.50] vol=2.3x ATR=6.60 |
| Stop hit — per-position SL triggered | 2024-09-09 12:20:00 | 1394.70 | 1399.27 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:10:00 | 1423.50 | 1409.76 | 0.00 | ORB-long ORB[1396.75,1415.75] vol=3.1x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:35:00 | 1431.61 | 1420.32 | 0.00 | T1 1.5R @ 1431.61 |
| Stop hit — per-position SL triggered | 2024-09-11 10:40:00 | 1423.50 | 1420.36 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1425.20 | 1430.04 | 0.00 | ORB-short ORB[1427.50,1448.90] vol=3.2x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:35:00 | 1414.33 | 1427.67 | 0.00 | T1 1.5R @ 1414.33 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 1425.20 | 1427.29 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 09:45:00 | 1449.75 | 1458.59 | 0.00 | ORB-short ORB[1451.00,1472.10] vol=3.7x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:10:00 | 1441.13 | 1452.06 | 0.00 | T1 1.5R @ 1441.13 |
| Stop hit — per-position SL triggered | 2024-09-23 12:50:00 | 1449.75 | 1448.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 1439.85 | 1444.88 | 0.00 | ORB-short ORB[1439.90,1454.25] vol=1.5x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:55:00 | 1433.14 | 1442.64 | 0.00 | T1 1.5R @ 1433.14 |
| Target hit | 2024-09-24 15:20:00 | 1428.35 | 1436.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-09-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:45:00 | 1450.00 | 1443.06 | 0.00 | ORB-long ORB[1439.70,1449.30] vol=2.2x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:15:00 | 1456.10 | 1446.16 | 0.00 | T1 1.5R @ 1456.10 |
| Stop hit — per-position SL triggered | 2024-09-27 11:20:00 | 1450.00 | 1447.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 1482.50 | 1477.26 | 0.00 | ORB-long ORB[1470.00,1482.35] vol=13.2x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-10-14 11:15:00 | 1477.94 | 1477.35 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 1491.55 | 1488.90 | 0.00 | ORB-long ORB[1470.00,1490.00] vol=2.1x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-10-16 09:55:00 | 1484.97 | 1488.68 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:50:00 | 1483.75 | 1487.05 | 0.00 | ORB-short ORB[1486.10,1497.85] vol=4.8x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-10-17 10:25:00 | 1489.25 | 1486.19 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1481.90 | 1472.30 | 0.00 | ORB-long ORB[1462.00,1479.40] vol=2.4x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:35:00 | 1491.59 | 1474.07 | 0.00 | T1 1.5R @ 1491.59 |
| Stop hit — per-position SL triggered | 2024-10-18 10:00:00 | 1481.90 | 1484.04 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:35:00 | 1368.45 | 1379.68 | 0.00 | ORB-short ORB[1382.50,1403.05] vol=1.5x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:45:00 | 1355.85 | 1376.78 | 0.00 | T1 1.5R @ 1355.85 |
| Stop hit — per-position SL triggered | 2024-10-25 11:50:00 | 1368.45 | 1367.42 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 1381.90 | 1392.12 | 0.00 | ORB-short ORB[1387.25,1406.80] vol=1.6x ATR=7.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 1370.19 | 1380.54 | 0.00 | T1 1.5R @ 1370.19 |
| Target hit | 2024-11-13 15:20:00 | 1338.85 | 1353.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-11-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:40:00 | 1329.75 | 1317.69 | 0.00 | ORB-long ORB[1295.00,1313.05] vol=5.9x ATR=8.09 |
| Stop hit — per-position SL triggered | 2024-11-22 09:45:00 | 1321.66 | 1318.22 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-11-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 11:05:00 | 1311.30 | 1316.41 | 0.00 | ORB-short ORB[1317.00,1330.05] vol=2.0x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:50:00 | 1305.41 | 1313.77 | 0.00 | T1 1.5R @ 1305.41 |
| Stop hit — per-position SL triggered | 2024-11-25 12:05:00 | 1311.30 | 1313.23 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:00:00 | 1365.20 | 1371.84 | 0.00 | ORB-short ORB[1367.65,1385.00] vol=2.2x ATR=7.41 |
| Stop hit — per-position SL triggered | 2024-11-27 10:30:00 | 1372.61 | 1371.43 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:55:00 | 1390.00 | 1381.21 | 0.00 | ORB-long ORB[1364.60,1384.15] vol=2.5x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 14:35:00 | 1396.18 | 1386.19 | 0.00 | T1 1.5R @ 1396.18 |
| Stop hit — per-position SL triggered | 2024-12-02 14:45:00 | 1390.00 | 1386.31 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:55:00 | 1412.15 | 1408.18 | 0.00 | ORB-long ORB[1389.30,1405.80] vol=17.7x ATR=5.24 |
| Stop hit — per-position SL triggered | 2024-12-03 10:00:00 | 1406.91 | 1408.19 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:15:00 | 1397.05 | 1388.18 | 0.00 | ORB-long ORB[1375.25,1392.75] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2024-12-04 10:25:00 | 1391.89 | 1388.51 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:15:00 | 1361.05 | 1354.48 | 0.00 | ORB-long ORB[1345.10,1360.35] vol=1.8x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:20:00 | 1366.25 | 1355.55 | 0.00 | T1 1.5R @ 1366.25 |
| Stop hit — per-position SL triggered | 2024-12-10 10:25:00 | 1361.05 | 1355.69 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:55:00 | 1366.10 | 1377.76 | 0.00 | ORB-short ORB[1368.50,1388.50] vol=2.4x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 13:40:00 | 1359.25 | 1375.96 | 0.00 | T1 1.5R @ 1359.25 |
| Stop hit — per-position SL triggered | 2024-12-11 14:30:00 | 1366.10 | 1374.22 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:40:00 | 1354.75 | 1351.84 | 0.00 | ORB-long ORB[1343.00,1354.65] vol=1.9x ATR=4.26 |
| Stop hit — per-position SL triggered | 2024-12-13 09:45:00 | 1350.49 | 1351.92 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:50:00 | 1387.65 | 1377.32 | 0.00 | ORB-long ORB[1359.90,1378.45] vol=8.0x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-12-16 09:55:00 | 1381.89 | 1377.79 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:45:00 | 1402.50 | 1396.48 | 0.00 | ORB-long ORB[1384.00,1400.00] vol=1.6x ATR=4.62 |
| Stop hit — per-position SL triggered | 2024-12-18 09:50:00 | 1397.88 | 1397.34 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 11:15:00 | 1365.00 | 1371.86 | 0.00 | ORB-short ORB[1369.55,1379.00] vol=1.8x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-12-20 11:25:00 | 1368.64 | 1371.80 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 1398.95 | 1391.47 | 0.00 | ORB-long ORB[1382.55,1394.75] vol=2.9x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:40:00 | 1405.50 | 1398.32 | 0.00 | T1 1.5R @ 1405.50 |
| Target hit | 2024-12-24 10:35:00 | 1400.60 | 1400.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1452.65 | 1443.99 | 0.00 | ORB-long ORB[1432.05,1450.00] vol=1.8x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-12-30 11:10:00 | 1445.14 | 1450.42 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 1454.60 | 1459.82 | 0.00 | ORB-short ORB[1460.25,1475.00] vol=2.8x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-01-02 11:20:00 | 1458.41 | 1458.93 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:30:00 | 1465.45 | 1472.07 | 0.00 | ORB-short ORB[1474.85,1491.90] vol=2.0x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:35:00 | 1458.61 | 1469.57 | 0.00 | T1 1.5R @ 1458.61 |
| Stop hit — per-position SL triggered | 2025-01-03 11:55:00 | 1465.45 | 1468.74 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1448.00 | 1454.68 | 0.00 | ORB-short ORB[1450.00,1463.90] vol=3.0x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:55:00 | 1441.44 | 1449.42 | 0.00 | T1 1.5R @ 1441.44 |
| Stop hit — per-position SL triggered | 2025-01-06 15:05:00 | 1448.00 | 1443.45 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:40:00 | 1454.65 | 1448.25 | 0.00 | ORB-long ORB[1435.60,1447.95] vol=4.7x ATR=6.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:30:00 | 1463.93 | 1454.20 | 0.00 | T1 1.5R @ 1463.93 |
| Stop hit — per-position SL triggered | 2025-01-09 11:20:00 | 1454.65 | 1455.40 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:50:00 | 1402.55 | 1411.37 | 0.00 | ORB-short ORB[1415.05,1431.65] vol=3.5x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:00:00 | 1392.39 | 1410.23 | 0.00 | T1 1.5R @ 1392.39 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 1402.55 | 1404.59 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:25:00 | 1370.70 | 1355.35 | 0.00 | ORB-long ORB[1350.00,1359.20] vol=2.5x ATR=5.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:35:00 | 1379.38 | 1357.07 | 0.00 | T1 1.5R @ 1379.38 |
| Stop hit — per-position SL triggered | 2025-01-15 11:05:00 | 1370.70 | 1362.34 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:05:00 | 1370.65 | 1377.05 | 0.00 | ORB-short ORB[1380.65,1397.40] vol=2.0x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-01-21 10:10:00 | 1375.18 | 1376.68 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:25:00 | 1333.00 | 1335.39 | 0.00 | ORB-short ORB[1336.30,1353.25] vol=15.1x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:35:00 | 1327.45 | 1334.97 | 0.00 | T1 1.5R @ 1327.45 |
| Target hit | 2025-01-24 15:20:00 | 1286.45 | 1325.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:55:00 | 1236.75 | 1223.80 | 0.00 | ORB-long ORB[1211.30,1224.80] vol=3.3x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:15:00 | 1244.95 | 1227.93 | 0.00 | T1 1.5R @ 1244.95 |
| Stop hit — per-position SL triggered | 2025-01-30 10:40:00 | 1236.75 | 1229.32 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:45:00 | 1231.35 | 1227.17 | 0.00 | ORB-long ORB[1220.15,1228.45] vol=6.2x ATR=4.76 |
| Stop hit — per-position SL triggered | 2025-01-31 09:50:00 | 1226.59 | 1227.54 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-02-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:05:00 | 1260.00 | 1247.42 | 0.00 | ORB-long ORB[1232.05,1250.00] vol=3.6x ATR=5.07 |
| Stop hit — per-position SL triggered | 2025-02-01 10:10:00 | 1254.93 | 1248.80 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 09:40:00 | 1251.20 | 1255.00 | 0.00 | ORB-short ORB[1253.80,1266.15] vol=1.5x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-02-05 09:55:00 | 1255.16 | 1255.24 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:15:00 | 960.00 | 954.93 | 0.00 | ORB-long ORB[940.75,954.85] vol=4.2x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:55:00 | 964.63 | 956.37 | 0.00 | T1 1.5R @ 964.63 |
| Target hit | 2025-03-05 15:20:00 | 967.00 | 963.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 995.10 | 984.79 | 0.00 | ORB-long ORB[972.05,984.35] vol=4.2x ATR=4.02 |
| Stop hit — per-position SL triggered | 2025-03-06 11:20:00 | 991.08 | 985.29 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:45:00 | 991.35 | 987.32 | 0.00 | ORB-long ORB[980.30,986.45] vol=1.5x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-03-07 10:50:00 | 988.01 | 987.64 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 954.00 | 948.21 | 0.00 | ORB-long ORB[936.00,949.90] vol=1.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 949.50 | 948.66 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:15:00 | 917.20 | 919.87 | 0.00 | ORB-short ORB[921.90,934.45] vol=1.7x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-03-17 10:35:00 | 921.20 | 919.49 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 1020.00 | 1012.33 | 0.00 | ORB-long ORB[994.10,1009.00] vol=2.2x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:30:00 | 1027.25 | 1017.43 | 0.00 | T1 1.5R @ 1027.25 |
| Target hit | 2025-03-21 12:50:00 | 1025.00 | 1026.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2025-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:25:00 | 1044.35 | 1058.30 | 0.00 | ORB-short ORB[1056.70,1069.00] vol=1.7x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 10:30:00 | 1037.51 | 1054.80 | 0.00 | T1 1.5R @ 1037.51 |
| Target hit | 2025-03-25 15:20:00 | 1006.95 | 1023.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-03-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:05:00 | 1031.40 | 1022.73 | 0.00 | ORB-long ORB[1009.95,1024.50] vol=1.9x ATR=6.24 |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 1025.16 | 1023.82 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:15:00 | 1027.20 | 1017.08 | 0.00 | ORB-long ORB[1008.50,1017.40] vol=1.7x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:35:00 | 1036.39 | 1022.37 | 0.00 | T1 1.5R @ 1036.39 |
| Target hit | 2025-04-17 15:20:00 | 1103.30 | 1074.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:00:00 | 1030.40 | 1025.01 | 0.00 | ORB-long ORB[1016.50,1029.90] vol=3.5x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-05-08 10:05:00 | 1026.18 | 1025.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-07-15 11:00:00 | 1344.00 | 2024-07-15 11:10:00 | 1340.25 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-07-15 11:00:00 | 1344.00 | 2024-07-15 11:25:00 | 1344.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:00:00 | 1359.00 | 2024-07-19 10:15:00 | 1354.18 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-19 10:00:00 | 1359.00 | 2024-07-19 10:15:00 | 1361.50 | TARGET_HIT | 0.50 | -0.18% |
| SELL | retest1 | 2024-07-26 11:05:00 | 1293.55 | 2024-07-26 11:50:00 | 1289.33 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-07-26 11:05:00 | 1293.55 | 2024-07-26 12:30:00 | 1293.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-29 11:05:00 | 1291.90 | 2024-07-29 11:25:00 | 1287.56 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-07-29 11:05:00 | 1291.90 | 2024-07-29 12:00:00 | 1291.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:30:00 | 1306.25 | 2024-07-31 12:50:00 | 1304.38 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-08-01 09:30:00 | 1325.55 | 2024-08-01 09:35:00 | 1320.77 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-12 10:00:00 | 1295.00 | 2024-08-12 10:10:00 | 1289.39 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-19 10:35:00 | 1278.50 | 2024-08-19 10:50:00 | 1285.38 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-19 10:35:00 | 1278.50 | 2024-08-19 14:40:00 | 1325.00 | TARGET_HIT | 0.50 | 3.64% |
| BUY | retest1 | 2024-08-21 09:45:00 | 1352.50 | 2024-08-21 10:05:00 | 1345.06 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-08-23 09:30:00 | 1378.00 | 2024-08-23 09:45:00 | 1372.81 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1373.00 | 2024-08-28 09:35:00 | 1379.08 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1381.60 | 2024-08-29 14:45:00 | 1374.64 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1381.60 | 2024-08-29 15:20:00 | 1373.95 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2024-09-03 09:30:00 | 1395.35 | 2024-09-03 09:50:00 | 1402.20 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-03 09:30:00 | 1395.35 | 2024-09-03 09:55:00 | 1395.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:50:00 | 1408.30 | 2024-09-04 10:25:00 | 1402.12 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-06 09:50:00 | 1394.85 | 2024-09-06 10:05:00 | 1385.76 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-09-06 09:50:00 | 1394.85 | 2024-09-06 10:20:00 | 1394.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-09 10:15:00 | 1401.30 | 2024-09-09 12:20:00 | 1394.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-09-11 10:10:00 | 1423.50 | 2024-09-11 10:35:00 | 1431.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-11 10:10:00 | 1423.50 | 2024-09-11 10:40:00 | 1423.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-13 09:30:00 | 1425.20 | 2024-09-13 09:35:00 | 1414.33 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-09-13 09:30:00 | 1425.20 | 2024-09-13 09:50:00 | 1425.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-23 09:45:00 | 1449.75 | 2024-09-23 11:10:00 | 1441.13 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-23 09:45:00 | 1449.75 | 2024-09-23 12:50:00 | 1449.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 09:55:00 | 1439.85 | 2024-09-24 11:55:00 | 1433.14 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-09-24 09:55:00 | 1439.85 | 2024-09-24 15:20:00 | 1428.35 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2024-09-27 10:45:00 | 1450.00 | 2024-09-27 11:15:00 | 1456.10 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-27 10:45:00 | 1450.00 | 2024-09-27 11:20:00 | 1450.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 11:10:00 | 1482.50 | 2024-10-14 11:15:00 | 1477.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-16 09:30:00 | 1491.55 | 2024-10-16 09:55:00 | 1484.97 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-17 09:50:00 | 1483.75 | 2024-10-17 10:25:00 | 1489.25 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1481.90 | 2024-10-18 09:35:00 | 1491.59 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1481.90 | 2024-10-18 10:00:00 | 1481.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:35:00 | 1368.45 | 2024-10-25 10:45:00 | 1355.85 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2024-10-25 10:35:00 | 1368.45 | 2024-10-25 11:50:00 | 1368.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1381.90 | 2024-11-13 09:40:00 | 1370.19 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1381.90 | 2024-11-13 15:20:00 | 1338.85 | TARGET_HIT | 0.50 | 3.12% |
| BUY | retest1 | 2024-11-22 09:40:00 | 1329.75 | 2024-11-22 09:45:00 | 1321.66 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-11-25 11:05:00 | 1311.30 | 2024-11-25 11:50:00 | 1305.41 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-11-25 11:05:00 | 1311.30 | 2024-11-25 12:05:00 | 1311.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 10:00:00 | 1365.20 | 2024-11-27 10:30:00 | 1372.61 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-12-02 10:55:00 | 1390.00 | 2024-12-02 14:35:00 | 1396.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-02 10:55:00 | 1390.00 | 2024-12-02 14:45:00 | 1390.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 09:55:00 | 1412.15 | 2024-12-03 10:00:00 | 1406.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-04 10:15:00 | 1397.05 | 2024-12-04 10:25:00 | 1391.89 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-10 10:15:00 | 1361.05 | 2024-12-10 10:20:00 | 1366.25 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-10 10:15:00 | 1361.05 | 2024-12-10 10:25:00 | 1361.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-11 10:55:00 | 1366.10 | 2024-12-11 13:40:00 | 1359.25 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-11 10:55:00 | 1366.10 | 2024-12-11 14:30:00 | 1366.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-13 09:40:00 | 1354.75 | 2024-12-13 09:45:00 | 1350.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-16 09:50:00 | 1387.65 | 2024-12-16 09:55:00 | 1381.89 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-18 09:45:00 | 1402.50 | 2024-12-18 09:50:00 | 1397.88 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-20 11:15:00 | 1365.00 | 2024-12-20 11:25:00 | 1368.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1398.95 | 2024-12-24 09:40:00 | 1405.50 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1398.95 | 2024-12-24 10:35:00 | 1400.60 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1452.65 | 2024-12-30 11:10:00 | 1445.14 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-01-02 11:05:00 | 1454.60 | 2025-01-02 11:20:00 | 1458.41 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-03 10:30:00 | 1465.45 | 2025-01-03 11:35:00 | 1458.61 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-03 10:30:00 | 1465.45 | 2025-01-03 11:55:00 | 1465.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1448.00 | 2025-01-06 11:55:00 | 1441.44 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1448.00 | 2025-01-06 15:05:00 | 1448.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:40:00 | 1454.65 | 2025-01-09 10:30:00 | 1463.93 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-09 09:40:00 | 1454.65 | 2025-01-09 11:20:00 | 1454.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:50:00 | 1402.55 | 2025-01-10 10:00:00 | 1392.39 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-01-10 09:50:00 | 1402.55 | 2025-01-10 10:05:00 | 1402.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-15 10:25:00 | 1370.70 | 2025-01-15 10:35:00 | 1379.38 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-15 10:25:00 | 1370.70 | 2025-01-15 11:05:00 | 1370.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 10:05:00 | 1370.65 | 2025-01-21 10:10:00 | 1375.18 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-24 10:25:00 | 1333.00 | 2025-01-24 10:35:00 | 1327.45 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-24 10:25:00 | 1333.00 | 2025-01-24 15:20:00 | 1286.45 | TARGET_HIT | 0.50 | 3.49% |
| BUY | retest1 | 2025-01-30 09:55:00 | 1236.75 | 2025-01-30 10:15:00 | 1244.95 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-30 09:55:00 | 1236.75 | 2025-01-30 10:40:00 | 1236.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 09:45:00 | 1231.35 | 2025-01-31 09:50:00 | 1226.59 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-01 10:05:00 | 1260.00 | 2025-02-01 10:10:00 | 1254.93 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-05 09:40:00 | 1251.20 | 2025-02-05 09:55:00 | 1255.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-05 11:15:00 | 960.00 | 2025-03-05 11:55:00 | 964.63 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-05 11:15:00 | 960.00 | 2025-03-05 15:20:00 | 967.00 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2025-03-06 11:05:00 | 995.10 | 2025-03-06 11:20:00 | 991.08 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-07 10:45:00 | 991.35 | 2025-03-07 10:50:00 | 988.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-11 11:00:00 | 954.00 | 2025-03-11 11:15:00 | 949.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-17 10:15:00 | 917.20 | 2025-03-17 10:35:00 | 921.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1020.00 | 2025-03-21 11:30:00 | 1027.25 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1020.00 | 2025-03-21 12:50:00 | 1025.00 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2025-03-25 10:25:00 | 1044.35 | 2025-03-25 10:30:00 | 1037.51 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-03-25 10:25:00 | 1044.35 | 2025-03-25 15:20:00 | 1006.95 | TARGET_HIT | 0.50 | 3.58% |
| BUY | retest1 | 2025-03-26 10:05:00 | 1031.40 | 2025-03-26 10:15:00 | 1025.16 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-04-17 10:15:00 | 1027.20 | 2025-04-17 10:35:00 | 1036.39 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-04-17 10:15:00 | 1027.20 | 2025-04-17 15:20:00 | 1103.30 | TARGET_HIT | 0.50 | 7.41% |
| BUY | retest1 | 2025-05-08 10:00:00 | 1030.40 | 2025-05-08 10:05:00 | 1026.18 | STOP_HIT | 1.00 | -0.41% |
