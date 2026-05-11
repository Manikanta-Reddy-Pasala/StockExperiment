# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-31 15:25:00 (27421 bars)
- **Last close:** 1202.30
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
| ENTRY1 | 88 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 16 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 72
- **Target hits / Stop hits / Partials:** 16 / 72 / 39
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 19.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 31 | 44.3% | 9 | 39 | 22 | 0.14% | 9.5% |
| BUY @ 2nd Alert (retest1) | 70 | 31 | 44.3% | 9 | 39 | 22 | 0.14% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 24 | 42.1% | 7 | 33 | 17 | 0.18% | 10.4% |
| SELL @ 2nd Alert (retest1) | 57 | 24 | 42.1% | 7 | 33 | 17 | 0.18% | 10.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 55 | 43.3% | 16 | 72 | 39 | 0.16% | 19.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 1321.00 | 1316.62 | 0.00 | ORB-long ORB[1311.05,1320.80] vol=1.6x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 12:15:00 | 1327.71 | 1320.95 | 0.00 | T1 1.5R @ 1327.71 |
| Stop hit — per-position SL triggered | 2024-05-14 13:20:00 | 1321.00 | 1322.94 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:10:00 | 1300.05 | 1295.40 | 0.00 | ORB-long ORB[1287.50,1298.90] vol=2.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-05-16 10:20:00 | 1296.40 | 1297.09 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:50:00 | 1335.70 | 1339.57 | 0.00 | ORB-short ORB[1336.65,1347.85] vol=1.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-05-23 10:10:00 | 1338.81 | 1339.06 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 1334.30 | 1339.50 | 0.00 | ORB-short ORB[1336.85,1350.00] vol=2.4x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:00:00 | 1328.40 | 1337.35 | 0.00 | T1 1.5R @ 1328.40 |
| Stop hit — per-position SL triggered | 2024-05-27 10:10:00 | 1334.30 | 1336.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:35:00 | 1334.75 | 1328.75 | 0.00 | ORB-long ORB[1322.70,1332.10] vol=3.4x ATR=5.20 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 1329.55 | 1331.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 1339.80 | 1336.00 | 0.00 | ORB-long ORB[1329.30,1337.45] vol=2.7x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:55:00 | 1345.55 | 1339.79 | 0.00 | T1 1.5R @ 1345.55 |
| Target hit | 2024-06-07 10:20:00 | 1343.05 | 1344.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2024-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:20:00 | 1349.75 | 1340.95 | 0.00 | ORB-long ORB[1332.20,1345.00] vol=1.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:30:00 | 1354.84 | 1342.39 | 0.00 | T1 1.5R @ 1354.84 |
| Target hit | 2024-06-10 12:10:00 | 1350.30 | 1352.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 1364.00 | 1358.02 | 0.00 | ORB-long ORB[1348.05,1361.35] vol=2.0x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-06-11 10:35:00 | 1359.91 | 1360.38 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 1404.95 | 1396.18 | 0.00 | ORB-long ORB[1385.05,1403.85] vol=1.5x ATR=5.17 |
| Stop hit — per-position SL triggered | 2024-06-12 10:00:00 | 1399.78 | 1396.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:50:00 | 1383.35 | 1391.36 | 0.00 | ORB-short ORB[1390.10,1400.25] vol=1.8x ATR=2.95 |
| Stop hit — per-position SL triggered | 2024-06-14 11:00:00 | 1386.30 | 1390.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 11:15:00 | 1389.10 | 1393.80 | 0.00 | ORB-short ORB[1391.05,1403.50] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-06-18 11:25:00 | 1391.26 | 1393.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:25:00 | 1386.55 | 1396.51 | 0.00 | ORB-short ORB[1394.00,1409.00] vol=1.7x ATR=3.77 |
| Stop hit — per-position SL triggered | 2024-06-19 10:55:00 | 1390.32 | 1393.35 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 1387.55 | 1390.14 | 0.00 | ORB-short ORB[1390.10,1396.25] vol=1.7x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-06-20 11:20:00 | 1389.79 | 1390.07 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 1391.85 | 1387.97 | 0.00 | ORB-long ORB[1382.95,1389.00] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:35:00 | 1395.56 | 1395.03 | 0.00 | T1 1.5R @ 1395.56 |
| Target hit | 2024-06-21 13:05:00 | 1432.00 | 1432.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 1441.00 | 1427.90 | 0.00 | ORB-long ORB[1417.00,1434.95] vol=3.5x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-06-25 09:50:00 | 1436.25 | 1429.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 1444.70 | 1439.52 | 0.00 | ORB-long ORB[1432.35,1442.95] vol=3.1x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:50:00 | 1450.84 | 1444.00 | 0.00 | T1 1.5R @ 1450.84 |
| Stop hit — per-position SL triggered | 2024-06-26 09:55:00 | 1444.70 | 1444.28 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 1469.80 | 1461.11 | 0.00 | ORB-long ORB[1455.30,1463.95] vol=2.0x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:15:00 | 1475.26 | 1467.22 | 0.00 | T1 1.5R @ 1475.26 |
| Stop hit — per-position SL triggered | 2024-07-04 10:25:00 | 1469.80 | 1467.51 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 1479.80 | 1472.29 | 0.00 | ORB-long ORB[1465.70,1475.75] vol=2.5x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-07-05 11:05:00 | 1475.93 | 1472.65 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 1459.55 | 1465.02 | 0.00 | ORB-short ORB[1462.55,1477.90] vol=3.1x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:10:00 | 1454.15 | 1463.46 | 0.00 | T1 1.5R @ 1454.15 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 1459.55 | 1463.37 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 1442.00 | 1450.72 | 0.00 | ORB-short ORB[1451.55,1465.75] vol=1.7x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 1436.39 | 1447.85 | 0.00 | T1 1.5R @ 1436.39 |
| Stop hit — per-position SL triggered | 2024-07-10 11:40:00 | 1442.00 | 1439.79 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:40:00 | 1448.65 | 1454.28 | 0.00 | ORB-short ORB[1453.00,1462.40] vol=1.6x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-07-16 09:45:00 | 1452.92 | 1453.71 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1416.60 | 1424.69 | 0.00 | ORB-short ORB[1422.00,1438.00] vol=2.2x ATR=5.41 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 1422.01 | 1422.27 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 09:35:00 | 1426.35 | 1416.80 | 0.00 | ORB-long ORB[1402.30,1420.75] vol=2.7x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-07-19 09:40:00 | 1421.04 | 1417.80 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:40:00 | 1496.30 | 1490.91 | 0.00 | ORB-long ORB[1481.00,1495.00] vol=2.2x ATR=6.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:25:00 | 1505.59 | 1494.09 | 0.00 | T1 1.5R @ 1505.59 |
| Stop hit — per-position SL triggered | 2024-07-26 11:05:00 | 1496.30 | 1496.45 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 1529.10 | 1524.76 | 0.00 | ORB-long ORB[1516.00,1529.00] vol=2.0x ATR=3.97 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 1525.13 | 1524.89 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:05:00 | 1490.85 | 1494.44 | 0.00 | ORB-short ORB[1494.60,1512.00] vol=2.4x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-08-01 11:25:00 | 1495.18 | 1494.21 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 1442.70 | 1430.30 | 0.00 | ORB-long ORB[1414.95,1435.50] vol=1.7x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:45:00 | 1450.86 | 1435.77 | 0.00 | T1 1.5R @ 1450.86 |
| Target hit | 2024-08-07 15:20:00 | 1465.30 | 1446.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 1439.30 | 1451.82 | 0.00 | ORB-short ORB[1456.30,1472.00] vol=1.8x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 12:30:00 | 1431.89 | 1446.68 | 0.00 | T1 1.5R @ 1431.89 |
| Target hit | 2024-08-14 15:20:00 | 1432.95 | 1438.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-08-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:25:00 | 1529.55 | 1523.87 | 0.00 | ORB-long ORB[1517.10,1526.00] vol=6.3x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-08-22 10:45:00 | 1525.65 | 1525.76 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 1496.55 | 1505.22 | 0.00 | ORB-short ORB[1505.00,1520.00] vol=3.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-08-23 10:45:00 | 1501.03 | 1499.13 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:05:00 | 1502.00 | 1495.12 | 0.00 | ORB-long ORB[1491.00,1499.00] vol=1.6x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 1497.97 | 1496.30 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:50:00 | 1542.25 | 1534.45 | 0.00 | ORB-long ORB[1518.10,1534.35] vol=3.0x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:00:00 | 1548.06 | 1538.35 | 0.00 | T1 1.5R @ 1548.06 |
| Stop hit — per-position SL triggered | 2024-08-28 10:45:00 | 1542.25 | 1541.46 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:45:00 | 1522.20 | 1516.24 | 0.00 | ORB-long ORB[1511.10,1520.00] vol=1.9x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-08-29 10:30:00 | 1517.91 | 1517.89 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1528.75 | 1519.09 | 0.00 | ORB-long ORB[1511.90,1518.45] vol=2.4x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-08-30 10:10:00 | 1524.12 | 1520.28 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:50:00 | 1523.10 | 1520.31 | 0.00 | ORB-long ORB[1514.00,1520.80] vol=1.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-09-03 10:05:00 | 1520.25 | 1520.49 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:40:00 | 1530.35 | 1518.07 | 0.00 | ORB-long ORB[1502.05,1520.20] vol=1.7x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:55:00 | 1537.57 | 1524.37 | 0.00 | T1 1.5R @ 1537.57 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 1530.35 | 1527.05 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1557.70 | 1546.08 | 0.00 | ORB-long ORB[1528.30,1539.90] vol=7.2x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:35:00 | 1565.12 | 1554.36 | 0.00 | T1 1.5R @ 1565.12 |
| Stop hit — per-position SL triggered | 2024-09-05 09:55:00 | 1557.70 | 1556.94 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:55:00 | 1590.35 | 1594.33 | 0.00 | ORB-short ORB[1592.75,1604.00] vol=1.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-09-12 12:20:00 | 1594.31 | 1593.04 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:35:00 | 1682.50 | 1671.23 | 0.00 | ORB-long ORB[1657.00,1669.95] vol=2.2x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 1676.66 | 1673.98 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 1689.50 | 1682.45 | 0.00 | ORB-long ORB[1663.55,1686.45] vol=3.2x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:40:00 | 1696.36 | 1684.27 | 0.00 | T1 1.5R @ 1696.36 |
| Target hit | 2024-09-18 11:20:00 | 1698.55 | 1701.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 1674.20 | 1663.64 | 0.00 | ORB-long ORB[1652.10,1664.50] vol=1.5x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-09-23 11:25:00 | 1670.71 | 1664.36 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:55:00 | 1694.50 | 1700.50 | 0.00 | ORB-short ORB[1699.30,1708.70] vol=1.9x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-09-25 10:05:00 | 1698.34 | 1699.37 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-09-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:10:00 | 1713.85 | 1723.58 | 0.00 | ORB-short ORB[1717.00,1738.05] vol=1.8x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:35:00 | 1705.84 | 1722.43 | 0.00 | T1 1.5R @ 1705.84 |
| Target hit | 2024-09-27 15:20:00 | 1677.70 | 1706.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 1634.80 | 1626.54 | 0.00 | ORB-long ORB[1612.05,1631.60] vol=2.4x ATR=5.89 |
| Stop hit — per-position SL triggered | 2024-10-03 09:55:00 | 1628.91 | 1628.42 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:55:00 | 1631.70 | 1616.12 | 0.00 | ORB-long ORB[1605.05,1624.00] vol=1.7x ATR=6.04 |
| Stop hit — per-position SL triggered | 2024-10-04 11:00:00 | 1625.66 | 1616.54 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1576.05 | 1590.05 | 0.00 | ORB-short ORB[1604.00,1620.00] vol=1.8x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1583.56 | 1589.09 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1565.25 | 1556.61 | 0.00 | ORB-long ORB[1548.15,1560.00] vol=1.7x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:40:00 | 1574.73 | 1559.62 | 0.00 | T1 1.5R @ 1574.73 |
| Target hit | 2024-10-08 11:55:00 | 1575.35 | 1576.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1610.50 | 1599.57 | 0.00 | ORB-long ORB[1588.45,1608.00] vol=1.5x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 09:50:00 | 1618.80 | 1604.33 | 0.00 | T1 1.5R @ 1618.80 |
| Stop hit — per-position SL triggered | 2024-10-10 09:55:00 | 1610.50 | 1604.58 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 1622.30 | 1615.22 | 0.00 | ORB-long ORB[1607.05,1618.95] vol=3.5x ATR=4.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:55:00 | 1629.76 | 1618.81 | 0.00 | T1 1.5R @ 1629.76 |
| Target hit | 2024-10-11 10:20:00 | 1623.85 | 1625.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2024-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1614.50 | 1606.30 | 0.00 | ORB-long ORB[1595.05,1612.95] vol=4.0x ATR=6.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:45:00 | 1624.43 | 1610.40 | 0.00 | T1 1.5R @ 1624.43 |
| Stop hit — per-position SL triggered | 2024-10-18 10:00:00 | 1614.50 | 1611.58 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:15:00 | 1511.45 | 1525.45 | 0.00 | ORB-short ORB[1517.50,1539.00] vol=5.1x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 11:30:00 | 1503.75 | 1522.91 | 0.00 | T1 1.5R @ 1503.75 |
| Target hit | 2024-10-24 15:20:00 | 1509.90 | 1510.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 1497.60 | 1505.13 | 0.00 | ORB-short ORB[1500.95,1518.50] vol=1.6x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 1490.33 | 1502.24 | 0.00 | T1 1.5R @ 1490.33 |
| Stop hit — per-position SL triggered | 2024-10-25 09:55:00 | 1497.60 | 1498.38 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:30:00 | 1496.60 | 1516.23 | 0.00 | ORB-short ORB[1522.55,1537.30] vol=1.8x ATR=5.09 |
| Stop hit — per-position SL triggered | 2024-10-29 10:35:00 | 1501.69 | 1515.64 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:15:00 | 1542.55 | 1537.59 | 0.00 | ORB-long ORB[1518.25,1538.15] vol=3.4x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-10-30 10:25:00 | 1537.78 | 1537.71 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 1517.20 | 1510.43 | 0.00 | ORB-long ORB[1502.05,1513.10] vol=2.3x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-11-06 09:40:00 | 1511.83 | 1511.22 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:55:00 | 1476.25 | 1488.68 | 0.00 | ORB-short ORB[1482.00,1500.00] vol=2.0x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 12:00:00 | 1468.90 | 1485.05 | 0.00 | T1 1.5R @ 1468.90 |
| Target hit | 2024-11-08 15:20:00 | 1467.40 | 1479.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2024-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:30:00 | 1474.35 | 1463.05 | 0.00 | ORB-long ORB[1452.50,1469.25] vol=2.2x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:45:00 | 1482.10 | 1468.51 | 0.00 | T1 1.5R @ 1482.10 |
| Stop hit — per-position SL triggered | 2024-11-11 12:05:00 | 1474.35 | 1473.43 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-11-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:00:00 | 1469.55 | 1460.05 | 0.00 | ORB-long ORB[1447.00,1464.95] vol=1.6x ATR=5.87 |
| Stop hit — per-position SL triggered | 2024-11-14 10:25:00 | 1463.68 | 1461.39 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-11-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 10:45:00 | 1438.00 | 1448.82 | 0.00 | ORB-short ORB[1443.05,1458.90] vol=1.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-11-22 10:55:00 | 1441.79 | 1447.56 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:50:00 | 1470.90 | 1474.81 | 0.00 | ORB-short ORB[1472.85,1488.55] vol=2.6x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:20:00 | 1464.15 | 1471.67 | 0.00 | T1 1.5R @ 1464.15 |
| Stop hit — per-position SL triggered | 2024-11-26 11:45:00 | 1470.90 | 1471.06 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 1466.00 | 1474.96 | 0.00 | ORB-short ORB[1469.40,1484.55] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-11-27 09:45:00 | 1470.17 | 1472.95 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 1538.05 | 1528.17 | 0.00 | ORB-long ORB[1510.05,1530.00] vol=6.0x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-11-28 10:25:00 | 1531.99 | 1530.10 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1546.75 | 1541.48 | 0.00 | ORB-long ORB[1525.00,1543.50] vol=3.0x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-11-29 09:35:00 | 1540.21 | 1540.09 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 1591.40 | 1579.85 | 0.00 | ORB-long ORB[1563.10,1583.00] vol=3.2x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 13:15:00 | 1600.57 | 1588.18 | 0.00 | T1 1.5R @ 1600.57 |
| Target hit | 2024-12-03 15:20:00 | 1598.25 | 1591.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2024-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 1589.75 | 1599.80 | 0.00 | ORB-short ORB[1591.65,1610.00] vol=1.7x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-12-04 11:10:00 | 1593.73 | 1599.37 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:50:00 | 1481.15 | 1489.75 | 0.00 | ORB-short ORB[1487.85,1503.25] vol=2.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-12-11 11:35:00 | 1483.77 | 1488.55 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 1468.95 | 1474.69 | 0.00 | ORB-short ORB[1471.40,1486.90] vol=1.8x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:10:00 | 1460.35 | 1468.80 | 0.00 | T1 1.5R @ 1460.35 |
| Target hit | 2024-12-12 15:20:00 | 1457.50 | 1464.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2024-12-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:35:00 | 1441.65 | 1448.94 | 0.00 | ORB-short ORB[1447.00,1463.95] vol=2.1x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 1446.12 | 1448.16 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1475.70 | 1469.58 | 0.00 | ORB-long ORB[1455.30,1475.00] vol=2.6x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:45:00 | 1482.62 | 1473.98 | 0.00 | T1 1.5R @ 1482.62 |
| Stop hit — per-position SL triggered | 2024-12-16 09:55:00 | 1475.70 | 1474.52 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:15:00 | 1412.00 | 1417.50 | 0.00 | ORB-short ORB[1418.80,1428.00] vol=1.8x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:15:00 | 1405.37 | 1414.42 | 0.00 | T1 1.5R @ 1405.37 |
| Stop hit — per-position SL triggered | 2024-12-20 11:55:00 | 1412.00 | 1413.19 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:15:00 | 1331.90 | 1337.55 | 0.00 | ORB-short ORB[1332.00,1344.10] vol=5.6x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:50:00 | 1326.48 | 1334.99 | 0.00 | T1 1.5R @ 1326.48 |
| Target hit | 2024-12-30 15:20:00 | 1308.60 | 1324.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2024-12-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:50:00 | 1292.15 | 1299.41 | 0.00 | ORB-short ORB[1295.00,1308.35] vol=1.9x ATR=5.69 |
| Stop hit — per-position SL triggered | 2024-12-31 10:55:00 | 1297.84 | 1297.62 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:45:00 | 1293.10 | 1298.65 | 0.00 | ORB-short ORB[1293.75,1312.45] vol=1.5x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:15:00 | 1287.08 | 1295.10 | 0.00 | T1 1.5R @ 1287.08 |
| Target hit | 2025-01-06 15:20:00 | 1251.85 | 1263.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2025-01-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:45:00 | 1108.00 | 1104.13 | 0.00 | ORB-long ORB[1092.90,1106.60] vol=1.6x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-01-21 09:55:00 | 1104.84 | 1104.15 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-01-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:20:00 | 1076.60 | 1083.21 | 0.00 | ORB-short ORB[1087.30,1099.20] vol=4.3x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-01-24 11:40:00 | 1080.61 | 1080.87 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 1045.95 | 1049.12 | 0.00 | ORB-short ORB[1046.00,1056.35] vol=2.2x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-01-27 09:40:00 | 1050.29 | 1048.70 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 1036.45 | 1041.24 | 0.00 | ORB-short ORB[1037.50,1048.00] vol=1.8x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:50:00 | 1030.28 | 1038.90 | 0.00 | T1 1.5R @ 1030.28 |
| Stop hit — per-position SL triggered | 2025-01-28 10:00:00 | 1036.45 | 1038.82 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-01-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:00:00 | 1072.05 | 1060.74 | 0.00 | ORB-long ORB[1045.90,1059.00] vol=1.6x ATR=4.03 |
| Stop hit — per-position SL triggered | 2025-01-29 10:45:00 | 1068.02 | 1065.00 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 1081.10 | 1073.37 | 0.00 | ORB-long ORB[1069.30,1077.00] vol=1.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:45:00 | 1085.84 | 1077.44 | 0.00 | T1 1.5R @ 1085.84 |
| Target hit | 2025-01-31 13:40:00 | 1086.20 | 1087.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1084.15 | 1091.89 | 0.00 | ORB-short ORB[1086.00,1097.90] vol=3.0x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:10:00 | 1079.19 | 1089.72 | 0.00 | T1 1.5R @ 1079.19 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 1084.15 | 1089.34 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:15:00 | 999.00 | 1004.64 | 0.00 | ORB-short ORB[1000.00,1011.70] vol=1.8x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:20:00 | 992.84 | 1001.86 | 0.00 | T1 1.5R @ 992.84 |
| Stop hit — per-position SL triggered | 2025-02-18 11:45:00 | 999.00 | 1001.21 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:05:00 | 1000.50 | 993.41 | 0.00 | ORB-long ORB[982.10,996.40] vol=1.9x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-02-20 10:45:00 | 996.56 | 995.30 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-02-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:10:00 | 1007.80 | 1017.96 | 0.00 | ORB-short ORB[1011.00,1022.15] vol=1.9x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-02-21 10:30:00 | 1011.99 | 1016.49 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:45:00 | 1007.55 | 1002.11 | 0.00 | ORB-long ORB[995.00,1003.95] vol=3.9x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:55:00 | 1014.22 | 1004.35 | 0.00 | T1 1.5R @ 1014.22 |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 1007.55 | 1008.99 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:05:00 | 922.00 | 932.92 | 0.00 | ORB-short ORB[933.40,944.15] vol=3.5x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:20:00 | 916.98 | 930.84 | 0.00 | T1 1.5R @ 916.98 |
| Stop hit — per-position SL triggered | 2025-03-12 13:20:00 | 922.00 | 925.80 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 971.20 | 964.69 | 0.00 | ORB-long ORB[958.90,965.60] vol=1.8x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-03-21 11:10:00 | 968.10 | 965.54 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-03-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:45:00 | 933.65 | 939.78 | 0.00 | ORB-short ORB[937.30,950.30] vol=2.0x ATR=3.37 |
| Stop hit — per-position SL triggered | 2025-03-26 10:20:00 | 937.02 | 937.20 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-04-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:25:00 | 986.55 | 975.90 | 0.00 | ORB-long ORB[965.40,978.95] vol=1.9x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:35:00 | 994.42 | 979.82 | 0.00 | T1 1.5R @ 994.42 |
| Stop hit — per-position SL triggered | 2025-04-22 10:50:00 | 986.55 | 981.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:35:00 | 1321.00 | 2024-05-14 12:15:00 | 1327.71 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-14 09:35:00 | 1321.00 | 2024-05-14 13:20:00 | 1321.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-16 10:10:00 | 1300.05 | 2024-05-16 10:20:00 | 1296.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-23 09:50:00 | 1335.70 | 2024-05-23 10:10:00 | 1338.81 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-27 09:45:00 | 1334.30 | 2024-05-27 10:00:00 | 1328.40 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-27 09:45:00 | 1334.30 | 2024-05-27 10:10:00 | 1334.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 09:35:00 | 1334.75 | 2024-06-06 11:30:00 | 1329.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-07 09:35:00 | 1339.80 | 2024-06-07 09:55:00 | 1345.55 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-07 09:35:00 | 1339.80 | 2024-06-07 10:20:00 | 1343.05 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-06-10 10:20:00 | 1349.75 | 2024-06-10 10:30:00 | 1354.84 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-10 10:20:00 | 1349.75 | 2024-06-10 12:10:00 | 1350.30 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-06-11 10:00:00 | 1364.00 | 2024-06-11 10:35:00 | 1359.91 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-12 09:50:00 | 1404.95 | 2024-06-12 10:00:00 | 1399.78 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-14 10:50:00 | 1383.35 | 2024-06-14 11:00:00 | 1386.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-18 11:15:00 | 1389.10 | 2024-06-18 11:25:00 | 1391.26 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-06-19 10:25:00 | 1386.55 | 2024-06-19 10:55:00 | 1390.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-20 11:00:00 | 1387.55 | 2024-06-20 11:20:00 | 1389.79 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1391.85 | 2024-06-21 09:35:00 | 1395.56 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1391.85 | 2024-06-21 13:05:00 | 1432.00 | TARGET_HIT | 0.50 | 2.88% |
| BUY | retest1 | 2024-06-25 09:40:00 | 1441.00 | 2024-06-25 09:50:00 | 1436.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-26 09:40:00 | 1444.70 | 2024-06-26 09:50:00 | 1450.84 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-26 09:40:00 | 1444.70 | 2024-06-26 09:55:00 | 1444.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 09:55:00 | 1469.80 | 2024-07-04 10:15:00 | 1475.26 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-07-04 09:55:00 | 1469.80 | 2024-07-04 10:25:00 | 1469.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 10:45:00 | 1479.80 | 2024-07-05 11:05:00 | 1475.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-08 10:50:00 | 1459.55 | 2024-07-08 11:10:00 | 1454.15 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-08 10:50:00 | 1459.55 | 2024-07-08 11:15:00 | 1459.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:10:00 | 1442.00 | 2024-07-10 10:20:00 | 1436.39 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 10:10:00 | 1442.00 | 2024-07-10 11:40:00 | 1442.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-16 09:40:00 | 1448.65 | 2024-07-16 09:45:00 | 1452.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1416.60 | 2024-07-18 09:45:00 | 1422.01 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-19 09:35:00 | 1426.35 | 2024-07-19 09:40:00 | 1421.04 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-26 09:40:00 | 1496.30 | 2024-07-26 10:25:00 | 1505.59 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-26 09:40:00 | 1496.30 | 2024-07-26 11:05:00 | 1496.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:40:00 | 1529.10 | 2024-07-31 09:45:00 | 1525.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-01 11:05:00 | 1490.85 | 2024-08-01 11:25:00 | 1495.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-07 10:25:00 | 1442.70 | 2024-08-07 10:45:00 | 1450.86 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-07 10:25:00 | 1442.70 | 2024-08-07 15:20:00 | 1465.30 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2024-08-14 10:50:00 | 1439.30 | 2024-08-14 12:30:00 | 1431.89 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-14 10:50:00 | 1439.30 | 2024-08-14 15:20:00 | 1432.95 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-22 10:25:00 | 1529.55 | 2024-08-22 10:45:00 | 1525.65 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-23 09:30:00 | 1496.55 | 2024-08-23 10:45:00 | 1501.03 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-26 10:05:00 | 1502.00 | 2024-08-26 10:15:00 | 1497.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-28 09:50:00 | 1542.25 | 2024-08-28 10:00:00 | 1548.06 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-28 09:50:00 | 1542.25 | 2024-08-28 10:45:00 | 1542.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:45:00 | 1522.20 | 2024-08-29 10:30:00 | 1517.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-30 10:00:00 | 1528.75 | 2024-08-30 10:10:00 | 1524.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-03 09:50:00 | 1523.10 | 2024-09-03 10:05:00 | 1520.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-04 09:40:00 | 1530.35 | 2024-09-04 09:55:00 | 1537.57 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-04 09:40:00 | 1530.35 | 2024-09-04 10:15:00 | 1530.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 09:30:00 | 1557.70 | 2024-09-05 09:35:00 | 1565.12 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-05 09:30:00 | 1557.70 | 2024-09-05 09:55:00 | 1557.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-12 10:55:00 | 1590.35 | 2024-09-12 12:20:00 | 1594.31 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-16 09:35:00 | 1682.50 | 2024-09-16 09:45:00 | 1676.66 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-18 09:35:00 | 1689.50 | 2024-09-18 09:40:00 | 1696.36 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-18 09:35:00 | 1689.50 | 2024-09-18 11:20:00 | 1698.55 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-23 11:10:00 | 1674.20 | 2024-09-23 11:25:00 | 1670.71 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-25 09:55:00 | 1694.50 | 2024-09-25 10:05:00 | 1698.34 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-27 11:10:00 | 1713.85 | 2024-09-27 11:35:00 | 1705.84 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-09-27 11:10:00 | 1713.85 | 2024-09-27 15:20:00 | 1677.70 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2024-10-03 09:35:00 | 1634.80 | 2024-10-03 09:55:00 | 1628.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-04 10:55:00 | 1631.70 | 2024-10-04 11:00:00 | 1625.66 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1576.05 | 2024-10-07 11:15:00 | 1583.56 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-10-08 09:30:00 | 1565.25 | 2024-10-08 09:40:00 | 1574.73 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-08 09:30:00 | 1565.25 | 2024-10-08 11:55:00 | 1575.35 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-10 09:30:00 | 1610.50 | 2024-10-10 09:50:00 | 1618.80 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-10 09:30:00 | 1610.50 | 2024-10-10 09:55:00 | 1610.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:50:00 | 1622.30 | 2024-10-11 09:55:00 | 1629.76 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-10-11 09:50:00 | 1622.30 | 2024-10-11 10:20:00 | 1623.85 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1614.50 | 2024-10-18 09:45:00 | 1624.43 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1614.50 | 2024-10-18 10:00:00 | 1614.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-24 11:15:00 | 1511.45 | 2024-10-24 11:30:00 | 1503.75 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-24 11:15:00 | 1511.45 | 2024-10-24 15:20:00 | 1509.90 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1497.60 | 2024-10-25 09:40:00 | 1490.33 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1497.60 | 2024-10-25 09:55:00 | 1497.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:30:00 | 1496.60 | 2024-10-29 10:35:00 | 1501.69 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-30 10:15:00 | 1542.55 | 2024-10-30 10:25:00 | 1537.78 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-06 09:30:00 | 1517.20 | 2024-11-06 09:40:00 | 1511.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-08 10:55:00 | 1476.25 | 2024-11-08 12:00:00 | 1468.90 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-08 10:55:00 | 1476.25 | 2024-11-08 15:20:00 | 1467.40 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-11 09:30:00 | 1474.35 | 2024-11-11 09:45:00 | 1482.10 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-11-11 09:30:00 | 1474.35 | 2024-11-11 12:05:00 | 1474.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-14 10:00:00 | 1469.55 | 2024-11-14 10:25:00 | 1463.68 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-22 10:45:00 | 1438.00 | 2024-11-22 10:55:00 | 1441.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-26 09:50:00 | 1470.90 | 2024-11-26 11:20:00 | 1464.15 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-26 09:50:00 | 1470.90 | 2024-11-26 11:45:00 | 1470.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 09:30:00 | 1466.00 | 2024-11-27 09:45:00 | 1470.17 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-28 09:45:00 | 1538.05 | 2024-11-28 10:25:00 | 1531.99 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-11-29 09:30:00 | 1546.75 | 2024-11-29 09:35:00 | 1540.21 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-03 09:40:00 | 1591.40 | 2024-12-03 13:15:00 | 1600.57 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-03 09:40:00 | 1591.40 | 2024-12-03 15:20:00 | 1598.25 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-04 11:00:00 | 1589.75 | 2024-12-04 11:10:00 | 1593.73 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-11 10:50:00 | 1481.15 | 2024-12-11 11:35:00 | 1483.77 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-12 09:35:00 | 1468.95 | 2024-12-12 12:10:00 | 1460.35 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-12-12 09:35:00 | 1468.95 | 2024-12-12 15:20:00 | 1457.50 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-12-13 10:35:00 | 1441.65 | 2024-12-13 10:50:00 | 1446.12 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1475.70 | 2024-12-16 09:45:00 | 1482.62 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1475.70 | 2024-12-16 09:55:00 | 1475.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:15:00 | 1412.00 | 2024-12-20 11:15:00 | 1405.37 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-20 10:15:00 | 1412.00 | 2024-12-20 11:55:00 | 1412.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 11:15:00 | 1331.90 | 2024-12-30 11:50:00 | 1326.48 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-30 11:15:00 | 1331.90 | 2024-12-30 15:20:00 | 1308.60 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2024-12-31 09:50:00 | 1292.15 | 2024-12-31 10:55:00 | 1297.84 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-06 09:45:00 | 1293.10 | 2025-01-06 10:15:00 | 1287.08 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-06 09:45:00 | 1293.10 | 2025-01-06 15:20:00 | 1251.85 | TARGET_HIT | 0.50 | 3.19% |
| BUY | retest1 | 2025-01-21 09:45:00 | 1108.00 | 2025-01-21 09:55:00 | 1104.84 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-24 10:20:00 | 1076.60 | 2025-01-24 11:40:00 | 1080.61 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-27 09:30:00 | 1045.95 | 2025-01-27 09:40:00 | 1050.29 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-28 09:30:00 | 1036.45 | 2025-01-28 09:50:00 | 1030.28 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-01-28 09:30:00 | 1036.45 | 2025-01-28 10:00:00 | 1036.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 10:00:00 | 1072.05 | 2025-01-29 10:45:00 | 1068.02 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-31 11:00:00 | 1081.10 | 2025-01-31 11:45:00 | 1085.84 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-31 11:00:00 | 1081.10 | 2025-01-31 13:40:00 | 1086.20 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1084.15 | 2025-02-01 11:10:00 | 1079.19 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1084.15 | 2025-02-01 11:15:00 | 1084.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 10:15:00 | 999.00 | 2025-02-18 11:20:00 | 992.84 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-02-18 10:15:00 | 999.00 | 2025-02-18 11:45:00 | 999.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 10:05:00 | 1000.50 | 2025-02-20 10:45:00 | 996.56 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-21 10:10:00 | 1007.80 | 2025-02-21 10:30:00 | 1011.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-25 09:45:00 | 1007.55 | 2025-02-25 09:55:00 | 1014.22 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-02-25 09:45:00 | 1007.55 | 2025-02-25 10:15:00 | 1007.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 11:05:00 | 922.00 | 2025-03-12 11:20:00 | 916.98 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-12 11:05:00 | 922.00 | 2025-03-12 13:20:00 | 922.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:50:00 | 971.20 | 2025-03-21 11:10:00 | 968.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-26 09:45:00 | 933.65 | 2025-03-26 10:20:00 | 937.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-22 10:25:00 | 986.55 | 2025-04-22 10:35:00 | 994.42 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-04-22 10:25:00 | 986.55 | 2025-04-22 10:50:00 | 986.55 | STOP_HIT | 0.50 | 0.00% |
