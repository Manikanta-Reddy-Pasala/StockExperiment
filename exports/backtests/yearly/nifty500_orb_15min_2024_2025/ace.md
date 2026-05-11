# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 949.90
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
| ENTRY1 | 33 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 29
- **Target hits / Stop hits / Partials:** 4 / 29 / 9
- **Avg / median % per leg:** 0.08% / -0.29%
- **Sum % (uncompounded):** 3.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 6 | 25.0% | 2 | 18 | 4 | 0.03% | 0.7% |
| BUY @ 2nd Alert (retest1) | 24 | 6 | 25.0% | 2 | 18 | 4 | 0.03% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.15% | 2.8% |
| SELL @ 2nd Alert (retest1) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.15% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 42 | 13 | 31.0% | 4 | 29 | 9 | 0.08% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 1415.95 | 1405.69 | 0.00 | ORB-long ORB[1390.90,1409.90] vol=3.3x ATR=8.89 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 1407.06 | 1405.98 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:15:00 | 1407.50 | 1416.28 | 0.00 | ORB-short ORB[1414.10,1435.00] vol=2.1x ATR=6.93 |
| Stop hit — per-position SL triggered | 2024-05-31 10:40:00 | 1414.43 | 1414.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:25:00 | 1510.00 | 1497.06 | 0.00 | ORB-long ORB[1484.25,1504.90] vol=1.8x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:55:00 | 1520.60 | 1504.54 | 0.00 | T1 1.5R @ 1520.60 |
| Stop hit — per-position SL triggered | 2024-06-26 11:35:00 | 1510.00 | 1505.90 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:35:00 | 1446.95 | 1442.57 | 0.00 | ORB-long ORB[1429.55,1442.10] vol=10.9x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-07-11 10:10:00 | 1440.98 | 1443.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1396.95 | 1405.15 | 0.00 | ORB-short ORB[1400.20,1417.80] vol=2.2x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:35:00 | 1387.09 | 1403.84 | 0.00 | T1 1.5R @ 1387.09 |
| Stop hit — per-position SL triggered | 2024-07-23 11:45:00 | 1396.95 | 1403.48 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:15:00 | 1402.70 | 1395.78 | 0.00 | ORB-long ORB[1385.75,1396.95] vol=1.9x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-07-26 10:20:00 | 1398.42 | 1395.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:00:00 | 1414.80 | 1398.10 | 0.00 | ORB-long ORB[1383.05,1396.40] vol=3.5x ATR=5.61 |
| Stop hit — per-position SL triggered | 2024-07-30 10:05:00 | 1409.19 | 1399.53 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:10:00 | 1453.10 | 1441.10 | 0.00 | ORB-long ORB[1432.00,1448.75] vol=3.8x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:15:00 | 1459.75 | 1442.51 | 0.00 | T1 1.5R @ 1459.75 |
| Stop hit — per-position SL triggered | 2024-07-31 11:20:00 | 1453.10 | 1442.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 1299.70 | 1303.91 | 0.00 | ORB-short ORB[1299.75,1316.80] vol=1.7x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:55:00 | 1293.13 | 1301.94 | 0.00 | T1 1.5R @ 1293.13 |
| Stop hit — per-position SL triggered | 2024-08-26 10:10:00 | 1299.70 | 1300.73 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1250.20 | 1257.33 | 0.00 | ORB-short ORB[1253.30,1268.00] vol=1.7x ATR=4.56 |
| Target hit | 2024-09-12 15:20:00 | 1248.80 | 1250.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-09-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:45:00 | 1262.35 | 1254.28 | 0.00 | ORB-long ORB[1249.10,1258.00] vol=2.0x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:50:00 | 1267.60 | 1256.92 | 0.00 | T1 1.5R @ 1267.60 |
| Target hit | 2024-09-16 15:20:00 | 1325.00 | 1297.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-09-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:00:00 | 1363.95 | 1342.80 | 0.00 | ORB-long ORB[1326.05,1345.00] vol=3.6x ATR=8.84 |
| Stop hit — per-position SL triggered | 2024-09-17 10:05:00 | 1355.11 | 1344.74 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 1450.80 | 1439.94 | 0.00 | ORB-long ORB[1430.60,1448.95] vol=4.1x ATR=4.81 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 1445.99 | 1440.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:50:00 | 1424.45 | 1435.04 | 0.00 | ORB-short ORB[1430.00,1445.95] vol=2.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-09-25 11:00:00 | 1428.60 | 1434.34 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1409.00 | 1417.73 | 0.00 | ORB-short ORB[1412.20,1427.80] vol=1.8x ATR=5.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:35:00 | 1401.24 | 1412.74 | 0.00 | T1 1.5R @ 1401.24 |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 1409.00 | 1406.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:40:00 | 1431.45 | 1419.79 | 0.00 | ORB-long ORB[1410.00,1425.70] vol=2.4x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-09-27 10:45:00 | 1427.83 | 1420.44 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-10-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:40:00 | 1408.50 | 1400.14 | 0.00 | ORB-long ORB[1393.00,1403.60] vol=1.8x ATR=5.90 |
| Stop hit — per-position SL triggered | 2024-10-01 09:50:00 | 1402.60 | 1400.49 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-10-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:55:00 | 1348.00 | 1344.44 | 0.00 | ORB-long ORB[1332.10,1347.70] vol=2.8x ATR=5.03 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 1342.97 | 1344.40 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 1257.30 | 1266.41 | 0.00 | ORB-short ORB[1261.60,1274.95] vol=1.9x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:45:00 | 1249.37 | 1263.30 | 0.00 | T1 1.5R @ 1249.37 |
| Target hit | 2024-10-25 15:20:00 | 1224.50 | 1237.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 1337.50 | 1324.74 | 0.00 | ORB-long ORB[1314.85,1333.55] vol=2.2x ATR=5.89 |
| Stop hit — per-position SL triggered | 2024-11-07 09:45:00 | 1331.61 | 1327.08 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1229.25 | 1240.11 | 0.00 | ORB-short ORB[1235.10,1251.95] vol=2.6x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-11-18 09:50:00 | 1235.50 | 1235.06 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:30:00 | 1270.85 | 1266.59 | 0.00 | ORB-long ORB[1253.45,1270.00] vol=2.6x ATR=5.32 |
| Stop hit — per-position SL triggered | 2024-11-26 09:45:00 | 1265.53 | 1266.74 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-11-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:05:00 | 1275.00 | 1266.14 | 0.00 | ORB-long ORB[1253.20,1269.55] vol=2.8x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-11-27 11:30:00 | 1270.94 | 1266.89 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:20:00 | 1307.90 | 1294.56 | 0.00 | ORB-long ORB[1282.55,1297.70] vol=1.5x ATR=5.14 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 1302.76 | 1298.98 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:15:00 | 1378.55 | 1365.76 | 0.00 | ORB-long ORB[1347.05,1366.90] vol=3.9x ATR=5.66 |
| Stop hit — per-position SL triggered | 2024-12-04 10:25:00 | 1372.89 | 1367.21 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:05:00 | 1442.00 | 1424.08 | 0.00 | ORB-long ORB[1407.25,1426.65] vol=4.5x ATR=8.06 |
| Stop hit — per-position SL triggered | 2024-12-09 11:05:00 | 1433.94 | 1431.81 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 1476.70 | 1487.94 | 0.00 | ORB-short ORB[1484.00,1499.00] vol=1.8x ATR=6.30 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 1483.00 | 1486.10 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:50:00 | 1445.00 | 1464.21 | 0.00 | ORB-short ORB[1461.45,1479.55] vol=1.7x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 1450.99 | 1462.99 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:40:00 | 1538.95 | 1545.26 | 0.00 | ORB-short ORB[1540.45,1558.70] vol=1.6x ATR=6.62 |
| Stop hit — per-position SL triggered | 2025-01-03 10:00:00 | 1545.57 | 1544.73 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 1305.30 | 1298.90 | 0.00 | ORB-long ORB[1291.00,1304.40] vol=2.9x ATR=6.11 |
| Stop hit — per-position SL triggered | 2025-01-20 09:55:00 | 1299.19 | 1300.52 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 1275.60 | 1287.82 | 0.00 | ORB-short ORB[1283.00,1301.85] vol=1.8x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:00:00 | 1267.52 | 1284.03 | 0.00 | T1 1.5R @ 1267.52 |
| Stop hit — per-position SL triggered | 2025-01-24 10:35:00 | 1275.60 | 1280.14 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 1255.50 | 1247.65 | 0.00 | ORB-long ORB[1237.80,1253.30] vol=2.6x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:35:00 | 1263.44 | 1252.74 | 0.00 | T1 1.5R @ 1263.44 |
| Target hit | 2025-04-21 10:15:00 | 1258.90 | 1260.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 1257.90 | 1265.77 | 0.00 | ORB-short ORB[1259.40,1273.30] vol=2.7x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 1262.86 | 1265.18 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:35:00 | 1415.95 | 2024-05-16 09:40:00 | 1407.06 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2024-05-31 10:15:00 | 1407.50 | 2024-05-31 10:40:00 | 1414.43 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-26 10:25:00 | 1510.00 | 2024-06-26 10:55:00 | 1520.60 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-06-26 10:25:00 | 1510.00 | 2024-06-26 11:35:00 | 1510.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:35:00 | 1446.95 | 2024-07-11 10:10:00 | 1440.98 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1396.95 | 2024-07-23 11:35:00 | 1387.09 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1396.95 | 2024-07-23 11:45:00 | 1396.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:15:00 | 1402.70 | 2024-07-26 10:20:00 | 1398.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-30 10:00:00 | 1414.80 | 2024-07-30 10:05:00 | 1409.19 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-31 11:10:00 | 1453.10 | 2024-07-31 11:15:00 | 1459.75 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-31 11:10:00 | 1453.10 | 2024-07-31 11:20:00 | 1453.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 09:35:00 | 1299.70 | 2024-08-26 09:55:00 | 1293.13 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-08-26 09:35:00 | 1299.70 | 2024-08-26 10:10:00 | 1299.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-12 09:30:00 | 1250.20 | 2024-09-12 15:20:00 | 1248.80 | TARGET_HIT | 1.00 | 0.11% |
| BUY | retest1 | 2024-09-16 10:45:00 | 1262.35 | 2024-09-16 10:50:00 | 1267.60 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-16 10:45:00 | 1262.35 | 2024-09-16 15:20:00 | 1325.00 | TARGET_HIT | 0.50 | 4.96% |
| BUY | retest1 | 2024-09-17 10:00:00 | 1363.95 | 2024-09-17 10:05:00 | 1355.11 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2024-09-24 11:00:00 | 1450.80 | 2024-09-24 11:05:00 | 1445.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-25 10:50:00 | 1424.45 | 2024-09-25 11:00:00 | 1428.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-26 09:30:00 | 1409.00 | 2024-09-26 09:35:00 | 1401.24 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-26 09:30:00 | 1409.00 | 2024-09-26 10:15:00 | 1409.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:40:00 | 1431.45 | 2024-09-27 10:45:00 | 1427.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-01 09:40:00 | 1408.50 | 2024-10-01 09:50:00 | 1402.60 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-15 09:55:00 | 1348.00 | 2024-10-15 10:00:00 | 1342.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1257.30 | 2024-10-25 09:45:00 | 1249.37 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1257.30 | 2024-10-25 15:20:00 | 1224.50 | TARGET_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2024-11-07 09:35:00 | 1337.50 | 2024-11-07 09:45:00 | 1331.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-18 09:30:00 | 1229.25 | 2024-11-18 09:50:00 | 1235.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-11-26 09:30:00 | 1270.85 | 2024-11-26 09:45:00 | 1265.53 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-11-27 11:05:00 | 1275.00 | 2024-11-27 11:30:00 | 1270.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-28 10:20:00 | 1307.90 | 2024-11-28 10:30:00 | 1302.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-04 10:15:00 | 1378.55 | 2024-12-04 10:25:00 | 1372.89 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-09 10:05:00 | 1442.00 | 2024-12-09 11:05:00 | 1433.94 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-12-24 09:30:00 | 1476.70 | 2024-12-24 09:40:00 | 1483.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-26 09:50:00 | 1445.00 | 2024-12-26 09:55:00 | 1450.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-03 09:40:00 | 1538.95 | 2025-01-03 10:00:00 | 1545.57 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-20 09:40:00 | 1305.30 | 2025-01-20 09:55:00 | 1299.19 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-01-24 09:35:00 | 1275.60 | 2025-01-24 10:00:00 | 1267.52 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-01-24 09:35:00 | 1275.60 | 2025-01-24 10:35:00 | 1275.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:30:00 | 1255.50 | 2025-04-21 09:35:00 | 1263.44 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-04-21 09:30:00 | 1255.50 | 2025-04-21 10:15:00 | 1258.90 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-04-23 09:35:00 | 1257.90 | 2025-04-23 09:45:00 | 1262.86 | STOP_HIT | 1.00 | -0.39% |
