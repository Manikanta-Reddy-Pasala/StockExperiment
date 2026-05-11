# INFY (INFY)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-06-04 15:25:00 (18431 bars)
- **Last close:** 1400.05
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 12 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 66
- **Target hits / Stop hits / Partials:** 12 / 66 / 33
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 5.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 12 | 24.0% | 1 | 38 | 11 | -0.03% | -1.4% |
| BUY @ 2nd Alert (retest1) | 50 | 12 | 24.0% | 1 | 38 | 11 | -0.03% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 33 | 54.1% | 11 | 28 | 22 | 0.12% | 7.1% |
| SELL @ 2nd Alert (retest1) | 61 | 33 | 54.1% | 11 | 28 | 22 | 0.12% | 7.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 45 | 40.5% | 12 | 66 | 33 | 0.05% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:00:00 | 1248.00 | 1253.84 | 0.00 | ORB-short ORB[1252.05,1260.95] vol=1.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-05-17 11:05:00 | 1249.83 | 1253.61 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:45:00 | 1250.65 | 1254.05 | 0.00 | ORB-short ORB[1252.95,1259.65] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2023-05-18 10:25:00 | 1253.79 | 1253.31 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 09:40:00 | 1264.70 | 1259.55 | 0.00 | ORB-long ORB[1252.80,1260.10] vol=1.9x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:10:00 | 1268.78 | 1263.68 | 0.00 | T1 1.5R @ 1268.78 |
| Stop hit — per-position SL triggered | 2023-05-19 10:25:00 | 1264.70 | 1264.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:35:00 | 1310.85 | 1306.22 | 0.00 | ORB-long ORB[1297.05,1310.00] vol=1.7x ATR=3.03 |
| Stop hit — per-position SL triggered | 2023-05-23 09:45:00 | 1307.82 | 1307.07 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:15:00 | 1303.20 | 1298.73 | 0.00 | ORB-long ORB[1291.30,1297.00] vol=2.3x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-05-24 11:35:00 | 1300.86 | 1299.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:30:00 | 1318.85 | 1314.33 | 0.00 | ORB-long ORB[1306.00,1314.80] vol=1.9x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-05-26 10:50:00 | 1316.47 | 1314.88 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 09:30:00 | 1319.95 | 1325.06 | 0.00 | ORB-short ORB[1322.90,1332.70] vol=1.5x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 10:15:00 | 1315.40 | 1321.95 | 0.00 | T1 1.5R @ 1315.40 |
| Target hit | 2023-05-29 15:20:00 | 1314.45 | 1316.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2023-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:55:00 | 1311.00 | 1316.28 | 0.00 | ORB-short ORB[1311.35,1320.90] vol=2.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2023-05-30 11:00:00 | 1312.87 | 1316.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:30:00 | 1328.55 | 1324.48 | 0.00 | ORB-long ORB[1318.00,1326.60] vol=1.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2023-05-31 09:35:00 | 1326.07 | 1324.96 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 09:35:00 | 1304.30 | 1307.47 | 0.00 | ORB-short ORB[1305.10,1312.90] vol=1.9x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 10:30:00 | 1300.54 | 1304.98 | 0.00 | T1 1.5R @ 1300.54 |
| Stop hit — per-position SL triggered | 2023-06-02 11:05:00 | 1304.30 | 1304.61 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:10:00 | 1269.30 | 1274.83 | 0.00 | ORB-short ORB[1275.50,1282.50] vol=1.6x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 10:45:00 | 1266.15 | 1272.82 | 0.00 | T1 1.5R @ 1266.15 |
| Target hit | 2023-06-09 15:20:00 | 1265.05 | 1268.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2023-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 11:05:00 | 1288.70 | 1283.88 | 0.00 | ORB-long ORB[1273.00,1284.90] vol=2.2x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-06-12 11:35:00 | 1286.87 | 1284.81 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 11:00:00 | 1284.40 | 1287.49 | 0.00 | ORB-short ORB[1285.50,1291.50] vol=2.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-06-16 11:45:00 | 1286.15 | 1286.34 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:10:00 | 1290.35 | 1286.08 | 0.00 | ORB-long ORB[1284.00,1290.00] vol=2.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 11:30:00 | 1292.66 | 1287.46 | 0.00 | T1 1.5R @ 1292.66 |
| Stop hit — per-position SL triggered | 2023-06-28 14:55:00 | 1290.35 | 1290.80 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 1328.10 | 1316.08 | 0.00 | ORB-long ORB[1304.45,1321.90] vol=2.4x ATR=3.85 |
| Stop hit — per-position SL triggered | 2023-06-30 09:40:00 | 1324.25 | 1317.32 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:50:00 | 1342.65 | 1339.89 | 0.00 | ORB-long ORB[1334.00,1341.90] vol=2.3x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-07-07 10:15:00 | 1340.18 | 1341.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 1340.45 | 1336.48 | 0.00 | ORB-long ORB[1327.00,1336.35] vol=1.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:00:00 | 1343.98 | 1338.19 | 0.00 | T1 1.5R @ 1343.98 |
| Stop hit — per-position SL triggered | 2023-07-11 10:30:00 | 1340.45 | 1339.87 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:45:00 | 1370.50 | 1358.51 | 0.00 | ORB-long ORB[1330.85,1348.20] vol=1.8x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 12:05:00 | 1375.92 | 1363.48 | 0.00 | T1 1.5R @ 1375.92 |
| Stop hit — per-position SL triggered | 2023-07-13 12:30:00 | 1370.50 | 1364.36 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 10:10:00 | 1454.30 | 1436.79 | 0.00 | ORB-long ORB[1414.30,1436.00] vol=1.7x ATR=4.74 |
| Stop hit — per-position SL triggered | 2023-07-17 10:20:00 | 1449.56 | 1439.59 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:30:00 | 1479.95 | 1488.38 | 0.00 | ORB-short ORB[1481.25,1498.80] vol=1.8x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:35:00 | 1473.05 | 1487.67 | 0.00 | T1 1.5R @ 1473.05 |
| Target hit | 2023-07-19 15:20:00 | 1476.00 | 1476.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-07-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:50:00 | 1345.00 | 1352.42 | 0.00 | ORB-short ORB[1353.85,1363.85] vol=1.6x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 11:50:00 | 1341.35 | 1350.26 | 0.00 | T1 1.5R @ 1341.35 |
| Target hit | 2023-07-28 15:20:00 | 1340.60 | 1345.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 11:10:00 | 1350.00 | 1344.39 | 0.00 | ORB-long ORB[1333.25,1348.95] vol=1.7x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 11:45:00 | 1353.55 | 1345.89 | 0.00 | T1 1.5R @ 1353.55 |
| Stop hit — per-position SL triggered | 2023-07-31 12:45:00 | 1350.00 | 1346.97 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:55:00 | 1381.70 | 1375.97 | 0.00 | ORB-long ORB[1362.00,1376.20] vol=2.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2023-08-04 10:35:00 | 1378.41 | 1378.19 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:35:00 | 1387.45 | 1391.32 | 0.00 | ORB-short ORB[1390.85,1396.85] vol=2.7x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 09:40:00 | 1383.61 | 1390.63 | 0.00 | T1 1.5R @ 1383.61 |
| Target hit | 2023-08-11 13:55:00 | 1378.85 | 1377.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2023-08-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:55:00 | 1389.95 | 1396.60 | 0.00 | ORB-short ORB[1397.00,1408.70] vol=1.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-08-18 11:35:00 | 1392.23 | 1395.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:50:00 | 1437.25 | 1432.70 | 0.00 | ORB-long ORB[1428.20,1434.75] vol=1.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2023-09-01 11:05:00 | 1434.75 | 1433.62 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 11:05:00 | 1474.50 | 1471.10 | 0.00 | ORB-long ORB[1464.50,1472.95] vol=2.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 12:30:00 | 1477.23 | 1472.48 | 0.00 | T1 1.5R @ 1477.23 |
| Stop hit — per-position SL triggered | 2023-09-11 13:25:00 | 1474.50 | 1473.14 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 1510.60 | 1506.76 | 0.00 | ORB-long ORB[1499.15,1508.00] vol=2.4x ATR=2.78 |
| Stop hit — per-position SL triggered | 2023-09-14 09:50:00 | 1507.82 | 1507.89 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:55:00 | 1473.90 | 1482.27 | 0.00 | ORB-short ORB[1484.85,1489.85] vol=1.6x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-09-25 12:00:00 | 1476.47 | 1479.90 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:55:00 | 1451.90 | 1453.78 | 0.00 | ORB-short ORB[1453.55,1459.50] vol=1.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-09-27 11:10:00 | 1453.59 | 1453.70 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 11:15:00 | 1425.20 | 1427.95 | 0.00 | ORB-short ORB[1427.15,1437.00] vol=1.7x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-10-03 11:30:00 | 1427.65 | 1427.84 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 11:05:00 | 1431.05 | 1423.33 | 0.00 | ORB-long ORB[1419.00,1427.70] vol=2.5x ATR=2.72 |
| Stop hit — per-position SL triggered | 2023-10-04 11:25:00 | 1428.33 | 1424.71 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:55:00 | 1470.50 | 1461.86 | 0.00 | ORB-long ORB[1450.20,1462.80] vol=1.5x ATR=4.20 |
| Stop hit — per-position SL triggered | 2023-10-05 11:45:00 | 1466.30 | 1464.77 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 1516.50 | 1509.06 | 0.00 | ORB-long ORB[1497.00,1508.60] vol=3.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-10-11 09:35:00 | 1513.28 | 1509.91 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-10-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:55:00 | 1481.85 | 1485.97 | 0.00 | ORB-short ORB[1484.20,1504.40] vol=6.0x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-10-12 11:55:00 | 1485.99 | 1485.48 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:25:00 | 1382.00 | 1385.89 | 0.00 | ORB-short ORB[1387.80,1393.85] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2023-11-09 11:25:00 | 1384.16 | 1384.35 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 10:55:00 | 1365.40 | 1368.75 | 0.00 | ORB-short ORB[1368.10,1373.15] vol=2.0x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 11:00:00 | 1362.43 | 1367.87 | 0.00 | T1 1.5R @ 1362.43 |
| Stop hit — per-position SL triggered | 2023-11-10 12:00:00 | 1365.40 | 1365.88 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-11-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 11:05:00 | 1399.90 | 1398.62 | 0.00 | ORB-long ORB[1393.00,1399.60] vol=3.7x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-11-15 12:00:00 | 1397.48 | 1398.64 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 1423.95 | 1416.52 | 0.00 | ORB-long ORB[1410.70,1419.00] vol=2.7x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:15:00 | 1427.08 | 1418.34 | 0.00 | T1 1.5R @ 1427.08 |
| Target hit | 2023-11-16 15:20:00 | 1442.55 | 1438.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-11-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:50:00 | 1448.00 | 1441.03 | 0.00 | ORB-long ORB[1436.85,1442.90] vol=2.6x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-11-17 11:15:00 | 1445.35 | 1442.80 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 11:10:00 | 1439.65 | 1443.98 | 0.00 | ORB-short ORB[1442.00,1450.00] vol=1.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-11-21 11:20:00 | 1441.39 | 1443.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 11:15:00 | 1441.45 | 1444.36 | 0.00 | ORB-short ORB[1445.55,1452.60] vol=1.8x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 13:00:00 | 1438.51 | 1443.10 | 0.00 | T1 1.5R @ 1438.51 |
| Target hit | 2023-11-24 15:20:00 | 1439.65 | 1440.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:40:00 | 1470.45 | 1465.41 | 0.00 | ORB-long ORB[1455.90,1468.95] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2023-12-04 10:00:00 | 1466.96 | 1465.89 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:45:00 | 1470.45 | 1459.59 | 0.00 | ORB-long ORB[1451.05,1458.15] vol=1.6x ATR=2.93 |
| Stop hit — per-position SL triggered | 2023-12-06 09:50:00 | 1467.52 | 1460.38 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 1476.30 | 1473.60 | 0.00 | ORB-long ORB[1466.25,1473.30] vol=1.8x ATR=2.61 |
| Stop hit — per-position SL triggered | 2023-12-08 11:10:00 | 1473.69 | 1473.64 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-12-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:55:00 | 1486.20 | 1479.89 | 0.00 | ORB-long ORB[1467.75,1482.90] vol=2.6x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-12-12 10:15:00 | 1482.60 | 1481.21 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:20:00 | 1563.20 | 1537.43 | 0.00 | ORB-long ORB[1521.00,1532.40] vol=1.5x ATR=5.17 |
| Stop hit — per-position SL triggered | 2023-12-15 10:25:00 | 1558.03 | 1539.79 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 1551.95 | 1562.15 | 0.00 | ORB-short ORB[1565.30,1573.95] vol=2.4x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:55:00 | 1545.70 | 1559.56 | 0.00 | T1 1.5R @ 1545.70 |
| Stop hit — per-position SL triggered | 2023-12-19 11:50:00 | 1551.95 | 1554.22 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 11:10:00 | 1560.40 | 1559.64 | 0.00 | ORB-long ORB[1544.00,1559.80] vol=1.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-12-27 12:30:00 | 1557.61 | 1559.96 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 11:05:00 | 1545.25 | 1541.17 | 0.00 | ORB-long ORB[1535.25,1542.90] vol=1.7x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 11:15:00 | 1548.67 | 1542.04 | 0.00 | T1 1.5R @ 1548.67 |
| Stop hit — per-position SL triggered | 2024-01-01 11:40:00 | 1545.25 | 1543.26 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:30:00 | 1531.55 | 1538.69 | 0.00 | ORB-short ORB[1538.50,1549.15] vol=2.5x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 11:15:00 | 1527.18 | 1536.62 | 0.00 | T1 1.5R @ 1527.18 |
| Stop hit — per-position SL triggered | 2024-01-02 14:00:00 | 1531.55 | 1532.78 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:35:00 | 1534.95 | 1523.45 | 0.00 | ORB-long ORB[1506.25,1527.00] vol=1.5x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-01-05 09:50:00 | 1530.91 | 1528.06 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-01-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 10:10:00 | 1641.85 | 1638.51 | 0.00 | ORB-long ORB[1620.25,1640.00] vol=9.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-01-17 10:30:00 | 1637.79 | 1638.59 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:15:00 | 1649.90 | 1657.28 | 0.00 | ORB-short ORB[1650.00,1665.50] vol=1.9x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 11:30:00 | 1644.83 | 1656.51 | 0.00 | T1 1.5R @ 1644.83 |
| Stop hit — per-position SL triggered | 2024-01-19 11:35:00 | 1649.90 | 1656.21 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 09:55:00 | 1665.50 | 1659.43 | 0.00 | ORB-long ORB[1651.50,1662.00] vol=1.5x ATR=5.65 |
| Stop hit — per-position SL triggered | 2024-01-24 10:05:00 | 1659.85 | 1659.68 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 10:40:00 | 1656.50 | 1663.51 | 0.00 | ORB-short ORB[1660.10,1679.35] vol=1.5x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 11:20:00 | 1649.04 | 1660.74 | 0.00 | T1 1.5R @ 1649.04 |
| Stop hit — per-position SL triggered | 2024-01-29 12:10:00 | 1656.50 | 1659.40 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 11:00:00 | 1673.40 | 1666.88 | 0.00 | ORB-long ORB[1650.35,1667.45] vol=1.9x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-02-01 11:05:00 | 1669.71 | 1666.97 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-02-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 09:35:00 | 1709.05 | 1692.04 | 0.00 | ORB-long ORB[1673.95,1694.80] vol=1.8x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 09:45:00 | 1717.09 | 1699.64 | 0.00 | T1 1.5R @ 1717.09 |
| Stop hit — per-position SL triggered | 2024-02-06 10:25:00 | 1709.05 | 1704.66 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 1701.50 | 1710.02 | 0.00 | ORB-short ORB[1710.65,1729.00] vol=1.9x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 11:55:00 | 1697.17 | 1706.87 | 0.00 | T1 1.5R @ 1697.17 |
| Stop hit — per-position SL triggered | 2024-02-07 12:20:00 | 1701.50 | 1705.88 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-02-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:40:00 | 1689.55 | 1698.39 | 0.00 | ORB-short ORB[1698.00,1706.35] vol=1.6x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:30:00 | 1683.54 | 1693.89 | 0.00 | T1 1.5R @ 1683.54 |
| Stop hit — per-position SL triggered | 2024-02-08 11:40:00 | 1689.55 | 1693.67 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 11:00:00 | 1662.95 | 1675.81 | 0.00 | ORB-short ORB[1675.95,1692.90] vol=1.7x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-02-09 11:10:00 | 1667.07 | 1674.54 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:30:00 | 1667.90 | 1675.52 | 0.00 | ORB-short ORB[1669.05,1689.10] vol=2.2x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-02-13 09:55:00 | 1672.32 | 1672.77 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:30:00 | 1641.70 | 1649.07 | 0.00 | ORB-short ORB[1642.35,1665.00] vol=1.6x ATR=5.74 |
| Stop hit — per-position SL triggered | 2024-02-14 09:35:00 | 1647.44 | 1648.72 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:50:00 | 1678.05 | 1683.18 | 0.00 | ORB-short ORB[1685.00,1694.25] vol=4.9x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 11:10:00 | 1672.90 | 1680.51 | 0.00 | T1 1.5R @ 1672.90 |
| Stop hit — per-position SL triggered | 2024-02-20 11:45:00 | 1678.05 | 1678.78 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 11:00:00 | 1655.55 | 1660.10 | 0.00 | ORB-short ORB[1661.10,1674.95] vol=1.7x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 11:40:00 | 1651.47 | 1658.25 | 0.00 | T1 1.5R @ 1651.47 |
| Target hit | 2024-02-26 13:35:00 | 1655.50 | 1653.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — SELL (started 2024-02-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:25:00 | 1656.25 | 1659.43 | 0.00 | ORB-short ORB[1661.00,1671.10] vol=3.2x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-02-27 10:50:00 | 1659.34 | 1658.94 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 10:40:00 | 1686.45 | 1675.41 | 0.00 | ORB-long ORB[1662.40,1670.50] vol=1.7x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-02-28 10:50:00 | 1682.60 | 1676.60 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 10:50:00 | 1648.15 | 1652.21 | 0.00 | ORB-short ORB[1649.00,1664.45] vol=2.1x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 10:55:00 | 1643.84 | 1651.71 | 0.00 | T1 1.5R @ 1643.84 |
| Target hit | 2024-03-04 15:20:00 | 1636.95 | 1641.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2024-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:00:00 | 1609.80 | 1622.41 | 0.00 | ORB-short ORB[1627.10,1637.35] vol=1.5x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-03-05 10:05:00 | 1613.95 | 1621.58 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:40:00 | 1618.50 | 1612.00 | 0.00 | ORB-long ORB[1605.20,1616.60] vol=1.6x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:50:00 | 1623.67 | 1613.43 | 0.00 | T1 1.5R @ 1623.67 |
| Stop hit — per-position SL triggered | 2024-03-07 12:40:00 | 1618.50 | 1618.17 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:00:00 | 1594.00 | 1601.72 | 0.00 | ORB-short ORB[1602.00,1613.80] vol=2.0x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:10:00 | 1588.52 | 1600.35 | 0.00 | T1 1.5R @ 1588.52 |
| Stop hit — per-position SL triggered | 2024-03-11 11:25:00 | 1594.00 | 1599.40 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-03-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 11:05:00 | 1558.30 | 1562.32 | 0.00 | ORB-short ORB[1561.50,1576.95] vol=4.7x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 11:35:00 | 1553.49 | 1561.62 | 0.00 | T1 1.5R @ 1553.49 |
| Target hit | 2024-03-21 15:20:00 | 1555.25 | 1556.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 11:00:00 | 1502.50 | 1514.04 | 0.00 | ORB-short ORB[1513.60,1529.95] vol=1.9x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 14:40:00 | 1497.51 | 1508.16 | 0.00 | T1 1.5R @ 1497.51 |
| Target hit | 2024-04-01 15:20:00 | 1495.20 | 1505.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2024-04-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 10:35:00 | 1492.00 | 1483.64 | 0.00 | ORB-long ORB[1475.00,1485.70] vol=1.7x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-04-03 10:40:00 | 1489.44 | 1483.92 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 1481.95 | 1485.36 | 0.00 | ORB-short ORB[1483.20,1492.95] vol=2.2x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:50:00 | 1477.46 | 1483.47 | 0.00 | T1 1.5R @ 1477.46 |
| Target hit | 2024-04-04 11:10:00 | 1478.50 | 1477.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — BUY (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 1435.00 | 1427.63 | 0.00 | ORB-long ORB[1417.10,1428.15] vol=2.0x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:45:00 | 1443.74 | 1432.17 | 0.00 | T1 1.5R @ 1443.74 |
| Stop hit — per-position SL triggered | 2024-05-06 10:05:00 | 1435.00 | 1434.03 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:45:00 | 1419.85 | 1421.00 | 0.00 | ORB-short ORB[1420.10,1429.20] vol=2.7x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:10:00 | 1415.80 | 1420.21 | 0.00 | T1 1.5R @ 1415.80 |
| Stop hit — per-position SL triggered | 2024-05-07 10:50:00 | 1419.85 | 1419.89 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 09:50:00 | 1430.00 | 1433.57 | 0.00 | ORB-short ORB[1431.00,1439.75] vol=1.6x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-05-08 10:05:00 | 1433.16 | 1432.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 11:00:00 | 1248.00 | 2023-05-17 11:05:00 | 1249.83 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-05-18 09:45:00 | 1250.65 | 2023-05-18 10:25:00 | 1253.79 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-19 09:40:00 | 1264.70 | 2023-05-19 10:10:00 | 1268.78 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-05-19 09:40:00 | 1264.70 | 2023-05-19 10:25:00 | 1264.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-23 09:35:00 | 1310.85 | 2023-05-23 09:45:00 | 1307.82 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-24 11:15:00 | 1303.20 | 2023-05-24 11:35:00 | 1300.86 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-05-26 10:30:00 | 1318.85 | 2023-05-26 10:50:00 | 1316.47 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-05-29 09:30:00 | 1319.95 | 2023-05-29 10:15:00 | 1315.40 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-05-29 09:30:00 | 1319.95 | 2023-05-29 15:20:00 | 1314.45 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2023-05-30 10:55:00 | 1311.00 | 2023-05-30 11:00:00 | 1312.87 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-05-31 09:30:00 | 1328.55 | 2023-05-31 09:35:00 | 1326.07 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-02 09:35:00 | 1304.30 | 2023-06-02 10:30:00 | 1300.54 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-06-02 09:35:00 | 1304.30 | 2023-06-02 11:05:00 | 1304.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 10:10:00 | 1269.30 | 2023-06-09 10:45:00 | 1266.15 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-06-09 10:10:00 | 1269.30 | 2023-06-09 15:20:00 | 1265.05 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2023-06-12 11:05:00 | 1288.70 | 2023-06-12 11:35:00 | 1286.87 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-16 11:00:00 | 1284.40 | 2023-06-16 11:45:00 | 1286.15 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-06-28 11:10:00 | 1290.35 | 2023-06-28 11:30:00 | 1292.66 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2023-06-28 11:10:00 | 1290.35 | 2023-06-28 14:55:00 | 1290.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-30 09:30:00 | 1328.10 | 2023-06-30 09:40:00 | 1324.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-07 09:50:00 | 1342.65 | 2023-07-07 10:15:00 | 1340.18 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-11 09:40:00 | 1340.45 | 2023-07-11 10:00:00 | 1343.98 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-07-11 09:40:00 | 1340.45 | 2023-07-11 10:30:00 | 1340.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-13 10:45:00 | 1370.50 | 2023-07-13 12:05:00 | 1375.92 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-13 10:45:00 | 1370.50 | 2023-07-13 12:30:00 | 1370.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-17 10:10:00 | 1454.30 | 2023-07-17 10:20:00 | 1449.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-19 10:30:00 | 1479.95 | 2023-07-19 10:35:00 | 1473.05 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-07-19 10:30:00 | 1479.95 | 2023-07-19 15:20:00 | 1476.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-28 10:50:00 | 1345.00 | 2023-07-28 11:50:00 | 1341.35 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-28 10:50:00 | 1345.00 | 2023-07-28 15:20:00 | 1340.60 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2023-07-31 11:10:00 | 1350.00 | 2023-07-31 11:45:00 | 1353.55 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-07-31 11:10:00 | 1350.00 | 2023-07-31 12:45:00 | 1350.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-04 09:55:00 | 1381.70 | 2023-08-04 10:35:00 | 1378.41 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-11 09:35:00 | 1387.45 | 2023-08-11 09:40:00 | 1383.61 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-08-11 09:35:00 | 1387.45 | 2023-08-11 13:55:00 | 1378.85 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2023-08-18 10:55:00 | 1389.95 | 2023-08-18 11:35:00 | 1392.23 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-01 10:50:00 | 1437.25 | 2023-09-01 11:05:00 | 1434.75 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-11 11:05:00 | 1474.50 | 2023-09-11 12:30:00 | 1477.23 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2023-09-11 11:05:00 | 1474.50 | 2023-09-11 13:25:00 | 1474.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-14 09:30:00 | 1510.60 | 2023-09-14 09:50:00 | 1507.82 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-09-25 10:55:00 | 1473.90 | 2023-09-25 12:00:00 | 1476.47 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-09-27 10:55:00 | 1451.90 | 2023-09-27 11:10:00 | 1453.59 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-10-03 11:15:00 | 1425.20 | 2023-10-03 11:30:00 | 1427.65 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-04 11:05:00 | 1431.05 | 2023-10-04 11:25:00 | 1428.33 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-05 10:55:00 | 1470.50 | 2023-10-05 11:45:00 | 1466.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-11 09:30:00 | 1516.50 | 2023-10-11 09:35:00 | 1513.28 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-12 10:55:00 | 1481.85 | 2023-10-12 11:55:00 | 1485.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-09 10:25:00 | 1382.00 | 2023-11-09 11:25:00 | 1384.16 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-11-10 10:55:00 | 1365.40 | 2023-11-10 11:00:00 | 1362.43 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-11-10 10:55:00 | 1365.40 | 2023-11-10 12:00:00 | 1365.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-15 11:05:00 | 1399.90 | 2023-11-15 12:00:00 | 1397.48 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1423.95 | 2023-11-16 11:15:00 | 1427.08 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1423.95 | 2023-11-16 15:20:00 | 1442.55 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2023-11-17 10:50:00 | 1448.00 | 2023-11-17 11:15:00 | 1445.35 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-21 11:10:00 | 1439.65 | 2023-11-21 11:20:00 | 1441.39 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-11-24 11:15:00 | 1441.45 | 2023-11-24 13:00:00 | 1438.51 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-24 11:15:00 | 1441.45 | 2023-11-24 15:20:00 | 1439.65 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2023-12-04 09:40:00 | 1470.45 | 2023-12-04 10:00:00 | 1466.96 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-06 09:45:00 | 1470.45 | 2023-12-06 09:50:00 | 1467.52 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-08 11:05:00 | 1476.30 | 2023-12-08 11:10:00 | 1473.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-12-12 09:55:00 | 1486.20 | 2023-12-12 10:15:00 | 1482.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-15 10:20:00 | 1563.20 | 2023-12-15 10:25:00 | 1558.03 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-19 09:40:00 | 1551.95 | 2023-12-19 09:55:00 | 1545.70 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-19 09:40:00 | 1551.95 | 2023-12-19 11:50:00 | 1551.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-27 11:10:00 | 1560.40 | 2023-12-27 12:30:00 | 1557.61 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-01-01 11:05:00 | 1545.25 | 2024-01-01 11:15:00 | 1548.67 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-01-01 11:05:00 | 1545.25 | 2024-01-01 11:40:00 | 1545.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 10:30:00 | 1531.55 | 2024-01-02 11:15:00 | 1527.18 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-01-02 10:30:00 | 1531.55 | 2024-01-02 14:00:00 | 1531.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 09:35:00 | 1534.95 | 2024-01-05 09:50:00 | 1530.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-01-17 10:10:00 | 1641.85 | 2024-01-17 10:30:00 | 1637.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-19 11:15:00 | 1649.90 | 2024-01-19 11:30:00 | 1644.83 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-01-19 11:15:00 | 1649.90 | 2024-01-19 11:35:00 | 1649.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-24 09:55:00 | 1665.50 | 2024-01-24 10:05:00 | 1659.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-01-29 10:40:00 | 1656.50 | 2024-01-29 11:20:00 | 1649.04 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-01-29 10:40:00 | 1656.50 | 2024-01-29 12:10:00 | 1656.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-01 11:00:00 | 1673.40 | 2024-02-01 11:05:00 | 1669.71 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-02-06 09:35:00 | 1709.05 | 2024-02-06 09:45:00 | 1717.09 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-02-06 09:35:00 | 1709.05 | 2024-02-06 10:25:00 | 1709.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-07 11:05:00 | 1701.50 | 2024-02-07 11:55:00 | 1697.17 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-07 11:05:00 | 1701.50 | 2024-02-07 12:20:00 | 1701.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 10:40:00 | 1689.55 | 2024-02-08 11:30:00 | 1683.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-02-08 10:40:00 | 1689.55 | 2024-02-08 11:40:00 | 1689.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-09 11:00:00 | 1662.95 | 2024-02-09 11:10:00 | 1667.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-13 09:30:00 | 1667.90 | 2024-02-13 09:55:00 | 1672.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-14 09:30:00 | 1641.70 | 2024-02-14 09:35:00 | 1647.44 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-20 10:50:00 | 1678.05 | 2024-02-20 11:10:00 | 1672.90 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-02-20 10:50:00 | 1678.05 | 2024-02-20 11:45:00 | 1678.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-26 11:00:00 | 1655.55 | 2024-02-26 11:40:00 | 1651.47 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-26 11:00:00 | 1655.55 | 2024-02-26 13:35:00 | 1655.50 | TARGET_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-27 10:25:00 | 1656.25 | 2024-02-27 10:50:00 | 1659.34 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-02-28 10:40:00 | 1686.45 | 2024-02-28 10:50:00 | 1682.60 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-04 10:50:00 | 1648.15 | 2024-03-04 10:55:00 | 1643.84 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-03-04 10:50:00 | 1648.15 | 2024-03-04 15:20:00 | 1636.95 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-03-05 10:00:00 | 1609.80 | 2024-03-05 10:05:00 | 1613.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-03-07 10:40:00 | 1618.50 | 2024-03-07 10:50:00 | 1623.67 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-07 10:40:00 | 1618.50 | 2024-03-07 12:40:00 | 1618.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-11 11:00:00 | 1594.00 | 2024-03-11 11:10:00 | 1588.52 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-03-11 11:00:00 | 1594.00 | 2024-03-11 11:25:00 | 1594.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-21 11:05:00 | 1558.30 | 2024-03-21 11:35:00 | 1553.49 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-03-21 11:05:00 | 1558.30 | 2024-03-21 15:20:00 | 1555.25 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-04-01 11:00:00 | 1502.50 | 2024-04-01 14:40:00 | 1497.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-04-01 11:00:00 | 1502.50 | 2024-04-01 15:20:00 | 1495.20 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-04-03 10:35:00 | 1492.00 | 2024-04-03 10:40:00 | 1489.44 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-04-04 09:30:00 | 1481.95 | 2024-04-04 09:50:00 | 1477.46 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-04-04 09:30:00 | 1481.95 | 2024-04-04 11:10:00 | 1478.50 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-05-06 09:35:00 | 1435.00 | 2024-05-06 09:45:00 | 1443.74 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-05-06 09:35:00 | 1435.00 | 2024-05-06 10:05:00 | 1435.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 09:45:00 | 1419.85 | 2024-05-07 10:10:00 | 1415.80 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-05-07 09:45:00 | 1419.85 | 2024-05-07 10:50:00 | 1419.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-08 09:50:00 | 1430.00 | 2024-05-08 10:05:00 | 1433.16 | STOP_HIT | 1.00 | -0.22% |
