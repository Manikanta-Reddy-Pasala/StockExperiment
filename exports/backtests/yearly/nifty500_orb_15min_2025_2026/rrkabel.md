# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1945.00
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 8 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 52
- **Target hits / Stop hits / Partials:** 8 / 52 / 22
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 15.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 16 | 34.0% | 4 | 31 | 12 | 0.20% | 9.3% |
| BUY @ 2nd Alert (retest1) | 47 | 16 | 34.0% | 4 | 31 | 12 | 0.20% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 14 | 40.0% | 4 | 21 | 10 | 0.18% | 6.1% |
| SELL @ 2nd Alert (retest1) | 35 | 14 | 40.0% | 4 | 21 | 10 | 0.18% | 6.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 82 | 30 | 36.6% | 8 | 52 | 22 | 0.19% | 15.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 1297.60 | 1316.83 | 0.00 | ORB-short ORB[1313.30,1328.00] vol=2.0x ATR=7.14 |
| Stop hit — per-position SL triggered | 2025-05-19 09:40:00 | 1304.74 | 1315.76 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:35:00 | 1297.50 | 1311.26 | 0.00 | ORB-short ORB[1311.00,1325.10] vol=3.3x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-05-28 10:45:00 | 1301.60 | 1310.95 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1330.10 | 1321.59 | 0.00 | ORB-long ORB[1310.90,1330.00] vol=2.1x ATR=6.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:25:00 | 1339.66 | 1324.02 | 0.00 | T1 1.5R @ 1339.66 |
| Target hit | 2025-05-30 15:20:00 | 1428.60 | 1397.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 10:05:00 | 1412.80 | 1422.83 | 0.00 | ORB-short ORB[1415.00,1433.90] vol=1.8x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:35:00 | 1403.42 | 1416.15 | 0.00 | T1 1.5R @ 1403.42 |
| Target hit | 2025-06-09 15:20:00 | 1387.30 | 1400.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1341.80 | 1333.65 | 0.00 | ORB-long ORB[1323.60,1338.10] vol=1.7x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 1336.85 | 1336.50 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1332.70 | 1343.84 | 0.00 | ORB-short ORB[1342.90,1356.40] vol=2.4x ATR=3.53 |
| Stop hit — per-position SL triggered | 2025-06-26 11:30:00 | 1336.23 | 1342.32 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 1365.00 | 1355.18 | 0.00 | ORB-long ORB[1347.10,1363.00] vol=5.6x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:50:00 | 1372.04 | 1360.15 | 0.00 | T1 1.5R @ 1372.04 |
| Target hit | 2025-06-27 13:15:00 | 1369.60 | 1369.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 1349.30 | 1351.04 | 0.00 | ORB-short ORB[1350.00,1369.30] vol=2.6x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-06-30 13:50:00 | 1352.76 | 1350.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 1363.60 | 1359.56 | 0.00 | ORB-long ORB[1346.10,1361.70] vol=3.6x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 12:05:00 | 1369.63 | 1361.25 | 0.00 | T1 1.5R @ 1369.63 |
| Target hit | 2025-07-01 14:45:00 | 1400.00 | 1400.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2025-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:05:00 | 1405.00 | 1388.02 | 0.00 | ORB-long ORB[1371.10,1384.20] vol=6.6x ATR=6.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 10:10:00 | 1414.47 | 1399.94 | 0.00 | T1 1.5R @ 1414.47 |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 1405.00 | 1402.39 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 1440.00 | 1432.81 | 0.00 | ORB-long ORB[1418.80,1437.90] vol=5.5x ATR=6.90 |
| Stop hit — per-position SL triggered | 2025-07-22 09:55:00 | 1433.10 | 1433.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 1390.50 | 1377.08 | 0.00 | ORB-long ORB[1357.60,1377.50] vol=1.8x ATR=9.67 |
| Stop hit — per-position SL triggered | 2025-07-29 10:10:00 | 1380.83 | 1382.88 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:45:00 | 1216.20 | 1228.71 | 0.00 | ORB-short ORB[1225.10,1241.70] vol=1.7x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 1222.18 | 1224.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:35:00 | 1217.60 | 1224.86 | 0.00 | ORB-short ORB[1220.70,1237.50] vol=1.6x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:20:00 | 1211.55 | 1221.16 | 0.00 | T1 1.5R @ 1211.55 |
| Target hit | 2025-08-12 14:00:00 | 1216.90 | 1216.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2025-08-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1206.80 | 1211.85 | 0.00 | ORB-short ORB[1212.00,1227.20] vol=3.0x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-08-14 10:25:00 | 1210.81 | 1210.36 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:00:00 | 1227.60 | 1218.36 | 0.00 | ORB-long ORB[1212.20,1225.70] vol=2.1x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:35:00 | 1232.73 | 1220.72 | 0.00 | T1 1.5R @ 1232.73 |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 1227.60 | 1222.46 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:00:00 | 1231.70 | 1223.79 | 0.00 | ORB-long ORB[1211.80,1229.40] vol=1.5x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:10:00 | 1238.50 | 1226.79 | 0.00 | T1 1.5R @ 1238.50 |
| Stop hit — per-position SL triggered | 2025-08-22 10:45:00 | 1231.70 | 1229.28 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 1199.50 | 1204.90 | 0.00 | ORB-short ORB[1201.00,1216.50] vol=3.2x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:40:00 | 1194.30 | 1202.27 | 0.00 | T1 1.5R @ 1194.30 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 1199.50 | 1202.00 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:30:00 | 1184.00 | 1191.60 | 0.00 | ORB-short ORB[1188.20,1202.90] vol=1.7x ATR=4.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:10:00 | 1177.33 | 1184.62 | 0.00 | T1 1.5R @ 1177.33 |
| Stop hit — per-position SL triggered | 2025-08-29 10:35:00 | 1184.00 | 1184.10 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:15:00 | 1185.70 | 1177.86 | 0.00 | ORB-long ORB[1170.00,1182.00] vol=1.5x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-09-01 11:25:00 | 1182.74 | 1178.13 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:50:00 | 1200.10 | 1193.67 | 0.00 | ORB-long ORB[1183.00,1197.10] vol=2.4x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-09-02 11:40:00 | 1195.83 | 1196.33 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:05:00 | 1210.70 | 1198.40 | 0.00 | ORB-long ORB[1182.80,1195.90] vol=2.0x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-09-03 10:20:00 | 1206.86 | 1199.60 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 11:05:00 | 1220.90 | 1215.75 | 0.00 | ORB-long ORB[1205.70,1218.30] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-09-09 11:10:00 | 1218.46 | 1215.92 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:45:00 | 1232.20 | 1227.55 | 0.00 | ORB-long ORB[1213.90,1230.00] vol=2.4x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-09-11 10:05:00 | 1227.60 | 1229.38 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:40:00 | 1239.30 | 1247.01 | 0.00 | ORB-short ORB[1245.30,1259.40] vol=6.1x ATR=4.82 |
| Stop hit — per-position SL triggered | 2025-09-17 11:55:00 | 1244.12 | 1245.38 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:25:00 | 1292.60 | 1273.44 | 0.00 | ORB-long ORB[1255.50,1270.00] vol=2.5x ATR=5.13 |
| Stop hit — per-position SL triggered | 2025-09-22 10:30:00 | 1287.47 | 1275.19 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:10:00 | 1275.20 | 1290.74 | 0.00 | ORB-short ORB[1282.40,1300.90] vol=1.6x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 12:50:00 | 1268.91 | 1287.69 | 0.00 | T1 1.5R @ 1268.91 |
| Target hit | 2025-09-24 15:20:00 | 1261.10 | 1279.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:35:00 | 1251.30 | 1254.57 | 0.00 | ORB-short ORB[1252.40,1260.80] vol=2.0x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:45:00 | 1246.66 | 1250.41 | 0.00 | T1 1.5R @ 1246.66 |
| Stop hit — per-position SL triggered | 2025-10-09 11:55:00 | 1251.30 | 1249.16 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:10:00 | 1261.60 | 1268.95 | 0.00 | ORB-short ORB[1261.80,1272.00] vol=2.5x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-10-13 11:25:00 | 1265.29 | 1267.66 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:30:00 | 1251.40 | 1256.51 | 0.00 | ORB-short ORB[1252.10,1266.80] vol=3.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-10-14 10:55:00 | 1254.79 | 1256.02 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 1249.10 | 1252.72 | 0.00 | ORB-short ORB[1254.10,1263.90] vol=2.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-10-15 11:00:00 | 1251.78 | 1252.49 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 1279.40 | 1274.90 | 0.00 | ORB-long ORB[1260.00,1272.60] vol=4.3x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-10-20 09:35:00 | 1275.27 | 1273.50 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 1423.70 | 1414.60 | 0.00 | ORB-long ORB[1401.50,1420.00] vol=4.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-10-31 10:05:00 | 1418.69 | 1416.48 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:00:00 | 1343.80 | 1349.89 | 0.00 | ORB-short ORB[1344.90,1358.10] vol=1.8x ATR=4.43 |
| Stop hit — per-position SL triggered | 2025-11-11 10:10:00 | 1348.23 | 1350.19 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:20:00 | 1393.00 | 1389.50 | 0.00 | ORB-long ORB[1376.70,1386.70] vol=5.9x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-11-17 10:55:00 | 1388.79 | 1389.89 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 1373.10 | 1367.61 | 0.00 | ORB-long ORB[1351.60,1369.90] vol=2.4x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:40:00 | 1379.24 | 1369.82 | 0.00 | T1 1.5R @ 1379.24 |
| Stop hit — per-position SL triggered | 2025-11-21 09:45:00 | 1373.10 | 1370.40 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 10:40:00 | 1351.40 | 1346.85 | 0.00 | ORB-long ORB[1335.30,1344.70] vol=3.0x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-11-25 10:50:00 | 1347.75 | 1347.08 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:10:00 | 1391.60 | 1398.46 | 0.00 | ORB-short ORB[1396.60,1414.50] vol=2.3x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:40:00 | 1386.55 | 1397.66 | 0.00 | T1 1.5R @ 1386.55 |
| Stop hit — per-position SL triggered | 2025-12-01 11:55:00 | 1391.60 | 1396.71 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:50:00 | 1388.00 | 1392.53 | 0.00 | ORB-short ORB[1389.60,1401.00] vol=2.8x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-12-02 10:00:00 | 1391.50 | 1392.35 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 1371.00 | 1362.36 | 0.00 | ORB-long ORB[1353.60,1367.00] vol=3.1x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-12-08 11:30:00 | 1366.51 | 1362.94 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:35:00 | 1343.10 | 1348.54 | 0.00 | ORB-short ORB[1346.40,1356.60] vol=2.8x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-12-09 09:40:00 | 1347.84 | 1348.13 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 1433.00 | 1427.31 | 0.00 | ORB-long ORB[1408.80,1428.80] vol=6.9x ATR=4.32 |
| Stop hit — per-position SL triggered | 2025-12-12 09:50:00 | 1428.68 | 1427.70 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 09:30:00 | 1443.20 | 1434.79 | 0.00 | ORB-long ORB[1419.90,1438.60] vol=1.6x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-12-15 09:40:00 | 1438.15 | 1435.80 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 09:50:00 | 1425.00 | 1428.68 | 0.00 | ORB-short ORB[1426.00,1438.30] vol=1.7x ATR=5.08 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 1430.08 | 1428.46 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 1460.60 | 1447.07 | 0.00 | ORB-long ORB[1432.60,1451.00] vol=1.6x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:20:00 | 1471.55 | 1454.52 | 0.00 | T1 1.5R @ 1471.55 |
| Stop hit — per-position SL triggered | 2025-12-30 10:25:00 | 1460.60 | 1455.84 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:40:00 | 1464.00 | 1453.89 | 0.00 | ORB-long ORB[1441.90,1460.00] vol=2.1x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-01-01 10:00:00 | 1458.63 | 1458.52 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:00:00 | 1501.00 | 1491.91 | 0.00 | ORB-long ORB[1480.00,1492.60] vol=2.3x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:05:00 | 1506.77 | 1493.83 | 0.00 | T1 1.5R @ 1506.77 |
| Stop hit — per-position SL triggered | 2026-01-16 10:20:00 | 1501.00 | 1494.96 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 1411.50 | 1418.01 | 0.00 | ORB-short ORB[1420.40,1437.50] vol=3.5x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:40:00 | 1402.58 | 1414.99 | 0.00 | T1 1.5R @ 1402.58 |
| Stop hit — per-position SL triggered | 2026-01-20 09:45:00 | 1411.50 | 1414.63 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 1439.90 | 1422.60 | 0.00 | ORB-long ORB[1411.40,1421.00] vol=2.8x ATR=6.76 |
| Stop hit — per-position SL triggered | 2026-01-22 09:50:00 | 1433.14 | 1429.09 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 1459.60 | 1473.53 | 0.00 | ORB-short ORB[1469.40,1488.00] vol=2.0x ATR=6.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1449.15 | 1466.22 | 0.00 | T1 1.5R @ 1449.15 |
| Target hit | 2026-02-10 15:20:00 | 1419.40 | 1436.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 1419.00 | 1409.72 | 0.00 | ORB-long ORB[1396.10,1412.30] vol=4.6x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 1413.91 | 1411.46 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 1433.00 | 1427.27 | 0.00 | ORB-long ORB[1415.50,1426.00] vol=2.4x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:30:00 | 1441.08 | 1431.49 | 0.00 | T1 1.5R @ 1441.08 |
| Target hit | 2026-02-19 11:30:00 | 1442.20 | 1442.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1425.90 | 1422.78 | 0.00 | ORB-long ORB[1407.10,1425.00] vol=1.8x ATR=5.96 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 1419.94 | 1422.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 1471.60 | 1456.87 | 0.00 | ORB-long ORB[1441.00,1454.90] vol=3.6x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 1465.46 | 1457.86 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 1452.00 | 1478.05 | 0.00 | ORB-short ORB[1492.00,1508.30] vol=1.6x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:05:00 | 1440.87 | 1472.06 | 0.00 | T1 1.5R @ 1440.87 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 1452.00 | 1462.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:15:00 | 1536.70 | 1519.56 | 0.00 | ORB-long ORB[1491.30,1511.70] vol=2.0x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 13:05:00 | 1547.19 | 1532.32 | 0.00 | T1 1.5R @ 1547.19 |
| Stop hit — per-position SL triggered | 2026-03-06 14:40:00 | 1536.70 | 1535.76 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 1387.10 | 1382.22 | 0.00 | ORB-long ORB[1365.70,1386.00] vol=3.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 1381.32 | 1382.25 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 1380.40 | 1378.05 | 0.00 | ORB-long ORB[1366.40,1380.00] vol=2.9x ATR=5.89 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 1374.51 | 1377.33 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 1461.40 | 1475.43 | 0.00 | ORB-short ORB[1469.20,1489.00] vol=1.7x ATR=5.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 1467.25 | 1469.58 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1469.20 | 1456.21 | 0.00 | ORB-long ORB[1445.60,1464.30] vol=2.1x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:25:00 | 1477.99 | 1465.62 | 0.00 | T1 1.5R @ 1477.99 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 1469.20 | 1470.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 09:35:00 | 1297.60 | 2025-05-19 09:40:00 | 1304.74 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-05-28 10:35:00 | 1297.50 | 2025-05-28 10:45:00 | 1301.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-30 10:15:00 | 1330.10 | 2025-05-30 10:25:00 | 1339.66 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-05-30 10:15:00 | 1330.10 | 2025-05-30 15:20:00 | 1428.60 | TARGET_HIT | 0.50 | 7.41% |
| SELL | retest1 | 2025-06-09 10:05:00 | 1412.80 | 2025-06-09 10:35:00 | 1403.42 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-06-09 10:05:00 | 1412.80 | 2025-06-09 15:20:00 | 1387.30 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2025-06-19 09:30:00 | 1341.80 | 2025-06-19 09:50:00 | 1336.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-26 11:15:00 | 1332.70 | 2025-06-26 11:30:00 | 1336.23 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-27 09:30:00 | 1365.00 | 2025-06-27 09:50:00 | 1372.04 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-06-27 09:30:00 | 1365.00 | 2025-06-27 13:15:00 | 1369.60 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-30 11:15:00 | 1349.30 | 2025-06-30 13:50:00 | 1352.76 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-01 10:50:00 | 1363.60 | 2025-07-01 12:05:00 | 1369.63 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-01 10:50:00 | 1363.60 | 2025-07-01 14:45:00 | 1400.00 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2025-07-16 10:05:00 | 1405.00 | 2025-07-16 10:10:00 | 1414.47 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-07-16 10:05:00 | 1405.00 | 2025-07-16 10:15:00 | 1405.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 09:30:00 | 1440.00 | 2025-07-22 09:55:00 | 1433.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-07-29 09:40:00 | 1390.50 | 2025-07-29 10:10:00 | 1380.83 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest1 | 2025-08-11 09:45:00 | 1216.20 | 2025-08-11 10:15:00 | 1222.18 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-08-12 09:35:00 | 1217.60 | 2025-08-12 10:20:00 | 1211.55 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-08-12 09:35:00 | 1217.60 | 2025-08-12 14:00:00 | 1216.90 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-08-14 09:45:00 | 1206.80 | 2025-08-14 10:25:00 | 1210.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-21 11:00:00 | 1227.60 | 2025-08-21 11:35:00 | 1232.73 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-21 11:00:00 | 1227.60 | 2025-08-21 12:15:00 | 1227.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 10:00:00 | 1231.70 | 2025-08-22 10:10:00 | 1238.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-08-22 10:00:00 | 1231.70 | 2025-08-22 10:45:00 | 1231.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:30:00 | 1199.50 | 2025-08-26 09:40:00 | 1194.30 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-08-26 09:30:00 | 1199.50 | 2025-08-26 09:45:00 | 1199.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:30:00 | 1184.00 | 2025-08-29 10:10:00 | 1177.33 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-08-29 09:30:00 | 1184.00 | 2025-08-29 10:35:00 | 1184.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 11:15:00 | 1185.70 | 2025-09-01 11:25:00 | 1182.74 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-02 09:50:00 | 1200.10 | 2025-09-02 11:40:00 | 1195.83 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-03 10:05:00 | 1210.70 | 2025-09-03 10:20:00 | 1206.86 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-09 11:05:00 | 1220.90 | 2025-09-09 11:10:00 | 1218.46 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-11 09:45:00 | 1232.20 | 2025-09-11 10:05:00 | 1227.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-17 10:40:00 | 1239.30 | 2025-09-17 11:55:00 | 1244.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-22 10:25:00 | 1292.60 | 2025-09-22 10:30:00 | 1287.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-24 11:10:00 | 1275.20 | 2025-09-24 12:50:00 | 1268.91 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-24 11:10:00 | 1275.20 | 2025-09-24 15:20:00 | 1261.10 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2025-10-09 10:35:00 | 1251.30 | 2025-10-09 10:45:00 | 1246.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-09 10:35:00 | 1251.30 | 2025-10-09 11:55:00 | 1251.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 11:10:00 | 1261.60 | 2025-10-13 11:25:00 | 1265.29 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-14 10:30:00 | 1251.40 | 2025-10-14 10:55:00 | 1254.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-15 10:50:00 | 1249.10 | 2025-10-15 11:00:00 | 1251.78 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-20 09:30:00 | 1279.40 | 2025-10-20 09:35:00 | 1275.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-31 09:50:00 | 1423.70 | 2025-10-31 10:05:00 | 1418.69 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-11 10:00:00 | 1343.80 | 2025-11-11 10:10:00 | 1348.23 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-17 10:20:00 | 1393.00 | 2025-11-17 10:55:00 | 1388.79 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-21 09:35:00 | 1373.10 | 2025-11-21 09:40:00 | 1379.24 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-21 09:35:00 | 1373.10 | 2025-11-21 09:45:00 | 1373.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-25 10:40:00 | 1351.40 | 2025-11-25 10:50:00 | 1347.75 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-01 11:10:00 | 1391.60 | 2025-12-01 11:40:00 | 1386.55 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-01 11:10:00 | 1391.60 | 2025-12-01 11:55:00 | 1391.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 09:50:00 | 1388.00 | 2025-12-02 10:00:00 | 1391.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-08 11:00:00 | 1371.00 | 2025-12-08 11:30:00 | 1366.51 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-09 09:35:00 | 1343.10 | 2025-12-09 09:40:00 | 1347.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-12 09:40:00 | 1433.00 | 2025-12-12 09:50:00 | 1428.68 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-15 09:30:00 | 1443.20 | 2025-12-15 09:40:00 | 1438.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-17 09:50:00 | 1425.00 | 2025-12-17 10:00:00 | 1430.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-30 09:40:00 | 1460.60 | 2025-12-30 10:20:00 | 1471.55 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-12-30 09:40:00 | 1460.60 | 2025-12-30 10:25:00 | 1460.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 09:40:00 | 1464.00 | 2026-01-01 10:00:00 | 1458.63 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-16 10:00:00 | 1501.00 | 2026-01-16 10:05:00 | 1506.77 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-16 10:00:00 | 1501.00 | 2026-01-16 10:20:00 | 1501.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 09:30:00 | 1411.50 | 2026-01-20 09:40:00 | 1402.58 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-01-20 09:30:00 | 1411.50 | 2026-01-20 09:45:00 | 1411.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 09:30:00 | 1439.90 | 2026-01-22 09:50:00 | 1433.14 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-02-10 09:35:00 | 1459.60 | 2026-02-10 10:10:00 | 1449.15 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-02-10 09:35:00 | 1459.60 | 2026-02-10 15:20:00 | 1419.40 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2026-02-17 09:50:00 | 1419.00 | 2026-02-17 10:40:00 | 1413.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-19 10:00:00 | 1433.00 | 2026-02-19 10:30:00 | 1441.08 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-19 10:00:00 | 1433.00 | 2026-02-19 11:30:00 | 1442.20 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-20 10:15:00 | 1425.90 | 2026-02-20 10:55:00 | 1419.94 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-23 09:45:00 | 1471.60 | 2026-02-23 09:50:00 | 1465.46 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1452.00 | 2026-03-05 11:05:00 | 1440.87 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1452.00 | 2026-03-05 11:25:00 | 1452.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1536.70 | 2026-03-06 13:05:00 | 1547.19 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1536.70 | 2026-03-06 14:40:00 | 1536.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 09:45:00 | 1387.10 | 2026-03-17 09:55:00 | 1381.32 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 09:30:00 | 1380.40 | 2026-03-25 09:35:00 | 1374.51 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-21 09:30:00 | 1461.40 | 2026-04-21 09:45:00 | 1467.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1469.20 | 2026-04-22 10:25:00 | 1477.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1469.20 | 2026-04-22 11:40:00 | 1469.20 | STOP_HIT | 0.50 | 0.00% |
