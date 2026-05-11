# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 515.50
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 10 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 44
- **Target hits / Stop hits / Partials:** 10 / 43 / 21
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 13.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 10 | 31.2% | 2 | 22 | 8 | -0.03% | -0.9% |
| BUY @ 2nd Alert (retest1) | 32 | 10 | 31.2% | 2 | 22 | 8 | -0.03% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 20 | 47.6% | 8 | 21 | 13 | 0.33% | 13.9% |
| SELL @ 2nd Alert (retest1) | 42 | 20 | 47.6% | 8 | 21 | 13 | 0.33% | 13.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 74 | 30 | 40.5% | 10 | 43 | 21 | 0.18% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:45:00 | 1188.00 | 1196.41 | 0.00 | ORB-short ORB[1188.85,1204.70] vol=1.7x ATR=7.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:55:00 | 1176.58 | 1188.74 | 0.00 | T1 1.5R @ 1176.58 |
| Target hit | 2024-05-15 11:25:00 | 1182.35 | 1179.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 1215.10 | 1206.83 | 0.00 | ORB-long ORB[1196.10,1210.00] vol=1.9x ATR=7.52 |
| Stop hit — per-position SL triggered | 2024-05-16 09:45:00 | 1207.58 | 1209.80 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 1148.00 | 1157.03 | 0.00 | ORB-short ORB[1156.00,1165.90] vol=1.8x ATR=8.45 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 1156.45 | 1156.32 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:05:00 | 1188.85 | 1174.40 | 0.00 | ORB-long ORB[1165.00,1175.00] vol=5.3x ATR=5.78 |
| Stop hit — per-position SL triggered | 2024-05-23 11:10:00 | 1183.07 | 1174.81 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 1157.75 | 1162.40 | 0.00 | ORB-short ORB[1161.05,1168.95] vol=1.6x ATR=7.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:10:00 | 1147.20 | 1153.17 | 0.00 | T1 1.5R @ 1147.20 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 1157.75 | 1153.00 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:50:00 | 1187.05 | 1172.98 | 0.00 | ORB-long ORB[1160.00,1173.00] vol=5.9x ATR=5.62 |
| Stop hit — per-position SL triggered | 2024-05-29 11:00:00 | 1181.43 | 1174.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 1132.50 | 1137.61 | 0.00 | ORB-short ORB[1138.00,1150.00] vol=3.1x ATR=7.21 |
| Stop hit — per-position SL triggered | 2024-05-31 09:45:00 | 1139.71 | 1137.64 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:50:00 | 1388.85 | 1377.72 | 0.00 | ORB-long ORB[1370.00,1388.40] vol=1.8x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:55:00 | 1402.10 | 1383.27 | 0.00 | T1 1.5R @ 1402.10 |
| Stop hit — per-position SL triggered | 2024-06-13 10:00:00 | 1388.85 | 1384.64 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 1379.60 | 1366.64 | 0.00 | ORB-long ORB[1355.60,1372.70] vol=1.5x ATR=6.37 |
| Stop hit — per-position SL triggered | 2024-06-18 09:40:00 | 1373.23 | 1370.61 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:45:00 | 1411.50 | 1425.36 | 0.00 | ORB-short ORB[1425.00,1440.00] vol=1.6x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-06-25 09:55:00 | 1419.01 | 1423.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 1394.80 | 1403.25 | 0.00 | ORB-short ORB[1403.00,1414.00] vol=2.4x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-06-26 10:10:00 | 1401.34 | 1398.33 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 1412.65 | 1397.83 | 0.00 | ORB-long ORB[1387.00,1399.40] vol=2.3x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:45:00 | 1422.21 | 1419.50 | 0.00 | T1 1.5R @ 1422.21 |
| Target hit | 2024-06-27 10:10:00 | 1421.00 | 1425.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2024-07-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:35:00 | 1425.70 | 1417.42 | 0.00 | ORB-long ORB[1408.00,1423.50] vol=1.9x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:10:00 | 1433.76 | 1421.48 | 0.00 | T1 1.5R @ 1433.76 |
| Target hit | 2024-07-01 12:10:00 | 1435.10 | 1436.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 1415.90 | 1425.15 | 0.00 | ORB-short ORB[1425.00,1436.45] vol=3.2x ATR=6.96 |
| Stop hit — per-position SL triggered | 2024-07-02 09:35:00 | 1422.86 | 1424.47 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 1455.05 | 1448.58 | 0.00 | ORB-long ORB[1436.00,1454.05] vol=1.8x ATR=7.07 |
| Stop hit — per-position SL triggered | 2024-07-03 14:25:00 | 1447.98 | 1452.51 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:05:00 | 1454.95 | 1439.68 | 0.00 | ORB-long ORB[1424.10,1440.90] vol=10.2x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-07-05 11:10:00 | 1447.70 | 1440.71 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 1466.15 | 1455.03 | 0.00 | ORB-long ORB[1440.95,1454.55] vol=5.6x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 1459.09 | 1456.51 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:20:00 | 1408.00 | 1421.19 | 0.00 | ORB-short ORB[1417.90,1436.55] vol=2.0x ATR=5.70 |
| Stop hit — per-position SL triggered | 2024-07-09 10:40:00 | 1413.70 | 1419.26 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:00:00 | 1364.95 | 1376.96 | 0.00 | ORB-short ORB[1374.00,1393.00] vol=1.5x ATR=8.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 1351.49 | 1373.82 | 0.00 | T1 1.5R @ 1351.49 |
| Target hit | 2024-07-10 11:05:00 | 1347.45 | 1346.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2024-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:10:00 | 1362.80 | 1375.09 | 0.00 | ORB-short ORB[1370.15,1384.00] vol=3.7x ATR=7.97 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 1370.77 | 1369.32 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 1411.05 | 1427.58 | 0.00 | ORB-short ORB[1420.10,1439.35] vol=1.6x ATR=9.00 |
| Stop hit — per-position SL triggered | 2024-07-15 09:45:00 | 1420.05 | 1426.01 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1426.00 | 1437.39 | 0.00 | ORB-short ORB[1431.40,1448.95] vol=1.7x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:35:00 | 1418.29 | 1435.09 | 0.00 | T1 1.5R @ 1418.29 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1426.00 | 1433.87 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:40:00 | 1278.00 | 1288.07 | 0.00 | ORB-short ORB[1285.10,1299.15] vol=1.5x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 12:35:00 | 1271.51 | 1283.54 | 0.00 | T1 1.5R @ 1271.51 |
| Target hit | 2024-07-31 15:20:00 | 1239.95 | 1261.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:40:00 | 1314.15 | 1300.39 | 0.00 | ORB-long ORB[1290.10,1300.00] vol=6.1x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-08-22 10:50:00 | 1308.00 | 1303.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:15:00 | 1337.40 | 1324.63 | 0.00 | ORB-long ORB[1318.40,1330.50] vol=2.5x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:25:00 | 1345.36 | 1327.84 | 0.00 | T1 1.5R @ 1345.36 |
| Stop hit — per-position SL triggered | 2024-08-30 10:40:00 | 1337.40 | 1328.77 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 1260.05 | 1263.44 | 0.00 | ORB-short ORB[1262.00,1275.00] vol=1.8x ATR=3.15 |
| Stop hit — per-position SL triggered | 2024-09-17 11:15:00 | 1263.20 | 1262.99 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:00:00 | 1244.95 | 1253.65 | 0.00 | ORB-short ORB[1250.10,1262.95] vol=2.2x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:10:00 | 1237.70 | 1250.54 | 0.00 | T1 1.5R @ 1237.70 |
| Stop hit — per-position SL triggered | 2024-09-18 11:00:00 | 1244.95 | 1247.96 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:15:00 | 1222.75 | 1236.84 | 0.00 | ORB-short ORB[1237.00,1253.25] vol=4.8x ATR=6.27 |
| Stop hit — per-position SL triggered | 2024-09-20 10:35:00 | 1229.02 | 1234.81 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1225.00 | 1226.83 | 0.00 | ORB-short ORB[1226.00,1233.00] vol=1.6x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:55:00 | 1219.59 | 1225.07 | 0.00 | T1 1.5R @ 1219.59 |
| Stop hit — per-position SL triggered | 2024-09-26 10:10:00 | 1225.00 | 1224.89 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:15:00 | 1195.00 | 1204.46 | 0.00 | ORB-short ORB[1197.35,1211.00] vol=1.8x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-10-01 10:30:00 | 1198.93 | 1203.76 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:35:00 | 1202.75 | 1199.32 | 0.00 | ORB-long ORB[1187.00,1198.90] vol=4.5x ATR=5.54 |
| Stop hit — per-position SL triggered | 2024-10-09 10:40:00 | 1197.21 | 1199.30 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1228.00 | 1209.91 | 0.00 | ORB-long ORB[1181.30,1199.00] vol=7.7x ATR=8.23 |
| Stop hit — per-position SL triggered | 2024-10-10 09:35:00 | 1219.77 | 1215.03 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 1198.95 | 1196.45 | 0.00 | ORB-long ORB[1188.40,1198.65] vol=4.5x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:10:00 | 1206.25 | 1198.80 | 0.00 | T1 1.5R @ 1206.25 |
| Stop hit — per-position SL triggered | 2024-10-11 10:45:00 | 1198.95 | 1200.10 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 1203.70 | 1195.90 | 0.00 | ORB-long ORB[1190.00,1200.80] vol=1.5x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-10-14 09:50:00 | 1198.90 | 1196.20 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:50:00 | 1192.25 | 1197.01 | 0.00 | ORB-short ORB[1195.50,1207.15] vol=2.5x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 11:50:00 | 1186.41 | 1195.90 | 0.00 | T1 1.5R @ 1186.41 |
| Stop hit — per-position SL triggered | 2024-10-15 12:15:00 | 1192.25 | 1195.49 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:30:00 | 1330.50 | 1320.15 | 0.00 | ORB-long ORB[1303.30,1319.70] vol=6.0x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 09:35:00 | 1338.49 | 1326.60 | 0.00 | T1 1.5R @ 1338.49 |
| Stop hit — per-position SL triggered | 2024-11-26 09:40:00 | 1330.50 | 1327.12 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:45:00 | 1363.35 | 1343.96 | 0.00 | ORB-long ORB[1326.15,1339.90] vol=4.4x ATR=8.22 |
| Stop hit — per-position SL triggered | 2024-11-27 09:50:00 | 1355.13 | 1348.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:05:00 | 1327.15 | 1335.27 | 0.00 | ORB-short ORB[1330.55,1347.00] vol=2.1x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:10:00 | 1318.40 | 1333.70 | 0.00 | T1 1.5R @ 1318.40 |
| Target hit | 2024-11-29 13:50:00 | 1323.75 | 1323.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 1336.60 | 1323.08 | 0.00 | ORB-long ORB[1307.00,1326.90] vol=3.6x ATR=5.68 |
| Stop hit — per-position SL triggered | 2024-12-02 09:35:00 | 1330.92 | 1331.16 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:05:00 | 1353.00 | 1350.58 | 0.00 | ORB-long ORB[1335.25,1352.15] vol=1.5x ATR=5.55 |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 1347.45 | 1350.47 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 1313.90 | 1322.04 | 0.00 | ORB-short ORB[1318.15,1329.00] vol=1.8x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:40:00 | 1306.95 | 1314.01 | 0.00 | T1 1.5R @ 1306.95 |
| Target hit | 2024-12-12 13:40:00 | 1297.50 | 1297.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2024-12-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 10:05:00 | 1307.00 | 1299.84 | 0.00 | ORB-long ORB[1292.00,1306.30] vol=2.0x ATR=6.18 |
| Stop hit — per-position SL triggered | 2024-12-13 10:10:00 | 1300.82 | 1299.97 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 1310.95 | 1329.82 | 0.00 | ORB-short ORB[1325.15,1337.70] vol=2.2x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-12-16 14:00:00 | 1315.47 | 1323.63 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:00:00 | 1285.50 | 1291.24 | 0.00 | ORB-short ORB[1288.00,1297.20] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-12-18 11:25:00 | 1289.67 | 1290.49 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 1264.00 | 1258.44 | 0.00 | ORB-long ORB[1253.00,1263.00] vol=1.5x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 1260.13 | 1258.52 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:35:00 | 1197.65 | 1203.03 | 0.00 | ORB-short ORB[1200.55,1209.85] vol=1.5x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 12:15:00 | 1192.69 | 1198.37 | 0.00 | T1 1.5R @ 1192.69 |
| Target hit | 2024-12-27 12:15:00 | 1199.90 | 1198.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2024-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:35:00 | 1184.00 | 1191.38 | 0.00 | ORB-short ORB[1186.75,1199.50] vol=1.8x ATR=4.53 |
| Stop hit — per-position SL triggered | 2024-12-30 09:40:00 | 1188.53 | 1191.14 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:45:00 | 1159.60 | 1166.93 | 0.00 | ORB-short ORB[1163.50,1179.90] vol=1.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 1164.38 | 1164.62 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 1194.30 | 1187.62 | 0.00 | ORB-long ORB[1181.00,1192.95] vol=2.3x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:40:00 | 1200.84 | 1189.95 | 0.00 | T1 1.5R @ 1200.84 |
| Stop hit — per-position SL triggered | 2025-01-02 09:45:00 | 1194.30 | 1190.94 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:25:00 | 902.10 | 881.23 | 0.00 | ORB-long ORB[860.00,872.75] vol=1.6x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:30:00 | 911.30 | 889.52 | 0.00 | T1 1.5R @ 911.30 |
| Stop hit — per-position SL triggered | 2025-01-31 10:45:00 | 902.10 | 896.04 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 868.00 | 875.09 | 0.00 | ORB-short ORB[870.80,882.00] vol=3.3x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:00:00 | 861.11 | 871.29 | 0.00 | T1 1.5R @ 861.11 |
| Target hit | 2025-02-10 15:20:00 | 856.75 | 861.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:35:00 | 750.50 | 757.85 | 0.00 | ORB-short ORB[755.00,765.70] vol=1.9x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-02-27 09:40:00 | 753.79 | 757.36 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 790.95 | 799.97 | 0.00 | ORB-short ORB[795.90,807.00] vol=2.1x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:35:00 | 784.16 | 797.05 | 0.00 | T1 1.5R @ 784.16 |
| Target hit | 2025-03-26 15:20:00 | 748.50 | 768.63 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 09:45:00 | 1188.00 | 2024-05-15 09:55:00 | 1176.58 | PARTIAL | 0.50 | 0.96% |
| SELL | retest1 | 2024-05-15 09:45:00 | 1188.00 | 2024-05-15 11:25:00 | 1182.35 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-16 09:40:00 | 1215.10 | 2024-05-16 09:45:00 | 1207.58 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-05-22 09:45:00 | 1148.00 | 2024-05-22 09:55:00 | 1156.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-05-23 11:05:00 | 1188.85 | 2024-05-23 11:10:00 | 1183.07 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-05-28 09:35:00 | 1157.75 | 2024-05-28 10:10:00 | 1147.20 | PARTIAL | 0.50 | 0.91% |
| SELL | retest1 | 2024-05-28 09:35:00 | 1157.75 | 2024-05-28 10:15:00 | 1157.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 10:50:00 | 1187.05 | 2024-05-29 11:00:00 | 1181.43 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-05-31 09:40:00 | 1132.50 | 2024-05-31 09:45:00 | 1139.71 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-06-13 09:50:00 | 1388.85 | 2024-06-13 09:55:00 | 1402.10 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2024-06-13 09:50:00 | 1388.85 | 2024-06-13 10:00:00 | 1388.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 09:30:00 | 1379.60 | 2024-06-18 09:40:00 | 1373.23 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-06-25 09:45:00 | 1411.50 | 2024-06-25 09:55:00 | 1419.01 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-06-26 09:30:00 | 1394.80 | 2024-06-26 10:10:00 | 1401.34 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-06-27 09:40:00 | 1412.65 | 2024-06-27 09:45:00 | 1422.21 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-06-27 09:40:00 | 1412.65 | 2024-06-27 10:10:00 | 1421.00 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-01 10:35:00 | 1425.70 | 2024-07-01 11:10:00 | 1433.76 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-01 10:35:00 | 1425.70 | 2024-07-01 12:10:00 | 1435.10 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-02 09:30:00 | 1415.90 | 2024-07-02 09:35:00 | 1422.86 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-07-03 09:40:00 | 1455.05 | 2024-07-03 14:25:00 | 1447.98 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-07-05 11:05:00 | 1454.95 | 2024-07-05 11:10:00 | 1447.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-07-08 09:55:00 | 1466.15 | 2024-07-08 10:00:00 | 1459.09 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-09 10:20:00 | 1408.00 | 2024-07-09 10:40:00 | 1413.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-10 10:00:00 | 1364.95 | 2024-07-10 10:10:00 | 1351.49 | PARTIAL | 0.50 | 0.99% |
| SELL | retest1 | 2024-07-10 10:00:00 | 1364.95 | 2024-07-10 11:05:00 | 1347.45 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2024-07-11 10:10:00 | 1362.80 | 2024-07-11 11:40:00 | 1370.77 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-07-15 09:35:00 | 1411.05 | 2024-07-15 09:45:00 | 1420.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1426.00 | 2024-07-18 09:35:00 | 1418.29 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1426.00 | 2024-07-18 09:40:00 | 1426.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-31 10:40:00 | 1278.00 | 2024-07-31 12:35:00 | 1271.51 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-31 10:40:00 | 1278.00 | 2024-07-31 15:20:00 | 1239.95 | TARGET_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2024-08-22 10:40:00 | 1314.15 | 2024-08-22 10:50:00 | 1308.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-30 10:15:00 | 1337.40 | 2024-08-30 10:25:00 | 1345.36 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-30 10:15:00 | 1337.40 | 2024-08-30 10:40:00 | 1337.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:55:00 | 1260.05 | 2024-09-17 11:15:00 | 1263.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-18 10:00:00 | 1244.95 | 2024-09-18 10:10:00 | 1237.70 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-18 10:00:00 | 1244.95 | 2024-09-18 11:00:00 | 1244.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-20 10:15:00 | 1222.75 | 2024-09-20 10:35:00 | 1229.02 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-09-26 09:30:00 | 1225.00 | 2024-09-26 09:55:00 | 1219.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-26 09:30:00 | 1225.00 | 2024-09-26 10:10:00 | 1225.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 10:15:00 | 1195.00 | 2024-10-01 10:30:00 | 1198.93 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-09 10:35:00 | 1202.75 | 2024-10-09 10:40:00 | 1197.21 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-10-10 09:30:00 | 1228.00 | 2024-10-10 09:35:00 | 1219.77 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2024-10-11 09:55:00 | 1198.95 | 2024-10-11 10:10:00 | 1206.25 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-11 09:55:00 | 1198.95 | 2024-10-11 10:45:00 | 1198.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:45:00 | 1203.70 | 2024-10-14 09:50:00 | 1198.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-15 10:50:00 | 1192.25 | 2024-10-15 11:50:00 | 1186.41 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-15 10:50:00 | 1192.25 | 2024-10-15 12:15:00 | 1192.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 09:30:00 | 1330.50 | 2024-11-26 09:35:00 | 1338.49 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-26 09:30:00 | 1330.50 | 2024-11-26 09:40:00 | 1330.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:45:00 | 1363.35 | 2024-11-27 09:50:00 | 1355.13 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-11-29 10:05:00 | 1327.15 | 2024-11-29 10:10:00 | 1318.40 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-29 10:05:00 | 1327.15 | 2024-11-29 13:50:00 | 1323.75 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1336.60 | 2024-12-02 09:35:00 | 1330.92 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-06 11:05:00 | 1353.00 | 2024-12-06 11:15:00 | 1347.45 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-12 09:30:00 | 1313.90 | 2024-12-12 09:40:00 | 1306.95 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-12 09:30:00 | 1313.90 | 2024-12-12 13:40:00 | 1297.50 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2024-12-13 10:05:00 | 1307.00 | 2024-12-13 10:10:00 | 1300.82 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1310.95 | 2024-12-16 14:00:00 | 1315.47 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-18 11:00:00 | 1285.50 | 2024-12-18 11:25:00 | 1289.67 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1264.00 | 2024-12-20 09:45:00 | 1260.13 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-27 09:35:00 | 1197.65 | 2024-12-27 12:15:00 | 1192.69 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-27 09:35:00 | 1197.65 | 2024-12-27 12:15:00 | 1199.90 | TARGET_HIT | 0.50 | -0.19% |
| SELL | retest1 | 2024-12-30 09:35:00 | 1184.00 | 2024-12-30 09:40:00 | 1188.53 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-31 09:45:00 | 1159.60 | 2024-12-31 10:15:00 | 1164.38 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1194.30 | 2025-01-02 09:40:00 | 1200.84 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1194.30 | 2025-01-02 09:45:00 | 1194.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 10:25:00 | 902.10 | 2025-01-31 10:30:00 | 911.30 | PARTIAL | 0.50 | 1.02% |
| BUY | retest1 | 2025-01-31 10:25:00 | 902.10 | 2025-01-31 10:45:00 | 902.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 09:30:00 | 868.00 | 2025-02-10 10:00:00 | 861.11 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2025-02-10 09:30:00 | 868.00 | 2025-02-10 15:20:00 | 856.75 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2025-02-27 09:35:00 | 750.50 | 2025-02-27 09:40:00 | 753.79 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-03-26 09:30:00 | 790.95 | 2025-03-26 09:35:00 | 784.16 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2025-03-26 09:30:00 | 790.95 | 2025-03-26 15:20:00 | 748.50 | TARGET_HIT | 0.50 | 5.37% |
