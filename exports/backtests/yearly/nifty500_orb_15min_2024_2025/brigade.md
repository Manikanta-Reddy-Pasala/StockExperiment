# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 760.25
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
| ENTRY1 | 36 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 30
- **Target hits / Stop hits / Partials:** 6 / 30 / 13
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 8.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 9 | 32.1% | 2 | 19 | 7 | 0.04% | 1.1% |
| BUY @ 2nd Alert (retest1) | 28 | 9 | 32.1% | 2 | 19 | 7 | 0.04% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 10 | 47.6% | 4 | 11 | 6 | 0.33% | 7.0% |
| SELL @ 2nd Alert (retest1) | 21 | 10 | 47.6% | 4 | 11 | 6 | 0.33% | 7.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 49 | 19 | 38.8% | 6 | 30 | 13 | 0.17% | 8.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:55:00 | 1382.45 | 1398.03 | 0.00 | ORB-short ORB[1400.00,1411.85] vol=1.7x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:25:00 | 1374.67 | 1394.06 | 0.00 | T1 1.5R @ 1374.67 |
| Stop hit — per-position SL triggered | 2024-06-12 12:10:00 | 1382.45 | 1381.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:20:00 | 1371.15 | 1358.64 | 0.00 | ORB-long ORB[1345.90,1365.65] vol=1.9x ATR=6.08 |
| Stop hit — per-position SL triggered | 2024-06-13 10:35:00 | 1365.07 | 1361.48 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:15:00 | 1366.40 | 1373.89 | 0.00 | ORB-short ORB[1377.00,1396.00] vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-07-03 11:35:00 | 1369.90 | 1373.41 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 1367.35 | 1373.57 | 0.00 | ORB-short ORB[1367.65,1384.95] vol=1.6x ATR=5.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:50:00 | 1358.38 | 1370.11 | 0.00 | T1 1.5R @ 1358.38 |
| Target hit | 2024-07-05 15:20:00 | 1346.60 | 1352.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 1335.00 | 1341.94 | 0.00 | ORB-short ORB[1335.05,1355.10] vol=2.1x ATR=6.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:00:00 | 1325.05 | 1338.40 | 0.00 | T1 1.5R @ 1325.05 |
| Target hit | 2024-07-11 15:20:00 | 1300.00 | 1307.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:45:00 | 1242.05 | 1236.14 | 0.00 | ORB-long ORB[1217.30,1234.85] vol=1.7x ATR=4.97 |
| Stop hit — per-position SL triggered | 2024-07-26 12:30:00 | 1237.08 | 1238.15 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-08-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 10:25:00 | 1224.45 | 1225.96 | 0.00 | ORB-short ORB[1230.05,1245.05] vol=2.2x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-08-02 10:45:00 | 1229.78 | 1226.10 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-08-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:20:00 | 1154.95 | 1154.03 | 0.00 | ORB-long ORB[1140.80,1152.65] vol=18.2x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:25:00 | 1160.03 | 1154.52 | 0.00 | T1 1.5R @ 1160.03 |
| Target hit | 2024-08-20 15:20:00 | 1169.45 | 1158.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 1157.70 | 1151.73 | 0.00 | ORB-long ORB[1140.00,1155.10] vol=2.2x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:35:00 | 1164.95 | 1155.46 | 0.00 | T1 1.5R @ 1164.95 |
| Stop hit — per-position SL triggered | 2024-08-23 10:10:00 | 1157.70 | 1159.60 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 1167.30 | 1161.15 | 0.00 | ORB-long ORB[1147.05,1164.40] vol=1.8x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 14:25:00 | 1177.42 | 1168.23 | 0.00 | T1 1.5R @ 1177.42 |
| Target hit | 2024-08-27 15:20:00 | 1187.05 | 1175.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-09-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:10:00 | 1345.00 | 1341.92 | 0.00 | ORB-long ORB[1336.05,1344.35] vol=2.5x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-09-16 13:15:00 | 1342.47 | 1343.61 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:50:00 | 1429.55 | 1427.73 | 0.00 | ORB-long ORB[1410.30,1428.55] vol=12.8x ATR=8.01 |
| Stop hit — per-position SL triggered | 2024-10-01 12:10:00 | 1421.54 | 1427.55 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-10-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:35:00 | 1414.80 | 1388.01 | 0.00 | ORB-long ORB[1367.00,1379.30] vol=3.9x ATR=7.14 |
| Stop hit — per-position SL triggered | 2024-10-04 09:40:00 | 1407.66 | 1390.02 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-10-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:55:00 | 1267.15 | 1273.49 | 0.00 | ORB-short ORB[1272.00,1287.65] vol=2.0x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-10-15 11:00:00 | 1271.88 | 1271.65 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 1303.00 | 1315.26 | 0.00 | ORB-short ORB[1315.60,1332.00] vol=3.0x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 1293.61 | 1311.39 | 0.00 | T1 1.5R @ 1293.61 |
| Target hit | 2024-10-17 15:20:00 | 1286.10 | 1291.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:10:00 | 1197.95 | 1203.68 | 0.00 | ORB-short ORB[1200.60,1217.85] vol=2.5x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-10-24 11:25:00 | 1202.36 | 1203.13 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-11-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 09:50:00 | 1199.60 | 1208.82 | 0.00 | ORB-short ORB[1204.90,1221.50] vol=1.5x ATR=6.01 |
| Stop hit — per-position SL triggered | 2024-11-25 10:25:00 | 1205.61 | 1206.84 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:10:00 | 1315.45 | 1297.28 | 0.00 | ORB-long ORB[1292.10,1308.55] vol=4.0x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:35:00 | 1324.49 | 1303.55 | 0.00 | T1 1.5R @ 1324.49 |
| Stop hit — per-position SL triggered | 2024-12-06 12:25:00 | 1315.45 | 1311.09 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 1296.70 | 1289.42 | 0.00 | ORB-long ORB[1276.30,1288.55] vol=1.8x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:20:00 | 1302.90 | 1292.75 | 0.00 | T1 1.5R @ 1302.90 |
| Stop hit — per-position SL triggered | 2024-12-10 11:00:00 | 1296.70 | 1298.33 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:40:00 | 1252.85 | 1246.63 | 0.00 | ORB-long ORB[1238.65,1251.75] vol=3.5x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-12-27 10:50:00 | 1248.54 | 1246.84 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:10:00 | 1235.65 | 1230.27 | 0.00 | ORB-long ORB[1220.05,1231.10] vol=1.8x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 11:55:00 | 1241.71 | 1231.94 | 0.00 | T1 1.5R @ 1241.71 |
| Stop hit — per-position SL triggered | 2024-12-31 12:10:00 | 1235.65 | 1233.24 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:45:00 | 1254.90 | 1240.30 | 0.00 | ORB-long ORB[1239.05,1248.30] vol=2.4x ATR=4.63 |
| Stop hit — per-position SL triggered | 2025-01-01 10:50:00 | 1250.27 | 1241.60 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:00:00 | 1253.00 | 1247.56 | 0.00 | ORB-long ORB[1235.05,1249.00] vol=2.0x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:30:00 | 1258.68 | 1252.27 | 0.00 | T1 1.5R @ 1258.68 |
| Stop hit — per-position SL triggered | 2025-01-02 10:35:00 | 1253.00 | 1252.49 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 1295.10 | 1286.59 | 0.00 | ORB-long ORB[1269.40,1288.00] vol=2.3x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-01-03 10:50:00 | 1288.62 | 1290.17 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1237.95 | 1239.36 | 0.00 | ORB-short ORB[1241.55,1256.95] vol=1.6x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:55:00 | 1231.44 | 1237.66 | 0.00 | T1 1.5R @ 1231.44 |
| Target hit | 2025-01-08 15:20:00 | 1219.95 | 1231.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-01-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:55:00 | 1159.95 | 1168.81 | 0.00 | ORB-short ORB[1173.30,1190.70] vol=1.5x ATR=5.94 |
| Stop hit — per-position SL triggered | 2025-01-10 11:05:00 | 1165.89 | 1168.65 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:30:00 | 1142.50 | 1128.49 | 0.00 | ORB-long ORB[1113.90,1130.00] vol=2.4x ATR=7.00 |
| Stop hit — per-position SL triggered | 2025-01-16 09:35:00 | 1135.50 | 1129.77 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:50:00 | 1128.35 | 1134.89 | 0.00 | ORB-short ORB[1131.00,1141.45] vol=2.9x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 1120.67 | 1130.81 | 0.00 | T1 1.5R @ 1120.67 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 1128.35 | 1125.53 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:10:00 | 1180.00 | 1167.20 | 0.00 | ORB-long ORB[1154.15,1170.95] vol=6.1x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-02-01 11:40:00 | 1174.53 | 1169.27 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:50:00 | 950.95 | 946.20 | 0.00 | ORB-long ORB[936.10,945.55] vol=4.4x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-03-18 09:55:00 | 947.62 | 948.72 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 960.95 | 958.25 | 0.00 | ORB-long ORB[950.05,960.60] vol=2.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-03-20 09:50:00 | 958.11 | 959.03 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 978.70 | 968.40 | 0.00 | ORB-long ORB[958.00,970.20] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 974.46 | 972.24 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 983.85 | 978.81 | 0.00 | ORB-long ORB[969.10,983.05] vol=1.8x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-04-16 10:20:00 | 979.46 | 979.34 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:00:00 | 1024.80 | 1029.65 | 0.00 | ORB-short ORB[1025.25,1039.90] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-04-29 11:05:00 | 1028.00 | 1029.61 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:55:00 | 1009.50 | 1010.39 | 0.00 | ORB-short ORB[1014.30,1028.60] vol=12.6x ATR=4.36 |
| Stop hit — per-position SL triggered | 2025-05-06 10:15:00 | 1013.86 | 1010.41 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-05-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 09:45:00 | 986.30 | 994.15 | 0.00 | ORB-short ORB[988.90,1000.20] vol=1.9x ATR=5.55 |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 991.85 | 989.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-12 10:55:00 | 1382.45 | 2024-06-12 11:25:00 | 1374.67 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-06-12 10:55:00 | 1382.45 | 2024-06-12 12:10:00 | 1382.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 10:20:00 | 1371.15 | 2024-06-13 10:35:00 | 1365.07 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-03 11:15:00 | 1366.40 | 2024-07-03 11:35:00 | 1369.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-05 09:30:00 | 1367.35 | 2024-07-05 09:50:00 | 1358.38 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-05 09:30:00 | 1367.35 | 2024-07-05 15:20:00 | 1346.60 | TARGET_HIT | 0.50 | 1.52% |
| SELL | retest1 | 2024-07-11 09:30:00 | 1335.00 | 2024-07-11 10:00:00 | 1325.05 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-07-11 09:30:00 | 1335.00 | 2024-07-11 15:20:00 | 1300.00 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2024-07-26 10:45:00 | 1242.05 | 2024-07-26 12:30:00 | 1237.08 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-02 10:25:00 | 1224.45 | 2024-08-02 10:45:00 | 1229.78 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-20 10:20:00 | 1154.95 | 2024-08-20 10:25:00 | 1160.03 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-20 10:20:00 | 1154.95 | 2024-08-20 15:20:00 | 1169.45 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2024-08-23 09:30:00 | 1157.70 | 2024-08-23 09:35:00 | 1164.95 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-23 09:30:00 | 1157.70 | 2024-08-23 10:10:00 | 1157.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 09:30:00 | 1167.30 | 2024-08-27 14:25:00 | 1177.42 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-08-27 09:30:00 | 1167.30 | 2024-08-27 15:20:00 | 1187.05 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2024-09-16 10:10:00 | 1345.00 | 2024-09-16 13:15:00 | 1342.47 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-10-01 10:50:00 | 1429.55 | 2024-10-01 12:10:00 | 1421.54 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-10-04 09:35:00 | 1414.80 | 2024-10-04 09:40:00 | 1407.66 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-10-15 09:55:00 | 1267.15 | 2024-10-15 11:00:00 | 1271.88 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1303.00 | 2024-10-17 11:25:00 | 1293.61 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1303.00 | 2024-10-17 15:20:00 | 1286.10 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2024-10-24 11:10:00 | 1197.95 | 2024-10-24 11:25:00 | 1202.36 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-25 09:50:00 | 1199.60 | 2024-11-25 10:25:00 | 1205.61 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-06 11:10:00 | 1315.45 | 2024-12-06 11:35:00 | 1324.49 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-12-06 11:10:00 | 1315.45 | 2024-12-06 12:25:00 | 1315.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:55:00 | 1296.70 | 2024-12-10 10:20:00 | 1302.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-10 09:55:00 | 1296.70 | 2024-12-10 11:00:00 | 1296.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 10:40:00 | 1252.85 | 2024-12-27 10:50:00 | 1248.54 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-31 11:10:00 | 1235.65 | 2024-12-31 11:55:00 | 1241.71 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-31 11:10:00 | 1235.65 | 2024-12-31 12:10:00 | 1235.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:45:00 | 1254.90 | 2025-01-01 10:50:00 | 1250.27 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-02 10:00:00 | 1253.00 | 2025-01-02 10:30:00 | 1258.68 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-02 10:00:00 | 1253.00 | 2025-01-02 10:35:00 | 1253.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 09:55:00 | 1295.10 | 2025-01-03 10:50:00 | 1288.62 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1237.95 | 2025-01-08 11:55:00 | 1231.44 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1237.95 | 2025-01-08 15:20:00 | 1219.95 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2025-01-10 10:55:00 | 1159.95 | 2025-01-10 11:05:00 | 1165.89 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-16 09:30:00 | 1142.50 | 2025-01-16 09:35:00 | 1135.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-01-21 09:50:00 | 1128.35 | 2025-01-21 10:20:00 | 1120.67 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-21 09:50:00 | 1128.35 | 2025-01-21 11:45:00 | 1128.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 11:10:00 | 1180.00 | 2025-02-01 11:40:00 | 1174.53 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-18 09:50:00 | 950.95 | 2025-03-18 09:55:00 | 947.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-20 09:35:00 | 960.95 | 2025-03-20 09:50:00 | 958.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-02 09:30:00 | 978.70 | 2025-04-02 09:35:00 | 974.46 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-04-16 09:55:00 | 983.85 | 2025-04-16 10:20:00 | 979.46 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-29 11:00:00 | 1024.80 | 2025-04-29 11:05:00 | 1028.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-06 09:55:00 | 1009.50 | 2025-05-06 10:15:00 | 1013.86 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-05-09 09:45:00 | 986.30 | 2025-05-09 11:15:00 | 991.85 | STOP_HIT | 1.00 | -0.56% |
