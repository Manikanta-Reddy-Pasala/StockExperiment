# Dr. Reddy's Laboratories Ltd. (DRREDDY)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-03-06 15:25:00 (15408 bars)
- **Last close:** 1138.40
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
| ENTRY1 | 69 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 59
- **Target hits / Stop hits / Partials:** 10 / 59 / 19
- **Avg / median % per leg:** 0.04% / -0.17%
- **Sum % (uncompounded):** 3.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 14 | 33.3% | 4 | 28 | 10 | 0.04% | 1.7% |
| BUY @ 2nd Alert (retest1) | 42 | 14 | 33.3% | 4 | 28 | 10 | 0.04% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 15 | 32.6% | 6 | 31 | 9 | 0.04% | 2.0% |
| SELL @ 2nd Alert (retest1) | 46 | 15 | 32.6% | 6 | 31 | 9 | 0.04% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 29 | 33.0% | 10 | 59 | 19 | 0.04% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 1190.00 | 1183.82 | 0.00 | ORB-long ORB[1172.00,1184.39] vol=2.0x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-05-15 10:05:00 | 1186.36 | 1184.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 1178.41 | 1181.10 | 0.00 | ORB-short ORB[1178.42,1185.60] vol=2.4x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:40:00 | 1173.78 | 1179.47 | 0.00 | T1 1.5R @ 1173.78 |
| Target hit | 2024-05-16 12:20:00 | 1169.38 | 1167.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1161.20 | 1167.24 | 0.00 | ORB-short ORB[1164.25,1174.90] vol=2.5x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 1164.05 | 1166.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 1188.40 | 1183.71 | 0.00 | ORB-long ORB[1174.43,1186.40] vol=1.6x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-05-28 09:35:00 | 1185.41 | 1185.35 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:10:00 | 1181.59 | 1184.43 | 0.00 | ORB-short ORB[1184.20,1195.00] vol=1.6x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:25:00 | 1178.56 | 1183.50 | 0.00 | T1 1.5R @ 1178.56 |
| Stop hit — per-position SL triggered | 2024-05-30 12:55:00 | 1181.59 | 1181.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-04 09:35:00 | 1145.00 | 1150.36 | 0.00 | ORB-short ORB[1147.34,1162.06] vol=2.1x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-06-04 09:40:00 | 1149.02 | 1149.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 1174.99 | 1164.20 | 0.00 | ORB-long ORB[1156.32,1172.76] vol=1.5x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 11:05:00 | 1181.21 | 1172.38 | 0.00 | T1 1.5R @ 1181.21 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 1174.99 | 1172.67 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:40:00 | 1190.00 | 1186.01 | 0.00 | ORB-long ORB[1175.25,1186.88] vol=2.5x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:50:00 | 1195.67 | 1190.28 | 0.00 | T1 1.5R @ 1195.67 |
| Target hit | 2024-06-07 15:20:00 | 1212.38 | 1204.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:20:00 | 1218.92 | 1209.90 | 0.00 | ORB-long ORB[1200.81,1212.26] vol=1.9x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:25:00 | 1224.01 | 1215.02 | 0.00 | T1 1.5R @ 1224.01 |
| Stop hit — per-position SL triggered | 2024-06-10 14:25:00 | 1218.92 | 1218.26 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:30:00 | 1201.94 | 1209.12 | 0.00 | ORB-short ORB[1209.02,1217.00] vol=3.2x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 1204.82 | 1208.27 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 1210.83 | 1216.76 | 0.00 | ORB-short ORB[1213.35,1223.98] vol=1.6x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-06-13 12:40:00 | 1214.44 | 1213.01 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:10:00 | 1185.98 | 1191.34 | 0.00 | ORB-short ORB[1193.00,1199.93] vol=2.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-06-19 10:35:00 | 1188.05 | 1189.04 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:05:00 | 1192.00 | 1186.25 | 0.00 | ORB-long ORB[1177.60,1189.76] vol=1.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-06-20 11:25:00 | 1188.90 | 1186.66 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 1259.31 | 1262.47 | 0.00 | ORB-short ORB[1260.30,1269.97] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 1261.64 | 1262.39 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:45:00 | 1284.21 | 1280.29 | 0.00 | ORB-long ORB[1273.03,1280.00] vol=2.1x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:00:00 | 1288.67 | 1282.44 | 0.00 | T1 1.5R @ 1288.67 |
| Target hit | 2024-07-03 13:00:00 | 1293.33 | 1294.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2024-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:55:00 | 1286.83 | 1296.81 | 0.00 | ORB-short ORB[1299.27,1310.02] vol=1.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-07-09 10:00:00 | 1290.13 | 1295.89 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:15:00 | 1360.00 | 1352.77 | 0.00 | ORB-long ORB[1341.33,1350.00] vol=4.7x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-07-15 10:45:00 | 1355.90 | 1354.50 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1338.01 | 1348.47 | 0.00 | ORB-short ORB[1347.00,1356.59] vol=2.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-07-16 11:30:00 | 1340.54 | 1347.53 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:45:00 | 1332.00 | 1327.29 | 0.00 | ORB-long ORB[1318.08,1329.99] vol=1.6x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-07-18 09:55:00 | 1328.94 | 1327.63 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:05:00 | 1348.74 | 1352.83 | 0.00 | ORB-short ORB[1349.90,1357.03] vol=6.0x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 1352.01 | 1352.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:50:00 | 1365.98 | 1357.88 | 0.00 | ORB-long ORB[1351.41,1361.30] vol=2.5x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-07-25 10:55:00 | 1362.94 | 1358.28 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 10:50:00 | 1366.40 | 1373.14 | 0.00 | ORB-short ORB[1370.64,1381.08] vol=4.1x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-07-26 11:20:00 | 1370.00 | 1372.57 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 11:15:00 | 1367.36 | 1371.07 | 0.00 | ORB-short ORB[1369.01,1378.20] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:05:00 | 1363.96 | 1370.13 | 0.00 | T1 1.5R @ 1363.96 |
| Target hit | 2024-07-30 15:20:00 | 1360.00 | 1364.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:10:00 | 1370.68 | 1368.02 | 0.00 | ORB-long ORB[1357.33,1369.60] vol=2.8x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:30:00 | 1376.27 | 1369.48 | 0.00 | T1 1.5R @ 1376.27 |
| Stop hit — per-position SL triggered | 2024-08-01 10:45:00 | 1370.68 | 1369.93 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:50:00 | 1376.63 | 1374.30 | 0.00 | ORB-long ORB[1363.01,1373.78] vol=2.8x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-08-06 12:20:00 | 1372.98 | 1375.30 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:10:00 | 1384.91 | 1381.07 | 0.00 | ORB-long ORB[1368.00,1382.79] vol=2.3x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-08-07 11:35:00 | 1382.01 | 1381.47 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:35:00 | 1402.20 | 1395.51 | 0.00 | ORB-long ORB[1383.20,1398.79] vol=2.4x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-08-08 10:05:00 | 1398.66 | 1400.13 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:00:00 | 1392.81 | 1397.74 | 0.00 | ORB-short ORB[1394.80,1406.00] vol=1.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-08-12 10:05:00 | 1395.67 | 1397.61 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1385.04 | 1381.10 | 0.00 | ORB-long ORB[1370.25,1380.40] vol=1.5x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:30:00 | 1389.35 | 1381.87 | 0.00 | T1 1.5R @ 1389.35 |
| Stop hit — per-position SL triggered | 2024-08-13 13:55:00 | 1385.04 | 1384.88 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:15:00 | 1381.80 | 1386.98 | 0.00 | ORB-short ORB[1382.32,1393.14] vol=1.6x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-08-20 11:45:00 | 1384.06 | 1386.56 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:50:00 | 1394.80 | 1400.02 | 0.00 | ORB-short ORB[1397.22,1412.47] vol=1.9x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:10:00 | 1390.45 | 1397.22 | 0.00 | T1 1.5R @ 1390.45 |
| Target hit | 2024-08-22 12:35:00 | 1392.97 | 1392.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2024-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:40:00 | 1329.13 | 1332.60 | 0.00 | ORB-short ORB[1331.80,1338.80] vol=4.6x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 09:45:00 | 1324.49 | 1330.90 | 0.00 | T1 1.5R @ 1324.49 |
| Stop hit — per-position SL triggered | 2024-09-10 09:55:00 | 1329.13 | 1328.17 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1345.20 | 1337.85 | 0.00 | ORB-long ORB[1328.00,1340.00] vol=1.9x ATR=3.15 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 1342.05 | 1340.98 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 1326.76 | 1328.61 | 0.00 | ORB-short ORB[1326.82,1333.07] vol=2.0x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-09-17 11:00:00 | 1328.40 | 1328.40 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:00:00 | 1323.01 | 1326.02 | 0.00 | ORB-short ORB[1324.42,1329.60] vol=1.5x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:20:00 | 1319.64 | 1324.27 | 0.00 | T1 1.5R @ 1319.64 |
| Target hit | 2024-09-18 15:20:00 | 1314.01 | 1314.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:15:00 | 1320.00 | 1329.55 | 0.00 | ORB-short ORB[1330.89,1337.98] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-09-24 10:40:00 | 1322.86 | 1325.30 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:50:00 | 1320.58 | 1323.14 | 0.00 | ORB-short ORB[1324.00,1332.59] vol=9.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-09-25 11:30:00 | 1322.46 | 1321.50 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1333.60 | 1336.88 | 0.00 | ORB-short ORB[1336.88,1342.96] vol=3.3x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-09-26 09:35:00 | 1336.28 | 1336.84 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:45:00 | 1336.40 | 1342.87 | 0.00 | ORB-short ORB[1344.00,1357.08] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 1338.60 | 1341.92 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 1325.80 | 1323.63 | 0.00 | ORB-long ORB[1315.00,1325.59] vol=4.2x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-10-08 12:00:00 | 1322.83 | 1323.78 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 1323.40 | 1325.33 | 0.00 | ORB-short ORB[1327.27,1339.60] vol=2.2x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:30:00 | 1319.91 | 1324.61 | 0.00 | T1 1.5R @ 1319.91 |
| Stop hit — per-position SL triggered | 2024-10-10 12:35:00 | 1323.40 | 1322.73 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 1329.66 | 1326.70 | 0.00 | ORB-long ORB[1320.00,1328.48] vol=1.8x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-10-14 09:35:00 | 1327.26 | 1326.48 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 1339.00 | 1333.43 | 0.00 | ORB-long ORB[1325.81,1332.75] vol=2.4x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-10-16 11:25:00 | 1336.82 | 1333.60 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:50:00 | 1352.40 | 1346.89 | 0.00 | ORB-long ORB[1337.62,1348.96] vol=2.2x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-10-17 10:00:00 | 1348.82 | 1347.81 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:10:00 | 1353.00 | 1344.25 | 0.00 | ORB-long ORB[1332.48,1343.80] vol=2.7x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-10-18 10:20:00 | 1349.01 | 1345.00 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:50:00 | 1345.87 | 1341.00 | 0.00 | ORB-long ORB[1337.24,1345.40] vol=1.5x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-10-22 10:10:00 | 1342.33 | 1344.53 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 1297.62 | 1300.01 | 0.00 | ORB-short ORB[1302.41,1316.40] vol=3.3x ATR=3.43 |
| Stop hit — per-position SL triggered | 2024-10-25 11:00:00 | 1301.05 | 1300.09 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 10:45:00 | 1238.80 | 1240.03 | 0.00 | ORB-short ORB[1238.90,1253.00] vol=2.4x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 11:05:00 | 1233.70 | 1238.87 | 0.00 | T1 1.5R @ 1233.70 |
| Target hit | 2024-11-14 15:20:00 | 1227.35 | 1231.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 1201.75 | 1204.95 | 0.00 | ORB-short ORB[1203.05,1215.60] vol=2.3x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-11-27 12:45:00 | 1204.39 | 1202.31 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 1192.90 | 1200.25 | 0.00 | ORB-short ORB[1198.30,1209.90] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-11-28 12:45:00 | 1195.91 | 1194.40 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:40:00 | 1217.50 | 1215.81 | 0.00 | ORB-long ORB[1202.30,1214.25] vol=2.1x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-12-02 10:50:00 | 1214.70 | 1215.80 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:05:00 | 1215.00 | 1216.45 | 0.00 | ORB-short ORB[1218.10,1229.10] vol=1.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-12-04 11:35:00 | 1217.18 | 1216.38 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:05:00 | 1239.95 | 1244.84 | 0.00 | ORB-short ORB[1246.55,1253.60] vol=4.2x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-12-10 11:30:00 | 1242.56 | 1244.40 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 1226.60 | 1232.99 | 0.00 | ORB-short ORB[1232.55,1245.30] vol=4.1x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-12-12 11:10:00 | 1229.15 | 1232.57 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:50:00 | 1230.35 | 1235.01 | 0.00 | ORB-short ORB[1231.50,1241.60] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-12-13 11:00:00 | 1233.61 | 1234.75 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 1348.40 | 1335.97 | 0.00 | ORB-long ORB[1317.40,1334.75] vol=1.8x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:50:00 | 1355.41 | 1342.24 | 0.00 | T1 1.5R @ 1355.41 |
| Target hit | 2024-12-20 11:05:00 | 1354.00 | 1354.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2024-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 09:40:00 | 1338.70 | 1351.83 | 0.00 | ORB-short ORB[1347.30,1367.00] vol=2.1x ATR=6.56 |
| Stop hit — per-position SL triggered | 2024-12-23 10:05:00 | 1345.26 | 1348.82 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:40:00 | 1350.45 | 1345.95 | 0.00 | ORB-long ORB[1336.20,1348.45] vol=1.6x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:30:00 | 1356.27 | 1347.49 | 0.00 | T1 1.5R @ 1356.27 |
| Target hit | 2024-12-24 14:25:00 | 1353.30 | 1353.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2024-12-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:55:00 | 1357.85 | 1352.34 | 0.00 | ORB-long ORB[1344.75,1356.00] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-12-26 10:00:00 | 1353.83 | 1352.60 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1377.65 | 1364.25 | 0.00 | ORB-long ORB[1345.50,1365.00] vol=5.5x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-12-27 10:05:00 | 1372.15 | 1370.69 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:50:00 | 1386.95 | 1380.11 | 0.00 | ORB-long ORB[1366.55,1383.95] vol=7.4x ATR=4.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 12:15:00 | 1393.73 | 1382.75 | 0.00 | T1 1.5R @ 1393.73 |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 1386.95 | 1386.79 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:00:00 | 1342.60 | 1348.71 | 0.00 | ORB-short ORB[1350.55,1361.30] vol=1.6x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1346.00 | 1346.64 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:20:00 | 1326.80 | 1333.28 | 0.00 | ORB-short ORB[1333.25,1348.85] vol=2.4x ATR=3.85 |
| Stop hit — per-position SL triggered | 2025-01-14 10:50:00 | 1330.65 | 1332.44 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:05:00 | 1307.30 | 1296.29 | 0.00 | ORB-long ORB[1288.00,1303.45] vol=1.8x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-01-22 10:10:00 | 1302.49 | 1297.27 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1200.80 | 1196.13 | 0.00 | ORB-long ORB[1184.00,1200.45] vol=1.6x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-01-30 10:00:00 | 1197.26 | 1198.23 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-02-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-03 09:35:00 | 1215.90 | 1208.50 | 0.00 | ORB-long ORB[1196.80,1214.75] vol=2.0x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1211.44 | 1210.60 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 1248.25 | 1242.82 | 0.00 | ORB-long ORB[1230.20,1246.50] vol=2.5x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 09:35:00 | 1253.63 | 1244.30 | 0.00 | T1 1.5R @ 1253.63 |
| Stop hit — per-position SL triggered | 2025-02-06 09:45:00 | 1248.25 | 1245.14 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:30:00 | 1185.75 | 1195.54 | 0.00 | ORB-short ORB[1192.20,1204.95] vol=1.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-02-18 10:40:00 | 1189.23 | 1194.05 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:00:00 | 1154.40 | 1159.37 | 0.00 | ORB-short ORB[1155.85,1169.55] vol=2.3x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1149.39 | 1157.07 | 0.00 | T1 1.5R @ 1149.39 |
| Target hit | 2025-02-25 15:20:00 | 1123.85 | 1135.45 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:55:00 | 1190.00 | 2024-05-15 10:05:00 | 1186.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1178.41 | 2024-05-16 09:40:00 | 1173.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1178.41 | 2024-05-16 12:20:00 | 1169.38 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2024-05-27 09:30:00 | 1161.20 | 2024-05-27 09:40:00 | 1164.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-28 09:30:00 | 1188.40 | 2024-05-28 09:35:00 | 1185.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-30 11:10:00 | 1181.59 | 2024-05-30 11:25:00 | 1178.56 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-05-30 11:10:00 | 1181.59 | 2024-05-30 12:55:00 | 1181.59 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-04 09:35:00 | 1145.00 | 2024-06-04 09:40:00 | 1149.02 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-06 10:05:00 | 1174.99 | 2024-06-06 11:05:00 | 1181.21 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-06 10:05:00 | 1174.99 | 2024-06-06 11:30:00 | 1174.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:40:00 | 1190.00 | 2024-06-07 10:50:00 | 1195.67 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-07 09:40:00 | 1190.00 | 2024-06-07 15:20:00 | 1212.38 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2024-06-10 10:20:00 | 1218.92 | 2024-06-10 11:25:00 | 1224.01 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-10 10:20:00 | 1218.92 | 2024-06-10 14:25:00 | 1218.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:30:00 | 1201.94 | 2024-06-12 10:45:00 | 1204.82 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-13 09:35:00 | 1210.83 | 2024-06-13 12:40:00 | 1214.44 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-19 10:10:00 | 1185.98 | 2024-06-19 10:35:00 | 1188.05 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-06-20 11:05:00 | 1192.00 | 2024-06-20 11:25:00 | 1188.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-02 09:35:00 | 1259.31 | 2024-07-02 09:40:00 | 1261.64 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-03 09:45:00 | 1284.21 | 2024-07-03 10:00:00 | 1288.67 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-03 09:45:00 | 1284.21 | 2024-07-03 13:00:00 | 1293.33 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-07-09 09:55:00 | 1286.83 | 2024-07-09 10:00:00 | 1290.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-15 10:15:00 | 1360.00 | 2024-07-15 10:45:00 | 1355.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-16 11:15:00 | 1338.01 | 2024-07-16 11:30:00 | 1340.54 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-18 09:45:00 | 1332.00 | 2024-07-18 09:55:00 | 1328.94 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-23 11:05:00 | 1348.74 | 2024-07-23 11:20:00 | 1352.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-25 10:50:00 | 1365.98 | 2024-07-25 10:55:00 | 1362.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-26 10:50:00 | 1366.40 | 2024-07-26 11:20:00 | 1370.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-30 11:15:00 | 1367.36 | 2024-07-30 12:05:00 | 1363.96 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-07-30 11:15:00 | 1367.36 | 2024-07-30 15:20:00 | 1360.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-01 10:10:00 | 1370.68 | 2024-08-01 10:30:00 | 1376.27 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-01 10:10:00 | 1370.68 | 2024-08-01 10:45:00 | 1370.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 10:50:00 | 1376.63 | 2024-08-06 12:20:00 | 1372.98 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-07 11:10:00 | 1384.91 | 2024-08-07 11:35:00 | 1382.01 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-08 09:35:00 | 1402.20 | 2024-08-08 10:05:00 | 1398.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-12 10:00:00 | 1392.81 | 2024-08-12 10:05:00 | 1395.67 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-13 11:00:00 | 1385.04 | 2024-08-13 11:30:00 | 1389.35 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-13 11:00:00 | 1385.04 | 2024-08-13 13:55:00 | 1385.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 11:15:00 | 1381.80 | 2024-08-20 11:45:00 | 1384.06 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-08-22 09:50:00 | 1394.80 | 2024-08-22 10:10:00 | 1390.45 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-22 09:50:00 | 1394.80 | 2024-08-22 12:35:00 | 1392.97 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-09-10 09:40:00 | 1329.13 | 2024-09-10 09:45:00 | 1324.49 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-10 09:40:00 | 1329.13 | 2024-09-10 09:55:00 | 1329.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1345.20 | 2024-09-11 09:55:00 | 1342.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-17 10:55:00 | 1326.76 | 2024-09-17 11:00:00 | 1328.40 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2024-09-18 10:00:00 | 1323.01 | 2024-09-18 10:20:00 | 1319.64 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-09-18 10:00:00 | 1323.01 | 2024-09-18 15:20:00 | 1314.01 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-09-24 10:15:00 | 1320.00 | 2024-09-24 10:40:00 | 1322.86 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-25 10:50:00 | 1320.58 | 2024-09-25 11:30:00 | 1322.46 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-09-26 09:30:00 | 1333.60 | 2024-09-26 09:35:00 | 1336.28 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-01 10:45:00 | 1336.40 | 2024-10-01 10:55:00 | 1338.60 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-10-08 11:05:00 | 1325.80 | 2024-10-08 12:00:00 | 1322.83 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-10 11:10:00 | 1323.40 | 2024-10-10 11:30:00 | 1319.91 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-10-10 11:10:00 | 1323.40 | 2024-10-10 12:35:00 | 1323.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:30:00 | 1329.66 | 2024-10-14 09:35:00 | 1327.26 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-10-16 11:15:00 | 1339.00 | 2024-10-16 11:25:00 | 1336.82 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-10-17 09:50:00 | 1352.40 | 2024-10-17 10:00:00 | 1348.82 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-18 10:10:00 | 1353.00 | 2024-10-18 10:20:00 | 1349.01 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-22 09:50:00 | 1345.87 | 2024-10-22 10:10:00 | 1342.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-25 10:55:00 | 1297.62 | 2024-10-25 11:00:00 | 1301.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-14 10:45:00 | 1238.80 | 2024-11-14 11:05:00 | 1233.70 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-11-14 10:45:00 | 1238.80 | 2024-11-14 15:20:00 | 1227.35 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2024-11-27 10:50:00 | 1201.75 | 2024-11-27 12:45:00 | 1204.39 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1192.90 | 2024-11-28 12:45:00 | 1195.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-02 10:40:00 | 1217.50 | 2024-12-02 10:50:00 | 1214.70 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-04 11:05:00 | 1215.00 | 2024-12-04 11:35:00 | 1217.18 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-10 11:05:00 | 1239.95 | 2024-12-10 11:30:00 | 1242.56 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-12 11:00:00 | 1226.60 | 2024-12-12 11:10:00 | 1229.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-13 10:50:00 | 1230.35 | 2024-12-13 11:00:00 | 1233.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1348.40 | 2024-12-20 09:50:00 | 1355.41 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1348.40 | 2024-12-20 11:05:00 | 1354.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-23 09:40:00 | 1338.70 | 2024-12-23 10:05:00 | 1345.26 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-24 10:40:00 | 1350.45 | 2024-12-24 11:30:00 | 1356.27 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-24 10:40:00 | 1350.45 | 2024-12-24 14:25:00 | 1353.30 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-12-26 09:55:00 | 1357.85 | 2024-12-26 10:00:00 | 1353.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-27 09:30:00 | 1377.65 | 2024-12-27 10:05:00 | 1372.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-31 10:50:00 | 1386.95 | 2024-12-31 12:15:00 | 1393.73 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-31 10:50:00 | 1386.95 | 2024-12-31 15:15:00 | 1386.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:00:00 | 1342.60 | 2025-01-06 11:30:00 | 1346.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-14 10:20:00 | 1326.80 | 2025-01-14 10:50:00 | 1330.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-22 10:05:00 | 1307.30 | 2025-01-22 10:10:00 | 1302.49 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1200.80 | 2025-01-30 10:00:00 | 1197.26 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-03 09:35:00 | 1215.90 | 2025-02-03 10:15:00 | 1211.44 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-06 09:30:00 | 1248.25 | 2025-02-06 09:35:00 | 1253.63 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-02-06 09:30:00 | 1248.25 | 2025-02-06 09:45:00 | 1248.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 10:30:00 | 1185.75 | 2025-02-18 10:40:00 | 1189.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-25 10:00:00 | 1154.40 | 2025-02-25 10:15:00 | 1149.39 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-02-25 10:00:00 | 1154.40 | 2025-02-25 15:20:00 | 1123.85 | TARGET_HIT | 0.50 | 2.65% |
