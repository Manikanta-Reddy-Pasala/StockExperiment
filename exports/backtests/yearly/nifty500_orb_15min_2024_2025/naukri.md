# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 978.40
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
| PARTIAL | 27 |
| TARGET_HIT | 15 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 54
- **Target hits / Stop hits / Partials:** 15 / 54 / 27
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 12.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 24 | 46.2% | 9 | 28 | 15 | 0.10% | 5.3% |
| BUY @ 2nd Alert (retest1) | 52 | 24 | 46.2% | 9 | 28 | 15 | 0.10% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 18 | 40.9% | 6 | 26 | 12 | 0.16% | 6.9% |
| SELL @ 2nd Alert (retest1) | 44 | 18 | 40.9% | 6 | 26 | 12 | 0.16% | 6.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 42 | 43.8% | 15 | 54 | 27 | 0.13% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:05:00 | 1193.92 | 1194.13 | 0.00 | ORB-short ORB[1194.00,1205.46] vol=3.5x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-05-15 11:15:00 | 1197.66 | 1194.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:55:00 | 1186.40 | 1187.92 | 0.00 | ORB-short ORB[1190.62,1200.00] vol=1.7x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-05-16 11:45:00 | 1190.03 | 1187.50 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:15:00 | 1260.06 | 1251.09 | 0.00 | ORB-long ORB[1240.71,1257.53] vol=2.3x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:25:00 | 1268.92 | 1254.18 | 0.00 | T1 1.5R @ 1268.92 |
| Target hit | 2024-05-21 15:20:00 | 1280.35 | 1278.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:30:00 | 1278.14 | 1282.39 | 0.00 | ORB-short ORB[1278.99,1293.00] vol=6.1x ATR=4.91 |
| Stop hit — per-position SL triggered | 2024-05-22 10:35:00 | 1283.05 | 1282.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1166.00 | 1171.13 | 0.00 | ORB-short ORB[1168.20,1183.10] vol=1.5x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-05-30 09:45:00 | 1169.74 | 1170.55 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:55:00 | 1260.03 | 1251.48 | 0.00 | ORB-long ORB[1244.28,1254.38] vol=7.5x ATR=3.78 |
| Stop hit — per-position SL triggered | 2024-06-12 11:00:00 | 1256.25 | 1251.83 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:55:00 | 1266.23 | 1260.41 | 0.00 | ORB-long ORB[1252.01,1262.75] vol=1.7x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:05:00 | 1271.73 | 1262.06 | 0.00 | T1 1.5R @ 1271.73 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 1266.23 | 1264.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 1252.40 | 1246.78 | 0.00 | ORB-long ORB[1238.84,1251.00] vol=1.6x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:40:00 | 1259.54 | 1250.16 | 0.00 | T1 1.5R @ 1259.54 |
| Target hit | 2024-06-20 12:30:00 | 1256.02 | 1256.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 1297.75 | 1287.96 | 0.00 | ORB-long ORB[1266.59,1285.89] vol=4.9x ATR=5.72 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 1292.03 | 1288.82 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:40:00 | 1329.20 | 1312.37 | 0.00 | ORB-long ORB[1306.40,1319.76] vol=4.0x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:45:00 | 1336.70 | 1317.10 | 0.00 | T1 1.5R @ 1336.70 |
| Stop hit — per-position SL triggered | 2024-06-25 11:00:00 | 1329.20 | 1321.56 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 1352.57 | 1344.45 | 0.00 | ORB-long ORB[1335.02,1349.59] vol=1.5x ATR=4.66 |
| Stop hit — per-position SL triggered | 2024-06-27 10:25:00 | 1347.91 | 1346.33 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 1383.64 | 1370.92 | 0.00 | ORB-long ORB[1350.02,1367.39] vol=1.5x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-07-01 10:40:00 | 1378.92 | 1371.99 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:45:00 | 1353.20 | 1362.81 | 0.00 | ORB-short ORB[1356.00,1368.89] vol=1.5x ATR=3.75 |
| Stop hit — per-position SL triggered | 2024-07-04 10:55:00 | 1356.95 | 1362.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 1357.56 | 1350.09 | 0.00 | ORB-long ORB[1344.29,1355.00] vol=1.9x ATR=3.43 |
| Stop hit — per-position SL triggered | 2024-07-05 10:55:00 | 1354.13 | 1350.47 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:55:00 | 1391.42 | 1382.72 | 0.00 | ORB-long ORB[1366.02,1382.97] vol=1.8x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:00:00 | 1397.39 | 1385.81 | 0.00 | T1 1.5R @ 1397.39 |
| Target hit | 2024-07-09 10:50:00 | 1396.40 | 1398.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2024-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:40:00 | 1363.99 | 1375.11 | 0.00 | ORB-short ORB[1380.00,1394.50] vol=3.6x ATR=6.52 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 1370.51 | 1374.56 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:45:00 | 1330.61 | 1351.08 | 0.00 | ORB-short ORB[1354.01,1371.18] vol=1.6x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 12:00:00 | 1323.84 | 1341.48 | 0.00 | T1 1.5R @ 1323.84 |
| Stop hit — per-position SL triggered | 2024-07-11 13:10:00 | 1330.61 | 1334.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:15:00 | 1401.51 | 1379.91 | 0.00 | ORB-long ORB[1352.77,1372.39] vol=3.1x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-07-22 10:30:00 | 1395.70 | 1385.59 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 1415.61 | 1411.03 | 0.00 | ORB-long ORB[1401.52,1414.00] vol=3.0x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:55:00 | 1422.30 | 1415.40 | 0.00 | T1 1.5R @ 1422.30 |
| Target hit | 2024-07-26 11:40:00 | 1416.57 | 1417.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2024-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:40:00 | 1419.74 | 1413.69 | 0.00 | ORB-long ORB[1404.00,1413.92] vol=3.3x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-07-31 10:55:00 | 1415.69 | 1414.02 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:00:00 | 1389.26 | 1399.04 | 0.00 | ORB-short ORB[1400.32,1418.00] vol=3.0x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-08-01 11:05:00 | 1393.16 | 1398.80 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:05:00 | 1383.10 | 1376.56 | 0.00 | ORB-long ORB[1367.85,1380.02] vol=2.4x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:40:00 | 1391.82 | 1379.97 | 0.00 | T1 1.5R @ 1391.82 |
| Target hit | 2024-08-07 15:20:00 | 1409.99 | 1400.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:45:00 | 1390.23 | 1394.22 | 0.00 | ORB-short ORB[1390.41,1410.21] vol=2.5x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-08-08 11:10:00 | 1394.59 | 1393.93 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:35:00 | 1437.39 | 1428.41 | 0.00 | ORB-long ORB[1418.08,1427.89] vol=2.6x ATR=5.24 |
| Stop hit — per-position SL triggered | 2024-08-14 10:50:00 | 1432.15 | 1430.34 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 1499.14 | 1492.70 | 0.00 | ORB-long ORB[1477.53,1497.44] vol=2.4x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:45:00 | 1503.94 | 1495.38 | 0.00 | T1 1.5R @ 1503.94 |
| Stop hit — per-position SL triggered | 2024-08-26 09:55:00 | 1499.14 | 1496.46 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1495.77 | 1499.37 | 0.00 | ORB-short ORB[1497.43,1506.00] vol=2.8x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 1499.26 | 1499.29 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 11:15:00 | 1529.67 | 1532.09 | 0.00 | ORB-short ORB[1532.21,1545.00] vol=2.4x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-08-30 11:50:00 | 1533.03 | 1531.83 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 1486.96 | 1488.47 | 0.00 | ORB-short ORB[1488.88,1500.15] vol=4.4x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-09-06 10:10:00 | 1490.85 | 1488.73 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:40:00 | 1493.17 | 1477.04 | 0.00 | ORB-long ORB[1461.00,1482.98] vol=1.7x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-09-09 10:55:00 | 1488.54 | 1478.62 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:15:00 | 1486.41 | 1494.49 | 0.00 | ORB-short ORB[1492.34,1502.04] vol=1.7x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-09-10 11:25:00 | 1490.30 | 1494.08 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:10:00 | 1522.63 | 1509.50 | 0.00 | ORB-long ORB[1495.55,1504.00] vol=3.0x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:15:00 | 1530.01 | 1514.53 | 0.00 | T1 1.5R @ 1530.01 |
| Target hit | 2024-09-11 14:35:00 | 1537.62 | 1537.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-09-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:50:00 | 1565.20 | 1555.89 | 0.00 | ORB-long ORB[1543.00,1559.44] vol=4.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:55:00 | 1571.39 | 1558.90 | 0.00 | T1 1.5R @ 1571.39 |
| Target hit | 2024-09-16 11:50:00 | 1574.31 | 1574.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2024-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 1519.01 | 1537.72 | 0.00 | ORB-short ORB[1543.01,1561.99] vol=1.5x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-09-18 11:35:00 | 1524.22 | 1534.78 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 1597.30 | 1606.64 | 0.00 | ORB-short ORB[1604.07,1627.99] vol=2.0x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:40:00 | 1591.19 | 1603.60 | 0.00 | T1 1.5R @ 1591.19 |
| Target hit | 2024-09-25 15:20:00 | 1584.37 | 1591.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:30:00 | 1629.60 | 1618.91 | 0.00 | ORB-long ORB[1600.09,1623.83] vol=1.7x ATR=7.65 |
| Stop hit — per-position SL triggered | 2024-09-27 10:00:00 | 1621.95 | 1622.16 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 1647.40 | 1639.95 | 0.00 | ORB-long ORB[1628.12,1643.94] vol=1.6x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-10-01 09:55:00 | 1641.81 | 1643.49 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:40:00 | 1650.96 | 1638.76 | 0.00 | ORB-long ORB[1624.00,1645.53] vol=4.2x ATR=7.85 |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 1643.11 | 1651.27 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:10:00 | 1627.72 | 1621.82 | 0.00 | ORB-long ORB[1600.77,1615.39] vol=9.1x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:25:00 | 1635.31 | 1622.93 | 0.00 | T1 1.5R @ 1635.31 |
| Target hit | 2024-10-08 14:50:00 | 1631.60 | 1631.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2024-10-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:40:00 | 1684.39 | 1672.48 | 0.00 | ORB-long ORB[1660.00,1674.00] vol=2.2x ATR=5.92 |
| Stop hit — per-position SL triggered | 2024-10-09 10:55:00 | 1678.47 | 1674.48 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:50:00 | 1667.94 | 1674.49 | 0.00 | ORB-short ORB[1674.35,1687.89] vol=2.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:15:00 | 1661.74 | 1672.88 | 0.00 | T1 1.5R @ 1661.74 |
| Stop hit — per-position SL triggered | 2024-10-10 11:25:00 | 1667.94 | 1672.51 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:15:00 | 1564.51 | 1581.06 | 0.00 | ORB-short ORB[1573.87,1594.18] vol=2.6x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:25:00 | 1555.43 | 1576.99 | 0.00 | T1 1.5R @ 1555.43 |
| Stop hit — per-position SL triggered | 2024-10-22 10:55:00 | 1564.51 | 1570.60 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:35:00 | 1512.03 | 1521.84 | 0.00 | ORB-short ORB[1513.72,1535.32] vol=1.8x ATR=6.34 |
| Stop hit — per-position SL triggered | 2024-10-23 09:45:00 | 1518.37 | 1521.17 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 10:45:00 | 1550.28 | 1555.11 | 0.00 | ORB-short ORB[1556.15,1569.72] vol=1.8x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:55:00 | 1543.26 | 1553.73 | 0.00 | T1 1.5R @ 1543.26 |
| Stop hit — per-position SL triggered | 2024-10-24 12:00:00 | 1550.28 | 1549.19 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:55:00 | 1527.49 | 1535.55 | 0.00 | ORB-short ORB[1538.14,1551.99] vol=1.6x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:40:00 | 1518.18 | 1529.83 | 0.00 | T1 1.5R @ 1518.18 |
| Target hit | 2024-10-25 13:35:00 | 1514.77 | 1513.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 11:15:00 | 1537.90 | 1532.03 | 0.00 | ORB-long ORB[1520.16,1535.32] vol=1.9x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:30:00 | 1544.12 | 1532.61 | 0.00 | T1 1.5R @ 1544.12 |
| Stop hit — per-position SL triggered | 2024-11-19 14:20:00 | 1537.90 | 1537.75 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 1665.23 | 1653.98 | 0.00 | ORB-long ORB[1645.33,1657.46] vol=4.1x ATR=5.04 |
| Stop hit — per-position SL triggered | 2024-11-27 11:20:00 | 1660.19 | 1656.40 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:00:00 | 1647.81 | 1664.08 | 0.00 | ORB-short ORB[1654.74,1671.33] vol=1.9x ATR=6.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:05:00 | 1638.16 | 1663.13 | 0.00 | T1 1.5R @ 1638.16 |
| Stop hit — per-position SL triggered | 2024-11-28 11:30:00 | 1647.81 | 1660.99 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:00:00 | 1694.28 | 1674.24 | 0.00 | ORB-long ORB[1645.83,1663.80] vol=1.8x ATR=6.48 |
| Stop hit — per-position SL triggered | 2024-12-02 10:05:00 | 1687.80 | 1676.39 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1752.18 | 1745.33 | 0.00 | ORB-long ORB[1738.00,1750.00] vol=1.6x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:45:00 | 1759.31 | 1750.81 | 0.00 | T1 1.5R @ 1759.31 |
| Stop hit — per-position SL triggered | 2024-12-16 09:55:00 | 1752.18 | 1752.13 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 1720.00 | 1724.30 | 0.00 | ORB-short ORB[1722.39,1734.22] vol=1.9x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 1724.29 | 1723.36 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 1768.55 | 1756.87 | 0.00 | ORB-long ORB[1742.01,1760.00] vol=5.5x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:15:00 | 1777.47 | 1766.29 | 0.00 | T1 1.5R @ 1777.47 |
| Target hit | 2025-01-02 11:25:00 | 1769.20 | 1770.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 1796.80 | 1785.30 | 0.00 | ORB-long ORB[1776.00,1791.60] vol=1.7x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:10:00 | 1803.42 | 1790.85 | 0.00 | T1 1.5R @ 1803.42 |
| Stop hit — per-position SL triggered | 2025-01-03 11:10:00 | 1796.80 | 1796.64 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:45:00 | 1665.99 | 1680.18 | 0.00 | ORB-short ORB[1674.11,1693.99] vol=2.0x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:00:00 | 1656.40 | 1677.86 | 0.00 | T1 1.5R @ 1656.40 |
| Target hit | 2025-01-08 15:20:00 | 1647.56 | 1650.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 1613.40 | 1630.68 | 0.00 | ORB-short ORB[1635.68,1651.43] vol=4.0x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:30:00 | 1604.57 | 1623.00 | 0.00 | T1 1.5R @ 1604.57 |
| Stop hit — per-position SL triggered | 2025-01-09 12:10:00 | 1613.40 | 1619.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 1507.48 | 1484.88 | 0.00 | ORB-long ORB[1458.11,1479.75] vol=1.9x ATR=8.03 |
| Stop hit — per-position SL triggered | 2025-01-14 11:25:00 | 1499.45 | 1488.67 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1530.02 | 1536.47 | 0.00 | ORB-short ORB[1543.51,1563.95] vol=4.8x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 1533.86 | 1536.37 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:40:00 | 1529.23 | 1546.23 | 0.00 | ORB-short ORB[1553.05,1567.58] vol=2.2x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-02-04 10:50:00 | 1534.93 | 1544.23 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:35:00 | 1576.86 | 1566.48 | 0.00 | ORB-long ORB[1542.51,1565.56] vol=1.5x ATR=5.42 |
| Stop hit — per-position SL triggered | 2025-02-05 10:40:00 | 1571.44 | 1567.04 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:10:00 | 1487.18 | 1491.45 | 0.00 | ORB-short ORB[1492.58,1514.57] vol=1.8x ATR=6.01 |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 1493.19 | 1491.86 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 10:55:00 | 1534.70 | 1527.25 | 0.00 | ORB-long ORB[1511.00,1528.63] vol=1.5x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-02-13 13:35:00 | 1529.87 | 1532.04 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 11:15:00 | 1458.00 | 1476.18 | 0.00 | ORB-short ORB[1475.40,1493.00] vol=1.5x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:45:00 | 1449.81 | 1470.56 | 0.00 | T1 1.5R @ 1449.81 |
| Target hit | 2025-02-24 15:20:00 | 1436.90 | 1441.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:30:00 | 1367.97 | 1361.37 | 0.00 | ORB-long ORB[1352.28,1365.00] vol=1.6x ATR=4.33 |
| Stop hit — per-position SL triggered | 2025-03-05 09:35:00 | 1363.64 | 1362.36 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:10:00 | 1439.49 | 1441.39 | 0.00 | ORB-short ORB[1444.00,1462.75] vol=1.5x ATR=5.25 |
| Stop hit — per-position SL triggered | 2025-03-28 12:00:00 | 1444.74 | 1441.65 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1378.20 | 1363.57 | 0.00 | ORB-long ORB[1351.31,1370.02] vol=1.9x ATR=6.19 |
| Stop hit — per-position SL triggered | 2025-04-02 10:20:00 | 1372.01 | 1369.23 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:05:00 | 1423.00 | 1416.95 | 0.00 | ORB-long ORB[1406.30,1422.90] vol=1.8x ATR=4.80 |
| Stop hit — per-position SL triggered | 2025-04-24 10:20:00 | 1418.20 | 1417.46 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 1428.00 | 1437.76 | 0.00 | ORB-short ORB[1430.50,1449.90] vol=2.1x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 1418.68 | 1434.39 | 0.00 | T1 1.5R @ 1418.68 |
| Target hit | 2025-04-25 13:45:00 | 1416.80 | 1415.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 67 — SELL (started 2025-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 10:45:00 | 1389.70 | 1398.01 | 0.00 | ORB-short ORB[1398.30,1418.70] vol=3.3x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-04-28 10:50:00 | 1393.94 | 1396.77 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1438.90 | 1435.11 | 0.00 | ORB-long ORB[1422.80,1437.30] vol=2.2x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-05-05 11:30:00 | 1434.90 | 1435.12 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:00:00 | 1424.00 | 1433.20 | 0.00 | ORB-short ORB[1433.60,1453.60] vol=1.7x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:20:00 | 1416.31 | 1430.93 | 0.00 | T1 1.5R @ 1416.31 |
| Target hit | 2025-05-06 15:20:00 | 1396.60 | 1411.15 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 11:05:00 | 1193.92 | 2024-05-15 11:15:00 | 1197.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-16 10:55:00 | 1186.40 | 2024-05-16 11:45:00 | 1190.03 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-21 10:15:00 | 1260.06 | 2024-05-21 10:25:00 | 1268.92 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-05-21 10:15:00 | 1260.06 | 2024-05-21 15:20:00 | 1280.35 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2024-05-22 10:30:00 | 1278.14 | 2024-05-22 10:35:00 | 1283.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1166.00 | 2024-05-30 09:45:00 | 1169.74 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-12 10:55:00 | 1260.03 | 2024-06-12 11:00:00 | 1256.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-13 09:55:00 | 1266.23 | 2024-06-13 10:05:00 | 1271.73 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-13 09:55:00 | 1266.23 | 2024-06-13 11:25:00 | 1266.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:30:00 | 1252.40 | 2024-06-20 09:40:00 | 1259.54 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-20 09:30:00 | 1252.40 | 2024-06-20 12:30:00 | 1256.02 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-21 09:35:00 | 1297.75 | 2024-06-21 09:40:00 | 1292.03 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-25 10:40:00 | 1329.20 | 2024-06-25 10:45:00 | 1336.70 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-06-25 10:40:00 | 1329.20 | 2024-06-25 11:00:00 | 1329.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 09:45:00 | 1352.57 | 2024-06-27 10:25:00 | 1347.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-01 10:30:00 | 1383.64 | 2024-07-01 10:40:00 | 1378.92 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-04 10:45:00 | 1353.20 | 2024-07-04 10:55:00 | 1356.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-05 10:45:00 | 1357.56 | 2024-07-05 10:55:00 | 1354.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-09 09:55:00 | 1391.42 | 2024-07-09 10:00:00 | 1397.39 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-09 09:55:00 | 1391.42 | 2024-07-09 10:50:00 | 1396.40 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-10 10:40:00 | 1363.99 | 2024-07-10 10:55:00 | 1370.51 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-11 10:45:00 | 1330.61 | 2024-07-11 12:00:00 | 1323.84 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-11 10:45:00 | 1330.61 | 2024-07-11 13:10:00 | 1330.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-22 10:15:00 | 1401.51 | 2024-07-22 10:30:00 | 1395.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-26 09:30:00 | 1415.61 | 2024-07-26 10:55:00 | 1422.30 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-26 09:30:00 | 1415.61 | 2024-07-26 11:40:00 | 1416.57 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-07-31 10:40:00 | 1419.74 | 2024-07-31 10:55:00 | 1415.69 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-01 11:00:00 | 1389.26 | 2024-08-01 11:05:00 | 1393.16 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-07 10:05:00 | 1383.10 | 2024-08-07 10:40:00 | 1391.82 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-07 10:05:00 | 1383.10 | 2024-08-07 15:20:00 | 1409.99 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2024-08-08 10:45:00 | 1390.23 | 2024-08-08 11:10:00 | 1394.59 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-14 10:35:00 | 1437.39 | 2024-08-14 10:50:00 | 1432.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-26 09:30:00 | 1499.14 | 2024-08-26 09:45:00 | 1503.94 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-26 09:30:00 | 1499.14 | 2024-08-26 09:55:00 | 1499.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1495.77 | 2024-08-28 09:35:00 | 1499.26 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-30 11:15:00 | 1529.67 | 2024-08-30 11:50:00 | 1533.03 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 10:05:00 | 1486.96 | 2024-09-06 10:10:00 | 1490.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-09 10:40:00 | 1493.17 | 2024-09-09 10:55:00 | 1488.54 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-10 11:15:00 | 1486.41 | 2024-09-10 11:25:00 | 1490.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-11 10:10:00 | 1522.63 | 2024-09-11 10:15:00 | 1530.01 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-11 10:10:00 | 1522.63 | 2024-09-11 14:35:00 | 1537.62 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-09-16 10:50:00 | 1565.20 | 2024-09-16 10:55:00 | 1571.39 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-16 10:50:00 | 1565.20 | 2024-09-16 11:50:00 | 1574.31 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-18 11:00:00 | 1519.01 | 2024-09-18 11:35:00 | 1524.22 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-25 11:00:00 | 1597.30 | 2024-09-25 11:40:00 | 1591.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-25 11:00:00 | 1597.30 | 2024-09-25 15:20:00 | 1584.37 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2024-09-27 09:30:00 | 1629.60 | 2024-09-27 10:00:00 | 1621.95 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-01 09:30:00 | 1647.40 | 2024-10-01 09:55:00 | 1641.81 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-04 09:40:00 | 1650.96 | 2024-10-04 14:15:00 | 1643.11 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-10-08 11:10:00 | 1627.72 | 2024-10-08 11:25:00 | 1635.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-10-08 11:10:00 | 1627.72 | 2024-10-08 14:50:00 | 1631.60 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-10-09 10:40:00 | 1684.39 | 2024-10-09 10:55:00 | 1678.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-10 10:50:00 | 1667.94 | 2024-10-10 11:15:00 | 1661.74 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-10 10:50:00 | 1667.94 | 2024-10-10 11:25:00 | 1667.94 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:15:00 | 1564.51 | 2024-10-22 10:25:00 | 1555.43 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-22 10:15:00 | 1564.51 | 2024-10-22 10:55:00 | 1564.51 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-23 09:35:00 | 1512.03 | 2024-10-23 09:45:00 | 1518.37 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-24 10:45:00 | 1550.28 | 2024-10-24 10:55:00 | 1543.26 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-24 10:45:00 | 1550.28 | 2024-10-24 12:00:00 | 1550.28 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 09:55:00 | 1527.49 | 2024-10-25 10:40:00 | 1518.18 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-25 09:55:00 | 1527.49 | 2024-10-25 13:35:00 | 1514.77 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-11-19 11:15:00 | 1537.90 | 2024-11-19 11:30:00 | 1544.12 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-19 11:15:00 | 1537.90 | 2024-11-19 14:20:00 | 1537.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 11:00:00 | 1665.23 | 2024-11-27 11:20:00 | 1660.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-28 11:00:00 | 1647.81 | 2024-11-28 11:05:00 | 1638.16 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-28 11:00:00 | 1647.81 | 2024-11-28 11:30:00 | 1647.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 10:00:00 | 1694.28 | 2024-12-02 10:05:00 | 1687.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1752.18 | 2024-12-16 09:45:00 | 1759.31 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1752.18 | 2024-12-16 09:55:00 | 1752.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 09:40:00 | 1720.00 | 2024-12-30 10:05:00 | 1724.29 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1768.55 | 2025-01-02 10:15:00 | 1777.47 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-02 09:35:00 | 1768.55 | 2025-01-02 11:25:00 | 1769.20 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2025-01-03 10:00:00 | 1796.80 | 2025-01-03 10:10:00 | 1803.42 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-03 10:00:00 | 1796.80 | 2025-01-03 11:10:00 | 1796.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 09:45:00 | 1665.99 | 2025-01-08 10:00:00 | 1656.40 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-08 09:45:00 | 1665.99 | 2025-01-08 15:20:00 | 1647.56 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2025-01-09 10:50:00 | 1613.40 | 2025-01-09 11:30:00 | 1604.57 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-09 10:50:00 | 1613.40 | 2025-01-09 12:10:00 | 1613.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-14 11:00:00 | 1507.48 | 2025-01-14 11:25:00 | 1499.45 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1530.02 | 2025-02-01 11:10:00 | 1533.86 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-04 10:40:00 | 1529.23 | 2025-02-04 10:50:00 | 1534.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-05 10:35:00 | 1576.86 | 2025-02-05 10:40:00 | 1571.44 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-12 10:10:00 | 1487.18 | 2025-02-12 10:15:00 | 1493.19 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-13 10:55:00 | 1534.70 | 2025-02-13 13:35:00 | 1529.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-24 11:15:00 | 1458.00 | 2025-02-24 11:45:00 | 1449.81 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-24 11:15:00 | 1458.00 | 2025-02-24 15:20:00 | 1436.90 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-03-05 09:30:00 | 1367.97 | 2025-03-05 09:35:00 | 1363.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-28 11:10:00 | 1439.49 | 2025-03-28 12:00:00 | 1444.74 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-02 09:30:00 | 1378.20 | 2025-04-02 10:20:00 | 1372.01 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-24 10:05:00 | 1423.00 | 2025-04-24 10:20:00 | 1418.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-25 09:35:00 | 1428.00 | 2025-04-25 09:45:00 | 1418.68 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-04-25 09:35:00 | 1428.00 | 2025-04-25 13:45:00 | 1416.80 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-04-28 10:45:00 | 1389.70 | 2025-04-28 10:50:00 | 1393.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-05 11:15:00 | 1438.90 | 2025-05-05 11:30:00 | 1434.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-06 10:00:00 | 1424.00 | 2025-05-06 10:20:00 | 1416.31 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-05-06 10:00:00 | 1424.00 | 2025-05-06 15:20:00 | 1396.60 | TARGET_HIT | 0.50 | 1.92% |
