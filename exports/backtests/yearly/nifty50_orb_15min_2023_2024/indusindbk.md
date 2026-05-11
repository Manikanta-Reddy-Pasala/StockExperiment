# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (54004 bars)
- **Last close:** 948.45
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
| ENTRY1 | 97 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 19 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 78
- **Target hits / Stop hits / Partials:** 19 / 78 / 39
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 22.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 31 | 41.3% | 9 | 44 | 22 | 0.15% | 11.1% |
| BUY @ 2nd Alert (retest1) | 75 | 31 | 41.3% | 9 | 44 | 22 | 0.15% | 11.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 27 | 44.3% | 10 | 34 | 17 | 0.19% | 11.7% |
| SELL @ 2nd Alert (retest1) | 61 | 27 | 44.3% | 10 | 34 | 17 | 0.19% | 11.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 136 | 58 | 42.6% | 19 | 78 | 39 | 0.17% | 22.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 11:00:00 | 1213.10 | 1203.70 | 0.00 | ORB-long ORB[1196.00,1211.85] vol=1.5x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 11:20:00 | 1218.10 | 1205.39 | 0.00 | T1 1.5R @ 1218.10 |
| Target hit | 2023-05-15 15:20:00 | 1220.05 | 1213.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:55:00 | 1229.65 | 1226.13 | 0.00 | ORB-long ORB[1214.00,1227.00] vol=2.2x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 10:05:00 | 1234.24 | 1227.91 | 0.00 | T1 1.5R @ 1234.24 |
| Target hit | 2023-05-17 11:15:00 | 1233.05 | 1234.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2023-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 10:40:00 | 1264.35 | 1257.24 | 0.00 | ORB-long ORB[1245.00,1260.75] vol=1.6x ATR=3.13 |
| Stop hit — per-position SL triggered | 2023-05-23 11:00:00 | 1261.22 | 1258.08 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:40:00 | 1292.95 | 1287.01 | 0.00 | ORB-long ORB[1272.00,1287.95] vol=1.6x ATR=3.39 |
| Stop hit — per-position SL triggered | 2023-05-29 12:05:00 | 1289.56 | 1288.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:45:00 | 1287.00 | 1283.87 | 0.00 | ORB-long ORB[1276.40,1286.95] vol=2.5x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-05-30 10:50:00 | 1284.73 | 1283.90 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:25:00 | 1292.75 | 1289.64 | 0.00 | ORB-long ORB[1283.75,1290.90] vol=1.8x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:00:00 | 1296.52 | 1290.58 | 0.00 | T1 1.5R @ 1296.52 |
| Stop hit — per-position SL triggered | 2023-05-31 11:20:00 | 1292.75 | 1291.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 09:55:00 | 1276.75 | 1281.85 | 0.00 | ORB-short ORB[1278.05,1285.60] vol=1.7x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 10:00:00 | 1272.19 | 1280.64 | 0.00 | T1 1.5R @ 1272.19 |
| Stop hit — per-position SL triggered | 2023-06-02 10:10:00 | 1276.75 | 1280.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 10:50:00 | 1323.40 | 1317.99 | 0.00 | ORB-long ORB[1311.80,1316.65] vol=1.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2023-06-08 10:55:00 | 1320.99 | 1318.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 11:00:00 | 1319.50 | 1325.18 | 0.00 | ORB-short ORB[1320.35,1336.95] vol=1.7x ATR=2.46 |
| Stop hit — per-position SL triggered | 2023-06-12 11:10:00 | 1321.96 | 1324.78 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:05:00 | 1295.00 | 1303.18 | 0.00 | ORB-short ORB[1303.60,1317.25] vol=1.7x ATR=2.98 |
| Stop hit — per-position SL triggered | 2023-06-19 11:10:00 | 1297.98 | 1303.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:55:00 | 1301.30 | 1306.77 | 0.00 | ORB-short ORB[1305.60,1319.00] vol=2.1x ATR=2.76 |
| Stop hit — per-position SL triggered | 2023-06-27 10:20:00 | 1304.06 | 1305.80 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:30:00 | 1328.40 | 1322.78 | 0.00 | ORB-long ORB[1316.25,1325.00] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2023-06-28 09:40:00 | 1324.91 | 1323.66 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:35:00 | 1352.65 | 1350.01 | 0.00 | ORB-long ORB[1337.05,1351.00] vol=1.7x ATR=4.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-30 10:00:00 | 1359.43 | 1351.91 | 0.00 | T1 1.5R @ 1359.43 |
| Target hit | 2023-06-30 15:20:00 | 1377.10 | 1367.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:35:00 | 1370.00 | 1374.34 | 0.00 | ORB-short ORB[1374.00,1387.00] vol=4.0x ATR=3.51 |
| Stop hit — per-position SL triggered | 2023-07-03 09:40:00 | 1373.51 | 1374.23 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 09:30:00 | 1377.30 | 1384.90 | 0.00 | ORB-short ORB[1381.05,1396.70] vol=1.7x ATR=3.66 |
| Stop hit — per-position SL triggered | 2023-07-06 09:35:00 | 1380.96 | 1384.21 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:45:00 | 1366.00 | 1377.28 | 0.00 | ORB-short ORB[1378.00,1393.95] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 1368.84 | 1375.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 1382.20 | 1379.21 | 0.00 | ORB-long ORB[1367.00,1381.95] vol=3.2x ATR=4.19 |
| Stop hit — per-position SL triggered | 2023-07-14 09:40:00 | 1378.01 | 1379.37 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:35:00 | 1414.40 | 1429.77 | 0.00 | ORB-short ORB[1424.60,1443.40] vol=1.5x ATR=8.24 |
| Stop hit — per-position SL triggered | 2023-07-19 10:15:00 | 1422.64 | 1425.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:25:00 | 1431.50 | 1425.11 | 0.00 | ORB-long ORB[1420.00,1429.95] vol=1.6x ATR=3.51 |
| Stop hit — per-position SL triggered | 2023-07-20 10:35:00 | 1427.99 | 1425.44 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:30:00 | 1421.50 | 1416.03 | 0.00 | ORB-long ORB[1405.30,1420.00] vol=1.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2023-07-24 09:40:00 | 1417.41 | 1416.51 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 11:15:00 | 1418.10 | 1427.44 | 0.00 | ORB-short ORB[1420.10,1433.20] vol=1.7x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:25:00 | 1413.81 | 1426.70 | 0.00 | T1 1.5R @ 1413.81 |
| Target hit | 2023-07-27 14:30:00 | 1416.75 | 1416.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 11:15:00 | 1410.25 | 1415.70 | 0.00 | ORB-short ORB[1413.05,1418.75] vol=1.7x ATR=3.23 |
| Stop hit — per-position SL triggered | 2023-07-28 11:25:00 | 1413.48 | 1415.53 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 11:00:00 | 1411.00 | 1403.73 | 0.00 | ORB-long ORB[1400.15,1409.35] vol=1.8x ATR=3.01 |
| Stop hit — per-position SL triggered | 2023-07-31 11:45:00 | 1407.99 | 1406.40 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 11:15:00 | 1390.50 | 1379.31 | 0.00 | ORB-long ORB[1368.85,1387.00] vol=4.2x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 11:30:00 | 1397.54 | 1380.94 | 0.00 | T1 1.5R @ 1397.54 |
| Target hit | 2023-08-04 15:20:00 | 1408.05 | 1397.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2023-08-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:30:00 | 1411.50 | 1406.71 | 0.00 | ORB-long ORB[1400.10,1409.10] vol=1.6x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 09:40:00 | 1416.39 | 1409.27 | 0.00 | T1 1.5R @ 1416.39 |
| Stop hit — per-position SL triggered | 2023-08-10 09:55:00 | 1411.50 | 1411.51 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 10:55:00 | 1408.40 | 1416.16 | 0.00 | ORB-short ORB[1413.05,1431.75] vol=3.3x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 12:30:00 | 1403.87 | 1412.00 | 0.00 | T1 1.5R @ 1403.87 |
| Target hit | 2023-08-11 15:20:00 | 1400.00 | 1404.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2023-08-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:35:00 | 1390.50 | 1384.57 | 0.00 | ORB-long ORB[1377.25,1386.35] vol=2.1x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-08-17 10:45:00 | 1387.42 | 1384.84 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:45:00 | 1421.95 | 1415.28 | 0.00 | ORB-long ORB[1402.25,1417.00] vol=2.0x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 09:50:00 | 1426.81 | 1418.58 | 0.00 | T1 1.5R @ 1426.81 |
| Stop hit — per-position SL triggered | 2023-08-24 11:05:00 | 1421.95 | 1423.73 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 11:15:00 | 1396.45 | 1402.08 | 0.00 | ORB-short ORB[1404.00,1413.70] vol=3.5x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 11:20:00 | 1393.36 | 1401.35 | 0.00 | T1 1.5R @ 1393.36 |
| Stop hit — per-position SL triggered | 2023-08-29 11:35:00 | 1396.45 | 1401.00 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 11:05:00 | 1398.55 | 1404.41 | 0.00 | ORB-short ORB[1403.10,1409.00] vol=1.8x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-08-30 11:15:00 | 1400.83 | 1404.13 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 10:05:00 | 1386.00 | 1392.40 | 0.00 | ORB-short ORB[1386.25,1401.90] vol=2.2x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 10:35:00 | 1379.59 | 1388.70 | 0.00 | T1 1.5R @ 1379.59 |
| Target hit | 2023-08-31 15:00:00 | 1379.00 | 1376.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2023-09-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:40:00 | 1386.45 | 1383.29 | 0.00 | ORB-long ORB[1376.20,1384.00] vol=2.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 11:20:00 | 1391.95 | 1385.58 | 0.00 | T1 1.5R @ 1391.95 |
| Target hit | 2023-09-01 15:20:00 | 1416.35 | 1406.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 1429.40 | 1423.90 | 0.00 | ORB-long ORB[1418.00,1425.00] vol=1.7x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-09-05 09:55:00 | 1426.53 | 1424.84 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 11:00:00 | 1411.45 | 1406.51 | 0.00 | ORB-long ORB[1398.00,1409.90] vol=1.5x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 11:10:00 | 1415.64 | 1407.31 | 0.00 | T1 1.5R @ 1415.64 |
| Stop hit — per-position SL triggered | 2023-09-07 11:30:00 | 1411.45 | 1409.01 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:05:00 | 1451.75 | 1443.19 | 0.00 | ORB-long ORB[1434.10,1445.00] vol=2.0x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:15:00 | 1457.82 | 1446.52 | 0.00 | T1 1.5R @ 1457.82 |
| Stop hit — per-position SL triggered | 2023-09-08 10:20:00 | 1451.75 | 1446.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 10:35:00 | 1438.90 | 1425.87 | 0.00 | ORB-long ORB[1410.00,1427.90] vol=1.6x ATR=5.25 |
| Stop hit — per-position SL triggered | 2023-10-09 11:40:00 | 1433.65 | 1431.11 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 09:55:00 | 1421.05 | 1429.75 | 0.00 | ORB-short ORB[1428.90,1436.35] vol=1.5x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 10:30:00 | 1415.98 | 1424.82 | 0.00 | T1 1.5R @ 1415.98 |
| Target hit | 2023-10-10 12:10:00 | 1420.35 | 1420.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2023-10-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 09:45:00 | 1423.60 | 1427.63 | 0.00 | ORB-short ORB[1425.25,1431.90] vol=1.7x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:05:00 | 1418.80 | 1425.06 | 0.00 | T1 1.5R @ 1418.80 |
| Stop hit — per-position SL triggered | 2023-10-11 10:35:00 | 1423.60 | 1423.45 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:15:00 | 1438.20 | 1431.55 | 0.00 | ORB-long ORB[1420.00,1429.60] vol=1.7x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 10:30:00 | 1443.27 | 1435.08 | 0.00 | T1 1.5R @ 1443.27 |
| Target hit | 2023-10-13 15:20:00 | 1463.95 | 1452.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:30:00 | 1441.55 | 1435.17 | 0.00 | ORB-long ORB[1428.00,1439.35] vol=1.6x ATR=4.11 |
| Stop hit — per-position SL triggered | 2023-10-18 09:50:00 | 1437.44 | 1437.28 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:40:00 | 1463.65 | 1456.71 | 0.00 | ORB-long ORB[1439.10,1459.80] vol=2.4x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 09:50:00 | 1470.38 | 1461.90 | 0.00 | T1 1.5R @ 1470.38 |
| Stop hit — per-position SL triggered | 2023-11-02 10:05:00 | 1463.65 | 1465.57 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:30:00 | 1477.50 | 1472.40 | 0.00 | ORB-long ORB[1464.25,1474.80] vol=2.3x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 09:45:00 | 1482.20 | 1476.42 | 0.00 | T1 1.5R @ 1482.20 |
| Stop hit — per-position SL triggered | 2023-11-07 10:15:00 | 1477.50 | 1478.10 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:15:00 | 1483.00 | 1488.45 | 0.00 | ORB-short ORB[1488.60,1495.70] vol=2.2x ATR=3.37 |
| Stop hit — per-position SL triggered | 2023-11-08 11:25:00 | 1486.37 | 1488.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 09:45:00 | 1499.30 | 1495.32 | 0.00 | ORB-long ORB[1484.85,1494.95] vol=2.0x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 10:05:00 | 1504.67 | 1498.85 | 0.00 | T1 1.5R @ 1504.67 |
| Target hit | 2023-11-09 14:20:00 | 1507.00 | 1508.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2023-11-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:30:00 | 1506.35 | 1500.75 | 0.00 | ORB-long ORB[1493.20,1505.20] vol=1.5x ATR=2.95 |
| Stop hit — per-position SL triggered | 2023-11-10 10:35:00 | 1503.40 | 1500.81 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:55:00 | 1511.25 | 1505.34 | 0.00 | ORB-long ORB[1496.15,1504.80] vol=1.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2023-11-13 10:30:00 | 1508.57 | 1508.94 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:45:00 | 1499.95 | 1503.09 | 0.00 | ORB-short ORB[1502.25,1507.45] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 11:25:00 | 1496.97 | 1502.03 | 0.00 | T1 1.5R @ 1496.97 |
| Stop hit — per-position SL triggered | 2023-11-21 11:50:00 | 1499.95 | 1501.61 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 10:20:00 | 1489.10 | 1496.62 | 0.00 | ORB-short ORB[1494.15,1505.25] vol=1.7x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 11:45:00 | 1483.89 | 1493.03 | 0.00 | T1 1.5R @ 1483.89 |
| Target hit | 2023-11-22 15:20:00 | 1470.90 | 1477.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-11-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:40:00 | 1481.90 | 1489.44 | 0.00 | ORB-short ORB[1486.15,1500.00] vol=1.7x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:50:00 | 1475.64 | 1487.41 | 0.00 | T1 1.5R @ 1475.64 |
| Target hit | 2023-11-30 15:00:00 | 1473.80 | 1470.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 11:15:00 | 1495.05 | 1512.68 | 0.00 | ORB-short ORB[1505.95,1521.35] vol=1.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2023-12-05 11:20:00 | 1500.33 | 1511.35 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 1508.25 | 1513.62 | 0.00 | ORB-short ORB[1510.65,1521.35] vol=2.4x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-12-06 09:40:00 | 1511.92 | 1512.27 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 11:00:00 | 1513.90 | 1503.67 | 0.00 | ORB-long ORB[1496.05,1506.80] vol=2.0x ATR=3.12 |
| Stop hit — per-position SL triggered | 2023-12-07 11:05:00 | 1510.78 | 1504.94 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:45:00 | 1520.50 | 1515.96 | 0.00 | ORB-long ORB[1510.00,1519.70] vol=3.1x ATR=3.45 |
| Stop hit — per-position SL triggered | 2023-12-08 10:55:00 | 1517.05 | 1516.18 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-12-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 11:10:00 | 1516.70 | 1520.71 | 0.00 | ORB-short ORB[1522.40,1528.90] vol=2.4x ATR=2.48 |
| Stop hit — per-position SL triggered | 2023-12-12 11:15:00 | 1519.18 | 1520.64 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:10:00 | 1541.40 | 1530.13 | 0.00 | ORB-long ORB[1515.60,1527.50] vol=2.0x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 11:05:00 | 1547.37 | 1535.77 | 0.00 | T1 1.5R @ 1547.37 |
| Target hit | 2023-12-14 15:20:00 | 1551.95 | 1543.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2023-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 11:00:00 | 1580.85 | 1571.39 | 0.00 | ORB-long ORB[1559.05,1578.65] vol=2.3x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 12:10:00 | 1586.86 | 1575.47 | 0.00 | T1 1.5R @ 1586.86 |
| Stop hit — per-position SL triggered | 2023-12-22 12:30:00 | 1580.85 | 1577.05 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:40:00 | 1573.55 | 1567.80 | 0.00 | ORB-long ORB[1562.00,1569.10] vol=1.7x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-12-26 10:50:00 | 1569.41 | 1568.17 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:45:00 | 1588.55 | 1582.17 | 0.00 | ORB-long ORB[1567.30,1586.40] vol=1.7x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 11:10:00 | 1593.35 | 1584.37 | 0.00 | T1 1.5R @ 1593.35 |
| Stop hit — per-position SL triggered | 2023-12-27 11:55:00 | 1588.55 | 1587.06 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:35:00 | 1580.05 | 1589.19 | 0.00 | ORB-short ORB[1590.10,1597.65] vol=1.5x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-01-02 10:45:00 | 1583.84 | 1588.56 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-01-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:55:00 | 1621.00 | 1614.06 | 0.00 | ORB-long ORB[1593.00,1616.90] vol=2.1x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:10:00 | 1627.47 | 1617.13 | 0.00 | T1 1.5R @ 1627.47 |
| Target hit | 2024-01-04 15:20:00 | 1646.75 | 1638.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 10:15:00 | 1642.00 | 1640.12 | 0.00 | ORB-long ORB[1631.35,1641.35] vol=3.2x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-01-08 10:55:00 | 1637.48 | 1640.63 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:30:00 | 1634.45 | 1628.38 | 0.00 | ORB-long ORB[1620.00,1631.90] vol=1.9x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-01-10 12:45:00 | 1629.80 | 1632.80 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-01-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:55:00 | 1664.70 | 1656.52 | 0.00 | ORB-long ORB[1642.55,1657.50] vol=1.7x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-01-11 12:30:00 | 1660.61 | 1660.79 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-01-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 10:45:00 | 1666.50 | 1658.89 | 0.00 | ORB-long ORB[1647.40,1662.85] vol=2.3x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 11:00:00 | 1671.83 | 1660.59 | 0.00 | T1 1.5R @ 1671.83 |
| Stop hit — per-position SL triggered | 2024-01-12 11:05:00 | 1666.50 | 1660.87 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 11:15:00 | 1678.90 | 1686.03 | 0.00 | ORB-short ORB[1683.40,1691.50] vol=1.7x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 12:05:00 | 1674.41 | 1683.58 | 0.00 | T1 1.5R @ 1674.41 |
| Target hit | 2024-01-16 15:20:00 | 1667.00 | 1675.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 1523.70 | 1535.07 | 0.00 | ORB-short ORB[1532.00,1547.90] vol=1.6x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:20:00 | 1515.92 | 1530.62 | 0.00 | T1 1.5R @ 1515.92 |
| Target hit | 2024-01-23 15:20:00 | 1437.00 | 1478.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2024-01-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:05:00 | 1535.70 | 1523.16 | 0.00 | ORB-long ORB[1507.15,1519.90] vol=3.0x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 10:30:00 | 1544.29 | 1530.05 | 0.00 | T1 1.5R @ 1544.29 |
| Stop hit — per-position SL triggered | 2024-01-31 12:15:00 | 1535.70 | 1535.22 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 1501.25 | 1526.82 | 0.00 | ORB-short ORB[1521.20,1538.80] vol=1.8x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-02-08 12:10:00 | 1506.38 | 1518.18 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:50:00 | 1457.75 | 1449.93 | 0.00 | ORB-long ORB[1440.05,1455.00] vol=3.9x ATR=5.06 |
| Stop hit — per-position SL triggered | 2024-02-13 11:10:00 | 1452.69 | 1450.73 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-02-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:50:00 | 1521.35 | 1516.74 | 0.00 | ORB-long ORB[1509.25,1520.15] vol=1.6x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-02-21 10:10:00 | 1516.70 | 1518.79 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:40:00 | 1473.55 | 1476.82 | 0.00 | ORB-short ORB[1476.80,1484.55] vol=1.7x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 11:50:00 | 1468.00 | 1475.01 | 0.00 | T1 1.5R @ 1468.00 |
| Stop hit — per-position SL triggered | 2024-02-26 13:15:00 | 1473.55 | 1472.72 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:10:00 | 1459.75 | 1468.17 | 0.00 | ORB-short ORB[1462.75,1474.00] vol=1.8x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-02-27 10:15:00 | 1463.42 | 1467.91 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-02-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 10:10:00 | 1497.55 | 1491.87 | 0.00 | ORB-long ORB[1490.00,1497.05] vol=2.4x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-02-28 10:35:00 | 1493.10 | 1494.85 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 09:40:00 | 1553.60 | 1544.76 | 0.00 | ORB-long ORB[1537.15,1547.00] vol=2.0x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-03-06 09:55:00 | 1548.95 | 1546.39 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 10:55:00 | 1545.80 | 1554.41 | 0.00 | ORB-short ORB[1555.20,1568.75] vol=1.7x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:20:00 | 1540.03 | 1553.01 | 0.00 | T1 1.5R @ 1540.03 |
| Stop hit — per-position SL triggered | 2024-03-11 11:35:00 | 1545.80 | 1552.12 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:55:00 | 1535.90 | 1543.59 | 0.00 | ORB-short ORB[1536.20,1550.00] vol=2.0x ATR=5.10 |
| Stop hit — per-position SL triggered | 2024-03-12 11:00:00 | 1541.00 | 1543.46 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-03-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:30:00 | 1521.00 | 1527.13 | 0.00 | ORB-short ORB[1531.10,1540.60] vol=2.4x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-03-13 10:45:00 | 1526.05 | 1526.85 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-03-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 10:35:00 | 1464.25 | 1475.05 | 0.00 | ORB-short ORB[1471.55,1485.60] vol=1.9x ATR=4.66 |
| Stop hit — per-position SL triggered | 2024-03-15 10:45:00 | 1468.91 | 1474.51 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:15:00 | 1471.15 | 1481.90 | 0.00 | ORB-short ORB[1480.55,1491.60] vol=3.4x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-03-18 11:40:00 | 1475.66 | 1479.26 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:30:00 | 1455.40 | 1464.07 | 0.00 | ORB-short ORB[1469.20,1482.95] vol=2.7x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:45:00 | 1449.17 | 1461.76 | 0.00 | T1 1.5R @ 1449.17 |
| Stop hit — per-position SL triggered | 2024-03-19 11:20:00 | 1455.40 | 1459.11 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:40:00 | 1429.35 | 1446.02 | 0.00 | ORB-short ORB[1436.45,1449.65] vol=1.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2024-03-20 10:55:00 | 1435.13 | 1442.99 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-03-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 11:10:00 | 1484.45 | 1477.29 | 0.00 | ORB-long ORB[1459.00,1476.00] vol=2.1x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-03-21 11:40:00 | 1480.23 | 1477.67 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 11:15:00 | 1504.80 | 1519.08 | 0.00 | ORB-short ORB[1512.10,1523.60] vol=1.8x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-03-26 11:25:00 | 1508.16 | 1518.82 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 11:00:00 | 1530.00 | 1521.01 | 0.00 | ORB-long ORB[1514.45,1526.55] vol=2.3x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-03-27 11:05:00 | 1526.10 | 1521.16 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-03-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:00:00 | 1549.50 | 1544.36 | 0.00 | ORB-long ORB[1534.75,1543.95] vol=2.1x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 13:35:00 | 1555.51 | 1546.98 | 0.00 | T1 1.5R @ 1555.51 |
| Stop hit — per-position SL triggered | 2024-03-28 15:10:00 | 1549.50 | 1550.91 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:30:00 | 1562.50 | 1553.40 | 0.00 | ORB-long ORB[1537.00,1555.75] vol=3.5x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:05:00 | 1569.74 | 1558.74 | 0.00 | T1 1.5R @ 1569.74 |
| Stop hit — per-position SL triggered | 2024-04-02 10:25:00 | 1562.50 | 1559.95 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:50:00 | 1541.20 | 1545.00 | 0.00 | ORB-short ORB[1542.50,1556.90] vol=2.1x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-04-03 11:00:00 | 1544.51 | 1544.91 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 11:05:00 | 1574.40 | 1560.92 | 0.00 | ORB-long ORB[1553.45,1567.30] vol=2.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-04-08 11:20:00 | 1570.55 | 1562.63 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 09:40:00 | 1510.00 | 1518.81 | 0.00 | ORB-short ORB[1516.50,1534.80] vol=2.2x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-04-16 10:15:00 | 1514.68 | 1514.21 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:15:00 | 1487.65 | 1493.52 | 0.00 | ORB-short ORB[1490.00,1501.00] vol=1.6x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-04-22 11:10:00 | 1493.02 | 1490.34 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 11:10:00 | 1475.85 | 1478.49 | 0.00 | ORB-short ORB[1477.10,1485.80] vol=2.1x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-04-23 11:45:00 | 1478.89 | 1478.21 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:40:00 | 1483.25 | 1482.76 | 0.00 | ORB-long ORB[1470.55,1480.00] vol=6.4x ATR=3.95 |
| Stop hit — per-position SL triggered | 2024-04-25 10:45:00 | 1479.30 | 1482.32 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:15:00 | 1479.45 | 1471.15 | 0.00 | ORB-long ORB[1450.10,1471.90] vol=1.5x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-04-29 10:45:00 | 1474.34 | 1475.17 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 11:15:00 | 1502.25 | 1512.12 | 0.00 | ORB-short ORB[1508.00,1521.30] vol=1.5x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-05-02 11:30:00 | 1506.28 | 1511.39 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 09:30:00 | 1505.95 | 1498.17 | 0.00 | ORB-long ORB[1482.05,1503.80] vol=1.9x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-05-06 09:35:00 | 1500.48 | 1498.35 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:20:00 | 1483.75 | 1489.64 | 0.00 | ORB-short ORB[1489.00,1502.65] vol=1.9x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:30:00 | 1478.49 | 1488.31 | 0.00 | T1 1.5R @ 1478.49 |
| Target hit | 2024-05-07 15:20:00 | 1453.95 | 1464.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 97 — SELL (started 2024-05-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:05:00 | 1427.30 | 1441.36 | 0.00 | ORB-short ORB[1445.50,1456.60] vol=1.8x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:35:00 | 1421.32 | 1438.13 | 0.00 | T1 1.5R @ 1421.32 |
| Target hit | 2024-05-09 15:20:00 | 1403.50 | 1414.87 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 11:00:00 | 1213.10 | 2023-05-15 11:20:00 | 1218.10 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-05-15 11:00:00 | 1213.10 | 2023-05-15 15:20:00 | 1220.05 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2023-05-17 09:55:00 | 1229.65 | 2023-05-17 10:05:00 | 1234.24 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-05-17 09:55:00 | 1229.65 | 2023-05-17 11:15:00 | 1233.05 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2023-05-23 10:40:00 | 1264.35 | 2023-05-23 11:00:00 | 1261.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-29 10:40:00 | 1292.95 | 2023-05-29 12:05:00 | 1289.56 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-05-30 10:45:00 | 1287.00 | 2023-05-30 10:50:00 | 1284.73 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-05-31 10:25:00 | 1292.75 | 2023-05-31 11:00:00 | 1296.52 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-05-31 10:25:00 | 1292.75 | 2023-05-31 11:20:00 | 1292.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-02 09:55:00 | 1276.75 | 2023-06-02 10:00:00 | 1272.19 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-06-02 09:55:00 | 1276.75 | 2023-06-02 10:10:00 | 1276.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 10:50:00 | 1323.40 | 2023-06-08 10:55:00 | 1320.99 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-12 11:00:00 | 1319.50 | 2023-06-12 11:10:00 | 1321.96 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-19 11:05:00 | 1295.00 | 2023-06-19 11:10:00 | 1297.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-27 09:55:00 | 1301.30 | 2023-06-27 10:20:00 | 1304.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-28 09:30:00 | 1328.40 | 2023-06-28 09:40:00 | 1324.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-30 09:35:00 | 1352.65 | 2023-06-30 10:00:00 | 1359.43 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-06-30 09:35:00 | 1352.65 | 2023-06-30 15:20:00 | 1377.10 | TARGET_HIT | 0.50 | 1.81% |
| SELL | retest1 | 2023-07-03 09:35:00 | 1370.00 | 2023-07-03 09:40:00 | 1373.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-06 09:30:00 | 1377.30 | 2023-07-06 09:35:00 | 1380.96 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-07 10:45:00 | 1366.00 | 2023-07-07 11:15:00 | 1368.84 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-14 09:30:00 | 1382.20 | 2023-07-14 09:40:00 | 1378.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-07-19 09:35:00 | 1414.40 | 2023-07-19 10:15:00 | 1422.64 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2023-07-20 10:25:00 | 1431.50 | 2023-07-20 10:35:00 | 1427.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-24 09:30:00 | 1421.50 | 2023-07-24 09:40:00 | 1417.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-27 11:15:00 | 1418.10 | 2023-07-27 11:25:00 | 1413.81 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-07-27 11:15:00 | 1418.10 | 2023-07-27 14:30:00 | 1416.75 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-07-28 11:15:00 | 1410.25 | 2023-07-28 11:25:00 | 1413.48 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-31 11:00:00 | 1411.00 | 2023-07-31 11:45:00 | 1407.99 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-04 11:15:00 | 1390.50 | 2023-08-04 11:30:00 | 1397.54 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-04 11:15:00 | 1390.50 | 2023-08-04 15:20:00 | 1408.05 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2023-08-10 09:30:00 | 1411.50 | 2023-08-10 09:40:00 | 1416.39 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-08-10 09:30:00 | 1411.50 | 2023-08-10 09:55:00 | 1411.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-11 10:55:00 | 1408.40 | 2023-08-11 12:30:00 | 1403.87 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-08-11 10:55:00 | 1408.40 | 2023-08-11 15:20:00 | 1400.00 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2023-08-17 10:35:00 | 1390.50 | 2023-08-17 10:45:00 | 1387.42 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-24 09:45:00 | 1421.95 | 2023-08-24 09:50:00 | 1426.81 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-08-24 09:45:00 | 1421.95 | 2023-08-24 11:05:00 | 1421.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-29 11:15:00 | 1396.45 | 2023-08-29 11:20:00 | 1393.36 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-29 11:15:00 | 1396.45 | 2023-08-29 11:35:00 | 1396.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-30 11:05:00 | 1398.55 | 2023-08-30 11:15:00 | 1400.83 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-31 10:05:00 | 1386.00 | 2023-08-31 10:35:00 | 1379.59 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-08-31 10:05:00 | 1386.00 | 2023-08-31 15:00:00 | 1379.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-09-01 10:40:00 | 1386.45 | 2023-09-01 11:20:00 | 1391.95 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-09-01 10:40:00 | 1386.45 | 2023-09-01 15:20:00 | 1416.35 | TARGET_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2023-09-05 09:40:00 | 1429.40 | 2023-09-05 09:55:00 | 1426.53 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-07 11:00:00 | 1411.45 | 2023-09-07 11:10:00 | 1415.64 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-07 11:00:00 | 1411.45 | 2023-09-07 11:30:00 | 1411.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-08 10:05:00 | 1451.75 | 2023-09-08 10:15:00 | 1457.82 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-08 10:05:00 | 1451.75 | 2023-09-08 10:20:00 | 1451.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 10:35:00 | 1438.90 | 2023-10-09 11:40:00 | 1433.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-10-10 09:55:00 | 1421.05 | 2023-10-10 10:30:00 | 1415.98 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-10-10 09:55:00 | 1421.05 | 2023-10-10 12:10:00 | 1420.35 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2023-10-11 09:45:00 | 1423.60 | 2023-10-11 10:05:00 | 1418.80 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-11 09:45:00 | 1423.60 | 2023-10-11 10:35:00 | 1423.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-13 10:15:00 | 1438.20 | 2023-10-13 10:30:00 | 1443.27 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-10-13 10:15:00 | 1438.20 | 2023-10-13 15:20:00 | 1463.95 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2023-10-18 09:30:00 | 1441.55 | 2023-10-18 09:50:00 | 1437.44 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-02 09:40:00 | 1463.65 | 2023-11-02 09:50:00 | 1470.38 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-11-02 09:40:00 | 1463.65 | 2023-11-02 10:05:00 | 1463.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 09:30:00 | 1477.50 | 2023-11-07 09:45:00 | 1482.20 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-11-07 09:30:00 | 1477.50 | 2023-11-07 10:15:00 | 1477.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-08 11:15:00 | 1483.00 | 2023-11-08 11:25:00 | 1486.37 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-09 09:45:00 | 1499.30 | 2023-11-09 10:05:00 | 1504.67 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-09 09:45:00 | 1499.30 | 2023-11-09 14:20:00 | 1507.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-11-10 10:30:00 | 1506.35 | 2023-11-10 10:35:00 | 1503.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-13 09:55:00 | 1511.25 | 2023-11-13 10:30:00 | 1508.57 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-21 10:45:00 | 1499.95 | 2023-11-21 11:25:00 | 1496.97 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-21 10:45:00 | 1499.95 | 2023-11-21 11:50:00 | 1499.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-22 10:20:00 | 1489.10 | 2023-11-22 11:45:00 | 1483.89 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-11-22 10:20:00 | 1489.10 | 2023-11-22 15:20:00 | 1470.90 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2023-11-30 09:40:00 | 1481.90 | 2023-11-30 09:50:00 | 1475.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-11-30 09:40:00 | 1481.90 | 2023-11-30 15:00:00 | 1473.80 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2023-12-05 11:15:00 | 1495.05 | 2023-12-05 11:20:00 | 1500.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-12-06 09:30:00 | 1508.25 | 2023-12-06 09:40:00 | 1511.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-07 11:00:00 | 1513.90 | 2023-12-07 11:05:00 | 1510.78 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-08 10:45:00 | 1520.50 | 2023-12-08 10:55:00 | 1517.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-12 11:10:00 | 1516.70 | 2023-12-12 11:15:00 | 1519.18 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-12-14 10:10:00 | 1541.40 | 2023-12-14 11:05:00 | 1547.37 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-12-14 10:10:00 | 1541.40 | 2023-12-14 15:20:00 | 1551.95 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2023-12-22 11:00:00 | 1580.85 | 2023-12-22 12:10:00 | 1586.86 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-12-22 11:00:00 | 1580.85 | 2023-12-22 12:30:00 | 1580.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 10:40:00 | 1573.55 | 2023-12-26 10:50:00 | 1569.41 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-27 10:45:00 | 1588.55 | 2023-12-27 11:10:00 | 1593.35 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-12-27 10:45:00 | 1588.55 | 2023-12-27 11:55:00 | 1588.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 10:35:00 | 1580.05 | 2024-01-02 10:45:00 | 1583.84 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-01-04 09:55:00 | 1621.00 | 2024-01-04 10:10:00 | 1627.47 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-01-04 09:55:00 | 1621.00 | 2024-01-04 15:20:00 | 1646.75 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2024-01-08 10:15:00 | 1642.00 | 2024-01-08 10:55:00 | 1637.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-01-10 09:30:00 | 1634.45 | 2024-01-10 12:45:00 | 1629.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-01-11 09:55:00 | 1664.70 | 2024-01-11 12:30:00 | 1660.61 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-12 10:45:00 | 1666.50 | 2024-01-12 11:00:00 | 1671.83 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-01-12 10:45:00 | 1666.50 | 2024-01-12 11:05:00 | 1666.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-16 11:15:00 | 1678.90 | 2024-01-16 12:05:00 | 1674.41 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-01-16 11:15:00 | 1678.90 | 2024-01-16 15:20:00 | 1667.00 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-01-23 09:55:00 | 1523.70 | 2024-01-23 10:20:00 | 1515.92 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-01-23 09:55:00 | 1523.70 | 2024-01-23 15:20:00 | 1437.00 | TARGET_HIT | 0.50 | 5.69% |
| BUY | retest1 | 2024-01-31 10:05:00 | 1535.70 | 2024-01-31 10:30:00 | 1544.29 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-31 10:05:00 | 1535.70 | 2024-01-31 12:15:00 | 1535.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:00:00 | 1501.25 | 2024-02-08 12:10:00 | 1506.38 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-13 10:50:00 | 1457.75 | 2024-02-13 11:10:00 | 1452.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-02-21 09:50:00 | 1521.35 | 2024-02-21 10:10:00 | 1516.70 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-26 10:40:00 | 1473.55 | 2024-02-26 11:50:00 | 1468.00 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-02-26 10:40:00 | 1473.55 | 2024-02-26 13:15:00 | 1473.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-27 10:10:00 | 1459.75 | 2024-02-27 10:15:00 | 1463.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-28 10:10:00 | 1497.55 | 2024-02-28 10:35:00 | 1493.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-03-06 09:40:00 | 1553.60 | 2024-03-06 09:55:00 | 1548.95 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-11 10:55:00 | 1545.80 | 2024-03-11 11:20:00 | 1540.03 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-03-11 10:55:00 | 1545.80 | 2024-03-11 11:35:00 | 1545.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-12 10:55:00 | 1535.90 | 2024-03-12 11:00:00 | 1541.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-13 10:30:00 | 1521.00 | 2024-03-13 10:45:00 | 1526.05 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-15 10:35:00 | 1464.25 | 2024-03-15 10:45:00 | 1468.91 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-03-18 11:15:00 | 1471.15 | 2024-03-18 11:40:00 | 1475.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-19 10:30:00 | 1455.40 | 2024-03-19 10:45:00 | 1449.17 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-03-19 10:30:00 | 1455.40 | 2024-03-19 11:20:00 | 1455.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 10:40:00 | 1429.35 | 2024-03-20 10:55:00 | 1435.13 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-03-21 11:10:00 | 1484.45 | 2024-03-21 11:40:00 | 1480.23 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-26 11:15:00 | 1504.80 | 2024-03-26 11:25:00 | 1508.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-03-27 11:00:00 | 1530.00 | 2024-03-27 11:05:00 | 1526.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-03-28 11:00:00 | 1549.50 | 2024-03-28 13:35:00 | 1555.51 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-03-28 11:00:00 | 1549.50 | 2024-03-28 15:10:00 | 1549.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-02 09:30:00 | 1562.50 | 2024-04-02 10:05:00 | 1569.74 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-02 09:30:00 | 1562.50 | 2024-04-02 10:25:00 | 1562.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-03 10:50:00 | 1541.20 | 2024-04-03 11:00:00 | 1544.51 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-08 11:05:00 | 1574.40 | 2024-04-08 11:20:00 | 1570.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-16 09:40:00 | 1510.00 | 2024-04-16 10:15:00 | 1514.68 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-22 10:15:00 | 1487.65 | 2024-04-22 11:10:00 | 1493.02 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-23 11:10:00 | 1475.85 | 2024-04-23 11:45:00 | 1478.89 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-25 10:40:00 | 1483.25 | 2024-04-25 10:45:00 | 1479.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-04-29 10:15:00 | 1479.45 | 2024-04-29 10:45:00 | 1474.34 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-02 11:15:00 | 1502.25 | 2024-05-02 11:30:00 | 1506.28 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-06 09:30:00 | 1505.95 | 2024-05-06 09:35:00 | 1500.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-07 10:20:00 | 1483.75 | 2024-05-07 10:30:00 | 1478.49 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-07 10:20:00 | 1483.75 | 2024-05-07 15:20:00 | 1453.95 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2024-05-09 11:05:00 | 1427.30 | 2024-05-09 11:35:00 | 1421.32 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-09 11:05:00 | 1427.30 | 2024-05-09 15:20:00 | 1403.50 | TARGET_HIT | 0.50 | 1.67% |
