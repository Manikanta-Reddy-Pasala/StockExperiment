# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1253.00
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
| ENTRY1 | 26 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 22
- **Target hits / Stop hits / Partials:** 4 / 22 / 10
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 3.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.12% | 2.5% |
| BUY @ 2nd Alert (retest1) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.12% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.06% | 1.0% |
| SELL @ 2nd Alert (retest1) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.06% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 36 | 14 | 38.9% | 4 | 22 | 10 | 0.10% | 3.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1358.90 | 1351.83 | 0.00 | ORB-long ORB[1348.00,1358.00] vol=3.8x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:05:00 | 1365.72 | 1357.29 | 0.00 | T1 1.5R @ 1365.72 |
| Target hit | 2026-02-09 15:20:00 | 1369.60 | 1368.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 1387.30 | 1380.87 | 0.00 | ORB-long ORB[1369.90,1386.00] vol=1.9x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 1384.20 | 1382.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:45:00 | 1381.50 | 1374.02 | 0.00 | ORB-long ORB[1367.00,1378.80] vol=2.5x ATR=3.12 |
| Stop hit — per-position SL triggered | 2026-02-11 11:00:00 | 1378.38 | 1375.13 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:00:00 | 1386.00 | 1390.19 | 0.00 | ORB-short ORB[1386.90,1397.40] vol=2.5x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 1380.09 | 1389.82 | 0.00 | T1 1.5R @ 1380.09 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 1386.00 | 1389.69 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 1426.50 | 1423.58 | 0.00 | ORB-long ORB[1413.90,1426.30] vol=2.1x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 1432.01 | 1424.34 | 0.00 | T1 1.5R @ 1432.01 |
| Target hit | 2026-02-17 15:20:00 | 1435.30 | 1431.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1432.90 | 1434.45 | 0.00 | ORB-short ORB[1434.10,1442.30] vol=9.2x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 1435.86 | 1434.12 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 1418.40 | 1428.87 | 0.00 | ORB-short ORB[1429.30,1438.90] vol=1.6x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 1413.19 | 1426.24 | 0.00 | T1 1.5R @ 1413.19 |
| Target hit | 2026-02-19 15:20:00 | 1395.00 | 1410.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 1404.40 | 1414.54 | 0.00 | ORB-short ORB[1410.10,1423.90] vol=2.0x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1407.39 | 1414.34 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1401.60 | 1405.49 | 0.00 | ORB-short ORB[1402.00,1415.00] vol=2.2x ATR=4.32 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1405.92 | 1402.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1419.80 | 1417.25 | 0.00 | ORB-long ORB[1407.50,1417.60] vol=3.1x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 1425.45 | 1418.22 | 0.00 | T1 1.5R @ 1425.45 |
| Stop hit — per-position SL triggered | 2026-02-26 11:45:00 | 1419.80 | 1418.38 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 1403.60 | 1403.64 | 0.00 | ORB-short ORB[1406.00,1414.90] vol=1.6x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:30:00 | 1398.87 | 1403.51 | 0.00 | T1 1.5R @ 1398.87 |
| Stop hit — per-position SL triggered | 2026-02-27 11:50:00 | 1403.60 | 1403.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:45:00 | 1322.00 | 1325.92 | 0.00 | ORB-short ORB[1322.50,1340.00] vol=1.6x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-03-04 11:40:00 | 1326.84 | 1325.32 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 1338.40 | 1332.38 | 0.00 | ORB-long ORB[1327.20,1338.00] vol=2.0x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-03-05 11:35:00 | 1335.09 | 1333.59 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:15:00 | 1362.40 | 1355.46 | 0.00 | ORB-long ORB[1340.30,1355.00] vol=2.2x ATR=3.70 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 1358.70 | 1359.99 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-03-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:50:00 | 1354.00 | 1349.20 | 0.00 | ORB-long ORB[1338.40,1349.90] vol=1.5x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:15:00 | 1361.19 | 1354.13 | 0.00 | T1 1.5R @ 1361.19 |
| Target hit | 2026-03-10 13:15:00 | 1365.50 | 1370.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2026-03-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:05:00 | 1287.50 | 1295.98 | 0.00 | ORB-short ORB[1300.10,1316.90] vol=3.2x ATR=5.22 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 1292.72 | 1295.34 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-03-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:50:00 | 1308.40 | 1301.13 | 0.00 | ORB-long ORB[1291.50,1303.60] vol=3.2x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 1304.18 | 1303.17 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:30:00 | 1218.20 | 1202.94 | 0.00 | ORB-long ORB[1186.30,1202.90] vol=1.7x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-04-07 11:10:00 | 1213.14 | 1208.97 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1277.10 | 1269.76 | 0.00 | ORB-long ORB[1262.80,1275.00] vol=1.5x ATR=3.64 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 1273.46 | 1271.39 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 1278.10 | 1289.09 | 0.00 | ORB-short ORB[1282.80,1300.80] vol=1.7x ATR=3.68 |
| Stop hit — per-position SL triggered | 2026-04-15 11:10:00 | 1281.78 | 1288.86 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 1284.10 | 1293.57 | 0.00 | ORB-short ORB[1289.00,1308.00] vol=1.9x ATR=5.28 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 1289.38 | 1291.03 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 1312.60 | 1300.17 | 0.00 | ORB-long ORB[1285.20,1301.30] vol=2.1x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-04-17 10:25:00 | 1308.48 | 1301.44 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1330.00 | 1322.43 | 0.00 | ORB-long ORB[1308.60,1320.00] vol=3.8x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 1336.34 | 1326.42 | 0.00 | T1 1.5R @ 1336.34 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 1330.00 | 1327.60 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:05:00 | 1282.70 | 1280.36 | 0.00 | ORB-long ORB[1267.00,1278.50] vol=2.5x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:25:00 | 1288.93 | 1282.57 | 0.00 | T1 1.5R @ 1288.93 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 1282.70 | 1282.77 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1239.60 | 1244.74 | 0.00 | ORB-short ORB[1240.10,1254.00] vol=1.7x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:35:00 | 1234.80 | 1241.53 | 0.00 | T1 1.5R @ 1234.80 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 1239.60 | 1240.17 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 1274.90 | 1266.34 | 0.00 | ORB-long ORB[1254.10,1263.20] vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 1271.40 | 1268.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 1358.90 | 2026-02-09 11:05:00 | 1365.72 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-09 10:30:00 | 1358.90 | 2026-02-09 15:20:00 | 1369.60 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-10 10:25:00 | 1387.30 | 2026-02-10 10:40:00 | 1384.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-11 10:45:00 | 1381.50 | 2026-02-11 11:00:00 | 1378.38 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-13 10:00:00 | 1386.00 | 2026-02-13 10:15:00 | 1380.09 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-13 10:00:00 | 1386.00 | 2026-02-13 10:25:00 | 1386.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:55:00 | 1426.50 | 2026-02-17 10:10:00 | 1432.01 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 09:55:00 | 1426.50 | 2026-02-17 15:20:00 | 1435.30 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1432.90 | 2026-02-18 12:15:00 | 1435.86 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-19 10:50:00 | 1418.40 | 2026-02-19 11:20:00 | 1413.19 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-19 10:50:00 | 1418.40 | 2026-02-19 15:20:00 | 1395.00 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2026-02-23 11:05:00 | 1404.40 | 2026-02-23 11:15:00 | 1407.39 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-25 09:30:00 | 1401.60 | 2026-02-25 10:15:00 | 1405.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1419.80 | 2026-02-26 11:30:00 | 1425.45 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1419.80 | 2026-02-26 11:45:00 | 1419.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 11:10:00 | 1403.60 | 2026-02-27 11:30:00 | 1398.87 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-27 11:10:00 | 1403.60 | 2026-02-27 11:50:00 | 1403.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 10:45:00 | 1322.00 | 2026-03-04 11:40:00 | 1326.84 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-05 11:05:00 | 1338.40 | 2026-03-05 11:35:00 | 1335.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1362.40 | 2026-03-06 10:45:00 | 1358.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-10 09:50:00 | 1354.00 | 2026-03-10 10:15:00 | 1361.19 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-10 09:50:00 | 1354.00 | 2026-03-10 13:15:00 | 1365.50 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2026-03-16 11:05:00 | 1287.50 | 2026-03-16 11:15:00 | 1292.72 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-17 10:50:00 | 1308.40 | 2026-03-17 11:15:00 | 1304.18 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-07 10:30:00 | 1218.20 | 2026-04-07 11:10:00 | 1213.14 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1277.10 | 2026-04-10 11:45:00 | 1273.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-15 11:05:00 | 1278.10 | 2026-04-15 11:10:00 | 1281.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-16 09:35:00 | 1284.10 | 2026-04-16 10:30:00 | 1289.38 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-17 10:15:00 | 1312.60 | 2026-04-17 10:25:00 | 1308.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1330.00 | 2026-04-21 10:05:00 | 1336.34 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1330.00 | 2026-04-21 10:25:00 | 1330.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:05:00 | 1282.70 | 2026-04-29 10:25:00 | 1288.93 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-29 10:05:00 | 1282.70 | 2026-04-29 10:35:00 | 1282.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:15:00 | 1239.60 | 2026-05-05 10:35:00 | 1234.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-05 10:15:00 | 1239.60 | 2026-05-05 11:30:00 | 1239.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:40:00 | 1274.90 | 2026-05-07 11:05:00 | 1271.40 | STOP_HIT | 1.00 | -0.27% |
