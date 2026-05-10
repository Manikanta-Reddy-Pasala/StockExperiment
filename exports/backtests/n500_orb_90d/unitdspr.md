# United Spirits Ltd. (UNITDSPR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1284.00
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
| ENTRY1 | 28 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 9 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 19
- **Target hits / Stop hits / Partials:** 9 / 19 / 12
- **Avg / median % per leg:** 0.29% / 0.30%
- **Sum % (uncompounded):** 11.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 11 | 73.3% | 5 | 4 | 6 | 0.69% | 10.3% |
| BUY @ 2nd Alert (retest1) | 15 | 11 | 73.3% | 5 | 4 | 6 | 0.69% | 10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 10 | 40.0% | 4 | 15 | 6 | 0.06% | 1.5% |
| SELL @ 2nd Alert (retest1) | 25 | 10 | 40.0% | 4 | 15 | 6 | 0.06% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 40 | 21 | 52.5% | 9 | 19 | 12 | 0.29% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1380.80 | 1372.96 | 0.00 | ORB-long ORB[1368.70,1378.70] vol=2.0x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 1388.12 | 1376.32 | 0.00 | T1 1.5R @ 1388.12 |
| Target hit | 2026-02-09 15:20:00 | 1407.80 | 1396.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:15:00 | 1402.40 | 1406.35 | 0.00 | ORB-short ORB[1404.70,1412.30] vol=2.1x ATR=2.19 |
| Stop hit — per-position SL triggered | 2026-02-10 11:50:00 | 1404.59 | 1405.82 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1410.30 | 1412.28 | 0.00 | ORB-short ORB[1410.90,1420.20] vol=3.1x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:40:00 | 1405.86 | 1411.12 | 0.00 | T1 1.5R @ 1405.86 |
| Target hit | 2026-02-11 13:40:00 | 1408.90 | 1408.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2026-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:35:00 | 1425.60 | 1417.09 | 0.00 | ORB-long ORB[1407.40,1419.00] vol=3.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 1422.75 | 1420.52 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 1405.30 | 1411.94 | 0.00 | ORB-short ORB[1407.50,1417.50] vol=1.9x ATR=3.40 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 1408.70 | 1411.77 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 1418.00 | 1412.20 | 0.00 | ORB-long ORB[1407.00,1415.30] vol=1.8x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-02-17 10:20:00 | 1414.45 | 1413.26 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 1421.30 | 1423.66 | 0.00 | ORB-short ORB[1421.90,1431.30] vol=2.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-02-18 11:35:00 | 1423.65 | 1423.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 1413.00 | 1420.13 | 0.00 | ORB-short ORB[1417.10,1429.80] vol=1.7x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 1408.61 | 1417.41 | 0.00 | T1 1.5R @ 1408.61 |
| Stop hit — per-position SL triggered | 2026-02-19 12:55:00 | 1413.00 | 1415.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 1397.40 | 1400.70 | 0.00 | ORB-short ORB[1400.30,1409.80] vol=2.0x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:10:00 | 1392.77 | 1399.18 | 0.00 | T1 1.5R @ 1392.77 |
| Target hit | 2026-02-26 11:25:00 | 1393.20 | 1391.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1377.60 | 1380.47 | 0.00 | ORB-short ORB[1380.50,1392.70] vol=1.7x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:00:00 | 1372.22 | 1379.13 | 0.00 | T1 1.5R @ 1372.22 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 1377.60 | 1378.25 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:20:00 | 1339.90 | 1346.39 | 0.00 | ORB-short ORB[1340.10,1356.00] vol=2.0x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:40:00 | 1333.40 | 1344.46 | 0.00 | T1 1.5R @ 1333.40 |
| Target hit | 2026-03-04 15:20:00 | 1313.70 | 1329.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 1384.10 | 1370.25 | 0.00 | ORB-long ORB[1360.50,1368.90] vol=2.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:20:00 | 1390.48 | 1373.16 | 0.00 | T1 1.5R @ 1390.48 |
| Target hit | 2026-03-10 15:20:00 | 1409.90 | 1396.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-03-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:10:00 | 1315.70 | 1320.72 | 0.00 | ORB-short ORB[1318.60,1332.50] vol=2.2x ATR=4.40 |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 1320.10 | 1320.64 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:35:00 | 1315.90 | 1310.59 | 0.00 | ORB-long ORB[1305.20,1312.60] vol=3.3x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-18 10:40:00 | 1312.72 | 1310.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 1293.00 | 1295.89 | 0.00 | ORB-short ORB[1298.80,1317.60] vol=8.5x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-03-19 11:35:00 | 1297.05 | 1295.74 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-03-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:35:00 | 1284.30 | 1292.90 | 0.00 | ORB-short ORB[1285.80,1304.00] vol=1.5x ATR=4.55 |
| Stop hit — per-position SL triggered | 2026-03-24 10:45:00 | 1288.85 | 1292.49 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 1230.20 | 1242.72 | 0.00 | ORB-short ORB[1237.10,1250.00] vol=2.1x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 1234.06 | 1239.86 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 1246.00 | 1251.05 | 0.00 | ORB-short ORB[1251.00,1263.10] vol=1.8x ATR=3.47 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 1249.47 | 1248.83 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 1271.60 | 1267.02 | 0.00 | ORB-long ORB[1254.90,1270.00] vol=1.6x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:35:00 | 1277.44 | 1269.34 | 0.00 | T1 1.5R @ 1277.44 |
| Target hit | 2026-04-17 10:50:00 | 1283.10 | 1283.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 1326.10 | 1322.34 | 0.00 | ORB-long ORB[1310.20,1323.80] vol=2.9x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:20:00 | 1331.24 | 1326.17 | 0.00 | T1 1.5R @ 1331.24 |
| Target hit | 2026-04-21 15:20:00 | 1364.70 | 1345.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1379.80 | 1370.11 | 0.00 | ORB-long ORB[1357.00,1376.60] vol=1.7x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:20:00 | 1384.96 | 1370.82 | 0.00 | T1 1.5R @ 1384.96 |
| Target hit | 2026-04-22 15:20:00 | 1390.40 | 1385.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 1374.50 | 1379.05 | 0.00 | ORB-short ORB[1378.20,1395.00] vol=2.3x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-04-24 10:10:00 | 1379.26 | 1378.82 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 1402.20 | 1397.30 | 0.00 | ORB-long ORB[1391.90,1401.80] vol=2.3x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:05:00 | 1407.30 | 1398.73 | 0.00 | T1 1.5R @ 1407.30 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 1402.20 | 1400.71 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 1380.60 | 1385.92 | 0.00 | ORB-short ORB[1386.20,1398.00] vol=1.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-04-28 11:55:00 | 1384.01 | 1383.76 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 1373.90 | 1380.32 | 0.00 | ORB-short ORB[1379.90,1391.10] vol=1.6x ATR=2.98 |
| Stop hit — per-position SL triggered | 2026-04-29 11:35:00 | 1376.88 | 1379.05 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2026-04-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:10:00 | 1322.80 | 1336.60 | 0.00 | ORB-short ORB[1335.50,1350.00] vol=2.3x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 1326.66 | 1335.57 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 1309.50 | 1320.53 | 0.00 | ORB-short ORB[1321.80,1334.20] vol=1.7x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-05-05 10:55:00 | 1312.52 | 1319.83 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 1287.10 | 1293.01 | 0.00 | ORB-short ORB[1290.10,1301.20] vol=2.3x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:45:00 | 1282.69 | 1291.18 | 0.00 | T1 1.5R @ 1282.69 |
| Target hit | 2026-05-07 15:20:00 | 1281.00 | 1283.89 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 1380.80 | 2026-02-09 11:00:00 | 1388.12 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-09 10:30:00 | 1380.80 | 2026-02-09 15:20:00 | 1407.80 | TARGET_HIT | 0.50 | 1.96% |
| SELL | retest1 | 2026-02-10 11:15:00 | 1402.40 | 2026-02-10 11:50:00 | 1404.59 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-11 11:05:00 | 1410.30 | 2026-02-11 11:40:00 | 1405.86 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-11 11:05:00 | 1410.30 | 2026-02-11 13:40:00 | 1408.90 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-02-12 10:35:00 | 1425.60 | 2026-02-12 11:15:00 | 1422.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 09:35:00 | 1405.30 | 2026-02-13 09:40:00 | 1408.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-17 09:45:00 | 1418.00 | 2026-02-17 10:20:00 | 1414.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 10:50:00 | 1421.30 | 2026-02-18 11:35:00 | 1423.65 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1413.00 | 2026-02-19 12:05:00 | 1408.61 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1413.00 | 2026-02-19 12:55:00 | 1413.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:05:00 | 1397.40 | 2026-02-26 10:10:00 | 1392.77 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-26 10:05:00 | 1397.40 | 2026-02-26 11:25:00 | 1393.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1377.60 | 2026-02-27 11:00:00 | 1372.22 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1377.60 | 2026-02-27 11:25:00 | 1377.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 10:20:00 | 1339.90 | 2026-03-04 10:40:00 | 1333.40 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-04 10:20:00 | 1339.90 | 2026-03-04 15:20:00 | 1313.70 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2026-03-10 10:50:00 | 1384.10 | 2026-03-10 11:20:00 | 1390.48 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-10 10:50:00 | 1384.10 | 2026-03-10 15:20:00 | 1409.90 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2026-03-17 10:10:00 | 1315.70 | 2026-03-17 10:15:00 | 1320.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-18 10:35:00 | 1315.90 | 2026-03-18 10:40:00 | 1312.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-19 11:00:00 | 1293.00 | 2026-03-19 11:35:00 | 1297.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-24 10:35:00 | 1284.30 | 2026-03-24 10:45:00 | 1288.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-30 10:55:00 | 1230.20 | 2026-03-30 12:00:00 | 1234.06 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-16 09:55:00 | 1246.00 | 2026-04-16 10:20:00 | 1249.47 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-17 09:30:00 | 1271.60 | 2026-04-17 09:35:00 | 1277.44 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-17 09:30:00 | 1271.60 | 2026-04-17 10:50:00 | 1283.10 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-04-21 09:50:00 | 1326.10 | 2026-04-21 11:20:00 | 1331.24 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-21 09:50:00 | 1326.10 | 2026-04-21 15:20:00 | 1364.70 | TARGET_HIT | 0.50 | 2.91% |
| BUY | retest1 | 2026-04-22 11:15:00 | 1379.80 | 2026-04-22 11:20:00 | 1384.96 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-22 11:15:00 | 1379.80 | 2026-04-22 15:20:00 | 1390.40 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1374.50 | 2026-04-24 10:10:00 | 1379.26 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-27 10:55:00 | 1402.20 | 2026-04-27 11:05:00 | 1407.30 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-27 10:55:00 | 1402.20 | 2026-04-27 11:25:00 | 1402.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:00:00 | 1380.60 | 2026-04-28 11:55:00 | 1384.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-29 11:00:00 | 1373.90 | 2026-04-29 11:35:00 | 1376.88 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-30 11:10:00 | 1322.80 | 2026-04-30 11:25:00 | 1326.66 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-05 10:50:00 | 1309.50 | 2026-05-05 10:55:00 | 1312.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1287.10 | 2026-05-07 11:45:00 | 1282.69 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1287.10 | 2026-05-07 15:20:00 | 1281.00 | TARGET_HIT | 0.50 | 0.47% |
