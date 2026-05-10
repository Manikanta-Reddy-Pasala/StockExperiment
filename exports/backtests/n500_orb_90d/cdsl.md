# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1261.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 17
- **Target hits / Stop hits / Partials:** 4 / 17 / 8
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 5.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.11% | 1.2% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.11% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.23% | 4.1% |
| SELL @ 2nd Alert (retest1) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.23% | 4.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 12 | 41.4% | 4 | 17 | 8 | 0.19% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 1364.60 | 1355.08 | 0.00 | ORB-long ORB[1342.20,1356.00] vol=2.0x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 1359.54 | 1355.62 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1337.40 | 1348.84 | 0.00 | ORB-short ORB[1342.90,1359.00] vol=1.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 1341.68 | 1346.16 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 1347.50 | 1355.09 | 0.00 | ORB-short ORB[1352.90,1365.00] vol=1.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 1350.92 | 1353.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1336.00 | 1329.57 | 0.00 | ORB-long ORB[1323.10,1334.50] vol=2.3x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:50:00 | 1341.69 | 1331.07 | 0.00 | T1 1.5R @ 1341.69 |
| Stop hit — per-position SL triggered | 2026-02-24 10:05:00 | 1336.00 | 1331.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 1273.00 | 1283.95 | 0.00 | ORB-short ORB[1288.00,1299.00] vol=1.7x ATR=3.82 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 1276.82 | 1282.52 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 1242.60 | 1251.76 | 0.00 | ORB-short ORB[1246.10,1261.70] vol=2.5x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:45:00 | 1236.29 | 1247.16 | 0.00 | T1 1.5R @ 1236.29 |
| Target hit | 2026-03-11 15:20:00 | 1218.50 | 1237.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 1197.60 | 1203.94 | 0.00 | ORB-short ORB[1198.20,1213.40] vol=2.0x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 1202.16 | 1202.59 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 1233.00 | 1221.22 | 0.00 | ORB-long ORB[1205.30,1223.70] vol=1.9x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 1229.22 | 1221.61 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:50:00 | 1147.00 | 1160.55 | 0.00 | ORB-short ORB[1161.10,1176.90] vol=1.7x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:00:00 | 1140.24 | 1159.23 | 0.00 | T1 1.5R @ 1140.24 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1147.00 | 1158.83 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 11:10:00 | 1284.50 | 1264.32 | 0.00 | ORB-long ORB[1249.60,1268.30] vol=2.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-04-08 12:30:00 | 1279.49 | 1269.69 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1335.10 | 1326.10 | 0.00 | ORB-long ORB[1318.10,1331.00] vol=3.1x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:30:00 | 1344.26 | 1332.17 | 0.00 | T1 1.5R @ 1344.26 |
| Target hit | 2026-04-15 15:20:00 | 1339.20 | 1336.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1357.20 | 1369.64 | 0.00 | ORB-short ORB[1365.10,1373.50] vol=2.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 1360.46 | 1369.01 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 1347.60 | 1358.68 | 0.00 | ORB-short ORB[1354.10,1367.20] vol=1.6x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 1341.99 | 1354.43 | 0.00 | T1 1.5R @ 1341.99 |
| Target hit | 2026-04-22 15:20:00 | 1321.40 | 1331.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1319.60 | 1328.92 | 0.00 | ORB-short ORB[1323.60,1335.90] vol=2.2x ATR=3.81 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 1323.41 | 1327.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 1329.80 | 1322.38 | 0.00 | ORB-long ORB[1314.50,1325.70] vol=2.3x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 1325.92 | 1322.98 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 1319.10 | 1323.35 | 0.00 | ORB-short ORB[1320.20,1332.60] vol=2.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 1322.51 | 1322.88 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 1343.50 | 1333.56 | 0.00 | ORB-long ORB[1326.50,1338.60] vol=4.9x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 1340.26 | 1334.47 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 1279.90 | 1290.23 | 0.00 | ORB-short ORB[1282.00,1298.00] vol=1.7x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:30:00 | 1272.88 | 1285.67 | 0.00 | T1 1.5R @ 1272.88 |
| Stop hit — per-position SL triggered | 2026-04-30 13:50:00 | 1279.90 | 1280.51 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1243.00 | 1230.23 | 0.00 | ORB-long ORB[1222.10,1232.80] vol=3.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:25:00 | 1249.25 | 1232.70 | 0.00 | T1 1.5R @ 1249.25 |
| Target hit | 2026-05-05 15:20:00 | 1254.50 | 1243.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 1269.90 | 1277.88 | 0.00 | ORB-short ORB[1280.30,1291.00] vol=1.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 1272.82 | 1277.60 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 1261.20 | 1267.13 | 0.00 | ORB-short ORB[1262.70,1274.40] vol=3.3x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:45:00 | 1255.58 | 1262.73 | 0.00 | T1 1.5R @ 1255.58 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 1261.20 | 1261.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 1364.60 | 2026-02-09 11:30:00 | 1359.54 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-13 09:30:00 | 1337.40 | 2026-02-13 09:40:00 | 1341.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-19 09:35:00 | 1347.50 | 2026-02-19 09:45:00 | 1350.92 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1336.00 | 2026-02-24 09:50:00 | 1341.69 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1336.00 | 2026-02-24 10:05:00 | 1336.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1273.00 | 2026-02-27 11:00:00 | 1276.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-11 10:10:00 | 1242.60 | 2026-03-11 10:45:00 | 1236.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-11 10:10:00 | 1242.60 | 2026-03-11 15:20:00 | 1218.50 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2026-03-12 10:10:00 | 1197.60 | 2026-03-12 10:35:00 | 1202.16 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 11:05:00 | 1233.00 | 2026-03-18 11:10:00 | 1229.22 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1147.00 | 2026-03-23 11:00:00 | 1140.24 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1147.00 | 2026-03-23 11:05:00 | 1147.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 11:10:00 | 1284.50 | 2026-04-08 12:30:00 | 1279.49 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1335.10 | 2026-04-15 11:30:00 | 1344.26 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1335.10 | 2026-04-15 15:20:00 | 1339.20 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1357.20 | 2026-04-21 11:00:00 | 1360.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-22 09:55:00 | 1347.60 | 2026-04-22 10:15:00 | 1341.99 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-22 09:55:00 | 1347.60 | 2026-04-22 15:20:00 | 1321.40 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1319.60 | 2026-04-24 09:35:00 | 1323.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 09:55:00 | 1329.80 | 2026-04-27 10:05:00 | 1325.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-28 09:35:00 | 1319.10 | 2026-04-28 09:50:00 | 1322.51 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1343.50 | 2026-04-29 11:10:00 | 1340.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1279.90 | 2026-04-30 10:30:00 | 1272.88 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1279.90 | 2026-04-30 13:50:00 | 1279.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 11:15:00 | 1243.00 | 2026-05-05 11:25:00 | 1249.25 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-05 11:15:00 | 1243.00 | 2026-05-05 15:20:00 | 1254.50 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1269.90 | 2026-05-07 11:30:00 | 1272.82 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1261.20 | 2026-05-08 09:45:00 | 1255.58 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1261.20 | 2026-05-08 09:50:00 | 1261.20 | STOP_HIT | 0.50 | 0.00% |
