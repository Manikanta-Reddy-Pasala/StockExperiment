# Axis Bank Ltd. (AXISBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1270.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 12
- **Target hits / Stop hits / Partials:** 6 / 12 / 12
- **Avg / median % per leg:** 0.38% / 0.35%
- **Sum % (uncompounded):** 11.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.38% | 4.2% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.38% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 12 | 63.2% | 4 | 7 | 8 | 0.38% | 7.3% |
| SELL @ 2nd Alert (retest1) | 19 | 12 | 63.2% | 4 | 7 | 8 | 0.38% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 18 | 60.0% | 6 | 12 | 12 | 0.38% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:55:00 | 1324.70 | 1337.66 | 0.00 | ORB-short ORB[1339.30,1346.50] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:10:00 | 1319.63 | 1334.33 | 0.00 | T1 1.5R @ 1319.63 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 1324.70 | 1332.49 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 1359.80 | 1355.15 | 0.00 | ORB-long ORB[1351.40,1358.30] vol=3.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1357.37 | 1355.40 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1366.50 | 1360.04 | 0.00 | ORB-long ORB[1355.50,1364.00] vol=2.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-18 11:10:00 | 1364.07 | 1360.44 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 1395.70 | 1393.60 | 0.00 | ORB-long ORB[1387.60,1395.00] vol=7.2x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 1399.36 | 1394.18 | 0.00 | T1 1.5R @ 1399.36 |
| Stop hit — per-position SL triggered | 2026-02-25 12:35:00 | 1395.70 | 1395.05 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1332.90 | 1337.78 | 0.00 | ORB-short ORB[1335.80,1347.90] vol=1.5x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:55:00 | 1325.75 | 1334.23 | 0.00 | T1 1.5R @ 1325.75 |
| Target hit | 2026-03-06 12:05:00 | 1326.10 | 1325.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 1309.80 | 1298.61 | 0.00 | ORB-long ORB[1284.60,1298.50] vol=2.7x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:35:00 | 1315.16 | 1300.89 | 0.00 | T1 1.5R @ 1315.16 |
| Stop hit — per-position SL triggered | 2026-03-10 12:20:00 | 1309.80 | 1302.84 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1289.90 | 1295.46 | 0.00 | ORB-short ORB[1301.40,1317.50] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:10:00 | 1285.46 | 1293.97 | 0.00 | T1 1.5R @ 1285.46 |
| Target hit | 2026-03-11 15:20:00 | 1254.50 | 1274.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1208.20 | 1218.62 | 0.00 | ORB-short ORB[1222.90,1234.50] vol=1.6x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:55:00 | 1203.77 | 1215.93 | 0.00 | T1 1.5R @ 1203.77 |
| Target hit | 2026-03-13 15:20:00 | 1196.60 | 1206.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:40:00 | 1174.40 | 1179.98 | 0.00 | ORB-short ORB[1180.00,1189.80] vol=2.3x ATR=4.71 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1179.11 | 1178.72 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1214.90 | 1206.32 | 0.00 | ORB-long ORB[1196.40,1211.30] vol=2.5x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 11:45:00 | 1221.26 | 1207.83 | 0.00 | T1 1.5R @ 1221.26 |
| Target hit | 2026-04-06 15:20:00 | 1242.70 | 1223.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:55:00 | 1324.20 | 1314.24 | 0.00 | ORB-long ORB[1300.10,1318.10] vol=1.6x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-04-08 13:05:00 | 1318.99 | 1317.40 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 1339.10 | 1329.70 | 0.00 | ORB-long ORB[1315.30,1330.50] vol=1.9x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:45:00 | 1344.49 | 1331.61 | 0.00 | T1 1.5R @ 1344.49 |
| Target hit | 2026-04-13 15:20:00 | 1353.30 | 1346.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 1358.90 | 1367.64 | 0.00 | ORB-short ORB[1364.10,1384.50] vol=1.7x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 12:25:00 | 1353.94 | 1365.61 | 0.00 | T1 1.5R @ 1353.94 |
| Stop hit — per-position SL triggered | 2026-04-15 14:55:00 | 1358.90 | 1360.52 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 1351.90 | 1353.51 | 0.00 | ORB-short ORB[1353.30,1364.50] vol=1.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1355.51 | 1353.59 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 1355.50 | 1362.68 | 0.00 | ORB-short ORB[1366.00,1375.00] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 1350.71 | 1362.08 | 0.00 | T1 1.5R @ 1350.71 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 1355.50 | 1360.81 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1301.80 | 1310.03 | 0.00 | ORB-short ORB[1310.20,1319.70] vol=3.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 1297.97 | 1308.97 | 0.00 | T1 1.5R @ 1297.97 |
| Target hit | 2026-04-28 15:20:00 | 1289.40 | 1297.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1261.70 | 1266.39 | 0.00 | ORB-short ORB[1263.10,1271.50] vol=1.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 1264.63 | 1264.77 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1265.90 | 1271.97 | 0.00 | ORB-short ORB[1269.40,1286.30] vol=1.5x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:15:00 | 1260.95 | 1271.13 | 0.00 | T1 1.5R @ 1260.95 |
| Stop hit — per-position SL triggered | 2026-05-06 13:30:00 | 1265.90 | 1266.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:55:00 | 1324.70 | 2026-02-13 10:10:00 | 1319.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-13 09:55:00 | 1324.70 | 2026-02-13 10:25:00 | 1324.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:05:00 | 1359.80 | 2026-02-17 11:30:00 | 1357.37 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-18 11:00:00 | 1366.50 | 2026-02-18 11:10:00 | 1364.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-25 10:55:00 | 1395.70 | 2026-02-25 11:25:00 | 1399.36 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-25 10:55:00 | 1395.70 | 2026-02-25 12:35:00 | 1395.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 09:45:00 | 1332.90 | 2026-03-06 09:55:00 | 1325.75 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-06 09:45:00 | 1332.90 | 2026-03-06 12:05:00 | 1326.10 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-10 11:10:00 | 1309.80 | 2026-03-10 11:35:00 | 1315.16 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-10 11:10:00 | 1309.80 | 2026-03-10 12:20:00 | 1309.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:00:00 | 1289.90 | 2026-03-11 11:10:00 | 1285.46 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-11 11:00:00 | 1289.90 | 2026-03-11 15:20:00 | 1254.50 | TARGET_HIT | 0.50 | 2.74% |
| SELL | retest1 | 2026-03-13 11:15:00 | 1208.20 | 2026-03-13 11:55:00 | 1203.77 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-13 11:15:00 | 1208.20 | 2026-03-13 15:20:00 | 1196.60 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-03-24 09:40:00 | 1174.40 | 2026-03-24 10:15:00 | 1179.11 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-06 11:15:00 | 1214.90 | 2026-04-06 11:45:00 | 1221.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-06 11:15:00 | 1214.90 | 2026-04-06 15:20:00 | 1242.70 | TARGET_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2026-04-08 10:55:00 | 1324.20 | 2026-04-08 13:05:00 | 1318.99 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-13 10:55:00 | 1339.10 | 2026-04-13 11:45:00 | 1344.49 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-13 10:55:00 | 1339.10 | 2026-04-13 15:20:00 | 1353.30 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1358.90 | 2026-04-15 12:25:00 | 1353.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1358.90 | 2026-04-15 14:55:00 | 1358.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:00:00 | 1351.90 | 2026-04-16 11:15:00 | 1355.51 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1355.50 | 2026-04-24 11:00:00 | 1350.71 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1355.50 | 2026-04-24 11:20:00 | 1355.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1301.80 | 2026-04-28 11:20:00 | 1297.97 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1301.80 | 2026-04-28 15:20:00 | 1289.40 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2026-05-05 11:05:00 | 1261.70 | 2026-05-05 12:50:00 | 1264.63 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1265.90 | 2026-05-06 11:15:00 | 1260.95 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1265.90 | 2026-05-06 13:30:00 | 1265.90 | STOP_HIT | 0.50 | 0.00% |
