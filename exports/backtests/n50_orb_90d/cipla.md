# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1348.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 20
- **Target hits / Stop hits / Partials:** 3 / 20 / 10
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 2.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 6 | 30.0% | 1 | 14 | 5 | 0.02% | 0.4% |
| BUY @ 2nd Alert (retest1) | 20 | 6 | 30.0% | 1 | 14 | 5 | 0.02% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.18% | 2.3% |
| SELL @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.18% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 13 | 39.4% | 3 | 20 | 10 | 0.08% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:55:00 | 1336.60 | 1338.36 | 0.00 | ORB-short ORB[1338.40,1350.00] vol=4.2x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:05:00 | 1333.15 | 1337.11 | 0.00 | T1 1.5R @ 1333.15 |
| Stop hit — per-position SL triggered | 2026-02-12 11:45:00 | 1336.60 | 1336.37 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 1341.60 | 1343.00 | 0.00 | ORB-short ORB[1345.30,1352.90] vol=10.1x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 13:00:00 | 1338.33 | 1341.72 | 0.00 | T1 1.5R @ 1338.33 |
| Stop hit — per-position SL triggered | 2026-02-18 13:45:00 | 1341.60 | 1341.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 1356.50 | 1351.82 | 0.00 | ORB-long ORB[1347.00,1352.00] vol=2.4x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-02-19 11:10:00 | 1354.10 | 1352.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 1340.00 | 1335.76 | 0.00 | ORB-long ORB[1326.70,1338.90] vol=2.8x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:45:00 | 1343.97 | 1338.35 | 0.00 | T1 1.5R @ 1343.97 |
| Stop hit — per-position SL triggered | 2026-02-25 12:40:00 | 1340.00 | 1338.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1362.40 | 1353.08 | 0.00 | ORB-long ORB[1343.30,1353.00] vol=1.7x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-02-26 10:05:00 | 1359.09 | 1357.58 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:50:00 | 1330.60 | 1333.71 | 0.00 | ORB-short ORB[1332.00,1348.90] vol=1.8x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 12:00:00 | 1324.94 | 1332.10 | 0.00 | T1 1.5R @ 1324.94 |
| Target hit | 2026-03-04 15:20:00 | 1312.70 | 1324.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1319.30 | 1324.73 | 0.00 | ORB-short ORB[1321.10,1332.70] vol=2.0x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:50:00 | 1315.00 | 1323.40 | 0.00 | T1 1.5R @ 1315.00 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 1319.30 | 1323.17 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:15:00 | 1296.80 | 1302.93 | 0.00 | ORB-short ORB[1299.50,1309.00] vol=1.5x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-03-09 10:35:00 | 1300.30 | 1301.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1330.00 | 1324.03 | 0.00 | ORB-long ORB[1317.50,1327.80] vol=4.2x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-03-10 12:20:00 | 1327.03 | 1327.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 1317.90 | 1318.13 | 0.00 | ORB-short ORB[1322.10,1326.90] vol=2.2x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:55:00 | 1313.40 | 1317.55 | 0.00 | T1 1.5R @ 1313.40 |
| Target hit | 2026-03-13 14:10:00 | 1315.70 | 1313.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 1239.00 | 1231.58 | 0.00 | ORB-long ORB[1219.00,1227.50] vol=2.1x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:25:00 | 1242.38 | 1232.85 | 0.00 | T1 1.5R @ 1242.38 |
| Stop hit — per-position SL triggered | 2026-03-25 11:35:00 | 1239.00 | 1233.65 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 1245.30 | 1236.40 | 0.00 | ORB-long ORB[1226.50,1239.90] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-03-27 11:40:00 | 1241.97 | 1238.38 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1205.30 | 1197.74 | 0.00 | ORB-long ORB[1185.40,1198.90] vol=4.0x ATR=4.09 |
| Stop hit — per-position SL triggered | 2026-04-07 12:45:00 | 1201.21 | 1202.24 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:40:00 | 1228.30 | 1224.68 | 0.00 | ORB-long ORB[1213.10,1226.70] vol=3.3x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-04-09 11:00:00 | 1225.05 | 1224.99 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 1234.50 | 1231.95 | 0.00 | ORB-long ORB[1220.80,1234.40] vol=1.8x ATR=3.20 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1231.30 | 1232.52 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 1237.20 | 1235.18 | 0.00 | ORB-long ORB[1224.40,1234.60] vol=2.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1240.44 | 1235.37 | 0.00 | T1 1.5R @ 1240.44 |
| Stop hit — per-position SL triggered | 2026-04-17 11:50:00 | 1237.20 | 1235.82 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:50:00 | 1319.60 | 1310.62 | 0.00 | ORB-long ORB[1299.00,1312.50] vol=1.6x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-04-27 10:55:00 | 1315.86 | 1310.78 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:40:00 | 1325.30 | 1316.37 | 0.00 | ORB-long ORB[1308.00,1320.00] vol=2.0x ATR=3.72 |
| Stop hit — per-position SL triggered | 2026-04-30 09:55:00 | 1321.58 | 1320.50 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 1326.60 | 1323.04 | 0.00 | ORB-long ORB[1313.60,1324.60] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:00:00 | 1331.33 | 1325.46 | 0.00 | T1 1.5R @ 1331.33 |
| Stop hit — per-position SL triggered | 2026-05-04 12:30:00 | 1326.60 | 1328.04 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 1321.10 | 1326.45 | 0.00 | ORB-short ORB[1326.10,1339.40] vol=1.5x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 1324.26 | 1323.84 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:20:00 | 1345.70 | 1338.90 | 0.00 | ORB-long ORB[1333.10,1343.00] vol=1.9x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 1350.43 | 1342.95 | 0.00 | T1 1.5R @ 1350.43 |
| Target hit | 2026-05-06 15:20:00 | 1364.10 | 1358.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 1379.30 | 1372.82 | 0.00 | ORB-long ORB[1366.00,1374.00] vol=1.6x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-05-07 10:40:00 | 1376.43 | 1373.16 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 1350.30 | 1356.47 | 0.00 | ORB-short ORB[1352.00,1366.70] vol=2.1x ATR=4.32 |
| Stop hit — per-position SL triggered | 2026-05-08 10:00:00 | 1354.62 | 1355.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 09:55:00 | 1336.60 | 2026-02-12 11:05:00 | 1333.15 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-12 09:55:00 | 1336.60 | 2026-02-12 11:45:00 | 1336.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:45:00 | 1341.60 | 2026-02-18 13:00:00 | 1338.33 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-18 10:45:00 | 1341.60 | 2026-02-18 13:45:00 | 1341.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 10:50:00 | 1356.50 | 2026-02-19 11:10:00 | 1354.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-25 10:40:00 | 1340.00 | 2026-02-25 11:45:00 | 1343.97 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-25 10:40:00 | 1340.00 | 2026-02-25 12:40:00 | 1340.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1362.40 | 2026-02-26 10:05:00 | 1359.09 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-04 10:50:00 | 1330.60 | 2026-03-04 12:00:00 | 1324.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-04 10:50:00 | 1330.60 | 2026-03-04 15:20:00 | 1312.70 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1319.30 | 2026-03-06 10:50:00 | 1315.00 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1319.30 | 2026-03-06 11:00:00 | 1319.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-09 10:15:00 | 1296.80 | 2026-03-09 10:35:00 | 1300.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-10 11:15:00 | 1330.00 | 2026-03-10 12:20:00 | 1327.03 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-13 11:05:00 | 1317.90 | 2026-03-13 11:55:00 | 1313.40 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-13 11:05:00 | 1317.90 | 2026-03-13 14:10:00 | 1315.70 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-03-25 11:05:00 | 1239.00 | 2026-03-25 11:25:00 | 1242.38 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-03-25 11:05:00 | 1239.00 | 2026-03-25 11:35:00 | 1239.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 11:10:00 | 1245.30 | 2026-03-27 11:40:00 | 1241.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-07 10:15:00 | 1205.30 | 2026-04-07 12:45:00 | 1201.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-09 10:40:00 | 1228.30 | 2026-04-09 11:00:00 | 1225.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-10 09:35:00 | 1234.50 | 2026-04-10 10:05:00 | 1231.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1237.20 | 2026-04-17 11:10:00 | 1240.44 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1237.20 | 2026-04-17 11:50:00 | 1237.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:50:00 | 1319.60 | 2026-04-27 10:55:00 | 1315.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-30 09:40:00 | 1325.30 | 2026-04-30 09:55:00 | 1321.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1326.60 | 2026-05-04 12:00:00 | 1331.33 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1326.60 | 2026-05-04 12:30:00 | 1326.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:00:00 | 1321.10 | 2026-05-05 11:30:00 | 1324.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-06 10:20:00 | 1345.70 | 2026-05-06 10:55:00 | 1350.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-05-06 10:20:00 | 1345.70 | 2026-05-06 15:20:00 | 1364.10 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2026-05-07 10:35:00 | 1379.30 | 2026-05-07 10:40:00 | 1376.43 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-05-08 09:35:00 | 1350.30 | 2026-05-08 10:00:00 | 1354.62 | STOP_HIT | 1.00 | -0.32% |
