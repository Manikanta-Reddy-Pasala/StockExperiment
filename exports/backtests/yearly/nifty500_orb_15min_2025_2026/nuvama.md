# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1631.10
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 53
- **Target hits / Stop hits / Partials:** 9 / 53 / 23
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 9.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 15 | 33.3% | 4 | 30 | 11 | 0.07% | 3.1% |
| BUY @ 2nd Alert (retest1) | 45 | 15 | 33.3% | 4 | 30 | 11 | 0.07% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.17% | 6.8% |
| SELL @ 2nd Alert (retest1) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.17% | 6.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 85 | 32 | 37.6% | 9 | 53 | 23 | 0.12% | 9.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 09:50:00 | 1390.90 | 1384.70 | 0.00 | ORB-long ORB[1371.20,1389.20] vol=2.8x ATR=5.20 |
| Stop hit — per-position SL triggered | 2025-05-27 10:00:00 | 1385.70 | 1384.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 1461.20 | 1453.65 | 0.00 | ORB-long ORB[1440.00,1457.20] vol=2.4x ATR=5.17 |
| Stop hit — per-position SL triggered | 2025-06-06 09:50:00 | 1456.03 | 1455.68 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1426.00 | 1413.63 | 0.00 | ORB-long ORB[1402.20,1417.70] vol=1.9x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:35:00 | 1434.71 | 1420.95 | 0.00 | T1 1.5R @ 1434.71 |
| Target hit | 2025-06-17 11:00:00 | 1445.00 | 1453.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 1439.10 | 1450.91 | 0.00 | ORB-short ORB[1449.80,1464.90] vol=3.1x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:00:00 | 1431.07 | 1448.11 | 0.00 | T1 1.5R @ 1431.07 |
| Target hit | 2025-06-19 15:20:00 | 1393.20 | 1415.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:50:00 | 1430.00 | 1417.53 | 0.00 | ORB-long ORB[1406.90,1427.90] vol=1.7x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-06-24 10:00:00 | 1424.00 | 1418.72 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:30:00 | 1461.80 | 1452.22 | 0.00 | ORB-long ORB[1433.90,1455.00] vol=6.4x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 09:45:00 | 1471.71 | 1458.67 | 0.00 | T1 1.5R @ 1471.71 |
| Target hit | 2025-06-25 15:20:00 | 1502.50 | 1481.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1622.60 | 1633.81 | 0.00 | ORB-short ORB[1637.60,1659.80] vol=6.5x ATR=6.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:50:00 | 1612.67 | 1633.44 | 0.00 | T1 1.5R @ 1612.67 |
| Stop hit — per-position SL triggered | 2025-07-01 11:00:00 | 1622.60 | 1633.08 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:40:00 | 1463.40 | 1453.84 | 0.00 | ORB-long ORB[1441.60,1461.80] vol=1.8x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-07-30 09:45:00 | 1458.49 | 1454.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:50:00 | 1380.70 | 1388.70 | 0.00 | ORB-short ORB[1388.10,1401.80] vol=1.8x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:30:00 | 1373.64 | 1383.05 | 0.00 | T1 1.5R @ 1373.64 |
| Stop hit — per-position SL triggered | 2025-08-06 11:45:00 | 1380.70 | 1380.31 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:45:00 | 1384.60 | 1376.01 | 0.00 | ORB-long ORB[1362.10,1379.20] vol=2.5x ATR=6.16 |
| Stop hit — per-position SL triggered | 2025-08-12 10:00:00 | 1378.44 | 1377.31 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 1342.70 | 1346.96 | 0.00 | ORB-short ORB[1350.00,1365.80] vol=7.0x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-08-20 09:35:00 | 1346.76 | 1346.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 1403.20 | 1394.27 | 0.00 | ORB-long ORB[1380.80,1398.30] vol=2.2x ATR=6.29 |
| Stop hit — per-position SL triggered | 2025-08-22 09:40:00 | 1396.91 | 1394.76 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 1309.60 | 1302.38 | 0.00 | ORB-long ORB[1290.70,1299.80] vol=4.6x ATR=3.89 |
| Stop hit — per-position SL triggered | 2025-09-03 09:40:00 | 1305.71 | 1303.95 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:50:00 | 1309.90 | 1317.35 | 0.00 | ORB-short ORB[1313.80,1325.90] vol=2.2x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-09-04 10:00:00 | 1314.96 | 1316.10 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-09-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:55:00 | 1269.20 | 1278.97 | 0.00 | ORB-short ORB[1278.00,1296.20] vol=6.0x ATR=3.21 |
| Stop hit — per-position SL triggered | 2025-09-09 11:00:00 | 1272.41 | 1278.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 1293.00 | 1282.65 | 0.00 | ORB-long ORB[1276.50,1289.90] vol=2.3x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-09-10 09:40:00 | 1288.94 | 1283.17 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-09-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:40:00 | 1297.50 | 1294.21 | 0.00 | ORB-long ORB[1282.10,1295.60] vol=2.3x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-09-15 09:45:00 | 1292.36 | 1294.30 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 1269.90 | 1273.22 | 0.00 | ORB-short ORB[1270.10,1279.50] vol=1.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 09:45:00 | 1264.82 | 1271.09 | 0.00 | T1 1.5R @ 1264.82 |
| Stop hit — per-position SL triggered | 2025-09-17 10:30:00 | 1269.90 | 1267.11 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:15:00 | 1288.80 | 1282.42 | 0.00 | ORB-long ORB[1275.20,1286.00] vol=1.7x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:40:00 | 1294.54 | 1285.70 | 0.00 | T1 1.5R @ 1294.54 |
| Stop hit — per-position SL triggered | 2025-09-18 11:10:00 | 1288.80 | 1288.01 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:40:00 | 1239.60 | 1248.96 | 0.00 | ORB-short ORB[1250.00,1260.60] vol=1.5x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:05:00 | 1234.75 | 1244.17 | 0.00 | T1 1.5R @ 1234.75 |
| Stop hit — per-position SL triggered | 2025-09-23 11:55:00 | 1239.60 | 1240.47 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:10:00 | 1235.30 | 1239.74 | 0.00 | ORB-short ORB[1236.60,1244.40] vol=1.7x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 1239.17 | 1237.18 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:35:00 | 1219.70 | 1225.57 | 0.00 | ORB-short ORB[1222.50,1234.90] vol=1.7x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:45:00 | 1213.92 | 1223.07 | 0.00 | T1 1.5R @ 1213.92 |
| Stop hit — per-position SL triggered | 2025-09-25 10:00:00 | 1219.70 | 1221.79 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 1201.40 | 1208.53 | 0.00 | ORB-short ORB[1203.50,1216.00] vol=2.0x ATR=3.99 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 1205.39 | 1208.00 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 1244.70 | 1231.98 | 0.00 | ORB-long ORB[1222.00,1237.90] vol=2.4x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1240.55 | 1233.43 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:45:00 | 1276.70 | 1266.55 | 0.00 | ORB-long ORB[1255.00,1262.90] vol=2.7x ATR=3.64 |
| Stop hit — per-position SL triggered | 2025-10-01 09:50:00 | 1273.06 | 1268.81 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 1398.10 | 1391.86 | 0.00 | ORB-long ORB[1380.80,1395.20] vol=1.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:40:00 | 1403.61 | 1394.67 | 0.00 | T1 1.5R @ 1403.61 |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1398.10 | 1396.79 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 1432.90 | 1438.51 | 0.00 | ORB-short ORB[1433.50,1448.60] vol=1.7x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 1437.09 | 1438.13 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:10:00 | 1400.40 | 1405.49 | 0.00 | ORB-short ORB[1404.00,1419.60] vol=1.6x ATR=4.89 |
| Stop hit — per-position SL triggered | 2025-10-20 10:45:00 | 1405.29 | 1403.41 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 1459.80 | 1451.78 | 0.00 | ORB-long ORB[1440.00,1452.00] vol=2.3x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:45:00 | 1465.62 | 1459.31 | 0.00 | T1 1.5R @ 1465.62 |
| Stop hit — per-position SL triggered | 2025-10-24 09:50:00 | 1459.80 | 1459.60 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:40:00 | 1457.50 | 1448.53 | 0.00 | ORB-long ORB[1432.80,1454.10] vol=2.0x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:00:00 | 1464.30 | 1453.36 | 0.00 | T1 1.5R @ 1464.30 |
| Target hit | 2025-10-27 15:20:00 | 1481.60 | 1472.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 1487.00 | 1484.50 | 0.00 | ORB-long ORB[1475.50,1486.70] vol=2.1x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:50:00 | 1493.21 | 1486.51 | 0.00 | T1 1.5R @ 1493.21 |
| Stop hit — per-position SL triggered | 2025-10-28 10:10:00 | 1487.00 | 1487.61 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:00:00 | 1448.60 | 1462.44 | 0.00 | ORB-short ORB[1469.70,1482.20] vol=2.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-10-30 11:05:00 | 1452.66 | 1462.15 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:35:00 | 1477.80 | 1464.56 | 0.00 | ORB-long ORB[1440.10,1454.00] vol=7.2x ATR=5.53 |
| Stop hit — per-position SL triggered | 2025-10-31 09:40:00 | 1472.27 | 1465.37 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1465.00 | 1449.50 | 0.00 | ORB-long ORB[1428.20,1449.90] vol=1.5x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-11-04 11:10:00 | 1460.07 | 1450.06 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 1487.00 | 1472.35 | 0.00 | ORB-long ORB[1455.30,1474.80] vol=2.8x ATR=6.92 |
| Stop hit — per-position SL triggered | 2025-11-10 10:20:00 | 1480.08 | 1479.30 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 1484.60 | 1477.79 | 0.00 | ORB-long ORB[1471.10,1481.80] vol=3.7x ATR=4.18 |
| Stop hit — per-position SL triggered | 2025-11-12 09:40:00 | 1480.42 | 1478.78 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 1488.10 | 1482.02 | 0.00 | ORB-long ORB[1466.60,1486.80] vol=3.6x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:35:00 | 1494.73 | 1484.93 | 0.00 | T1 1.5R @ 1494.73 |
| Stop hit — per-position SL triggered | 2025-11-13 09:40:00 | 1488.10 | 1485.34 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:55:00 | 1475.50 | 1468.34 | 0.00 | ORB-long ORB[1460.30,1470.00] vol=1.6x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-11-20 10:05:00 | 1471.93 | 1468.79 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 1460.60 | 1464.71 | 0.00 | ORB-short ORB[1461.70,1472.00] vol=2.5x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-11-21 09:35:00 | 1463.76 | 1464.49 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:05:00 | 1440.80 | 1449.15 | 0.00 | ORB-short ORB[1443.10,1452.90] vol=4.7x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-11-24 11:10:00 | 1443.52 | 1447.92 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 11:15:00 | 1402.80 | 1408.33 | 0.00 | ORB-short ORB[1407.20,1420.00] vol=2.2x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 11:35:00 | 1398.46 | 1406.69 | 0.00 | T1 1.5R @ 1398.46 |
| Target hit | 2025-11-25 14:15:00 | 1401.90 | 1401.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2025-12-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:30:00 | 1510.60 | 1503.93 | 0.00 | ORB-long ORB[1493.00,1508.40] vol=2.4x ATR=5.40 |
| Stop hit — per-position SL triggered | 2025-12-01 09:45:00 | 1505.20 | 1507.12 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:50:00 | 1477.20 | 1486.02 | 0.00 | ORB-short ORB[1483.10,1497.60] vol=2.8x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:25:00 | 1469.62 | 1479.74 | 0.00 | T1 1.5R @ 1469.62 |
| Target hit | 2025-12-02 15:20:00 | 1453.20 | 1468.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1432.70 | 1440.73 | 0.00 | ORB-short ORB[1436.60,1454.90] vol=2.3x ATR=5.12 |
| Stop hit — per-position SL triggered | 2025-12-03 09:55:00 | 1437.82 | 1438.66 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 1404.00 | 1416.47 | 0.00 | ORB-short ORB[1409.00,1424.00] vol=1.8x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-12-10 11:55:00 | 1408.52 | 1414.30 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:10:00 | 1423.10 | 1424.47 | 0.00 | ORB-short ORB[1424.40,1442.80] vol=1.8x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-12-22 11:25:00 | 1426.10 | 1424.54 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:00:00 | 1436.00 | 1429.80 | 0.00 | ORB-long ORB[1418.40,1432.30] vol=2.2x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:10:00 | 1440.82 | 1431.36 | 0.00 | T1 1.5R @ 1440.82 |
| Stop hit — per-position SL triggered | 2025-12-23 11:25:00 | 1436.00 | 1432.39 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 1482.80 | 1491.92 | 0.00 | ORB-short ORB[1483.40,1504.90] vol=3.5x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:20:00 | 1475.28 | 1490.73 | 0.00 | T1 1.5R @ 1475.28 |
| Stop hit — per-position SL triggered | 2026-01-06 13:45:00 | 1482.80 | 1487.75 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 1471.20 | 1472.32 | 0.00 | ORB-short ORB[1472.90,1487.40] vol=8.0x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:55:00 | 1466.19 | 1470.47 | 0.00 | T1 1.5R @ 1466.19 |
| Target hit | 2026-01-08 12:10:00 | 1464.50 | 1463.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2026-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:55:00 | 1460.40 | 1444.00 | 0.00 | ORB-long ORB[1435.10,1449.00] vol=2.0x ATR=6.46 |
| Stop hit — per-position SL triggered | 2026-01-09 12:35:00 | 1453.94 | 1452.68 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 1409.00 | 1417.46 | 0.00 | ORB-short ORB[1413.10,1424.00] vol=1.8x ATR=5.00 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 1414.00 | 1415.15 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 1390.20 | 1405.46 | 0.00 | ORB-short ORB[1409.60,1425.90] vol=2.7x ATR=7.16 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 1397.36 | 1405.14 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:10:00 | 1351.30 | 1334.46 | 0.00 | ORB-long ORB[1300.00,1320.00] vol=2.0x ATR=7.13 |
| Stop hit — per-position SL triggered | 2026-01-30 10:25:00 | 1344.17 | 1335.86 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-02-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:30:00 | 1391.10 | 1380.46 | 0.00 | ORB-long ORB[1365.00,1384.70] vol=2.2x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 1401.11 | 1388.04 | 0.00 | T1 1.5R @ 1401.11 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 1391.10 | 1389.38 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1276.50 | 1285.39 | 0.00 | ORB-short ORB[1280.20,1289.90] vol=2.6x ATR=4.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:00:00 | 1269.17 | 1284.41 | 0.00 | T1 1.5R @ 1269.17 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1276.50 | 1283.63 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1304.80 | 1297.94 | 0.00 | ORB-long ORB[1288.20,1303.70] vol=2.1x ATR=5.12 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1299.68 | 1300.69 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1279.40 | 1270.61 | 0.00 | ORB-long ORB[1258.10,1273.90] vol=2.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 1274.94 | 1271.74 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1259.80 | 1265.97 | 0.00 | ORB-short ORB[1262.60,1280.00] vol=2.2x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:35:00 | 1253.32 | 1263.68 | 0.00 | T1 1.5R @ 1253.32 |
| Target hit | 2026-02-27 10:25:00 | 1250.30 | 1250.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — BUY (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 1257.40 | 1252.68 | 0.00 | ORB-long ORB[1235.50,1251.00] vol=2.8x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-03-06 09:40:00 | 1251.26 | 1252.74 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 1231.60 | 1237.64 | 0.00 | ORB-short ORB[1235.00,1246.80] vol=2.4x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-03-11 10:00:00 | 1236.02 | 1236.86 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 1367.20 | 1356.35 | 0.00 | ORB-long ORB[1347.00,1363.80] vol=4.9x ATR=4.87 |
| Stop hit — per-position SL triggered | 2026-04-27 12:35:00 | 1362.33 | 1359.71 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 1381.80 | 1376.20 | 0.00 | ORB-long ORB[1360.70,1372.10] vol=6.7x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 1392.22 | 1381.14 | 0.00 | T1 1.5R @ 1392.22 |
| Target hit | 2026-05-06 12:25:00 | 1385.40 | 1385.82 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-27 09:50:00 | 1390.90 | 2025-05-27 10:00:00 | 1385.70 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-06 09:35:00 | 1461.20 | 2025-06-06 09:50:00 | 1456.03 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1426.00 | 2025-06-17 09:35:00 | 1434.71 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1426.00 | 2025-06-17 11:00:00 | 1445.00 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2025-06-19 10:35:00 | 1439.10 | 2025-06-19 11:00:00 | 1431.07 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-19 10:35:00 | 1439.10 | 2025-06-19 15:20:00 | 1393.20 | TARGET_HIT | 0.50 | 3.19% |
| BUY | retest1 | 2025-06-24 09:50:00 | 1430.00 | 2025-06-24 10:00:00 | 1424.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-25 09:30:00 | 1461.80 | 2025-06-25 09:45:00 | 1471.71 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-06-25 09:30:00 | 1461.80 | 2025-06-25 15:20:00 | 1502.50 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2025-07-01 10:45:00 | 1622.60 | 2025-07-01 10:50:00 | 1612.67 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-01 10:45:00 | 1622.60 | 2025-07-01 11:00:00 | 1622.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 09:40:00 | 1463.40 | 2025-07-30 09:45:00 | 1458.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-06 09:50:00 | 1380.70 | 2025-08-06 10:30:00 | 1373.64 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-06 09:50:00 | 1380.70 | 2025-08-06 11:45:00 | 1380.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-12 09:45:00 | 1384.60 | 2025-08-12 10:00:00 | 1378.44 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-08-20 09:30:00 | 1342.70 | 2025-08-20 09:35:00 | 1346.76 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-22 09:35:00 | 1403.20 | 2025-08-22 09:40:00 | 1396.91 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-09-03 09:35:00 | 1309.60 | 2025-09-03 09:40:00 | 1305.71 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-04 09:50:00 | 1309.90 | 2025-09-04 10:00:00 | 1314.96 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-09-09 10:55:00 | 1269.20 | 2025-09-09 11:00:00 | 1272.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-10 09:35:00 | 1293.00 | 2025-09-10 09:40:00 | 1288.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-15 09:40:00 | 1297.50 | 2025-09-15 09:45:00 | 1292.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-17 09:30:00 | 1269.90 | 2025-09-17 09:45:00 | 1264.82 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-17 09:30:00 | 1269.90 | 2025-09-17 10:30:00 | 1269.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 10:15:00 | 1288.80 | 2025-09-18 10:40:00 | 1294.54 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-18 10:15:00 | 1288.80 | 2025-09-18 11:10:00 | 1288.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 09:40:00 | 1239.60 | 2025-09-23 10:05:00 | 1234.75 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-23 09:40:00 | 1239.60 | 2025-09-23 11:55:00 | 1239.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:10:00 | 1235.30 | 2025-09-24 12:15:00 | 1239.17 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-25 09:35:00 | 1219.70 | 2025-09-25 09:45:00 | 1213.92 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-09-25 09:35:00 | 1219.70 | 2025-09-25 10:00:00 | 1219.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-26 09:30:00 | 1201.40 | 2025-09-26 09:40:00 | 1205.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1244.70 | 2025-09-29 11:15:00 | 1240.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-01 09:45:00 | 1276.70 | 2025-10-01 09:50:00 | 1273.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-07 09:35:00 | 1398.10 | 2025-10-07 09:40:00 | 1403.61 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-10-07 09:35:00 | 1398.10 | 2025-10-07 10:15:00 | 1398.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 09:30:00 | 1432.90 | 2025-10-13 09:35:00 | 1437.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-20 10:10:00 | 1400.40 | 2025-10-20 10:45:00 | 1405.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-24 09:30:00 | 1459.80 | 2025-10-24 09:45:00 | 1465.62 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-24 09:30:00 | 1459.80 | 2025-10-24 09:50:00 | 1459.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 10:40:00 | 1457.50 | 2025-10-27 11:00:00 | 1464.30 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-27 10:40:00 | 1457.50 | 2025-10-27 15:20:00 | 1481.60 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2025-10-28 09:40:00 | 1487.00 | 2025-10-28 09:50:00 | 1493.21 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-28 09:40:00 | 1487.00 | 2025-10-28 10:10:00 | 1487.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 11:00:00 | 1448.60 | 2025-10-30 11:05:00 | 1452.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-31 09:35:00 | 1477.80 | 2025-10-31 09:40:00 | 1472.27 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-04 11:00:00 | 1465.00 | 2025-11-04 11:10:00 | 1460.07 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-10 09:35:00 | 1487.00 | 2025-11-10 10:20:00 | 1480.08 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-11-12 09:35:00 | 1484.60 | 2025-11-12 09:40:00 | 1480.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-13 09:30:00 | 1488.10 | 2025-11-13 09:35:00 | 1494.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-13 09:30:00 | 1488.10 | 2025-11-13 09:40:00 | 1488.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-20 09:55:00 | 1475.50 | 2025-11-20 10:05:00 | 1471.93 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-21 09:30:00 | 1460.60 | 2025-11-21 09:35:00 | 1463.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-24 11:05:00 | 1440.80 | 2025-11-24 11:10:00 | 1443.52 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-25 11:15:00 | 1402.80 | 2025-11-25 11:35:00 | 1398.46 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-25 11:15:00 | 1402.80 | 2025-11-25 14:15:00 | 1401.90 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-12-01 09:30:00 | 1510.60 | 2025-12-01 09:45:00 | 1505.20 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-02 09:50:00 | 1477.20 | 2025-12-02 11:25:00 | 1469.62 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-02 09:50:00 | 1477.20 | 2025-12-02 15:20:00 | 1453.20 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1432.70 | 2025-12-03 09:55:00 | 1437.82 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-10 10:55:00 | 1404.00 | 2025-12-10 11:55:00 | 1408.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-22 11:10:00 | 1423.10 | 2025-12-22 11:25:00 | 1426.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-23 11:00:00 | 1436.00 | 2025-12-23 11:10:00 | 1440.82 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-23 11:00:00 | 1436.00 | 2025-12-23 11:25:00 | 1436.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 10:50:00 | 1482.80 | 2026-01-06 11:20:00 | 1475.28 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-06 10:50:00 | 1482.80 | 2026-01-06 13:45:00 | 1482.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:50:00 | 1471.20 | 2026-01-08 10:55:00 | 1466.19 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-08 10:50:00 | 1471.20 | 2026-01-08 12:10:00 | 1464.50 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-09 09:55:00 | 1460.40 | 2026-01-09 12:35:00 | 1453.94 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-01-13 09:35:00 | 1409.00 | 2026-01-13 09:45:00 | 1414.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-21 10:55:00 | 1390.20 | 2026-01-21 11:00:00 | 1397.36 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-01-30 10:10:00 | 1351.30 | 2026-01-30 10:25:00 | 1344.17 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-02-09 09:30:00 | 1391.10 | 2026-02-09 11:00:00 | 1401.11 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-02-09 09:30:00 | 1391.10 | 2026-02-09 12:00:00 | 1391.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1276.50 | 2026-02-17 11:00:00 | 1269.17 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1276.50 | 2026-02-17 11:30:00 | 1276.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:30:00 | 1304.80 | 2026-02-20 13:15:00 | 1299.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1279.40 | 2026-02-24 11:40:00 | 1274.94 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 09:30:00 | 1259.80 | 2026-02-27 09:35:00 | 1253.32 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-27 09:30:00 | 1259.80 | 2026-02-27 10:25:00 | 1250.30 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2026-03-06 09:35:00 | 1257.40 | 2026-03-06 09:40:00 | 1251.26 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-11 09:40:00 | 1231.60 | 2026-03-11 10:00:00 | 1236.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 11:05:00 | 1367.20 | 2026-04-27 12:35:00 | 1362.33 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 09:55:00 | 1381.80 | 2026-05-06 11:00:00 | 1392.22 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-05-06 09:55:00 | 1381.80 | 2026-05-06 12:25:00 | 1385.40 | TARGET_HIT | 0.50 | 0.26% |
