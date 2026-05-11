# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-02-04 15:25:00 (13888 bars)
- **Last close:** 1453.30
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 12 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 30
- **Target hits / Stop hits / Partials:** 12 / 30 / 16
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 14.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 10 | 35.7% | 4 | 18 | 6 | 0.10% | 2.7% |
| BUY @ 2nd Alert (retest1) | 28 | 10 | 35.7% | 4 | 18 | 6 | 0.10% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 18 | 60.0% | 8 | 12 | 10 | 0.40% | 11.9% |
| SELL @ 2nd Alert (retest1) | 30 | 18 | 60.0% | 8 | 12 | 10 | 0.40% | 11.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 58 | 28 | 48.3% | 12 | 30 | 16 | 0.25% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1208.10 | 1200.36 | 0.00 | ORB-long ORB[1185.50,1202.40] vol=1.9x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-05-15 10:40:00 | 1203.22 | 1200.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:40:00 | 1333.30 | 1316.19 | 0.00 | ORB-long ORB[1288.80,1306.20] vol=4.1x ATR=7.02 |
| Stop hit — per-position SL triggered | 2025-06-04 09:45:00 | 1326.28 | 1316.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:35:00 | 1341.20 | 1336.70 | 0.00 | ORB-long ORB[1320.60,1336.00] vol=4.1x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 09:45:00 | 1349.46 | 1343.27 | 0.00 | T1 1.5R @ 1349.46 |
| Target hit | 2025-06-10 11:55:00 | 1385.00 | 1388.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-06-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:50:00 | 1318.40 | 1330.33 | 0.00 | ORB-short ORB[1328.90,1346.00] vol=1.6x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:50:00 | 1310.92 | 1324.81 | 0.00 | T1 1.5R @ 1310.92 |
| Target hit | 2025-06-12 15:20:00 | 1301.20 | 1309.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:10:00 | 1280.10 | 1290.20 | 0.00 | ORB-short ORB[1285.00,1302.00] vol=2.1x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 11:00:00 | 1273.92 | 1285.59 | 0.00 | T1 1.5R @ 1273.92 |
| Stop hit — per-position SL triggered | 2025-06-17 11:50:00 | 1280.10 | 1283.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:50:00 | 1278.80 | 1284.93 | 0.00 | ORB-short ORB[1280.50,1295.00] vol=1.7x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 11:10:00 | 1273.50 | 1284.20 | 0.00 | T1 1.5R @ 1273.50 |
| Target hit | 2025-06-18 15:20:00 | 1256.30 | 1269.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 09:40:00 | 1261.40 | 1256.87 | 0.00 | ORB-long ORB[1246.90,1260.70] vol=1.7x ATR=5.31 |
| Stop hit — per-position SL triggered | 2025-06-23 10:45:00 | 1256.09 | 1259.62 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 09:35:00 | 1259.40 | 1270.61 | 0.00 | ORB-short ORB[1265.00,1278.70] vol=1.7x ATR=4.64 |
| Stop hit — per-position SL triggered | 2025-06-24 09:40:00 | 1264.04 | 1269.47 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:55:00 | 1334.50 | 1326.31 | 0.00 | ORB-long ORB[1318.70,1334.20] vol=1.6x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:20:00 | 1341.87 | 1329.18 | 0.00 | T1 1.5R @ 1341.87 |
| Stop hit — per-position SL triggered | 2025-06-27 10:25:00 | 1334.50 | 1329.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:25:00 | 1499.60 | 1465.19 | 0.00 | ORB-long ORB[1386.20,1408.00] vol=6.5x ATR=13.43 |
| Stop hit — per-position SL triggered | 2025-06-30 10:30:00 | 1486.17 | 1469.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 10:10:00 | 1417.50 | 1426.84 | 0.00 | ORB-short ORB[1424.40,1444.90] vol=2.1x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 1410.62 | 1425.18 | 0.00 | T1 1.5R @ 1410.62 |
| Stop hit — per-position SL triggered | 2025-07-03 11:40:00 | 1417.50 | 1415.06 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:50:00 | 1378.60 | 1381.87 | 0.00 | ORB-short ORB[1386.00,1392.30] vol=2.7x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:05:00 | 1373.82 | 1380.80 | 0.00 | T1 1.5R @ 1373.82 |
| Target hit | 2025-07-08 14:00:00 | 1376.00 | 1375.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2025-07-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:05:00 | 1364.10 | 1357.24 | 0.00 | ORB-long ORB[1347.50,1359.90] vol=2.2x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:10:00 | 1373.36 | 1359.84 | 0.00 | T1 1.5R @ 1373.36 |
| Target hit | 2025-07-15 10:45:00 | 1379.90 | 1385.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2025-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1397.90 | 1404.28 | 0.00 | ORB-short ORB[1407.30,1423.30] vol=1.7x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-07-17 10:30:00 | 1402.09 | 1402.90 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:15:00 | 1405.90 | 1414.06 | 0.00 | ORB-short ORB[1410.10,1430.00] vol=2.9x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:30:00 | 1398.38 | 1412.31 | 0.00 | T1 1.5R @ 1398.38 |
| Target hit | 2025-07-25 15:20:00 | 1363.60 | 1383.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 1377.80 | 1383.10 | 0.00 | ORB-short ORB[1380.00,1400.00] vol=1.8x ATR=5.93 |
| Stop hit — per-position SL triggered | 2025-07-30 10:25:00 | 1383.73 | 1381.62 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:45:00 | 1469.20 | 1474.71 | 0.00 | ORB-short ORB[1471.80,1481.60] vol=2.2x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 1474.43 | 1473.72 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1555.20 | 1545.62 | 0.00 | ORB-long ORB[1530.00,1550.00] vol=2.7x ATR=5.74 |
| Stop hit — per-position SL triggered | 2025-10-03 10:40:00 | 1549.46 | 1545.92 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:15:00 | 1575.10 | 1585.23 | 0.00 | ORB-short ORB[1579.70,1599.00] vol=1.6x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 1567.04 | 1580.17 | 0.00 | T1 1.5R @ 1567.04 |
| Target hit | 2025-10-07 15:20:00 | 1534.00 | 1559.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:05:00 | 1548.90 | 1543.51 | 0.00 | ORB-long ORB[1526.10,1547.90] vol=1.8x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:10:00 | 1557.05 | 1546.27 | 0.00 | T1 1.5R @ 1557.05 |
| Target hit | 2025-10-10 11:50:00 | 1552.90 | 1553.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 1565.90 | 1562.32 | 0.00 | ORB-long ORB[1541.50,1564.90] vol=4.6x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 1559.24 | 1562.14 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1550.90 | 1530.80 | 0.00 | ORB-long ORB[1500.90,1514.80] vol=1.6x ATR=7.04 |
| Stop hit — per-position SL triggered | 2025-10-15 15:20:00 | 1548.70 | 1543.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 1566.80 | 1564.02 | 0.00 | ORB-long ORB[1549.90,1566.30] vol=3.4x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-10-17 09:55:00 | 1562.80 | 1564.62 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:10:00 | 1599.40 | 1593.52 | 0.00 | ORB-long ORB[1575.00,1590.00] vol=13.6x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 1594.07 | 1593.67 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:30:00 | 1486.90 | 1476.38 | 0.00 | ORB-long ORB[1470.00,1482.00] vol=2.3x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:45:00 | 1495.11 | 1478.68 | 0.00 | T1 1.5R @ 1495.11 |
| Target hit | 2025-11-03 15:00:00 | 1490.00 | 1490.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2025-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:40:00 | 1586.60 | 1594.89 | 0.00 | ORB-short ORB[1594.50,1612.30] vol=1.9x ATR=4.97 |
| Stop hit — per-position SL triggered | 2025-11-13 10:25:00 | 1591.57 | 1591.29 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:05:00 | 1577.00 | 1568.35 | 0.00 | ORB-long ORB[1552.60,1574.50] vol=3.6x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-11-14 12:05:00 | 1572.28 | 1569.42 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-11-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:00:00 | 1568.10 | 1571.18 | 0.00 | ORB-short ORB[1569.40,1584.00] vol=3.6x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 1573.06 | 1570.72 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:45:00 | 1738.80 | 1727.54 | 0.00 | ORB-long ORB[1698.00,1720.50] vol=4.3x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:55:00 | 1749.90 | 1728.76 | 0.00 | T1 1.5R @ 1749.90 |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 1738.80 | 1729.66 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:50:00 | 1720.50 | 1714.77 | 0.00 | ORB-long ORB[1704.10,1720.00] vol=1.8x ATR=4.86 |
| Stop hit — per-position SL triggered | 2025-11-26 10:55:00 | 1715.64 | 1716.06 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 11:10:00 | 1710.10 | 1699.09 | 0.00 | ORB-long ORB[1688.80,1705.20] vol=3.0x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 1705.18 | 1699.29 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 1696.50 | 1674.69 | 0.00 | ORB-long ORB[1647.80,1670.70] vol=2.9x ATR=5.61 |
| Stop hit — per-position SL triggered | 2025-12-01 11:00:00 | 1690.89 | 1680.96 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1682.00 | 1699.48 | 0.00 | ORB-short ORB[1687.00,1710.50] vol=1.6x ATR=5.76 |
| Stop hit — per-position SL triggered | 2025-12-03 12:30:00 | 1687.76 | 1692.09 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:35:00 | 1642.30 | 1653.44 | 0.00 | ORB-short ORB[1654.60,1670.20] vol=4.8x ATR=7.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:40:00 | 1630.98 | 1645.42 | 0.00 | T1 1.5R @ 1630.98 |
| Target hit | 2025-12-05 11:15:00 | 1629.30 | 1627.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 1667.60 | 1656.29 | 0.00 | ORB-long ORB[1640.60,1653.90] vol=3.9x ATR=4.98 |
| Stop hit — per-position SL triggered | 2025-12-12 10:00:00 | 1662.62 | 1660.89 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1657.70 | 1669.05 | 0.00 | ORB-short ORB[1665.40,1685.00] vol=5.0x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:40:00 | 1649.78 | 1665.24 | 0.00 | T1 1.5R @ 1649.78 |
| Target hit | 2025-12-18 10:40:00 | 1654.90 | 1653.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 1671.20 | 1658.43 | 0.00 | ORB-long ORB[1643.40,1666.00] vol=1.6x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-12-31 12:05:00 | 1667.10 | 1660.88 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 1546.70 | 1562.38 | 0.00 | ORB-short ORB[1560.00,1581.70] vol=3.9x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 1539.54 | 1557.22 | 0.00 | T1 1.5R @ 1539.54 |
| Target hit | 2026-01-08 14:40:00 | 1537.20 | 1536.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — SELL (started 2026-01-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:05:00 | 1510.40 | 1515.04 | 0.00 | ORB-short ORB[1515.20,1531.90] vol=3.9x ATR=4.43 |
| Stop hit — per-position SL triggered | 2026-01-14 10:10:00 | 1514.83 | 1514.92 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:50:00 | 1559.10 | 1555.20 | 0.00 | ORB-long ORB[1545.00,1556.60] vol=2.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-01-16 11:55:00 | 1554.88 | 1556.13 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-01-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:20:00 | 1476.50 | 1487.86 | 0.00 | ORB-short ORB[1480.40,1498.80] vol=3.6x ATR=5.69 |
| Stop hit — per-position SL triggered | 2026-01-19 10:25:00 | 1482.19 | 1488.30 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:10:00 | 1453.90 | 1467.29 | 0.00 | ORB-short ORB[1459.00,1477.90] vol=3.2x ATR=5.30 |
| Stop hit — per-position SL triggered | 2026-01-22 11:20:00 | 1459.20 | 1467.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:30:00 | 1208.10 | 2025-05-15 10:40:00 | 1203.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-04 09:40:00 | 1333.30 | 2025-06-04 09:45:00 | 1326.28 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-06-10 09:35:00 | 1341.20 | 2025-06-10 09:45:00 | 1349.46 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-10 09:35:00 | 1341.20 | 2025-06-10 11:55:00 | 1385.00 | TARGET_HIT | 0.50 | 3.27% |
| SELL | retest1 | 2025-06-12 10:50:00 | 1318.40 | 2025-06-12 11:50:00 | 1310.92 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-06-12 10:50:00 | 1318.40 | 2025-06-12 15:20:00 | 1301.20 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2025-06-17 10:10:00 | 1280.10 | 2025-06-17 11:00:00 | 1273.92 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-06-17 10:10:00 | 1280.10 | 2025-06-17 11:50:00 | 1280.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-18 10:50:00 | 1278.80 | 2025-06-18 11:10:00 | 1273.50 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-18 10:50:00 | 1278.80 | 2025-06-18 15:20:00 | 1256.30 | TARGET_HIT | 0.50 | 1.76% |
| BUY | retest1 | 2025-06-23 09:40:00 | 1261.40 | 2025-06-23 10:45:00 | 1256.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-06-24 09:35:00 | 1259.40 | 2025-06-24 09:40:00 | 1264.04 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-27 09:55:00 | 1334.50 | 2025-06-27 10:20:00 | 1341.87 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-27 09:55:00 | 1334.50 | 2025-06-27 10:25:00 | 1334.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 10:25:00 | 1499.60 | 2025-06-30 10:30:00 | 1486.17 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2025-07-03 10:10:00 | 1417.50 | 2025-07-03 10:15:00 | 1410.62 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-07-03 10:10:00 | 1417.50 | 2025-07-03 11:40:00 | 1417.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 10:50:00 | 1378.60 | 2025-07-08 11:05:00 | 1373.82 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-08 10:50:00 | 1378.60 | 2025-07-08 14:00:00 | 1376.00 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-07-15 10:05:00 | 1364.10 | 2025-07-15 10:10:00 | 1373.36 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-07-15 10:05:00 | 1364.10 | 2025-07-15 10:45:00 | 1379.90 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-07-17 09:45:00 | 1397.90 | 2025-07-17 10:30:00 | 1402.09 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-25 10:15:00 | 1405.90 | 2025-07-25 10:30:00 | 1398.38 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-25 10:15:00 | 1405.90 | 2025-07-25 15:20:00 | 1363.60 | TARGET_HIT | 0.50 | 3.01% |
| SELL | retest1 | 2025-07-30 09:45:00 | 1377.80 | 2025-07-30 10:25:00 | 1383.73 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-08-21 09:45:00 | 1469.20 | 2025-08-21 10:15:00 | 1474.43 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-03 10:30:00 | 1555.20 | 2025-10-03 10:40:00 | 1549.46 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-07 10:15:00 | 1575.10 | 2025-10-07 11:20:00 | 1567.04 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-07 10:15:00 | 1575.10 | 2025-10-07 15:20:00 | 1534.00 | TARGET_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2025-10-10 10:05:00 | 1548.90 | 2025-10-10 10:10:00 | 1557.05 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-10 10:05:00 | 1548.90 | 2025-10-10 11:50:00 | 1552.90 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-13 09:30:00 | 1565.90 | 2025-10-13 09:35:00 | 1559.24 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-15 10:45:00 | 1550.90 | 2025-10-15 15:20:00 | 1548.70 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-10-17 09:35:00 | 1566.80 | 2025-10-17 09:55:00 | 1562.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-20 10:10:00 | 1599.40 | 2025-10-20 10:15:00 | 1594.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-03 10:30:00 | 1486.90 | 2025-11-03 10:45:00 | 1495.11 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-03 10:30:00 | 1486.90 | 2025-11-03 15:00:00 | 1490.00 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-11-13 09:40:00 | 1586.60 | 2025-11-13 10:25:00 | 1591.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-14 11:05:00 | 1577.00 | 2025-11-14 12:05:00 | 1572.28 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-17 11:00:00 | 1568.10 | 2025-11-17 11:15:00 | 1573.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-24 10:45:00 | 1738.80 | 2025-11-24 10:55:00 | 1749.90 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-11-24 10:45:00 | 1738.80 | 2025-11-24 11:15:00 | 1738.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 09:50:00 | 1720.50 | 2025-11-26 10:55:00 | 1715.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-27 11:10:00 | 1710.10 | 2025-11-27 11:15:00 | 1705.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-01 10:45:00 | 1696.50 | 2025-12-01 11:00:00 | 1690.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-03 11:15:00 | 1682.00 | 2025-12-03 12:30:00 | 1687.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-05 09:35:00 | 1642.30 | 2025-12-05 09:40:00 | 1630.98 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-12-05 09:35:00 | 1642.30 | 2025-12-05 11:15:00 | 1629.30 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-12-12 09:35:00 | 1667.60 | 2025-12-12 10:00:00 | 1662.62 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-18 09:30:00 | 1657.70 | 2025-12-18 09:40:00 | 1649.78 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-18 09:30:00 | 1657.70 | 2025-12-18 10:40:00 | 1654.90 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-12-31 10:50:00 | 1671.20 | 2025-12-31 12:05:00 | 1667.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-08 10:55:00 | 1546.70 | 2026-01-08 11:10:00 | 1539.54 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-01-08 10:55:00 | 1546.70 | 2026-01-08 14:40:00 | 1537.20 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2026-01-14 10:05:00 | 1510.40 | 2026-01-14 10:10:00 | 1514.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-16 10:50:00 | 1559.10 | 2026-01-16 11:55:00 | 1554.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-19 10:20:00 | 1476.50 | 2026-01-19 10:25:00 | 1482.19 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-22 11:10:00 | 1453.90 | 2026-01-22 11:20:00 | 1459.20 | STOP_HIT | 1.00 | -0.36% |
