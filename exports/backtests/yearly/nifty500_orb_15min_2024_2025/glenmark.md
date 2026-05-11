# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 2361.20
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 11 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 63
- **Target hits / Stop hits / Partials:** 11 / 63 / 27
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 6.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 23 | 43.4% | 7 | 30 | 16 | 0.13% | 7.0% |
| BUY @ 2nd Alert (retest1) | 53 | 23 | 43.4% | 7 | 30 | 16 | 0.13% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 15 | 31.2% | 4 | 33 | 11 | -0.02% | -0.9% |
| SELL @ 2nd Alert (retest1) | 48 | 15 | 31.2% | 4 | 33 | 11 | -0.02% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 101 | 38 | 37.6% | 11 | 63 | 27 | 0.06% | 6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:00:00 | 998.00 | 1004.90 | 0.00 | ORB-short ORB[1006.10,1017.60] vol=5.8x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:05:00 | 992.88 | 1001.30 | 0.00 | T1 1.5R @ 992.88 |
| Stop hit — per-position SL triggered | 2024-05-15 10:10:00 | 998.00 | 1000.83 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 998.40 | 1002.66 | 0.00 | ORB-short ORB[1002.00,1011.95] vol=1.5x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-05-16 11:30:00 | 1001.30 | 1002.56 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:35:00 | 1041.40 | 1033.25 | 0.00 | ORB-long ORB[1019.55,1027.70] vol=3.5x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-05-18 09:40:00 | 1037.77 | 1034.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:55:00 | 1165.95 | 1171.76 | 0.00 | ORB-short ORB[1169.05,1180.20] vol=4.1x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:25:00 | 1161.24 | 1170.42 | 0.00 | T1 1.5R @ 1161.24 |
| Target hit | 2024-05-30 15:20:00 | 1158.75 | 1162.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-06-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:40:00 | 1204.75 | 1193.88 | 0.00 | ORB-long ORB[1190.55,1204.70] vol=1.6x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 13:30:00 | 1211.59 | 1200.83 | 0.00 | T1 1.5R @ 1211.59 |
| Stop hit — per-position SL triggered | 2024-06-10 14:35:00 | 1204.75 | 1202.13 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:40:00 | 1186.75 | 1191.58 | 0.00 | ORB-short ORB[1186.80,1204.00] vol=1.6x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-06-12 11:00:00 | 1190.33 | 1190.58 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:45:00 | 1245.00 | 1240.35 | 0.00 | ORB-long ORB[1232.25,1244.00] vol=1.7x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:55:00 | 1250.81 | 1242.28 | 0.00 | T1 1.5R @ 1250.81 |
| Stop hit — per-position SL triggered | 2024-06-18 10:40:00 | 1245.00 | 1249.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 1252.20 | 1246.89 | 0.00 | ORB-long ORB[1241.25,1247.00] vol=2.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:50:00 | 1256.46 | 1249.95 | 0.00 | T1 1.5R @ 1256.46 |
| Stop hit — per-position SL triggered | 2024-06-21 10:40:00 | 1252.20 | 1253.23 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 11:05:00 | 1217.15 | 1224.87 | 0.00 | ORB-short ORB[1222.05,1231.95] vol=2.8x ATR=3.15 |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 1220.30 | 1223.50 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1214.20 | 1220.59 | 0.00 | ORB-short ORB[1215.10,1231.85] vol=2.2x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-06-25 10:25:00 | 1217.68 | 1219.44 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 1223.00 | 1213.44 | 0.00 | ORB-long ORB[1200.10,1212.80] vol=2.2x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:15:00 | 1228.51 | 1214.70 | 0.00 | T1 1.5R @ 1228.51 |
| Stop hit — per-position SL triggered | 2024-06-26 15:00:00 | 1223.00 | 1224.31 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:40:00 | 1241.00 | 1237.41 | 0.00 | ORB-long ORB[1232.10,1240.85] vol=1.9x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 13:30:00 | 1246.79 | 1242.16 | 0.00 | T1 1.5R @ 1246.79 |
| Target hit | 2024-07-01 15:20:00 | 1264.05 | 1255.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:35:00 | 1277.00 | 1284.54 | 0.00 | ORB-short ORB[1280.00,1291.95] vol=2.5x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-07-04 10:55:00 | 1281.10 | 1282.85 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:30:00 | 1354.20 | 1357.20 | 0.00 | ORB-short ORB[1355.05,1364.95] vol=2.4x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-07-09 10:45:00 | 1359.41 | 1357.31 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 1371.00 | 1378.09 | 0.00 | ORB-short ORB[1380.60,1396.00] vol=1.6x ATR=5.41 |
| Stop hit — per-position SL triggered | 2024-07-12 10:45:00 | 1376.41 | 1375.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 1395.75 | 1390.80 | 0.00 | ORB-long ORB[1382.50,1394.00] vol=1.6x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:40:00 | 1404.39 | 1392.84 | 0.00 | T1 1.5R @ 1404.39 |
| Stop hit — per-position SL triggered | 2024-07-15 09:50:00 | 1395.75 | 1393.57 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:10:00 | 1441.90 | 1432.35 | 0.00 | ORB-long ORB[1414.00,1429.55] vol=2.2x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-07-24 10:30:00 | 1435.91 | 1434.45 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:50:00 | 1409.95 | 1414.67 | 0.00 | ORB-short ORB[1413.55,1425.65] vol=2.3x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-07-25 11:05:00 | 1413.81 | 1414.30 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:45:00 | 1459.00 | 1452.21 | 0.00 | ORB-long ORB[1434.50,1455.00] vol=4.1x ATR=5.12 |
| Stop hit — per-position SL triggered | 2024-07-31 11:00:00 | 1453.88 | 1456.71 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:00:00 | 1448.90 | 1462.39 | 0.00 | ORB-short ORB[1461.10,1479.40] vol=1.8x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:00:00 | 1442.84 | 1459.29 | 0.00 | T1 1.5R @ 1442.84 |
| Stop hit — per-position SL triggered | 2024-08-01 13:00:00 | 1448.90 | 1453.25 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:30:00 | 1472.00 | 1466.08 | 0.00 | ORB-long ORB[1458.50,1471.00] vol=1.6x ATR=4.67 |
| Stop hit — per-position SL triggered | 2024-08-08 10:45:00 | 1467.33 | 1466.31 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:30:00 | 1494.65 | 1483.15 | 0.00 | ORB-long ORB[1472.30,1490.20] vol=2.6x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:40:00 | 1501.49 | 1488.09 | 0.00 | T1 1.5R @ 1501.49 |
| Stop hit — per-position SL triggered | 2024-08-12 09:50:00 | 1494.65 | 1490.07 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:10:00 | 1618.95 | 1630.03 | 0.00 | ORB-short ORB[1621.75,1642.55] vol=1.6x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:30:00 | 1611.86 | 1629.03 | 0.00 | T1 1.5R @ 1611.86 |
| Stop hit — per-position SL triggered | 2024-08-20 12:00:00 | 1618.95 | 1627.61 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:45:00 | 1718.85 | 1712.99 | 0.00 | ORB-long ORB[1683.15,1706.95] vol=3.0x ATR=6.11 |
| Stop hit — per-position SL triggered | 2024-08-30 10:55:00 | 1712.74 | 1713.18 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 1710.20 | 1727.26 | 0.00 | ORB-short ORB[1733.15,1750.50] vol=3.2x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:25:00 | 1702.64 | 1725.23 | 0.00 | T1 1.5R @ 1702.64 |
| Target hit | 2024-09-02 15:20:00 | 1690.20 | 1705.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 1685.25 | 1673.34 | 0.00 | ORB-long ORB[1666.50,1682.00] vol=3.5x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:35:00 | 1693.83 | 1674.28 | 0.00 | T1 1.5R @ 1693.83 |
| Stop hit — per-position SL triggered | 2024-09-04 11:20:00 | 1685.25 | 1680.34 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 1689.50 | 1702.36 | 0.00 | ORB-short ORB[1700.30,1724.00] vol=5.5x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 1696.35 | 1701.89 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1748.95 | 1742.40 | 0.00 | ORB-long ORB[1729.95,1746.00] vol=1.8x ATR=5.39 |
| Stop hit — per-position SL triggered | 2024-09-12 09:45:00 | 1743.56 | 1745.91 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:05:00 | 1747.25 | 1749.82 | 0.00 | ORB-short ORB[1750.35,1769.60] vol=1.6x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-09-16 10:55:00 | 1751.93 | 1749.14 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:20:00 | 1677.55 | 1692.69 | 0.00 | ORB-short ORB[1695.25,1719.25] vol=2.9x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-09-18 10:25:00 | 1683.39 | 1692.20 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 1688.70 | 1684.70 | 0.00 | ORB-long ORB[1671.00,1685.35] vol=2.5x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-09-27 11:10:00 | 1684.43 | 1684.60 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 11:05:00 | 1669.15 | 1683.61 | 0.00 | ORB-short ORB[1689.05,1701.95] vol=2.0x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-09-30 11:25:00 | 1674.65 | 1682.19 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:50:00 | 1652.75 | 1663.45 | 0.00 | ORB-short ORB[1662.80,1675.55] vol=2.2x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 1658.01 | 1662.75 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 11:10:00 | 1690.00 | 1674.05 | 0.00 | ORB-long ORB[1629.95,1654.80] vol=3.4x ATR=7.18 |
| Stop hit — per-position SL triggered | 2024-10-04 13:30:00 | 1682.82 | 1682.36 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:55:00 | 1664.40 | 1672.40 | 0.00 | ORB-short ORB[1665.25,1682.90] vol=1.5x ATR=7.30 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 1671.70 | 1671.78 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:45:00 | 1697.70 | 1684.82 | 0.00 | ORB-long ORB[1663.70,1685.05] vol=1.6x ATR=7.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:05:00 | 1708.64 | 1690.55 | 0.00 | T1 1.5R @ 1708.64 |
| Target hit | 2024-10-08 12:05:00 | 1709.30 | 1709.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2024-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:35:00 | 1752.85 | 1745.64 | 0.00 | ORB-long ORB[1736.00,1751.40] vol=2.0x ATR=6.35 |
| Stop hit — per-position SL triggered | 2024-10-09 09:45:00 | 1746.50 | 1747.16 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:40:00 | 1808.00 | 1799.05 | 0.00 | ORB-long ORB[1784.25,1805.00] vol=1.7x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-10-14 09:45:00 | 1802.16 | 1799.34 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:40:00 | 1772.00 | 1785.66 | 0.00 | ORB-short ORB[1787.35,1809.90] vol=3.3x ATR=6.99 |
| Stop hit — per-position SL triggered | 2024-10-16 10:10:00 | 1778.99 | 1781.60 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:50:00 | 1769.35 | 1774.10 | 0.00 | ORB-short ORB[1771.05,1789.75] vol=1.6x ATR=6.10 |
| Stop hit — per-position SL triggered | 2024-10-17 09:55:00 | 1775.45 | 1773.87 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:45:00 | 1735.75 | 1725.49 | 0.00 | ORB-long ORB[1716.05,1727.60] vol=1.9x ATR=6.38 |
| Stop hit — per-position SL triggered | 2024-10-22 10:10:00 | 1729.37 | 1730.26 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-11-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:30:00 | 1680.45 | 1668.43 | 0.00 | ORB-long ORB[1651.00,1674.00] vol=1.7x ATR=6.81 |
| Stop hit — per-position SL triggered | 2024-11-11 11:45:00 | 1673.64 | 1671.19 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:55:00 | 1509.90 | 1489.87 | 0.00 | ORB-long ORB[1473.35,1494.90] vol=3.4x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-11-19 14:20:00 | 1503.32 | 1499.29 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 1508.35 | 1519.56 | 0.00 | ORB-short ORB[1515.15,1533.85] vol=2.1x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 1502.39 | 1516.50 | 0.00 | T1 1.5R @ 1502.39 |
| Target hit | 2024-11-28 15:20:00 | 1495.90 | 1505.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2024-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:50:00 | 1516.50 | 1509.18 | 0.00 | ORB-long ORB[1495.85,1512.95] vol=2.0x ATR=4.66 |
| Stop hit — per-position SL triggered | 2024-11-29 10:05:00 | 1511.84 | 1510.86 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 1541.75 | 1536.50 | 0.00 | ORB-long ORB[1519.95,1540.35] vol=1.7x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:35:00 | 1548.83 | 1539.10 | 0.00 | T1 1.5R @ 1548.83 |
| Target hit | 2024-12-02 10:35:00 | 1549.40 | 1550.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2024-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:25:00 | 1536.70 | 1542.43 | 0.00 | ORB-short ORB[1540.00,1552.00] vol=1.9x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 10:55:00 | 1530.74 | 1539.36 | 0.00 | T1 1.5R @ 1530.74 |
| Target hit | 2024-12-05 12:15:00 | 1533.50 | 1530.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 1529.60 | 1537.02 | 0.00 | ORB-short ORB[1532.05,1550.00] vol=2.5x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 1533.99 | 1536.65 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1530.65 | 1541.07 | 0.00 | ORB-short ORB[1540.45,1554.90] vol=2.5x ATR=4.70 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 1535.35 | 1538.60 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:35:00 | 1536.00 | 1527.40 | 0.00 | ORB-long ORB[1507.95,1529.85] vol=2.0x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-12-18 10:00:00 | 1530.16 | 1531.19 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 1517.35 | 1509.14 | 0.00 | ORB-long ORB[1498.20,1515.75] vol=1.9x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:00:00 | 1526.45 | 1512.36 | 0.00 | T1 1.5R @ 1526.45 |
| Target hit | 2024-12-19 15:20:00 | 1544.00 | 1532.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:25:00 | 1596.65 | 1603.77 | 0.00 | ORB-short ORB[1607.00,1619.00] vol=2.1x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-01-02 10:45:00 | 1600.90 | 1603.11 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 1619.05 | 1630.75 | 0.00 | ORB-short ORB[1628.70,1645.85] vol=2.3x ATR=4.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 1611.79 | 1627.17 | 0.00 | T1 1.5R @ 1611.79 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1619.05 | 1625.37 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:00:00 | 1623.55 | 1628.45 | 0.00 | ORB-short ORB[1623.65,1638.40] vol=1.7x ATR=5.54 |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 1629.09 | 1628.24 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:40:00 | 1629.10 | 1641.62 | 0.00 | ORB-short ORB[1638.10,1657.70] vol=1.6x ATR=5.46 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 1634.56 | 1639.86 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:55:00 | 1572.60 | 1585.45 | 0.00 | ORB-short ORB[1580.15,1596.55] vol=1.5x ATR=5.90 |
| Stop hit — per-position SL triggered | 2025-01-10 11:10:00 | 1578.50 | 1579.44 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 11:15:00 | 1490.10 | 1500.34 | 0.00 | ORB-short ORB[1502.70,1518.40] vol=2.9x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 11:55:00 | 1482.69 | 1497.82 | 0.00 | T1 1.5R @ 1482.69 |
| Stop hit — per-position SL triggered | 2025-01-14 12:00:00 | 1490.10 | 1497.38 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:40:00 | 1456.40 | 1447.71 | 0.00 | ORB-long ORB[1437.95,1454.35] vol=2.5x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:10:00 | 1464.33 | 1452.39 | 0.00 | T1 1.5R @ 1464.33 |
| Stop hit — per-position SL triggered | 2025-01-17 10:40:00 | 1456.40 | 1452.95 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:55:00 | 1480.15 | 1457.59 | 0.00 | ORB-long ORB[1449.60,1463.75] vol=1.9x ATR=4.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:05:00 | 1487.62 | 1460.52 | 0.00 | T1 1.5R @ 1487.62 |
| Target hit | 2025-01-20 15:20:00 | 1503.50 | 1482.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 1484.05 | 1494.63 | 0.00 | ORB-short ORB[1491.00,1508.75] vol=1.5x ATR=4.87 |
| Stop hit — per-position SL triggered | 2025-01-22 09:55:00 | 1488.92 | 1493.67 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 1508.45 | 1500.19 | 0.00 | ORB-long ORB[1485.55,1505.65] vol=1.7x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:15:00 | 1515.94 | 1506.08 | 0.00 | T1 1.5R @ 1515.94 |
| Target hit | 2025-01-23 11:50:00 | 1510.00 | 1510.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1465.10 | 1459.78 | 0.00 | ORB-long ORB[1447.95,1461.65] vol=1.7x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-01-30 09:35:00 | 1461.34 | 1460.58 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 1471.65 | 1463.46 | 0.00 | ORB-long ORB[1447.55,1466.10] vol=2.4x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 1465.87 | 1466.97 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:45:00 | 1437.45 | 1441.58 | 0.00 | ORB-short ORB[1449.00,1458.05] vol=1.8x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 1441.12 | 1440.52 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:40:00 | 1521.05 | 1503.67 | 0.00 | ORB-long ORB[1485.10,1506.80] vol=4.7x ATR=7.00 |
| Stop hit — per-position SL triggered | 2025-02-07 10:45:00 | 1514.05 | 1505.00 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 11:15:00 | 1318.45 | 1312.91 | 0.00 | ORB-long ORB[1288.55,1306.45] vol=1.8x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-03-04 11:20:00 | 1313.77 | 1313.09 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-03-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 09:40:00 | 1487.75 | 1494.32 | 0.00 | ORB-short ORB[1498.00,1513.55] vol=2.0x ATR=7.03 |
| Stop hit — per-position SL triggered | 2025-03-24 09:55:00 | 1494.78 | 1492.09 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 1468.70 | 1475.24 | 0.00 | ORB-short ORB[1474.30,1489.75] vol=3.2x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:50:00 | 1460.49 | 1471.20 | 0.00 | T1 1.5R @ 1460.49 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 1468.70 | 1471.13 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:15:00 | 1518.00 | 1505.22 | 0.00 | ORB-long ORB[1492.55,1512.70] vol=2.1x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 10:45:00 | 1525.80 | 1513.80 | 0.00 | T1 1.5R @ 1525.80 |
| Stop hit — per-position SL triggered | 2025-04-02 11:40:00 | 1518.00 | 1516.23 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:05:00 | 1360.90 | 1368.92 | 0.00 | ORB-short ORB[1364.60,1380.00] vol=1.5x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:40:00 | 1354.72 | 1367.58 | 0.00 | T1 1.5R @ 1354.72 |
| Stop hit — per-position SL triggered | 2025-04-16 11:50:00 | 1360.90 | 1367.28 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:30:00 | 1348.50 | 1358.59 | 0.00 | ORB-short ORB[1360.00,1376.80] vol=2.3x ATR=4.63 |
| Stop hit — per-position SL triggered | 2025-04-17 10:45:00 | 1353.13 | 1357.72 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:10:00 | 1386.70 | 1373.47 | 0.00 | ORB-long ORB[1359.60,1377.50] vol=1.7x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:40:00 | 1393.42 | 1376.43 | 0.00 | T1 1.5R @ 1393.42 |
| Target hit | 2025-04-22 12:10:00 | 1387.00 | 1388.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2025-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:30:00 | 1405.10 | 1396.82 | 0.00 | ORB-long ORB[1388.40,1404.70] vol=2.2x ATR=6.22 |
| Stop hit — per-position SL triggered | 2025-04-23 10:45:00 | 1398.88 | 1398.09 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:55:00 | 1399.60 | 1397.04 | 0.00 | ORB-long ORB[1381.10,1396.80] vol=6.8x ATR=5.24 |
| Stop hit — per-position SL triggered | 2025-04-30 11:05:00 | 1394.36 | 1398.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:00:00 | 998.00 | 2024-05-15 10:05:00 | 992.88 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-15 10:00:00 | 998.00 | 2024-05-15 10:10:00 | 998.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:15:00 | 998.40 | 2024-05-16 11:30:00 | 1001.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-18 09:35:00 | 1041.40 | 2024-05-18 09:40:00 | 1037.77 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-30 10:55:00 | 1165.95 | 2024-05-30 11:25:00 | 1161.24 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-30 10:55:00 | 1165.95 | 2024-05-30 15:20:00 | 1158.75 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2024-06-10 10:40:00 | 1204.75 | 2024-06-10 13:30:00 | 1211.59 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-10 10:40:00 | 1204.75 | 2024-06-10 14:35:00 | 1204.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:40:00 | 1186.75 | 2024-06-12 11:00:00 | 1190.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-18 09:45:00 | 1245.00 | 2024-06-18 09:55:00 | 1250.81 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-18 09:45:00 | 1245.00 | 2024-06-18 10:40:00 | 1245.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:35:00 | 1252.20 | 2024-06-21 09:50:00 | 1256.46 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-21 09:35:00 | 1252.20 | 2024-06-21 10:40:00 | 1252.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-24 11:05:00 | 1217.15 | 2024-06-24 11:15:00 | 1220.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-25 10:15:00 | 1214.20 | 2024-06-25 10:25:00 | 1217.68 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1223.00 | 2024-06-26 11:15:00 | 1228.51 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1223.00 | 2024-06-26 15:00:00 | 1223.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:40:00 | 1241.00 | 2024-07-01 13:30:00 | 1246.79 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-01 09:40:00 | 1241.00 | 2024-07-01 15:20:00 | 1264.05 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2024-07-04 10:35:00 | 1277.00 | 2024-07-04 10:55:00 | 1281.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-09 10:30:00 | 1354.20 | 2024-07-09 10:45:00 | 1359.41 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-12 09:55:00 | 1371.00 | 2024-07-12 10:45:00 | 1376.41 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-15 09:35:00 | 1395.75 | 2024-07-15 09:40:00 | 1404.39 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-15 09:35:00 | 1395.75 | 2024-07-15 09:50:00 | 1395.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:10:00 | 1441.90 | 2024-07-24 10:30:00 | 1435.91 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-25 10:50:00 | 1409.95 | 2024-07-25 11:05:00 | 1413.81 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-31 09:45:00 | 1459.00 | 2024-07-31 11:00:00 | 1453.88 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-01 11:00:00 | 1448.90 | 2024-08-01 12:00:00 | 1442.84 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-01 11:00:00 | 1448.90 | 2024-08-01 13:00:00 | 1448.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:30:00 | 1472.00 | 2024-08-08 10:45:00 | 1467.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-12 09:30:00 | 1494.65 | 2024-08-12 09:40:00 | 1501.49 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-12 09:30:00 | 1494.65 | 2024-08-12 09:50:00 | 1494.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 11:10:00 | 1618.95 | 2024-08-20 11:30:00 | 1611.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-20 11:10:00 | 1618.95 | 2024-08-20 12:00:00 | 1618.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:45:00 | 1718.85 | 2024-08-30 10:55:00 | 1712.74 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1710.20 | 2024-09-02 11:25:00 | 1702.64 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1710.20 | 2024-09-02 15:20:00 | 1690.20 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2024-09-04 09:30:00 | 1685.25 | 2024-09-04 09:35:00 | 1693.83 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-04 09:30:00 | 1685.25 | 2024-09-04 11:20:00 | 1685.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-09 09:30:00 | 1689.50 | 2024-09-09 09:35:00 | 1696.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-09-12 09:30:00 | 1748.95 | 2024-09-12 09:45:00 | 1743.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-16 10:05:00 | 1747.25 | 2024-09-16 10:55:00 | 1751.93 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-18 10:20:00 | 1677.55 | 2024-09-18 10:25:00 | 1683.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-27 11:00:00 | 1688.70 | 2024-09-27 11:10:00 | 1684.43 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-30 11:05:00 | 1669.15 | 2024-09-30 11:25:00 | 1674.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-01 10:50:00 | 1652.75 | 2024-10-01 10:55:00 | 1658.01 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-04 11:10:00 | 1690.00 | 2024-10-04 13:30:00 | 1682.82 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-07 10:55:00 | 1664.40 | 2024-10-07 11:05:00 | 1671.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-08 09:45:00 | 1697.70 | 2024-10-08 10:05:00 | 1708.64 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-08 09:45:00 | 1697.70 | 2024-10-08 12:05:00 | 1709.30 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-10-09 09:35:00 | 1752.85 | 2024-10-09 09:45:00 | 1746.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-14 09:40:00 | 1808.00 | 2024-10-14 09:45:00 | 1802.16 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-16 09:40:00 | 1772.00 | 2024-10-16 10:10:00 | 1778.99 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-17 09:50:00 | 1769.35 | 2024-10-17 09:55:00 | 1775.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-22 09:45:00 | 1735.75 | 2024-10-22 10:10:00 | 1729.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-11 10:30:00 | 1680.45 | 2024-11-11 11:45:00 | 1673.64 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-19 10:55:00 | 1509.90 | 2024-11-19 14:20:00 | 1503.32 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1508.35 | 2024-11-28 10:45:00 | 1502.39 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1508.35 | 2024-11-28 15:20:00 | 1495.90 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-11-29 09:50:00 | 1516.50 | 2024-11-29 10:05:00 | 1511.84 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1541.75 | 2024-12-02 09:35:00 | 1548.83 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1541.75 | 2024-12-02 10:35:00 | 1549.40 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-05 10:25:00 | 1536.70 | 2024-12-05 10:55:00 | 1530.74 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-05 10:25:00 | 1536.70 | 2024-12-05 12:15:00 | 1533.50 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2024-12-12 09:35:00 | 1529.60 | 2024-12-12 09:50:00 | 1533.99 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-17 09:30:00 | 1530.65 | 2024-12-17 09:35:00 | 1535.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-18 09:35:00 | 1536.00 | 2024-12-18 10:00:00 | 1530.16 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-19 09:40:00 | 1517.35 | 2024-12-19 10:00:00 | 1526.45 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-12-19 09:40:00 | 1517.35 | 2024-12-19 15:20:00 | 1544.00 | TARGET_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2025-01-02 10:25:00 | 1596.65 | 2025-01-02 10:45:00 | 1600.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1619.05 | 2025-01-06 11:10:00 | 1611.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1619.05 | 2025-01-06 11:30:00 | 1619.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 10:00:00 | 1623.55 | 2025-01-07 10:15:00 | 1629.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-08 09:40:00 | 1629.10 | 2025-01-08 09:50:00 | 1634.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-10 09:55:00 | 1572.60 | 2025-01-10 11:10:00 | 1578.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-14 11:15:00 | 1490.10 | 2025-01-14 11:55:00 | 1482.69 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-14 11:15:00 | 1490.10 | 2025-01-14 12:00:00 | 1490.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:40:00 | 1456.40 | 2025-01-17 10:10:00 | 1464.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-17 09:40:00 | 1456.40 | 2025-01-17 10:40:00 | 1456.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-20 10:55:00 | 1480.15 | 2025-01-20 11:05:00 | 1487.62 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-20 10:55:00 | 1480.15 | 2025-01-20 15:20:00 | 1503.50 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2025-01-22 09:50:00 | 1484.05 | 2025-01-22 09:55:00 | 1488.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-23 09:35:00 | 1508.45 | 2025-01-23 10:15:00 | 1515.94 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-23 09:35:00 | 1508.45 | 2025-01-23 11:50:00 | 1510.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1465.10 | 2025-01-30 09:35:00 | 1461.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-31 09:40:00 | 1471.65 | 2025-01-31 10:05:00 | 1465.87 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-01 10:45:00 | 1437.45 | 2025-02-01 11:00:00 | 1441.12 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-07 10:40:00 | 1521.05 | 2025-02-07 10:45:00 | 1514.05 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-04 11:15:00 | 1318.45 | 2025-03-04 11:20:00 | 1313.77 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-24 09:40:00 | 1487.75 | 2025-03-24 09:55:00 | 1494.78 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-26 09:35:00 | 1468.70 | 2025-03-26 09:50:00 | 1460.49 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-03-26 09:35:00 | 1468.70 | 2025-03-26 09:55:00 | 1468.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 10:15:00 | 1518.00 | 2025-04-02 10:45:00 | 1525.80 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-04-02 10:15:00 | 1518.00 | 2025-04-02 11:40:00 | 1518.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-16 11:05:00 | 1360.90 | 2025-04-16 11:40:00 | 1354.72 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-16 11:05:00 | 1360.90 | 2025-04-16 11:50:00 | 1360.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-17 10:30:00 | 1348.50 | 2025-04-17 10:45:00 | 1353.13 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-22 10:10:00 | 1386.70 | 2025-04-22 10:40:00 | 1393.42 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-22 10:10:00 | 1386.70 | 2025-04-22 12:10:00 | 1387.00 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2025-04-23 10:30:00 | 1405.10 | 2025-04-23 10:45:00 | 1398.88 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-30 09:55:00 | 1399.60 | 2025-04-30 11:05:00 | 1394.36 | STOP_HIT | 1.00 | -0.37% |
