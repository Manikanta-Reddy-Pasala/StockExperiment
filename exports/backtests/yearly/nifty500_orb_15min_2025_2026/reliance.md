# Reliance Industries Ltd. (RELIANCE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1436.00
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
| ENTRY1 | 83 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 9 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 74
- **Target hits / Stop hits / Partials:** 9 / 74 / 25
- **Avg / median % per leg:** 0.04% / -0.12%
- **Sum % (uncompounded):** 4.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 21 | 33.3% | 5 | 42 | 16 | 0.03% | 1.8% |
| BUY @ 2nd Alert (retest1) | 63 | 21 | 33.3% | 5 | 42 | 16 | 0.03% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 13 | 28.9% | 4 | 32 | 9 | 0.06% | 2.7% |
| SELL @ 2nd Alert (retest1) | 45 | 13 | 28.9% | 4 | 32 | 9 | 0.06% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 34 | 31.5% | 9 | 74 | 25 | 0.04% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1415.60 | 1417.91 | 0.00 | ORB-short ORB[1417.10,1426.00] vol=2.4x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-05-15 10:00:00 | 1418.47 | 1417.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 1432.20 | 1428.11 | 0.00 | ORB-long ORB[1418.10,1426.50] vol=1.7x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-05-21 11:05:00 | 1429.79 | 1428.28 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 1466.60 | 1455.75 | 0.00 | ORB-long ORB[1443.10,1457.70] vol=2.2x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-06-11 09:35:00 | 1463.03 | 1457.23 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:30:00 | 1434.80 | 1430.16 | 0.00 | ORB-long ORB[1424.40,1433.00] vol=1.9x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:05:00 | 1439.23 | 1432.69 | 0.00 | T1 1.5R @ 1439.23 |
| Stop hit — per-position SL triggered | 2025-06-16 12:00:00 | 1434.80 | 1433.89 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:05:00 | 1431.20 | 1435.96 | 0.00 | ORB-short ORB[1432.50,1448.00] vol=3.0x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:25:00 | 1427.49 | 1434.35 | 0.00 | T1 1.5R @ 1427.49 |
| Stop hit — per-position SL triggered | 2025-06-17 11:25:00 | 1431.20 | 1432.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:55:00 | 1470.40 | 1466.31 | 0.00 | ORB-long ORB[1460.50,1467.50] vol=4.0x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1467.85 | 1467.32 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:55:00 | 1480.80 | 1473.79 | 0.00 | ORB-long ORB[1465.10,1474.70] vol=1.5x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:00:00 | 1484.79 | 1475.74 | 0.00 | T1 1.5R @ 1484.79 |
| Stop hit — per-position SL triggered | 2025-06-26 10:40:00 | 1480.80 | 1478.98 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:10:00 | 1510.00 | 1503.52 | 0.00 | ORB-long ORB[1497.30,1509.00] vol=2.7x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:30:00 | 1514.06 | 1504.57 | 0.00 | T1 1.5R @ 1514.06 |
| Target hit | 2025-06-27 15:20:00 | 1512.40 | 1511.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1517.90 | 1522.53 | 0.00 | ORB-short ORB[1521.70,1530.00] vol=1.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-07-02 11:25:00 | 1519.92 | 1522.26 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:15:00 | 1528.90 | 1522.55 | 0.00 | ORB-long ORB[1513.00,1524.40] vol=2.7x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-07-03 11:45:00 | 1526.48 | 1523.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:50:00 | 1526.40 | 1523.04 | 0.00 | ORB-long ORB[1517.80,1524.90] vol=2.5x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-07-04 11:10:00 | 1523.75 | 1523.20 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:55:00 | 1538.30 | 1535.09 | 0.00 | ORB-long ORB[1525.00,1535.80] vol=2.7x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-07-07 11:00:00 | 1535.95 | 1535.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 1533.50 | 1536.59 | 0.00 | ORB-short ORB[1535.00,1541.20] vol=2.4x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-07-08 10:50:00 | 1535.35 | 1536.66 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:55:00 | 1493.10 | 1501.74 | 0.00 | ORB-short ORB[1506.50,1515.00] vol=1.5x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 1495.39 | 1500.55 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:00:00 | 1497.70 | 1494.59 | 0.00 | ORB-long ORB[1487.80,1495.20] vol=2.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-07-14 10:05:00 | 1495.34 | 1494.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:00:00 | 1494.80 | 1489.41 | 0.00 | ORB-long ORB[1486.00,1493.60] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1492.41 | 1490.48 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1481.00 | 1481.50 | 0.00 | ORB-short ORB[1483.10,1489.60] vol=2.3x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-07-17 10:55:00 | 1482.74 | 1481.50 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:00:00 | 1471.50 | 1477.35 | 0.00 | ORB-short ORB[1476.00,1484.80] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-07-18 12:35:00 | 1473.71 | 1474.93 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:05:00 | 1412.50 | 1417.13 | 0.00 | ORB-short ORB[1414.50,1423.00] vol=1.5x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.88 | 1414.62 | 0.00 | T1 1.5R @ 1409.88 |
| Target hit | 2025-07-24 15:20:00 | 1402.30 | 1407.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:55:00 | 1405.60 | 1411.67 | 0.00 | ORB-short ORB[1411.10,1423.30] vol=1.5x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-07-30 11:10:00 | 1408.54 | 1411.17 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 1393.80 | 1388.94 | 0.00 | ORB-long ORB[1382.20,1391.50] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-07-31 12:10:00 | 1390.66 | 1391.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:15:00 | 1396.80 | 1396.55 | 0.00 | ORB-long ORB[1384.30,1393.00] vol=2.3x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:05:00 | 1401.43 | 1396.86 | 0.00 | T1 1.5R @ 1401.43 |
| Stop hit — per-position SL triggered | 2025-08-01 15:00:00 | 1396.80 | 1398.36 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:30:00 | 1379.30 | 1383.89 | 0.00 | ORB-short ORB[1383.80,1390.70] vol=2.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-08-07 12:10:00 | 1381.37 | 1381.30 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:45:00 | 1373.30 | 1384.71 | 0.00 | ORB-short ORB[1384.90,1397.20] vol=2.4x ATR=3.27 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 1376.57 | 1381.11 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:15:00 | 1420.90 | 1419.90 | 0.00 | ORB-long ORB[1411.00,1417.30] vol=1.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-08-20 11:30:00 | 1418.71 | 1419.90 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:40:00 | 1430.00 | 1427.75 | 0.00 | ORB-long ORB[1420.30,1428.90] vol=1.9x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-08-21 10:45:00 | 1427.50 | 1427.81 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:10:00 | 1414.40 | 1417.57 | 0.00 | ORB-short ORB[1415.00,1423.40] vol=5.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-08-22 11:20:00 | 1416.39 | 1417.35 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:45:00 | 1401.40 | 1406.01 | 0.00 | ORB-short ORB[1404.00,1410.00] vol=1.7x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 10:40:00 | 1397.28 | 1404.23 | 0.00 | T1 1.5R @ 1397.28 |
| Target hit | 2025-08-26 15:20:00 | 1383.00 | 1391.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:50:00 | 1360.60 | 1367.39 | 0.00 | ORB-short ORB[1363.00,1371.80] vol=2.0x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-09-05 11:00:00 | 1363.38 | 1366.63 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:10:00 | 1373.80 | 1375.44 | 0.00 | ORB-short ORB[1375.70,1381.20] vol=2.6x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 12:00:00 | 1371.00 | 1374.81 | 0.00 | T1 1.5R @ 1371.00 |
| Stop hit — per-position SL triggered | 2025-09-09 12:30:00 | 1373.80 | 1374.28 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:45:00 | 1383.50 | 1381.31 | 0.00 | ORB-long ORB[1375.00,1383.00] vol=1.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-11 12:20:00 | 1381.98 | 1382.31 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 1393.10 | 1389.65 | 0.00 | ORB-long ORB[1380.50,1388.20] vol=2.8x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:25:00 | 1396.09 | 1390.71 | 0.00 | T1 1.5R @ 1396.09 |
| Stop hit — per-position SL triggered | 2025-09-12 12:10:00 | 1393.10 | 1391.21 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 10:45:00 | 1391.60 | 1393.06 | 0.00 | ORB-short ORB[1392.10,1396.90] vol=2.2x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-09-15 11:25:00 | 1393.06 | 1392.93 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:05:00 | 1405.50 | 1408.00 | 0.00 | ORB-short ORB[1408.10,1417.00] vol=1.9x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-09-19 11:35:00 | 1407.19 | 1407.82 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:55:00 | 1380.60 | 1385.94 | 0.00 | ORB-short ORB[1387.00,1393.00] vol=5.4x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-09-23 11:00:00 | 1383.02 | 1385.83 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 10:45:00 | 1375.10 | 1370.37 | 0.00 | ORB-long ORB[1367.20,1372.10] vol=1.7x ATR=2.15 |
| Stop hit — per-position SL triggered | 2025-09-26 10:55:00 | 1372.95 | 1370.79 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 1385.00 | 1383.75 | 0.00 | ORB-long ORB[1379.00,1384.70] vol=1.6x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:10:00 | 1387.86 | 1384.11 | 0.00 | T1 1.5R @ 1387.86 |
| Stop hit — per-position SL triggered | 2025-09-29 11:30:00 | 1385.00 | 1384.28 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 1359.50 | 1363.74 | 0.00 | ORB-short ORB[1363.10,1371.60] vol=1.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-10-03 12:05:00 | 1361.56 | 1362.50 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:20:00 | 1389.20 | 1384.58 | 0.00 | ORB-long ORB[1375.90,1383.70] vol=2.4x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-10-07 10:25:00 | 1387.02 | 1384.69 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:10:00 | 1379.50 | 1375.52 | 0.00 | ORB-long ORB[1369.10,1379.30] vol=2.5x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 1377.51 | 1375.67 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:55:00 | 1414.80 | 1407.31 | 0.00 | ORB-long ORB[1399.60,1410.80] vol=2.2x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-10-17 11:40:00 | 1412.29 | 1408.97 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 1479.30 | 1475.73 | 0.00 | ORB-long ORB[1470.40,1478.90] vol=1.6x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-10-23 09:35:00 | 1475.54 | 1475.87 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1472.40 | 1466.63 | 0.00 | ORB-long ORB[1458.00,1468.60] vol=1.7x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 09:40:00 | 1477.50 | 1470.94 | 0.00 | T1 1.5R @ 1477.50 |
| Target hit | 2025-10-27 13:15:00 | 1478.70 | 1479.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 1479.60 | 1485.14 | 0.00 | ORB-short ORB[1481.20,1489.00] vol=1.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 1482.18 | 1484.52 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:55:00 | 1493.00 | 1494.00 | 0.00 | ORB-short ORB[1494.00,1501.30] vol=1.7x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-10-30 11:00:00 | 1495.41 | 1494.03 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:40:00 | 1488.00 | 1490.05 | 0.00 | ORB-short ORB[1488.10,1497.50] vol=1.8x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:00:00 | 1484.08 | 1489.71 | 0.00 | T1 1.5R @ 1484.08 |
| Stop hit — per-position SL triggered | 2025-10-31 11:55:00 | 1488.00 | 1488.73 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 1498.50 | 1491.33 | 0.00 | ORB-long ORB[1476.20,1496.50] vol=1.5x ATR=4.30 |
| Stop hit — per-position SL triggered | 2025-11-06 10:10:00 | 1494.20 | 1494.48 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:15:00 | 1515.20 | 1509.08 | 0.00 | ORB-long ORB[1500.00,1513.50] vol=1.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:25:00 | 1518.66 | 1509.57 | 0.00 | T1 1.5R @ 1518.66 |
| Stop hit — per-position SL triggered | 2025-11-12 14:40:00 | 1515.20 | 1515.70 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 11:15:00 | 1516.60 | 1512.22 | 0.00 | ORB-long ORB[1506.80,1516.00] vol=1.7x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 12:55:00 | 1519.81 | 1514.85 | 0.00 | T1 1.5R @ 1519.81 |
| Stop hit — per-position SL triggered | 2025-11-13 13:55:00 | 1516.60 | 1516.10 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 1517.90 | 1514.05 | 0.00 | ORB-long ORB[1505.50,1513.50] vol=1.7x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 1515.50 | 1514.30 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1514.10 | 1517.07 | 0.00 | ORB-short ORB[1514.90,1519.50] vol=1.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-11-19 11:40:00 | 1515.88 | 1516.71 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:35:00 | 1540.10 | 1534.52 | 0.00 | ORB-long ORB[1526.70,1537.00] vol=3.2x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:10:00 | 1544.20 | 1537.10 | 0.00 | T1 1.5R @ 1544.20 |
| Target hit | 2025-11-20 15:20:00 | 1548.50 | 1543.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 1561.30 | 1552.60 | 0.00 | ORB-long ORB[1540.50,1552.50] vol=2.6x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 12:25:00 | 1565.84 | 1557.01 | 0.00 | T1 1.5R @ 1565.84 |
| Stop hit — per-position SL triggered | 2025-11-26 13:25:00 | 1561.30 | 1558.75 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 1563.60 | 1568.01 | 0.00 | ORB-short ORB[1565.90,1575.50] vol=1.5x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 13:05:00 | 1560.45 | 1566.25 | 0.00 | T1 1.5R @ 1560.45 |
| Stop hit — per-position SL triggered | 2025-11-27 15:05:00 | 1563.60 | 1563.37 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 1578.00 | 1571.24 | 0.00 | ORB-long ORB[1563.00,1573.80] vol=1.6x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 1574.24 | 1572.34 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:10:00 | 1559.90 | 1548.77 | 0.00 | ORB-long ORB[1535.50,1545.00] vol=1.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1557.84 | 1549.32 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 1543.00 | 1544.05 | 0.00 | ORB-short ORB[1545.00,1551.70] vol=1.5x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-12-16 11:20:00 | 1545.22 | 1544.04 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:00:00 | 1541.70 | 1544.65 | 0.00 | ORB-short ORB[1542.10,1550.00] vol=2.3x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-12-17 11:45:00 | 1543.91 | 1544.05 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:40:00 | 1571.90 | 1564.49 | 0.00 | ORB-long ORB[1551.00,1567.90] vol=1.6x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-12-19 11:40:00 | 1569.01 | 1566.66 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:35:00 | 1552.30 | 1549.37 | 0.00 | ORB-long ORB[1543.10,1549.40] vol=2.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-12-30 09:45:00 | 1550.04 | 1549.48 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 1552.00 | 1547.99 | 0.00 | ORB-long ORB[1541.00,1549.60] vol=2.6x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:15:00 | 1555.11 | 1549.13 | 0.00 | T1 1.5R @ 1555.11 |
| Target hit | 2025-12-31 15:20:00 | 1568.50 | 1566.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 1588.40 | 1586.92 | 0.00 | ORB-long ORB[1578.20,1587.50] vol=1.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-01-02 10:45:00 | 1586.14 | 1586.89 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 1484.80 | 1490.86 | 0.00 | ORB-short ORB[1491.10,1503.90] vol=2.1x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 1480.62 | 1489.65 | 0.00 | T1 1.5R @ 1480.62 |
| Target hit | 2026-01-08 15:20:00 | 1472.00 | 1477.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 1465.00 | 1471.56 | 0.00 | ORB-short ORB[1468.00,1485.80] vol=1.8x ATR=3.40 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 1468.40 | 1469.83 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:10:00 | 1399.90 | 1407.32 | 0.00 | ORB-short ORB[1405.60,1416.00] vol=1.6x ATR=2.51 |
| Stop hit — per-position SL triggered | 2026-01-20 11:45:00 | 1402.41 | 1406.44 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1390.90 | 1395.48 | 0.00 | ORB-short ORB[1393.70,1402.90] vol=2.3x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-01-29 10:55:00 | 1393.59 | 1392.89 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1389.20 | 1386.84 | 0.00 | ORB-long ORB[1378.50,1386.90] vol=9.7x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-01-30 11:20:00 | 1386.96 | 1386.91 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1404.80 | 1402.11 | 0.00 | ORB-long ORB[1388.00,1401.90] vol=3.4x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:05:00 | 1409.11 | 1402.63 | 0.00 | T1 1.5R @ 1409.11 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 1404.80 | 1403.70 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1459.70 | 1464.41 | 0.00 | ORB-short ORB[1464.50,1473.00] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 1457.01 | 1463.98 | 0.00 | T1 1.5R @ 1457.01 |
| Stop hit — per-position SL triggered | 2026-02-12 11:55:00 | 1459.70 | 1462.09 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1431.00 | 1433.10 | 0.00 | ORB-short ORB[1432.00,1440.50] vol=2.2x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:10:00 | 1427.95 | 1432.33 | 0.00 | T1 1.5R @ 1427.95 |
| Target hit | 2026-02-25 15:20:00 | 1398.40 | 1414.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 1404.20 | 1396.41 | 0.00 | ORB-long ORB[1381.10,1397.00] vol=1.5x ATR=3.95 |
| Stop hit — per-position SL triggered | 2026-03-12 13:05:00 | 1400.25 | 1399.02 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 1411.60 | 1407.87 | 0.00 | ORB-long ORB[1397.20,1409.70] vol=1.7x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 1408.64 | 1408.20 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:00:00 | 1427.50 | 1420.65 | 0.00 | ORB-long ORB[1414.10,1421.90] vol=1.9x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-03-25 11:35:00 | 1424.83 | 1422.38 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:40:00 | 1332.20 | 1341.29 | 0.00 | ORB-short ORB[1338.00,1350.00] vol=1.6x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-04-09 10:50:00 | 1335.22 | 1340.80 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 1347.50 | 1344.73 | 0.00 | ORB-long ORB[1331.50,1344.50] vol=1.5x ATR=3.43 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1344.07 | 1344.76 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 1335.50 | 1338.71 | 0.00 | ORB-short ORB[1340.00,1353.80] vol=2.1x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 1338.60 | 1338.04 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1357.00 | 1348.90 | 0.00 | ORB-long ORB[1340.00,1351.90] vol=2.1x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1360.46 | 1350.09 | 0.00 | T1 1.5R @ 1360.46 |
| Stop hit — per-position SL triggered | 2026-04-17 13:20:00 | 1357.00 | 1356.90 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1361.60 | 1355.43 | 0.00 | ORB-long ORB[1350.20,1358.00] vol=1.5x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:05:00 | 1365.24 | 1357.24 | 0.00 | T1 1.5R @ 1365.24 |
| Stop hit — per-position SL triggered | 2026-04-22 11:35:00 | 1361.60 | 1358.30 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 1331.60 | 1337.45 | 0.00 | ORB-short ORB[1337.30,1345.90] vol=1.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 1334.02 | 1337.34 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 1342.40 | 1327.75 | 0.00 | ORB-long ORB[1311.00,1331.00] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-27 10:20:00 | 1338.38 | 1329.95 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1448.10 | 1443.16 | 0.00 | ORB-long ORB[1433.40,1446.50] vol=1.5x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:35:00 | 1452.74 | 1445.68 | 0.00 | T1 1.5R @ 1452.74 |
| Target hit | 2026-05-04 15:20:00 | 1465.10 | 1453.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1445.10 | 1460.09 | 0.00 | ORB-short ORB[1463.00,1473.30] vol=1.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 1448.67 | 1458.75 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 1429.90 | 1425.40 | 0.00 | ORB-long ORB[1417.50,1429.50] vol=2.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 1426.76 | 1425.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-15 09:45:00 | 1415.60 | 2025-05-15 10:00:00 | 1418.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-05-21 11:00:00 | 1432.20 | 2025-05-21 11:05:00 | 1429.79 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-06-11 09:30:00 | 1466.60 | 2025-06-11 09:35:00 | 1463.03 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-16 10:30:00 | 1434.80 | 2025-06-16 11:05:00 | 1439.23 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-06-16 10:30:00 | 1434.80 | 2025-06-16 12:00:00 | 1434.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 10:05:00 | 1431.20 | 2025-06-17 10:25:00 | 1427.49 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-17 10:05:00 | 1431.20 | 2025-06-17 11:25:00 | 1431.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 10:55:00 | 1470.40 | 2025-06-25 12:15:00 | 1467.85 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-06-26 09:55:00 | 1480.80 | 2025-06-26 10:00:00 | 1484.79 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-26 09:55:00 | 1480.80 | 2025-06-26 10:40:00 | 1480.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 11:10:00 | 1510.00 | 2025-06-27 11:30:00 | 1514.06 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-27 11:10:00 | 1510.00 | 2025-06-27 15:20:00 | 1512.40 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-07-02 11:15:00 | 1517.90 | 2025-07-02 11:25:00 | 1519.92 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-07-03 11:15:00 | 1528.90 | 2025-07-03 11:45:00 | 1526.48 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-04 10:50:00 | 1526.40 | 2025-07-04 11:10:00 | 1523.75 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-07 10:55:00 | 1538.30 | 2025-07-07 11:00:00 | 1535.95 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-08 10:45:00 | 1533.50 | 2025-07-08 10:50:00 | 1535.35 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-07-11 10:55:00 | 1493.10 | 2025-07-11 11:10:00 | 1495.39 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-14 10:00:00 | 1497.70 | 2025-07-14 10:05:00 | 1495.34 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-15 11:00:00 | 1494.80 | 2025-07-15 11:15:00 | 1492.41 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-17 10:45:00 | 1481.00 | 2025-07-17 10:55:00 | 1482.74 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-07-18 11:00:00 | 1471.50 | 2025-07-18 12:35:00 | 1473.71 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-24 11:05:00 | 1412.50 | 2025-07-24 11:15:00 | 1409.88 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-07-24 11:05:00 | 1412.50 | 2025-07-24 15:20:00 | 1402.30 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2025-07-30 10:55:00 | 1405.60 | 2025-07-30 11:10:00 | 1408.54 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-31 11:00:00 | 1393.80 | 2025-07-31 12:10:00 | 1390.66 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-01 11:15:00 | 1396.80 | 2025-08-01 12:05:00 | 1401.43 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-08-01 11:15:00 | 1396.80 | 2025-08-01 15:00:00 | 1396.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 10:30:00 | 1379.30 | 2025-08-07 12:10:00 | 1381.37 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-08 10:45:00 | 1373.30 | 2025-08-08 11:15:00 | 1376.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-20 11:15:00 | 1420.90 | 2025-08-20 11:30:00 | 1418.71 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-21 10:40:00 | 1430.00 | 2025-08-21 10:45:00 | 1427.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-22 11:10:00 | 1414.40 | 2025-08-22 11:20:00 | 1416.39 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-26 09:45:00 | 1401.40 | 2025-08-26 10:40:00 | 1397.28 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-26 09:45:00 | 1401.40 | 2025-08-26 15:20:00 | 1383.00 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-09-05 10:50:00 | 1360.60 | 2025-09-05 11:00:00 | 1363.38 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-09 11:10:00 | 1373.80 | 2025-09-09 12:00:00 | 1371.00 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-09-09 11:10:00 | 1373.80 | 2025-09-09 12:30:00 | 1373.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 10:45:00 | 1383.50 | 2025-09-11 12:20:00 | 1381.98 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-09-12 11:00:00 | 1393.10 | 2025-09-12 11:25:00 | 1396.09 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-09-12 11:00:00 | 1393.10 | 2025-09-12 12:10:00 | 1393.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-15 10:45:00 | 1391.60 | 2025-09-15 11:25:00 | 1393.06 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2025-09-19 11:05:00 | 1405.50 | 2025-09-19 11:35:00 | 1407.19 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-09-23 10:55:00 | 1380.60 | 2025-09-23 11:00:00 | 1383.02 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-26 10:45:00 | 1375.10 | 2025-09-26 10:55:00 | 1372.95 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1385.00 | 2025-09-29 11:10:00 | 1387.86 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-09-29 10:55:00 | 1385.00 | 2025-09-29 11:30:00 | 1385.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-03 10:55:00 | 1359.50 | 2025-10-03 12:05:00 | 1361.56 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-07 10:20:00 | 1389.20 | 2025-10-07 10:25:00 | 1387.02 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-09 11:10:00 | 1379.50 | 2025-10-09 11:15:00 | 1377.51 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-10-17 10:55:00 | 1414.80 | 2025-10-17 11:40:00 | 1412.29 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-23 09:30:00 | 1479.30 | 2025-10-23 09:35:00 | 1475.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-27 09:30:00 | 1472.40 | 2025-10-27 09:40:00 | 1477.50 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-27 09:30:00 | 1472.40 | 2025-10-27 13:15:00 | 1478.70 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-28 10:50:00 | 1479.60 | 2025-10-28 11:15:00 | 1482.18 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-30 10:55:00 | 1493.00 | 2025-10-30 11:00:00 | 1495.41 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-10-31 10:40:00 | 1488.00 | 2025-10-31 11:00:00 | 1484.08 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-31 10:40:00 | 1488.00 | 2025-10-31 11:55:00 | 1488.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-06 09:35:00 | 1498.50 | 2025-11-06 10:10:00 | 1494.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-12 11:15:00 | 1515.20 | 2025-11-12 11:25:00 | 1518.66 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-11-12 11:15:00 | 1515.20 | 2025-11-12 14:40:00 | 1515.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 11:15:00 | 1516.60 | 2025-11-13 12:55:00 | 1519.81 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-11-13 11:15:00 | 1516.60 | 2025-11-13 13:55:00 | 1516.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 10:55:00 | 1517.90 | 2025-11-14 11:15:00 | 1515.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-11-19 11:15:00 | 1514.10 | 2025-11-19 11:40:00 | 1515.88 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-11-20 10:35:00 | 1540.10 | 2025-11-20 12:10:00 | 1544.20 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-11-20 10:35:00 | 1540.10 | 2025-11-20 15:20:00 | 1548.50 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-26 10:45:00 | 1561.30 | 2025-11-26 12:25:00 | 1565.84 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-26 10:45:00 | 1561.30 | 2025-11-26 13:25:00 | 1561.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 11:15:00 | 1563.60 | 2025-11-27 13:05:00 | 1560.45 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-11-27 11:15:00 | 1563.60 | 2025-11-27 15:05:00 | 1563.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 09:30:00 | 1578.00 | 2025-11-28 09:35:00 | 1574.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-04 11:10:00 | 1559.90 | 2025-12-04 11:15:00 | 1557.84 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-12-16 11:15:00 | 1543.00 | 2025-12-16 11:20:00 | 1545.22 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-17 11:00:00 | 1541.70 | 2025-12-17 11:45:00 | 1543.91 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-12-19 10:40:00 | 1571.90 | 2025-12-19 11:40:00 | 1569.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-30 09:35:00 | 1552.30 | 2025-12-30 09:45:00 | 1550.04 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1552.00 | 2025-12-31 11:15:00 | 1555.11 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1552.00 | 2025-12-31 15:20:00 | 1568.50 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2026-01-02 10:40:00 | 1588.40 | 2026-01-02 10:45:00 | 1586.14 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2026-01-08 10:55:00 | 1484.80 | 2026-01-08 11:20:00 | 1480.62 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-01-08 10:55:00 | 1484.80 | 2026-01-08 15:20:00 | 1472.00 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2026-01-13 09:35:00 | 1465.00 | 2026-01-13 09:45:00 | 1468.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-20 11:10:00 | 1399.90 | 2026-01-20 11:45:00 | 1402.41 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-29 09:45:00 | 1390.90 | 2026-01-29 10:55:00 | 1393.59 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-30 11:15:00 | 1389.20 | 2026-01-30 11:20:00 | 1386.96 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-01 11:00:00 | 1404.80 | 2026-02-01 11:05:00 | 1409.11 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-01 11:00:00 | 1404.80 | 2026-02-01 11:10:00 | 1404.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 11:15:00 | 1459.70 | 2026-02-12 11:20:00 | 1457.01 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2026-02-12 11:15:00 | 1459.70 | 2026-02-12 11:55:00 | 1459.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:15:00 | 1431.00 | 2026-02-25 12:10:00 | 1427.95 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2026-02-25 11:15:00 | 1431.00 | 2026-02-25 15:20:00 | 1398.40 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2026-03-12 11:15:00 | 1404.20 | 2026-03-12 13:05:00 | 1400.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-18 11:00:00 | 1411.60 | 2026-03-18 11:25:00 | 1408.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-25 11:00:00 | 1427.50 | 2026-03-25 11:35:00 | 1424.83 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-04-09 10:40:00 | 1332.20 | 2026-04-09 10:50:00 | 1335.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-10 09:55:00 | 1347.50 | 2026-04-10 10:05:00 | 1344.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1335.50 | 2026-04-16 11:00:00 | 1338.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1357.00 | 2026-04-17 11:10:00 | 1360.46 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1357.00 | 2026-04-17 13:20:00 | 1357.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1361.60 | 2026-04-22 11:05:00 | 1365.24 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1361.60 | 2026-04-22 11:35:00 | 1361.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1331.60 | 2026-04-24 11:20:00 | 1334.02 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-04-27 09:55:00 | 1342.40 | 2026-04-27 10:20:00 | 1338.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 11:15:00 | 1448.10 | 2026-05-04 12:35:00 | 1452.74 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-05-04 11:15:00 | 1448.10 | 2026-05-04 15:20:00 | 1465.10 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1445.10 | 2026-05-06 11:10:00 | 1448.67 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-08 11:00:00 | 1429.90 | 2026-05-08 11:15:00 | 1426.76 | STOP_HIT | 1.00 | -0.22% |
