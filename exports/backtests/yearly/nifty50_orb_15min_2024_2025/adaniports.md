# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 1760.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 12 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 65
- **Target hits / Stop hits / Partials:** 12 / 65 / 32
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 17.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 23 | 34.8% | 5 | 43 | 18 | 0.11% | 7.5% |
| BUY @ 2nd Alert (retest1) | 66 | 23 | 34.8% | 5 | 43 | 18 | 0.11% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 21 | 48.8% | 7 | 22 | 14 | 0.22% | 9.6% |
| SELL @ 2nd Alert (retest1) | 43 | 21 | 48.8% | 7 | 22 | 14 | 0.22% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 44 | 40.4% | 12 | 65 | 32 | 0.16% | 17.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 1315.95 | 1301.04 | 0.00 | ORB-long ORB[1295.10,1312.85] vol=1.9x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:35:00 | 1322.47 | 1305.63 | 0.00 | T1 1.5R @ 1322.47 |
| Target hit | 2024-05-14 15:20:00 | 1333.00 | 1321.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 1349.05 | 1343.15 | 0.00 | ORB-long ORB[1334.05,1344.05] vol=2.4x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-05-15 10:00:00 | 1344.94 | 1346.11 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 1354.85 | 1343.60 | 0.00 | ORB-long ORB[1333.00,1344.95] vol=3.1x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:15:00 | 1361.19 | 1348.17 | 0.00 | T1 1.5R @ 1361.19 |
| Target hit | 2024-05-21 15:20:00 | 1387.00 | 1374.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:15:00 | 1388.60 | 1377.98 | 0.00 | ORB-long ORB[1370.70,1384.00] vol=3.0x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 1383.88 | 1378.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 1418.60 | 1424.58 | 0.00 | ORB-short ORB[1422.20,1438.00] vol=1.8x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:40:00 | 1412.66 | 1422.83 | 0.00 | T1 1.5R @ 1412.66 |
| Stop hit — per-position SL triggered | 2024-05-28 09:45:00 | 1418.60 | 1422.45 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 1392.00 | 1402.15 | 0.00 | ORB-short ORB[1402.65,1411.00] vol=1.5x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-06-13 11:40:00 | 1395.12 | 1400.65 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 1411.50 | 1404.07 | 0.00 | ORB-long ORB[1395.45,1410.00] vol=1.9x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-06-14 09:55:00 | 1407.63 | 1404.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:00:00 | 1440.95 | 1445.97 | 0.00 | ORB-short ORB[1443.10,1458.60] vol=4.4x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-06-18 11:50:00 | 1445.70 | 1443.88 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 1466.50 | 1454.32 | 0.00 | ORB-long ORB[1444.10,1463.90] vol=1.9x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:15:00 | 1474.29 | 1457.34 | 0.00 | T1 1.5R @ 1474.29 |
| Stop hit — per-position SL triggered | 2024-06-20 10:30:00 | 1466.50 | 1459.69 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:00:00 | 1449.35 | 1456.22 | 0.00 | ORB-short ORB[1455.35,1468.90] vol=1.7x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:00:00 | 1444.58 | 1453.79 | 0.00 | T1 1.5R @ 1444.58 |
| Stop hit — per-position SL triggered | 2024-06-25 14:20:00 | 1449.35 | 1449.03 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 1485.50 | 1477.07 | 0.00 | ORB-long ORB[1471.70,1483.95] vol=2.2x ATR=3.94 |
| Stop hit — per-position SL triggered | 2024-07-02 11:00:00 | 1481.56 | 1480.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1482.35 | 1478.20 | 0.00 | ORB-long ORB[1470.80,1480.75] vol=1.8x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:15:00 | 1487.79 | 1481.51 | 0.00 | T1 1.5R @ 1487.79 |
| Stop hit — per-position SL triggered | 2024-07-03 10:35:00 | 1482.35 | 1481.73 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 11:10:00 | 1504.90 | 1506.03 | 0.00 | ORB-short ORB[1505.70,1516.15] vol=1.9x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-07-04 11:30:00 | 1508.92 | 1505.96 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 1500.30 | 1489.31 | 0.00 | ORB-long ORB[1477.00,1492.75] vol=3.0x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-07-09 09:40:00 | 1496.10 | 1493.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:00:00 | 1494.00 | 1489.15 | 0.00 | ORB-long ORB[1484.35,1492.95] vol=3.3x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 1490.60 | 1490.40 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1476.65 | 1487.63 | 0.00 | ORB-short ORB[1484.25,1499.05] vol=2.5x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1481.06 | 1485.03 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:55:00 | 1489.80 | 1476.53 | 0.00 | ORB-long ORB[1467.00,1477.00] vol=1.9x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:00:00 | 1495.63 | 1478.02 | 0.00 | T1 1.5R @ 1495.63 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 1489.80 | 1481.54 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:10:00 | 1486.00 | 1474.40 | 0.00 | ORB-long ORB[1466.85,1475.40] vol=2.0x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:30:00 | 1493.30 | 1478.50 | 0.00 | T1 1.5R @ 1493.30 |
| Stop hit — per-position SL triggered | 2024-07-25 13:30:00 | 1486.00 | 1484.86 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:30:00 | 1513.70 | 1502.45 | 0.00 | ORB-long ORB[1490.00,1498.95] vol=2.2x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:00:00 | 1519.48 | 1508.23 | 0.00 | T1 1.5R @ 1519.48 |
| Target hit | 2024-07-26 15:20:00 | 1536.40 | 1530.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:45:00 | 1564.00 | 1557.37 | 0.00 | ORB-long ORB[1547.00,1563.75] vol=2.2x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:55:00 | 1571.07 | 1560.44 | 0.00 | T1 1.5R @ 1571.07 |
| Stop hit — per-position SL triggered | 2024-07-31 11:25:00 | 1564.00 | 1562.77 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:30:00 | 1594.00 | 1585.38 | 0.00 | ORB-long ORB[1575.00,1589.95] vol=2.0x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 09:35:00 | 1600.39 | 1589.10 | 0.00 | T1 1.5R @ 1600.39 |
| Stop hit — per-position SL triggered | 2024-08-01 09:50:00 | 1594.00 | 1592.76 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:00:00 | 1528.55 | 1522.88 | 0.00 | ORB-long ORB[1514.20,1526.55] vol=1.7x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:25:00 | 1538.24 | 1526.15 | 0.00 | T1 1.5R @ 1538.24 |
| Target hit | 2024-08-07 13:20:00 | 1530.75 | 1531.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-08-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:35:00 | 1553.15 | 1542.27 | 0.00 | ORB-long ORB[1537.60,1547.55] vol=1.6x ATR=5.07 |
| Stop hit — per-position SL triggered | 2024-08-08 10:55:00 | 1548.08 | 1544.17 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 1504.80 | 1499.12 | 0.00 | ORB-long ORB[1492.55,1502.75] vol=1.5x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-08-21 09:50:00 | 1501.59 | 1499.48 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 1490.00 | 1484.39 | 0.00 | ORB-long ORB[1479.50,1487.00] vol=1.5x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:50:00 | 1494.52 | 1486.94 | 0.00 | T1 1.5R @ 1494.52 |
| Stop hit — per-position SL triggered | 2024-08-27 09:55:00 | 1490.00 | 1487.05 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:05:00 | 1485.00 | 1480.73 | 0.00 | ORB-long ORB[1476.00,1482.00] vol=2.3x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-08-28 10:35:00 | 1482.24 | 1481.85 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1465.45 | 1468.44 | 0.00 | ORB-short ORB[1468.20,1475.40] vol=2.1x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:25:00 | 1461.50 | 1467.84 | 0.00 | T1 1.5R @ 1461.50 |
| Stop hit — per-position SL triggered | 2024-08-29 12:20:00 | 1465.45 | 1465.36 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:05:00 | 1484.60 | 1488.58 | 0.00 | ORB-short ORB[1488.60,1498.10] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-09-03 13:00:00 | 1486.74 | 1487.48 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 1457.65 | 1461.61 | 0.00 | ORB-short ORB[1459.00,1468.75] vol=1.9x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:55:00 | 1452.20 | 1459.23 | 0.00 | T1 1.5R @ 1452.20 |
| Target hit | 2024-09-06 12:05:00 | 1445.00 | 1444.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:45:00 | 1460.35 | 1451.75 | 0.00 | ORB-long ORB[1440.20,1454.80] vol=1.8x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:05:00 | 1468.29 | 1457.07 | 0.00 | T1 1.5R @ 1468.29 |
| Stop hit — per-position SL triggered | 2024-09-12 10:40:00 | 1460.35 | 1459.22 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 1428.95 | 1433.33 | 0.00 | ORB-short ORB[1431.30,1445.00] vol=1.7x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:15:00 | 1424.54 | 1430.85 | 0.00 | T1 1.5R @ 1424.54 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 1428.95 | 1430.21 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:10:00 | 1415.80 | 1425.63 | 0.00 | ORB-short ORB[1428.80,1438.00] vol=2.0x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:20:00 | 1410.46 | 1424.20 | 0.00 | T1 1.5R @ 1410.46 |
| Target hit | 2024-09-19 15:20:00 | 1410.10 | 1410.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1431.05 | 1419.83 | 0.00 | ORB-long ORB[1409.80,1428.00] vol=1.9x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-09-20 11:00:00 | 1427.12 | 1422.30 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 1464.50 | 1458.51 | 0.00 | ORB-long ORB[1451.15,1458.15] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-09-24 10:05:00 | 1461.30 | 1459.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 1447.30 | 1452.91 | 0.00 | ORB-short ORB[1450.50,1464.00] vol=1.6x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-09-25 09:45:00 | 1450.98 | 1451.98 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 1459.50 | 1451.93 | 0.00 | ORB-long ORB[1444.40,1458.40] vol=1.5x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:05:00 | 1467.03 | 1453.08 | 0.00 | T1 1.5R @ 1467.03 |
| Stop hit — per-position SL triggered | 2024-09-30 11:20:00 | 1459.50 | 1453.67 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:40:00 | 1422.75 | 1418.91 | 0.00 | ORB-long ORB[1414.00,1419.95] vol=2.1x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-10-14 10:10:00 | 1419.55 | 1420.26 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 1407.80 | 1412.47 | 0.00 | ORB-short ORB[1408.10,1418.40] vol=3.2x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:40:00 | 1403.77 | 1410.97 | 0.00 | T1 1.5R @ 1403.77 |
| Stop hit — per-position SL triggered | 2024-10-16 14:25:00 | 1407.80 | 1404.16 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 1399.40 | 1402.51 | 0.00 | ORB-short ORB[1401.05,1408.00] vol=1.6x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:20:00 | 1393.98 | 1400.33 | 0.00 | T1 1.5R @ 1393.98 |
| Stop hit — per-position SL triggered | 2024-10-17 10:35:00 | 1399.40 | 1400.03 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:10:00 | 1390.80 | 1393.85 | 0.00 | ORB-short ORB[1398.00,1412.90] vol=1.7x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:30:00 | 1385.63 | 1392.66 | 0.00 | T1 1.5R @ 1385.63 |
| Target hit | 2024-10-21 15:20:00 | 1375.55 | 1383.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 1360.00 | 1372.63 | 0.00 | ORB-short ORB[1373.65,1387.10] vol=2.2x ATR=4.82 |
| Stop hit — per-position SL triggered | 2024-10-22 11:05:00 | 1364.82 | 1365.81 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 11:15:00 | 1353.60 | 1328.02 | 0.00 | ORB-long ORB[1308.65,1326.85] vol=1.5x ATR=5.09 |
| Stop hit — per-position SL triggered | 2024-10-28 11:45:00 | 1348.51 | 1330.42 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:05:00 | 1334.70 | 1328.37 | 0.00 | ORB-long ORB[1315.20,1330.95] vol=2.0x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:10:00 | 1342.59 | 1329.25 | 0.00 | T1 1.5R @ 1342.59 |
| Target hit | 2024-11-06 15:20:00 | 1370.20 | 1348.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 1359.90 | 1351.23 | 0.00 | ORB-long ORB[1338.60,1356.10] vol=1.6x ATR=4.62 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 1355.28 | 1351.56 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:05:00 | 1241.50 | 1249.93 | 0.00 | ORB-short ORB[1250.40,1258.35] vol=2.2x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:15:00 | 1236.07 | 1245.74 | 0.00 | T1 1.5R @ 1236.07 |
| Target hit | 2024-12-11 15:20:00 | 1232.20 | 1239.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 1243.10 | 1231.38 | 0.00 | ORB-long ORB[1226.60,1237.95] vol=2.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 1239.14 | 1232.32 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 1244.20 | 1250.14 | 0.00 | ORB-short ORB[1251.55,1261.85] vol=2.4x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-12-16 12:35:00 | 1247.46 | 1248.77 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1250.15 | 1246.27 | 0.00 | ORB-long ORB[1234.35,1249.80] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 1247.19 | 1246.50 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 1209.70 | 1205.37 | 0.00 | ORB-long ORB[1199.30,1209.05] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 1206.41 | 1206.43 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 1199.85 | 1193.53 | 0.00 | ORB-long ORB[1182.55,1194.50] vol=2.2x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:10:00 | 1204.97 | 1195.32 | 0.00 | T1 1.5R @ 1204.97 |
| Stop hit — per-position SL triggered | 2024-12-26 12:00:00 | 1199.85 | 1197.22 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1249.85 | 1246.02 | 0.00 | ORB-long ORB[1233.00,1249.00] vol=2.7x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:35:00 | 1256.85 | 1248.30 | 0.00 | T1 1.5R @ 1256.85 |
| Stop hit — per-position SL triggered | 2024-12-30 09:55:00 | 1249.85 | 1250.03 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 11:15:00 | 1213.60 | 1216.55 | 0.00 | ORB-short ORB[1216.75,1225.80] vol=2.4x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-12-31 12:00:00 | 1217.60 | 1216.39 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:45:00 | 1216.05 | 1222.09 | 0.00 | ORB-short ORB[1220.00,1227.90] vol=2.2x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:00:00 | 1209.38 | 1218.07 | 0.00 | T1 1.5R @ 1209.38 |
| Stop hit — per-position SL triggered | 2025-01-01 10:35:00 | 1216.05 | 1216.02 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 1220.75 | 1225.89 | 0.00 | ORB-short ORB[1222.00,1237.60] vol=2.1x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 12:05:00 | 1215.13 | 1224.21 | 0.00 | T1 1.5R @ 1215.13 |
| Target hit | 2025-01-03 15:20:00 | 1198.50 | 1213.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1172.10 | 1186.90 | 0.00 | ORB-short ORB[1189.65,1200.90] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1175.54 | 1185.96 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:05:00 | 1086.80 | 1094.76 | 0.00 | ORB-short ORB[1091.65,1106.45] vol=2.6x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:30:00 | 1080.39 | 1091.02 | 0.00 | T1 1.5R @ 1080.39 |
| Target hit | 2025-01-13 15:20:00 | 1066.50 | 1082.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:40:00 | 1139.65 | 1131.27 | 0.00 | ORB-long ORB[1120.05,1131.80] vol=2.3x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-01-15 09:50:00 | 1134.97 | 1132.27 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 1121.80 | 1134.82 | 0.00 | ORB-short ORB[1141.50,1154.25] vol=2.2x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-01-21 10:25:00 | 1125.19 | 1134.09 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:20:00 | 1095.30 | 1098.83 | 0.00 | ORB-short ORB[1102.05,1113.20] vol=2.1x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-01-24 10:35:00 | 1098.95 | 1098.76 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:00:00 | 1091.95 | 1084.49 | 0.00 | ORB-long ORB[1074.50,1089.45] vol=3.6x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-01-29 11:10:00 | 1088.26 | 1085.16 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 1108.45 | 1103.59 | 0.00 | ORB-long ORB[1093.50,1107.85] vol=2.1x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:15:00 | 1115.06 | 1109.48 | 0.00 | T1 1.5R @ 1115.06 |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 1108.45 | 1109.98 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:35:00 | 1160.00 | 1150.44 | 0.00 | ORB-long ORB[1136.95,1153.20] vol=2.4x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:45:00 | 1166.40 | 1154.91 | 0.00 | T1 1.5R @ 1166.40 |
| Stop hit — per-position SL triggered | 2025-03-10 10:25:00 | 1160.00 | 1158.72 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:45:00 | 1152.60 | 1144.87 | 0.00 | ORB-long ORB[1135.80,1144.65] vol=2.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-03-12 10:00:00 | 1148.68 | 1148.17 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 1152.00 | 1145.28 | 0.00 | ORB-long ORB[1137.95,1150.00] vol=1.7x ATR=3.45 |
| Stop hit — per-position SL triggered | 2025-03-18 10:35:00 | 1148.55 | 1147.09 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:35:00 | 1180.25 | 1169.79 | 0.00 | ORB-long ORB[1158.80,1169.95] vol=1.8x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-03-19 12:05:00 | 1176.71 | 1175.26 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 1190.45 | 1181.67 | 0.00 | ORB-long ORB[1172.05,1183.95] vol=1.6x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 1186.87 | 1182.31 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:15:00 | 1205.45 | 1201.19 | 0.00 | ORB-long ORB[1194.05,1203.15] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-03-24 11:40:00 | 1202.21 | 1201.66 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:55:00 | 1181.10 | 1193.71 | 0.00 | ORB-short ORB[1196.30,1209.10] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-03-25 12:05:00 | 1185.88 | 1190.89 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:15:00 | 1194.40 | 1191.15 | 0.00 | ORB-long ORB[1173.05,1190.90] vol=1.9x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-03-26 11:45:00 | 1190.75 | 1191.66 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-04-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:05:00 | 1116.75 | 1119.57 | 0.00 | ORB-short ORB[1119.00,1133.85] vol=3.5x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 1120.73 | 1119.32 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 1165.85 | 1158.82 | 0.00 | ORB-long ORB[1148.00,1164.95] vol=2.3x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-04-11 10:05:00 | 1160.97 | 1161.97 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 1214.00 | 1209.26 | 0.00 | ORB-long ORB[1202.40,1211.90] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-04-16 11:40:00 | 1210.27 | 1210.10 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:45:00 | 1252.50 | 1243.44 | 0.00 | ORB-long ORB[1233.90,1249.90] vol=2.2x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-04-22 10:50:00 | 1248.88 | 1243.88 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 1222.00 | 1232.60 | 0.00 | ORB-short ORB[1230.10,1241.90] vol=2.0x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 1226.17 | 1228.30 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 1220.70 | 1228.73 | 0.00 | ORB-short ORB[1226.20,1243.40] vol=2.7x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 1215.06 | 1225.76 | 0.00 | T1 1.5R @ 1215.06 |
| Target hit | 2025-04-25 12:55:00 | 1199.80 | 1198.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 1200.80 | 1192.31 | 0.00 | ORB-long ORB[1184.10,1199.00] vol=1.9x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 1195.69 | 1192.86 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 1312.40 | 1301.05 | 0.00 | ORB-long ORB[1285.70,1304.90] vol=2.5x ATR=6.77 |
| Stop hit — per-position SL triggered | 2025-05-05 09:35:00 | 1305.63 | 1302.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:15:00 | 1315.95 | 2024-05-14 11:35:00 | 1322.47 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-05-14 11:15:00 | 1315.95 | 2024-05-14 15:20:00 | 1333.00 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2024-05-15 09:30:00 | 1349.05 | 2024-05-15 10:00:00 | 1344.94 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-21 09:55:00 | 1354.85 | 2024-05-21 10:15:00 | 1361.19 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-21 09:55:00 | 1354.85 | 2024-05-21 15:20:00 | 1387.00 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2024-05-23 11:15:00 | 1388.60 | 2024-05-23 11:20:00 | 1383.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-28 09:35:00 | 1418.60 | 2024-05-28 09:40:00 | 1412.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-28 09:35:00 | 1418.60 | 2024-05-28 09:45:00 | 1418.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:15:00 | 1392.00 | 2024-06-13 11:40:00 | 1395.12 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-14 09:50:00 | 1411.50 | 2024-06-14 09:55:00 | 1407.63 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-18 10:00:00 | 1440.95 | 2024-06-18 11:50:00 | 1445.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-20 10:10:00 | 1466.50 | 2024-06-20 10:15:00 | 1474.29 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-20 10:10:00 | 1466.50 | 2024-06-20 10:30:00 | 1466.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 10:00:00 | 1449.35 | 2024-06-25 11:00:00 | 1444.58 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-25 10:00:00 | 1449.35 | 2024-06-25 14:20:00 | 1449.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 10:35:00 | 1485.50 | 2024-07-02 11:00:00 | 1481.56 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1482.35 | 2024-07-03 10:15:00 | 1487.79 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1482.35 | 2024-07-03 10:35:00 | 1482.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-04 11:10:00 | 1504.90 | 2024-07-04 11:30:00 | 1508.92 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-09 09:30:00 | 1500.30 | 2024-07-09 09:40:00 | 1496.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 11:00:00 | 1494.00 | 2024-07-12 11:15:00 | 1490.60 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1476.65 | 2024-07-18 09:40:00 | 1481.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-23 10:55:00 | 1489.80 | 2024-07-23 11:00:00 | 1495.63 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-23 10:55:00 | 1489.80 | 2024-07-23 11:15:00 | 1489.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 10:10:00 | 1486.00 | 2024-07-25 10:30:00 | 1493.30 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-25 10:10:00 | 1486.00 | 2024-07-25 13:30:00 | 1486.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:30:00 | 1513.70 | 2024-07-26 11:00:00 | 1519.48 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-26 10:30:00 | 1513.70 | 2024-07-26 15:20:00 | 1536.40 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2024-07-31 10:45:00 | 1564.00 | 2024-07-31 10:55:00 | 1571.07 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-31 10:45:00 | 1564.00 | 2024-07-31 11:25:00 | 1564.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 09:30:00 | 1594.00 | 2024-08-01 09:35:00 | 1600.39 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-01 09:30:00 | 1594.00 | 2024-08-01 09:50:00 | 1594.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-07 10:00:00 | 1528.55 | 2024-08-07 10:25:00 | 1538.24 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-07 10:00:00 | 1528.55 | 2024-08-07 13:20:00 | 1530.75 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-08-08 10:35:00 | 1553.15 | 2024-08-08 10:55:00 | 1548.08 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-21 09:45:00 | 1504.80 | 2024-08-21 09:50:00 | 1501.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-27 09:40:00 | 1490.00 | 2024-08-27 09:50:00 | 1494.52 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-08-27 09:40:00 | 1490.00 | 2024-08-27 09:55:00 | 1490.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 10:05:00 | 1485.00 | 2024-08-28 10:35:00 | 1482.24 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1465.45 | 2024-08-29 11:25:00 | 1461.50 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1465.45 | 2024-08-29 12:20:00 | 1465.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-03 11:05:00 | 1484.60 | 2024-09-03 13:00:00 | 1486.74 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-09-06 09:45:00 | 1457.65 | 2024-09-06 09:55:00 | 1452.20 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-06 09:45:00 | 1457.65 | 2024-09-06 12:05:00 | 1445.00 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-09-12 09:45:00 | 1460.35 | 2024-09-12 10:05:00 | 1468.29 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-12 09:45:00 | 1460.35 | 2024-09-12 10:40:00 | 1460.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 09:35:00 | 1428.95 | 2024-09-17 10:15:00 | 1424.54 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-09-17 09:35:00 | 1428.95 | 2024-09-17 10:35:00 | 1428.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:10:00 | 1415.80 | 2024-09-19 11:20:00 | 1410.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-19 11:10:00 | 1415.80 | 2024-09-19 15:20:00 | 1410.10 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1431.05 | 2024-09-20 11:00:00 | 1427.12 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-24 09:50:00 | 1464.50 | 2024-09-24 10:05:00 | 1461.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-25 09:30:00 | 1447.30 | 2024-09-25 09:45:00 | 1450.98 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-30 10:55:00 | 1459.50 | 2024-09-30 11:05:00 | 1467.03 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-30 10:55:00 | 1459.50 | 2024-09-30 11:20:00 | 1459.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:40:00 | 1422.75 | 2024-10-14 10:10:00 | 1419.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-16 11:15:00 | 1407.80 | 2024-10-16 11:40:00 | 1403.77 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-16 11:15:00 | 1407.80 | 2024-10-16 14:25:00 | 1407.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:40:00 | 1399.40 | 2024-10-17 10:20:00 | 1393.98 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-10-17 09:40:00 | 1399.40 | 2024-10-17 10:35:00 | 1399.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 11:10:00 | 1390.80 | 2024-10-21 11:30:00 | 1385.63 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-21 11:10:00 | 1390.80 | 2024-10-21 15:20:00 | 1375.55 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2024-10-22 10:30:00 | 1360.00 | 2024-10-22 11:05:00 | 1364.82 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-28 11:15:00 | 1353.60 | 2024-10-28 11:45:00 | 1348.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-06 10:05:00 | 1334.70 | 2024-11-06 10:10:00 | 1342.59 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-11-06 10:05:00 | 1334.70 | 2024-11-06 15:20:00 | 1370.20 | TARGET_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2024-11-08 09:45:00 | 1359.90 | 2024-11-08 09:50:00 | 1355.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1241.50 | 2024-12-11 11:15:00 | 1236.07 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1241.50 | 2024-12-11 15:20:00 | 1232.20 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2024-12-12 10:30:00 | 1243.10 | 2024-12-12 10:35:00 | 1239.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-16 11:10:00 | 1244.20 | 2024-12-16 12:35:00 | 1247.46 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-17 09:30:00 | 1250.15 | 2024-12-17 09:35:00 | 1247.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1209.70 | 2024-12-20 09:45:00 | 1206.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-26 11:00:00 | 1199.85 | 2024-12-26 11:10:00 | 1204.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-26 11:00:00 | 1199.85 | 2024-12-26 12:00:00 | 1199.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1249.85 | 2024-12-30 09:35:00 | 1256.85 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1249.85 | 2024-12-30 09:55:00 | 1249.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-31 11:15:00 | 1213.60 | 2024-12-31 12:00:00 | 1217.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-01 09:45:00 | 1216.05 | 2025-01-01 10:00:00 | 1209.38 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-01 09:45:00 | 1216.05 | 2025-01-01 10:35:00 | 1216.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 10:55:00 | 1220.75 | 2025-01-03 12:05:00 | 1215.13 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-03 10:55:00 | 1220.75 | 2025-01-03 15:20:00 | 1198.50 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2025-01-06 11:15:00 | 1172.10 | 2025-01-06 11:30:00 | 1175.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-13 11:05:00 | 1086.80 | 2025-01-13 13:30:00 | 1080.39 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-13 11:05:00 | 1086.80 | 2025-01-13 15:20:00 | 1066.50 | TARGET_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2025-01-15 09:40:00 | 1139.65 | 2025-01-15 09:50:00 | 1134.97 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1121.80 | 2025-01-21 10:25:00 | 1125.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-24 10:20:00 | 1095.30 | 2025-01-24 10:35:00 | 1098.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-29 11:00:00 | 1091.95 | 2025-01-29 11:10:00 | 1088.26 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1108.45 | 2025-01-30 11:15:00 | 1115.06 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1108.45 | 2025-01-30 12:15:00 | 1108.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 09:35:00 | 1160.00 | 2025-03-10 09:45:00 | 1166.40 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-10 09:35:00 | 1160.00 | 2025-03-10 10:25:00 | 1160.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 09:45:00 | 1152.60 | 2025-03-12 10:00:00 | 1148.68 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-18 10:00:00 | 1152.00 | 2025-03-18 10:35:00 | 1148.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-19 10:35:00 | 1180.25 | 2025-03-19 12:05:00 | 1176.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-21 09:40:00 | 1190.45 | 2025-03-21 09:45:00 | 1186.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-24 11:15:00 | 1205.45 | 2025-03-24 11:40:00 | 1202.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-25 10:55:00 | 1181.10 | 2025-03-25 12:05:00 | 1185.88 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-26 11:15:00 | 1194.40 | 2025-03-26 11:45:00 | 1190.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-09 11:05:00 | 1116.75 | 2025-04-09 11:15:00 | 1120.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-11 09:35:00 | 1165.85 | 2025-04-11 10:05:00 | 1160.97 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-16 11:15:00 | 1214.00 | 2025-04-16 11:40:00 | 1210.27 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-22 10:45:00 | 1252.50 | 2025-04-22 10:50:00 | 1248.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 10:05:00 | 1222.00 | 2025-04-23 11:10:00 | 1226.17 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1220.70 | 2025-04-25 09:35:00 | 1215.06 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1220.70 | 2025-04-25 12:55:00 | 1199.80 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2025-04-28 09:30:00 | 1200.80 | 2025-04-28 09:40:00 | 1195.69 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-05 09:30:00 | 1312.40 | 2025-05-05 09:35:00 | 1305.63 | STOP_HIT | 1.00 | -0.52% |
