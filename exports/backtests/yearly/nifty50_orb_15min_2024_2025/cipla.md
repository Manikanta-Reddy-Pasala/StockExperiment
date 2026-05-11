# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 12 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 70
- **Target hits / Stop hits / Partials:** 12 / 70 / 29
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 5.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 23 | 35.4% | 6 | 42 | 17 | 0.04% | 2.6% |
| BUY @ 2nd Alert (retest1) | 65 | 23 | 35.4% | 6 | 42 | 17 | 0.04% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 18 | 39.1% | 6 | 28 | 12 | 0.07% | 3.3% |
| SELL @ 2nd Alert (retest1) | 46 | 18 | 39.1% | 6 | 28 | 12 | 0.07% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 41 | 36.9% | 12 | 70 | 29 | 0.05% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 1394.15 | 1400.03 | 0.00 | ORB-short ORB[1396.05,1413.35] vol=2.0x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 1398.28 | 1398.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 1405.60 | 1409.50 | 0.00 | ORB-short ORB[1406.00,1419.90] vol=1.5x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 12:25:00 | 1400.44 | 1406.83 | 0.00 | T1 1.5R @ 1400.44 |
| Target hit | 2024-05-17 15:20:00 | 1399.80 | 1403.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 1410.15 | 1407.17 | 0.00 | ORB-long ORB[1396.40,1410.00] vol=4.7x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:05:00 | 1415.35 | 1407.84 | 0.00 | T1 1.5R @ 1415.35 |
| Target hit | 2024-05-21 15:20:00 | 1443.70 | 1425.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-05-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:00:00 | 1453.20 | 1445.79 | 0.00 | ORB-long ORB[1443.00,1450.95] vol=1.5x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:15:00 | 1459.97 | 1450.16 | 0.00 | T1 1.5R @ 1459.97 |
| Stop hit — per-position SL triggered | 2024-05-22 10:30:00 | 1453.20 | 1451.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 1493.00 | 1485.20 | 0.00 | ORB-long ORB[1475.20,1486.05] vol=2.1x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-05-28 09:55:00 | 1489.16 | 1487.80 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:55:00 | 1484.15 | 1479.65 | 0.00 | ORB-long ORB[1472.60,1483.20] vol=1.5x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:25:00 | 1488.14 | 1480.13 | 0.00 | T1 1.5R @ 1488.14 |
| Stop hit — per-position SL triggered | 2024-05-29 12:05:00 | 1484.15 | 1480.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:00:00 | 1479.95 | 1483.06 | 0.00 | ORB-short ORB[1481.45,1496.55] vol=3.0x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:30:00 | 1474.55 | 1481.19 | 0.00 | T1 1.5R @ 1474.55 |
| Target hit | 2024-05-30 15:20:00 | 1471.50 | 1471.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 1549.00 | 1540.40 | 0.00 | ORB-long ORB[1532.00,1542.45] vol=5.0x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-06-12 11:05:00 | 1545.87 | 1541.06 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:45:00 | 1574.25 | 1570.43 | 0.00 | ORB-long ORB[1563.60,1572.65] vol=1.6x ATR=2.89 |
| Stop hit — per-position SL triggered | 2024-06-18 10:05:00 | 1571.36 | 1571.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 1503.30 | 1506.79 | 0.00 | ORB-short ORB[1505.05,1517.80] vol=3.9x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-06-25 11:10:00 | 1506.03 | 1505.77 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:05:00 | 1480.75 | 1491.48 | 0.00 | ORB-short ORB[1487.90,1507.20] vol=1.6x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-06-26 10:10:00 | 1485.08 | 1491.03 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 1476.00 | 1478.55 | 0.00 | ORB-short ORB[1478.05,1486.10] vol=1.8x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:00:00 | 1470.98 | 1475.63 | 0.00 | T1 1.5R @ 1470.98 |
| Stop hit — per-position SL triggered | 2024-06-27 14:50:00 | 1476.00 | 1473.62 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 1483.55 | 1475.19 | 0.00 | ORB-long ORB[1467.00,1483.20] vol=1.5x ATR=3.72 |
| Stop hit — per-position SL triggered | 2024-07-04 11:00:00 | 1479.83 | 1477.19 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:00:00 | 1509.00 | 1516.09 | 0.00 | ORB-short ORB[1510.05,1524.65] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-07-08 10:10:00 | 1513.17 | 1515.60 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:40:00 | 1494.90 | 1505.37 | 0.00 | ORB-short ORB[1510.35,1517.25] vol=1.9x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 1498.75 | 1505.13 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 1499.05 | 1503.81 | 0.00 | ORB-short ORB[1503.15,1519.00] vol=1.5x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 1502.56 | 1503.25 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:45:00 | 1528.15 | 1523.69 | 0.00 | ORB-long ORB[1516.20,1525.65] vol=1.7x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-07-15 10:35:00 | 1523.84 | 1525.49 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:50:00 | 1510.70 | 1503.74 | 0.00 | ORB-long ORB[1485.60,1506.00] vol=6.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-07-18 10:40:00 | 1506.64 | 1504.73 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:30:00 | 1501.00 | 1484.86 | 0.00 | ORB-long ORB[1470.65,1482.00] vol=1.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-07-22 12:40:00 | 1496.79 | 1493.62 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:35:00 | 1498.00 | 1495.52 | 0.00 | ORB-long ORB[1485.00,1496.95] vol=3.7x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 11:35:00 | 1503.83 | 1496.14 | 0.00 | T1 1.5R @ 1503.83 |
| Stop hit — per-position SL triggered | 2024-07-24 12:20:00 | 1498.00 | 1497.33 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:20:00 | 1535.00 | 1520.73 | 0.00 | ORB-long ORB[1501.00,1519.40] vol=1.6x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:00:00 | 1543.33 | 1526.91 | 0.00 | T1 1.5R @ 1543.33 |
| Stop hit — per-position SL triggered | 2024-07-26 11:35:00 | 1535.00 | 1528.56 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:30:00 | 1586.50 | 1581.28 | 0.00 | ORB-long ORB[1571.05,1584.75] vol=2.7x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:40:00 | 1591.77 | 1584.78 | 0.00 | T1 1.5R @ 1591.77 |
| Stop hit — per-position SL triggered | 2024-08-12 09:45:00 | 1586.50 | 1585.18 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:00:00 | 1604.25 | 1595.77 | 0.00 | ORB-long ORB[1585.30,1600.00] vol=2.3x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 1599.89 | 1597.14 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:55:00 | 1586.85 | 1578.12 | 0.00 | ORB-long ORB[1561.00,1574.85] vol=1.8x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-08-21 11:05:00 | 1583.84 | 1578.61 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 1580.10 | 1586.47 | 0.00 | ORB-short ORB[1586.05,1599.00] vol=2.4x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-08-22 11:10:00 | 1582.38 | 1586.09 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 11:00:00 | 1585.85 | 1580.68 | 0.00 | ORB-long ORB[1566.55,1576.30] vol=1.6x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 14:15:00 | 1589.90 | 1583.49 | 0.00 | T1 1.5R @ 1589.90 |
| Target hit | 2024-08-26 15:20:00 | 1593.35 | 1587.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 1608.00 | 1602.22 | 0.00 | ORB-long ORB[1586.50,1605.95] vol=1.9x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:00:00 | 1613.37 | 1606.76 | 0.00 | T1 1.5R @ 1613.37 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 1608.00 | 1610.75 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 1640.65 | 1633.78 | 0.00 | ORB-long ORB[1621.05,1636.65] vol=2.2x ATR=4.83 |
| Stop hit — per-position SL triggered | 2024-08-29 09:40:00 | 1635.82 | 1635.04 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 1630.15 | 1627.74 | 0.00 | ORB-long ORB[1615.00,1627.30] vol=1.5x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:50:00 | 1637.04 | 1628.98 | 0.00 | T1 1.5R @ 1637.04 |
| Target hit | 2024-08-30 10:20:00 | 1631.00 | 1631.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 1655.55 | 1651.52 | 0.00 | ORB-long ORB[1643.30,1655.00] vol=2.0x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 1651.85 | 1652.71 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 1619.60 | 1622.65 | 0.00 | ORB-short ORB[1622.00,1631.95] vol=1.7x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 1624.28 | 1622.55 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:00:00 | 1633.00 | 1624.05 | 0.00 | ORB-long ORB[1610.00,1622.60] vol=1.9x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-09-09 12:00:00 | 1629.29 | 1627.73 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1646.10 | 1638.75 | 0.00 | ORB-long ORB[1624.60,1640.85] vol=1.7x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-09-11 09:45:00 | 1642.07 | 1639.55 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:10:00 | 1651.55 | 1662.60 | 0.00 | ORB-short ORB[1660.10,1673.00] vol=2.3x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 1655.20 | 1661.77 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 1653.30 | 1655.76 | 0.00 | ORB-short ORB[1654.00,1660.65] vol=1.7x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:45:00 | 1647.61 | 1651.66 | 0.00 | T1 1.5R @ 1647.61 |
| Stop hit — per-position SL triggered | 2024-09-19 10:20:00 | 1653.30 | 1646.27 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:25:00 | 1616.25 | 1622.24 | 0.00 | ORB-short ORB[1627.00,1650.05] vol=3.4x ATR=5.69 |
| Stop hit — per-position SL triggered | 2024-09-20 10:30:00 | 1621.94 | 1622.18 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 11:00:00 | 1673.50 | 1659.71 | 0.00 | ORB-long ORB[1640.35,1661.25] vol=1.7x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 1668.63 | 1661.89 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:15:00 | 1650.75 | 1641.46 | 0.00 | ORB-long ORB[1623.85,1647.40] vol=2.2x ATR=5.25 |
| Stop hit — per-position SL triggered | 2024-10-08 11:55:00 | 1645.50 | 1642.70 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 1577.00 | 1567.13 | 0.00 | ORB-long ORB[1560.00,1571.00] vol=2.1x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-10-17 10:00:00 | 1572.48 | 1567.66 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1536.05 | 1543.04 | 0.00 | ORB-short ORB[1549.75,1561.25] vol=1.9x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:45:00 | 1529.91 | 1539.34 | 0.00 | T1 1.5R @ 1529.91 |
| Target hit | 2024-10-21 15:20:00 | 1524.00 | 1532.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-10-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 11:05:00 | 1506.40 | 1513.87 | 0.00 | ORB-short ORB[1513.00,1525.70] vol=2.5x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-10-22 11:25:00 | 1510.44 | 1513.09 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:50:00 | 1487.70 | 1494.17 | 0.00 | ORB-short ORB[1488.45,1506.75] vol=3.4x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:10:00 | 1480.45 | 1492.65 | 0.00 | T1 1.5R @ 1480.45 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 1487.70 | 1491.72 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:15:00 | 1593.30 | 1600.52 | 0.00 | ORB-short ORB[1599.45,1612.35] vol=1.5x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 12:05:00 | 1587.36 | 1598.45 | 0.00 | T1 1.5R @ 1587.36 |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 1593.30 | 1592.44 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 11:05:00 | 1579.80 | 1577.10 | 0.00 | ORB-long ORB[1568.70,1576.95] vol=6.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 1576.51 | 1577.17 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 11:15:00 | 1542.20 | 1544.69 | 0.00 | ORB-short ORB[1543.40,1556.95] vol=1.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 1546.48 | 1544.67 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:40:00 | 1486.45 | 1478.37 | 0.00 | ORB-long ORB[1460.05,1476.95] vol=1.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-11-22 10:45:00 | 1482.61 | 1478.54 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:55:00 | 1540.70 | 1527.76 | 0.00 | ORB-long ORB[1508.50,1522.30] vol=1.8x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-12-03 11:05:00 | 1537.28 | 1528.86 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 1506.85 | 1519.15 | 0.00 | ORB-short ORB[1521.10,1536.75] vol=2.5x ATR=3.32 |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 1510.17 | 1517.53 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:15:00 | 1482.65 | 1488.89 | 0.00 | ORB-short ORB[1494.05,1507.00] vol=2.9x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 10:35:00 | 1477.13 | 1487.57 | 0.00 | T1 1.5R @ 1477.13 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 1482.65 | 1483.04 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 1453.90 | 1457.52 | 0.00 | ORB-short ORB[1455.00,1461.35] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 1456.60 | 1457.38 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:05:00 | 1440.20 | 1444.22 | 0.00 | ORB-short ORB[1445.85,1457.55] vol=5.9x ATR=2.84 |
| Stop hit — per-position SL triggered | 2024-12-16 11:30:00 | 1443.04 | 1443.70 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:00:00 | 1477.75 | 1469.20 | 0.00 | ORB-long ORB[1453.20,1466.90] vol=1.6x ATR=3.94 |
| Stop hit — per-position SL triggered | 2024-12-18 11:05:00 | 1473.81 | 1469.36 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:40:00 | 1484.35 | 1468.11 | 0.00 | ORB-long ORB[1449.90,1472.00] vol=1.9x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:50:00 | 1490.64 | 1471.68 | 0.00 | T1 1.5R @ 1490.64 |
| Stop hit — per-position SL triggered | 2024-12-19 11:00:00 | 1484.35 | 1472.95 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 1482.25 | 1472.94 | 0.00 | ORB-long ORB[1469.40,1478.95] vol=1.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-12-26 12:25:00 | 1479.08 | 1476.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 1505.90 | 1496.01 | 0.00 | ORB-long ORB[1484.00,1494.20] vol=1.5x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-12-27 09:55:00 | 1501.86 | 1499.00 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 1515.95 | 1506.72 | 0.00 | ORB-long ORB[1495.10,1509.00] vol=1.6x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:55:00 | 1521.58 | 1509.19 | 0.00 | T1 1.5R @ 1521.58 |
| Target hit | 2024-12-30 15:20:00 | 1520.60 | 1517.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 1490.15 | 1493.49 | 0.00 | ORB-short ORB[1491.30,1509.25] vol=2.5x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1493.91 | 1492.90 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:10:00 | 1502.55 | 1495.56 | 0.00 | ORB-long ORB[1488.20,1501.50] vol=1.7x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 10:50:00 | 1508.97 | 1497.77 | 0.00 | T1 1.5R @ 1508.97 |
| Stop hit — per-position SL triggered | 2025-01-07 12:05:00 | 1502.55 | 1500.27 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 1448.00 | 1453.63 | 0.00 | ORB-short ORB[1451.55,1463.95] vol=3.2x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-01-13 11:45:00 | 1452.13 | 1451.65 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:10:00 | 1451.45 | 1443.24 | 0.00 | ORB-long ORB[1435.25,1450.45] vol=2.0x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 1448.01 | 1443.47 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:50:00 | 1452.60 | 1444.71 | 0.00 | ORB-long ORB[1432.55,1441.95] vol=1.5x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:05:00 | 1457.20 | 1445.86 | 0.00 | T1 1.5R @ 1457.20 |
| Stop hit — per-position SL triggered | 2025-01-23 11:45:00 | 1452.60 | 1447.90 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:45:00 | 1435.80 | 1438.38 | 0.00 | ORB-short ORB[1436.55,1451.00] vol=2.8x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-01-24 11:00:00 | 1439.42 | 1438.23 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:50:00 | 1400.90 | 1407.96 | 0.00 | ORB-short ORB[1405.15,1414.90] vol=2.4x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 1393.60 | 1404.69 | 0.00 | T1 1.5R @ 1393.60 |
| Target hit | 2025-01-27 11:50:00 | 1398.60 | 1398.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 1369.00 | 1378.14 | 0.00 | ORB-short ORB[1376.40,1396.05] vol=2.2x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-01-28 09:40:00 | 1373.71 | 1376.84 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:05:00 | 1458.50 | 1447.37 | 0.00 | ORB-long ORB[1426.00,1445.00] vol=1.7x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:50:00 | 1464.94 | 1452.04 | 0.00 | T1 1.5R @ 1464.94 |
| Stop hit — per-position SL triggered | 2025-01-30 12:45:00 | 1458.50 | 1455.59 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:30:00 | 1485.90 | 1474.88 | 0.00 | ORB-long ORB[1459.90,1479.15] vol=1.5x ATR=5.84 |
| Stop hit — per-position SL triggered | 2025-01-31 09:40:00 | 1480.06 | 1477.47 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-03 09:30:00 | 1438.45 | 1427.63 | 0.00 | ORB-long ORB[1415.35,1436.50] vol=1.8x ATR=6.21 |
| Stop hit — per-position SL triggered | 2025-02-03 09:35:00 | 1432.24 | 1428.34 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 1466.50 | 1467.53 | 0.00 | ORB-short ORB[1470.20,1481.85] vol=2.4x ATR=3.83 |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 1470.33 | 1467.56 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:20:00 | 1445.55 | 1454.21 | 0.00 | ORB-short ORB[1460.10,1480.15] vol=1.8x ATR=5.26 |
| Stop hit — per-position SL triggered | 2025-02-14 10:25:00 | 1450.81 | 1453.85 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:00:00 | 1450.90 | 1456.80 | 0.00 | ORB-short ORB[1452.00,1464.00] vol=1.9x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:05:00 | 1446.11 | 1455.94 | 0.00 | T1 1.5R @ 1446.11 |
| Target hit | 2025-02-27 15:20:00 | 1440.45 | 1445.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:20:00 | 1479.95 | 1472.68 | 0.00 | ORB-long ORB[1457.00,1472.70] vol=1.7x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-03-10 10:30:00 | 1476.45 | 1473.34 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:00:00 | 1515.20 | 1509.26 | 0.00 | ORB-long ORB[1495.45,1507.45] vol=1.7x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-03-20 11:35:00 | 1512.48 | 1510.34 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 1523.95 | 1519.32 | 0.00 | ORB-long ORB[1510.00,1519.10] vol=2.1x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-03-21 10:10:00 | 1520.35 | 1520.20 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 1531.15 | 1523.78 | 0.00 | ORB-long ORB[1516.55,1528.35] vol=2.0x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-03-24 12:35:00 | 1527.84 | 1526.26 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:50:00 | 1498.80 | 1504.01 | 0.00 | ORB-short ORB[1503.80,1512.35] vol=3.4x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:05:00 | 1493.82 | 1503.01 | 0.00 | T1 1.5R @ 1493.82 |
| Target hit | 2025-03-26 15:20:00 | 1475.00 | 1489.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 11:15:00 | 1466.60 | 1475.21 | 0.00 | ORB-short ORB[1467.90,1484.60] vol=1.5x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-04-15 13:25:00 | 1470.11 | 1472.01 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1506.30 | 1500.59 | 0.00 | ORB-long ORB[1486.00,1504.50] vol=2.0x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:30:00 | 1511.12 | 1501.77 | 0.00 | T1 1.5R @ 1511.12 |
| Target hit | 2025-04-17 15:20:00 | 1513.10 | 1511.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2025-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:40:00 | 1518.90 | 1511.39 | 0.00 | ORB-long ORB[1507.10,1517.50] vol=2.0x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:00:00 | 1524.28 | 1514.52 | 0.00 | T1 1.5R @ 1524.28 |
| Target hit | 2025-04-22 15:20:00 | 1527.90 | 1523.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1547.60 | 1539.73 | 0.00 | ORB-long ORB[1524.90,1543.00] vol=2.1x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-04-24 09:35:00 | 1543.52 | 1540.30 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-04-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 11:10:00 | 1524.50 | 1537.94 | 0.00 | ORB-short ORB[1554.50,1566.70] vol=4.8x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:40:00 | 1516.27 | 1536.08 | 0.00 | T1 1.5R @ 1516.27 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 1524.50 | 1533.74 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:35:00 | 1567.80 | 1558.55 | 0.00 | ORB-long ORB[1538.00,1560.00] vol=2.0x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-04-30 09:40:00 | 1563.27 | 1559.02 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:10:00 | 1539.30 | 1535.53 | 0.00 | ORB-long ORB[1528.30,1538.00] vol=1.6x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 11:30:00 | 1544.58 | 1536.44 | 0.00 | T1 1.5R @ 1544.58 |
| Stop hit — per-position SL triggered | 2025-05-05 12:25:00 | 1539.30 | 1539.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 1394.15 | 2024-05-16 09:40:00 | 1398.28 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-17 10:55:00 | 1405.60 | 2024-05-17 12:25:00 | 1400.44 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-05-17 10:55:00 | 1405.60 | 2024-05-17 15:20:00 | 1399.80 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-05-21 10:00:00 | 1410.15 | 2024-05-21 10:05:00 | 1415.35 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-21 10:00:00 | 1410.15 | 2024-05-21 15:20:00 | 1443.70 | TARGET_HIT | 0.50 | 2.38% |
| BUY | retest1 | 2024-05-22 10:00:00 | 1453.20 | 2024-05-22 10:15:00 | 1459.97 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-22 10:00:00 | 1453.20 | 2024-05-22 10:30:00 | 1453.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-28 09:45:00 | 1493.00 | 2024-05-28 09:55:00 | 1489.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-29 10:55:00 | 1484.15 | 2024-05-29 11:25:00 | 1488.14 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-05-29 10:55:00 | 1484.15 | 2024-05-29 12:05:00 | 1484.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 10:00:00 | 1479.95 | 2024-05-30 11:30:00 | 1474.55 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-30 10:00:00 | 1479.95 | 2024-05-30 15:20:00 | 1471.50 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-12 11:00:00 | 1549.00 | 2024-06-12 11:05:00 | 1545.87 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-18 09:45:00 | 1574.25 | 2024-06-18 10:05:00 | 1571.36 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-06-25 11:00:00 | 1503.30 | 2024-06-25 11:10:00 | 1506.03 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-06-26 10:05:00 | 1480.75 | 2024-06-26 10:10:00 | 1485.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-27 09:40:00 | 1476.00 | 2024-06-27 13:00:00 | 1470.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-27 09:40:00 | 1476.00 | 2024-06-27 14:50:00 | 1476.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:25:00 | 1483.55 | 2024-07-04 11:00:00 | 1479.83 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-08 10:00:00 | 1509.00 | 2024-07-08 10:10:00 | 1513.17 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-10 10:40:00 | 1494.90 | 2024-07-10 10:45:00 | 1498.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-11 11:05:00 | 1499.05 | 2024-07-11 11:40:00 | 1502.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-15 09:45:00 | 1528.15 | 2024-07-15 10:35:00 | 1523.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-18 09:50:00 | 1510.70 | 2024-07-18 10:40:00 | 1506.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-22 10:30:00 | 1501.00 | 2024-07-22 12:40:00 | 1496.79 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-24 10:35:00 | 1498.00 | 2024-07-24 11:35:00 | 1503.83 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-24 10:35:00 | 1498.00 | 2024-07-24 12:20:00 | 1498.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:20:00 | 1535.00 | 2024-07-26 11:00:00 | 1543.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-26 10:20:00 | 1535.00 | 2024-07-26 11:35:00 | 1535.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 09:30:00 | 1586.50 | 2024-08-12 09:40:00 | 1591.77 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-12 09:30:00 | 1586.50 | 2024-08-12 09:45:00 | 1586.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 10:00:00 | 1604.25 | 2024-08-13 10:15:00 | 1599.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-21 10:55:00 | 1586.85 | 2024-08-21 11:05:00 | 1583.84 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-22 11:00:00 | 1580.10 | 2024-08-22 11:10:00 | 1582.38 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-08-26 11:00:00 | 1585.85 | 2024-08-26 14:15:00 | 1589.90 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-08-26 11:00:00 | 1585.85 | 2024-08-26 15:20:00 | 1593.35 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-08-27 09:35:00 | 1608.00 | 2024-08-27 10:00:00 | 1613.37 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-27 09:35:00 | 1608.00 | 2024-08-27 11:30:00 | 1608.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:30:00 | 1640.65 | 2024-08-29 09:40:00 | 1635.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-30 09:35:00 | 1630.15 | 2024-08-30 09:50:00 | 1637.04 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-30 09:35:00 | 1630.15 | 2024-08-30 10:20:00 | 1631.00 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-09-03 09:30:00 | 1655.55 | 2024-09-03 09:40:00 | 1651.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 11:05:00 | 1619.60 | 2024-09-06 11:40:00 | 1624.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-09 11:00:00 | 1633.00 | 2024-09-09 12:00:00 | 1629.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-11 09:35:00 | 1646.10 | 2024-09-11 09:45:00 | 1642.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-18 10:10:00 | 1651.55 | 2024-09-18 10:15:00 | 1655.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-19 09:35:00 | 1653.30 | 2024-09-19 09:45:00 | 1647.61 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-19 09:35:00 | 1653.30 | 2024-09-19 10:20:00 | 1653.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-20 10:25:00 | 1616.25 | 2024-09-20 10:30:00 | 1621.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-03 11:00:00 | 1673.50 | 2024-10-03 11:15:00 | 1668.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-08 11:15:00 | 1650.75 | 2024-10-08 11:55:00 | 1645.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-17 09:55:00 | 1577.00 | 2024-10-17 10:00:00 | 1572.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-21 10:30:00 | 1536.05 | 2024-10-21 11:45:00 | 1529.91 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-21 10:30:00 | 1536.05 | 2024-10-21 15:20:00 | 1524.00 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-10-22 11:05:00 | 1506.40 | 2024-10-22 11:25:00 | 1510.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-25 09:50:00 | 1487.70 | 2024-10-25 10:10:00 | 1480.45 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-25 09:50:00 | 1487.70 | 2024-10-25 10:20:00 | 1487.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 11:15:00 | 1593.30 | 2024-11-06 12:05:00 | 1587.36 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-11-06 11:15:00 | 1593.30 | 2024-11-06 14:15:00 | 1593.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 11:05:00 | 1579.80 | 2024-11-08 11:15:00 | 1576.51 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-12 11:15:00 | 1542.20 | 2024-11-12 11:20:00 | 1546.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-22 10:40:00 | 1486.45 | 2024-11-22 10:45:00 | 1482.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-03 10:55:00 | 1540.70 | 2024-12-03 11:05:00 | 1537.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-04 11:00:00 | 1506.85 | 2024-12-04 11:15:00 | 1510.17 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 10:15:00 | 1482.65 | 2024-12-05 10:35:00 | 1477.13 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-05 10:15:00 | 1482.65 | 2024-12-05 12:05:00 | 1482.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:45:00 | 1453.90 | 2024-12-12 09:50:00 | 1456.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-16 11:05:00 | 1440.20 | 2024-12-16 11:30:00 | 1443.04 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-18 11:00:00 | 1477.75 | 2024-12-18 11:05:00 | 1473.81 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-19 10:40:00 | 1484.35 | 2024-12-19 10:50:00 | 1490.64 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-19 10:40:00 | 1484.35 | 2024-12-19 11:00:00 | 1484.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 11:05:00 | 1482.25 | 2024-12-26 12:25:00 | 1479.08 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-27 09:40:00 | 1505.90 | 2024-12-27 09:55:00 | 1501.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-30 10:35:00 | 1515.95 | 2024-12-30 10:55:00 | 1521.58 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-30 10:35:00 | 1515.95 | 2024-12-30 15:20:00 | 1520.60 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-06 11:10:00 | 1490.15 | 2025-01-06 11:30:00 | 1493.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-07 10:10:00 | 1502.55 | 2025-01-07 10:50:00 | 1508.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-07 10:10:00 | 1502.55 | 2025-01-07 12:05:00 | 1502.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-13 11:10:00 | 1448.00 | 2025-01-13 11:45:00 | 1452.13 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-14 11:10:00 | 1451.45 | 2025-01-14 11:15:00 | 1448.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-23 10:50:00 | 1452.60 | 2025-01-23 11:05:00 | 1457.20 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-01-23 10:50:00 | 1452.60 | 2025-01-23 11:45:00 | 1452.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 10:45:00 | 1435.80 | 2025-01-24 11:00:00 | 1439.42 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-27 09:50:00 | 1400.90 | 2025-01-27 10:15:00 | 1393.60 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-27 09:50:00 | 1400.90 | 2025-01-27 11:50:00 | 1398.60 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-01-28 09:30:00 | 1369.00 | 2025-01-28 09:40:00 | 1373.71 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 11:05:00 | 1458.50 | 2025-01-30 11:50:00 | 1464.94 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-30 11:05:00 | 1458.50 | 2025-01-30 12:45:00 | 1458.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 09:30:00 | 1485.90 | 2025-01-31 09:40:00 | 1480.06 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-03 09:30:00 | 1438.45 | 2025-02-03 09:35:00 | 1432.24 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-02-07 11:10:00 | 1466.50 | 2025-02-07 11:15:00 | 1470.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-14 10:20:00 | 1445.55 | 2025-02-14 10:25:00 | 1450.81 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-27 11:00:00 | 1450.90 | 2025-02-27 11:05:00 | 1446.11 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-02-27 11:00:00 | 1450.90 | 2025-02-27 15:20:00 | 1440.45 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2025-03-10 10:20:00 | 1479.95 | 2025-03-10 10:30:00 | 1476.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-20 11:00:00 | 1515.20 | 2025-03-20 11:35:00 | 1512.48 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-03-21 09:40:00 | 1523.95 | 2025-03-21 10:10:00 | 1520.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-24 11:10:00 | 1531.15 | 2025-03-24 12:35:00 | 1527.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-03-26 10:50:00 | 1498.80 | 2025-03-26 11:05:00 | 1493.82 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-03-26 10:50:00 | 1498.80 | 2025-03-26 15:20:00 | 1475.00 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2025-04-15 11:15:00 | 1466.60 | 2025-04-15 13:25:00 | 1470.11 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1506.30 | 2025-04-17 11:30:00 | 1511.12 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1506.30 | 2025-04-17 15:20:00 | 1513.10 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2025-04-22 10:40:00 | 1518.90 | 2025-04-22 11:00:00 | 1524.28 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-04-22 10:40:00 | 1518.90 | 2025-04-22 15:20:00 | 1527.90 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-24 09:30:00 | 1547.60 | 2025-04-24 09:35:00 | 1543.52 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-25 11:10:00 | 1524.50 | 2025-04-25 11:40:00 | 1516.27 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-25 11:10:00 | 1524.50 | 2025-04-25 12:15:00 | 1524.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 09:35:00 | 1567.80 | 2025-04-30 09:40:00 | 1563.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-05 11:10:00 | 1539.30 | 2025-05-05 11:30:00 | 1544.58 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-05 11:10:00 | 1539.30 | 2025-05-05 12:25:00 | 1539.30 | STOP_HIT | 0.50 | 0.00% |
