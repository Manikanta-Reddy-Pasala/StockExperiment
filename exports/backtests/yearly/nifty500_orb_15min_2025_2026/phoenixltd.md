# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1845.00
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
| ENTRY1 | 72 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 11 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 61
- **Target hits / Stop hits / Partials:** 11 / 61 / 24
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 2.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 27 | 45.8% | 11 | 32 | 16 | 0.09% | 5.1% |
| BUY @ 2nd Alert (retest1) | 59 | 27 | 45.8% | 11 | 32 | 16 | 0.09% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 8 | 21.6% | 0 | 29 | 8 | -0.07% | -2.6% |
| SELL @ 2nd Alert (retest1) | 37 | 8 | 21.6% | 0 | 29 | 8 | -0.07% | -2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 35 | 36.5% | 11 | 61 | 24 | 0.03% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 1621.30 | 1607.53 | 0.00 | ORB-long ORB[1590.50,1612.40] vol=1.7x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-05-26 09:40:00 | 1616.07 | 1610.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1587.80 | 1595.13 | 0.00 | ORB-short ORB[1590.00,1609.70] vol=2.7x ATR=6.04 |
| Stop hit — per-position SL triggered | 2025-05-28 10:00:00 | 1593.84 | 1592.48 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1568.30 | 1549.17 | 0.00 | ORB-long ORB[1525.70,1546.70] vol=1.7x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:35:00 | 1577.60 | 1553.35 | 0.00 | T1 1.5R @ 1577.60 |
| Target hit | 2025-06-02 15:20:00 | 1586.50 | 1576.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:00:00 | 1588.40 | 1592.51 | 0.00 | ORB-short ORB[1599.30,1613.90] vol=7.5x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:10:00 | 1581.26 | 1592.18 | 0.00 | T1 1.5R @ 1581.26 |
| Stop hit — per-position SL triggered | 2025-06-04 15:00:00 | 1588.40 | 1586.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:55:00 | 1660.00 | 1654.20 | 0.00 | ORB-long ORB[1631.10,1656.00] vol=1.6x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 10:10:00 | 1669.01 | 1657.45 | 0.00 | T1 1.5R @ 1669.01 |
| Target hit | 2025-06-11 10:35:00 | 1665.60 | 1669.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1641.90 | 1634.58 | 0.00 | ORB-long ORB[1619.60,1639.10] vol=2.8x ATR=6.22 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 1635.68 | 1635.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:30:00 | 1569.00 | 1587.02 | 0.00 | ORB-short ORB[1592.20,1610.20] vol=1.7x ATR=5.02 |
| Stop hit — per-position SL triggered | 2025-06-19 10:45:00 | 1574.02 | 1584.70 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:30:00 | 1599.30 | 1580.87 | 0.00 | ORB-long ORB[1558.30,1578.40] vol=1.7x ATR=7.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:45:00 | 1610.27 | 1588.74 | 0.00 | T1 1.5R @ 1610.27 |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 1599.30 | 1589.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 09:55:00 | 1623.70 | 1618.39 | 0.00 | ORB-long ORB[1596.30,1618.20] vol=9.9x ATR=7.86 |
| Stop hit — per-position SL triggered | 2025-06-23 10:40:00 | 1615.84 | 1618.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:05:00 | 1586.40 | 1576.02 | 0.00 | ORB-long ORB[1572.20,1582.00] vol=2.6x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-06-30 11:30:00 | 1582.01 | 1578.08 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:00:00 | 1514.60 | 1500.98 | 0.00 | ORB-long ORB[1486.40,1508.40] vol=1.8x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-07-03 11:45:00 | 1509.56 | 1501.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 11:00:00 | 1552.40 | 1545.21 | 0.00 | ORB-long ORB[1530.40,1542.90] vol=2.5x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-07-07 11:05:00 | 1548.44 | 1545.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:30:00 | 1495.10 | 1502.62 | 0.00 | ORB-short ORB[1503.60,1513.00] vol=1.8x ATR=3.43 |
| Stop hit — per-position SL triggered | 2025-07-11 10:35:00 | 1498.53 | 1502.44 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 11:00:00 | 1497.30 | 1501.73 | 0.00 | ORB-short ORB[1505.00,1518.50] vol=1.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-07-15 12:25:00 | 1500.54 | 1499.27 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1455.20 | 1457.75 | 0.00 | ORB-short ORB[1468.80,1481.40] vol=8.9x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-07-23 10:25:00 | 1459.46 | 1457.63 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1467.70 | 1454.78 | 0.00 | ORB-long ORB[1442.80,1460.60] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-07-24 09:55:00 | 1462.92 | 1455.69 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 11:00:00 | 1523.00 | 1510.33 | 0.00 | ORB-long ORB[1497.10,1515.00] vol=1.7x ATR=7.44 |
| Stop hit — per-position SL triggered | 2025-07-28 12:55:00 | 1515.56 | 1514.87 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:30:00 | 1470.50 | 1481.36 | 0.00 | ORB-short ORB[1481.00,1492.40] vol=1.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-07-31 10:35:00 | 1474.71 | 1480.75 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 09:50:00 | 1442.40 | 1444.88 | 0.00 | ORB-short ORB[1445.10,1458.50] vol=5.9x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:40:00 | 1435.16 | 1443.90 | 0.00 | T1 1.5R @ 1435.16 |
| Stop hit — per-position SL triggered | 2025-08-08 10:50:00 | 1442.40 | 1443.32 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:00:00 | 1431.80 | 1421.50 | 0.00 | ORB-long ORB[1402.50,1418.40] vol=2.2x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:20:00 | 1438.75 | 1422.33 | 0.00 | T1 1.5R @ 1438.75 |
| Target hit | 2025-08-11 11:50:00 | 1433.00 | 1433.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2025-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:05:00 | 1426.70 | 1433.50 | 0.00 | ORB-short ORB[1437.30,1448.00] vol=1.7x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:25:00 | 1421.37 | 1432.32 | 0.00 | T1 1.5R @ 1421.37 |
| Stop hit — per-position SL triggered | 2025-08-14 11:45:00 | 1426.70 | 1431.62 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:55:00 | 1491.40 | 1481.26 | 0.00 | ORB-long ORB[1475.00,1490.60] vol=1.8x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:00:00 | 1497.67 | 1482.57 | 0.00 | T1 1.5R @ 1497.67 |
| Stop hit — per-position SL triggered | 2025-08-19 11:05:00 | 1491.40 | 1482.83 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 1532.30 | 1518.04 | 0.00 | ORB-long ORB[1503.30,1522.00] vol=1.6x ATR=5.02 |
| Stop hit — per-position SL triggered | 2025-08-20 10:35:00 | 1527.28 | 1522.27 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:40:00 | 1591.40 | 1584.98 | 0.00 | ORB-long ORB[1566.20,1588.30] vol=2.6x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:20:00 | 1598.94 | 1588.25 | 0.00 | T1 1.5R @ 1598.94 |
| Stop hit — per-position SL triggered | 2025-08-25 11:25:00 | 1591.40 | 1588.41 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:20:00 | 1516.80 | 1520.40 | 0.00 | ORB-short ORB[1518.00,1540.70] vol=10.4x ATR=4.99 |
| Stop hit — per-position SL triggered | 2025-09-04 10:25:00 | 1521.79 | 1520.30 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:55:00 | 1552.20 | 1560.66 | 0.00 | ORB-short ORB[1555.00,1565.10] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-09-11 11:05:00 | 1555.37 | 1560.38 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1555.50 | 1561.91 | 0.00 | ORB-short ORB[1557.50,1566.10] vol=2.0x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:05:00 | 1551.06 | 1560.70 | 0.00 | T1 1.5R @ 1551.06 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1555.50 | 1560.57 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:00:00 | 1625.80 | 1620.95 | 0.00 | ORB-long ORB[1608.00,1621.90] vol=2.2x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:25:00 | 1631.75 | 1625.61 | 0.00 | T1 1.5R @ 1631.75 |
| Target hit | 2025-09-18 11:05:00 | 1630.40 | 1632.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 1618.00 | 1625.72 | 0.00 | ORB-short ORB[1619.00,1632.80] vol=2.6x ATR=4.16 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 1622.16 | 1624.81 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:50:00 | 1653.70 | 1642.98 | 0.00 | ORB-long ORB[1625.00,1639.60] vol=2.3x ATR=4.50 |
| Stop hit — per-position SL triggered | 2025-09-22 11:10:00 | 1649.20 | 1644.31 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:15:00 | 1620.80 | 1632.68 | 0.00 | ORB-short ORB[1632.30,1646.90] vol=1.6x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-09-23 12:35:00 | 1625.51 | 1629.96 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 1603.70 | 1608.18 | 0.00 | ORB-short ORB[1604.50,1620.00] vol=1.5x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:40:00 | 1597.16 | 1606.17 | 0.00 | T1 1.5R @ 1597.16 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 1603.70 | 1599.54 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:10:00 | 1570.10 | 1572.49 | 0.00 | ORB-short ORB[1575.00,1585.90] vol=1.9x ATR=4.70 |
| Stop hit — per-position SL triggered | 2025-09-25 11:20:00 | 1574.80 | 1572.40 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 1564.20 | 1557.44 | 0.00 | ORB-long ORB[1544.50,1563.10] vol=2.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-09-26 11:10:00 | 1559.19 | 1557.83 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:10:00 | 1562.50 | 1556.05 | 0.00 | ORB-long ORB[1542.50,1555.30] vol=8.4x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 12:20:00 | 1570.03 | 1558.33 | 0.00 | T1 1.5R @ 1570.03 |
| Stop hit — per-position SL triggered | 2025-10-01 15:10:00 | 1562.50 | 1562.86 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:35:00 | 1601.00 | 1594.24 | 0.00 | ORB-long ORB[1585.00,1594.30] vol=2.0x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:05:00 | 1608.23 | 1597.55 | 0.00 | T1 1.5R @ 1608.23 |
| Stop hit — per-position SL triggered | 2025-10-10 10:35:00 | 1601.00 | 1599.35 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 1674.50 | 1668.45 | 0.00 | ORB-long ORB[1652.00,1674.30] vol=1.6x ATR=6.46 |
| Stop hit — per-position SL triggered | 2025-10-23 09:45:00 | 1668.04 | 1668.79 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:20:00 | 1701.00 | 1694.87 | 0.00 | ORB-long ORB[1686.60,1700.00] vol=1.8x ATR=6.03 |
| Stop hit — per-position SL triggered | 2025-10-29 11:30:00 | 1694.97 | 1697.88 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 1688.20 | 1697.45 | 0.00 | ORB-short ORB[1688.50,1707.50] vol=2.1x ATR=4.40 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 1692.60 | 1697.19 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:25:00 | 1757.70 | 1745.80 | 0.00 | ORB-long ORB[1731.10,1752.70] vol=2.6x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:40:00 | 1767.15 | 1749.39 | 0.00 | T1 1.5R @ 1767.15 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 1757.70 | 1751.16 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 11:10:00 | 1742.70 | 1728.73 | 0.00 | ORB-long ORB[1720.20,1738.00] vol=3.1x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-11-17 11:20:00 | 1738.66 | 1729.56 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:55:00 | 1722.10 | 1725.72 | 0.00 | ORB-short ORB[1725.30,1749.00] vol=3.2x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-11-18 11:10:00 | 1726.37 | 1725.43 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 1702.10 | 1713.23 | 0.00 | ORB-short ORB[1707.50,1727.80] vol=2.1x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:20:00 | 1694.83 | 1710.51 | 0.00 | T1 1.5R @ 1694.83 |
| Stop hit — per-position SL triggered | 2025-11-19 11:30:00 | 1702.10 | 1708.66 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:45:00 | 1740.70 | 1743.54 | 0.00 | ORB-short ORB[1740.90,1749.60] vol=1.6x ATR=3.32 |
| Stop hit — per-position SL triggered | 2025-11-27 09:55:00 | 1744.02 | 1743.43 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1732.50 | 1736.85 | 0.00 | ORB-short ORB[1736.00,1751.00] vol=6.9x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:20:00 | 1726.61 | 1736.61 | 0.00 | T1 1.5R @ 1726.61 |
| Stop hit — per-position SL triggered | 2025-11-28 10:25:00 | 1732.50 | 1736.51 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:50:00 | 1704.70 | 1718.55 | 0.00 | ORB-short ORB[1724.60,1737.90] vol=1.6x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-12-03 11:20:00 | 1708.71 | 1715.44 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:55:00 | 1730.80 | 1724.15 | 0.00 | ORB-long ORB[1715.60,1730.00] vol=2.8x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-12-04 11:45:00 | 1727.30 | 1725.65 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 1738.70 | 1730.09 | 0.00 | ORB-long ORB[1709.90,1734.10] vol=1.6x ATR=4.44 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 1734.26 | 1730.41 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 11:05:00 | 1741.80 | 1737.83 | 0.00 | ORB-long ORB[1725.00,1740.90] vol=2.0x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:35:00 | 1748.53 | 1739.73 | 0.00 | T1 1.5R @ 1748.53 |
| Target hit | 2025-12-10 14:45:00 | 1753.50 | 1754.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2025-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:05:00 | 1755.30 | 1752.43 | 0.00 | ORB-long ORB[1727.10,1752.70] vol=2.1x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-12-12 11:45:00 | 1750.49 | 1752.56 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:10:00 | 1770.60 | 1762.87 | 0.00 | ORB-long ORB[1746.10,1769.60] vol=2.6x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 12:00:00 | 1778.64 | 1768.84 | 0.00 | T1 1.5R @ 1778.64 |
| Target hit | 2025-12-15 15:20:00 | 1787.90 | 1779.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:00:00 | 1789.50 | 1775.52 | 0.00 | ORB-long ORB[1760.00,1782.60] vol=2.9x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 12:00:00 | 1797.20 | 1778.29 | 0.00 | T1 1.5R @ 1797.20 |
| Target hit | 2025-12-18 14:15:00 | 1793.40 | 1793.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2025-12-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:05:00 | 1820.00 | 1813.74 | 0.00 | ORB-long ORB[1790.60,1812.80] vol=8.0x ATR=6.16 |
| Stop hit — per-position SL triggered | 2025-12-19 10:10:00 | 1813.84 | 1813.82 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:45:00 | 1864.10 | 1853.35 | 0.00 | ORB-long ORB[1834.70,1857.90] vol=2.6x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-12-24 11:05:00 | 1859.44 | 1855.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 1857.80 | 1847.38 | 0.00 | ORB-long ORB[1835.00,1852.60] vol=3.0x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-12-29 09:35:00 | 1852.52 | 1848.22 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 1869.70 | 1860.77 | 0.00 | ORB-long ORB[1846.60,1857.10] vol=2.1x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-01-01 11:25:00 | 1865.68 | 1863.33 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 1875.20 | 1874.51 | 0.00 | ORB-long ORB[1860.30,1874.00] vol=4.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:50:00 | 1881.39 | 1876.03 | 0.00 | T1 1.5R @ 1881.39 |
| Target hit | 2026-01-02 11:40:00 | 1890.20 | 1890.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — BUY (started 2026-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:45:00 | 1933.80 | 1932.05 | 0.00 | ORB-long ORB[1914.00,1931.40] vol=9.1x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-01-06 11:05:00 | 1929.66 | 1931.65 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 1923.00 | 1931.30 | 0.00 | ORB-short ORB[1930.10,1946.50] vol=1.8x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 1927.30 | 1930.29 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:55:00 | 1863.90 | 1874.22 | 0.00 | ORB-short ORB[1870.40,1889.50] vol=2.1x ATR=6.78 |
| Stop hit — per-position SL triggered | 2026-01-14 10:20:00 | 1870.68 | 1869.59 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 1748.90 | 1745.77 | 0.00 | ORB-long ORB[1725.10,1747.90] vol=3.9x ATR=5.24 |
| Target hit | 2026-02-09 15:20:00 | 1750.00 | 1749.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 1800.90 | 1770.53 | 0.00 | ORB-long ORB[1743.80,1768.20] vol=2.9x ATR=6.00 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 1794.90 | 1777.99 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:50:00 | 1747.70 | 1756.68 | 0.00 | ORB-short ORB[1755.00,1772.50] vol=2.5x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:00:00 | 1739.88 | 1754.22 | 0.00 | T1 1.5R @ 1739.88 |
| Stop hit — per-position SL triggered | 2026-02-13 11:40:00 | 1747.70 | 1745.48 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-02-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:10:00 | 1746.00 | 1740.36 | 0.00 | ORB-long ORB[1720.00,1744.80] vol=2.6x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 1741.44 | 1740.56 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1741.30 | 1753.51 | 0.00 | ORB-short ORB[1755.00,1775.00] vol=2.0x ATR=5.36 |
| Stop hit — per-position SL triggered | 2026-02-18 10:05:00 | 1746.66 | 1751.26 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1737.20 | 1750.61 | 0.00 | ORB-short ORB[1753.90,1768.20] vol=1.9x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-19 12:00:00 | 1742.00 | 1743.98 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1694.20 | 1694.44 | 0.00 | ORB-short ORB[1697.50,1710.70] vol=3.3x ATR=6.23 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 1700.43 | 1694.51 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1603.80 | 1615.13 | 0.00 | ORB-short ORB[1604.10,1626.00] vol=1.9x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-03-06 11:30:00 | 1607.92 | 1614.11 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 1564.20 | 1557.71 | 0.00 | ORB-long ORB[1533.90,1552.30] vol=1.6x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 1557.68 | 1558.69 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 1744.10 | 1729.42 | 0.00 | ORB-long ORB[1712.60,1735.30] vol=1.6x ATR=9.43 |
| Stop hit — per-position SL triggered | 2026-04-10 09:45:00 | 1734.67 | 1732.89 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 1787.00 | 1779.75 | 0.00 | ORB-long ORB[1768.90,1784.20] vol=1.8x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 1796.02 | 1784.06 | 0.00 | T1 1.5R @ 1796.02 |
| Target hit | 2026-04-29 14:10:00 | 1807.00 | 1812.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1802.20 | 1791.47 | 0.00 | ORB-long ORB[1782.70,1798.00] vol=2.1x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:45:00 | 1811.27 | 1799.23 | 0.00 | T1 1.5R @ 1811.27 |
| Target hit | 2026-05-06 13:00:00 | 1804.40 | 1804.98 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-26 09:30:00 | 1621.30 | 2025-05-26 09:40:00 | 1616.07 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-28 09:30:00 | 1587.80 | 2025-05-28 10:00:00 | 1593.84 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-02 11:00:00 | 1568.30 | 2025-06-02 11:35:00 | 1577.60 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-06-02 11:00:00 | 1568.30 | 2025-06-02 15:20:00 | 1586.50 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-06-04 11:00:00 | 1588.40 | 2025-06-04 11:10:00 | 1581.26 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-04 11:00:00 | 1588.40 | 2025-06-04 15:00:00 | 1588.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 09:55:00 | 1660.00 | 2025-06-11 10:10:00 | 1669.01 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-11 09:55:00 | 1660.00 | 2025-06-11 10:35:00 | 1665.60 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1641.90 | 2025-06-17 09:40:00 | 1635.68 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-19 10:30:00 | 1569.00 | 2025-06-19 10:45:00 | 1574.02 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-20 10:30:00 | 1599.30 | 2025-06-20 11:45:00 | 1610.27 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-06-20 10:30:00 | 1599.30 | 2025-06-20 12:15:00 | 1599.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-23 09:55:00 | 1623.70 | 2025-06-23 10:40:00 | 1615.84 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-06-30 11:05:00 | 1586.40 | 2025-06-30 11:30:00 | 1582.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-03 11:00:00 | 1514.60 | 2025-07-03 11:45:00 | 1509.56 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-07 11:00:00 | 1552.40 | 2025-07-07 11:05:00 | 1548.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-11 10:30:00 | 1495.10 | 2025-07-11 10:35:00 | 1498.53 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-15 11:00:00 | 1497.30 | 2025-07-15 12:25:00 | 1500.54 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-23 10:00:00 | 1455.20 | 2025-07-23 10:25:00 | 1459.46 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-24 09:45:00 | 1467.70 | 2025-07-24 09:55:00 | 1462.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-28 11:00:00 | 1523.00 | 2025-07-28 12:55:00 | 1515.56 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-07-31 10:30:00 | 1470.50 | 2025-07-31 10:35:00 | 1474.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-08 09:50:00 | 1442.40 | 2025-08-08 10:40:00 | 1435.16 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-08-08 09:50:00 | 1442.40 | 2025-08-08 10:50:00 | 1442.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 11:00:00 | 1431.80 | 2025-08-11 11:20:00 | 1438.75 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-08-11 11:00:00 | 1431.80 | 2025-08-11 11:50:00 | 1433.00 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-08-14 11:05:00 | 1426.70 | 2025-08-14 11:25:00 | 1421.37 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-14 11:05:00 | 1426.70 | 2025-08-14 11:45:00 | 1426.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:55:00 | 1491.40 | 2025-08-19 11:00:00 | 1497.67 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-19 10:55:00 | 1491.40 | 2025-08-19 11:05:00 | 1491.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:10:00 | 1532.30 | 2025-08-20 10:35:00 | 1527.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-25 10:40:00 | 1591.40 | 2025-08-25 11:20:00 | 1598.94 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-25 10:40:00 | 1591.40 | 2025-08-25 11:25:00 | 1591.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-04 10:20:00 | 1516.80 | 2025-09-04 10:25:00 | 1521.79 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-11 10:55:00 | 1552.20 | 2025-09-11 11:05:00 | 1555.37 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-12 10:45:00 | 1555.50 | 2025-09-12 11:05:00 | 1551.06 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-12 10:45:00 | 1555.50 | 2025-09-12 11:15:00 | 1555.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 10:00:00 | 1625.80 | 2025-09-18 10:25:00 | 1631.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-18 10:00:00 | 1625.80 | 2025-09-18 11:05:00 | 1630.40 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-19 09:55:00 | 1618.00 | 2025-09-19 10:05:00 | 1622.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-22 10:50:00 | 1653.70 | 2025-09-22 11:10:00 | 1649.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-23 11:15:00 | 1620.80 | 2025-09-23 12:35:00 | 1625.51 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-24 09:35:00 | 1603.70 | 2025-09-24 09:40:00 | 1597.16 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-24 09:35:00 | 1603.70 | 2025-09-24 10:15:00 | 1603.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 11:10:00 | 1570.10 | 2025-09-25 11:20:00 | 1574.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-26 11:00:00 | 1564.20 | 2025-09-26 11:10:00 | 1559.19 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-01 11:10:00 | 1562.50 | 2025-10-01 12:20:00 | 1570.03 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-01 11:10:00 | 1562.50 | 2025-10-01 15:10:00 | 1562.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:35:00 | 1601.00 | 2025-10-10 10:05:00 | 1608.23 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-10-10 09:35:00 | 1601.00 | 2025-10-10 10:35:00 | 1601.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-23 09:30:00 | 1674.50 | 2025-10-23 09:45:00 | 1668.04 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-29 10:20:00 | 1701.00 | 2025-10-29 11:30:00 | 1694.97 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-31 11:10:00 | 1688.20 | 2025-10-31 11:20:00 | 1692.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-07 10:25:00 | 1757.70 | 2025-11-07 10:40:00 | 1767.15 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-11-07 10:25:00 | 1757.70 | 2025-11-07 10:55:00 | 1757.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-17 11:10:00 | 1742.70 | 2025-11-17 11:20:00 | 1738.66 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-18 10:55:00 | 1722.10 | 2025-11-18 11:10:00 | 1726.37 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-19 10:55:00 | 1702.10 | 2025-11-19 11:20:00 | 1694.83 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-19 10:55:00 | 1702.10 | 2025-11-19 11:30:00 | 1702.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 09:45:00 | 1740.70 | 2025-11-27 09:55:00 | 1744.02 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-28 10:15:00 | 1732.50 | 2025-11-28 10:20:00 | 1726.61 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-28 10:15:00 | 1732.50 | 2025-11-28 10:25:00 | 1732.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:50:00 | 1704.70 | 2025-12-03 11:20:00 | 1708.71 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-04 10:55:00 | 1730.80 | 2025-12-04 11:45:00 | 1727.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-05 10:00:00 | 1738.70 | 2025-12-05 10:05:00 | 1734.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-10 11:05:00 | 1741.80 | 2025-12-10 11:35:00 | 1748.53 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-10 11:05:00 | 1741.80 | 2025-12-10 14:45:00 | 1753.50 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2025-12-12 11:05:00 | 1755.30 | 2025-12-12 11:45:00 | 1750.49 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-15 10:10:00 | 1770.60 | 2025-12-15 12:00:00 | 1778.64 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-15 10:10:00 | 1770.60 | 2025-12-15 15:20:00 | 1787.90 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-12-18 11:00:00 | 1789.50 | 2025-12-18 12:00:00 | 1797.20 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-12-18 11:00:00 | 1789.50 | 2025-12-18 14:15:00 | 1793.40 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-12-19 10:05:00 | 1820.00 | 2025-12-19 10:10:00 | 1813.84 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-24 10:45:00 | 1864.10 | 2025-12-24 11:05:00 | 1859.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-29 09:30:00 | 1857.80 | 2025-12-29 09:35:00 | 1852.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-01 11:15:00 | 1869.70 | 2026-01-01 11:25:00 | 1865.68 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 09:35:00 | 1875.20 | 2026-01-02 09:50:00 | 1881.39 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-01-02 09:35:00 | 1875.20 | 2026-01-02 11:40:00 | 1890.20 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2026-01-06 10:45:00 | 1933.80 | 2026-01-06 11:05:00 | 1929.66 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-08 11:10:00 | 1923.00 | 2026-01-08 11:35:00 | 1927.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-14 09:55:00 | 1863.90 | 2026-01-14 10:20:00 | 1870.68 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-09 10:40:00 | 1748.90 | 2026-02-09 15:20:00 | 1750.00 | TARGET_HIT | 1.00 | 0.06% |
| BUY | retest1 | 2026-02-11 10:50:00 | 1800.90 | 2026-02-11 11:30:00 | 1794.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-13 09:50:00 | 1747.70 | 2026-02-13 10:00:00 | 1739.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-13 09:50:00 | 1747.70 | 2026-02-13 11:40:00 | 1747.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 11:10:00 | 1746.00 | 2026-02-16 11:25:00 | 1741.44 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1741.30 | 2026-02-18 10:05:00 | 1746.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1737.20 | 2026-02-19 12:00:00 | 1742.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1694.20 | 2026-02-27 10:35:00 | 1700.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1603.80 | 2026-03-06 11:30:00 | 1607.92 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-17 10:35:00 | 1564.20 | 2026-03-17 11:25:00 | 1557.68 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-10 09:35:00 | 1744.10 | 2026-04-10 09:45:00 | 1734.67 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1787.00 | 2026-04-29 11:15:00 | 1796.02 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-29 11:05:00 | 1787.00 | 2026-04-29 14:10:00 | 1807.00 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1802.20 | 2026-05-06 10:45:00 | 1811.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1802.20 | 2026-05-06 13:00:00 | 1804.40 | TARGET_HIT | 0.50 | 0.12% |
