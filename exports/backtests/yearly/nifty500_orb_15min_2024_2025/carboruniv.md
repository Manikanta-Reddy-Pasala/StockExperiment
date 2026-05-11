# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1020.20
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
| ENTRY1 | 64 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 16 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 48
- **Target hits / Stop hits / Partials:** 16 / 48 / 31
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 20.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 4 | 20 | 10 | 0.08% | 2.9% |
| BUY @ 2nd Alert (retest1) | 34 | 14 | 41.2% | 4 | 20 | 10 | 0.08% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 33 | 54.1% | 12 | 28 | 21 | 0.29% | 17.8% |
| SELL @ 2nd Alert (retest1) | 61 | 33 | 54.1% | 12 | 28 | 21 | 0.29% | 17.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 47 | 49.5% | 16 | 48 | 31 | 0.22% | 20.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 1527.50 | 1517.47 | 0.00 | ORB-long ORB[1499.35,1520.00] vol=1.6x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:10:00 | 1535.32 | 1521.08 | 0.00 | T1 1.5R @ 1535.32 |
| Stop hit — per-position SL triggered | 2024-05-15 10:15:00 | 1527.50 | 1521.86 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:55:00 | 1688.85 | 1699.70 | 0.00 | ORB-short ORB[1698.05,1714.95] vol=1.9x ATR=7.30 |
| Stop hit — per-position SL triggered | 2024-05-24 11:25:00 | 1696.15 | 1697.83 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:40:00 | 1611.70 | 1616.96 | 0.00 | ORB-short ORB[1617.95,1631.20] vol=1.5x ATR=8.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:20:00 | 1598.57 | 1612.27 | 0.00 | T1 1.5R @ 1598.57 |
| Stop hit — per-position SL triggered | 2024-05-28 11:30:00 | 1611.70 | 1612.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:40:00 | 1710.00 | 1693.14 | 0.00 | ORB-long ORB[1669.80,1694.00] vol=1.8x ATR=8.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:05:00 | 1722.65 | 1705.18 | 0.00 | T1 1.5R @ 1722.65 |
| Stop hit — per-position SL triggered | 2024-07-01 10:35:00 | 1710.00 | 1710.46 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:10:00 | 1692.35 | 1698.99 | 0.00 | ORB-short ORB[1698.25,1721.90] vol=2.0x ATR=6.48 |
| Stop hit — per-position SL triggered | 2024-07-05 12:30:00 | 1698.83 | 1695.62 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 1689.45 | 1709.30 | 0.00 | ORB-short ORB[1702.00,1724.95] vol=1.8x ATR=8.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 1677.14 | 1698.08 | 0.00 | T1 1.5R @ 1677.14 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 1689.45 | 1695.65 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:15:00 | 1676.30 | 1677.12 | 0.00 | ORB-short ORB[1678.05,1689.05] vol=19.1x ATR=7.39 |
| Stop hit — per-position SL triggered | 2024-07-11 10:20:00 | 1683.69 | 1677.13 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:45:00 | 1751.90 | 1736.30 | 0.00 | ORB-long ORB[1720.00,1745.00] vol=6.3x ATR=7.33 |
| Stop hit — per-position SL triggered | 2024-07-16 11:35:00 | 1744.57 | 1740.50 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 1712.90 | 1727.59 | 0.00 | ORB-short ORB[1730.60,1745.00] vol=2.4x ATR=7.71 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1720.61 | 1726.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:55:00 | 1703.00 | 1687.37 | 0.00 | ORB-long ORB[1672.45,1685.90] vol=2.2x ATR=10.08 |
| Stop hit — per-position SL triggered | 2024-07-23 10:05:00 | 1692.92 | 1689.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:05:00 | 1712.00 | 1701.31 | 0.00 | ORB-long ORB[1688.05,1711.75] vol=1.9x ATR=5.87 |
| Stop hit — per-position SL triggered | 2024-07-25 10:25:00 | 1706.13 | 1703.76 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:10:00 | 1725.10 | 1708.93 | 0.00 | ORB-long ORB[1691.45,1715.00] vol=6.1x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 13:10:00 | 1732.93 | 1718.89 | 0.00 | T1 1.5R @ 1732.93 |
| Target hit | 2024-07-26 15:20:00 | 1742.35 | 1727.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 1718.25 | 1701.43 | 0.00 | ORB-long ORB[1695.60,1715.00] vol=3.7x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:40:00 | 1733.50 | 1706.95 | 0.00 | T1 1.5R @ 1733.50 |
| Stop hit — per-position SL triggered | 2024-07-31 15:00:00 | 1718.25 | 1719.67 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 11:10:00 | 1529.90 | 1541.51 | 0.00 | ORB-short ORB[1541.30,1558.95] vol=2.0x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-08-28 12:45:00 | 1533.39 | 1536.74 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1510.00 | 1522.53 | 0.00 | ORB-short ORB[1513.55,1532.00] vol=2.6x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 1513.83 | 1521.57 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 11:10:00 | 1511.55 | 1515.07 | 0.00 | ORB-short ORB[1513.50,1524.00] vol=1.8x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 13:10:00 | 1506.76 | 1512.78 | 0.00 | T1 1.5R @ 1506.76 |
| Stop hit — per-position SL triggered | 2024-08-30 14:25:00 | 1511.55 | 1511.89 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:35:00 | 1514.25 | 1514.40 | 0.00 | ORB-short ORB[1515.60,1529.80] vol=2.6x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-09-02 10:40:00 | 1518.35 | 1514.34 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:35:00 | 1525.05 | 1520.44 | 0.00 | ORB-long ORB[1507.50,1523.90] vol=1.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-09-04 09:45:00 | 1520.32 | 1518.51 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 10:25:00 | 1569.75 | 1557.65 | 0.00 | ORB-long ORB[1549.60,1568.00] vol=2.3x ATR=5.86 |
| Stop hit — per-position SL triggered | 2024-09-06 10:30:00 | 1563.89 | 1559.75 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:00:00 | 1520.65 | 1527.85 | 0.00 | ORB-short ORB[1525.20,1534.20] vol=2.1x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-09-12 11:10:00 | 1525.93 | 1525.31 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:25:00 | 1516.95 | 1517.32 | 0.00 | ORB-short ORB[1517.10,1524.00] vol=3.3x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:45:00 | 1511.38 | 1516.25 | 0.00 | T1 1.5R @ 1511.38 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 1516.95 | 1514.14 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 1493.75 | 1500.51 | 0.00 | ORB-short ORB[1494.05,1511.50] vol=1.6x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:55:00 | 1486.01 | 1497.74 | 0.00 | T1 1.5R @ 1486.01 |
| Target hit | 2024-09-19 15:10:00 | 1491.40 | 1490.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:45:00 | 1504.60 | 1492.67 | 0.00 | ORB-long ORB[1483.10,1492.90] vol=1.6x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:50:00 | 1511.95 | 1499.18 | 0.00 | T1 1.5R @ 1511.95 |
| Stop hit — per-position SL triggered | 2024-09-20 09:55:00 | 1504.60 | 1499.69 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:00:00 | 1494.05 | 1501.72 | 0.00 | ORB-short ORB[1500.15,1515.15] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 1499.23 | 1501.17 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:00:00 | 1502.30 | 1505.75 | 0.00 | ORB-short ORB[1503.30,1515.00] vol=2.0x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-09-26 10:05:00 | 1506.28 | 1505.75 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:35:00 | 1489.30 | 1498.60 | 0.00 | ORB-short ORB[1499.70,1515.00] vol=1.6x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:40:00 | 1481.58 | 1496.63 | 0.00 | T1 1.5R @ 1481.58 |
| Stop hit — per-position SL triggered | 2024-09-30 11:50:00 | 1489.30 | 1491.56 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:10:00 | 1413.25 | 1400.19 | 0.00 | ORB-long ORB[1391.65,1410.40] vol=2.3x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:25:00 | 1422.85 | 1409.52 | 0.00 | T1 1.5R @ 1422.85 |
| Target hit | 2024-10-08 14:15:00 | 1420.00 | 1423.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2024-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:00:00 | 1463.90 | 1454.48 | 0.00 | ORB-long ORB[1439.40,1454.55] vol=1.6x ATR=6.67 |
| Stop hit — per-position SL triggered | 2024-10-10 10:55:00 | 1457.23 | 1457.80 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:05:00 | 1498.35 | 1502.26 | 0.00 | ORB-short ORB[1503.35,1512.45] vol=1.9x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:35:00 | 1493.59 | 1495.87 | 0.00 | T1 1.5R @ 1493.59 |
| Stop hit — per-position SL triggered | 2024-10-16 15:00:00 | 1498.35 | 1493.90 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 1481.45 | 1486.36 | 0.00 | ORB-short ORB[1490.10,1499.95] vol=2.7x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 1473.50 | 1485.07 | 0.00 | T1 1.5R @ 1473.50 |
| Target hit | 2024-10-17 15:20:00 | 1469.20 | 1473.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-10-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:20:00 | 1474.05 | 1463.88 | 0.00 | ORB-long ORB[1451.05,1466.45] vol=3.1x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 10:30:00 | 1482.55 | 1468.56 | 0.00 | T1 1.5R @ 1482.55 |
| Stop hit — per-position SL triggered | 2024-10-18 11:05:00 | 1474.05 | 1470.42 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:35:00 | 1393.00 | 1403.69 | 0.00 | ORB-short ORB[1404.05,1420.00] vol=2.7x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-10-23 09:45:00 | 1399.44 | 1399.92 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:10:00 | 1438.45 | 1441.90 | 0.00 | ORB-short ORB[1442.00,1461.80] vol=1.9x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:10:00 | 1429.80 | 1436.15 | 0.00 | T1 1.5R @ 1429.80 |
| Target hit | 2024-11-07 15:20:00 | 1429.00 | 1435.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 09:40:00 | 1431.50 | 1423.14 | 0.00 | ORB-long ORB[1414.50,1423.95] vol=1.9x ATR=7.15 |
| Stop hit — per-position SL triggered | 2024-11-14 09:45:00 | 1424.35 | 1425.15 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 09:55:00 | 1436.00 | 1440.69 | 0.00 | ORB-short ORB[1438.65,1458.95] vol=3.3x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:05:00 | 1426.96 | 1438.46 | 0.00 | T1 1.5R @ 1426.96 |
| Target hit | 2024-11-19 15:20:00 | 1410.30 | 1414.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-11-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 10:00:00 | 1387.55 | 1393.36 | 0.00 | ORB-short ORB[1395.00,1414.10] vol=1.6x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 12:40:00 | 1378.38 | 1389.76 | 0.00 | T1 1.5R @ 1378.38 |
| Stop hit — per-position SL triggered | 2024-11-22 13:25:00 | 1387.55 | 1389.10 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 1420.85 | 1414.41 | 0.00 | ORB-long ORB[1407.00,1419.90] vol=7.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-11-27 11:05:00 | 1417.35 | 1414.60 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:00:00 | 1437.90 | 1427.92 | 0.00 | ORB-long ORB[1417.50,1428.90] vol=1.9x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-11-29 10:05:00 | 1433.61 | 1428.06 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:45:00 | 1435.15 | 1442.42 | 0.00 | ORB-short ORB[1436.40,1456.30] vol=2.2x ATR=6.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 12:20:00 | 1424.92 | 1438.36 | 0.00 | T1 1.5R @ 1424.92 |
| Target hit | 2024-12-03 15:20:00 | 1413.05 | 1433.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 1382.00 | 1388.63 | 0.00 | ORB-short ORB[1385.80,1396.85] vol=1.6x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-12-06 09:35:00 | 1385.63 | 1388.09 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:15:00 | 1375.60 | 1364.90 | 0.00 | ORB-long ORB[1360.60,1369.00] vol=1.5x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:25:00 | 1381.52 | 1367.16 | 0.00 | T1 1.5R @ 1381.52 |
| Stop hit — per-position SL triggered | 2024-12-09 10:45:00 | 1375.60 | 1369.93 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:25:00 | 1360.00 | 1365.93 | 0.00 | ORB-short ORB[1366.75,1376.30] vol=3.1x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-12-10 11:25:00 | 1363.64 | 1364.38 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:00:00 | 1369.65 | 1378.91 | 0.00 | ORB-short ORB[1375.70,1396.05] vol=2.0x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:35:00 | 1362.89 | 1374.73 | 0.00 | T1 1.5R @ 1362.89 |
| Target hit | 2024-12-12 15:20:00 | 1359.45 | 1364.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 1338.65 | 1348.07 | 0.00 | ORB-short ORB[1346.30,1361.00] vol=4.5x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 1344.17 | 1345.97 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 1332.25 | 1335.88 | 0.00 | ORB-short ORB[1334.30,1347.30] vol=1.9x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:00:00 | 1323.95 | 1334.15 | 0.00 | T1 1.5R @ 1323.95 |
| Target hit | 2024-12-17 15:20:00 | 1325.70 | 1326.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-12-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:10:00 | 1317.55 | 1312.36 | 0.00 | ORB-long ORB[1301.00,1315.65] vol=1.7x ATR=5.12 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 1312.43 | 1314.82 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 1266.40 | 1275.24 | 0.00 | ORB-short ORB[1275.05,1293.55] vol=1.8x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:20:00 | 1259.52 | 1269.24 | 0.00 | T1 1.5R @ 1259.52 |
| Stop hit — per-position SL triggered | 2024-12-26 13:10:00 | 1266.40 | 1263.19 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 1281.15 | 1277.47 | 0.00 | ORB-long ORB[1265.00,1278.35] vol=2.3x ATR=4.58 |
| Target hit | 2024-12-27 15:20:00 | 1286.00 | 1281.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 1276.35 | 1280.45 | 0.00 | ORB-short ORB[1278.95,1288.00] vol=2.0x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:00:00 | 1270.82 | 1279.75 | 0.00 | T1 1.5R @ 1270.82 |
| Stop hit — per-position SL triggered | 2024-12-30 11:30:00 | 1276.35 | 1278.97 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:50:00 | 1278.80 | 1277.61 | 0.00 | ORB-long ORB[1258.95,1278.00] vol=2.3x ATR=4.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:55:00 | 1286.27 | 1278.23 | 0.00 | T1 1.5R @ 1286.27 |
| Stop hit — per-position SL triggered | 2024-12-31 10:10:00 | 1278.80 | 1279.15 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 1276.45 | 1288.75 | 0.00 | ORB-short ORB[1287.45,1304.80] vol=1.5x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-01-02 10:55:00 | 1279.75 | 1287.27 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 1319.50 | 1313.27 | 0.00 | ORB-long ORB[1303.80,1317.60] vol=1.6x ATR=5.26 |
| Stop hit — per-position SL triggered | 2025-01-03 09:55:00 | 1314.24 | 1314.85 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 1278.55 | 1291.03 | 0.00 | ORB-short ORB[1290.00,1308.45] vol=1.6x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:05:00 | 1273.00 | 1289.67 | 0.00 | T1 1.5R @ 1273.00 |
| Target hit | 2025-01-06 15:20:00 | 1249.35 | 1264.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:10:00 | 1250.00 | 1258.78 | 0.00 | ORB-short ORB[1255.15,1267.90] vol=1.8x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 12:00:00 | 1243.96 | 1257.49 | 0.00 | T1 1.5R @ 1243.96 |
| Target hit | 2025-01-08 15:20:00 | 1245.00 | 1251.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:35:00 | 1257.75 | 1249.15 | 0.00 | ORB-long ORB[1236.00,1252.45] vol=3.0x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:05:00 | 1263.84 | 1255.81 | 0.00 | T1 1.5R @ 1263.84 |
| Target hit | 2025-01-09 15:20:00 | 1265.00 | 1263.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:15:00 | 1187.20 | 1195.75 | 0.00 | ORB-short ORB[1200.90,1213.20] vol=2.5x ATR=6.61 |
| Stop hit — per-position SL triggered | 2025-01-24 10:20:00 | 1193.81 | 1195.63 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:40:00 | 1170.20 | 1180.61 | 0.00 | ORB-short ORB[1181.75,1191.95] vol=3.0x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-02-01 10:50:00 | 1173.86 | 1176.58 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:15:00 | 1135.25 | 1143.14 | 0.00 | ORB-short ORB[1136.95,1151.75] vol=2.4x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:35:00 | 1128.21 | 1141.70 | 0.00 | T1 1.5R @ 1128.21 |
| Target hit | 2025-02-04 13:30:00 | 1130.70 | 1130.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 1083.65 | 1090.86 | 0.00 | ORB-short ORB[1085.55,1101.75] vol=1.6x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:40:00 | 1077.63 | 1087.99 | 0.00 | T1 1.5R @ 1077.63 |
| Target hit | 2025-02-10 15:20:00 | 1048.30 | 1063.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:45:00 | 1034.40 | 1045.55 | 0.00 | ORB-short ORB[1045.10,1058.20] vol=2.0x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-02-11 10:05:00 | 1039.21 | 1041.20 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:05:00 | 888.80 | 901.75 | 0.00 | ORB-short ORB[915.80,923.00] vol=5.0x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-02-21 11:20:00 | 891.43 | 900.53 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 995.35 | 989.09 | 0.00 | ORB-long ORB[976.65,990.00] vol=3.0x ATR=6.62 |
| Stop hit — per-position SL triggered | 2025-03-12 09:45:00 | 988.73 | 989.73 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 11:00:00 | 981.95 | 987.46 | 0.00 | ORB-short ORB[985.10,999.00] vol=1.9x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:15:00 | 976.61 | 985.50 | 0.00 | T1 1.5R @ 976.61 |
| Target hit | 2025-03-19 14:35:00 | 977.30 | 975.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2025-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:30:00 | 1040.20 | 1034.08 | 0.00 | ORB-long ORB[1022.00,1036.20] vol=3.2x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-04-30 10:35:00 | 1035.89 | 1034.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 10:05:00 | 1527.50 | 2024-05-15 10:10:00 | 1535.32 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-15 10:05:00 | 1527.50 | 2024-05-15 10:15:00 | 1527.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-24 10:55:00 | 1688.85 | 2024-05-24 11:25:00 | 1696.15 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-28 09:40:00 | 1611.70 | 2024-05-28 11:20:00 | 1598.57 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2024-05-28 09:40:00 | 1611.70 | 2024-05-28 11:30:00 | 1611.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:40:00 | 1710.00 | 2024-07-01 10:05:00 | 1722.65 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-07-01 09:40:00 | 1710.00 | 2024-07-01 10:35:00 | 1710.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 10:10:00 | 1692.35 | 2024-07-05 12:30:00 | 1698.83 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1689.45 | 2024-07-10 10:35:00 | 1677.14 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1689.45 | 2024-07-10 10:45:00 | 1689.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 10:15:00 | 1676.30 | 2024-07-11 10:20:00 | 1683.69 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-16 10:45:00 | 1751.90 | 2024-07-16 11:35:00 | 1744.57 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-18 09:35:00 | 1712.90 | 2024-07-18 09:40:00 | 1720.61 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-23 09:55:00 | 1703.00 | 2024-07-23 10:05:00 | 1692.92 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-07-25 10:05:00 | 1712.00 | 2024-07-25 10:25:00 | 1706.13 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-26 11:10:00 | 1725.10 | 2024-07-26 13:10:00 | 1732.93 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-26 11:10:00 | 1725.10 | 2024-07-26 15:20:00 | 1742.35 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2024-07-31 10:00:00 | 1718.25 | 2024-07-31 10:40:00 | 1733.50 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-07-31 10:00:00 | 1718.25 | 2024-07-31 15:00:00 | 1718.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 11:10:00 | 1529.90 | 2024-08-28 12:45:00 | 1533.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1510.00 | 2024-08-29 11:10:00 | 1513.83 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-30 11:10:00 | 1511.55 | 2024-08-30 13:10:00 | 1506.76 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-08-30 11:10:00 | 1511.55 | 2024-08-30 14:25:00 | 1511.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-02 10:35:00 | 1514.25 | 2024-09-02 10:40:00 | 1518.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-04 09:35:00 | 1525.05 | 2024-09-04 09:45:00 | 1520.32 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-06 10:25:00 | 1569.75 | 2024-09-06 10:30:00 | 1563.89 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-12 10:00:00 | 1520.65 | 2024-09-12 11:10:00 | 1525.93 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-13 10:25:00 | 1516.95 | 2024-09-13 10:45:00 | 1511.38 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-13 10:25:00 | 1516.95 | 2024-09-13 11:20:00 | 1516.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:35:00 | 1493.75 | 2024-09-19 09:55:00 | 1486.01 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-19 09:35:00 | 1493.75 | 2024-09-19 15:10:00 | 1491.40 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2024-09-20 09:45:00 | 1504.60 | 2024-09-20 09:50:00 | 1511.95 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-20 09:45:00 | 1504.60 | 2024-09-20 09:55:00 | 1504.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-23 10:00:00 | 1494.05 | 2024-09-23 10:15:00 | 1499.23 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-26 10:00:00 | 1502.30 | 2024-09-26 10:05:00 | 1506.28 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-30 10:35:00 | 1489.30 | 2024-09-30 10:40:00 | 1481.58 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-30 10:35:00 | 1489.30 | 2024-09-30 11:50:00 | 1489.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 11:10:00 | 1413.25 | 2024-10-08 11:25:00 | 1422.85 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-10-08 11:10:00 | 1413.25 | 2024-10-08 14:15:00 | 1420.00 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-10 10:00:00 | 1463.90 | 2024-10-10 10:55:00 | 1457.23 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-16 11:05:00 | 1498.35 | 2024-10-16 11:35:00 | 1493.59 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-10-16 11:05:00 | 1498.35 | 2024-10-16 15:00:00 | 1498.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1481.45 | 2024-10-17 11:25:00 | 1473.50 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1481.45 | 2024-10-17 15:20:00 | 1469.20 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-10-18 10:20:00 | 1474.05 | 2024-10-18 10:30:00 | 1482.55 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-10-18 10:20:00 | 1474.05 | 2024-10-18 11:05:00 | 1474.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-23 09:35:00 | 1393.00 | 2024-10-23 09:45:00 | 1399.44 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-11-07 10:10:00 | 1438.45 | 2024-11-07 15:10:00 | 1429.80 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-07 10:10:00 | 1438.45 | 2024-11-07 15:20:00 | 1429.00 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-14 09:40:00 | 1431.50 | 2024-11-14 09:45:00 | 1424.35 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-11-19 09:55:00 | 1436.00 | 2024-11-19 10:05:00 | 1426.96 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-11-19 09:55:00 | 1436.00 | 2024-11-19 15:20:00 | 1410.30 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2024-11-22 10:00:00 | 1387.55 | 2024-11-22 12:40:00 | 1378.38 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-22 10:00:00 | 1387.55 | 2024-11-22 13:25:00 | 1387.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 11:00:00 | 1420.85 | 2024-11-27 11:05:00 | 1417.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-29 10:00:00 | 1437.90 | 2024-11-29 10:05:00 | 1433.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-03 10:45:00 | 1435.15 | 2024-12-03 12:20:00 | 1424.92 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-12-03 10:45:00 | 1435.15 | 2024-12-03 15:20:00 | 1413.05 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2024-12-06 09:30:00 | 1382.00 | 2024-12-06 09:35:00 | 1385.63 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-09 10:15:00 | 1375.60 | 2024-12-09 10:25:00 | 1381.52 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-09 10:15:00 | 1375.60 | 2024-12-09 10:45:00 | 1375.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 10:25:00 | 1360.00 | 2024-12-10 11:25:00 | 1363.64 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-12 10:00:00 | 1369.65 | 2024-12-12 11:35:00 | 1362.89 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-12 10:00:00 | 1369.65 | 2024-12-12 15:20:00 | 1359.45 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2024-12-13 10:15:00 | 1338.65 | 2024-12-13 10:55:00 | 1344.17 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-17 09:40:00 | 1332.25 | 2024-12-17 10:00:00 | 1323.95 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-17 09:40:00 | 1332.25 | 2024-12-17 15:20:00 | 1325.70 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-19 10:10:00 | 1317.55 | 2024-12-19 10:15:00 | 1312.43 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-26 10:30:00 | 1266.40 | 2024-12-26 11:20:00 | 1259.52 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-12-26 10:30:00 | 1266.40 | 2024-12-26 13:10:00 | 1266.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 11:05:00 | 1281.15 | 2024-12-27 15:20:00 | 1286.00 | TARGET_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2024-12-30 10:35:00 | 1276.35 | 2024-12-30 11:00:00 | 1270.82 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-30 10:35:00 | 1276.35 | 2024-12-30 11:30:00 | 1276.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 09:50:00 | 1278.80 | 2024-12-31 09:55:00 | 1286.27 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-31 09:50:00 | 1278.80 | 2024-12-31 10:10:00 | 1278.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 10:45:00 | 1276.45 | 2025-01-02 10:55:00 | 1279.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-03 09:35:00 | 1319.50 | 2025-01-03 09:55:00 | 1314.24 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1278.55 | 2025-01-06 11:05:00 | 1273.00 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1278.55 | 2025-01-06 15:20:00 | 1249.35 | TARGET_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2025-01-08 11:10:00 | 1250.00 | 2025-01-08 12:00:00 | 1243.96 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-08 11:10:00 | 1250.00 | 2025-01-08 15:20:00 | 1245.00 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-09 09:35:00 | 1257.75 | 2025-01-09 10:05:00 | 1263.84 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-09 09:35:00 | 1257.75 | 2025-01-09 15:20:00 | 1265.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-24 10:15:00 | 1187.20 | 2025-01-24 10:20:00 | 1193.81 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-02-01 10:40:00 | 1170.20 | 2025-02-01 10:50:00 | 1173.86 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-04 10:15:00 | 1135.25 | 2025-02-04 10:35:00 | 1128.21 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-02-04 10:15:00 | 1135.25 | 2025-02-04 13:30:00 | 1130.70 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-10 09:30:00 | 1083.65 | 2025-02-10 09:40:00 | 1077.63 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-10 09:30:00 | 1083.65 | 2025-02-10 15:20:00 | 1048.30 | TARGET_HIT | 0.50 | 3.26% |
| SELL | retest1 | 2025-02-11 09:45:00 | 1034.40 | 2025-02-11 10:05:00 | 1039.21 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-02-21 11:05:00 | 888.80 | 2025-02-21 11:20:00 | 891.43 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-12 09:30:00 | 995.35 | 2025-03-12 09:45:00 | 988.73 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-03-19 11:00:00 | 981.95 | 2025-03-19 12:15:00 | 976.61 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-19 11:00:00 | 981.95 | 2025-03-19 14:35:00 | 977.30 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-30 10:30:00 | 1040.20 | 2025-04-30 10:35:00 | 1035.89 | STOP_HIT | 1.00 | -0.41% |
