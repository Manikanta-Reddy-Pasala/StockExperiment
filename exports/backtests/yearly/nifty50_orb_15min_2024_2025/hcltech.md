# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1198.00
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
| ENTRY1 | 58 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 12 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 46
- **Target hits / Stop hits / Partials:** 12 / 46 / 18
- **Avg / median % per leg:** 0.07% / -0.16%
- **Sum % (uncompounded):** 5.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 14 | 36.8% | 5 | 24 | 9 | 0.09% | 3.5% |
| BUY @ 2nd Alert (retest1) | 38 | 14 | 36.8% | 5 | 24 | 9 | 0.09% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 16 | 42.1% | 7 | 22 | 9 | 0.06% | 2.1% |
| SELL @ 2nd Alert (retest1) | 38 | 16 | 42.1% | 7 | 22 | 9 | 0.06% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 30 | 39.5% | 12 | 46 | 18 | 0.07% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:40:00 | 1311.15 | 1315.86 | 0.00 | ORB-short ORB[1317.55,1326.75] vol=2.1x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-05-14 11:05:00 | 1313.63 | 1314.98 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 1331.50 | 1324.89 | 0.00 | ORB-long ORB[1319.25,1326.90] vol=1.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-05-15 11:25:00 | 1328.62 | 1326.35 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:55:00 | 1334.30 | 1340.72 | 0.00 | ORB-short ORB[1342.10,1351.55] vol=2.2x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-05-30 11:25:00 | 1337.06 | 1338.87 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:15:00 | 1338.20 | 1330.92 | 0.00 | ORB-long ORB[1325.05,1335.85] vol=2.4x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-05-31 11:20:00 | 1334.17 | 1331.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:00:00 | 1352.50 | 1339.92 | 0.00 | ORB-long ORB[1316.10,1335.00] vol=1.5x ATR=6.95 |
| Stop hit — per-position SL triggered | 2024-06-05 11:25:00 | 1345.55 | 1341.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:30:00 | 1370.75 | 1363.40 | 0.00 | ORB-long ORB[1350.90,1369.80] vol=1.8x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 10:40:00 | 1379.65 | 1369.30 | 0.00 | T1 1.5R @ 1379.65 |
| Target hit | 2024-06-06 11:45:00 | 1372.05 | 1372.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:00:00 | 1429.95 | 1432.30 | 0.00 | ORB-short ORB[1430.30,1446.00] vol=1.8x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:30:00 | 1424.85 | 1431.10 | 0.00 | T1 1.5R @ 1424.85 |
| Stop hit — per-position SL triggered | 2024-06-14 10:55:00 | 1429.95 | 1430.57 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:30:00 | 1431.30 | 1436.20 | 0.00 | ORB-short ORB[1435.00,1442.90] vol=3.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-06-19 10:50:00 | 1434.17 | 1435.56 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:05:00 | 1469.10 | 1464.84 | 0.00 | ORB-long ORB[1455.80,1467.00] vol=2.1x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:40:00 | 1474.10 | 1466.57 | 0.00 | T1 1.5R @ 1474.10 |
| Stop hit — per-position SL triggered | 2024-07-01 14:25:00 | 1469.10 | 1469.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:00:00 | 1511.00 | 1501.00 | 0.00 | ORB-long ORB[1485.00,1502.70] vol=1.5x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:20:00 | 1516.76 | 1504.68 | 0.00 | T1 1.5R @ 1516.76 |
| Target hit | 2024-07-04 15:20:00 | 1524.10 | 1519.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-07-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 11:05:00 | 1545.50 | 1536.79 | 0.00 | ORB-long ORB[1533.40,1543.95] vol=1.6x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-07-09 11:40:00 | 1540.98 | 1537.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:50:00 | 1510.95 | 1518.20 | 0.00 | ORB-short ORB[1522.00,1532.60] vol=6.0x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:30:00 | 1504.94 | 1511.84 | 0.00 | T1 1.5R @ 1504.94 |
| Target hit | 2024-07-10 12:05:00 | 1507.25 | 1503.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 1531.05 | 1522.44 | 0.00 | ORB-long ORB[1510.85,1528.00] vol=2.1x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-07-11 09:45:00 | 1526.40 | 1524.47 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:40:00 | 1534.70 | 1522.23 | 0.00 | ORB-long ORB[1510.50,1530.00] vol=2.4x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:50:00 | 1542.11 | 1526.07 | 0.00 | T1 1.5R @ 1542.11 |
| Target hit | 2024-07-12 15:20:00 | 1562.25 | 1550.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:10:00 | 1587.05 | 1582.75 | 0.00 | ORB-long ORB[1562.85,1580.55] vol=1.8x ATR=4.35 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 1582.70 | 1582.80 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:50:00 | 1621.05 | 1625.86 | 0.00 | ORB-short ORB[1622.10,1632.20] vol=5.2x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-07-30 11:40:00 | 1624.03 | 1625.07 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:45:00 | 1634.95 | 1628.02 | 0.00 | ORB-long ORB[1621.00,1630.00] vol=1.5x ATR=2.89 |
| Stop hit — per-position SL triggered | 2024-07-31 10:50:00 | 1632.06 | 1628.13 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 10:55:00 | 1570.70 | 1582.24 | 0.00 | ORB-short ORB[1574.00,1591.90] vol=3.1x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:00:00 | 1563.09 | 1580.31 | 0.00 | T1 1.5R @ 1563.09 |
| Target hit | 2024-08-05 12:45:00 | 1560.00 | 1559.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 1595.75 | 1583.45 | 0.00 | ORB-long ORB[1577.00,1594.45] vol=2.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-08-12 11:50:00 | 1592.14 | 1586.94 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:45:00 | 1599.35 | 1588.78 | 0.00 | ORB-long ORB[1577.65,1587.80] vol=2.0x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:10:00 | 1606.65 | 1596.65 | 0.00 | T1 1.5R @ 1606.65 |
| Stop hit — per-position SL triggered | 2024-08-13 11:35:00 | 1599.35 | 1598.52 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:45:00 | 1668.05 | 1673.89 | 0.00 | ORB-short ORB[1672.80,1685.95] vol=1.9x ATR=3.32 |
| Stop hit — per-position SL triggered | 2024-08-23 10:50:00 | 1671.37 | 1673.75 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1694.25 | 1703.48 | 0.00 | ORB-short ORB[1699.00,1718.20] vol=1.6x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 1698.98 | 1702.55 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 1745.70 | 1725.41 | 0.00 | ORB-long ORB[1697.25,1715.00] vol=2.3x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-08-29 11:45:00 | 1740.18 | 1734.55 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:45:00 | 1772.85 | 1777.51 | 0.00 | ORB-short ORB[1777.05,1795.60] vol=1.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-09-05 14:30:00 | 1776.64 | 1774.38 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 1750.40 | 1768.45 | 0.00 | ORB-short ORB[1777.65,1795.75] vol=1.7x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-09-06 10:40:00 | 1756.46 | 1767.43 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 11:00:00 | 1817.40 | 1816.09 | 0.00 | ORB-long ORB[1807.20,1816.95] vol=1.9x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 12:20:00 | 1822.16 | 1817.14 | 0.00 | T1 1.5R @ 1822.16 |
| Target hit | 2024-09-17 14:25:00 | 1820.75 | 1821.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 1780.00 | 1788.39 | 0.00 | ORB-short ORB[1786.70,1808.10] vol=2.1x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-09-18 09:40:00 | 1784.77 | 1787.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 1738.70 | 1757.97 | 0.00 | ORB-short ORB[1751.55,1776.00] vol=2.3x ATR=8.43 |
| Stop hit — per-position SL triggered | 2024-09-19 09:40:00 | 1747.13 | 1756.61 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 1757.90 | 1746.85 | 0.00 | ORB-long ORB[1738.35,1754.40] vol=1.6x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-09-24 10:00:00 | 1753.85 | 1747.37 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 11:05:00 | 1801.55 | 1798.94 | 0.00 | ORB-long ORB[1778.90,1800.00] vol=3.6x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-09-26 12:10:00 | 1797.49 | 1799.86 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:00:00 | 1771.55 | 1786.62 | 0.00 | ORB-short ORB[1781.45,1804.20] vol=1.7x ATR=5.03 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 1776.58 | 1782.59 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:35:00 | 1837.00 | 1826.71 | 0.00 | ORB-long ORB[1810.80,1827.50] vol=3.4x ATR=6.07 |
| Stop hit — per-position SL triggered | 2024-10-10 09:40:00 | 1830.93 | 1827.38 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:15:00 | 1853.05 | 1861.26 | 0.00 | ORB-short ORB[1858.05,1877.05] vol=1.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2024-10-17 11:35:00 | 1857.51 | 1860.39 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-18 09:50:00 | 1855.75 | 1865.04 | 0.00 | ORB-short ORB[1857.60,1873.90] vol=1.6x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 10:10:00 | 1847.66 | 1861.60 | 0.00 | T1 1.5R @ 1847.66 |
| Target hit | 2024-10-18 12:30:00 | 1855.00 | 1854.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-10-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:10:00 | 1862.45 | 1866.91 | 0.00 | ORB-short ORB[1862.95,1888.50] vol=1.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:45:00 | 1856.40 | 1866.03 | 0.00 | T1 1.5R @ 1856.40 |
| Target hit | 2024-10-21 15:20:00 | 1844.15 | 1856.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:35:00 | 1814.90 | 1800.48 | 0.00 | ORB-long ORB[1783.00,1806.50] vol=1.7x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:00:00 | 1824.29 | 1808.79 | 0.00 | T1 1.5R @ 1824.29 |
| Stop hit — per-position SL triggered | 2024-11-06 10:30:00 | 1814.90 | 1811.71 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 10:50:00 | 1893.00 | 1898.91 | 0.00 | ORB-short ORB[1897.65,1919.95] vol=1.7x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-11-25 11:00:00 | 1897.68 | 1898.41 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:35:00 | 1867.55 | 1855.30 | 0.00 | ORB-long ORB[1833.95,1854.65] vol=1.5x ATR=6.14 |
| Stop hit — per-position SL triggered | 2024-11-29 10:55:00 | 1861.41 | 1859.55 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 11:10:00 | 1930.30 | 1927.06 | 0.00 | ORB-long ORB[1912.20,1930.00] vol=1.9x ATR=3.08 |
| Stop hit — per-position SL triggered | 2024-12-10 11:20:00 | 1927.22 | 1927.12 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:55:00 | 1960.55 | 1968.76 | 0.00 | ORB-short ORB[1966.80,1980.00] vol=1.9x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:25:00 | 1954.70 | 1964.12 | 0.00 | T1 1.5R @ 1954.70 |
| Target hit | 2024-12-16 15:20:00 | 1953.30 | 1958.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 1932.60 | 1923.55 | 0.00 | ORB-long ORB[1912.95,1928.45] vol=2.6x ATR=4.41 |
| Stop hit — per-position SL triggered | 2025-01-01 11:30:00 | 1928.19 | 1925.56 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 1930.95 | 1926.46 | 0.00 | ORB-long ORB[1910.05,1922.25] vol=3.2x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:55:00 | 1937.84 | 1928.00 | 0.00 | T1 1.5R @ 1937.84 |
| Target hit | 2025-01-02 15:20:00 | 1969.80 | 1955.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-01-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:05:00 | 1950.80 | 1972.79 | 0.00 | ORB-short ORB[1975.00,1992.10] vol=2.0x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-01-03 11:25:00 | 1956.25 | 1968.82 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 09:45:00 | 1942.20 | 1963.06 | 0.00 | ORB-short ORB[1960.00,1981.00] vol=2.1x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 10:35:00 | 1930.85 | 1953.82 | 0.00 | T1 1.5R @ 1930.85 |
| Target hit | 2025-01-07 15:20:00 | 1920.05 | 1930.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:40:00 | 1933.45 | 1932.63 | 0.00 | ORB-long ORB[1920.00,1930.00] vol=6.6x ATR=5.75 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 1927.70 | 1932.55 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:55:00 | 1797.40 | 1812.67 | 0.00 | ORB-short ORB[1818.25,1837.95] vol=1.9x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:50:00 | 1790.60 | 1805.08 | 0.00 | T1 1.5R @ 1790.60 |
| Stop hit — per-position SL triggered | 2025-01-16 14:30:00 | 1797.40 | 1797.83 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 11:15:00 | 1807.20 | 1818.13 | 0.00 | ORB-short ORB[1814.10,1827.30] vol=2.3x ATR=4.59 |
| Stop hit — per-position SL triggered | 2025-01-23 11:30:00 | 1811.79 | 1817.33 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 1714.75 | 1728.53 | 0.00 | ORB-short ORB[1722.60,1739.90] vol=1.6x ATR=4.47 |
| Target hit | 2025-02-05 15:20:00 | 1713.45 | 1718.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:45:00 | 1709.60 | 1725.54 | 0.00 | ORB-short ORB[1723.90,1742.40] vol=3.0x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 11:00:00 | 1702.54 | 1722.83 | 0.00 | T1 1.5R @ 1702.54 |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 1709.60 | 1719.26 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 10:50:00 | 1699.90 | 1706.31 | 0.00 | ORB-short ORB[1703.00,1714.55] vol=2.1x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-02-20 11:35:00 | 1703.70 | 1705.05 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-02-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 10:30:00 | 1697.00 | 1692.84 | 0.00 | ORB-long ORB[1685.15,1695.30] vol=2.2x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-02-21 10:35:00 | 1693.30 | 1692.88 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 11:05:00 | 1644.00 | 1643.15 | 0.00 | ORB-long ORB[1631.25,1642.20] vol=1.9x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-02-25 11:10:00 | 1640.32 | 1643.09 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 1638.75 | 1631.13 | 0.00 | ORB-long ORB[1620.00,1638.00] vol=1.7x ATR=6.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:45:00 | 1648.41 | 1634.75 | 0.00 | T1 1.5R @ 1648.41 |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 1638.75 | 1639.94 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:05:00 | 1638.35 | 1635.35 | 0.00 | ORB-long ORB[1624.90,1638.10] vol=4.0x ATR=4.11 |
| Stop hit — per-position SL triggered | 2025-03-26 11:10:00 | 1634.24 | 1635.41 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:30:00 | 1631.25 | 1628.57 | 0.00 | ORB-long ORB[1613.55,1630.00] vol=1.6x ATR=5.61 |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1625.64 | 1628.49 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 1554.05 | 1567.83 | 0.00 | ORB-short ORB[1563.45,1581.40] vol=2.1x ATR=5.65 |
| Stop hit — per-position SL triggered | 2025-04-01 11:15:00 | 1559.70 | 1566.51 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:50:00 | 1479.85 | 1487.60 | 0.00 | ORB-short ORB[1481.10,1500.00] vol=1.7x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-04-03 10:05:00 | 1485.83 | 1487.14 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 1575.70 | 1586.66 | 0.00 | ORB-short ORB[1579.00,1597.70] vol=1.7x ATR=3.74 |
| Stop hit — per-position SL triggered | 2025-05-05 11:30:00 | 1579.44 | 1585.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:40:00 | 1311.15 | 2024-05-14 11:05:00 | 1313.63 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-05-15 11:00:00 | 1331.50 | 2024-05-15 11:25:00 | 1328.62 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-30 10:55:00 | 1334.30 | 2024-05-30 11:25:00 | 1337.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-31 11:15:00 | 1338.20 | 2024-05-31 11:20:00 | 1334.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-05 11:00:00 | 1352.50 | 2024-06-05 11:25:00 | 1345.55 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-06-06 09:30:00 | 1370.75 | 2024-06-06 10:40:00 | 1379.65 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-06 09:30:00 | 1370.75 | 2024-06-06 11:45:00 | 1372.05 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2024-06-14 10:00:00 | 1429.95 | 2024-06-14 10:30:00 | 1424.85 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-06-14 10:00:00 | 1429.95 | 2024-06-14 10:55:00 | 1429.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 10:30:00 | 1431.30 | 2024-06-19 10:50:00 | 1434.17 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-01 11:05:00 | 1469.10 | 2024-07-01 11:40:00 | 1474.10 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-07-01 11:05:00 | 1469.10 | 2024-07-01 14:25:00 | 1469.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:00:00 | 1511.00 | 2024-07-04 10:20:00 | 1516.76 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-04 10:00:00 | 1511.00 | 2024-07-04 15:20:00 | 1524.10 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-07-09 11:05:00 | 1545.50 | 2024-07-09 11:40:00 | 1540.98 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-10 09:50:00 | 1510.95 | 2024-07-10 10:30:00 | 1504.94 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-10 09:50:00 | 1510.95 | 2024-07-10 12:05:00 | 1507.25 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-07-11 09:30:00 | 1531.05 | 2024-07-11 09:45:00 | 1526.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-12 10:40:00 | 1534.70 | 2024-07-12 10:50:00 | 1542.11 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-12 10:40:00 | 1534.70 | 2024-07-12 15:20:00 | 1562.25 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2024-07-25 11:10:00 | 1587.05 | 2024-07-25 11:15:00 | 1582.70 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-30 10:50:00 | 1621.05 | 2024-07-30 11:40:00 | 1624.03 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-07-31 10:45:00 | 1634.95 | 2024-07-31 10:50:00 | 1632.06 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-05 10:55:00 | 1570.70 | 2024-08-05 11:00:00 | 1563.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-05 10:55:00 | 1570.70 | 2024-08-05 12:45:00 | 1560.00 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-08-12 11:15:00 | 1595.75 | 2024-08-12 11:50:00 | 1592.14 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-13 09:45:00 | 1599.35 | 2024-08-13 11:10:00 | 1606.65 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-13 09:45:00 | 1599.35 | 2024-08-13 11:35:00 | 1599.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 10:45:00 | 1668.05 | 2024-08-23 10:50:00 | 1671.37 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1694.25 | 2024-08-28 09:40:00 | 1698.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-29 10:20:00 | 1745.70 | 2024-08-29 11:45:00 | 1740.18 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-05 10:45:00 | 1772.85 | 2024-09-05 14:30:00 | 1776.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-06 10:30:00 | 1750.40 | 2024-09-06 10:40:00 | 1756.46 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-17 11:00:00 | 1817.40 | 2024-09-17 12:20:00 | 1822.16 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-09-17 11:00:00 | 1817.40 | 2024-09-17 14:25:00 | 1820.75 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-09-18 09:35:00 | 1780.00 | 2024-09-18 09:40:00 | 1784.77 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-19 09:35:00 | 1738.70 | 2024-09-19 09:40:00 | 1747.13 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-09-24 09:55:00 | 1757.90 | 2024-09-24 10:00:00 | 1753.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-26 11:05:00 | 1801.55 | 2024-09-26 12:10:00 | 1797.49 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-07 11:00:00 | 1771.55 | 2024-10-07 11:25:00 | 1776.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-10 09:35:00 | 1837.00 | 2024-10-10 09:40:00 | 1830.93 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-17 11:15:00 | 1853.05 | 2024-10-17 11:35:00 | 1857.51 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-18 09:50:00 | 1855.75 | 2024-10-18 10:10:00 | 1847.66 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-18 09:50:00 | 1855.75 | 2024-10-18 12:30:00 | 1855.00 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-10-21 11:10:00 | 1862.45 | 2024-10-21 11:45:00 | 1856.40 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-10-21 11:10:00 | 1862.45 | 2024-10-21 15:20:00 | 1844.15 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-11-06 09:35:00 | 1814.90 | 2024-11-06 10:00:00 | 1824.29 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-11-06 09:35:00 | 1814.90 | 2024-11-06 10:30:00 | 1814.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-25 10:50:00 | 1893.00 | 2024-11-25 11:00:00 | 1897.68 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-29 10:35:00 | 1867.55 | 2024-11-29 10:55:00 | 1861.41 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-10 11:10:00 | 1930.30 | 2024-12-10 11:20:00 | 1927.22 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-16 10:55:00 | 1960.55 | 2024-12-16 12:25:00 | 1954.70 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-16 10:55:00 | 1960.55 | 2024-12-16 15:20:00 | 1953.30 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-01 11:05:00 | 1932.60 | 2025-01-01 11:30:00 | 1928.19 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-02 10:50:00 | 1930.95 | 2025-01-02 10:55:00 | 1937.84 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-02 10:50:00 | 1930.95 | 2025-01-02 15:20:00 | 1969.80 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2025-01-03 11:05:00 | 1950.80 | 2025-01-03 11:25:00 | 1956.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-07 09:45:00 | 1942.20 | 2025-01-07 10:35:00 | 1930.85 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-07 09:45:00 | 1942.20 | 2025-01-07 15:20:00 | 1920.05 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-01-09 10:40:00 | 1933.45 | 2025-01-09 10:45:00 | 1927.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-16 10:55:00 | 1797.40 | 2025-01-16 11:50:00 | 1790.60 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-16 10:55:00 | 1797.40 | 2025-01-16 14:30:00 | 1797.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-23 11:15:00 | 1807.20 | 2025-01-23 11:30:00 | 1811.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-05 11:15:00 | 1714.75 | 2025-02-05 15:20:00 | 1713.45 | TARGET_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2025-02-12 10:45:00 | 1709.60 | 2025-02-12 11:00:00 | 1702.54 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-12 10:45:00 | 1709.60 | 2025-02-12 11:15:00 | 1709.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-20 10:50:00 | 1699.90 | 2025-02-20 11:35:00 | 1703.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-02-21 10:30:00 | 1697.00 | 2025-02-21 10:35:00 | 1693.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-02-25 11:05:00 | 1644.00 | 2025-02-25 11:10:00 | 1640.32 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-25 09:35:00 | 1638.75 | 2025-03-25 09:45:00 | 1648.41 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-25 09:35:00 | 1638.75 | 2025-03-25 10:15:00 | 1638.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 11:05:00 | 1638.35 | 2025-03-26 11:10:00 | 1634.24 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-27 10:30:00 | 1631.25 | 2025-03-27 11:15:00 | 1625.64 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-01 10:55:00 | 1554.05 | 2025-04-01 11:15:00 | 1559.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-03 09:50:00 | 1479.85 | 2025-04-03 10:05:00 | 1485.83 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-05 11:05:00 | 1575.70 | 2025-05-05 11:30:00 | 1579.44 | STOP_HIT | 1.00 | -0.24% |
