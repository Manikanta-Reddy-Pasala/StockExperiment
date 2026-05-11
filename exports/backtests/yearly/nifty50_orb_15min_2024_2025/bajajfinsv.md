# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-06-04 15:25:00 (19758 bars)
- **Last close:** 1959.50
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
| ENTRY1 | 105 |
| ENTRY2 | 0 |
| PARTIAL | 45 |
| TARGET_HIT | 24 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 81
- **Target hits / Stop hits / Partials:** 24 / 81 / 45
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 21.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 35 | 48.6% | 13 | 37 | 22 | 0.16% | 11.4% |
| BUY @ 2nd Alert (retest1) | 72 | 35 | 48.6% | 13 | 37 | 22 | 0.16% | 11.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 78 | 34 | 43.6% | 11 | 44 | 23 | 0.13% | 9.9% |
| SELL @ 2nd Alert (retest1) | 78 | 34 | 43.6% | 11 | 44 | 23 | 0.13% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 150 | 69 | 46.0% | 24 | 81 | 45 | 0.14% | 21.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1567.60 | 1570.25 | 0.00 | ORB-short ORB[1570.75,1579.35] vol=2.1x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-05-16 10:30:00 | 1570.85 | 1570.12 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 1573.55 | 1579.96 | 0.00 | ORB-short ORB[1574.00,1595.30] vol=2.7x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-05-21 10:00:00 | 1577.88 | 1577.93 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 1593.80 | 1588.85 | 0.00 | ORB-long ORB[1581.25,1591.85] vol=2.0x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 1590.35 | 1589.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 1587.70 | 1591.07 | 0.00 | ORB-short ORB[1590.40,1602.35] vol=1.6x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:00:00 | 1582.53 | 1588.57 | 0.00 | T1 1.5R @ 1582.53 |
| Target hit | 2024-05-29 15:20:00 | 1569.40 | 1576.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1547.25 | 1562.11 | 0.00 | ORB-short ORB[1558.00,1575.20] vol=1.8x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:20:00 | 1539.95 | 1553.28 | 0.00 | T1 1.5R @ 1539.95 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 1547.25 | 1552.55 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:15:00 | 1504.55 | 1490.19 | 0.00 | ORB-long ORB[1476.95,1498.95] vol=3.1x ATR=8.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 12:00:00 | 1517.01 | 1493.42 | 0.00 | T1 1.5R @ 1517.01 |
| Target hit | 2024-06-05 15:20:00 | 1520.00 | 1508.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:45:00 | 1537.45 | 1529.86 | 0.00 | ORB-long ORB[1515.20,1530.00] vol=3.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-06-06 11:20:00 | 1531.93 | 1532.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:40:00 | 1555.90 | 1546.44 | 0.00 | ORB-long ORB[1530.55,1543.90] vol=1.8x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-06-07 10:55:00 | 1550.52 | 1547.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:00:00 | 1575.70 | 1568.94 | 0.00 | ORB-long ORB[1564.65,1573.90] vol=4.0x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:20:00 | 1582.46 | 1569.74 | 0.00 | T1 1.5R @ 1582.46 |
| Target hit | 2024-06-12 15:20:00 | 1580.55 | 1576.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 1584.55 | 1590.99 | 0.00 | ORB-short ORB[1590.00,1600.00] vol=3.0x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-06-19 10:40:00 | 1588.56 | 1590.75 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 1580.00 | 1583.45 | 0.00 | ORB-short ORB[1580.55,1590.50] vol=1.8x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-06-25 09:55:00 | 1583.28 | 1582.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1596.30 | 1603.91 | 0.00 | ORB-short ORB[1603.20,1616.50] vol=2.9x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-06-26 11:30:00 | 1599.31 | 1603.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 10:35:00 | 1581.50 | 1585.98 | 0.00 | ORB-short ORB[1583.00,1590.95] vol=2.2x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-07-01 10:55:00 | 1584.51 | 1585.21 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 1568.00 | 1568.33 | 0.00 | ORB-short ORB[1572.75,1585.00] vol=3.3x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-07-02 10:45:00 | 1571.41 | 1568.40 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 1565.50 | 1570.58 | 0.00 | ORB-short ORB[1571.85,1579.60] vol=1.6x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 1568.79 | 1570.47 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 1575.35 | 1579.27 | 0.00 | ORB-short ORB[1575.85,1586.00] vol=4.7x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 1579.06 | 1578.82 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1596.00 | 1604.88 | 0.00 | ORB-short ORB[1601.60,1613.60] vol=1.6x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1599.35 | 1603.29 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:40:00 | 1627.60 | 1636.88 | 0.00 | ORB-short ORB[1633.30,1647.95] vol=1.7x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:55:00 | 1619.19 | 1634.72 | 0.00 | T1 1.5R @ 1619.19 |
| Stop hit — per-position SL triggered | 2024-07-23 11:00:00 | 1627.60 | 1627.60 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 10:55:00 | 1586.20 | 1592.69 | 0.00 | ORB-short ORB[1600.05,1611.40] vol=8.4x ATR=5.48 |
| Stop hit — per-position SL triggered | 2024-07-24 12:10:00 | 1591.68 | 1590.84 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:35:00 | 1598.00 | 1591.51 | 0.00 | ORB-long ORB[1582.50,1590.50] vol=2.3x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 10:55:00 | 1603.75 | 1593.25 | 0.00 | T1 1.5R @ 1603.75 |
| Stop hit — per-position SL triggered | 2024-07-29 12:15:00 | 1598.00 | 1597.51 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:30:00 | 1621.00 | 1608.32 | 0.00 | ORB-long ORB[1595.20,1615.50] vol=1.6x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:45:00 | 1630.58 | 1612.14 | 0.00 | T1 1.5R @ 1630.58 |
| Target hit | 2024-07-30 15:05:00 | 1640.45 | 1642.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2024-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:10:00 | 1654.70 | 1646.43 | 0.00 | ORB-long ORB[1637.65,1652.00] vol=2.3x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-07-31 11:25:00 | 1650.39 | 1646.73 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 10:45:00 | 1582.75 | 1592.15 | 0.00 | ORB-short ORB[1583.65,1605.00] vol=2.1x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:15:00 | 1574.67 | 1587.29 | 0.00 | T1 1.5R @ 1574.67 |
| Stop hit — per-position SL triggered | 2024-08-05 11:40:00 | 1582.75 | 1584.04 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:05:00 | 1575.00 | 1576.54 | 0.00 | ORB-short ORB[1576.50,1584.80] vol=13.6x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:20:00 | 1569.49 | 1576.23 | 0.00 | T1 1.5R @ 1569.49 |
| Stop hit — per-position SL triggered | 2024-08-07 12:10:00 | 1575.00 | 1574.62 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 1561.40 | 1568.07 | 0.00 | ORB-short ORB[1562.45,1576.45] vol=1.8x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1565.38 | 1567.62 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:10:00 | 1565.40 | 1556.49 | 0.00 | ORB-long ORB[1548.75,1564.00] vol=2.8x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-08-12 12:10:00 | 1562.24 | 1559.50 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:55:00 | 1530.85 | 1533.58 | 0.00 | ORB-short ORB[1535.00,1548.00] vol=1.5x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 11:05:00 | 1525.70 | 1532.78 | 0.00 | T1 1.5R @ 1525.70 |
| Stop hit — per-position SL triggered | 2024-08-16 11:25:00 | 1530.85 | 1531.99 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:55:00 | 1569.70 | 1563.27 | 0.00 | ORB-long ORB[1551.50,1560.00] vol=1.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-08-20 10:00:00 | 1566.91 | 1563.91 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:15:00 | 1617.55 | 1613.60 | 0.00 | ORB-long ORB[1599.95,1616.30] vol=3.8x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:30:00 | 1623.96 | 1615.17 | 0.00 | T1 1.5R @ 1623.96 |
| Stop hit — per-position SL triggered | 2024-08-21 11:05:00 | 1617.55 | 1616.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 1625.30 | 1622.97 | 0.00 | ORB-long ORB[1614.05,1625.00] vol=1.5x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 1622.76 | 1623.15 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 1666.20 | 1652.43 | 0.00 | ORB-long ORB[1638.75,1655.00] vol=2.5x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-08-26 09:35:00 | 1662.13 | 1653.84 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 1681.10 | 1693.40 | 0.00 | ORB-short ORB[1685.00,1702.35] vol=2.0x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:10:00 | 1673.39 | 1687.89 | 0.00 | T1 1.5R @ 1673.39 |
| Stop hit — per-position SL triggered | 2024-08-27 11:05:00 | 1681.10 | 1673.93 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 1727.45 | 1721.66 | 0.00 | ORB-long ORB[1711.55,1724.00] vol=2.8x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:40:00 | 1735.04 | 1726.02 | 0.00 | T1 1.5R @ 1735.04 |
| Target hit | 2024-08-29 12:10:00 | 1752.90 | 1753.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-09-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:10:00 | 1829.60 | 1810.05 | 0.00 | ORB-long ORB[1787.80,1803.45] vol=1.9x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:30:00 | 1840.19 | 1820.23 | 0.00 | T1 1.5R @ 1840.19 |
| Target hit | 2024-09-02 15:20:00 | 1839.00 | 1837.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-09-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:10:00 | 1859.80 | 1864.27 | 0.00 | ORB-short ORB[1862.10,1871.75] vol=1.6x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:20:00 | 1853.47 | 1863.93 | 0.00 | T1 1.5R @ 1853.47 |
| Stop hit — per-position SL triggered | 2024-09-05 11:35:00 | 1859.80 | 1863.17 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 11:05:00 | 1893.60 | 1877.01 | 0.00 | ORB-long ORB[1846.50,1874.80] vol=2.0x ATR=7.23 |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 1886.37 | 1877.55 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 1914.30 | 1901.54 | 0.00 | ORB-long ORB[1887.05,1909.60] vol=3.1x ATR=6.71 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 1907.59 | 1906.48 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 1867.00 | 1854.12 | 0.00 | ORB-long ORB[1838.55,1859.50] vol=2.5x ATR=6.65 |
| Stop hit — per-position SL triggered | 2024-09-17 10:25:00 | 1860.35 | 1859.94 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1908.70 | 1902.49 | 0.00 | ORB-long ORB[1892.10,1907.80] vol=1.7x ATR=5.42 |
| Stop hit — per-position SL triggered | 2024-09-20 11:05:00 | 1903.28 | 1903.44 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:50:00 | 1921.40 | 1916.34 | 0.00 | ORB-long ORB[1910.05,1921.00] vol=1.8x ATR=4.53 |
| Stop hit — per-position SL triggered | 2024-09-24 12:00:00 | 1916.87 | 1917.01 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:05:00 | 1952.90 | 1942.66 | 0.00 | ORB-long ORB[1932.15,1947.45] vol=2.5x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:45:00 | 1961.46 | 1947.87 | 0.00 | T1 1.5R @ 1961.46 |
| Stop hit — per-position SL triggered | 2024-09-26 11:40:00 | 1952.90 | 1951.50 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-09-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:30:00 | 1998.45 | 1985.62 | 0.00 | ORB-long ORB[1966.30,1979.95] vol=2.3x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:45:00 | 2007.27 | 1989.94 | 0.00 | T1 1.5R @ 2007.27 |
| Target hit | 2024-09-27 13:05:00 | 2003.80 | 2004.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1875.00 | 1886.46 | 0.00 | ORB-short ORB[1880.70,1901.60] vol=1.7x ATR=7.07 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 1882.07 | 1884.88 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:15:00 | 1877.15 | 1861.83 | 0.00 | ORB-long ORB[1835.55,1857.70] vol=1.6x ATR=5.46 |
| Stop hit — per-position SL triggered | 2024-10-09 11:25:00 | 1871.69 | 1862.29 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 1881.55 | 1876.79 | 0.00 | ORB-long ORB[1860.00,1878.00] vol=1.6x ATR=5.19 |
| Stop hit — per-position SL triggered | 2024-10-10 11:15:00 | 1876.36 | 1877.12 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 1886.60 | 1881.36 | 0.00 | ORB-long ORB[1871.00,1884.90] vol=1.7x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:40:00 | 1893.80 | 1884.96 | 0.00 | T1 1.5R @ 1893.80 |
| Stop hit — per-position SL triggered | 2024-10-15 09:45:00 | 1886.60 | 1886.43 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:50:00 | 1840.85 | 1843.40 | 0.00 | ORB-short ORB[1850.00,1869.45] vol=2.1x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:55:00 | 1834.49 | 1841.64 | 0.00 | T1 1.5R @ 1834.49 |
| Target hit | 2024-10-17 15:20:00 | 1814.00 | 1823.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-10-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:00:00 | 1802.15 | 1802.36 | 0.00 | ORB-short ORB[1809.05,1823.90] vol=4.2x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 10:20:00 | 1793.53 | 1801.70 | 0.00 | T1 1.5R @ 1793.53 |
| Target hit | 2024-10-21 15:20:00 | 1758.35 | 1778.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 1714.95 | 1728.73 | 0.00 | ORB-short ORB[1733.80,1751.90] vol=1.5x ATR=6.33 |
| Stop hit — per-position SL triggered | 2024-10-25 11:10:00 | 1721.28 | 1727.61 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 10:55:00 | 1733.20 | 1740.55 | 0.00 | ORB-short ORB[1736.20,1752.00] vol=1.6x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:10:00 | 1726.81 | 1739.13 | 0.00 | T1 1.5R @ 1726.81 |
| Stop hit — per-position SL triggered | 2024-10-31 11:20:00 | 1733.20 | 1737.28 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:30:00 | 1724.15 | 1734.41 | 0.00 | ORB-short ORB[1744.75,1758.00] vol=2.3x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:45:00 | 1715.25 | 1732.11 | 0.00 | T1 1.5R @ 1715.25 |
| Target hit | 2024-11-04 15:20:00 | 1712.55 | 1714.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-11-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:30:00 | 1708.00 | 1718.43 | 0.00 | ORB-short ORB[1717.50,1732.95] vol=1.5x ATR=6.19 |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 1714.19 | 1714.91 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:10:00 | 1650.80 | 1661.36 | 0.00 | ORB-short ORB[1656.65,1671.75] vol=2.3x ATR=4.86 |
| Stop hit — per-position SL triggered | 2024-11-14 12:00:00 | 1655.66 | 1659.79 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 10:40:00 | 1620.25 | 1621.95 | 0.00 | ORB-short ORB[1625.55,1641.95] vol=13.0x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 13:25:00 | 1612.09 | 1621.53 | 0.00 | T1 1.5R @ 1612.09 |
| Target hit | 2024-11-18 13:35:00 | 1619.50 | 1618.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 11:15:00 | 1597.85 | 1607.07 | 0.00 | ORB-short ORB[1606.95,1625.95] vol=2.2x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:40:00 | 1592.88 | 1600.89 | 0.00 | T1 1.5R @ 1592.88 |
| Stop hit — per-position SL triggered | 2024-11-19 12:45:00 | 1597.85 | 1600.06 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-11-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:40:00 | 1633.20 | 1624.43 | 0.00 | ORB-long ORB[1611.00,1626.30] vol=2.0x ATR=4.37 |
| Stop hit — per-position SL triggered | 2024-11-25 10:55:00 | 1628.83 | 1625.10 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:45:00 | 1606.55 | 1604.77 | 0.00 | ORB-long ORB[1595.05,1603.00] vol=4.1x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-12-03 10:40:00 | 1602.25 | 1604.76 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:30:00 | 1620.55 | 1616.64 | 0.00 | ORB-long ORB[1608.00,1619.00] vol=10.3x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:35:00 | 1627.04 | 1617.38 | 0.00 | T1 1.5R @ 1627.04 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 1620.55 | 1619.59 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1610.70 | 1623.20 | 0.00 | ORB-short ORB[1625.40,1636.75] vol=1.8x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-12-05 12:00:00 | 1615.09 | 1620.08 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:30:00 | 1657.20 | 1648.90 | 0.00 | ORB-long ORB[1637.10,1653.65] vol=2.0x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:40:00 | 1663.44 | 1654.03 | 0.00 | T1 1.5R @ 1663.44 |
| Target hit | 2024-12-10 12:05:00 | 1662.25 | 1663.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2024-12-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:20:00 | 1648.50 | 1656.68 | 0.00 | ORB-short ORB[1661.30,1674.60] vol=5.8x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:40:00 | 1641.40 | 1655.18 | 0.00 | T1 1.5R @ 1641.40 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 1648.50 | 1652.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 1665.20 | 1674.19 | 0.00 | ORB-short ORB[1672.05,1683.95] vol=2.0x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-12-16 12:30:00 | 1669.03 | 1670.05 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 1655.20 | 1661.61 | 0.00 | ORB-short ORB[1662.25,1673.65] vol=1.5x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:50:00 | 1650.14 | 1657.79 | 0.00 | T1 1.5R @ 1650.14 |
| Target hit | 2024-12-17 10:50:00 | 1650.95 | 1646.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2024-12-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:20:00 | 1629.00 | 1632.06 | 0.00 | ORB-short ORB[1635.10,1647.40] vol=1.6x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-12-18 10:30:00 | 1633.18 | 1632.02 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 09:45:00 | 1599.50 | 1606.01 | 0.00 | ORB-short ORB[1601.10,1624.15] vol=1.6x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 12:45:00 | 1591.77 | 1600.32 | 0.00 | T1 1.5R @ 1591.77 |
| Target hit | 2024-12-19 15:20:00 | 1593.40 | 1594.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2024-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:10:00 | 1580.45 | 1574.90 | 0.00 | ORB-long ORB[1561.10,1574.85] vol=1.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-12-27 12:10:00 | 1576.49 | 1575.50 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 1579.65 | 1573.35 | 0.00 | ORB-long ORB[1565.00,1576.35] vol=1.7x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-01-01 13:30:00 | 1575.90 | 1576.29 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 11:15:00 | 1713.85 | 1702.90 | 0.00 | ORB-long ORB[1692.00,1710.70] vol=2.2x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-01-03 12:20:00 | 1708.52 | 1706.54 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 09:35:00 | 1683.10 | 1677.12 | 0.00 | ORB-long ORB[1670.00,1682.40] vol=1.6x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:45:00 | 1689.84 | 1680.33 | 0.00 | T1 1.5R @ 1689.84 |
| Target hit | 2025-01-08 10:20:00 | 1688.75 | 1692.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1680.10 | 1685.10 | 0.00 | ORB-short ORB[1685.05,1700.50] vol=4.5x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-01-09 10:50:00 | 1684.65 | 1684.90 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:10:00 | 1699.15 | 1690.68 | 0.00 | ORB-long ORB[1682.00,1696.80] vol=2.1x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:20:00 | 1707.22 | 1692.45 | 0.00 | T1 1.5R @ 1707.22 |
| Stop hit — per-position SL triggered | 2025-01-10 10:35:00 | 1699.15 | 1695.37 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:05:00 | 1707.15 | 1699.80 | 0.00 | ORB-long ORB[1684.00,1706.75] vol=1.9x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-01-14 11:25:00 | 1701.96 | 1701.16 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-01-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:45:00 | 1696.85 | 1687.20 | 0.00 | ORB-long ORB[1674.40,1691.00] vol=2.0x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:50:00 | 1702.91 | 1690.42 | 0.00 | T1 1.5R @ 1702.91 |
| Stop hit — per-position SL triggered | 2025-01-20 11:05:00 | 1696.85 | 1691.43 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:35:00 | 1735.00 | 1743.31 | 0.00 | ORB-short ORB[1736.35,1747.50] vol=1.7x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-01-21 09:45:00 | 1739.95 | 1742.77 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:50:00 | 1754.55 | 1750.26 | 0.00 | ORB-long ORB[1730.45,1748.85] vol=2.8x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:10:00 | 1761.15 | 1751.80 | 0.00 | T1 1.5R @ 1761.15 |
| Stop hit — per-position SL triggered | 2025-01-23 11:45:00 | 1754.55 | 1752.99 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-01-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:00:00 | 1737.50 | 1737.56 | 0.00 | ORB-short ORB[1739.10,1754.90] vol=1.8x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-01-24 11:40:00 | 1741.97 | 1737.84 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-27 09:35:00 | 1726.70 | 1720.99 | 0.00 | ORB-long ORB[1705.00,1726.30] vol=1.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 1720.92 | 1724.09 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-01-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:25:00 | 1792.90 | 1781.56 | 0.00 | ORB-long ORB[1767.25,1776.95] vol=2.0x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:10:00 | 1801.17 | 1786.52 | 0.00 | T1 1.5R @ 1801.17 |
| Stop hit — per-position SL triggered | 2025-01-29 12:45:00 | 1792.90 | 1790.52 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-02-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:10:00 | 1797.00 | 1805.25 | 0.00 | ORB-short ORB[1800.00,1814.80] vol=1.5x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:20:00 | 1786.98 | 1801.97 | 0.00 | T1 1.5R @ 1786.98 |
| Target hit | 2025-02-04 13:30:00 | 1792.95 | 1791.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 1781.25 | 1802.29 | 0.00 | ORB-short ORB[1804.75,1826.55] vol=1.9x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 1785.99 | 1801.91 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-02-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:55:00 | 1790.95 | 1801.11 | 0.00 | ORB-short ORB[1794.50,1812.95] vol=1.7x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:10:00 | 1784.84 | 1798.26 | 0.00 | T1 1.5R @ 1784.84 |
| Stop hit — per-position SL triggered | 2025-02-06 15:05:00 | 1790.95 | 1791.25 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:40:00 | 1773.90 | 1776.38 | 0.00 | ORB-short ORB[1774.35,1795.30] vol=5.6x ATR=4.94 |
| Stop hit — per-position SL triggered | 2025-02-11 10:40:00 | 1778.84 | 1775.15 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-02-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:55:00 | 1732.80 | 1745.36 | 0.00 | ORB-short ORB[1744.85,1766.30] vol=2.5x ATR=7.43 |
| Stop hit — per-position SL triggered | 2025-02-12 10:05:00 | 1740.23 | 1743.91 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 1821.50 | 1808.41 | 0.00 | ORB-long ORB[1796.65,1814.70] vol=1.9x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:45:00 | 1831.33 | 1814.57 | 0.00 | T1 1.5R @ 1831.33 |
| Target hit | 2025-02-13 10:55:00 | 1830.35 | 1834.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2025-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 09:50:00 | 1906.50 | 1899.43 | 0.00 | ORB-long ORB[1880.80,1892.95] vol=1.7x ATR=7.28 |
| Stop hit — per-position SL triggered | 2025-02-18 11:15:00 | 1899.22 | 1902.92 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 1879.50 | 1870.46 | 0.00 | ORB-long ORB[1857.80,1874.00] vol=2.1x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:35:00 | 1885.92 | 1874.41 | 0.00 | T1 1.5R @ 1885.92 |
| Target hit | 2025-02-25 11:00:00 | 1883.40 | 1884.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 87 — SELL (started 2025-02-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:35:00 | 1888.15 | 1899.51 | 0.00 | ORB-short ORB[1899.30,1920.00] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-02-28 11:30:00 | 1893.85 | 1896.15 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-03-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 11:10:00 | 1806.25 | 1816.85 | 0.00 | ORB-short ORB[1818.75,1833.50] vol=2.0x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:30:00 | 1799.99 | 1815.01 | 0.00 | T1 1.5R @ 1799.99 |
| Target hit | 2025-03-04 15:20:00 | 1789.05 | 1799.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2025-03-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:00:00 | 1857.60 | 1849.12 | 0.00 | ORB-long ORB[1837.30,1856.95] vol=2.9x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-03-07 10:50:00 | 1850.73 | 1852.81 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-11 09:30:00 | 1810.00 | 1817.38 | 0.00 | ORB-short ORB[1813.55,1836.85] vol=2.9x ATR=6.31 |
| Stop hit — per-position SL triggered | 2025-03-11 09:35:00 | 1816.31 | 1816.16 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 1839.50 | 1830.31 | 0.00 | ORB-long ORB[1820.00,1835.45] vol=1.7x ATR=5.12 |
| Stop hit — per-position SL triggered | 2025-03-13 09:40:00 | 1834.38 | 1830.76 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2025-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 11:05:00 | 1850.00 | 1852.80 | 0.00 | ORB-short ORB[1851.55,1860.90] vol=16.8x ATR=3.97 |
| Stop hit — per-position SL triggered | 2025-03-19 12:55:00 | 1853.97 | 1852.21 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-03-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:20:00 | 1818.00 | 1839.50 | 0.00 | ORB-short ORB[1840.05,1852.50] vol=1.7x ATR=5.91 |
| Stop hit — per-position SL triggered | 2025-03-20 10:25:00 | 1823.91 | 1838.71 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 1853.10 | 1858.91 | 0.00 | ORB-short ORB[1853.50,1880.00] vol=7.0x ATR=6.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:50:00 | 1843.33 | 1857.71 | 0.00 | T1 1.5R @ 1843.33 |
| Target hit | 2025-03-21 14:15:00 | 1842.90 | 1842.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 95 — BUY (started 2025-03-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:50:00 | 1875.60 | 1868.30 | 0.00 | ORB-long ORB[1845.50,1866.90] vol=3.9x ATR=5.77 |
| Stop hit — per-position SL triggered | 2025-03-24 10:05:00 | 1869.83 | 1868.61 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 1915.25 | 1903.98 | 0.00 | ORB-long ORB[1894.90,1908.00] vol=2.8x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:35:00 | 1922.48 | 1908.63 | 0.00 | T1 1.5R @ 1922.48 |
| Target hit | 2025-03-25 11:35:00 | 1927.80 | 1933.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 97 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:15:00 | 1966.95 | 1953.50 | 0.00 | ORB-long ORB[1932.60,1959.90] vol=2.4x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:40:00 | 1974.45 | 1956.24 | 0.00 | T1 1.5R @ 1974.45 |
| Target hit | 2025-03-27 15:20:00 | 2004.25 | 1992.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 98 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 1954.00 | 1983.30 | 0.00 | ORB-short ORB[1980.10,2002.00] vol=1.7x ATR=7.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:10:00 | 1943.15 | 1975.17 | 0.00 | T1 1.5R @ 1943.15 |
| Stop hit — per-position SL triggered | 2025-04-01 12:30:00 | 1954.00 | 1966.67 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2025-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:35:00 | 1926.00 | 1918.96 | 0.00 | ORB-long ORB[1908.00,1925.95] vol=1.8x ATR=5.40 |
| Stop hit — per-position SL triggered | 2025-04-04 09:40:00 | 1920.60 | 1918.09 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2025-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:45:00 | 1981.90 | 1968.51 | 0.00 | ORB-long ORB[1955.00,1970.60] vol=2.8x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-04-17 10:00:00 | 1976.28 | 1971.20 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2025-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:35:00 | 2088.40 | 2062.51 | 0.00 | ORB-long ORB[2035.70,2062.50] vol=1.9x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:50:00 | 2099.75 | 2070.77 | 0.00 | T1 1.5R @ 2099.75 |
| Target hit | 2025-04-21 15:20:00 | 2106.00 | 2090.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 102 — SELL (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 11:15:00 | 2074.00 | 2087.91 | 0.00 | ORB-short ORB[2087.60,2110.00] vol=1.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-04-22 11:20:00 | 2079.28 | 2087.61 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 2077.10 | 2101.24 | 0.00 | ORB-short ORB[2104.70,2125.00] vol=2.0x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 2066.35 | 2095.33 | 0.00 | T1 1.5R @ 2066.35 |
| Target hit | 2025-04-25 13:15:00 | 2062.30 | 2060.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 104 — SELL (started 2025-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:05:00 | 2020.00 | 2024.47 | 0.00 | ORB-short ORB[2021.60,2042.40] vol=1.6x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-05-06 13:05:00 | 2025.98 | 2023.53 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2025-05-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 10:50:00 | 1966.60 | 1975.48 | 0.00 | ORB-short ORB[1970.00,1999.00] vol=2.0x ATR=7.54 |
| Stop hit — per-position SL triggered | 2025-05-09 11:25:00 | 1974.14 | 1975.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:20:00 | 1567.60 | 2024-05-16 10:30:00 | 1570.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-21 09:30:00 | 1573.55 | 2024-05-21 10:00:00 | 1577.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-23 09:40:00 | 1593.80 | 2024-05-23 09:50:00 | 1590.35 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-29 10:45:00 | 1587.70 | 2024-05-29 11:00:00 | 1582.53 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-29 10:45:00 | 1587.70 | 2024-05-29 15:20:00 | 1569.40 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1547.25 | 2024-05-30 10:20:00 | 1539.95 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1547.25 | 2024-05-30 10:35:00 | 1547.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1504.55 | 2024-06-05 12:00:00 | 1517.01 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1504.55 | 2024-06-05 15:20:00 | 1520.00 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-06-06 10:45:00 | 1537.45 | 2024-06-06 11:20:00 | 1531.93 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-07 10:40:00 | 1555.90 | 2024-06-07 10:55:00 | 1550.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-12 10:00:00 | 1575.70 | 2024-06-12 10:20:00 | 1582.46 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-12 10:00:00 | 1575.70 | 2024-06-12 15:20:00 | 1580.55 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-06-19 10:35:00 | 1584.55 | 2024-06-19 10:40:00 | 1588.56 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1580.00 | 2024-06-25 09:55:00 | 1583.28 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-26 11:15:00 | 1596.30 | 2024-06-26 11:30:00 | 1599.31 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-01 10:35:00 | 1581.50 | 2024-07-01 10:55:00 | 1584.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-02 10:35:00 | 1568.00 | 2024-07-02 10:45:00 | 1571.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-08 09:55:00 | 1565.50 | 2024-07-08 10:00:00 | 1568.79 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-10 10:35:00 | 1575.35 | 2024-07-10 10:45:00 | 1579.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1596.00 | 2024-07-18 09:40:00 | 1599.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-23 09:40:00 | 1627.60 | 2024-07-23 09:55:00 | 1619.19 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-23 09:40:00 | 1627.60 | 2024-07-23 11:00:00 | 1627.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-24 10:55:00 | 1586.20 | 2024-07-24 12:10:00 | 1591.68 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-29 10:35:00 | 1598.00 | 2024-07-29 10:55:00 | 1603.75 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-29 10:35:00 | 1598.00 | 2024-07-29 12:15:00 | 1598.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 09:30:00 | 1621.00 | 2024-07-30 09:45:00 | 1630.58 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-30 09:30:00 | 1621.00 | 2024-07-30 15:05:00 | 1640.45 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2024-07-31 11:10:00 | 1654.70 | 2024-07-31 11:25:00 | 1650.39 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-05 10:45:00 | 1582.75 | 2024-08-05 11:15:00 | 1574.67 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-08-05 10:45:00 | 1582.75 | 2024-08-05 11:40:00 | 1582.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-07 11:05:00 | 1575.00 | 2024-08-07 11:20:00 | 1569.49 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-08-07 11:05:00 | 1575.00 | 2024-08-07 12:10:00 | 1575.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:05:00 | 1561.40 | 2024-08-08 10:15:00 | 1565.38 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-12 11:10:00 | 1565.40 | 2024-08-12 12:10:00 | 1562.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-16 10:55:00 | 1530.85 | 2024-08-16 11:05:00 | 1525.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-08-16 10:55:00 | 1530.85 | 2024-08-16 11:25:00 | 1530.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 09:55:00 | 1569.70 | 2024-08-20 10:00:00 | 1566.91 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-21 10:15:00 | 1617.55 | 2024-08-21 10:30:00 | 1623.96 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-21 10:15:00 | 1617.55 | 2024-08-21 11:05:00 | 1617.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 11:00:00 | 1625.30 | 2024-08-22 11:15:00 | 1622.76 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-08-26 09:30:00 | 1666.20 | 2024-08-26 09:35:00 | 1662.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-27 09:35:00 | 1681.10 | 2024-08-27 10:10:00 | 1673.39 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-27 09:35:00 | 1681.10 | 2024-08-27 11:05:00 | 1681.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:35:00 | 1727.45 | 2024-08-29 09:40:00 | 1735.04 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-29 09:35:00 | 1727.45 | 2024-08-29 12:10:00 | 1752.90 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2024-09-02 10:10:00 | 1829.60 | 2024-09-02 10:30:00 | 1840.19 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-02 10:10:00 | 1829.60 | 2024-09-02 15:20:00 | 1839.00 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-05 11:10:00 | 1859.80 | 2024-09-05 11:20:00 | 1853.47 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-05 11:10:00 | 1859.80 | 2024-09-05 11:35:00 | 1859.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 11:05:00 | 1893.60 | 2024-09-13 11:15:00 | 1886.37 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-16 09:30:00 | 1914.30 | 2024-09-16 09:45:00 | 1907.59 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-17 10:10:00 | 1867.00 | 2024-09-17 10:25:00 | 1860.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1908.70 | 2024-09-20 11:05:00 | 1903.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-24 10:50:00 | 1921.40 | 2024-09-24 12:00:00 | 1916.87 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-26 10:05:00 | 1952.90 | 2024-09-26 10:45:00 | 1961.46 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-26 10:05:00 | 1952.90 | 2024-09-26 11:40:00 | 1952.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:30:00 | 1998.45 | 2024-09-27 10:45:00 | 2007.27 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-27 10:30:00 | 1998.45 | 2024-09-27 13:05:00 | 2003.80 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2024-10-07 10:45:00 | 1875.00 | 2024-10-07 10:55:00 | 1882.07 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-09 11:15:00 | 1877.15 | 2024-10-09 11:25:00 | 1871.69 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-10 11:00:00 | 1881.55 | 2024-10-10 11:15:00 | 1876.36 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-15 09:30:00 | 1886.60 | 2024-10-15 09:40:00 | 1893.80 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-10-15 09:30:00 | 1886.60 | 2024-10-15 09:45:00 | 1886.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:50:00 | 1840.85 | 2024-10-17 10:55:00 | 1834.49 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-17 10:50:00 | 1840.85 | 2024-10-17 15:20:00 | 1814.00 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2024-10-21 10:00:00 | 1802.15 | 2024-10-21 10:20:00 | 1793.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-21 10:00:00 | 1802.15 | 2024-10-21 15:20:00 | 1758.35 | TARGET_HIT | 0.50 | 2.43% |
| SELL | retest1 | 2024-10-25 10:55:00 | 1714.95 | 2024-10-25 11:10:00 | 1721.28 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-31 10:55:00 | 1733.20 | 2024-10-31 11:10:00 | 1726.81 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-31 10:55:00 | 1733.20 | 2024-10-31 11:20:00 | 1733.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:30:00 | 1724.15 | 2024-11-04 10:45:00 | 1715.25 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-04 10:30:00 | 1724.15 | 2024-11-04 15:20:00 | 1712.55 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-12 10:30:00 | 1708.00 | 2024-11-12 11:15:00 | 1714.19 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-14 11:10:00 | 1650.80 | 2024-11-14 12:00:00 | 1655.66 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-18 10:40:00 | 1620.25 | 2024-11-18 13:25:00 | 1612.09 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-18 10:40:00 | 1620.25 | 2024-11-18 13:35:00 | 1619.50 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-11-19 11:15:00 | 1597.85 | 2024-11-19 11:40:00 | 1592.88 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-11-19 11:15:00 | 1597.85 | 2024-11-19 12:45:00 | 1597.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 10:40:00 | 1633.20 | 2024-11-25 10:55:00 | 1628.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-03 09:45:00 | 1606.55 | 2024-12-03 10:40:00 | 1602.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1620.55 | 2024-12-04 10:35:00 | 1627.04 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1620.55 | 2024-12-04 11:55:00 | 1620.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1610.70 | 2024-12-05 12:00:00 | 1615.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-10 09:30:00 | 1657.20 | 2024-12-10 09:40:00 | 1663.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-10 09:30:00 | 1657.20 | 2024-12-10 12:05:00 | 1662.25 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-13 10:20:00 | 1648.50 | 2024-12-13 10:40:00 | 1641.40 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-13 10:20:00 | 1648.50 | 2024-12-13 11:10:00 | 1648.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1665.20 | 2024-12-16 12:30:00 | 1669.03 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-17 09:40:00 | 1655.20 | 2024-12-17 09:50:00 | 1650.14 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-17 09:40:00 | 1655.20 | 2024-12-17 10:50:00 | 1650.95 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-12-18 10:20:00 | 1629.00 | 2024-12-18 10:30:00 | 1633.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-19 09:45:00 | 1599.50 | 2024-12-19 12:45:00 | 1591.77 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-19 09:45:00 | 1599.50 | 2024-12-19 15:20:00 | 1593.40 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-27 11:10:00 | 1580.45 | 2024-12-27 12:10:00 | 1576.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-01 10:50:00 | 1579.65 | 2025-01-01 13:30:00 | 1575.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-03 11:15:00 | 1713.85 | 2025-01-03 12:20:00 | 1708.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-08 09:35:00 | 1683.10 | 2025-01-08 09:45:00 | 1689.84 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-08 09:35:00 | 1683.10 | 2025-01-08 10:20:00 | 1688.75 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1680.10 | 2025-01-09 10:50:00 | 1684.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-10 10:10:00 | 1699.15 | 2025-01-10 10:20:00 | 1707.22 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-10 10:10:00 | 1699.15 | 2025-01-10 10:35:00 | 1699.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-14 11:05:00 | 1707.15 | 2025-01-14 11:25:00 | 1701.96 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-20 10:45:00 | 1696.85 | 2025-01-20 10:50:00 | 1702.91 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-20 10:45:00 | 1696.85 | 2025-01-20 11:05:00 | 1696.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 09:35:00 | 1735.00 | 2025-01-21 09:45:00 | 1739.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-23 10:50:00 | 1754.55 | 2025-01-23 11:10:00 | 1761.15 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-23 10:50:00 | 1754.55 | 2025-01-23 11:45:00 | 1754.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 11:00:00 | 1737.50 | 2025-01-24 11:40:00 | 1741.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-27 09:35:00 | 1726.70 | 2025-01-27 10:15:00 | 1720.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-29 10:25:00 | 1792.90 | 2025-01-29 11:10:00 | 1801.17 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-29 10:25:00 | 1792.90 | 2025-01-29 12:45:00 | 1792.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 10:10:00 | 1797.00 | 2025-02-04 10:20:00 | 1786.98 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-04 10:10:00 | 1797.00 | 2025-02-04 13:30:00 | 1792.95 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-02-05 11:15:00 | 1781.25 | 2025-02-05 11:20:00 | 1785.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-06 10:55:00 | 1790.95 | 2025-02-06 11:10:00 | 1784.84 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-02-06 10:55:00 | 1790.95 | 2025-02-06 15:05:00 | 1790.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 09:40:00 | 1773.90 | 2025-02-11 10:40:00 | 1778.84 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-12 09:55:00 | 1732.80 | 2025-02-12 10:05:00 | 1740.23 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-13 09:30:00 | 1821.50 | 2025-02-13 09:45:00 | 1831.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-13 09:30:00 | 1821.50 | 2025-02-13 10:55:00 | 1830.35 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-18 09:50:00 | 1906.50 | 2025-02-18 11:15:00 | 1899.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-25 09:30:00 | 1879.50 | 2025-02-25 09:35:00 | 1885.92 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-02-25 09:30:00 | 1879.50 | 2025-02-25 11:00:00 | 1883.40 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-02-28 10:35:00 | 1888.15 | 2025-02-28 11:30:00 | 1893.85 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-04 11:10:00 | 1806.25 | 2025-03-04 11:30:00 | 1799.99 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-03-04 11:10:00 | 1806.25 | 2025-03-04 15:20:00 | 1789.05 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2025-03-07 10:00:00 | 1857.60 | 2025-03-07 10:50:00 | 1850.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-11 09:30:00 | 1810.00 | 2025-03-11 09:35:00 | 1816.31 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-13 09:35:00 | 1839.50 | 2025-03-13 09:40:00 | 1834.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-03-19 11:05:00 | 1850.00 | 2025-03-19 12:55:00 | 1853.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-03-20 10:20:00 | 1818.00 | 2025-03-20 10:25:00 | 1823.91 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-21 09:40:00 | 1853.10 | 2025-03-21 09:50:00 | 1843.33 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-03-21 09:40:00 | 1853.10 | 2025-03-21 14:15:00 | 1842.90 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-24 09:50:00 | 1875.60 | 2025-03-24 10:05:00 | 1869.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-25 09:30:00 | 1915.25 | 2025-03-25 09:35:00 | 1922.48 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-25 09:30:00 | 1915.25 | 2025-03-25 11:35:00 | 1927.80 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-03-27 11:15:00 | 1966.95 | 2025-03-27 11:40:00 | 1974.45 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-27 11:15:00 | 1966.95 | 2025-03-27 15:20:00 | 2004.25 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2025-04-01 10:55:00 | 1954.00 | 2025-04-01 11:10:00 | 1943.15 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-01 10:55:00 | 1954.00 | 2025-04-01 12:30:00 | 1954.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-04 09:35:00 | 1926.00 | 2025-04-04 09:40:00 | 1920.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-17 09:45:00 | 1981.90 | 2025-04-17 10:00:00 | 1976.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-21 10:35:00 | 2088.40 | 2025-04-21 10:50:00 | 2099.75 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-21 10:35:00 | 2088.40 | 2025-04-21 15:20:00 | 2106.00 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-04-22 11:15:00 | 2074.00 | 2025-04-22 11:20:00 | 2079.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-04-25 09:40:00 | 2077.10 | 2025-04-25 09:45:00 | 2066.35 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 09:40:00 | 2077.10 | 2025-04-25 13:15:00 | 2062.30 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-05-06 11:05:00 | 2020.00 | 2025-05-06 13:05:00 | 2025.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-05-09 10:50:00 | 1966.60 | 2025-05-09 11:25:00 | 1974.14 | STOP_HIT | 1.00 | -0.38% |
