# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-11-29 15:25:00 (27292 bars)
- **Last close:** 1409.80
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
| PARTIAL | 40 |
| TARGET_HIT | 13 |
| STOP_HIT | 92 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 92
- **Target hits / Stop hits / Partials:** 13 / 92 / 40
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 6.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 29 | 37.7% | 7 | 48 | 22 | 0.04% | 2.9% |
| BUY @ 2nd Alert (retest1) | 77 | 29 | 37.7% | 7 | 48 | 22 | 0.04% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 24 | 35.3% | 6 | 44 | 18 | 0.05% | 3.1% |
| SELL @ 2nd Alert (retest1) | 68 | 24 | 35.3% | 6 | 44 | 18 | 0.05% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 145 | 53 | 36.6% | 13 | 92 | 40 | 0.04% | 6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-12 11:15:00 | 1528.00 | 1533.40 | 0.00 | ORB-short ORB[1530.05,1539.95] vol=1.8x ATR=3.44 |
| Stop hit — per-position SL triggered | 2023-05-12 11:40:00 | 1531.44 | 1533.25 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 10:10:00 | 1531.55 | 1527.27 | 0.00 | ORB-long ORB[1521.15,1530.00] vol=1.6x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 10:15:00 | 1536.70 | 1528.50 | 0.00 | T1 1.5R @ 1536.70 |
| Stop hit — per-position SL triggered | 2023-05-15 11:20:00 | 1531.55 | 1530.35 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:55:00 | 1535.25 | 1541.57 | 0.00 | ORB-short ORB[1542.00,1551.50] vol=3.6x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-05-17 11:00:00 | 1538.12 | 1541.44 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:55:00 | 1533.00 | 1537.05 | 0.00 | ORB-short ORB[1534.80,1544.45] vol=1.9x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 11:25:00 | 1529.10 | 1535.66 | 0.00 | T1 1.5R @ 1529.10 |
| Target hit | 2023-05-18 13:55:00 | 1531.00 | 1529.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2023-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 09:50:00 | 1514.80 | 1519.35 | 0.00 | ORB-short ORB[1514.85,1530.00] vol=2.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2023-05-23 10:20:00 | 1519.10 | 1518.68 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:45:00 | 1553.10 | 1546.18 | 0.00 | ORB-long ORB[1535.00,1548.40] vol=3.1x ATR=3.95 |
| Stop hit — per-position SL triggered | 2023-05-25 09:55:00 | 1549.15 | 1547.26 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 11:00:00 | 1574.25 | 1567.59 | 0.00 | ORB-long ORB[1560.00,1574.00] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-05-26 11:40:00 | 1571.03 | 1569.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 1568.65 | 1574.36 | 0.00 | ORB-short ORB[1572.05,1583.75] vol=1.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-06-02 11:10:00 | 1571.40 | 1573.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:30:00 | 1567.35 | 1573.47 | 0.00 | ORB-short ORB[1569.25,1583.45] vol=1.7x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:55:00 | 1563.04 | 1572.33 | 0.00 | T1 1.5R @ 1563.04 |
| Stop hit — per-position SL triggered | 2023-06-06 11:30:00 | 1567.35 | 1570.85 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:55:00 | 1576.95 | 1573.79 | 0.00 | ORB-long ORB[1560.80,1573.90] vol=5.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-06-12 13:40:00 | 1574.08 | 1574.72 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 1614.20 | 1606.02 | 0.00 | ORB-long ORB[1597.05,1613.30] vol=3.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2023-06-14 11:05:00 | 1609.99 | 1606.56 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:00:00 | 1613.00 | 1604.78 | 0.00 | ORB-long ORB[1593.90,1608.45] vol=1.7x ATR=3.73 |
| Stop hit — per-position SL triggered | 2023-06-15 10:10:00 | 1609.27 | 1606.16 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:50:00 | 1631.80 | 1625.65 | 0.00 | ORB-long ORB[1615.75,1628.35] vol=1.8x ATR=3.36 |
| Stop hit — per-position SL triggered | 2023-06-16 09:55:00 | 1628.44 | 1626.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 09:40:00 | 1649.25 | 1646.68 | 0.00 | ORB-long ORB[1640.95,1649.00] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2023-06-21 10:05:00 | 1646.52 | 1647.64 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:35:00 | 1630.05 | 1626.84 | 0.00 | ORB-long ORB[1617.35,1624.90] vol=5.4x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 10:15:00 | 1635.13 | 1629.79 | 0.00 | T1 1.5R @ 1635.13 |
| Stop hit — per-position SL triggered | 2023-06-26 10:30:00 | 1630.05 | 1634.07 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:45:00 | 1659.60 | 1653.24 | 0.00 | ORB-long ORB[1641.65,1653.85] vol=1.6x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 10:20:00 | 1665.05 | 1657.34 | 0.00 | T1 1.5R @ 1665.05 |
| Target hit | 2023-06-27 12:50:00 | 1663.75 | 1664.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2023-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:50:00 | 1665.75 | 1673.70 | 0.00 | ORB-short ORB[1674.40,1681.75] vol=1.7x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 10:00:00 | 1660.56 | 1670.37 | 0.00 | T1 1.5R @ 1660.56 |
| Target hit | 2023-07-03 14:25:00 | 1647.00 | 1646.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2023-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:30:00 | 1720.90 | 1702.16 | 0.00 | ORB-long ORB[1673.80,1690.80] vol=9.8x ATR=6.12 |
| Stop hit — per-position SL triggered | 2023-07-07 09:40:00 | 1714.78 | 1707.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:35:00 | 1685.90 | 1679.02 | 0.00 | ORB-long ORB[1671.40,1682.00] vol=2.2x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 11:45:00 | 1690.34 | 1684.17 | 0.00 | T1 1.5R @ 1690.34 |
| Target hit | 2023-07-11 15:20:00 | 1691.10 | 1691.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 10:40:00 | 1682.00 | 1687.06 | 0.00 | ORB-short ORB[1690.00,1695.35] vol=2.2x ATR=2.59 |
| Stop hit — per-position SL triggered | 2023-07-18 10:50:00 | 1684.59 | 1686.40 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:45:00 | 1700.55 | 1695.61 | 0.00 | ORB-long ORB[1688.45,1699.00] vol=1.7x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:15:00 | 1707.36 | 1702.37 | 0.00 | T1 1.5R @ 1707.36 |
| Target hit | 2023-07-19 10:25:00 | 1702.15 | 1702.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2023-07-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:45:00 | 1687.60 | 1691.91 | 0.00 | ORB-short ORB[1688.00,1700.30] vol=1.8x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:10:00 | 1683.40 | 1687.56 | 0.00 | T1 1.5R @ 1683.40 |
| Stop hit — per-position SL triggered | 2023-07-20 11:00:00 | 1687.60 | 1687.29 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:50:00 | 1709.20 | 1701.17 | 0.00 | ORB-long ORB[1693.05,1707.15] vol=6.6x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:55:00 | 1716.28 | 1703.82 | 0.00 | T1 1.5R @ 1716.28 |
| Stop hit — per-position SL triggered | 2023-07-26 11:00:00 | 1709.20 | 1703.91 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:30:00 | 1698.95 | 1706.08 | 0.00 | ORB-short ORB[1703.05,1716.80] vol=1.6x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:25:00 | 1693.96 | 1703.51 | 0.00 | T1 1.5R @ 1693.96 |
| Stop hit — per-position SL triggered | 2023-07-27 11:55:00 | 1698.95 | 1702.02 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 09:30:00 | 1709.70 | 1717.97 | 0.00 | ORB-short ORB[1712.10,1725.80] vol=2.2x ATR=4.06 |
| Stop hit — per-position SL triggered | 2023-07-31 09:40:00 | 1713.76 | 1717.23 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 10:00:00 | 1748.40 | 1742.46 | 0.00 | ORB-long ORB[1735.00,1744.95] vol=1.8x ATR=4.46 |
| Stop hit — per-position SL triggered | 2023-08-01 10:20:00 | 1743.94 | 1743.87 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 09:45:00 | 1740.00 | 1748.71 | 0.00 | ORB-short ORB[1750.00,1758.00] vol=1.6x ATR=4.06 |
| Stop hit — per-position SL triggered | 2023-08-04 10:10:00 | 1744.06 | 1743.24 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:40:00 | 1747.25 | 1755.03 | 0.00 | ORB-short ORB[1749.30,1762.20] vol=2.3x ATR=4.07 |
| Stop hit — per-position SL triggered | 2023-08-07 10:45:00 | 1751.32 | 1754.62 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:30:00 | 1664.00 | 1654.45 | 0.00 | ORB-long ORB[1638.40,1656.05] vol=2.1x ATR=6.18 |
| Stop hit — per-position SL triggered | 2023-08-11 11:00:00 | 1657.82 | 1654.98 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:05:00 | 1650.00 | 1647.82 | 0.00 | ORB-long ORB[1636.25,1645.40] vol=7.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 10:10:00 | 1654.99 | 1657.70 | 0.00 | T1 1.5R @ 1654.99 |
| Stop hit — per-position SL triggered | 2023-08-17 10:20:00 | 1650.00 | 1657.37 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:30:00 | 1706.00 | 1700.49 | 0.00 | ORB-long ORB[1693.00,1702.85] vol=3.8x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 10:50:00 | 1711.58 | 1705.22 | 0.00 | T1 1.5R @ 1711.58 |
| Stop hit — per-position SL triggered | 2023-08-29 10:55:00 | 1706.00 | 1705.38 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:40:00 | 1705.00 | 1700.49 | 0.00 | ORB-long ORB[1691.10,1700.35] vol=1.7x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:30:00 | 1710.48 | 1704.07 | 0.00 | T1 1.5R @ 1710.48 |
| Target hit | 2023-08-30 13:35:00 | 1711.50 | 1712.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 1673.35 | 1683.37 | 0.00 | ORB-short ORB[1682.05,1691.00] vol=1.8x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 11:20:00 | 1668.84 | 1680.34 | 0.00 | T1 1.5R @ 1668.84 |
| Target hit | 2023-09-04 15:20:00 | 1655.00 | 1660.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2023-09-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:40:00 | 1692.95 | 1688.15 | 0.00 | ORB-long ORB[1681.90,1690.00] vol=1.8x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-09-06 09:50:00 | 1689.01 | 1689.10 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:05:00 | 1704.05 | 1695.28 | 0.00 | ORB-long ORB[1687.40,1702.95] vol=3.8x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:10:00 | 1710.79 | 1696.91 | 0.00 | T1 1.5R @ 1710.79 |
| Target hit | 2023-09-07 11:25:00 | 1723.15 | 1726.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2023-09-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 10:30:00 | 1681.55 | 1696.63 | 0.00 | ORB-short ORB[1711.05,1724.95] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2023-09-12 10:40:00 | 1686.88 | 1695.94 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:40:00 | 1668.90 | 1675.73 | 0.00 | ORB-short ORB[1670.05,1684.15] vol=1.7x ATR=4.01 |
| Stop hit — per-position SL triggered | 2023-09-14 11:35:00 | 1672.91 | 1674.05 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:10:00 | 1682.80 | 1673.17 | 0.00 | ORB-long ORB[1658.60,1673.15] vol=4.5x ATR=4.48 |
| Stop hit — per-position SL triggered | 2023-09-15 10:20:00 | 1678.32 | 1673.75 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 11:10:00 | 1673.50 | 1667.09 | 0.00 | ORB-long ORB[1660.00,1670.45] vol=1.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 11:15:00 | 1676.77 | 1668.01 | 0.00 | T1 1.5R @ 1676.77 |
| Stop hit — per-position SL triggered | 2023-09-21 11:20:00 | 1673.50 | 1668.48 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:55:00 | 1641.60 | 1652.84 | 0.00 | ORB-short ORB[1650.10,1663.00] vol=2.1x ATR=4.34 |
| Stop hit — per-position SL triggered | 2023-09-22 10:05:00 | 1645.94 | 1651.63 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 10:40:00 | 1641.70 | 1636.52 | 0.00 | ORB-long ORB[1621.35,1638.50] vol=4.2x ATR=3.76 |
| Stop hit — per-position SL triggered | 2023-09-25 11:05:00 | 1637.94 | 1637.48 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 09:30:00 | 1598.85 | 1605.25 | 0.00 | ORB-short ORB[1603.10,1617.30] vol=2.8x ATR=3.59 |
| Stop hit — per-position SL triggered | 2023-09-27 09:35:00 | 1602.44 | 1605.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 09:30:00 | 1605.00 | 1597.19 | 0.00 | ORB-long ORB[1588.00,1600.20] vol=2.2x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 09:50:00 | 1612.86 | 1601.09 | 0.00 | T1 1.5R @ 1612.86 |
| Stop hit — per-position SL triggered | 2023-09-29 10:30:00 | 1605.00 | 1604.08 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 11:15:00 | 1615.20 | 1621.65 | 0.00 | ORB-short ORB[1618.20,1630.45] vol=2.2x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-10-06 11:30:00 | 1618.31 | 1621.25 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 11:10:00 | 1621.35 | 1616.10 | 0.00 | ORB-long ORB[1597.10,1614.90] vol=2.3x ATR=3.23 |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 1618.12 | 1617.60 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 10:05:00 | 1630.60 | 1626.82 | 0.00 | ORB-long ORB[1620.35,1629.75] vol=2.6x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 10:30:00 | 1634.67 | 1628.77 | 0.00 | T1 1.5R @ 1634.67 |
| Target hit | 2023-10-10 15:20:00 | 1642.10 | 1639.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2023-10-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:30:00 | 1658.65 | 1652.37 | 0.00 | ORB-long ORB[1645.65,1654.95] vol=1.5x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:40:00 | 1662.76 | 1653.87 | 0.00 | T1 1.5R @ 1662.76 |
| Stop hit — per-position SL triggered | 2023-10-11 12:35:00 | 1658.65 | 1659.48 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:55:00 | 1620.90 | 1628.47 | 0.00 | ORB-short ORB[1624.55,1633.05] vol=5.4x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:20:00 | 1615.36 | 1626.96 | 0.00 | T1 1.5R @ 1615.36 |
| Stop hit — per-position SL triggered | 2023-10-13 12:20:00 | 1620.90 | 1623.54 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 11:10:00 | 1634.25 | 1626.83 | 0.00 | ORB-long ORB[1615.00,1631.95] vol=2.0x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 13:35:00 | 1639.02 | 1629.93 | 0.00 | T1 1.5R @ 1639.02 |
| Stop hit — per-position SL triggered | 2023-10-16 14:20:00 | 1634.25 | 1632.73 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 09:55:00 | 1620.10 | 1626.80 | 0.00 | ORB-short ORB[1626.10,1633.55] vol=3.2x ATR=3.35 |
| Stop hit — per-position SL triggered | 2023-10-18 10:10:00 | 1623.45 | 1624.55 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 11:00:00 | 1625.85 | 1622.08 | 0.00 | ORB-long ORB[1616.50,1625.45] vol=2.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2023-10-19 11:10:00 | 1623.45 | 1622.15 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 09:40:00 | 1614.65 | 1618.14 | 0.00 | ORB-short ORB[1614.80,1624.00] vol=1.5x ATR=2.67 |
| Stop hit — per-position SL triggered | 2023-10-20 09:45:00 | 1617.32 | 1618.00 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 11:10:00 | 1594.55 | 1602.54 | 0.00 | ORB-short ORB[1606.10,1615.00] vol=2.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 11:40:00 | 1589.86 | 1600.87 | 0.00 | T1 1.5R @ 1589.86 |
| Stop hit — per-position SL triggered | 2023-10-23 11:50:00 | 1594.55 | 1599.43 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-10-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:45:00 | 1564.25 | 1566.85 | 0.00 | ORB-short ORB[1570.00,1587.10] vol=9.5x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 11:20:00 | 1558.27 | 1565.06 | 0.00 | T1 1.5R @ 1558.27 |
| Stop hit — per-position SL triggered | 2023-10-26 11:45:00 | 1564.25 | 1564.71 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-10-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 10:25:00 | 1562.00 | 1568.83 | 0.00 | ORB-short ORB[1566.70,1586.55] vol=2.1x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-10-31 11:00:00 | 1565.68 | 1568.16 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:05:00 | 1593.95 | 1589.30 | 0.00 | ORB-long ORB[1582.20,1592.85] vol=1.9x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:25:00 | 1599.49 | 1591.98 | 0.00 | T1 1.5R @ 1599.49 |
| Stop hit — per-position SL triggered | 2023-11-16 11:40:00 | 1593.95 | 1592.04 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 10:50:00 | 1592.85 | 1599.45 | 0.00 | ORB-short ORB[1593.20,1604.45] vol=2.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:15:00 | 1588.28 | 1597.85 | 0.00 | T1 1.5R @ 1588.28 |
| Target hit | 2023-11-20 15:20:00 | 1582.30 | 1588.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2023-11-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:10:00 | 1597.15 | 1604.25 | 0.00 | ORB-short ORB[1605.00,1611.35] vol=3.6x ATR=3.50 |
| Stop hit — per-position SL triggered | 2023-11-24 10:15:00 | 1600.65 | 1603.92 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 10:15:00 | 1622.30 | 1615.02 | 0.00 | ORB-long ORB[1600.15,1619.00] vol=1.8x ATR=4.97 |
| Stop hit — per-position SL triggered | 2023-11-28 12:40:00 | 1617.33 | 1620.46 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:30:00 | 1632.00 | 1627.63 | 0.00 | ORB-long ORB[1619.55,1631.10] vol=2.7x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:20:00 | 1637.45 | 1634.50 | 0.00 | T1 1.5R @ 1637.45 |
| Stop hit — per-position SL triggered | 2023-11-29 10:30:00 | 1632.00 | 1634.61 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 09:55:00 | 1618.70 | 1624.87 | 0.00 | ORB-short ORB[1623.00,1641.00] vol=3.5x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 11:05:00 | 1612.63 | 1621.25 | 0.00 | T1 1.5R @ 1612.63 |
| Stop hit — per-position SL triggered | 2023-12-04 13:55:00 | 1618.70 | 1614.96 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1638.00 | 1655.84 | 0.00 | ORB-short ORB[1657.80,1672.00] vol=1.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:35:00 | 1632.50 | 1652.40 | 0.00 | T1 1.5R @ 1632.50 |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 1638.00 | 1647.78 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:30:00 | 1681.30 | 1675.57 | 0.00 | ORB-long ORB[1660.00,1681.00] vol=1.6x ATR=5.86 |
| Stop hit — per-position SL triggered | 2023-12-11 09:40:00 | 1675.44 | 1676.22 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 11:05:00 | 1683.00 | 1677.19 | 0.00 | ORB-long ORB[1668.90,1682.00] vol=2.7x ATR=3.62 |
| Stop hit — per-position SL triggered | 2023-12-12 11:40:00 | 1679.38 | 1678.78 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 11:00:00 | 1645.55 | 1642.59 | 0.00 | ORB-long ORB[1636.00,1645.00] vol=3.2x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-12-14 11:10:00 | 1643.26 | 1642.60 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:40:00 | 1675.55 | 1669.20 | 0.00 | ORB-long ORB[1656.65,1666.00] vol=1.5x ATR=4.92 |
| Stop hit — per-position SL triggered | 2023-12-15 10:00:00 | 1670.63 | 1670.55 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:45:00 | 1688.00 | 1681.28 | 0.00 | ORB-long ORB[1672.30,1682.65] vol=3.0x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 09:50:00 | 1694.59 | 1685.28 | 0.00 | T1 1.5R @ 1694.59 |
| Target hit | 2023-12-20 12:25:00 | 1705.80 | 1707.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — SELL (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:15:00 | 1619.25 | 1625.23 | 0.00 | ORB-short ORB[1620.30,1630.00] vol=2.3x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:30:00 | 1612.70 | 1623.27 | 0.00 | T1 1.5R @ 1612.70 |
| Stop hit — per-position SL triggered | 2023-12-22 10:50:00 | 1619.25 | 1619.38 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:45:00 | 1609.75 | 1614.31 | 0.00 | ORB-short ORB[1612.10,1621.95] vol=1.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2023-12-26 10:50:00 | 1613.02 | 1614.08 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 1632.90 | 1636.91 | 0.00 | ORB-short ORB[1634.20,1645.15] vol=3.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2023-12-28 10:45:00 | 1636.78 | 1633.58 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:40:00 | 1652.00 | 1641.65 | 0.00 | ORB-long ORB[1628.70,1642.65] vol=3.0x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 12:15:00 | 1659.24 | 1648.14 | 0.00 | T1 1.5R @ 1659.24 |
| Stop hit — per-position SL triggered | 2023-12-29 13:30:00 | 1652.00 | 1650.00 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 09:35:00 | 1650.15 | 1652.53 | 0.00 | ORB-short ORB[1652.05,1658.65] vol=1.9x ATR=3.07 |
| Stop hit — per-position SL triggered | 2024-01-01 09:45:00 | 1653.22 | 1652.41 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:00:00 | 1616.00 | 1628.54 | 0.00 | ORB-short ORB[1630.15,1637.00] vol=1.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:30:00 | 1609.96 | 1623.05 | 0.00 | T1 1.5R @ 1609.96 |
| Target hit | 2024-01-02 15:20:00 | 1597.55 | 1607.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2024-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:35:00 | 1621.90 | 1615.91 | 0.00 | ORB-long ORB[1609.00,1616.00] vol=2.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-01-05 09:45:00 | 1619.12 | 1617.62 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:45:00 | 1447.00 | 1442.24 | 0.00 | ORB-long ORB[1430.00,1441.95] vol=1.5x ATR=7.11 |
| Stop hit — per-position SL triggered | 2024-02-06 12:50:00 | 1439.89 | 1443.54 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:50:00 | 1416.70 | 1425.96 | 0.00 | ORB-short ORB[1423.70,1434.55] vol=1.7x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:10:00 | 1411.76 | 1424.32 | 0.00 | T1 1.5R @ 1411.76 |
| Stop hit — per-position SL triggered | 2024-02-08 12:00:00 | 1416.70 | 1421.57 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-02-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:40:00 | 1406.90 | 1414.93 | 0.00 | ORB-short ORB[1416.30,1425.00] vol=1.7x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-02-12 11:20:00 | 1410.31 | 1412.13 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 09:45:00 | 1408.30 | 1404.35 | 0.00 | ORB-long ORB[1394.10,1408.00] vol=1.5x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-02-14 10:00:00 | 1404.93 | 1404.67 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 10:20:00 | 1450.95 | 1444.44 | 0.00 | ORB-long ORB[1437.15,1446.60] vol=1.5x ATR=3.77 |
| Stop hit — per-position SL triggered | 2024-02-19 10:25:00 | 1447.18 | 1444.99 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:15:00 | 1421.00 | 1427.65 | 0.00 | ORB-short ORB[1426.15,1438.45] vol=1.8x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 10:55:00 | 1415.06 | 1425.26 | 0.00 | T1 1.5R @ 1415.06 |
| Stop hit — per-position SL triggered | 2024-02-22 11:20:00 | 1421.00 | 1421.74 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:30:00 | 1432.30 | 1428.38 | 0.00 | ORB-long ORB[1419.90,1431.50] vol=2.9x ATR=3.46 |
| Stop hit — per-position SL triggered | 2024-02-26 09:35:00 | 1428.84 | 1428.59 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:40:00 | 1439.50 | 1435.62 | 0.00 | ORB-long ORB[1423.00,1434.95] vol=2.4x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 09:45:00 | 1444.52 | 1438.01 | 0.00 | T1 1.5R @ 1444.52 |
| Stop hit — per-position SL triggered | 2024-02-27 10:00:00 | 1439.50 | 1438.86 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 11:00:00 | 1448.50 | 1441.31 | 0.00 | ORB-long ORB[1435.10,1444.90] vol=4.2x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:35:00 | 1453.43 | 1444.08 | 0.00 | T1 1.5R @ 1453.43 |
| Stop hit — per-position SL triggered | 2024-03-07 13:40:00 | 1448.50 | 1447.68 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-03-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 10:40:00 | 1456.85 | 1450.22 | 0.00 | ORB-long ORB[1437.30,1449.90] vol=5.0x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-03-12 10:45:00 | 1452.70 | 1450.79 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:00:00 | 1400.00 | 1412.10 | 0.00 | ORB-short ORB[1421.70,1432.60] vol=2.2x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-03-13 10:10:00 | 1404.59 | 1410.36 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:05:00 | 1400.50 | 1409.30 | 0.00 | ORB-short ORB[1400.65,1418.85] vol=2.0x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-03-15 11:45:00 | 1404.46 | 1408.25 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 10:30:00 | 1382.20 | 1389.79 | 0.00 | ORB-short ORB[1388.85,1398.45] vol=1.6x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 10:55:00 | 1376.78 | 1385.02 | 0.00 | T1 1.5R @ 1376.78 |
| Stop hit — per-position SL triggered | 2024-03-18 11:55:00 | 1382.20 | 1381.36 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:35:00 | 1351.70 | 1359.98 | 0.00 | ORB-short ORB[1370.05,1380.00] vol=2.3x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-03-20 10:40:00 | 1355.34 | 1358.82 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-03-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:05:00 | 1384.70 | 1380.76 | 0.00 | ORB-long ORB[1374.75,1383.70] vol=2.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-03-21 10:30:00 | 1381.71 | 1381.16 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:50:00 | 1369.30 | 1374.23 | 0.00 | ORB-short ORB[1372.25,1381.90] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-03-26 10:55:00 | 1372.18 | 1374.12 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:30:00 | 1362.65 | 1369.85 | 0.00 | ORB-short ORB[1369.50,1377.95] vol=1.6x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-03-27 10:40:00 | 1365.84 | 1369.50 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-04-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:10:00 | 1369.65 | 1374.40 | 0.00 | ORB-short ORB[1373.65,1381.95] vol=1.8x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-04-04 10:15:00 | 1372.27 | 1374.30 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:30:00 | 1355.85 | 1360.69 | 0.00 | ORB-short ORB[1356.55,1367.00] vol=2.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-04-05 09:55:00 | 1359.02 | 1359.53 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-04-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:50:00 | 1353.70 | 1357.62 | 0.00 | ORB-short ORB[1358.30,1367.00] vol=1.8x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-04-08 11:40:00 | 1356.04 | 1356.91 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:20:00 | 1358.15 | 1352.31 | 0.00 | ORB-long ORB[1345.05,1354.60] vol=6.7x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:25:00 | 1362.49 | 1353.94 | 0.00 | T1 1.5R @ 1362.49 |
| Stop hit — per-position SL triggered | 2024-04-09 10:35:00 | 1358.15 | 1354.54 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:50:00 | 1355.50 | 1348.82 | 0.00 | ORB-long ORB[1342.00,1351.95] vol=1.6x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-04-10 09:55:00 | 1352.21 | 1349.14 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-04-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:55:00 | 1400.95 | 1391.65 | 0.00 | ORB-long ORB[1378.50,1399.00] vol=1.8x ATR=5.14 |
| Stop hit — per-position SL triggered | 2024-04-12 11:15:00 | 1395.81 | 1392.68 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 10:15:00 | 1362.70 | 1353.91 | 0.00 | ORB-long ORB[1346.00,1359.90] vol=1.7x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-04-18 13:25:00 | 1358.88 | 1360.30 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 11:15:00 | 1364.75 | 1360.16 | 0.00 | ORB-long ORB[1356.10,1362.95] vol=4.4x ATR=3.07 |
| Stop hit — per-position SL triggered | 2024-04-24 11:45:00 | 1361.68 | 1360.63 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 11:00:00 | 1348.90 | 1357.39 | 0.00 | ORB-short ORB[1356.85,1366.00] vol=1.6x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-04-25 11:05:00 | 1351.17 | 1356.70 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-05-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:40:00 | 1341.40 | 1350.53 | 0.00 | ORB-short ORB[1354.10,1361.40] vol=1.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-05-03 10:55:00 | 1343.98 | 1349.81 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:40:00 | 1332.90 | 1338.00 | 0.00 | ORB-short ORB[1335.55,1345.00] vol=2.1x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 10:15:00 | 1328.21 | 1335.03 | 0.00 | T1 1.5R @ 1328.21 |
| Target hit | 2024-05-06 15:20:00 | 1330.30 | 1330.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 103 — BUY (started 2024-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 10:25:00 | 1339.00 | 1329.87 | 0.00 | ORB-long ORB[1325.50,1333.85] vol=3.1x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-05-07 10:40:00 | 1335.51 | 1331.47 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-05-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:45:00 | 1313.25 | 1319.34 | 0.00 | ORB-short ORB[1316.80,1328.00] vol=1.6x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-05-09 11:45:00 | 1316.18 | 1317.93 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-05-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 11:05:00 | 1311.50 | 1306.78 | 0.00 | ORB-long ORB[1298.05,1308.50] vol=1.8x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-05-10 12:10:00 | 1308.28 | 1307.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-12 11:15:00 | 1528.00 | 2023-05-12 11:40:00 | 1531.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-15 10:10:00 | 1531.55 | 2023-05-15 10:15:00 | 1536.70 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-05-15 10:10:00 | 1531.55 | 2023-05-15 11:20:00 | 1531.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-17 10:55:00 | 1535.25 | 2023-05-17 11:00:00 | 1538.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-05-18 10:55:00 | 1533.00 | 2023-05-18 11:25:00 | 1529.10 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-05-18 10:55:00 | 1533.00 | 2023-05-18 13:55:00 | 1531.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2023-05-23 09:50:00 | 1514.80 | 2023-05-23 10:20:00 | 1519.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-05-25 09:45:00 | 1553.10 | 2023-05-25 09:55:00 | 1549.15 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-26 11:00:00 | 1574.25 | 2023-05-26 11:40:00 | 1571.03 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-02 10:55:00 | 1568.65 | 2023-06-02 11:10:00 | 1571.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-06 10:30:00 | 1567.35 | 2023-06-06 10:55:00 | 1563.04 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-06-06 10:30:00 | 1567.35 | 2023-06-06 11:30:00 | 1567.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-12 10:55:00 | 1576.95 | 2023-06-12 13:40:00 | 1574.08 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-14 11:00:00 | 1614.20 | 2023-06-14 11:05:00 | 1609.99 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-15 10:00:00 | 1613.00 | 2023-06-15 10:10:00 | 1609.27 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-16 09:50:00 | 1631.80 | 2023-06-16 09:55:00 | 1628.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-21 09:40:00 | 1649.25 | 2023-06-21 10:05:00 | 1646.52 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-26 09:35:00 | 1630.05 | 2023-06-26 10:15:00 | 1635.13 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-26 09:35:00 | 1630.05 | 2023-06-26 10:30:00 | 1630.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-27 09:45:00 | 1659.60 | 2023-06-27 10:20:00 | 1665.05 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-06-27 09:45:00 | 1659.60 | 2023-06-27 12:50:00 | 1663.75 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-03 09:50:00 | 1665.75 | 2023-07-03 10:00:00 | 1660.56 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-03 09:50:00 | 1665.75 | 2023-07-03 14:25:00 | 1647.00 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2023-07-07 09:30:00 | 1720.90 | 2023-07-07 09:40:00 | 1714.78 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-11 10:35:00 | 1685.90 | 2023-07-11 11:45:00 | 1690.34 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-07-11 10:35:00 | 1685.90 | 2023-07-11 15:20:00 | 1691.10 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-18 10:40:00 | 1682.00 | 2023-07-18 10:50:00 | 1684.59 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-07-19 09:45:00 | 1700.55 | 2023-07-19 10:15:00 | 1707.36 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-19 09:45:00 | 1700.55 | 2023-07-19 10:25:00 | 1702.15 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2023-07-20 09:45:00 | 1687.60 | 2023-07-20 10:10:00 | 1683.40 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-20 09:45:00 | 1687.60 | 2023-07-20 11:00:00 | 1687.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-26 10:50:00 | 1709.20 | 2023-07-26 10:55:00 | 1716.28 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-26 10:50:00 | 1709.20 | 2023-07-26 11:00:00 | 1709.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-27 10:30:00 | 1698.95 | 2023-07-27 11:25:00 | 1693.96 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-07-27 10:30:00 | 1698.95 | 2023-07-27 11:55:00 | 1698.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-31 09:30:00 | 1709.70 | 2023-07-31 09:40:00 | 1713.76 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-01 10:00:00 | 1748.40 | 2023-08-01 10:20:00 | 1743.94 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-04 09:45:00 | 1740.00 | 2023-08-04 10:10:00 | 1744.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-08-07 10:40:00 | 1747.25 | 2023-08-07 10:45:00 | 1751.32 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-11 10:30:00 | 1664.00 | 2023-08-11 11:00:00 | 1657.82 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-08-17 10:05:00 | 1650.00 | 2023-08-17 10:10:00 | 1654.99 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-08-17 10:05:00 | 1650.00 | 2023-08-17 10:20:00 | 1650.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-29 09:30:00 | 1706.00 | 2023-08-29 10:50:00 | 1711.58 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-08-29 09:30:00 | 1706.00 | 2023-08-29 10:55:00 | 1706.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 09:40:00 | 1705.00 | 2023-08-30 10:30:00 | 1710.48 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-08-30 09:40:00 | 1705.00 | 2023-08-30 13:35:00 | 1711.50 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2023-09-04 10:50:00 | 1673.35 | 2023-09-04 11:20:00 | 1668.84 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-09-04 10:50:00 | 1673.35 | 2023-09-04 15:20:00 | 1655.00 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2023-09-06 09:40:00 | 1692.95 | 2023-09-06 09:50:00 | 1689.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-07 10:05:00 | 1704.05 | 2023-09-07 10:10:00 | 1710.79 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-09-07 10:05:00 | 1704.05 | 2023-09-07 11:25:00 | 1723.15 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2023-09-12 10:30:00 | 1681.55 | 2023-09-12 10:40:00 | 1686.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-14 10:40:00 | 1668.90 | 2023-09-14 11:35:00 | 1672.91 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-15 10:10:00 | 1682.80 | 2023-09-15 10:20:00 | 1678.32 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-21 11:10:00 | 1673.50 | 2023-09-21 11:15:00 | 1676.77 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-09-21 11:10:00 | 1673.50 | 2023-09-21 11:20:00 | 1673.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-22 09:55:00 | 1641.60 | 2023-09-22 10:05:00 | 1645.94 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-25 10:40:00 | 1641.70 | 2023-09-25 11:05:00 | 1637.94 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-27 09:30:00 | 1598.85 | 2023-09-27 09:35:00 | 1602.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-09-29 09:30:00 | 1605.00 | 2023-09-29 09:50:00 | 1612.86 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-09-29 09:30:00 | 1605.00 | 2023-09-29 10:30:00 | 1605.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-06 11:15:00 | 1615.20 | 2023-10-06 11:30:00 | 1618.31 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-09 11:10:00 | 1621.35 | 2023-10-09 13:15:00 | 1618.12 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-10 10:05:00 | 1630.60 | 2023-10-10 10:30:00 | 1634.67 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-10-10 10:05:00 | 1630.60 | 2023-10-10 15:20:00 | 1642.10 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2023-10-11 10:30:00 | 1658.65 | 2023-10-11 10:40:00 | 1662.76 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-10-11 10:30:00 | 1658.65 | 2023-10-11 12:35:00 | 1658.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-13 10:55:00 | 1620.90 | 2023-10-13 11:20:00 | 1615.36 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-13 10:55:00 | 1620.90 | 2023-10-13 12:20:00 | 1620.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 11:10:00 | 1634.25 | 2023-10-16 13:35:00 | 1639.02 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-10-16 11:10:00 | 1634.25 | 2023-10-16 14:20:00 | 1634.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 09:55:00 | 1620.10 | 2023-10-18 10:10:00 | 1623.45 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-10-19 11:00:00 | 1625.85 | 2023-10-19 11:10:00 | 1623.45 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-10-20 09:40:00 | 1614.65 | 2023-10-20 09:45:00 | 1617.32 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-23 11:10:00 | 1594.55 | 2023-10-23 11:40:00 | 1589.86 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-10-23 11:10:00 | 1594.55 | 2023-10-23 11:50:00 | 1594.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-26 10:45:00 | 1564.25 | 2023-10-26 11:20:00 | 1558.27 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-10-26 10:45:00 | 1564.25 | 2023-10-26 11:45:00 | 1564.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-31 10:25:00 | 1562.00 | 2023-10-31 11:00:00 | 1565.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-16 10:05:00 | 1593.95 | 2023-11-16 11:25:00 | 1599.49 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-16 10:05:00 | 1593.95 | 2023-11-16 11:40:00 | 1593.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 10:50:00 | 1592.85 | 2023-11-20 11:15:00 | 1588.28 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-11-20 10:50:00 | 1592.85 | 2023-11-20 15:20:00 | 1582.30 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2023-11-24 10:10:00 | 1597.15 | 2023-11-24 10:15:00 | 1600.65 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-28 10:15:00 | 1622.30 | 2023-11-28 12:40:00 | 1617.33 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-29 09:30:00 | 1632.00 | 2023-11-29 10:20:00 | 1637.45 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-29 09:30:00 | 1632.00 | 2023-11-29 10:30:00 | 1632.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-04 09:55:00 | 1618.70 | 2023-12-04 11:05:00 | 1612.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-04 09:55:00 | 1618.70 | 2023-12-04 13:55:00 | 1618.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1638.00 | 2023-12-08 11:35:00 | 1632.50 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1638.00 | 2023-12-08 12:15:00 | 1638.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-11 09:30:00 | 1681.30 | 2023-12-11 09:40:00 | 1675.44 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-12 11:05:00 | 1683.00 | 2023-12-12 11:40:00 | 1679.38 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-14 11:00:00 | 1645.55 | 2023-12-14 11:10:00 | 1643.26 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-12-15 09:40:00 | 1675.55 | 2023-12-15 10:00:00 | 1670.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-20 09:45:00 | 1688.00 | 2023-12-20 09:50:00 | 1694.59 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-12-20 09:45:00 | 1688.00 | 2023-12-20 12:25:00 | 1705.80 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2023-12-22 10:15:00 | 1619.25 | 2023-12-22 10:30:00 | 1612.70 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-22 10:15:00 | 1619.25 | 2023-12-22 10:50:00 | 1619.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-26 10:45:00 | 1609.75 | 2023-12-26 10:50:00 | 1613.02 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-28 09:30:00 | 1632.90 | 2023-12-28 10:45:00 | 1636.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-29 10:40:00 | 1652.00 | 2023-12-29 12:15:00 | 1659.24 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-12-29 10:40:00 | 1652.00 | 2023-12-29 13:30:00 | 1652.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-01 09:35:00 | 1650.15 | 2024-01-01 09:45:00 | 1653.22 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-01-02 10:00:00 | 1616.00 | 2024-01-02 10:30:00 | 1609.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-01-02 10:00:00 | 1616.00 | 2024-01-02 15:20:00 | 1597.55 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2024-01-05 09:35:00 | 1621.90 | 2024-01-05 09:45:00 | 1619.12 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-02-06 10:45:00 | 1447.00 | 2024-02-06 12:50:00 | 1439.89 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-02-08 10:50:00 | 1416.70 | 2024-02-08 11:10:00 | 1411.76 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-08 10:50:00 | 1416.70 | 2024-02-08 12:00:00 | 1416.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-12 10:40:00 | 1406.90 | 2024-02-12 11:20:00 | 1410.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-14 09:45:00 | 1408.30 | 2024-02-14 10:00:00 | 1404.93 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-19 10:20:00 | 1450.95 | 2024-02-19 10:25:00 | 1447.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-22 10:15:00 | 1421.00 | 2024-02-22 10:55:00 | 1415.06 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-22 10:15:00 | 1421.00 | 2024-02-22 11:20:00 | 1421.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-26 09:30:00 | 1432.30 | 2024-02-26 09:35:00 | 1428.84 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-27 09:40:00 | 1439.50 | 2024-02-27 09:45:00 | 1444.52 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-02-27 09:40:00 | 1439.50 | 2024-02-27 10:00:00 | 1439.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-07 11:00:00 | 1448.50 | 2024-03-07 11:35:00 | 1453.43 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-03-07 11:00:00 | 1448.50 | 2024-03-07 13:40:00 | 1448.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 10:40:00 | 1456.85 | 2024-03-12 10:45:00 | 1452.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-13 10:00:00 | 1400.00 | 2024-03-13 10:10:00 | 1404.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-15 11:05:00 | 1400.50 | 2024-03-15 11:45:00 | 1404.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-18 10:30:00 | 1382.20 | 2024-03-18 10:55:00 | 1376.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-18 10:30:00 | 1382.20 | 2024-03-18 11:55:00 | 1382.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 10:35:00 | 1351.70 | 2024-03-20 10:40:00 | 1355.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-03-21 10:05:00 | 1384.70 | 2024-03-21 10:30:00 | 1381.71 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-03-26 10:50:00 | 1369.30 | 2024-03-26 10:55:00 | 1372.18 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-03-27 10:30:00 | 1362.65 | 2024-03-27 10:40:00 | 1365.84 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-04 10:10:00 | 1369.65 | 2024-04-04 10:15:00 | 1372.27 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-04-05 09:30:00 | 1355.85 | 2024-04-05 09:55:00 | 1359.02 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-08 10:50:00 | 1353.70 | 2024-04-08 11:40:00 | 1356.04 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-04-09 10:20:00 | 1358.15 | 2024-04-09 10:25:00 | 1362.49 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-04-09 10:20:00 | 1358.15 | 2024-04-09 10:35:00 | 1358.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-10 09:50:00 | 1355.50 | 2024-04-10 09:55:00 | 1352.21 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-12 10:55:00 | 1400.95 | 2024-04-12 11:15:00 | 1395.81 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-18 10:15:00 | 1362.70 | 2024-04-18 13:25:00 | 1358.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-24 11:15:00 | 1364.75 | 2024-04-24 11:45:00 | 1361.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-04-25 11:00:00 | 1348.90 | 2024-04-25 11:05:00 | 1351.17 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-05-03 10:40:00 | 1341.40 | 2024-05-03 10:55:00 | 1343.98 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-06 09:40:00 | 1332.90 | 2024-05-06 10:15:00 | 1328.21 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-06 09:40:00 | 1332.90 | 2024-05-06 15:20:00 | 1330.30 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-05-07 10:25:00 | 1339.00 | 2024-05-07 10:40:00 | 1335.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-09 10:45:00 | 1313.25 | 2024-05-09 11:45:00 | 1316.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-10 11:05:00 | 1311.50 | 2024-05-10 12:10:00 | 1308.28 | STOP_HIT | 1.00 | -0.25% |
