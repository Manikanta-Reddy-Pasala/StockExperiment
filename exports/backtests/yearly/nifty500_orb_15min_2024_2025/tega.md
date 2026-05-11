# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1659.00
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
| ENTRY1 | 59 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 12 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 80 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 47
- **Target hits / Stop hits / Partials:** 12 / 47 / 21
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 12.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 16 | 38.1% | 6 | 26 | 10 | 0.08% | 3.2% |
| BUY @ 2nd Alert (retest1) | 42 | 16 | 38.1% | 6 | 26 | 10 | 0.08% | 3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 17 | 44.7% | 6 | 21 | 11 | 0.24% | 9.2% |
| SELL @ 2nd Alert (retest1) | 38 | 17 | 44.7% | 6 | 21 | 11 | 0.24% | 9.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 80 | 33 | 41.2% | 12 | 47 | 21 | 0.15% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 10:35:00 | 1576.35 | 1567.69 | 0.00 | ORB-long ORB[1535.30,1558.50] vol=14.3x ATR=7.96 |
| Stop hit — per-position SL triggered | 2024-05-15 10:40:00 | 1568.39 | 1567.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1541.75 | 1550.01 | 0.00 | ORB-short ORB[1547.10,1566.60] vol=1.5x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:45:00 | 1532.70 | 1547.85 | 0.00 | T1 1.5R @ 1532.70 |
| Stop hit — per-position SL triggered | 2024-05-16 14:30:00 | 1541.75 | 1542.45 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:50:00 | 1586.30 | 1567.59 | 0.00 | ORB-long ORB[1542.15,1564.80] vol=9.7x ATR=11.95 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 1574.35 | 1567.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:55:00 | 1480.05 | 1473.35 | 0.00 | ORB-long ORB[1457.65,1477.60] vol=6.8x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:10:00 | 1488.09 | 1474.16 | 0.00 | T1 1.5R @ 1488.09 |
| Stop hit — per-position SL triggered | 2024-05-28 12:00:00 | 1480.05 | 1476.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 11:05:00 | 1487.90 | 1484.03 | 0.00 | ORB-long ORB[1471.90,1487.10] vol=2.0x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:40:00 | 1495.48 | 1486.80 | 0.00 | T1 1.5R @ 1495.48 |
| Stop hit — per-position SL triggered | 2024-05-29 13:10:00 | 1487.90 | 1487.37 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:15:00 | 1679.65 | 1688.99 | 0.00 | ORB-short ORB[1683.20,1699.00] vol=2.2x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-06-12 12:00:00 | 1684.33 | 1688.65 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 1679.00 | 1665.63 | 0.00 | ORB-long ORB[1650.70,1673.40] vol=2.3x ATR=6.68 |
| Target hit | 2024-06-18 15:20:00 | 1680.45 | 1675.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:20:00 | 1630.50 | 1618.10 | 0.00 | ORB-long ORB[1596.95,1619.65] vol=2.5x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-06-26 13:10:00 | 1625.01 | 1626.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:35:00 | 1633.00 | 1627.96 | 0.00 | ORB-long ORB[1617.35,1631.20] vol=2.4x ATR=6.77 |
| Stop hit — per-position SL triggered | 2024-06-27 10:10:00 | 1626.23 | 1629.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:55:00 | 1639.85 | 1637.08 | 0.00 | ORB-long ORB[1610.00,1630.95] vol=14.4x ATR=6.07 |
| Stop hit — per-position SL triggered | 2024-07-03 10:40:00 | 1633.78 | 1635.76 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:40:00 | 1672.30 | 1663.96 | 0.00 | ORB-long ORB[1653.00,1663.90] vol=2.9x ATR=8.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:00:00 | 1684.70 | 1675.73 | 0.00 | T1 1.5R @ 1684.70 |
| Target hit | 2024-07-04 12:15:00 | 1702.35 | 1703.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:55:00 | 1812.35 | 1820.81 | 0.00 | ORB-short ORB[1818.00,1833.25] vol=1.6x ATR=5.83 |
| Stop hit — per-position SL triggered | 2024-07-11 11:00:00 | 1818.18 | 1820.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:30:00 | 1708.85 | 1696.72 | 0.00 | ORB-long ORB[1675.05,1700.00] vol=7.3x ATR=6.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:35:00 | 1718.47 | 1703.19 | 0.00 | T1 1.5R @ 1718.47 |
| Target hit | 2024-08-19 10:55:00 | 1709.50 | 1712.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:50:00 | 1721.00 | 1690.11 | 0.00 | ORB-long ORB[1685.80,1705.00] vol=1.5x ATR=7.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:45:00 | 1732.69 | 1694.06 | 0.00 | T1 1.5R @ 1732.69 |
| Stop hit — per-position SL triggered | 2024-08-20 12:15:00 | 1721.00 | 1695.29 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:30:00 | 1754.20 | 1734.67 | 0.00 | ORB-long ORB[1715.45,1740.00] vol=3.1x ATR=6.60 |
| Stop hit — per-position SL triggered | 2024-08-21 10:45:00 | 1747.60 | 1735.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:55:00 | 1737.45 | 1720.32 | 0.00 | ORB-long ORB[1705.55,1723.00] vol=4.8x ATR=10.08 |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 1727.37 | 1721.83 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:55:00 | 1706.30 | 1717.93 | 0.00 | ORB-short ORB[1715.05,1735.20] vol=1.5x ATR=6.67 |
| Stop hit — per-position SL triggered | 2024-08-26 10:25:00 | 1712.97 | 1713.38 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:15:00 | 1680.25 | 1692.68 | 0.00 | ORB-short ORB[1680.45,1705.00] vol=2.3x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-08-27 10:20:00 | 1687.37 | 1692.43 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:20:00 | 1689.05 | 1695.91 | 0.00 | ORB-short ORB[1692.60,1709.95] vol=3.4x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 12:10:00 | 1678.63 | 1690.47 | 0.00 | T1 1.5R @ 1678.63 |
| Target hit | 2024-08-28 15:20:00 | 1672.85 | 1683.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-09-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:55:00 | 1732.50 | 1712.25 | 0.00 | ORB-long ORB[1705.05,1723.00] vol=11.5x ATR=7.15 |
| Stop hit — per-position SL triggered | 2024-09-02 11:00:00 | 1725.35 | 1714.74 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 1687.45 | 1703.74 | 0.00 | ORB-short ORB[1707.90,1722.00] vol=3.7x ATR=6.97 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 1694.42 | 1702.60 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1779.20 | 1783.04 | 0.00 | ORB-short ORB[1780.00,1801.85] vol=1.5x ATR=10.56 |
| Stop hit — per-position SL triggered | 2024-09-11 11:10:00 | 1789.76 | 1777.88 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 1802.05 | 1787.48 | 0.00 | ORB-long ORB[1772.50,1798.80] vol=3.0x ATR=7.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:05:00 | 1813.49 | 1794.37 | 0.00 | T1 1.5R @ 1813.49 |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 1802.05 | 1796.75 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 1851.45 | 1846.17 | 0.00 | ORB-long ORB[1832.95,1849.95] vol=3.8x ATR=7.58 |
| Stop hit — per-position SL triggered | 2024-09-19 09:45:00 | 1843.87 | 1846.53 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:30:00 | 1879.55 | 1875.83 | 0.00 | ORB-long ORB[1856.00,1878.00] vol=4.1x ATR=9.92 |
| Stop hit — per-position SL triggered | 2024-09-23 09:35:00 | 1869.63 | 1875.43 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 11:05:00 | 1835.80 | 1826.44 | 0.00 | ORB-long ORB[1817.10,1832.40] vol=5.9x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-09-25 11:10:00 | 1829.65 | 1826.51 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:45:00 | 1805.15 | 1815.31 | 0.00 | ORB-short ORB[1807.30,1829.45] vol=1.8x ATR=5.89 |
| Stop hit — per-position SL triggered | 2024-09-26 10:25:00 | 1811.04 | 1812.69 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:00:00 | 1848.95 | 1832.31 | 0.00 | ORB-long ORB[1811.55,1835.00] vol=3.7x ATR=8.64 |
| Stop hit — per-position SL triggered | 2024-09-30 10:05:00 | 1840.31 | 1833.38 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 1889.00 | 1913.46 | 0.00 | ORB-short ORB[1920.00,1933.55] vol=2.2x ATR=10.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:55:00 | 1874.00 | 1901.74 | 0.00 | T1 1.5R @ 1874.00 |
| Stop hit — per-position SL triggered | 2024-10-07 10:10:00 | 1889.00 | 1897.53 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:10:00 | 1926.00 | 1933.27 | 0.00 | ORB-short ORB[1928.80,1954.00] vol=1.9x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 10:45:00 | 1916.53 | 1929.29 | 0.00 | T1 1.5R @ 1916.53 |
| Target hit | 2024-10-10 15:20:00 | 1910.35 | 1922.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2024-10-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:40:00 | 1897.05 | 1908.02 | 0.00 | ORB-short ORB[1901.80,1920.70] vol=1.8x ATR=5.10 |
| Stop hit — per-position SL triggered | 2024-10-11 13:30:00 | 1902.15 | 1901.22 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 1871.25 | 1883.70 | 0.00 | ORB-short ORB[1881.00,1900.20] vol=1.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2024-10-17 09:45:00 | 1877.58 | 1882.78 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 1836.30 | 1850.27 | 0.00 | ORB-short ORB[1839.25,1865.50] vol=3.3x ATR=7.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:35:00 | 1825.36 | 1841.00 | 0.00 | T1 1.5R @ 1825.36 |
| Stop hit — per-position SL triggered | 2024-10-21 09:40:00 | 1836.30 | 1840.76 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:15:00 | 1768.45 | 1786.49 | 0.00 | ORB-short ORB[1785.30,1811.70] vol=2.3x ATR=7.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:30:00 | 1757.25 | 1782.01 | 0.00 | T1 1.5R @ 1757.25 |
| Stop hit — per-position SL triggered | 2024-10-22 10:40:00 | 1768.45 | 1781.10 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 10:05:00 | 1859.25 | 1850.23 | 0.00 | ORB-long ORB[1842.05,1857.15] vol=1.5x ATR=8.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:05:00 | 1872.15 | 1859.51 | 0.00 | T1 1.5R @ 1872.15 |
| Target hit | 2024-10-29 11:15:00 | 1860.40 | 1860.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2024-10-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:50:00 | 1892.50 | 1875.40 | 0.00 | ORB-long ORB[1857.25,1883.90] vol=2.5x ATR=8.37 |
| Stop hit — per-position SL triggered | 2024-10-30 10:25:00 | 1884.13 | 1884.39 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:10:00 | 1701.40 | 1709.88 | 0.00 | ORB-short ORB[1706.40,1730.00] vol=2.5x ATR=6.33 |
| Stop hit — per-position SL triggered | 2024-12-03 10:45:00 | 1707.73 | 1707.97 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:40:00 | 1646.75 | 1653.80 | 0.00 | ORB-short ORB[1647.00,1669.80] vol=2.8x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:05:00 | 1635.45 | 1647.51 | 0.00 | T1 1.5R @ 1635.45 |
| Target hit | 2024-12-06 15:20:00 | 1604.35 | 1627.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:15:00 | 1585.55 | 1578.58 | 0.00 | ORB-long ORB[1557.20,1579.95] vol=8.2x ATR=6.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:20:00 | 1595.72 | 1579.40 | 0.00 | T1 1.5R @ 1595.72 |
| Target hit | 2024-12-11 15:20:00 | 1637.45 | 1609.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:30:00 | 1702.50 | 1695.47 | 0.00 | ORB-long ORB[1677.30,1694.70] vol=3.0x ATR=9.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:40:00 | 1716.05 | 1702.00 | 0.00 | T1 1.5R @ 1716.05 |
| Stop hit — per-position SL triggered | 2024-12-18 09:50:00 | 1702.50 | 1703.63 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 1592.25 | 1583.41 | 0.00 | ORB-long ORB[1569.15,1585.95] vol=1.9x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:40:00 | 1601.71 | 1587.78 | 0.00 | T1 1.5R @ 1601.71 |
| Target hit | 2024-12-24 10:30:00 | 1596.65 | 1597.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 1567.10 | 1582.66 | 0.00 | ORB-short ORB[1584.00,1597.50] vol=2.3x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:50:00 | 1560.38 | 1577.53 | 0.00 | T1 1.5R @ 1560.38 |
| Target hit | 2024-12-26 15:20:00 | 1540.00 | 1559.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1562.35 | 1555.10 | 0.00 | ORB-long ORB[1542.80,1558.40] vol=1.7x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-12-27 09:35:00 | 1557.55 | 1556.37 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 10:25:00 | 1539.65 | 1549.93 | 0.00 | ORB-short ORB[1539.95,1560.00] vol=1.5x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 11:00:00 | 1529.64 | 1545.93 | 0.00 | T1 1.5R @ 1529.64 |
| Target hit | 2024-12-31 13:20:00 | 1535.85 | 1535.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2025-01-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:25:00 | 1541.50 | 1550.27 | 0.00 | ORB-short ORB[1545.00,1559.90] vol=1.6x ATR=4.61 |
| Stop hit — per-position SL triggered | 2025-01-03 11:00:00 | 1546.11 | 1548.91 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1620.10 | 1639.94 | 0.00 | ORB-short ORB[1638.45,1654.35] vol=1.6x ATR=8.17 |
| Stop hit — per-position SL triggered | 2025-01-15 09:40:00 | 1628.27 | 1636.48 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:50:00 | 1650.00 | 1646.38 | 0.00 | ORB-long ORB[1626.00,1648.90] vol=4.1x ATR=4.65 |
| Stop hit — per-position SL triggered | 2025-01-16 10:55:00 | 1645.35 | 1646.38 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 1662.30 | 1658.22 | 0.00 | ORB-long ORB[1645.00,1660.00] vol=2.1x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-01-21 09:35:00 | 1658.11 | 1657.95 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:05:00 | 1618.90 | 1609.28 | 0.00 | ORB-long ORB[1595.05,1607.65] vol=1.8x ATR=5.44 |
| Stop hit — per-position SL triggered | 2025-01-23 12:10:00 | 1613.46 | 1611.64 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:15:00 | 1577.55 | 1585.83 | 0.00 | ORB-short ORB[1586.20,1597.55] vol=7.0x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:35:00 | 1572.13 | 1584.18 | 0.00 | T1 1.5R @ 1572.13 |
| Target hit | 2025-01-24 15:20:00 | 1534.25 | 1552.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-01-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:55:00 | 1422.95 | 1457.92 | 0.00 | ORB-short ORB[1469.95,1483.90] vol=3.4x ATR=10.16 |
| Stop hit — per-position SL triggered | 2025-01-28 10:00:00 | 1433.11 | 1456.03 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-29 10:55:00 | 1464.05 | 1471.64 | 0.00 | ORB-short ORB[1466.05,1484.50] vol=1.8x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 12:45:00 | 1457.82 | 1467.01 | 0.00 | T1 1.5R @ 1457.82 |
| Stop hit — per-position SL triggered | 2025-01-29 14:00:00 | 1464.05 | 1466.14 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 11:05:00 | 1496.95 | 1504.45 | 0.00 | ORB-short ORB[1501.95,1515.05] vol=5.3x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-01-31 11:20:00 | 1499.54 | 1503.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:45:00 | 1305.00 | 1297.61 | 0.00 | ORB-long ORB[1293.55,1303.20] vol=2.4x ATR=6.36 |
| Stop hit — per-position SL triggered | 2025-03-13 11:20:00 | 1298.64 | 1297.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:25:00 | 1315.00 | 1310.51 | 0.00 | ORB-long ORB[1291.05,1309.35] vol=5.3x ATR=6.94 |
| Stop hit — per-position SL triggered | 2025-03-18 10:30:00 | 1308.06 | 1310.56 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:05:00 | 1369.60 | 1362.50 | 0.00 | ORB-long ORB[1350.00,1366.95] vol=3.5x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 1364.79 | 1362.88 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:55:00 | 1465.40 | 1462.34 | 0.00 | ORB-long ORB[1451.80,1464.20] vol=3.3x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-04-17 10:05:00 | 1461.42 | 1462.33 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:10:00 | 1390.10 | 1410.34 | 0.00 | ORB-short ORB[1410.60,1425.50] vol=1.6x ATR=4.58 |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1394.68 | 1409.48 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 1393.00 | 1402.28 | 0.00 | ORB-short ORB[1397.80,1413.60] vol=3.1x ATR=5.48 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 1398.48 | 1401.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 10:35:00 | 1576.35 | 2024-05-15 10:40:00 | 1568.39 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-16 10:20:00 | 1541.75 | 2024-05-16 10:45:00 | 1532.70 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-05-16 10:20:00 | 1541.75 | 2024-05-16 14:30:00 | 1541.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-22 09:50:00 | 1586.30 | 2024-05-22 09:55:00 | 1574.35 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2024-05-28 10:55:00 | 1480.05 | 2024-05-28 11:10:00 | 1488.09 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-05-28 10:55:00 | 1480.05 | 2024-05-28 12:00:00 | 1480.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 11:05:00 | 1487.90 | 2024-05-29 11:40:00 | 1495.48 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-29 11:05:00 | 1487.90 | 2024-05-29 13:10:00 | 1487.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 11:15:00 | 1679.65 | 2024-06-12 12:00:00 | 1684.33 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-18 09:35:00 | 1679.00 | 2024-06-18 15:20:00 | 1680.45 | TARGET_HIT | 1.00 | 0.09% |
| BUY | retest1 | 2024-06-26 10:20:00 | 1630.50 | 2024-06-26 13:10:00 | 1625.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-27 09:35:00 | 1633.00 | 2024-06-27 10:10:00 | 1626.23 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-03 09:55:00 | 1639.85 | 2024-07-03 10:40:00 | 1633.78 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-04 09:40:00 | 1672.30 | 2024-07-04 11:00:00 | 1684.70 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-07-04 09:40:00 | 1672.30 | 2024-07-04 12:15:00 | 1702.35 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2024-07-11 10:55:00 | 1812.35 | 2024-07-11 11:00:00 | 1818.18 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-19 10:30:00 | 1708.85 | 2024-08-19 10:35:00 | 1718.47 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-19 10:30:00 | 1708.85 | 2024-08-19 10:55:00 | 1709.50 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-08-20 10:50:00 | 1721.00 | 2024-08-20 11:45:00 | 1732.69 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-08-20 10:50:00 | 1721.00 | 2024-08-20 12:15:00 | 1721.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:30:00 | 1754.20 | 2024-08-21 10:45:00 | 1747.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-23 09:55:00 | 1737.45 | 2024-08-23 10:15:00 | 1727.37 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-08-26 09:55:00 | 1706.30 | 2024-08-26 10:25:00 | 1712.97 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-27 10:15:00 | 1680.25 | 2024-08-27 10:20:00 | 1687.37 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-28 10:20:00 | 1689.05 | 2024-08-28 12:10:00 | 1678.63 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-08-28 10:20:00 | 1689.05 | 2024-08-28 15:20:00 | 1672.85 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2024-09-02 10:55:00 | 1732.50 | 2024-09-02 11:00:00 | 1725.35 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-09 09:30:00 | 1687.45 | 2024-09-09 09:35:00 | 1694.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-11 09:35:00 | 1779.20 | 2024-09-11 11:10:00 | 1789.76 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-09-13 09:35:00 | 1802.05 | 2024-09-13 10:05:00 | 1813.49 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-09-13 09:35:00 | 1802.05 | 2024-09-13 10:15:00 | 1802.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:30:00 | 1851.45 | 2024-09-19 09:45:00 | 1843.87 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-09-23 09:30:00 | 1879.55 | 2024-09-23 09:35:00 | 1869.63 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-09-25 11:05:00 | 1835.80 | 2024-09-25 11:10:00 | 1829.65 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-26 09:45:00 | 1805.15 | 2024-09-26 10:25:00 | 1811.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-30 10:00:00 | 1848.95 | 2024-09-30 10:05:00 | 1840.31 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-10-07 09:45:00 | 1889.00 | 2024-10-07 09:55:00 | 1874.00 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-10-07 09:45:00 | 1889.00 | 2024-10-07 10:10:00 | 1889.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 10:10:00 | 1926.00 | 2024-10-10 10:45:00 | 1916.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-10 10:10:00 | 1926.00 | 2024-10-10 15:20:00 | 1910.35 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2024-10-11 10:40:00 | 1897.05 | 2024-10-11 13:30:00 | 1902.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-17 09:40:00 | 1871.25 | 2024-10-17 09:45:00 | 1877.58 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-21 09:30:00 | 1836.30 | 2024-10-21 09:35:00 | 1825.36 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-21 09:30:00 | 1836.30 | 2024-10-21 09:40:00 | 1836.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:15:00 | 1768.45 | 2024-10-22 10:30:00 | 1757.25 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-22 10:15:00 | 1768.45 | 2024-10-22 10:40:00 | 1768.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-29 10:05:00 | 1859.25 | 2024-10-29 11:05:00 | 1872.15 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-10-29 10:05:00 | 1859.25 | 2024-10-29 11:15:00 | 1860.40 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-10-30 09:50:00 | 1892.50 | 2024-10-30 10:25:00 | 1884.13 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-03 10:10:00 | 1701.40 | 2024-12-03 10:45:00 | 1707.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-06 09:40:00 | 1646.75 | 2024-12-06 10:05:00 | 1635.45 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-06 09:40:00 | 1646.75 | 2024-12-06 15:20:00 | 1604.35 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2024-12-11 10:15:00 | 1585.55 | 2024-12-11 10:20:00 | 1595.72 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-12-11 10:15:00 | 1585.55 | 2024-12-11 15:20:00 | 1637.45 | TARGET_HIT | 0.50 | 3.27% |
| BUY | retest1 | 2024-12-18 09:30:00 | 1702.50 | 2024-12-18 09:40:00 | 1716.05 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-12-18 09:30:00 | 1702.50 | 2024-12-18 09:50:00 | 1702.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1592.25 | 2024-12-24 09:40:00 | 1601.71 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1592.25 | 2024-12-24 10:30:00 | 1596.65 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2024-12-26 11:05:00 | 1567.10 | 2024-12-26 11:50:00 | 1560.38 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-26 11:05:00 | 1567.10 | 2024-12-26 15:20:00 | 1540.00 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2024-12-27 09:30:00 | 1562.35 | 2024-12-27 09:35:00 | 1557.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-31 10:25:00 | 1539.65 | 2024-12-31 11:00:00 | 1529.64 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-12-31 10:25:00 | 1539.65 | 2024-12-31 13:20:00 | 1535.85 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-01-03 10:25:00 | 1541.50 | 2025-01-03 11:00:00 | 1546.11 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1620.10 | 2025-01-15 09:40:00 | 1628.27 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-16 10:50:00 | 1650.00 | 2025-01-16 10:55:00 | 1645.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-21 09:30:00 | 1662.30 | 2025-01-21 09:35:00 | 1658.11 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-23 10:05:00 | 1618.90 | 2025-01-23 12:10:00 | 1613.46 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-24 11:15:00 | 1577.55 | 2025-01-24 11:35:00 | 1572.13 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-24 11:15:00 | 1577.55 | 2025-01-24 15:20:00 | 1534.25 | TARGET_HIT | 0.50 | 2.74% |
| SELL | retest1 | 2025-01-28 09:55:00 | 1422.95 | 2025-01-28 10:00:00 | 1433.11 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2025-01-29 10:55:00 | 1464.05 | 2025-01-29 12:45:00 | 1457.82 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-29 10:55:00 | 1464.05 | 2025-01-29 14:00:00 | 1464.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-31 11:05:00 | 1496.95 | 2025-01-31 11:20:00 | 1499.54 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-03-13 10:45:00 | 1305.00 | 2025-03-13 11:20:00 | 1298.64 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-18 10:25:00 | 1315.00 | 2025-03-18 10:30:00 | 1308.06 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-03-19 11:05:00 | 1369.60 | 2025-03-19 11:15:00 | 1364.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-17 09:55:00 | 1465.40 | 2025-04-17 10:05:00 | 1461.42 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-23 10:10:00 | 1390.10 | 2025-04-23 10:15:00 | 1394.68 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-29 09:45:00 | 1393.00 | 2025-04-29 09:55:00 | 1398.48 | STOP_HIT | 1.00 | -0.39% |
