# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (36496 bars)
- **Last close:** 2312.30
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 11 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 74
- **Target hits / Stop hits / Partials:** 11 / 74 / 29
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 10.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 29 | 39.7% | 8 | 44 | 21 | 0.12% | 9.0% |
| BUY @ 2nd Alert (retest1) | 73 | 29 | 39.7% | 8 | 44 | 21 | 0.12% | 9.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 11 | 26.8% | 3 | 30 | 8 | 0.04% | 1.4% |
| SELL @ 2nd Alert (retest1) | 41 | 11 | 26.8% | 3 | 30 | 8 | 0.04% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 40 | 35.1% | 11 | 74 | 29 | 0.09% | 10.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1666.00 | 1658.33 | 0.00 | ORB-long ORB[1648.50,1664.40] vol=1.6x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:25:00 | 1674.24 | 1659.88 | 0.00 | T1 1.5R @ 1674.24 |
| Stop hit — per-position SL triggered | 2024-05-16 10:35:00 | 1666.00 | 1661.66 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:05:00 | 1643.50 | 1632.22 | 0.00 | ORB-long ORB[1621.20,1640.00] vol=1.5x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 11:35:00 | 1653.92 | 1637.51 | 0.00 | T1 1.5R @ 1653.92 |
| Target hit | 2024-05-21 15:20:00 | 1689.35 | 1661.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 1677.80 | 1691.38 | 0.00 | ORB-short ORB[1685.15,1703.80] vol=3.2x ATR=8.29 |
| Stop hit — per-position SL triggered | 2024-05-22 10:20:00 | 1686.09 | 1685.89 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 11:15:00 | 1623.00 | 1630.08 | 0.00 | ORB-short ORB[1630.10,1642.00] vol=2.0x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-05-24 12:30:00 | 1627.15 | 1627.92 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:40:00 | 1635.30 | 1626.71 | 0.00 | ORB-long ORB[1611.90,1635.20] vol=1.7x ATR=6.07 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 1629.23 | 1629.61 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:05:00 | 1586.70 | 1592.62 | 0.00 | ORB-short ORB[1592.00,1614.85] vol=3.9x ATR=4.53 |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 1591.23 | 1592.19 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:45:00 | 1643.50 | 1629.58 | 0.00 | ORB-long ORB[1618.20,1629.90] vol=1.8x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:50:00 | 1649.74 | 1631.86 | 0.00 | T1 1.5R @ 1649.74 |
| Stop hit — per-position SL triggered | 2024-06-07 11:10:00 | 1643.50 | 1634.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 1634.00 | 1648.81 | 0.00 | ORB-short ORB[1635.55,1660.00] vol=3.1x ATR=6.93 |
| Stop hit — per-position SL triggered | 2024-06-10 09:35:00 | 1640.93 | 1646.72 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:30:00 | 1599.65 | 1613.23 | 0.00 | ORB-short ORB[1615.75,1625.00] vol=2.2x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-06-12 10:35:00 | 1603.72 | 1611.96 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 11:10:00 | 1575.10 | 1580.19 | 0.00 | ORB-short ORB[1581.10,1591.40] vol=1.5x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 11:50:00 | 1569.91 | 1578.22 | 0.00 | T1 1.5R @ 1569.91 |
| Stop hit — per-position SL triggered | 2024-06-19 13:30:00 | 1575.10 | 1575.64 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 1565.50 | 1557.06 | 0.00 | ORB-long ORB[1543.00,1563.55] vol=1.5x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:35:00 | 1571.37 | 1558.66 | 0.00 | T1 1.5R @ 1571.37 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 1565.50 | 1559.09 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 09:30:00 | 1583.50 | 1577.99 | 0.00 | ORB-long ORB[1562.90,1581.55] vol=2.0x ATR=4.69 |
| Stop hit — per-position SL triggered | 2024-06-24 09:40:00 | 1578.81 | 1580.14 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 1572.70 | 1558.54 | 0.00 | ORB-long ORB[1545.00,1564.90] vol=1.5x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:35:00 | 1577.96 | 1562.49 | 0.00 | T1 1.5R @ 1577.96 |
| Target hit | 2024-06-26 15:20:00 | 1588.90 | 1576.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 1629.25 | 1621.67 | 0.00 | ORB-long ORB[1615.50,1624.00] vol=1.6x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-07-02 10:10:00 | 1625.63 | 1625.49 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 1763.15 | 1771.54 | 0.00 | ORB-short ORB[1763.30,1785.25] vol=1.9x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 1768.65 | 1771.25 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:10:00 | 1800.15 | 1813.45 | 0.00 | ORB-short ORB[1817.00,1834.50] vol=1.6x ATR=6.14 |
| Stop hit — per-position SL triggered | 2024-07-10 11:55:00 | 1806.29 | 1812.28 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:50:00 | 1810.50 | 1816.75 | 0.00 | ORB-short ORB[1817.30,1828.95] vol=3.1x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-07-11 11:00:00 | 1814.43 | 1816.10 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 1805.40 | 1810.77 | 0.00 | ORB-short ORB[1807.00,1825.00] vol=2.1x ATR=5.08 |
| Stop hit — per-position SL triggered | 2024-07-15 11:05:00 | 1810.48 | 1810.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 11:05:00 | 1783.80 | 1797.85 | 0.00 | ORB-short ORB[1802.85,1823.80] vol=2.7x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 1788.28 | 1797.28 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:00:00 | 1810.00 | 1793.18 | 0.00 | ORB-long ORB[1766.05,1787.90] vol=2.1x ATR=6.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:30:00 | 1819.77 | 1805.48 | 0.00 | T1 1.5R @ 1819.77 |
| Target hit | 2024-07-22 13:00:00 | 1825.05 | 1826.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1798.75 | 1809.46 | 0.00 | ORB-short ORB[1803.30,1823.90] vol=2.4x ATR=5.61 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 1804.36 | 1808.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:10:00 | 1821.40 | 1809.45 | 0.00 | ORB-long ORB[1792.00,1808.90] vol=1.6x ATR=5.79 |
| Stop hit — per-position SL triggered | 2024-07-24 10:35:00 | 1815.61 | 1811.68 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:35:00 | 1864.80 | 1857.82 | 0.00 | ORB-long ORB[1841.55,1863.65] vol=1.6x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:35:00 | 1873.21 | 1860.89 | 0.00 | T1 1.5R @ 1873.21 |
| Stop hit — per-position SL triggered | 2024-07-29 12:05:00 | 1864.80 | 1861.96 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 1894.50 | 1886.82 | 0.00 | ORB-long ORB[1862.50,1887.85] vol=4.0x ATR=5.24 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 1889.26 | 1888.19 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:25:00 | 1948.20 | 1936.26 | 0.00 | ORB-long ORB[1909.65,1935.60] vol=5.2x ATR=8.25 |
| Stop hit — per-position SL triggered | 2024-08-01 12:00:00 | 1939.95 | 1943.02 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:40:00 | 2033.10 | 2014.95 | 0.00 | ORB-long ORB[2003.10,2020.00] vol=1.7x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:30:00 | 2044.88 | 2026.48 | 0.00 | T1 1.5R @ 2044.88 |
| Stop hit — per-position SL triggered | 2024-08-08 10:45:00 | 2033.10 | 2027.15 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:50:00 | 2095.05 | 2105.62 | 0.00 | ORB-short ORB[2103.80,2128.95] vol=3.8x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-08-12 11:20:00 | 2100.52 | 2104.54 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:45:00 | 2130.00 | 2118.89 | 0.00 | ORB-long ORB[2092.85,2119.80] vol=2.1x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-08-13 09:50:00 | 2123.50 | 2119.39 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:50:00 | 2111.00 | 2098.81 | 0.00 | ORB-long ORB[2081.55,2099.90] vol=1.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-08-21 10:55:00 | 2106.66 | 2099.43 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:55:00 | 2128.00 | 2109.97 | 0.00 | ORB-long ORB[2096.25,2125.00] vol=1.6x ATR=4.81 |
| Stop hit — per-position SL triggered | 2024-08-22 11:00:00 | 2123.19 | 2110.99 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 2113.60 | 2109.54 | 0.00 | ORB-long ORB[2092.10,2106.00] vol=3.6x ATR=5.51 |
| Stop hit — per-position SL triggered | 2024-08-26 10:55:00 | 2108.09 | 2109.62 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 2215.25 | 2210.72 | 0.00 | ORB-long ORB[2197.00,2214.70] vol=2.2x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-08-29 10:10:00 | 2209.68 | 2212.97 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 2251.70 | 2236.45 | 0.00 | ORB-long ORB[2223.80,2241.35] vol=3.0x ATR=8.63 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 2243.07 | 2239.18 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 2265.55 | 2253.20 | 0.00 | ORB-long ORB[2235.00,2254.00] vol=2.1x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 2260.22 | 2257.55 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:25:00 | 2239.30 | 2236.68 | 0.00 | ORB-long ORB[2206.60,2233.00] vol=6.6x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:45:00 | 2247.79 | 2237.11 | 0.00 | T1 1.5R @ 2247.79 |
| Target hit | 2024-09-04 15:20:00 | 2280.15 | 2259.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:00:00 | 2249.00 | 2243.62 | 0.00 | ORB-long ORB[2210.00,2240.00] vol=1.7x ATR=5.54 |
| Stop hit — per-position SL triggered | 2024-09-11 11:10:00 | 2243.46 | 2243.77 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 2243.85 | 2240.83 | 0.00 | ORB-long ORB[2217.85,2238.25] vol=14.7x ATR=7.03 |
| Stop hit — per-position SL triggered | 2024-09-12 09:40:00 | 2236.82 | 2240.82 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:15:00 | 2271.05 | 2268.27 | 0.00 | ORB-long ORB[2251.05,2267.30] vol=1.5x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-09-17 11:10:00 | 2265.30 | 2270.05 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:15:00 | 2233.10 | 2229.54 | 0.00 | ORB-long ORB[2216.00,2233.00] vol=4.4x ATR=6.22 |
| Stop hit — per-position SL triggered | 2024-09-26 10:30:00 | 2226.88 | 2229.58 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:05:00 | 2180.60 | 2193.66 | 0.00 | ORB-short ORB[2189.50,2219.00] vol=1.6x ATR=6.16 |
| Stop hit — per-position SL triggered | 2024-10-01 11:35:00 | 2186.76 | 2192.60 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:50:00 | 2200.30 | 2183.11 | 0.00 | ORB-long ORB[2163.00,2187.20] vol=2.7x ATR=8.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 10:20:00 | 2212.81 | 2192.41 | 0.00 | T1 1.5R @ 2212.81 |
| Target hit | 2024-10-04 13:30:00 | 2213.55 | 2214.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 2167.80 | 2191.26 | 0.00 | ORB-short ORB[2177.55,2200.95] vol=2.0x ATR=9.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:50:00 | 2154.10 | 2181.88 | 0.00 | T1 1.5R @ 2154.10 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 2167.80 | 2177.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:10:00 | 2204.80 | 2185.31 | 0.00 | ORB-long ORB[2157.10,2174.95] vol=1.8x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:20:00 | 2214.66 | 2191.91 | 0.00 | T1 1.5R @ 2214.66 |
| Stop hit — per-position SL triggered | 2024-10-08 11:35:00 | 2204.80 | 2193.40 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:30:00 | 2259.25 | 2246.67 | 0.00 | ORB-long ORB[2221.55,2252.45] vol=1.7x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-10-09 09:35:00 | 2251.74 | 2249.32 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 2246.95 | 2266.53 | 0.00 | ORB-short ORB[2271.80,2304.90] vol=1.7x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:10:00 | 2237.04 | 2264.50 | 0.00 | T1 1.5R @ 2237.04 |
| Target hit | 2024-10-10 15:20:00 | 2154.25 | 2172.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:55:00 | 2203.10 | 2215.95 | 0.00 | ORB-short ORB[2206.90,2232.75] vol=2.0x ATR=8.31 |
| Stop hit — per-position SL triggered | 2024-10-14 11:25:00 | 2211.41 | 2213.67 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2268.35 | 2260.01 | 0.00 | ORB-long ORB[2235.95,2264.85] vol=1.7x ATR=7.88 |
| Stop hit — per-position SL triggered | 2024-10-15 09:45:00 | 2260.47 | 2262.20 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:00:00 | 2165.25 | 2176.96 | 0.00 | ORB-short ORB[2180.65,2207.00] vol=1.6x ATR=7.55 |
| Stop hit — per-position SL triggered | 2024-10-17 10:25:00 | 2172.80 | 2171.61 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:35:00 | 2186.85 | 2174.55 | 0.00 | ORB-long ORB[2153.65,2185.95] vol=1.5x ATR=6.50 |
| Stop hit — per-position SL triggered | 2024-10-18 11:05:00 | 2180.35 | 2176.87 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:00:00 | 2087.60 | 2079.47 | 0.00 | ORB-long ORB[2062.20,2087.45] vol=2.0x ATR=8.48 |
| Stop hit — per-position SL triggered | 2024-10-24 11:50:00 | 2079.12 | 2082.13 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:10:00 | 2126.20 | 2145.90 | 0.00 | ORB-short ORB[2135.00,2161.20] vol=1.7x ATR=10.82 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 2137.02 | 2144.05 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:20:00 | 2131.05 | 2185.45 | 0.00 | ORB-short ORB[2187.15,2210.50] vol=2.6x ATR=9.70 |
| Stop hit — per-position SL triggered | 2024-11-05 10:25:00 | 2140.75 | 2180.28 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 2118.35 | 2109.13 | 0.00 | ORB-long ORB[2085.15,2110.95] vol=1.9x ATR=7.21 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 2111.14 | 2115.09 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:35:00 | 2044.00 | 2039.80 | 0.00 | ORB-long ORB[2024.20,2042.70] vol=2.0x ATR=6.35 |
| Stop hit — per-position SL triggered | 2024-11-19 09:40:00 | 2037.65 | 2039.71 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:05:00 | 2054.25 | 2031.78 | 0.00 | ORB-long ORB[2012.65,2031.25] vol=1.7x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:20:00 | 2064.07 | 2035.13 | 0.00 | T1 1.5R @ 2064.07 |
| Stop hit — per-position SL triggered | 2024-11-21 11:25:00 | 2054.25 | 2035.55 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 11:05:00 | 2017.25 | 2025.29 | 0.00 | ORB-short ORB[2017.35,2041.85] vol=3.3x ATR=5.40 |
| Stop hit — per-position SL triggered | 2024-11-27 11:40:00 | 2022.65 | 2024.08 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:50:00 | 2037.20 | 2022.97 | 0.00 | ORB-long ORB[1999.00,2019.80] vol=2.9x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-11-29 10:00:00 | 2030.66 | 2024.46 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 2077.80 | 2097.81 | 0.00 | ORB-short ORB[2099.95,2114.80] vol=1.8x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:15:00 | 2069.54 | 2095.69 | 0.00 | T1 1.5R @ 2069.54 |
| Stop hit — per-position SL triggered | 2024-12-05 11:40:00 | 2077.80 | 2092.58 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 2119.35 | 2130.66 | 0.00 | ORB-short ORB[2123.35,2142.00] vol=1.7x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:35:00 | 2112.85 | 2129.25 | 0.00 | T1 1.5R @ 2112.85 |
| Stop hit — per-position SL triggered | 2024-12-09 12:05:00 | 2119.35 | 2128.36 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:15:00 | 2155.00 | 2146.14 | 0.00 | ORB-long ORB[2128.70,2140.00] vol=1.6x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:25:00 | 2160.49 | 2147.33 | 0.00 | T1 1.5R @ 2160.49 |
| Stop hit — per-position SL triggered | 2024-12-11 11:35:00 | 2155.00 | 2147.80 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 2059.60 | 2068.23 | 0.00 | ORB-short ORB[2085.25,2112.00] vol=1.7x ATR=6.02 |
| Stop hit — per-position SL triggered | 2024-12-13 11:20:00 | 2065.62 | 2067.65 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:45:00 | 2122.00 | 2105.67 | 0.00 | ORB-long ORB[2079.20,2108.00] vol=2.6x ATR=6.09 |
| Stop hit — per-position SL triggered | 2024-12-19 10:55:00 | 2115.91 | 2107.24 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:50:00 | 2175.90 | 2155.99 | 0.00 | ORB-long ORB[2138.55,2165.00] vol=1.9x ATR=7.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:40:00 | 2186.72 | 2166.30 | 0.00 | T1 1.5R @ 2186.72 |
| Stop hit — per-position SL triggered | 2024-12-23 12:20:00 | 2175.90 | 2169.77 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:45:00 | 2180.00 | 2167.65 | 0.00 | ORB-long ORB[2150.00,2171.90] vol=2.4x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-12-24 12:45:00 | 2174.74 | 2175.09 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 2181.00 | 2169.64 | 0.00 | ORB-long ORB[2164.90,2180.00] vol=1.9x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:05:00 | 2189.27 | 2174.23 | 0.00 | T1 1.5R @ 2189.27 |
| Target hit | 2024-12-26 12:20:00 | 2182.05 | 2182.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2024-12-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:45:00 | 2208.10 | 2202.86 | 0.00 | ORB-long ORB[2182.10,2204.25] vol=2.1x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:50:00 | 2217.12 | 2207.20 | 0.00 | T1 1.5R @ 2217.12 |
| Stop hit — per-position SL triggered | 2024-12-27 11:00:00 | 2208.10 | 2208.57 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 2254.20 | 2244.10 | 0.00 | ORB-long ORB[2226.85,2252.85] vol=1.6x ATR=5.95 |
| Stop hit — per-position SL triggered | 2024-12-30 09:40:00 | 2248.25 | 2245.70 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 2368.35 | 2356.63 | 0.00 | ORB-long ORB[2351.45,2367.70] vol=1.6x ATR=5.71 |
| Stop hit — per-position SL triggered | 2025-01-02 11:40:00 | 2362.64 | 2363.35 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:50:00 | 2374.70 | 2364.82 | 0.00 | ORB-long ORB[2355.65,2369.00] vol=4.7x ATR=6.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:20:00 | 2384.97 | 2367.22 | 0.00 | T1 1.5R @ 2384.97 |
| Stop hit — per-position SL triggered | 2025-01-07 11:35:00 | 2374.70 | 2368.96 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:40:00 | 2316.00 | 2340.35 | 0.00 | ORB-short ORB[2348.00,2371.15] vol=2.6x ATR=8.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:45:00 | 2303.69 | 2336.64 | 0.00 | T1 1.5R @ 2303.69 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 2316.00 | 2333.86 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:30:00 | 2229.55 | 2241.99 | 0.00 | ORB-short ORB[2233.05,2262.60] vol=2.1x ATR=7.29 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 2236.84 | 2240.37 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:55:00 | 2148.15 | 2158.96 | 0.00 | ORB-short ORB[2153.00,2179.05] vol=2.9x ATR=7.21 |
| Stop hit — per-position SL triggered | 2025-01-14 11:00:00 | 2155.36 | 2158.37 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:10:00 | 2103.00 | 2129.06 | 0.00 | ORB-short ORB[2131.00,2155.45] vol=1.5x ATR=6.12 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 2109.12 | 2126.71 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:35:00 | 2083.75 | 2093.92 | 0.00 | ORB-short ORB[2096.65,2126.45] vol=2.9x ATR=7.79 |
| Stop hit — per-position SL triggered | 2025-01-27 10:40:00 | 2091.54 | 2093.69 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 2084.90 | 2075.61 | 0.00 | ORB-long ORB[2063.00,2079.85] vol=1.9x ATR=6.81 |
| Stop hit — per-position SL triggered | 2025-01-30 10:10:00 | 2078.09 | 2080.49 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 2207.35 | 2187.02 | 0.00 | ORB-long ORB[2165.15,2194.65] vol=1.9x ATR=7.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:20:00 | 2219.24 | 2196.55 | 0.00 | T1 1.5R @ 2219.24 |
| Stop hit — per-position SL triggered | 2025-02-07 12:10:00 | 2207.35 | 2202.58 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:15:00 | 1970.05 | 1964.59 | 0.00 | ORB-long ORB[1936.85,1965.20] vol=1.9x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:40:00 | 1978.56 | 1966.57 | 0.00 | T1 1.5R @ 1978.56 |
| Stop hit — per-position SL triggered | 2025-03-05 12:40:00 | 1970.05 | 1972.77 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 2012.00 | 2021.09 | 0.00 | ORB-short ORB[2014.95,2034.00] vol=2.1x ATR=7.93 |
| Stop hit — per-position SL triggered | 2025-03-06 10:20:00 | 2019.93 | 2018.02 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 2038.10 | 2026.42 | 0.00 | ORB-long ORB[2004.00,2033.05] vol=1.8x ATR=6.69 |
| Stop hit — per-position SL triggered | 2025-03-07 10:35:00 | 2031.41 | 2034.43 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:15:00 | 2077.45 | 2064.45 | 0.00 | ORB-long ORB[2057.30,2075.00] vol=2.2x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 11:30:00 | 2085.74 | 2066.55 | 0.00 | T1 1.5R @ 2085.74 |
| Target hit | 2025-03-20 15:20:00 | 2084.60 | 2081.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 2123.95 | 2108.95 | 0.00 | ORB-long ORB[2080.40,2106.45] vol=2.6x ATR=5.80 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 2118.15 | 2110.68 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 2110.90 | 2111.09 | 0.00 | ORB-short ORB[2115.00,2135.00] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2025-03-24 11:25:00 | 2116.06 | 2111.23 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:00:00 | 2027.35 | 2036.87 | 0.00 | ORB-short ORB[2030.50,2054.00] vol=1.7x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:35:00 | 2016.79 | 2031.34 | 0.00 | T1 1.5R @ 2016.79 |
| Target hit | 2025-03-27 12:40:00 | 2015.50 | 2011.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 84 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 1977.35 | 2002.90 | 0.00 | ORB-short ORB[1996.00,2017.65] vol=2.0x ATR=7.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:25:00 | 1966.82 | 1998.27 | 0.00 | T1 1.5R @ 1966.82 |
| Target hit | 2025-04-01 15:20:00 | 1953.20 | 1972.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — BUY (started 2025-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:10:00 | 2015.90 | 2003.32 | 0.00 | ORB-long ORB[1987.00,2013.00] vol=1.6x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:30:00 | 2025.53 | 2005.98 | 0.00 | T1 1.5R @ 2025.53 |
| Target hit | 2025-04-22 12:25:00 | 2041.00 | 2041.13 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:20:00 | 1666.00 | 2024-05-16 10:25:00 | 1674.24 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-05-16 10:20:00 | 1666.00 | 2024-05-16 10:35:00 | 1666.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 10:05:00 | 1643.50 | 2024-05-21 11:35:00 | 1653.92 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-05-21 10:05:00 | 1643.50 | 2024-05-21 15:20:00 | 1689.35 | TARGET_HIT | 0.50 | 2.79% |
| SELL | retest1 | 2024-05-22 09:40:00 | 1677.80 | 2024-05-22 10:20:00 | 1686.09 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-05-24 11:15:00 | 1623.00 | 2024-05-24 12:30:00 | 1627.15 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-27 09:40:00 | 1635.30 | 2024-05-27 10:05:00 | 1629.23 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-30 10:05:00 | 1586.70 | 2024-05-30 10:15:00 | 1591.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-07 10:45:00 | 1643.50 | 2024-06-07 10:50:00 | 1649.74 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-07 10:45:00 | 1643.50 | 2024-06-07 11:10:00 | 1643.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 09:30:00 | 1634.00 | 2024-06-10 09:35:00 | 1640.93 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-12 10:30:00 | 1599.65 | 2024-06-12 10:35:00 | 1603.72 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-19 11:10:00 | 1575.10 | 2024-06-19 11:50:00 | 1569.91 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-19 11:10:00 | 1575.10 | 2024-06-19 13:30:00 | 1575.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1565.50 | 2024-06-21 09:35:00 | 1571.37 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1565.50 | 2024-06-21 09:40:00 | 1565.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 09:30:00 | 1583.50 | 2024-06-24 09:40:00 | 1578.81 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-26 11:00:00 | 1572.70 | 2024-06-26 11:35:00 | 1577.96 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-06-26 11:00:00 | 1572.70 | 2024-06-26 15:20:00 | 1588.90 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-07-02 09:50:00 | 1629.25 | 2024-07-02 10:10:00 | 1625.63 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-08 09:40:00 | 1763.15 | 2024-07-08 09:45:00 | 1768.65 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 11:10:00 | 1800.15 | 2024-07-10 11:55:00 | 1806.29 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-11 10:50:00 | 1810.50 | 2024-07-11 11:00:00 | 1814.43 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-15 10:50:00 | 1805.40 | 2024-07-15 11:05:00 | 1810.48 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-19 11:05:00 | 1783.80 | 2024-07-19 11:15:00 | 1788.28 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-22 10:00:00 | 1810.00 | 2024-07-22 10:30:00 | 1819.77 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-22 10:00:00 | 1810.00 | 2024-07-22 13:00:00 | 1825.05 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1798.75 | 2024-07-23 11:20:00 | 1804.36 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-24 10:10:00 | 1821.40 | 2024-07-24 10:35:00 | 1815.61 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-29 10:35:00 | 1864.80 | 2024-07-29 11:35:00 | 1873.21 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-29 10:35:00 | 1864.80 | 2024-07-29 12:05:00 | 1864.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:30:00 | 1894.50 | 2024-07-31 09:35:00 | 1889.26 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-01 10:25:00 | 1948.20 | 2024-08-01 12:00:00 | 1939.95 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-08 09:40:00 | 2033.10 | 2024-08-08 10:30:00 | 2044.88 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-08-08 09:40:00 | 2033.10 | 2024-08-08 10:45:00 | 2033.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-12 10:50:00 | 2095.05 | 2024-08-12 11:20:00 | 2100.52 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-13 09:45:00 | 2130.00 | 2024-08-13 09:50:00 | 2123.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-21 10:50:00 | 2111.00 | 2024-08-21 10:55:00 | 2106.66 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-22 10:55:00 | 2128.00 | 2024-08-22 11:00:00 | 2123.19 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-26 10:50:00 | 2113.60 | 2024-08-26 10:55:00 | 2108.09 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-29 09:30:00 | 2215.25 | 2024-08-29 10:10:00 | 2209.68 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-30 09:30:00 | 2251.70 | 2024-08-30 09:40:00 | 2243.07 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-03 09:35:00 | 2265.55 | 2024-09-03 09:40:00 | 2260.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-04 10:25:00 | 2239.30 | 2024-09-04 10:45:00 | 2247.79 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-04 10:25:00 | 2239.30 | 2024-09-04 15:20:00 | 2280.15 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2024-09-11 11:00:00 | 2249.00 | 2024-09-11 11:10:00 | 2243.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-12 09:30:00 | 2243.85 | 2024-09-12 09:40:00 | 2236.82 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-17 10:15:00 | 2271.05 | 2024-09-17 11:10:00 | 2265.30 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-26 10:15:00 | 2233.10 | 2024-09-26 10:30:00 | 2226.88 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-01 11:05:00 | 2180.60 | 2024-10-01 11:35:00 | 2186.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-04 09:50:00 | 2200.30 | 2024-10-04 10:20:00 | 2212.81 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-04 09:50:00 | 2200.30 | 2024-10-04 13:30:00 | 2213.55 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-07 10:45:00 | 2167.80 | 2024-10-07 10:50:00 | 2154.10 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-07 10:45:00 | 2167.80 | 2024-10-07 11:15:00 | 2167.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 11:10:00 | 2204.80 | 2024-10-08 11:20:00 | 2214.66 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-08 11:10:00 | 2204.80 | 2024-10-08 11:35:00 | 2204.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 09:30:00 | 2259.25 | 2024-10-09 09:35:00 | 2251.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-10 11:00:00 | 2246.95 | 2024-10-10 11:10:00 | 2237.04 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-10 11:00:00 | 2246.95 | 2024-10-10 15:20:00 | 2154.25 | TARGET_HIT | 0.50 | 4.13% |
| SELL | retest1 | 2024-10-14 10:55:00 | 2203.10 | 2024-10-14 11:25:00 | 2211.41 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-15 09:30:00 | 2268.35 | 2024-10-15 09:45:00 | 2260.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-17 10:00:00 | 2165.25 | 2024-10-17 10:25:00 | 2172.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-18 10:35:00 | 2186.85 | 2024-10-18 11:05:00 | 2180.35 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-24 10:00:00 | 2087.60 | 2024-10-24 11:50:00 | 2079.12 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-25 10:10:00 | 2126.20 | 2024-10-25 10:20:00 | 2137.02 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-11-05 10:20:00 | 2131.05 | 2024-11-05 10:25:00 | 2140.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-11-12 09:40:00 | 2118.35 | 2024-11-12 11:20:00 | 2111.14 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-19 09:35:00 | 2044.00 | 2024-11-19 09:40:00 | 2037.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-21 11:05:00 | 2054.25 | 2024-11-21 11:20:00 | 2064.07 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-21 11:05:00 | 2054.25 | 2024-11-21 11:25:00 | 2054.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 11:05:00 | 2017.25 | 2024-11-27 11:40:00 | 2022.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-29 09:50:00 | 2037.20 | 2024-11-29 10:00:00 | 2030.66 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2077.80 | 2024-12-05 11:15:00 | 2069.54 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2077.80 | 2024-12-05 11:40:00 | 2077.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-09 11:00:00 | 2119.35 | 2024-12-09 11:35:00 | 2112.85 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-09 11:00:00 | 2119.35 | 2024-12-09 12:05:00 | 2119.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:15:00 | 2155.00 | 2024-12-11 11:25:00 | 2160.49 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-12-11 11:15:00 | 2155.00 | 2024-12-11 11:35:00 | 2155.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:10:00 | 2059.60 | 2024-12-13 11:20:00 | 2065.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-19 10:45:00 | 2122.00 | 2024-12-19 10:55:00 | 2115.91 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-23 10:50:00 | 2175.90 | 2024-12-23 11:40:00 | 2186.72 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-23 10:50:00 | 2175.90 | 2024-12-23 12:20:00 | 2175.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:45:00 | 2180.00 | 2024-12-24 12:45:00 | 2174.74 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-26 10:55:00 | 2181.00 | 2024-12-26 11:05:00 | 2189.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-26 10:55:00 | 2181.00 | 2024-12-26 12:20:00 | 2182.05 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-12-27 09:45:00 | 2208.10 | 2024-12-27 10:50:00 | 2217.12 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-27 09:45:00 | 2208.10 | 2024-12-27 11:00:00 | 2208.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:30:00 | 2254.20 | 2024-12-30 09:40:00 | 2248.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-02 10:45:00 | 2368.35 | 2025-01-02 11:40:00 | 2362.64 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-07 10:50:00 | 2374.70 | 2025-01-07 11:20:00 | 2384.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-07 10:50:00 | 2374.70 | 2025-01-07 11:35:00 | 2374.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 09:40:00 | 2316.00 | 2025-01-08 09:45:00 | 2303.69 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-08 09:40:00 | 2316.00 | 2025-01-08 09:50:00 | 2316.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:30:00 | 2229.55 | 2025-01-10 09:40:00 | 2236.84 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-14 10:55:00 | 2148.15 | 2025-01-14 11:00:00 | 2155.36 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-21 11:10:00 | 2103.00 | 2025-01-21 11:35:00 | 2109.12 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-27 10:35:00 | 2083.75 | 2025-01-27 10:40:00 | 2091.54 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-30 09:30:00 | 2084.90 | 2025-01-30 10:10:00 | 2078.09 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-07 11:00:00 | 2207.35 | 2025-02-07 11:20:00 | 2219.24 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-07 11:00:00 | 2207.35 | 2025-02-07 12:10:00 | 2207.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 10:15:00 | 1970.05 | 2025-03-05 10:40:00 | 1978.56 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-03-05 10:15:00 | 1970.05 | 2025-03-05 12:40:00 | 1970.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-06 09:30:00 | 2012.00 | 2025-03-06 10:20:00 | 2019.93 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-07 09:30:00 | 2038.10 | 2025-03-07 10:35:00 | 2031.41 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-20 11:15:00 | 2077.45 | 2025-03-20 11:30:00 | 2085.74 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-20 11:15:00 | 2077.45 | 2025-03-20 15:20:00 | 2084.60 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-03-21 09:40:00 | 2123.95 | 2025-03-21 09:45:00 | 2118.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-24 11:10:00 | 2110.90 | 2025-03-24 11:25:00 | 2116.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-27 10:00:00 | 2027.35 | 2025-03-27 10:35:00 | 2016.79 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-03-27 10:00:00 | 2027.35 | 2025-03-27 12:40:00 | 2015.50 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-01 11:00:00 | 1977.35 | 2025-04-01 11:25:00 | 1966.82 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-01 11:00:00 | 1977.35 | 2025-04-01 15:20:00 | 1953.20 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-04-22 10:10:00 | 2015.90 | 2025-04-22 10:30:00 | 2025.53 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-22 10:10:00 | 2015.90 | 2025-04-22 12:25:00 | 2041.00 | TARGET_HIT | 0.50 | 1.25% |
